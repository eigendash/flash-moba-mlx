"""A Metal kernel for block-sparse MoBA attention, forward only.

This is the toy counterpart of the paper's ``gather-and-densify`` forward pass
(Section 4.2 and Appendix D.2 of arXiv:2511.11571).  One thread owns one query
row.  For each selected block it walks the ``B`` keys of that block in sequence,
masking keys beyond the query, and updates a running max / running sum /
accumulator triple -- the same online softmax as :mod:`moba.tiled`, with only the
selected key blocks ever read from memory.

Differences from the paper's kernel, stated up front:

* It is one kernel, not the paper's three-stage ``Flash TopK`` plus forward plus
  backward pipeline.  There is no fused tiled top-k kernel here: the router runs
  in MLX and this kernel consumes its indices.
* There is no backward pass, because MLX custom Metal kernels are not
  autodifferentiable.  Training goes through :mod:`moba.tiled` instead, and
  ``tests/test_kernel.py`` pins this kernel to be bit-comparable with that path,
  so the forward semantics are the same even though the gradients come from a
  different implementation.
* No SRAM tiling of Q or of the physical ``Br x Bc`` tiles, no varlen index
  reformatting, no grouped-query attention.  Head dimensions and block sizes are
  baked into the Metal source at build time.
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_CACHE: dict[tuple[int, object], object] = {}


def _metal_source(dim: int) -> str:
    """Metal source with the head dimension specialised in.

    The accumulator is a fixed-size register array, so ``d`` has to be a compile
    time constant; baking it into the source is simpler than threading an integer
    template parameter through MLX's kernel signature.
    """
    return f"""
    uint gid = thread_position_in_grid.x;
    uint L = (uint)info[0];
    uint SLOTS = (uint)info[1];
    uint BB = (uint)info[2];

    uint bh = gid / L;
    uint t = gid % L;

    const device T* qp = q + ((size_t)bh * L + t) * {dim};
    const device int* idx = block_ids + ((size_t)bh * L + t) * SLOTS;
    const device int* sel = selected + ((size_t)bh * L + t) * SLOTS;

    float acc[{dim}];
    for (uint i = 0; i < {dim}; ++i) {{
        acc[i] = 0.0f;
    }}
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint slot = 0; slot < SLOTS; ++slot) {{
        if (sel[slot] == 0) {{
            continue;
        }}
        uint first = (uint)idx[slot] * BB;
        for (uint j = 0; j < BB; ++j) {{
            uint pos = first + j;
            if (pos > t || pos >= L) {{
                continue;
            }}
            const device T* kp = k + ((size_t)bh * L + pos) * {dim};
            float score = 0.0f;
            for (uint i = 0; i < {dim}; ++i) {{
                score += (float)qp[i] * (float)kp[i];
            }}
            score *= scale[0];
            float new_max = metal::max(running_max, score);
            float alpha = metal::exp(running_max - new_max);
            float weight = metal::exp(score - new_max);
            running_sum = running_sum * alpha + weight;
            const device T* vp = v + ((size_t)bh * L + pos) * {dim};
            for (uint i = 0; i < {dim}; ++i) {{
                acc[i] = acc[i] * alpha + weight * (float)vp[i];
            }}
            running_max = new_max;
        }}
    }}

    device T* op = out + ((size_t)bh * L + t) * {dim};
    float inv = 1.0f / metal::max(running_sum, 1e-30f);
    for (uint i = 0; i < {dim}; ++i) {{
        op[i] = (T)(acc[i] * inv);
    }}
    """


def _get_kernel(dim: int, dtype: mx.Dtype):
    key = (dim, str(dtype))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"moba_block_sparse_d{dim}_{str(dtype).replace('.', '_')}",
            input_names=["q", "k", "v", "block_ids", "selected", "info", "scale"],
            output_names=["out"],
            source=_metal_source(dim),
        )
    return _KERNEL_CACHE[key]


def metal_available() -> bool:
    return bool(mx.metal.is_available())


def metal_moba_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    indices: mx.array,
    selected: mx.array,
    block_size: int,
    scale: float | None = None,
) -> mx.array:
    """Forward-only block-sparse attention on the GPU.

    Shapes match :func:`moba.tiled.tiled_attention`: ``(..., L, d)`` inputs and
    ``(..., L, k)`` selection.  The output is the same tensor the tiled path
    produces, up to float rounding.
    """
    if not metal_available():
        raise RuntimeError("Metal is not available on this device")
    *lead, length, dim = queries.shape
    if queries.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise TypeError(f"unsupported dtype {queries.dtype}")
    if scale is None:
        scale = dim**-0.5
    batch = 1
    for size in lead:
        batch *= size
    slots = indices.shape[-1]

    q = mx.contiguous(queries.reshape(batch, length, dim))
    k = mx.contiguous(keys.reshape(batch, length, dim))
    v = mx.contiguous(values.reshape(batch, length, dim))
    ids = mx.contiguous(indices.reshape(batch, length, slots).astype(mx.int32))
    sel = mx.contiguous(selected.reshape(batch, length, slots).astype(mx.int32))
    info = mx.array([length, slots, block_size], dtype=mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)

    kernel = _get_kernel(dim, queries.dtype)
    threads = batch * length
    out = kernel(
        inputs=[q, k, v, ids, sel, info, scale_arr],
        template=[("T", queries.dtype)],
        grid=(threads, 1, 1),
        threadgroup=(min(128, threads), 1, 1),
        output_shapes=[(batch, length, dim)],
        output_dtypes=[queries.dtype],
    )[0]
    return out.reshape(*lead, length, dim)
