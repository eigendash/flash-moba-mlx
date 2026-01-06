"""Tiled block-sparse MoBA attention with an online (running) softmax.

This is the fast path.  It never materialises an ``L x L`` score matrix and never
reads a key it did not select: for each of the ``k`` selected slots it gathers the
``B`` keys and values of that one block and folds them into a running
max/sum/accumulator, FlashAttention style.  Work per query is ``O(k B)`` instead
of ``O(L)`` and the memory traffic is ``O(k B d)`` instead of ``O(L d)``.

Section 4 of arXiv:2511.11571 is what motivates this shape of implementation: the
paper's complaint about the original MoBA code is that it materialises the score
matrix and gathers non-contiguous blocks per query, and its ``gather-and-densify``
kernel fixes both by working block-by-block with a running softmax.  The loop
below is the same idea expressed in MLX, one gather per logical key block, with a
per-query rescale between blocks.  It is autodifferentiable, which the Metal
kernel in :mod:`moba.kernel` is not, so this is also the path the toy LM trains
through.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def _gather_rows(source: mx.array, positions: mx.array, length: int) -> mx.array:
    """Batched row gather: ``source[..., positions[..., i], :]``.

    MLX's ``take`` follows NumPy and inserts the *whole* index shape after
    ``source.shape[:axis]``, so a ``(..., L, B)`` index against a ``(..., L, d)``
    source yields a rank-6 tensor instead of ``(..., L, B, d)``.  Flattening every
    leading axis into one batch and gathering along axis 0 gives the batched row
    gather that is actually wanted.  This is the only operation that reads KV rows
    from memory, so it is also where "only the selected blocks are ever loaded" is
    enforced.
    """
    *lead, src_len, dim = source.shape
    batch = 1
    for size in lead:
        batch *= size
    flat_source = source.reshape(batch * src_len, dim)
    rows = positions.reshape(batch, *positions.shape[len(lead) :])
    row_ids = mx.arange(batch).reshape([batch] + [1] * (rows.ndim - 1)) * src_len
    gathered = mx.take(flat_source, row_ids + rows, axis=0)
    return gathered.reshape(*positions.shape, dim)


def tiled_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    indices: mx.array,
    selected: mx.array,
    block_size: int,
    scale: float | None = None,
) -> mx.array:
    """Block-sparse attention over exactly the selected blocks.

    ``queries``/``keys``/``values`` are ``(..., L, d)``; ``indices`` and
    ``selected`` are ``(..., L, k)`` from :class:`moba.router.Router`.  Slots with
    ``selected=False`` are skipped, keys later than the query are masked, and the
    accumulation is float32 regardless of the input dtype.
    """
    dim = queries.shape[-1]
    length = queries.shape[-2]
    lead = queries.shape[:-2]
    if scale is None:
        scale = dim**-0.5
    slots = indices.shape[-1]
    offsets = mx.arange(block_size).astype(mx.int32)
    query_pos = mx.arange(length).reshape([1] * len(lead) + [length, 1])

    q = queries.astype(mx.float32)
    k = keys.astype(mx.float32)
    v = values.astype(mx.float32)

    running_max = mx.full((*lead, length), -mx.inf, dtype=mx.float32)
    running_sum = mx.zeros((*lead, length), dtype=mx.float32)
    acc = mx.zeros((*lead, length, dim), dtype=mx.float32)

    for slot in range(slots):
        block = indices[..., slot]  # (..., L)
        pos = block[..., None] * block_size + offsets  # (..., L, B)
        ok = selected[..., slot][..., None] & (pos <= query_pos)
        kb = _gather_rows(k, pos, length)  # (..., L, B, d)
        vb = _gather_rows(v, pos, length)
        scores = mx.sum(q[..., :, None, :] * kb, axis=-1) * scale  # (..., L, B)
        scores = mx.where(ok, scores, -mx.inf)

        block_max = mx.max(scores, axis=-1)
        new_max = mx.maximum(running_max, block_max)
        # If a whole slot is masked the two maxima are both -inf and the
        # difference is NaN, so the rescale is explicitly zeroed there.
        alpha = mx.where(
            mx.isinf(new_max), mx.zeros_like(new_max), mx.exp(running_max - new_max)
        )
        probs = mx.where(
            ok,
            mx.exp(scores - new_max[..., None]),
            mx.zeros_like(scores),
        )
        running_sum = running_sum * alpha + mx.sum(probs, axis=-1)
        acc = acc * alpha[..., None] + mx.sum(probs[..., None] * vb, axis=-2)
        running_max = new_max

    # Every query's own block is always a legal candidate, so the denominator is
    # positive; the clamp only protects an explicitly all-masked call.
    denominator = mx.maximum(running_sum, 1e-30)
    return (acc / denominator[..., None]).astype(queries.dtype)


@lru_cache(maxsize=128)
def _compiled(block_size: int, slots: int, dim: int, scale: float):
    def run(q, k, v, idx, sel):
        return tiled_attention(q, k, v, idx, sel, block_size, scale)

    return mx.compile(run)


def tiled_attention_compiled(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    indices: mx.array,
    selected: mx.array,
    block_size: int,
    scale: float | None = None,
) -> mx.array:
    """``tiled_attention`` traced once with ``mx.compile``.

    The block size, slot count and scale are baked into the trace, so the cache is
    keyed on them; ``mx.compile`` then fuses the per-block online-softmax update
    into a single graph and removes the Python-level loop dispatch.
    """
    dim = queries.shape[-1]
    if scale is None:
        scale = dim**-0.5
    fn = _compiled(block_size, indices.shape[-1], dim, float(scale))
    return fn(queries, keys, values, indices, selected)


def unselected_block_is_ignored(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    indices: mx.array,
    selected: mx.array,
    block_size: int,
    scale: float | None = None,
) -> mx.array:
    """Same as :func:`tiled_attention`; exists so tests can state the intent."""
    return tiled_attention(
        queries, keys, values, indices, selected, block_size, scale
    )
