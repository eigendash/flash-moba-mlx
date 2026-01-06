"""Dense and mask-based MoBA attention: the slow paths everything else matches.

Two things live here:

* :func:`dense_attention` -- ordinary causal attention, the baseline the
  experiment trains against and the limit that ``k = K`` must reproduce.
* :func:`moba_reference_attention` -- MoBA written the obvious way: score every
  query against every key, keep the scores whose key block was selected, softmax
  and reduce.  It is O(L^2) time and memory and loads every key whether or not it
  is selected, which is exactly what the paper's Section 4 complains about.  It
  is the correctness oracle for the tiled path in :mod:`moba.attention` and for
  the Metal kernel in :mod:`moba.kernel`.
"""

from __future__ import annotations

import mlx.core as mx

from .blocks import n_blocks
from .router import token_selection_mask


def causal_token_mask(
    length: int,
    keys_length: int | None = None,
    dtype: mx.Dtype = mx.bool_,
) -> mx.array:
    """``(L, S)`` bool lower-triangular mask, true where a key is visible."""
    keys_length = length if keys_length is None else keys_length
    rows = mx.arange(length)[:, None]
    cols = mx.arange(keys_length)[None, :]
    return (cols <= rows).astype(dtype)


def attention_mask_bias(mask: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Additive bias: 0 where ``mask`` is true and ``-inf`` where it is false."""
    return mx.where(mask, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


def dense_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float | None = None,
) -> mx.array:
    """``softmax(Q K^T / sqrt(d)) V`` with a causal mask. ``(..., L, d) -> (..., L, d)``."""
    dim = queries.shape[-1]
    if scale is None:
        scale = dim**-0.5
    scores = mx.matmul(queries, mx.swapaxes(keys, -1, -2)) * scale
    bias = attention_mask_bias(
        causal_token_mask(queries.shape[-2], keys.shape[-2]), scores.dtype
    )
    probs = mx.softmax(scores + bias, axis=-1)
    return mx.matmul(probs, values)


def moba_reference_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    indices: mx.array,
    selected: mx.array,
    block_size: int,
    scale: float | None = None,
    return_probs: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Full attention restricted to the selected blocks by a materialised mask.

    ``indices``/``selected`` are ``(..., L, k)`` as returned by
    :class:`moba.router.Router`.  Nothing is gathered: the L x L score matrix is
    built regardless, the mask is applied with ``-inf``, and the softmax is taken
    over the surviving keys.  A query with no visible key at all (only possible
    if the caller passes an all-false mask) gets a uniform row rather than NaNs.
    """
    dim = queries.shape[-1]
    length = queries.shape[-2]
    if scale is None:
        scale = dim**-0.5
    scores = mx.matmul(queries, mx.swapaxes(keys, -1, -2)) * scale
    mask = token_selection_mask(indices, selected, length, block_size)
    masked = scores + attention_mask_bias(mask, scores.dtype)
    probs = mx.softmax(masked, axis=-1)
    out = mx.matmul(probs, values)
    if return_probs:
        return out, probs
    return out


def moba_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    indices: mx.array,
    selected: mx.array,
    block_size: int,
    scale: float | None = None,
    impl: str = "reference",
) -> mx.array:
    """Dispatch across the four implementations of MoBA attention.

    ``"reference"`` is the mask-based path above, ``"tiled"`` and ``"compiled"``
    are the block-sparse online-softmax loop in :mod:`moba.tiled`, and ``"metal"``
    is the GPU kernel in :mod:`moba.kernel`.  ``"dense"`` ignores the selection
    and is only useful as a baseline.  All four agree to 1e-5 for a given
    selection; the tests assert exactly that.
    """
    if impl == "reference":
        return moba_reference_attention(
            queries, keys, values, indices, selected, block_size, scale
        )
    if impl == "dense":
        return dense_attention(queries, keys, values, scale)
    if impl in ("tiled", "compiled"):
        from .tiled import tiled_attention, tiled_attention_compiled

        fn = tiled_attention if impl == "tiled" else tiled_attention_compiled
        return fn(queries, keys, values, indices, selected, block_size, scale)
    if impl == "metal":
        from .kernel import metal_moba_attention

        return metal_moba_attention(
            queries, keys, values, indices, selected, block_size, scale
        )
    raise ValueError(f"unknown impl {impl!r}")


def selection_statistics(
    indices: mx.array,
    selected: mx.array,
    seq_len: int,
    block_size: int,
) -> dict[str, mx.array]:
    """Per-query number of selected blocks and the loaded-block fraction.

    ``count`` is ``(..., L)`` float: how many key blocks the query actually
    attends to.  ``fraction`` divides that by ``K``, the number of blocks dense
    attention would have touched.
    """
    count = mx.sum(selected, axis=-1).astype(mx.float32)
    total = n_blocks(seq_len, block_size)
    return {"count": count, "fraction": count / total}
