"""The MoBA router: query-to-block scoring and causal top-k block selection.

Section 2 of arXiv:2511.11571 reviews the router as ``s_i = q^T k~_i`` with
``k~_i`` the mean of block ``i``, followed by a top-k.  Section 3 of the same
paper is the part that is new: it models exactly this selection as a
needle-in-a-haystack problem and derives the signal-to-noise ratio
``SNR = Delta_mu_eff * sqrt(d / (2B))`` that governs how often a noise block
outranks the signal block.  The router here is deliberately the plain centroid
router the model describes, so that the experiment can look at its accuracy
directly instead of only at the end-to-end loss.
"""

from __future__ import annotations

import mlx.core as mx

from .blocks import block_index, block_valid_mask, causal_block_scores, n_blocks


def topk_blocks(
    scores: mx.array,
    k: int,
    valid: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Top-k block ids per query, deterministically.

    ``scores`` is ``(..., L, K)``, ``valid`` is ``(..., L, K)`` bool and defaults
    to "all blocks are candidates", which is the non-causal router used to test
    the selection rule itself.

    Returns ``(indices, selected)`` of shape ``(..., L, k)``: the block ids in
    descending score order, and a bool flag saying whether that slot is a real
    selection.  A query with fewer than ``k`` candidates repeats index 0 in the
    unused slots with ``selected=False``, so downstream code can mask them out
    instead of silently attending to block 0.

    ``mx.argsort`` on the negated scores is used rather than a loop of argmax so
    that the whole selection is one kernel; that also fixes the tie order to
    whatever argsort does, which the tests pin down as "reproducible across
    calls" rather than "lowest index wins".
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    *lead, count = scores.shape
    if valid is None:
        valid = mx.ones_like(scores, dtype=mx.bool_)
    masked = mx.where(valid, scores, -mx.inf)
    take = min(k, count)
    order = mx.argsort(-masked, axis=-1)[..., :take]
    selected = mx.take_along_axis(masked, order, axis=-1) > -mx.inf
    if k > count:
        pad_shape = (*lead, k - count)
        order = mx.concatenate(
            [order, mx.zeros(pad_shape, dtype=order.dtype)], axis=-1
        )
        selected = mx.concatenate(
            [selected, mx.zeros(pad_shape, dtype=mx.bool_)], axis=-1
        )
    return order.astype(mx.int32), selected


def block_selection_mask(
    indices: mx.array,
    selected: mx.array,
    count: int,
) -> mx.array:
    """Scatter ``(..., L, k)`` selections into a ``(..., L, K)`` block mask.

    Unselected slots are written to a throwaway sentinel column that is dropped
    afterwards.  Writing them to column 0 instead would let a padding slot
    overwrite a genuine selection of block 0.
    """
    if indices.shape != selected.shape:
        raise ValueError("indices and selected must have the same shape")
    sentinel = mx.where(selected, indices, count)
    base = mx.zeros((*indices.shape[:-1], count + 1), dtype=mx.bool_)
    full = mx.put_along_axis(base, sentinel, selected, axis=-1)
    return full[..., :count]


def token_selection_mask(
    indices: mx.array,
    selected: mx.array,
    seq_len: int,
    block_size: int,
) -> mx.array:
    """Expand a block selection into the ``(..., L, L)`` causal token mask.

    Entry ``[t, s]`` is true iff key token ``s`` is in a block that query ``t``
    selected.  The causal triangle is intersected in, which is what makes a
    query inside a partially-visible block attend only to the part of it that
    already happened.
    """
    count = n_blocks(seq_len, block_size)
    blocks = block_selection_mask(indices, selected, count)
    tokens = block_index(seq_len, block_size)
    mask = mx.take(blocks, tokens, axis=-1)
    causal = mx.arange(seq_len)[None, :] <= mx.arange(seq_len)[:, None]
    return mask & causal


class Router:
    """Centroid router with a configurable top-k.

    ``Router.__call__`` returns ``(indices, selected, scores)`` for queries and
    keys shaped ``(..., L, d)``.  It is stateless: the affine projections live in
    the attention module, exactly as in MoBA, where the gate reuses the attention
    query.
    """

    def __init__(
        self,
        k: int,
        block_size: int,
        scale: float | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self.block_size = int(block_size)
        self.scale = scale

    def scores(self, queries: mx.array, keys: mx.array) -> mx.array:
        return causal_block_scores(
            queries, keys, self.block_size, scale=self.scale
        )

    def __call__(
        self, queries: mx.array, keys: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        scores = self.scores(queries, keys)
        length = queries.shape[-2]
        valid = mx.broadcast_to(
            block_valid_mask(length, self.block_size), scores.shape
        )
        indices, selected = topk_blocks(scores, self.k, valid)
        # The selection is a discrete decision: nothing is differentiable through
        # argsort or through the resulting integer indices, and MLX would
        # otherwise try to build a VJP for the integer gather in the tiled path.
        return (
            mx.stop_gradient(indices),
            mx.stop_gradient(selected),
            scores,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Router(k={self.k}, block_size={self.block_size}, "
            f"scale={self.scale})"
        )
