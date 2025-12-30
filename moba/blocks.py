"""Key/value block partitioning and the block statistics the router scores.

MoBA (Lu et al., arXiv:2502.13189, as reviewed in Section 2 of
arXiv:2511.11571) splits the key sequence into ``K = L / B`` contiguous blocks of
``B`` tokens and scores block ``j`` for a query ``q`` by the dot product with the
block *centroid* ``k~_j = (1/B) sum_{s in block j} k_s``.

Everything here is written for a trailing ``(..., L, d)`` layout, so the leading
dimensions can be batch, head or both.
"""

from __future__ import annotations

import mlx.core as mx


def n_blocks(seq_len: int, block_size: int) -> int:
    """Number of blocks; the sequence length must divide evenly."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if seq_len % block_size:
        raise ValueError(
            f"seq_len {seq_len} must be a multiple of block_size {block_size}"
        )
    return seq_len // block_size


def block_index(length: int, block_size: int) -> mx.array:
    """``(L,)`` int32: the block that each token belongs to."""
    n_blocks(length, block_size)
    return (mx.arange(length) // block_size).astype(mx.int32)


def block_centroids(keys: mx.array, block_size: int) -> mx.array:
    """Mean-pool each block of keys: ``(..., L, d) -> (..., K, d)``."""
    *lead, length, dim = keys.shape
    n_blocks(length, block_size)
    grouped = keys.reshape(*lead, length // block_size, block_size, dim)
    return grouped.mean(axis=-2)


def prefix_mean_within_block(keys: mx.array, block_size: int) -> mx.array:
    """Causal prefix mean *inside* each block: ``(..., L, d) -> (..., L, d)``.

    Row ``t`` holds the mean of the keys from the start of ``t``'s block up to
    and including ``t``.  For ``t`` at a block boundary this is just the key
    itself, so scoring a block with it never reads a future key.  This is what
    makes the router causal: the current block is scored on the part of itself
    that the query can actually see, not on the whole block.

    The cumulative sum is taken *inside* each block rather than globally: the
    global form subtracts two nearby cumulative sums and loses ~1e-5 to
    cancellation at block boundaries in float32.
    """
    *lead, length, dim = keys.shape
    count = n_blocks(length, block_size)
    grouped = keys.reshape(*lead, count, block_size, dim)
    running = mx.cumsum(grouped, axis=-2)
    offset = mx.arange(1, block_size + 1).astype(keys.dtype)
    mean = running / offset.reshape([1] * (keys.ndim - 2) + [block_size, 1])
    return mean.reshape(*lead, length, dim)


def block_valid_mask(length: int, block_size: int) -> mx.array:
    """``(L, K)`` bool: block ``j`` is a candidate for query ``t`` iff ``j <= t // B``.

    A query can only select its own block or an earlier one.  Blocks after the
    query's own block contain only future keys and are never candidates, which is
    the causality rule the paper states as "blocks containing future tokens are
    masked during selection".
    """
    count = n_blocks(length, block_size)
    tokens = mx.arange(length)[:, None]
    blocks = mx.arange(count)[None, :]
    return blocks <= (tokens // block_size)


def causal_block_centroids(keys: mx.array, block_size: int) -> mx.array:
    """``(..., L, K, d)``: the centroid each query may score, per candidate block.

    This materialises the full per-query centroid tensor and is only used as a
    readable reference for tests.  ``causal_block_scores`` computes the same
    numbers without building it.
    """
    *lead, length, dim = keys.shape
    count = n_blocks(length, block_size)
    means = block_centroids(keys, block_size)  # (..., K, d)
    within = prefix_mean_within_block(keys, block_size)  # (..., L, d)
    tokens = (mx.arange(length) // block_size)[:, None]  # (L, 1)
    blocks = mx.arange(count)[None, :]  # (1, K)
    is_own = blocks == tokens  # (L, K)
    is_past = blocks < tokens
    past = mx.broadcast_to(
        means[..., None, :, :], (*lead, length, count, dim)
    )
    own = mx.broadcast_to(
        within[..., :, None, :], (*lead, length, count, dim)
    )
    zeros = mx.zeros_like(past)
    grid = mx.broadcast_to(is_own[..., None], (*lead, length, count, 1))
    past_grid = mx.broadcast_to(is_past[..., None], (*lead, length, count, 1))
    return mx.where(grid, own, mx.where(past_grid, past, zeros))


def causal_block_scores(
    queries: mx.array,
    keys: mx.array,
    block_size: int,
    scale: float | None = None,
) -> mx.array:
    """Router scores ``s[t, j] = q_t . k~_j / sqrt(d)`` for causally legal ``j``.

    ``(..., L, d) x (..., L, d) -> (..., L, K)``.  Scores of blocks that are not
    candidates (``j > t // B``) are still returned but are meaningless; the
    router masks them with :func:`moba.router.topk_blocks`.

    The candidate centroid is built in two pieces: a single matmul against all
    ``K`` full-block centroids, then an overwrite of the diagonal ``j == t // B``
    entry with the causal prefix mean of the query's own block.
    """
    *lead, length, dim = queries.shape
    count = n_blocks(length, block_size)
    if scale is None:
        scale = dim**-0.5
    means = block_centroids(keys, block_size)  # (..., K, d)
    scores = mx.matmul(queries, mx.swapaxes(means, -1, -2)) * scale  # (..., L, K)
    within = prefix_mean_within_block(keys, block_size)  # (..., L, d)
    own = mx.sum(queries * within, axis=-1, keepdims=True) * scale  # (..., L, 1)
    tokens = (mx.arange(length) // block_size)[:, None]
    blocks = mx.arange(count)[None, :]
    return mx.where(blocks == tokens, own, scores)
