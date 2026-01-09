"""Measurements: router retrieval, loaded blocks, KV traffic and tile schedules.

The paper's Section 4 compares FlashMoBA against dense attention and against the
original MoBA on time and memory.  There is no CUDA here and no 8K-512K context,
so what is measured instead is the quantity that actually drives the cost: how
many key blocks each query reads, expressed as a fraction of what dense causal
attention would read, plus wall-clock on this machine.  The tile schedule is the
query-block x key-block grid the paper's kernel iterates over, and is what the
plots in :mod:`moba.plots` visualise.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from .router import block_selection_mask


def selection_counts(selected: mx.array) -> mx.array:
    """``(..., L, k)`` bool -> ``(..., L)`` float32 number of blocks per query."""
    return mx.sum(selected.astype(mx.float32), axis=-1)


def dense_counts(seq_len: int, block_size: int) -> np.ndarray:
    """Blocks a fully causal attention would read for each query position."""
    tokens = np.arange(seq_len)
    return (tokens // block_size + 1).astype(np.float64)


def traffic_stats(
    selected: mx.array,
    seq_len: int,
    block_size: int,
    head_dim: int,
    dtype_bytes: int = 4,
) -> dict[str, float]:
    """Loaded-block counts and KV bytes for a sparse selection vs dense causal.

    ``selected`` may carry batch and head axes; every element counts, because in
    this model each (batch, head) pair routes independently and reads its own KV.
    Sparse bytes assume only the selected blocks are read (which is what
    :func:`moba.tiled.tiled_attention` does); dense bytes assume every block up to
    and including the query's own.
    """
    counts = selection_counts(selected).astype(mx.float32)
    lead = counts.shape[:-1]
    queries = 1
    for size in lead:
        queries *= size
    length = counts.shape[-1]
    if length != seq_len:
        raise ValueError("selected does not match seq_len")
    sparse_blocks = float(mx.sum(counts))
    query_positions = queries * length
    dense_per_position = dense_counts(seq_len, block_size)
    dense_blocks = float(dense_per_position.sum()) * queries
    elements_per_block = block_size * head_dim
    sparse_bytes = sparse_blocks * elements_per_block * 2 * dtype_bytes
    dense_bytes = dense_blocks * elements_per_block * 2 * dtype_bytes
    return {
        "queries": float(queries),
        "sparse_blocks": sparse_blocks,
        "dense_blocks": dense_blocks,
        "blocks_per_query": sparse_blocks / max(query_positions, 1.0),
        "dense_blocks_per_query": dense_blocks / max(query_positions, 1.0),
        "loaded_fraction_of_dense": sparse_blocks / max(dense_blocks, 1e-30),
        # Mean blocks read per query divided by K, the number of blocks in the
        # sequence.  This is the quantity that should track k / K.
        "loaded_fraction_of_all_blocks": (
            sparse_blocks / max(query_positions, 1.0)
        )
        / max(seq_len // block_size, 1),
        "sparse_kv_bytes": sparse_bytes,
        "dense_kv_bytes": dense_bytes,
        "kv_bytes_fraction_of_dense": sparse_bytes / max(dense_bytes, 1e-30),
    }


def router_recall(
    indices: mx.array,
    selected: mx.array,
    signal_blocks: np.ndarray,
    position: int | None = None,
) -> dict[str, float]:
    """Does the top-k contain the block holding the answer binding?

    ``signal_blocks`` is ``(B,)`` (see :func:`moba.task.signal_block`).  The
    selection has shape ``(B, H, L, k)``.  With ``position`` only that query
    position is scored; otherwise every position is.
    """
    idx = np.array(indices)
    sel = np.array(selected)
    if idx.ndim != 4:
        raise ValueError("expected (batch, heads, length, k) indices")
    batch, heads, length, k = idx.shape
    if position is None:
        chosen = idx, sel
    else:
        chosen = idx[:, :, position], sel[:, :, position]
    idx_v, sel_v = chosen
    hits = np.zeros((batch, heads), np.float64)
    totals = np.zeros((batch, heads), np.float64)
    for b in range(batch):
        target = int(signal_blocks[b])
        for h in range(heads):
            if position is None:
                rows_idx = idx_v[b, h]
                rows_sel = sel_v[b, h]
            else:
                rows_idx = idx_v[b, h][None, :]
                rows_sel = sel_v[b, h][None, :]
            for row in range(rows_idx.shape[0]):
                ok = rows_sel[row]
                totals[b, h] += 1
                if target in set(rows_idx[row][ok].tolist()):
                    hits[b, h] += 1
    return {
        "recall": float(hits.sum() / max(totals.sum(), 1.0)),
        "per_head": (hits.sum(axis=0) / np.maximum(totals.sum(axis=0), 1e-30)).tolist(),
        "rows": float(totals.sum()),
    }


def selection_matrix(
    indices: mx.array,
    selected: mx.array,
    seq_len: int,
    block_size: int,
) -> np.ndarray:
    """``(L, K)`` fraction of (batch, head) rows that selected each block."""
    blocks = block_selection_mask(
        indices, selected, seq_len // block_size
    )
    mask = np.array(blocks.astype(mx.float32))
    lead = mask.shape[:-2]
    scale = 1.0
    for size in lead:
        scale *= size
    return mask.reshape(-1, mask.shape[-2], mask.shape[-1]).sum(axis=0) / scale


def tile_utilisation(
    indices: mx.array,
    selected: mx.array,
    seq_len: int,
    block_size: int,
) -> np.ndarray:
    """``(Q, K)`` tile schedule: share of queries in a query block using a key block.

    Row ``i`` is the logical query block, column ``j`` the logical key block, and
    the value is the fraction of that block's query rows (over every batch and
    head) whose top-k contains ``j``.  A dense causal schedule would have 1.0
    everywhere on and below the diagonal and 0.0 above it; this is the actual
    sparse schedule the kernel would iterate.
    """
    per_query = selection_matrix(indices, selected, seq_len, block_size)
    count = seq_len // block_size
    grouped = per_query.reshape(count, block_size, count)
    return grouped.mean(axis=1)


def schedule_stats(
    indices: mx.array,
    selected: mx.array,
    seq_len: int,
    block_size: int,
) -> dict[str, float]:
    """How much of the dense tile grid the sparse schedule touches."""
    schedule = tile_utilisation(indices, selected, seq_len, block_size)
    count = seq_len // block_size
    live = (schedule > 0).sum()
    dense = count * (count + 1) // 2
    return {
        "live_tiles": float(live),
        "dense_tiles": float(dense),
        "tile_fraction": float(live / dense),
        "mean_occupancy_of_live_tiles": float(
            schedule[schedule > 0].mean() if live else 0.0
        ),
        "mean_occupancy_of_causal_tiles": float(
            schedule[np.tril(np.ones((count, count), bool))].mean()
        ),
    }


def benchmark(fn, warmup: int = 2, iters: int = 5) -> dict[str, float]:
    """Median wall-clock milliseconds for ``fn`` after warm-up."""
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    return {
        "median_ms": float(times[len(times) // 2]),
        "min_ms": float(times[0]),
        "max_ms": float(times[-1]),
        "iters": float(iters),
    }
