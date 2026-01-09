import mlx.core as mx
import numpy as np
import pytest

from moba.blocks import n_blocks
from moba.metrics import (
    benchmark,
    dense_counts,
    router_recall,
    schedule_stats,
    selection_counts,
    selection_matrix,
    tile_utilisation,
    traffic_stats,
)
from moba.router import Router
from moba.task import TaskConfig, generate_batch, signal_block


def _selected_from(counts):
    """Build a (1, 1, L, k) selection whose per-query counts are ``counts``."""
    length = len(counts)
    k = max(counts)
    idx = np.zeros((1, 1, length, k), np.int32)
    sel = np.zeros((1, 1, length, k), bool)
    for t, c in enumerate(counts):
        idx[0, 0, t, :c] = np.arange(c)
        sel[0, 0, t, :c] = True
    return mx.array(idx), mx.array(sel)


def test_dense_counts_follow_the_causal_triangle():
    np.testing.assert_array_equal(dense_counts(8, 4), [1, 1, 1, 1, 2, 2, 2, 2])


def test_selection_counts_sum_the_slots():
    sel = mx.array([[[True, True], [True, False], [False, False]]])
    np.testing.assert_allclose(
        np.array(selection_counts(sel)), [[2.0, 1.0, 0.0]], atol=1e-6
    )


def test_traffic_fraction_tracks_k_over_blocks():
    """Loaded blocks as a fraction of dense should fall roughly like k / K."""
    length, block_size = 128, 16
    count = n_blocks(length, block_size)
    fractions = {}
    dense_fractions = {}
    for k in (1, 2, 4, 8, 16):
        idx, sel = _selected_from(
            [min(k, t // block_size + 1) for t in range(length)]
        )
        stats = traffic_stats(sel, length, block_size, head_dim=16)
        fractions[k] = stats["loaded_fraction_of_all_blocks"]
        dense_fractions[k] = stats["loaded_fraction_of_dense"]
        assert stats["sparse_blocks"] == sum(
            min(k, t // block_size + 1) for t in range(length)
        )
        # Once a query has k candidate blocks behind it, the loaded count is
        # exactly k, so its share of all blocks is exactly k / K.
        if k <= count:
            late = int(np.argmax(np.arange(length) // block_size + 1 >= k))
            late_counts = [
                min(k, t // block_size + 1) for t in range(late, length)
            ]
            assert np.mean(late_counts) / count == pytest.approx(k / count)
    # Monotone in k, always at most k / K, and a clear trend towards it.
    keys = sorted(fractions)
    assert all(fractions[a] <= fractions[b] for a, b in zip(keys, keys[1:]))
    for k in keys:
        assert fractions[k] <= k / count + 1e-9
    assert fractions[1] < 0.2 and fractions[4] < 0.5 and fractions[8] < 0.75
    # k = K selects every causally legal block, which is exactly dense causal
    # attention, so the loaded fraction of dense is one -- not k / K, because
    # early queries never had K candidates to begin with.
    assert dense_fractions[count] == pytest.approx(1.0)
    assert dense_fractions[count] > fractions[count]


def test_traffic_bytes_are_two_kv_reads_per_selected_block():
    length, block_size, head_dim = 64, 8, 4
    idx, sel = _selected_from([1] * length)
    stats = traffic_stats(sel, length, block_size, head_dim=head_dim)
    # One block per query, K and V, 4 bytes each.
    assert stats["sparse_blocks"] == length
    assert stats["sparse_kv_bytes"] == length * block_size * head_dim * 2 * 4
    assert stats["dense_kv_bytes"] > stats["sparse_kv_bytes"]
    assert stats["kv_bytes_fraction_of_dense"] == pytest.approx(
        stats["loaded_fraction_of_dense"]
    )


def test_traffic_stats_reject_a_mismatched_length():
    idx, sel = _selected_from([1] * 16)
    with pytest.raises(ValueError):
        traffic_stats(sel, 32, 8, head_dim=4)


def test_traffic_scales_with_the_number_of_heads_and_batches():
    length, block_size = 32, 8
    single = _selected_from([1] * length)
    stats_one = traffic_stats(single[1], length, block_size, head_dim=4)
    doubled = mx.concatenate([single[1], single[1]], axis=0)
    stats_two = traffic_stats(doubled, length, block_size, head_dim=4)
    assert stats_two["sparse_blocks"] == 2 * stats_one["sparse_blocks"]
    assert stats_two["loaded_fraction_of_dense"] == pytest.approx(
        stats_one["loaded_fraction_of_dense"]
    )


def test_router_recall_is_one_when_the_signal_block_is_selected():
    cfg = TaskConfig(
        seq_len=64, block_size=8, n_pairs=2, n_filler=8, n_keys=16, n_values=16
    )
    batch = generate_batch(cfg, 4, np.random.default_rng(0))
    signal = signal_block(batch)
    length, block = cfg.seq_len, cfg.block_size
    # Select exactly the signal block, for every query, every head.
    idx = np.repeat(signal[:, None, None, None], 2, axis=1).repeat(length, axis=2)
    sel = np.ones((4, 2, length, 1), bool)
    stats = router_recall(mx.array(idx.astype(np.int32)), mx.array(sel), signal)
    assert stats["recall"] == pytest.approx(1.0)


def test_router_recall_is_zero_when_the_signal_block_is_never_selected():
    cfg = TaskConfig(
        seq_len=64, block_size=8, n_pairs=2, n_filler=8, n_keys=16, n_values=16
    )
    batch = generate_batch(cfg, 4, np.random.default_rng(1))
    signal = signal_block(batch)
    length = cfg.seq_len
    wrong = ((signal + 1) % (length // cfg.block_size)).astype(np.int32)
    idx = np.repeat(wrong[:, None, None, None], 2, axis=1).repeat(length, axis=2)
    sel = np.ones((4, 2, length, 1), bool)
    stats = router_recall(mx.array(idx), mx.array(sel), signal)
    assert stats["recall"] == pytest.approx(0.0)


def test_router_recall_reports_per_head_and_row_counts():
    cfg = TaskConfig(
        seq_len=32, block_size=8, n_pairs=1, n_filler=8, n_keys=8, n_values=8
    )
    batch = generate_batch(cfg, 2, np.random.default_rng(2))
    signal = signal_block(batch)
    length = cfg.seq_len
    idx = np.repeat(signal[:, None, None, None], 3, axis=1).repeat(length, axis=2)
    sel = np.ones((2, 3, length, 1), bool)
    stats = router_recall(mx.array(idx), mx.array(sel), signal)
    assert len(stats["per_head"]) == 3
    assert stats["rows"] == 2 * 3 * length


def test_selection_matrix_has_the_shape_of_the_block_grid():
    length, block_size, k = 32, 8, 2
    rng = np.random.default_rng(3)
    q = mx.array(rng.normal(size=(2, 4, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(2, 4, length, 8)).astype(np.float32))
    idx, sel, _ = Router(k=k, block_size=block_size)(q, keys)
    matrix = selection_matrix(idx, sel, length, block_size)
    assert matrix.shape == (length, length // block_size)
    assert matrix.max() <= 1.0 + 1e-6
    assert matrix.min() >= 0.0
    # Nothing above the causal block diagonal.
    for t in range(length):
        assert matrix[t, t // block_size + 1 :].sum() == 0.0


def test_tile_utilisation_is_dense_on_the_diagonal_and_sparse_off_it():
    length, block_size, k = 64, 8, 2
    rng = np.random.default_rng(4)
    q = mx.array(rng.normal(size=(2, 2, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(2, 2, length, 8)).astype(np.float32))
    idx, sel, _ = Router(k=k, block_size=block_size)(q, keys)
    schedule = tile_utilisation(idx, sel, length, block_size)
    count = length // block_size
    assert schedule.shape == (count, count)
    # Every query always has a legal block to attend to, so the diagonal is used.
    assert (np.diag(schedule) > 0).all()
    # The first query block has only one candidate.
    assert schedule[0, 1:].sum() == 0.0


def test_schedule_stats_count_live_tiles_against_the_causal_grid():
    length, block_size = 64, 8
    idx, sel = _selected_from([1] * length)
    stats = schedule_stats(idx, sel, length, block_size)
    count = length // block_size
    assert stats["dense_tiles"] == count * (count + 1) // 2
    assert stats["live_tiles"] == count  # one block per query -> one tile per row
    assert stats["tile_fraction"] == pytest.approx(count / stats["dense_tiles"])


def test_schedule_stats_with_every_block_selected_fill_the_causal_grid():
    length, block_size = 64, 8
    count = length // block_size
    idx, sel = _selected_from([t // block_size + 1 for t in range(length)])
    stats = schedule_stats(idx, sel, length, block_size)
    assert stats["tile_fraction"] == pytest.approx(1.0)
    assert stats["mean_occupancy_of_causal_tiles"] == pytest.approx(1.0)


def test_benchmark_returns_ordered_timings():
    out = benchmark(lambda: mx.ones((64, 64)) * 2.0, warmup=1, iters=3)
    assert out["min_ms"] <= out["median_ms"] <= out["max_ms"]
    assert out["iters"] == 3.0
