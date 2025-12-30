import mlx.core as mx
import numpy as np
import pytest

from moba.blocks import (
    block_centroids,
    block_index,
    block_valid_mask,
    causal_block_centroids,
    causal_block_scores,
    n_blocks,
    prefix_mean_within_block,
)


def test_n_blocks_rejects_ragged_lengths():
    assert n_blocks(64, 16) == 4
    with pytest.raises(ValueError):
        n_blocks(65, 16)
    with pytest.raises(ValueError):
        n_blocks(64, 0)


def test_block_centroids_are_plain_block_means():
    rng = np.random.default_rng(0)
    keys = mx.array(rng.normal(size=(2, 12, 5)).astype(np.float32))
    got = np.array(block_centroids(keys, 4))
    want = keys.reshape(2, 3, 4, 5).mean(axis=2)
    assert got.shape == (2, 3, 5)
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_prefix_mean_within_block_uses_only_the_past():
    rng = np.random.default_rng(1)
    keys = mx.array(rng.normal(size=(3, 8, 4)).astype(np.float32))
    got = np.array(prefix_mean_within_block(keys, 4))
    for b in range(3):
        for t in range(8):
            start = (t // 4) * 4
            want = keys[b, start : t + 1].mean(axis=0)
            np.testing.assert_allclose(got[b, t], want, atol=1e-6)


def test_prefix_mean_is_invariant_to_future_keys():
    """Changing a key must not change any prefix mean at or before it."""
    rng = np.random.default_rng(2)
    keys = mx.array(rng.normal(size=(2, 12, 3)).astype(np.float32))
    before = np.array(prefix_mean_within_block(keys, 4))
    bumped = np.array(keys).copy()
    bumped[:, 5] += 100.0
    after = np.array(prefix_mean_within_block(mx.array(bumped), 4))
    # Positions strictly before 5 are unaffected; position 5 and later are not.
    np.testing.assert_allclose(before[:, :5], after[:, :5], atol=1e-6)
    assert not np.allclose(before[:, 5], after[:, 5])
    np.testing.assert_allclose(before[:, 8], after[:, 8], atol=1e-6)


def test_block_valid_mask_is_causal():
    mask = np.array(block_valid_mask(16, 4))
    assert mask.shape == (16, 4)
    for t in range(16):
        own = t // 4
        assert mask[t, : own + 1].all()
        assert not mask[t, own + 1 :].any()
    # A query may never select a block made only of future keys.
    assert not mask[0, 1:].any()


def test_causal_block_centroids_match_a_direct_computation():
    rng = np.random.default_rng(3)
    keys = mx.array(rng.normal(size=(2, 12, 3)).astype(np.float32))
    got = np.array(causal_block_centroids(keys, 4))
    assert got.shape == (2, 12, 3, 3)
    for b in range(2):
        for t in range(12):
            own = t // 4
            for j in range(3):
                if j < own:
                    want = keys[b, j * 4 : (j + 1) * 4].mean(axis=0)
                elif j == own:
                    want = keys[b, own * 4 : t + 1].mean(axis=0)
                else:
                    want = np.zeros(3, np.float32)
                np.testing.assert_allclose(got[b, t, j], want, atol=1e-6)


def test_causal_block_scores_agree_with_the_materialised_centroids():
    rng = np.random.default_rng(4)
    q = mx.array(rng.normal(size=(2, 16, 4)).astype(np.float32))
    k = mx.array(rng.normal(size=(2, 16, 4)).astype(np.float32))
    fast = np.array(causal_block_scores(q, k, 4))
    cents = np.array(causal_block_centroids(k, 4))
    want = np.einsum("btd,btjd->btj", q, cents) / np.sqrt(4.0)
    valid = np.broadcast_to(np.array(block_valid_mask(16, 4))[None, :, :], fast.shape)
    np.testing.assert_allclose(fast[valid], want[valid], atol=1e-5)


def test_causal_block_scores_ignore_future_keys():
    rng = np.random.default_rng(5)
    q = mx.array(rng.normal(size=(1, 20, 6)).astype(np.float32))
    k = mx.array(rng.normal(size=(1, 20, 6)).astype(np.float32))
    base = np.array(causal_block_scores(q, k, 4))
    bumped = np.array(k).copy()
    bumped[:, 17] += 50.0
    after = np.array(causal_block_scores(q, mx.array(bumped), 4))
    # Position 17 lives in block 4, so every candidate score of a query before
    # that block is bit-identical; queries from block 4 onwards can change.
    np.testing.assert_allclose(base[:, :16, :4], after[:, :16, :4], atol=1e-6)
    assert not np.allclose(base[:, 17, :5], after[:, 17, :5])
    assert not np.allclose(base[:, 19, :5], after[:, 19, :5])


def test_block_index_contract():
    idx = np.array(block_index(10, 5))
    np.testing.assert_array_equal(idx, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
