import mlx.core as mx
import numpy as np
import pytest

from moba.attention import (
    attention_mask_bias,
    causal_token_mask,
    dense_attention,
    moba_reference_attention,
    selection_statistics,
)
from moba.blocks import n_blocks
from moba.router import Router


def test_causal_token_mask_is_lower_triangular():
    mask = np.array(causal_token_mask(5))
    assert mask.shape == (5, 5)
    assert mask.tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_attention_mask_bias_has_zero_on_true_and_neg_inf_on_false():
    bias = np.array(attention_mask_bias(mx.array([[True, False]])))
    assert bias[0, 0] == 0.0
    assert np.isneginf(bias[0, 1])


def test_dense_attention_matches_a_hand_computed_single_query():
    q = mx.array([[[1.0, 0.0]]])
    k = mx.array([[[1.0, 0.0], [0.0, 1.0]]])
    v = mx.array([[[2.0, 0.0], [0.0, 4.0]]])
    out = np.array(dense_attention(q, k, v, scale=1.0))
    # Query 0 sees only key 0.
    np.testing.assert_allclose(out[0, 0], [2.0, 0.0], atol=1e-6)


def test_dense_attention_softmaxes_over_the_visible_prefix():
    q = mx.zeros((1, 3, 2))
    k = mx.ones((1, 3, 2))
    v = mx.array([[[1.0], [2.0], [3.0]]])
    out = np.array(dense_attention(q, k, v, scale=1.0))
    # With a flat score the row is the mean of the visible values.
    np.testing.assert_allclose(out[0, :, 0], [1.0, 1.5, 2.0], atol=1e-6)


def test_dense_attention_is_invariant_to_masked_future_keys():
    rng = np.random.default_rng(0)
    q = mx.array(rng.normal(size=(2, 12, 8)).astype(np.float32))
    k = mx.array(rng.normal(size=(2, 12, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(2, 12, 8)).astype(np.float32))
    base = np.array(dense_attention(q, k, v))
    bumped = np.array(k).copy()
    bumped[:, 11] += 10.0
    after = np.array(dense_attention(q, mx.array(bumped), v))
    # Key 11 is a future key for rows 0..10, so those rows cannot move.
    np.testing.assert_allclose(base[:, :11], after[:, :11], atol=1e-6)
    assert not np.allclose(base[:, 11], after[:, 11])


def test_dense_attention_supports_leading_batch_and_head_axes():
    rng = np.random.default_rng(1)
    q = mx.array(rng.normal(size=(3, 2, 16, 8)).astype(np.float32))
    k = mx.array(rng.normal(size=(3, 2, 16, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(3, 2, 16, 8)).astype(np.float32))
    out = dense_attention(q, k, v)
    assert out.shape == (3, 2, 16, 8)
    assert mx.isfinite(out).all()


def _reference_by_hand(q, k, v, indices, selected, block_size):
    """NumPy: pick out the visible keys per query and softmax over just those."""
    length = q.shape[-2]
    dim = q.shape[-1]
    scale = dim**-0.5
    out = np.zeros((length, dim), np.float32)
    counts = np.zeros(length, np.float32)
    for t in range(length):
        visible = []
        for slot in range(indices.shape[-1]):
            if not selected[t, slot]:
                continue
            block = int(indices[t, slot])
            for s in range(block * block_size, (block + 1) * block_size):
                if s <= t:
                    visible.append(s)
        if not visible:
            continue
        counts[t] = len(visible)
        s = np.array([float(q[t] @ k[j]) for j in visible]) * scale
        p = np.exp(s - s.max())
        p = p / p.sum()
        out[t] = sum(pi * np.asarray(v[j]) for pi, j in zip(p, visible))
    return out, counts


def test_moba_reference_matches_a_by_hand_softmax_over_visible_keys():
    rng = np.random.default_rng(2)
    length, block_size, k = 12, 4, 2
    q = mx.array(rng.normal(size=(1, length, 5)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, length, 5)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, length, 5)).astype(np.float32))
    idx, sel, _ = Router(k=k, block_size=block_size)(q, keys)
    got = np.array(
        moba_reference_attention(q, keys, v, idx, sel, block_size)
    )[0]
    want, _ = _reference_by_hand(
        np.array(q)[0], np.array(keys)[0], np.array(v)[0],
        np.array(idx)[0], np.array(sel)[0], block_size,
    )
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_moba_reference_with_every_block_selected_equals_dense_attention():
    rng = np.random.default_rng(3)
    length, block_size = 32, 8
    count = n_blocks(length, block_size)
    q = mx.array(rng.normal(size=(2, 3, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(2, 3, length, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(2, 3, length, 8)).astype(np.float32))
    idx, sel, _ = Router(k=count, block_size=block_size)(q, keys)
    got = np.array(moba_reference_attention(q, keys, v, idx, sel, block_size))
    want = np.array(dense_attention(q, keys, v))
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_moba_reference_with_k_one_attends_to_exactly_one_block():
    rng = np.random.default_rng(4)
    length, block_size = 16, 4
    q = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    idx, sel, scores = Router(k=1, block_size=block_size)(q, keys)
    got = np.array(
        moba_reference_attention(q, keys, v, idx, sel, block_size, scale=1.0)
    )[0]
    # Rebuild the answer from the argmax block alone, per query.
    masked = np.array(scores)[0]
    for t in range(length):
        own = t // block_size
        for j in range(masked.shape[-1]):
            if j > own:
                masked[t, j] = -np.inf
    for t in range(length):
        best = int(np.argmax(masked[t]))
        assert int(idx[0, t, 0]) == best
        rows = [s for s in range(best * block_size, (best + 1) * block_size) if s <= t]
        s = np.array([float(q[0, t] @ keys[0, j]) for j in rows])
        p = np.exp(s - s.max())
        p /= p.sum()
        want = sum(pi * np.asarray(v[0, j]) for pi, j in zip(p, rows))
        np.testing.assert_allclose(got[t], want, atol=1e-5)


def test_moba_reference_never_reads_a_future_key():
    rng = np.random.default_rng(5)
    length, block_size = 24, 4
    q = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    idx, sel, _ = Router(k=2, block_size=block_size)(q, keys)
    base = np.array(moba_reference_attention(q, keys, v, idx, sel, block_size))
    bumped = np.array(v).copy()
    bumped[:, 20] += 30.0
    after = np.array(
        moba_reference_attention(q, keys, mx.array(bumped), idx, sel, block_size)
    )
    # Only queries whose selected set includes position 20 may change.
    for t in range(length):
        blocks = {
            int(idx[0, t, s]) for s in range(2) if bool(sel[0, t, s])
        }
        if 20 // block_size in blocks and t >= 20:
            continue
        np.testing.assert_allclose(base[0, t], after[0, t], atol=1e-6)


def test_selection_statistics_count_blocks_and_fraction():
    idx = mx.array([[[0, 1], [1, 2]]], dtype=mx.int32)
    sel = mx.array([[[True, True], [True, False]]])
    stats = selection_statistics(idx, sel, 12, 4)
    np.testing.assert_allclose(
        np.array(stats["count"]), [[2.0, 1.0]], atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(stats["fraction"]), [[2 / 3, 1 / 3]], atol=1e-6
    )


@pytest.mark.parametrize("impl", ["reference", "tiled", "compiled", "metal"])
def test_dispatch_agrees_across_implementations(impl):
    from moba.attention import moba_attention

    if impl == "metal" and not mx.metal.is_available():
        pytest.skip("Metal is not available")
    rng = np.random.default_rng(12)
    length, block_size, k = 48, 8, 3
    q = mx.array(rng.normal(size=(2, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(2, length, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(2, length, 8)).astype(np.float32))
    idx, sel, _ = Router(k=k, block_size=block_size)(q, keys)
    got = np.array(moba_attention(q, keys, v, idx, sel, block_size, impl=impl))
    want = np.array(
        moba_attention(q, keys, v, idx, sel, block_size, impl="reference")
    )
    assert np.max(np.abs(got - want)) <= 1e-5


def test_dispatch_rejects_an_unknown_implementation():
    from moba.attention import moba_attention

    q = mx.zeros((1, 8, 4))
    with pytest.raises(ValueError):
        moba_attention(
            q, q, q, mx.zeros((1, 8, 1), mx.int32),
            mx.ones((1, 8, 1), mx.bool_), 8, impl="nope",
        )
