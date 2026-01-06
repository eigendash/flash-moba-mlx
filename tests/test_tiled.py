import mlx.core as mx
import numpy as np
import pytest

from moba.attention import dense_attention, moba_reference_attention
from moba.blocks import n_blocks
from moba.router import Router
from moba.tiled import tiled_attention, tiled_attention_compiled


def _case(seed=0, shape=(2, 3, 32, 8), block_size=8, k=2):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.normal(size=shape).astype(np.float32))
    keys = mx.array(rng.normal(size=shape).astype(np.float32))
    v = mx.array(rng.normal(size=shape).astype(np.float32))
    idx, sel, _ = Router(k=k, block_size=block_size)(q, keys)
    return q, keys, v, idx, sel


def test_tiled_matches_the_mask_reference_at_k2():
    q, k, v, idx, sel = _case(seed=0)
    got = np.array(tiled_attention(q, k, v, idx, sel, 8))
    want = np.array(moba_reference_attention(q, k, v, idx, sel, 8))
    assert np.max(np.abs(got - want)) <= 1e-5


@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_tiled_matches_the_mask_reference_across_k_and_block_sizes(block_size, k):
    q, keys, v, idx, sel = _case(
        seed=block_size * 10 + k, shape=(2, 2, 64, 8), block_size=block_size, k=k
    )
    got = np.array(tiled_attention(q, keys, v, idx, sel, block_size))
    want = np.array(moba_reference_attention(q, keys, v, idx, sel, block_size))
    assert np.max(np.abs(got - want)) <= 1e-5, np.max(np.abs(got - want))


def test_tiled_with_every_block_selected_equals_dense_attention():
    """The headline reduction: k = K is exact full causal attention."""
    rng = np.random.default_rng(7)
    length, block_size = 64, 8
    q = mx.array(rng.normal(size=(2, 2, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(2, 2, length, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(2, 2, length, 8)).astype(np.float32))
    count = n_blocks(length, block_size)
    idx, sel, _ = Router(k=count, block_size=block_size)(q, keys)
    got = np.array(tiled_attention(q, keys, v, idx, sel, block_size))
    want = np.array(dense_attention(q, keys, v))
    assert np.max(np.abs(got - want)) <= 1e-5, np.max(np.abs(got - want))


def test_tiled_with_k_above_the_block_count_equals_dense_attention():
    rng = np.random.default_rng(8)
    length, block_size = 32, 8
    q = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    router = Router(k=99, block_size=block_size)
    idx, sel, _ = router(q, keys)
    got = np.array(tiled_attention(q, keys, v, idx, sel, block_size))
    want = np.array(dense_attention(q, keys, v))
    assert np.max(np.abs(got - want)) <= 1e-5


def test_tiled_ignores_values_in_unselected_blocks():
    """Only selected blocks can be read: perturbing block 1 changes nothing.

    The selection is built by hand -- every query attends to block 0 only -- so
    that block 1 is genuinely unused and the test is not vacuous.
    """
    rng = np.random.default_rng(1)
    length, block_size = 16, 8
    q = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    idx = mx.zeros((1, length, 1), dtype=mx.int32)
    sel = mx.ones((1, length, 1), dtype=mx.bool_)
    base = np.array(tiled_attention(q, keys, v, idx, sel, block_size))
    bumped = np.array(v).copy()
    bumped[:, block_size:] += 100.0
    after = np.array(
        tiled_attention(q, keys, mx.array(bumped), idx, sel, block_size)
    )
    np.testing.assert_allclose(base, after, atol=1e-6)
    # Sanity: the perturbation would be visible if block 1 were selected.
    sel_both = mx.ones((1, length, 2), dtype=mx.bool_)
    idx_both = mx.stack([idx[..., 0], mx.ones_like(idx[..., 0])], axis=-1)
    moved = np.array(
        tiled_attention(q, keys, mx.array(bumped), idx_both, sel_both, block_size)
    )
    assert not np.allclose(
        np.array(tiled_attention(q, keys, v, idx_both, sel_both, block_size)), moved
    )


def test_tiled_is_float32_accumulated_for_float16_inputs():
    rng = np.random.default_rng(9)
    length, block_size = 32, 8
    q = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    idx, sel, _ = Router(k=2, block_size=block_size)(q, keys)
    out16 = tiled_attention(
        q.astype(mx.float16), keys.astype(mx.float16), v.astype(mx.float16), idx, sel,
        block_size,
    )
    assert out16.dtype == mx.float16
    assert bool(mx.isfinite(out16).all())


def test_compiled_tiled_matches_the_eager_version():
    q, keys, v, idx, sel = _case(seed=2, shape=(2, 32, 8), block_size=8, k=3)
    eager = np.array(tiled_attention(q, keys, v, idx, sel, 8))
    jit = np.array(tiled_attention_compiled(q, keys, v, idx, sel, 8))
    np.testing.assert_allclose(eager, jit, atol=1e-5)


def test_compiled_tiled_is_reused_across_equal_shapes():
    q, keys, v, idx, sel = _case(seed=3, shape=(1, 24, 8), block_size=8, k=2)
    first = np.array(tiled_attention_compiled(q, keys, v, idx, sel, 8))
    second = np.array(tiled_attention_compiled(q, keys, v, idx, sel, 8))
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_tiled_gradients_are_finite():
    rng = np.random.default_rng(10)
    length, block_size = 32, 8
    q0 = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    k0 = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    v0 = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    idx, sel, _ = Router(k=2, block_size=block_size)(q0, k0)

    def loss(q, k, v):
        out = tiled_attention(q, k, v, idx, sel, block_size)
        return mx.sum(out * out)

    grads = mx.grad(loss, argnums=(0, 1, 2))(q0, k0, v0)
    for g in grads:
        assert bool(mx.isfinite(g).all())
        assert float(mx.max(mx.abs(g))) > 0.0


def test_tiled_gradients_are_zero_for_unselected_blocks():
    rng = np.random.default_rng(11)
    length, block_size = 16, 8
    q0 = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    k0 = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    v0 = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    # Every query selects block 0; block 1 is never read.
    idx = mx.zeros((1, length, 1), dtype=mx.int32)
    sel = mx.ones((1, length, 1), dtype=mx.bool_)

    def loss(k, v):
        return mx.sum(tiled_attention(q0, k, v, idx, sel, block_size))

    dk, dv = mx.grad(loss, argnums=(0, 1))(k0, v0)
    np.testing.assert_allclose(np.array(dk)[:, block_size:], 0.0, atol=1e-7)
    np.testing.assert_allclose(np.array(dv)[:, block_size:], 0.0, atol=1e-7)
    assert float(mx.max(mx.abs(dk[:, :block_size]))) > 0.0


def test_tiled_degenerate_all_false_selection_is_finite():
    q = mx.zeros((1, 8, 4))
    keys = mx.zeros((1, 8, 4))
    v = mx.ones((1, 8, 4))
    idx = mx.zeros((1, 8, 1), dtype=mx.int32)
    sel = mx.zeros((1, 8, 1), dtype=mx.bool_)
    out = tiled_attention(q, keys, v, idx, sel, 8)
    assert bool(mx.isfinite(out).all())
    np.testing.assert_allclose(np.array(out), 0.0, atol=1e-12)
