import mlx.core as mx
import numpy as np
import pytest

from moba.attention import dense_attention, moba_reference_attention
from moba.blocks import n_blocks
from moba.router import Router
from moba.tiled import tiled_attention

from moba.kernel import metal_available, metal_moba_attention

pytestmark = pytest.mark.skipif(
    not metal_available(), reason="Metal is not available on this device"
)


def _case(seed=0, shape=(2, 32, 8), block_size=8, k=2, dtype=mx.float32):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.normal(size=shape).astype(np.float32)).astype(dtype)
    keys = mx.array(rng.normal(size=shape).astype(np.float32)).astype(dtype)
    v = mx.array(rng.normal(size=shape).astype(np.float32)).astype(dtype)
    idx, sel, _ = Router(k=k, block_size=block_size)(q, keys)
    return q, keys, v, idx, sel


@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_metal_matches_the_mask_reference(block_size, k):
    q, keys, v, idx, sel = _case(
        seed=block_size + k, shape=(2, 2, 64, 8), block_size=block_size, k=k
    )
    got = np.array(metal_moba_attention(q, keys, v, idx, sel, block_size))
    want = np.array(moba_reference_attention(q, keys, v, idx, sel, block_size))
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) <= 1e-5, np.max(np.abs(got - want))


def test_metal_matches_the_tiled_path():
    q, keys, v, idx, sel = _case(seed=5, shape=(4, 64, 8), block_size=8, k=3)
    got = np.array(metal_moba_attention(q, keys, v, idx, sel, 8))
    want = np.array(tiled_attention(q, keys, v, idx, sel, 8))
    assert np.max(np.abs(got - want)) <= 1e-5


def test_metal_reduces_to_dense_attention_when_every_block_is_selected():
    rng = np.random.default_rng(6)
    length, block_size = 64, 8
    q = mx.array(rng.normal(size=(2, 2, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(2, 2, length, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(2, 2, length, 8)).astype(np.float32))
    count = n_blocks(length, block_size)
    idx, sel, _ = Router(k=count, block_size=block_size)(q, keys)
    got = np.array(metal_moba_attention(q, keys, v, idx, sel, block_size))
    want = np.array(dense_attention(q, keys, v))
    assert np.max(np.abs(got - want)) <= 1e-5


def test_metal_handles_head_dimensions_that_are_not_a_power_of_two():
    q, keys, v, idx, sel = _case(seed=7, shape=(1, 24, 12), block_size=8, k=2)
    got = np.array(metal_moba_attention(q, keys, v, idx, sel, 8))
    want = np.array(moba_reference_attention(q, keys, v, idx, sel, 8))
    assert np.max(np.abs(got - want)) <= 1e-5


def test_metal_supports_float16_inputs():
    q, keys, v, idx, sel = _case(
        seed=8, shape=(1, 32, 8), block_size=8, k=2, dtype=mx.float16
    )
    got = metal_moba_attention(q, keys, v, idx, sel, 8)
    assert got.dtype == mx.float16
    assert bool(mx.isfinite(got).all())
    want = np.array(tiled_attention(q, keys, v, idx, sel, 8))
    assert np.max(np.abs(np.array(got) - want)) <= 5e-3


def test_metal_is_deterministic():
    q, keys, v, idx, sel = _case(seed=9, shape=(2, 32, 8), block_size=8, k=2)
    first = np.array(metal_moba_attention(q, keys, v, idx, sel, 8))
    for _ in range(3):
        again = np.array(metal_moba_attention(q, keys, v, idx, sel, 8))
        np.testing.assert_array_equal(first, again)


def test_metal_ignores_unselected_blocks():
    rng = np.random.default_rng(10)
    length, block_size = 16, 8
    q = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    idx = mx.zeros((1, length, 1), dtype=mx.int32)
    sel = mx.ones((1, length, 1), dtype=mx.bool_)
    base = np.array(metal_moba_attention(q, keys, v, idx, sel, block_size))
    bumped = np.array(v).copy()
    bumped[:, block_size:] += 50.0
    after = np.array(
        metal_moba_attention(q, keys, mx.array(bumped), idx, sel, block_size)
    )
    np.testing.assert_allclose(base, after, atol=1e-6)


def test_metal_never_reads_past_the_query():
    q, keys, v, idx, sel = _case(seed=11, shape=(1, 32, 8), block_size=8, k=2)
    base = np.array(metal_moba_attention(q, keys, v, idx, sel, 8))
    bumped = np.array(keys).copy()
    bumped[:, 31] += 25.0
    after = np.array(
        metal_moba_attention(q, mx.array(bumped), v, idx, sel, 8)
    )
    np.testing.assert_allclose(base[:, :31], after[:, :31], atol=1e-6)


def test_metal_rejects_an_unsupported_dtype():
    q = mx.zeros((1, 8, 8), dtype=mx.int32)
    with pytest.raises(TypeError):
        metal_moba_attention(q, q, q, mx.zeros((1, 8, 1), mx.int32),
                             mx.ones((1, 8, 1), mx.bool_), 8)


def test_metal_all_masked_selection_is_finite():
    q = mx.zeros((1, 8, 4))
    keys = mx.zeros((1, 8, 4))
    v = mx.ones((1, 8, 4))
    idx = mx.zeros((1, 8, 1), dtype=mx.int32)
    sel = mx.zeros((1, 8, 1), dtype=mx.bool_)
    out = np.array(metal_moba_attention(q, keys, v, idx, sel, 8))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 0.0, atol=1e-12)
