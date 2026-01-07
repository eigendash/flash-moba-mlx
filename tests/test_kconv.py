import mlx.core as mx
import numpy as np
import pytest

from moba.kconv import KeyConvolution, depthwise_causal_conv1d, silu


def test_silu_matches_its_definition():
    x = mx.array([-2.0, 0.0, 3.0])
    want = np.array([-2.0, 0.0, 3.0]) / (1.0 + np.exp(-np.array([-2.0, 0.0, 3.0])))
    np.testing.assert_allclose(np.array(silu(x)), want, atol=1e-6)


def test_kernel_size_one_convolution_is_a_per_channel_scaling():
    rng = np.random.default_rng(0)
    x = mx.array(rng.normal(size=(2, 5, 3)).astype(np.float32))
    w = mx.array([[1.0, 2.0, -1.0]], dtype=mx.float32)
    got = np.array(depthwise_causal_conv1d(x, w, 1))
    np.testing.assert_allclose(got, np.array(x) * np.array([1.0, 2.0, -1.0]), atol=1e-6)


def test_convolution_is_causal():
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(1, 9, 4)).astype(np.float32))
    w = mx.array(rng.normal(size=(5, 4)).astype(np.float32))
    base = np.array(depthwise_causal_conv1d(x, w, 5))
    bumped = np.array(x).copy()
    bumped[:, 6] += 10.0
    after = np.array(depthwise_causal_conv1d(mx.array(bumped), w, 5))
    # A tapped convolution can only look backwards.
    np.testing.assert_allclose(base[:, :6], after[:, :6], atol=1e-6)
    assert not np.allclose(base[:, 6], after[:, 6])


def test_convolution_matches_a_hand_computed_two_tap_case():
    x = mx.array([[[1.0], [2.0], [3.0], [4.0]]])
    w = mx.array([[2.0], [10.0]])
    # y_t = 2 x_t + 10 x_{t-1}, with x_{-1} = 0
    got = np.array(depthwise_causal_conv1d(x, w, 2))[0, :, 0]
    np.testing.assert_allclose(got, [2.0, 14.0, 26.0, 38.0], atol=1e-6)


def test_convolution_is_depthwise():
    rng = np.random.default_rng(2)
    x = mx.array(rng.normal(size=(1, 7, 3)).astype(np.float32))
    w = mx.array(rng.normal(size=(3, 3)).astype(np.float32))
    base = np.array(depthwise_causal_conv1d(x, w, 3))
    bumped = np.array(x).copy()
    bumped[:, :, 0] += 5.0
    after = np.array(depthwise_causal_conv1d(mx.array(bumped), w, 3))
    # Channel 0 moves, channels 1 and 2 do not.
    assert not np.allclose(base[..., 0], after[..., 0])
    np.testing.assert_allclose(base[..., 1:], after[..., 1:], atol=1e-6)


def test_convolution_rejects_a_mismatched_weight():
    with pytest.raises(ValueError):
        depthwise_causal_conv1d(mx.zeros((1, 4, 3)), mx.zeros((2, 2)), 2)


def test_key_convolution_is_the_identity_when_disabled():
    rng = np.random.default_rng(3)
    keys = mx.array(rng.normal(size=(2, 6, 4)).astype(np.float32))
    layer = KeyConvolution(4, kernel_size=5, enabled=False)
    np.testing.assert_array_equal(np.array(layer(keys)), np.array(keys))


def test_key_convolution_starts_near_the_identity_residual():
    rng = np.random.default_rng(4)
    keys = mx.array(rng.normal(size=(1, 8, 4)).astype(np.float32))
    layer = KeyConvolution(4, kernel_size=5)
    out = np.array(layer(keys))
    want = np.array(keys) + np.array(silu(keys))
    np.testing.assert_allclose(out, want, atol=1e-6)


def test_key_convolution_without_residual_or_activation_is_a_bare_convolution():
    rng = np.random.default_rng(5)
    keys = mx.array(rng.normal(size=(1, 8, 4)).astype(np.float32))
    layer = KeyConvolution(4, kernel_size=3, use_activation=False, use_residual=False)
    got = np.array(layer(keys))
    want = np.array(
        depthwise_causal_conv1d(keys, layer.weight, 3)
    )
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_key_convolution_preserves_shape_dtype_and_finiteness():
    rng = np.random.default_rng(6)
    keys = mx.array(rng.normal(size=(3, 2, 16, 8)).astype(np.float32))
    layer = KeyConvolution(8, kernel_size=3)
    out = layer(keys)
    assert out.shape == keys.shape
    assert out.dtype == keys.dtype
    assert bool(mx.isfinite(out).all())


def test_key_convolution_is_causal_as_a_whole():
    rng = np.random.default_rng(7)
    keys = mx.array(rng.normal(size=(1, 12, 4)).astype(np.float32))
    layer = KeyConvolution(4, kernel_size=5)
    base = np.array(layer(keys))
    bumped = np.array(keys).copy()
    bumped[:, 10] += 4.0
    after = np.array(layer(mx.array(bumped)))
    np.testing.assert_allclose(base[:, :10], after[:, :10], atol=1e-6)


def test_key_convolution_has_a_gradient():
    rng = np.random.default_rng(8)
    keys = mx.array(rng.normal(size=(1, 10, 4)).astype(np.float32))
    layer = KeyConvolution(4, kernel_size=3)

    def loss(weight, x):
        conv = depthwise_causal_conv1d(x, weight, 3)
        return mx.sum(silu(conv) ** 2)

    grad = mx.grad(loss)(layer.weight, keys)
    assert bool(mx.isfinite(grad).all())
    assert float(mx.max(mx.abs(grad))) > 0.0
