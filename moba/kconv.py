"""Key convolution: the paper's second design principle.

Appendix C of arXiv:2511.11571 ("Key Convolution Design") describes a short
depthwise causal convolution applied to keys, with kernel size 3 or 5, an
activation and a residual connection.  The motivation is the ``Delta_mu_eff`` term
of the paper's SNR: if semantically related tokens appear near each other, the
convolution smears each key into its neighbours so that several *related* keys
land in the same block, and the block centroid then carries more signal instead of
being diluted by ``1/B``.

The reading implemented here is: left-pad by ``kernel_size - 1`` so that the
convolution is strictly causal, convolve depthwise over the sequence axis, apply
SiLU, and add the input back:

    y_t = x_t + SiLU( sum_{i=0}^{c-1} w_i * x_{t-i} )

with no bias and one kernel per (tap, channel).  The paper does not give the
initialisation, the presence of a bias, or the exact residual form, so those are
choices; they are listed in the README.
"""

from __future__ import annotations

import mlx.core as mx


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def depthwise_causal_conv1d(
    inputs: mx.array,
    weight: mx.array,
    kernel_size: int,
) -> mx.array:
    """``(..., L, d)`` depthwise causal convolution with ``weight`` of shape ``(c, d)``."""
    if weight.shape != (kernel_size, inputs.shape[-1]):
        raise ValueError(
            f"weight must be ({kernel_size}, {inputs.shape[-1]}), got {weight.shape}"
        )
    length = inputs.shape[-2]
    pad = mx.zeros((*inputs.shape[:-2], kernel_size - 1, inputs.shape[-1]), inputs.dtype)
    padded = mx.concatenate([pad, inputs], axis=-2)
    out = mx.zeros_like(inputs)
    for tap in range(kernel_size):
        start = kernel_size - 1 - tap
        window = padded[..., start : start + length, :]
        out = out + window * weight[tap]
    return out


class KeyConvolution:
    """Optional depthwise causal short convolution on keys.

    ``kernel_size=1`` with the default initialisation (weight 1, residual) makes
    the layer ``x + SiLU(x)``; ``enabled=False`` makes it the identity, which is
    how the experiment ablates it.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        enabled: bool = True,
        use_activation: bool = True,
        use_residual: bool = True,
    ) -> None:
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.enabled = bool(enabled)
        self.use_activation = bool(use_activation)
        self.use_residual = bool(use_residual)
        # Near-identity start: tap 0 looks at the current key, later taps start
        # at zero, which keeps an untrained model close to the no-convolution
        # baseline instead of injecting noise into the router immediately.
        weight = mx.zeros((self.kernel_size, self.dim), dtype=mx.float32)
        weight = weight.at[0].add(mx.ones((self.dim,), dtype=mx.float32))
        self.weight = weight

    def __call__(self, keys: mx.array) -> mx.array:
        if not self.enabled:
            return keys
        dtype = keys.dtype
        conv = depthwise_causal_conv1d(
            keys.astype(mx.float32), self.weight, self.kernel_size
        )
        if self.use_activation:
            conv = silu(conv)
        out = conv + keys.astype(mx.float32) if self.use_residual else conv
        return out.astype(dtype)

    def parameters(self) -> dict[str, mx.array]:
        return {"weight": self.weight}
