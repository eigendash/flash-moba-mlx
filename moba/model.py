"""A small causal decoder-only LM whose attention layers are MoBA layers.

Toy scale on purpose: a few hundred thousand parameters, rotary positions on the
queries and keys, an ordinary GELU MLP, and one block-sparse attention layer per
block.  The router is per head and reuses that head's attention query, so it
scores block centroids of exactly the keys the attention head will read -- which
is what Section 2 of arXiv:2511.11571 describes and what the paper's Section 3
statistical model then analyses.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .attention import moba_attention
from .kconv import KeyConvolution
from .router import Router


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int | None = None
    max_seq_len: int = 256
    block_size: int = 16
    k: int = 2
    kconv_size: int = 0
    impl: str = "tiled"
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len % self.block_size:
            raise ValueError("max_seq_len must be a multiple of block_size")
        if self.k < 1:
            raise ValueError("k must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def apply_rope(x: mx.array, base: float = 10000.0, offset: int = 0) -> mx.array:
    """Rotary positions on ``(..., L, d)``.

    Absolute learned positions cannot express "attend to the previous token": the
    score between two positions would be ``p_t^T W p_{t-1}``, which no single
    bilinear form can make uniformly large.  The multi-key recall task needs
    exactly that shift head, so positions are rotary here, as in the paper's base
    model.
    """
    if x.shape[-1] % 2:
        raise ValueError("RoPE needs an even head dimension")
    return mx.fast.rope(
        x,
        dims=x.shape[-1],
        traditional=False,
        base=base,
        scale=1.0,
        offset=offset,
    )


class MoBAAttention(nn.Module):
    """Multi-head attention whose core is block sparse."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.router = Router(k=cfg.k, block_size=cfg.block_size)
        self.key_conv = (
            KeyConvolution(cfg.head_dim, cfg.kconv_size)
            if cfg.kconv_size > 1
            else None
        )

    def _heads(self, x: mx.array) -> mx.array:
        b, length, _ = x.shape
        h, d = self.cfg.n_heads, self.cfg.head_dim
        return mx.transpose(x.reshape(b, length, h, d), (0, 2, 1, 3))

    def __call__(
        self,
        x: mx.array,
        trace: list | None = None,
        impl: str | None = None,
    ) -> mx.array:
        b, length, _ = x.shape
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))
        if self.key_conv is not None:
            k = self.key_conv(k)
        q = apply_rope(q, self.cfg.rope_base)
        k = apply_rope(k, self.cfg.rope_base)
        indices, selected, scores = self.router(q, k)
        out = moba_attention(
            q,
            k,
            v,
            indices,
            selected,
            self.cfg.block_size,
            impl=impl or self.cfg.impl,
        )
        if trace is not None:
            trace.append(
                {
                    "indices": indices,
                    "selected": selected,
                    "scores": scores,
                    "length": length,
                }
            )
        merged = mx.transpose(out, (0, 2, 1, 3)).reshape(b, length, self.cfg.d_model)
        return self.o_proj(merged)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = MoBAAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def __call__(self, x, trace=None, impl=None):
        x = x + self.attn(self.norm1(x), trace=trace, impl=impl)
        return x + self.mlp(self.norm2(x))


class MoBALM(nn.Module):
    """Decoder-only LM with MoBA attention in every block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(
        self,
        tokens: mx.array,
        trace: list | None = None,
        impl: str | None = None,
    ) -> mx.array:
        _, length = tokens.shape
        if length > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {length} exceeds max_seq_len "
                f"{self.cfg.max_seq_len}"
            )
        h = self.token_emb(tokens)
        for block in self.blocks:
            h = block(h, trace=trace, impl=impl)
        return self.lm_head(self.norm(h))

    def loss(
        self,
        tokens: mx.array,
        targets: mx.array,
        mask: mx.array | None = None,
        trace: list | None = None,
        impl: str | None = None,
    ) -> mx.array:
        """Cross-entropy, averaged over ``mask`` when given.

        The toy task only scores the final position, so ``mask`` is how the
        answer-only objective is expressed; without it the loss is the ordinary
        next-token loss over the whole sequence.
        """
        logits = self(tokens, trace=trace, impl=impl)
        log_probs = nn.log_softmax(logits, axis=-1)
        picked = mx.take_along_axis(
            log_probs, targets[..., None], axis=-1
        )[..., 0]
        nll = -picked
        if mask is None:
            return nll.mean()
        weights = mask.astype(mx.float32)
        return mx.sum(nll * weights) / mx.maximum(mx.sum(weights), 1.0)


def parameter_count(model) -> int:
    """Total number of scalar parameters in a module tree."""
    from mlx.utils import tree_flatten

    return int(sum(v.size for _, v in tree_flatten(model.parameters())))


def routing_trace(
    model: MoBALM,
    tokens: mx.array,
    impl: str = "reference",
) -> list:
    """Run a forward pass and return the per-layer routing decisions."""
    trace: list = []
    model(tokens, trace=trace, impl=impl)
    mx.eval([t["indices"] for t in trace] + [t["selected"] for t in trace])
    return trace


def answer_accuracy(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
) -> float:
    """Fraction of scored positions whose argmax equals the target."""
    preds = mx.argmax(logits, axis=-1)
    correct = (preds == targets).astype(mx.float32)
    if mask is None:
        return float(mx.mean(correct))
    weights = mask.astype(mx.float32)
    return float(mx.sum(correct * weights) / mx.maximum(mx.sum(weights), 1.0))
