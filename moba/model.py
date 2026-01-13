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

from .attention import attention_mask_bias, causal_token_mask, moba_attention
from .blocks import block_valid_mask
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
    # Weight of the auxiliary objective that trains the router (see
    # MoBAAttention.router_loss).  The selection is discrete, so without it the
    # router receives no gradient at all and stays at its initialisation.
    router_distill: float = 0.0

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

    def router_loss(self, queries: mx.array, keys: mx.array, scores: mx.array) -> mx.array:
        """Cross-entropy that trains the router to follow the attention mass.

        The top-k is discrete, so nothing in the forward pass depends on the
        router's scores: ``stop_gradient`` on the indices is necessary for the
        gather but it also removes the only path from the loss back to the
        router.  A first version of this model trained a router that never left
        its initialisation, and recall@k sat at chance for every k.

        The fix is an auxiliary objective that costs one L x L softmax per layer
        during training only.  The block mass ``sum_{s in block j} p(t, s)`` of
        the *same* q and k under dense causal attention is a label-free target
        for where the router ought to look, and the loss is the cross-entropy
        between it and the router's softmax over candidate blocks.
        """
        length = queries.shape[-2]
        block_size = self.cfg.block_size
        count = length // block_size
        scale = queries.shape[-1] ** -0.5
        dense = mx.matmul(queries, mx.swapaxes(keys, -1, -2)) * scale
        bias = attention_mask_bias(
            causal_token_mask(length, length), dense.dtype
        )
        mass = mx.softmax(dense + bias, axis=-1)
        mass = mx.reshape(mass, (*mass.shape[:-1], count, block_size))
        target = mx.stop_gradient(mx.sum(mass, axis=-1))
        valid = mx.broadcast_to(block_valid_mask(length, block_size), scores.shape)
        log_probs = nn.log_softmax(mx.where(valid, scores, -mx.inf), axis=-1)
        # 0 * -inf is NaN, and the target is exactly zero on non-candidate
        # blocks, so the product has to be masked rather than multiplied.
        terms = mx.where(valid, target * log_probs, mx.zeros_like(log_probs))
        return -mx.mean(mx.sum(terms, axis=-1))

    def __call__(
        self,
        x: mx.array,
        trace: list | None = None,
        impl: str | None = None,
        aux: dict | None = None,
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
        if aux is not None and self.cfg.router_distill > 0:
            aux["router"] = aux.get("router", 0.0) + self.router_loss(q, k, scores)
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

    def __call__(self, x, trace=None, impl=None, aux=None):
        x = x + self.attn(self.norm1(x), trace=trace, impl=impl, aux=aux)
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
        aux: dict | None = None,
    ) -> mx.array:
        _, length = tokens.shape
        if length > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {length} exceeds max_seq_len "
                f"{self.cfg.max_seq_len}"
            )
        h = self.token_emb(tokens)
        for block in self.blocks:
            h = block(h, trace=trace, impl=impl, aux=aux)
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
        aux: dict = {}
        logits = self(tokens, trace=trace, impl=impl, aux=aux)
        log_probs = nn.log_softmax(logits, axis=-1)
        picked = mx.take_along_axis(
            log_probs, targets[..., None], axis=-1
        )[..., 0]
        nll = -picked
        if mask is None:
            task_loss = nll.mean()
        else:
            weights = mask.astype(mx.float32)
            task_loss = mx.sum(nll * weights) / mx.maximum(mx.sum(weights), 1.0)
        if self.cfg.router_distill > 0 and "router" in aux:
            return task_loss + self.cfg.router_distill * aux["router"] / len(self.blocks)
        return task_loss


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
