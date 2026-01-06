"""Toy-scale MLX implementation of Mixture of Block Attention (MoBA).

The router, the block statistics and the block-sparse attention paths live in
``moba.blocks``, ``moba.router``, ``moba.attention`` and ``moba.kernel``.  See
the README for what is taken from *Optimizing Mixture of Block Attention*
(arXiv:2511.11571) and what comes from the earlier MoBA paper.
"""

from .blocks import (
    block_centroids,
    block_index,
    block_valid_mask,
    causal_block_centroids,
    causal_block_scores,
    n_blocks,
    prefix_mean_within_block,
)
from .attention import (
    attention_mask_bias,
    causal_token_mask,
    dense_attention,
    moba_attention,
    moba_reference_attention,
    selection_statistics,
)
from .kernel import metal_available, metal_moba_attention
from .tiled import tiled_attention, tiled_attention_compiled
from .router import (
    Router,
    block_selection_mask,
    token_selection_mask,
    topk_blocks,
)

__all__ = [
    "block_centroids",
    "block_index",
    "block_valid_mask",
    "causal_block_centroids",
    "causal_block_scores",
    "n_blocks",
    "prefix_mean_within_block",
    "Router",
    "block_selection_mask",
    "token_selection_mask",
    "topk_blocks",
    "attention_mask_bias",
    "causal_token_mask",
    "dense_attention",
    "moba_attention",
    "moba_reference_attention",
    "selection_statistics",
    "tiled_attention",
    "tiled_attention_compiled",
    "metal_available",
    "metal_moba_attention",
]

__version__ = "0.1.0"
