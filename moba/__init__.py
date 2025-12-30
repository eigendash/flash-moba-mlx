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

__all__ = [
    "block_centroids",
    "block_index",
    "block_valid_mask",
    "causal_block_centroids",
    "causal_block_scores",
    "n_blocks",
    "prefix_mean_within_block",
]

__version__ = "0.1.0"
