"""Multi-key associative recall with long filler: a task where selection matters.

The sequence is cut into blocks.  A handful of *pair blocks* each start with a
``(key, value)`` binding; the value's token position is the key's position plus
one, so both live in the same block.  Every other position is filler or a random
*noise key* drawn from the key vocabulary.  The last block carries the query: its
final token is a copy of one of the keys, and the model has to emit the value bound
to it at that same position.

Two properties make the router the interesting part:

* the query key is one key among ``B`` inside its block, so the signal is diluted
  by ``1/B`` in the block centroid -- exactly the ``Delta_mu_eff / B`` term of the
  paper's Eq. 1, which is why the task suits the SNR analysis;
* the answer requires selecting the *pair block* that holds the matching key out
  of ``K`` candidates, roughly ``K - 1`` of which contain unrelated keys.  With
  ``k = 1`` the model must pick exactly one block and be right.

Everything is generated with NumPy for reproducibility and returned as MLX
arrays.  ``mask`` selects the final position, which is the only one scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class TaskConfig:
    seq_len: int = 256
    block_size: int = 16
    n_pairs: int = 3
    n_filler: int = 16
    n_keys: int = 32
    n_values: int = 32
    noise_key_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.seq_len % self.block_size:
            raise ValueError("seq_len must be a multiple of block_size")
        if self.n_blocks < self.n_pairs + 2:
            raise ValueError(
                f"need at least {self.n_pairs + 2} blocks, have {self.n_blocks}"
            )
        if self.n_keys < self.n_pairs + 1:
            raise ValueError("need more key types than pairs")
        if self.n_values < self.n_pairs:
            raise ValueError("need at least one value per pair")

    @property
    def n_blocks(self) -> int:
        return self.seq_len // self.block_size

    @property
    def filler_ids(self) -> tuple[int, int]:
        return 0, self.n_filler

    @property
    def key_ids(self) -> tuple[int, int]:
        return self.n_filler, self.n_filler + self.n_keys

    @property
    def value_ids(self) -> tuple[int, int]:
        return self.n_filler + self.n_keys, self.n_filler + self.n_keys + self.n_values

    @property
    def vocab_size(self) -> int:
        return self.n_filler + self.n_keys + self.n_values


@dataclass
class TaskBatch:
    tokens: mx.array
    targets: mx.array
    mask: mx.array
    pair_blocks: np.ndarray
    target_pair: np.ndarray
    answer: np.ndarray
    query_key: np.ndarray

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def answer_position(self) -> int:
        return int(self.tokens.shape[1]) - 1

    def numpy(self) -> dict:
        return {
            "tokens": np.array(self.tokens),
            "targets": np.array(self.targets),
            "mask": np.array(self.mask),
            "pair_blocks": self.pair_blocks,
            "target_pair": self.target_pair,
            "answer": self.answer,
            "query_key": self.query_key,
        }


def generate_batch(
    cfg: TaskConfig,
    batch_size: int,
    rng: np.random.Generator,
) -> TaskBatch:
    """One seeded batch of the associative-recall task."""
    length, block = cfg.seq_len, cfg.block_size
    n_blocks = cfg.n_blocks
    low_filler, high_filler = cfg.filler_ids
    low_key, high_key = cfg.key_ids
    low_value, high_value = cfg.value_ids

    tokens = rng.integers(low_filler, high_filler, size=(batch_size, length))
    # Noise keys sprinkled through the filler: these are the distractor signals
    # the router has to outrank when it scores a block centroid.
    noise = rng.random((batch_size, length)) < cfg.noise_key_rate
    tokens = np.where(
        noise, rng.integers(low_key, high_key, size=(batch_size, length)), tokens
    ).astype(np.int32)
    horizon = np.arange(length, dtype=np.int32)

    targets = np.zeros((batch_size, length), np.int32)
    pair_blocks = np.zeros((batch_size, cfg.n_pairs), np.int32)
    target_pair = np.zeros((batch_size,), np.int32)
    answer = np.zeros((batch_size,), np.int32)
    query_key = np.zeros((batch_size,), np.int32)

    # Candidate pair blocks exclude block 0 and the query block.
    candidates = np.arange(1, n_blocks - 1)
    key_slots = np.arange(low_key, high_key)

    for b in range(batch_size):
        chosen = rng.choice(candidates, size=cfg.n_pairs, replace=False)
        chosen.sort()
        pair_blocks[b] = chosen
        keys = rng.choice(key_slots, size=cfg.n_pairs + 1, replace=False)
        query_key[b] = keys[0]
        values = rng.choice(
            np.arange(low_value, high_value), size=cfg.n_pairs, replace=False
        )
        which = int(rng.integers(0, cfg.n_pairs))
        target_pair[b] = which
        answer[b] = values[which]
        for p in range(cfg.n_pairs):
            start = int(chosen[p]) * block
            key = int(keys[0] if p == which else keys[p + 1])
            tokens[b, start] = key
            tokens[b, start + 1] = values[p]
            # The rest of the pair block is filler plus distractor keys.  The
            # pair's own key is excluded so that the binding stays unique.
            rest = np.arange(start + 2, start + block)
            if rest.size:
                allowed = key_slots[key_slots != key]
                distract = rng.random(rest.size) < cfg.noise_key_rate
                extra_keys = rng.choice(allowed, size=rest.size, replace=True)
                tokens[b, rest] = np.where(
                    distract,
                    extra_keys,
                    rng.integers(low_filler, high_filler, size=rest.size),
                )
        # The query: the key again, as the last token of the sequence.
        tokens[b, length - 1] = query_key[b]
        targets[b, length - 1] = answer[b]

    mask = np.zeros((batch_size, length), np.float32)
    mask[:, length - 1] = 1.0
    # Next-token targets everywhere, with the last position replaced by the
    # retrieved value.  Only the last position is scored by the experiments.
    shifted = np.concatenate([tokens[:, 1:], tokens[:, -1:]], axis=1)
    targets = np.where(horizon[None] == length - 1, targets, shifted)

    return TaskBatch(
        tokens=mx.array(tokens.astype(np.int32)),
        targets=mx.array(targets.astype(np.int32)),
        mask=mx.array(mask),
        pair_blocks=pair_blocks,
        target_pair=target_pair,
        answer=answer.astype(np.int32),
        query_key=query_key.astype(np.int32),
    )


def held_out_batches(
    cfg: TaskConfig,
    batch_size: int,
    n_batches: int,
    seed: int,
) -> list[TaskBatch]:
    """A fixed evaluation set, generated from its own seed."""
    rng = np.random.default_rng(seed)
    return [generate_batch(cfg, batch_size, rng) for _ in range(n_batches)]


def signal_block(batch: TaskBatch, row: int | None = None) -> np.ndarray:
    """Block indices holding the answer key/value binding, shape ``(B,)``."""
    rows = np.arange(len(batch)) if row is None else np.asarray([row])
    return batch.pair_blocks[rows, batch.target_pair[rows]]
