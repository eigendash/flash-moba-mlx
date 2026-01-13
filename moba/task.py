"""Multi-key associative recall with long filler: a task where selection matters.

The sequence is cut into blocks.  A handful of *pair blocks* each start with a
``(key, value)`` binding; the value sits immediately after its key so both live in
the same block.  Every other position is filler or a random *noise key* drawn from
the key vocabulary.  The tail of the sequence is the query region: the last
``n_queries`` positions are copies of keys that were bound earlier, and at each of
those positions the target is the value bound to that key.  The query key is the
input token there, so the router at that position sees the key identity directly
instead of having to recover it from a preceding token.

Three properties make the router the interesting part:

* the query key is one key among ``B`` inside its block, so the signal is diluted
  by ``1/B`` in the block centroid -- exactly the ``Delta_mu_eff / B`` term of the
  paper's Eq. 1, which is why this task suits the SNR analysis;
* answering requires selecting the *pair block* that holds the matching key out of
  ``K`` candidates, roughly ``K - 1`` of which contain unrelated keys, and with
  ``k = 1`` the model must pick exactly one block and be right.  The scorer at a
  query position is the key token itself, so a layer-0 router can decide where to
  look from the token alone; what still has to be learned is the two-hop
  induction circuit that turns "I found the key" into "read the token after it";
* supervision happens at ``n_queries`` positions per sequence rather than one, so
  a few hundred steps are enough to see the effect of ``k``.

Everything is generated with NumPy for reproducibility and returned as MLX
arrays.  ``mask`` selects the query positions, the only ones scored by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np


@dataclass
class TaskConfig:
    seq_len: int = 128
    block_size: int = 16
    n_pairs: int = 6
    n_queries: int = 4
    n_filler: int = 16
    n_keys: int = 32
    n_values: int = 16
    noise_key_rate: float = 0.4
    noise_value_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.seq_len % self.block_size:
            raise ValueError("seq_len must be a multiple of block_size")
        if self.n_pairs > self.n_blocks - 2:
            raise ValueError(
                f"need at least {self.n_pairs + 2} blocks, have {self.n_blocks}"
            )
        if self.n_queries < 1 or self.n_queries > self.n_pairs:
            raise ValueError("need 1 <= n_queries <= n_pairs")
        if self.n_keys < self.n_pairs:
            raise ValueError("need at least as many key types as pairs")
        if self.n_values < self.n_pairs:
            raise ValueError("need at least as many values as pairs")
        if self.n_queries >= self.block_size:
            raise ValueError(
                "the query region must fit inside the final block; reduce "
                "n_queries or increase block_size"
            )

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

    @property
    def answer_positions(self) -> np.ndarray:
        """Positions of the query keys, in query order."""
        return (self.seq_len - self.n_queries + np.arange(self.n_queries)).astype(
            np.int64
        )


@dataclass
class TaskBatch:
    tokens: mx.array
    targets: mx.array
    mask: mx.array
    pair_blocks: np.ndarray
    query_pair: np.ndarray
    answer: np.ndarray
    query_key: np.ndarray
    answer_positions: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_queries(self) -> int:
        return int(self.answer_positions.shape[0])

    @property
    def answer_position(self) -> int:
        return int(self.tokens.shape[1]) - 1

    def numpy(self) -> dict:
        return {
            "tokens": np.array(self.tokens),
            "targets": np.array(self.targets),
            "mask": np.array(self.mask),
            "pair_blocks": self.pair_blocks,
            "query_pair": self.query_pair,
            "answer": self.answer,
            "query_key": self.query_key,
            "answer_positions": self.answer_positions,
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
    key_slots = np.arange(low_key, high_key)
    answer_positions = cfg.answer_positions

    tokens = np.zeros((batch_size, length), np.int32)
    targets = np.zeros((batch_size, length), np.int32)
    pair_blocks = np.zeros((batch_size, cfg.n_pairs), np.int32)
    query_pair = np.zeros((batch_size, cfg.n_queries), np.int32)
    answer = np.zeros((batch_size, cfg.n_queries), np.int32)
    query_key = np.zeros((batch_size, cfg.n_queries), np.int32)

    # Block 0 is filler and the last block carries the queries.
    candidates = np.arange(1, n_blocks - 1)
    for b in range(batch_size):
        chosen = np.sort(rng.choice(candidates, size=cfg.n_pairs, replace=False))
        pair_blocks[b] = chosen
        keys = rng.choice(key_slots, size=cfg.n_pairs, replace=False)
        values = rng.choice(
            np.arange(low_value, high_value), size=cfg.n_pairs, replace=False
        )
        # Noise keys are drawn only from keys that hold no binding in this row.
        # Otherwise a distractor can repeat a binding key, the same key then
        # points at two different values, and the task is ambiguous rather than
        # hard -- which is exactly what a first attempt at this generator did.
        allowed = key_slots[~np.isin(key_slots, keys)]
        if allowed.size == 0:
            raise ValueError("not enough key types to fill the noise positions")
        row = rng.integers(low_filler, high_filler, size=length)
        noise = rng.random(length) < cfg.noise_key_rate
        row = np.where(noise, rng.choice(allowed, size=length), row).astype(np.int32)
        if cfg.noise_value_rate > 0:
            # Value tokens sprinkled through the filler too.  Without this the
            # task can be solved by narrowing to the values that appear in the
            # sequence, which caps accuracy at 1 / n_pairs without any retrieval.
            value_noise = rng.random(length) < cfg.noise_value_rate
            row = np.where(
                value_noise,
                rng.integers(low_value, high_value, size=length),
                row,
            ).astype(np.int32)
        for p in range(cfg.n_pairs):
            start = int(chosen[p]) * block
            row[start] = keys[p]
            row[start + 1] = values[p]
            # The rest of the pair block is filler plus distractor keys.
            rest = np.arange(start + 2, start + block)
            if rest.size:
                distract = rng.random(rest.size) < cfg.noise_key_rate
                row[rest] = np.where(
                    distract,
                    rng.choice(allowed, size=rest.size),
                    rng.integers(low_filler, high_filler, size=rest.size),
                )
                if cfg.noise_value_rate > 0:
                    value_noise = rng.random(rest.size) < cfg.noise_value_rate
                    row[rest] = np.where(
                        value_noise,
                        rng.integers(low_value, high_value, size=rest.size),
                        row[rest],
                    ).astype(np.int32)
        asked = rng.choice(cfg.n_pairs, size=cfg.n_queries, replace=False)
        query_pair[b] = asked
        for i, p in enumerate(asked):
            pos = int(answer_positions[i])
            row[pos] = keys[p]
            query_key[b, i] = keys[p]
            answer[b, i] = values[p]
            targets[b, pos] = values[p]
        tokens[b] = row

    mask = np.zeros((batch_size, length), np.float32)
    mask[:, answer_positions] = 1.0
    # Next-token targets everywhere, with the answer slots replaced by the
    # retrieved values.  The experiments score the answer slots only.
    shifted = np.concatenate([tokens[:, 1:], tokens[:, -1:]], axis=1)
    horizon = np.arange(length)
    targets = np.where(
        np.isin(horizon, answer_positions)[None], targets, shifted
    )

    return TaskBatch(
        tokens=mx.array(tokens.astype(np.int32)),
        targets=mx.array(targets.astype(np.int32)),
        mask=mx.array(mask),
        pair_blocks=pair_blocks,
        query_pair=query_pair,
        answer=answer.astype(np.int32),
        query_key=query_key.astype(np.int32),
        answer_positions=answer_positions,
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


def signal_blocks(batch: TaskBatch) -> np.ndarray:
    """``(B, n_queries)`` block indices holding each query's key/value binding."""
    return batch.pair_blocks[np.arange(len(batch))[:, None], batch.query_pair]


def signal_block(batch: TaskBatch, row: int | None = None) -> np.ndarray:
    """Backwards-compatible alias: the first query's signal block per row."""
    return signal_blocks(batch)[:, 0] if row is None else signal_blocks(batch)[row]
