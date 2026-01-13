"""Training and evaluation helpers for the toy MoBA language model.

Kept out of the experiment script so the invariants around it (determinism, the
answer-only objective, the routing statistics collected next to the loss) can be
tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .metrics import router_recall, traffic_stats
from .model import MoBALM, ModelConfig, answer_accuracy, routing_trace
from .task import TaskBatch, TaskConfig, generate_batch, held_out_batches, signal_blocks


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0
    eval_batches: int = 4
    eval_batch_size: int = 32
    eval_seed: int = 999
    trace_layer: int = -1
    eval_impl: str = "reference"
    train_impl: str = "tiled"


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    router_recall_all: float
    router_recall_answer: float
    router_recall_first_layer: float
    loaded_fraction_of_dense: float
    loaded_fraction_of_all_blocks: float
    kv_bytes_fraction_of_dense: float
    blocks_per_query: float
    per_batch_accuracy: list[float] = field(default_factory=list)


def make_model(task: TaskConfig, model_cfg: ModelConfig, seed: int) -> MoBALM:
    mx.random.seed(seed)
    model = MoBALM(model_cfg)
    mx.eval(model.parameters())
    return model


def train(
    task: TaskConfig,
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    progress=None,
) -> MoBALM:
    """Train one model and return it.

    The loop evaluates every step: MLX is lazy, so without that the graph grows
    across thousands of steps and eventually trips the Metal buffer limit.
    """
    model = make_model(task, model_cfg, cfg.seed)
    data_rng = np.random.default_rng(cfg.seed + 12345)
    optimizer = optim.AdamW(
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    def loss_fn(m, tokens, targets, mask):
        return m.loss(tokens, targets, mask=mask, impl=cfg.train_impl)

    grad_fn = nn.value_and_grad(model, loss_fn)
    for step in range(cfg.steps):
        batch = generate_batch(task, cfg.batch_size, data_rng)
        loss, grads = grad_fn(model, batch.tokens, batch.targets, batch.mask)
        optimizer.update(model, grads)
        mx.eval(loss, optimizer.state, model.state)
        if progress is not None and (step % max(1, cfg.steps // 4) == 0):
            progress(step, float(loss))
    return model


def evaluate(
    model: MoBALM,
    task: TaskConfig,
    model_cfg: ModelConfig,
    cfg: TrainConfig,
) -> EvalResult:
    """Held-out loss, answer accuracy and the routing measurements behind it."""
    batches = held_out_batches(
        task, cfg.eval_batch_size, cfg.eval_batches, seed=cfg.eval_seed
    )
    losses, accuracies = [], []
    recalls_all, recalls_answer, recalls_first = [], [], []
    sparse_blocks = dense_blocks = 0.0
    query_positions = 0.0
    head_dim = model_cfg.head_dim
    dtype_bytes = 4
    for batch in batches:
        logits = model(batch.tokens, impl=cfg.eval_impl)
        mx.eval(logits)
        losses.append(float(model.loss(
            batch.tokens, batch.targets, batch.mask, impl=cfg.eval_impl
        )))
        accuracies.append(answer_accuracy(logits, batch.targets, batch.mask))

        trace = []
        model(batch.tokens, trace=trace, impl=cfg.eval_impl)
        entry = trace[cfg.trace_layer]
        first = trace[0]
        mx.eval(entry["indices"], entry["selected"],
               first["indices"], first["selected"])
        signals = signal_blocks(batch)
        # Recall is scored per query slot, so run it once per query index.
        for i in range(signals.shape[1]):
            rec = router_recall(
                entry["indices"],
                entry["selected"],
                signals[:, i],
                position=int(batch.answer_positions[i]),
            )
            recalls_answer.append(rec["recall"])
            recalls_first.append(
                router_recall(
                    first["indices"],
                    first["selected"],
                    signals[:, i],
                    position=int(batch.answer_positions[i]),
                )["recall"]
            )
        # Recall over every query position too, using the first query slot's
        # signal block: a cheap aggregate of "did the router look in the right
        # place anywhere in the sequence".
        recalls_all.append(
            router_recall(
                entry["indices"], entry["selected"], signals[:, 0]
            )["recall"]
        )
        stats = traffic_stats(
            entry["selected"], task.seq_len, task.block_size, head_dim=head_dim,
            dtype_bytes=dtype_bytes,
        )
        sparse_blocks += stats["sparse_blocks"]
        dense_blocks += stats["dense_blocks"]
        query_positions += stats["queries"] * task.seq_len
    return EvalResult(
        loss=float(np.mean(losses)),
        accuracy=float(np.mean(accuracies)),
        router_recall_all=float(np.mean(recalls_all)),
        router_recall_answer=float(np.mean(recalls_answer)),
        router_recall_first_layer=float(np.mean(recalls_first)),
        loaded_fraction_of_dense=sparse_blocks / max(dense_blocks, 1e-30),
        loaded_fraction_of_all_blocks=(sparse_blocks / max(query_positions, 1.0))
        / (task.seq_len // task.block_size),
        kv_bytes_fraction_of_dense=sparse_blocks / max(dense_blocks, 1e-30),
        blocks_per_query=sparse_blocks / max(query_positions, 1.0),
        per_batch_accuracy=accuracies,
    )


def evaluate_batches(
    model: MoBALM,
    batches: list[TaskBatch],
    impl: str = "reference",
) -> tuple[float, float]:
    """Mean loss and accuracy over an explicit list of batches."""
    losses, accuracies = [], []
    for batch in batches:
        logits = model(batch.tokens, impl=impl)
        mx.eval(logits)
        losses.append(float(model.loss(batch.tokens, batch.targets, batch.mask,
                                       impl=impl)))
        accuracies.append(answer_accuracy(logits, batch.targets, batch.mask))
    return float(np.mean(losses)), float(np.mean(accuracies))
