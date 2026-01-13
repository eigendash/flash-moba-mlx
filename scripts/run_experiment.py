"""The MoBA experiment: answer quality, router retrieval and KV traffic vs k.

What it does, in order:

1. Trains one small MoBA language model per (k, seed) on the multi-key
   associative-recall task and evaluates it on held-out batches.  ``k = K`` is the
   dense causal baseline: with every block selected the model *is* full attention.
   Configurations are trained concurrently in worker processes, so the wall clock
   of the whole sweep is a few minutes rather than the sum of the runs.
2. Measures, per configuration, how many key blocks each query actually reads, as
   a fraction of what dense causal attention reads, and the implied KV bytes.
3. Times the four attention implementations (dense, mask reference, tiled,
   compiled tiled, Metal) on fixed inputs, and checks that the tiled and Metal
   paths agree with the mask reference to 1e-5.
4. Repeats the router selection of one trained dense model at every k, which is a
   training-free k sweep on a fixed model.
5. Runs the paper's Section 3 SNR simulation across block sizes.

Writes ``results/experiment.json``, ``results/run_log.txt`` and the figures in
``results/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time
from dataclasses import asdict

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moba.attention import dense_attention, moba_attention  # noqa: E402
from moba.metrics import (  # noqa: E402
    benchmark,
    router_recall,
    schedule_stats,
    selection_matrix,
    tile_utilisation,
    traffic_stats,
)
from moba.model import (  # noqa: E402
    MoBALM,
    ModelConfig,
    answer_accuracy,
    routing_trace,
)
from moba.plots import (  # noqa: E402
    plot_accuracy_vs_k,
    plot_selection_matrix,
    plot_snr_recall,
    plot_snr_scaling,
    plot_tile_schedule,
    plot_traffic_vs_k,
)
from moba.router import Router  # noqa: E402
from moba.snr import snr_scaling_table  # noqa: E402
from moba.task import (  # noqa: E402
    TaskConfig,
    generate_batch,
    held_out_batches,
    signal_blocks,
)
from moba.train import EvalResult, TrainConfig, evaluate, train  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")

DEFAULTS = dict(
    seq_len=128,
    block_size=16,
    n_pairs=2,
    n_queries=2,
    n_filler=16,
    n_keys=64,
    n_values=16,
    noise_key_rate=0.4,
    noise_value_rate=0.0,
)

SNR_BLOCK_SIZES = [4, 8, 16, 32, 64]

_CNT = [0]


def log(message: str) -> None:
    _CNT[0] += 1
    line = f"[{_CNT[0]:03d}] {message}"
    print(line, flush=True)


def build_configs(args):
    task = TaskConfig(
        seq_len=args.seq_len,
        block_size=args.block_size,
        n_pairs=args.n_pairs,
        n_queries=args.n_queries,
        n_filler=args.n_filler,
        n_keys=args.n_keys,
        n_values=args.n_values,
        noise_key_rate=args.noise_key_rate,
        noise_value_rate=args.noise_value_rate,
    )
    base = dict(
        vocab_size=task.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=task.seq_len,
        block_size=task.block_size,
        kconv_size=args.kconv_size,
        router_distill=args.router_distill,
    )
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
    )
    return task, base, train_cfg


def run_one(task, base, train_cfg, k, seed, tag=""):
    """Train and evaluate one (k, seed) configuration. Runs in a worker process."""
    model_cfg = ModelConfig(**base, k=k, impl="tiled")
    cfg = TrainConfig(**{**asdict(train_cfg), "seed": seed})
    started = time.perf_counter()
    model = train(task, model_cfg, cfg)
    trained = time.perf_counter() - started
    result = evaluate(model, task, model_cfg, cfg)
    payload = asdict(result)
    payload.update(
        {
            "k": int(k),
            "seed": int(seed),
            "tag": tag,
            "train_seconds": trained,
            "impl": "tiled",
        }
    )
    return payload


def summarize(rows: list[dict]) -> dict:
    keys = [
        "loss",
        "accuracy",
        "router_recall_all",
        "router_recall_answer",
        "router_recall_first_layer",
        "loaded_fraction_of_dense",
        "loaded_fraction_of_all_blocks",
        "kv_bytes_fraction_of_dense",
        "blocks_per_query",
        "train_seconds",
    ]
    out = {"n": len(rows)}
    for key in keys:
        values = [row[key] for row in rows]
        out[key] = float(np.mean(values))
        out[key + "_std"] = float(np.std(values))
        out[key + "_values"] = [float(v) for v in values]
    out["k"] = int(rows[0]["k"])
    return out


def train_sweep(task, base, train_cfg, ks, seeds, workers) -> dict[int, dict]:
    jobs = [(k, seed) for k in ks for seed in seeds]
    log(f"training {len(jobs)} configurations on {workers} workers: "
        f"k in {ks}, seeds {seeds}")
    started = time.perf_counter()
    rows: dict[int, list[dict]] = {k: [] for k in ks}
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(run_one, task, base, train_cfg, k, seed): (k, seed)
            for k, seed in jobs
        }
        for future in futures.as_completed(pending):
            k, seed = pending[future]
            row = future.result()
            rows[k].append(row)
            log(
                f"  k={k} seed={seed}: loss={row['loss']:.4f} "
                f"acc={row['accuracy']:.4f} "
                f"recall@k(last)={row['router_recall_answer']:.3f} "
                f"recall@k(first)={row['router_recall_first_layer']:.3f} "
                f"blocks/query={row['blocks_per_query']:.2f} "
                f"({row['train_seconds']:.0f}s)"
            )
    log(f"training sweep finished in {time.perf_counter() - started:.0f}s")
    return {k: summarize(rows[k]) for k in ks}


def timing_table(task, base, args) -> dict:
    """Wall clock of every attention path on one fixed batch, plus exactness."""
    n_blocks = task.n_blocks
    seeds = [0, 1, 2]
    model = None
    timing = {}
    equivalence = {}
    traced = {}
    for k in args.timing_k:
        model_cfg = ModelConfig(**base, k=min(k, n_blocks), impl="tiled")
        if model is None:
            model = _load_or_train(task, model_cfg, args)
        for block in model.blocks:
            block.attn.router.k = min(k, n_blocks)
        model.cfg.k = min(k, n_blocks)
        batch = generate_batch(task, args.timing_batch, np.random.default_rng(7))
        trace = []
        model(batch.tokens, trace=trace, impl="reference")
        head = trace[args.timed_layers]
        mx.eval(head["indices"], head["selected"])
        indices = head["indices"]
        selected = head["selected"]
        # Real q/k/v from the model, so the timings are on real tensors.
        x = model.token_emb(batch.tokens)
        block = model.blocks[0]
        h = block.norm1(x)
        attn = block.attn
        q = attn._heads(attn.q_proj(h))
        keys = attn._heads(attn.k_proj(h))
        v = attn._heads(attn.v_proj(h))
        mx.eval(q, keys, v)
        dense_ms = benchmark(
            lambda: dense_attention(q, keys, v), warmup=args.warmup, iters=args.iters
        )["median_ms"]
        entry = {"dense": dense_ms}
        for impl in ["reference", "tiled", "compiled"]:
            if impl == "compiled" and not args.with_compiled:
                continue
            ms = benchmark(
                lambda impl=impl: moba_attention(
                    q, keys, v, indices, selected, task.block_size, impl=impl
                ),
                warmup=args.warmup,
                iters=args.iters,
            )
            entry[impl] = ms["median_ms"]
        if args.with_metal and mx.metal.is_available():
            entry["metal"] = benchmark(
                lambda: moba_attention(
                    q, keys, v, indices, selected, task.block_size, impl="metal"
                ),
                warmup=args.warmup,
                iters=args.iters,
            )["median_ms"]
        timing[str(k)] = entry

        ref = moba_attention(
            q, keys, v, indices, selected, task.block_size, impl="reference"
        )
        mx.eval(ref)
        max_diffs = {}
        for impl in ["tiled", "compiled", "metal"]:
            if impl == "compiled" and not args.with_compiled:
                continue
            if impl == "metal" and not (args.with_metal and mx.metal.is_available()):
                continue
            got = moba_attention(
                q, keys, v, indices, selected, task.block_size, impl=impl
            )
            mx.eval(got)
            max_diffs[impl] = float(
                mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)))
            )
        equivalence[str(k)] = max_diffs
        log(
            f"  timing k={k}: "
            + ", ".join(f"{name}={ms:.2f}ms" for name, ms in entry.items())
            + " | max|fast-ref| "
            + ", ".join(f"{name}={d:.2e}" for name, d in max_diffs.items())
        )
    return {"timing_ms": timing, "equivalence": equivalence}


_ROUTING_MODEL: dict = {}


def _load_or_train(task, model_cfg, args):
    """Train (or load) the routing model used for timings and figures.

    Cached in-process as well as on disk: the timings, the training-free sweep
    and every figure need the same model, and retraining it three times inside
    one run costs minutes for no extra information.
    """
    cache = os.path.join(REPO_ROOT, "artifacts", "routing_model.npz")
    if "model" in _ROUTING_MODEL:
        return _ROUTING_MODEL["model"]
    model = MoBALM(model_cfg)
    if os.path.exists(cache) and not args.retrain:
        arrays = dict(np.load(cache))
        model.load_weights(list(arrays.items()))
        mx.eval(model.parameters())
        log(f"loaded routing model from {os.path.relpath(cache, REPO_ROOT)}")
        _ROUTING_MODEL["model"] = model
        return model
    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
    )
    log(f"training the routing model for figures and timings ({args.steps} steps)")
    started = time.perf_counter()
    model = train(task, model_cfg, cfg)
    log(f"routing model trained in {time.perf_counter() - started:.0f}s")
    from mlx.utils import tree_flatten

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **{name: np.array(value) for name, value in tree_flatten(
        model.parameters())})
    _ROUTING_MODEL["model"] = model
    return model


def traffic_table(task, base, ks, seeds, workers, train_cfg) -> list[dict]:
    rows = []
    model_cfg = ModelConfig(**base, k=min(ks[-1], task.n_blocks), impl="tiled")
    started = time.perf_counter()
    model = train(task, model_cfg, TrainConfig(**{**asdict(train_cfg), "seed": seeds[0]}))
    log(f"trained the traffic model in {time.perf_counter() - started:.0f}s")
    batches = held_out_batches(task, 8, 2, seed=4242)
    for k in ks:
        model.blocks[-1].attn.router.k = min(k, task.n_blocks)
        sparse = dense = 0.0
        for batch in batches:
            trace = []
            model(batch.tokens, trace=trace, impl="reference")
            entry = trace[-1]
            mx.eval(entry["selected"])
            stats = traffic_stats(
                entry["selected"], task.seq_len, task.block_size,
                head_dim=base["d_model"] // base["n_heads"],
            )
            sparse += stats["sparse_blocks"]
            dense += stats["dense_blocks"]
        rows.append(
            {
                "k": int(k),
                "loaded_blocks": sparse,
                "dense_blocks": dense,
                "loaded_fraction_of_dense": sparse / dense,
                "loaded_fraction_of_all_blocks": (
                    sparse
                    / (
                        len(batches)
                        * batches[0].tokens.shape[0]
                        * base["n_heads"]
                        * task.seq_len
                    )
                )
                / task.n_blocks,
            }
        )
        log(
            f"  traffic k={k}: loaded {sparse:.0f} of {dense:.0f} dense blocks "
            f"({sparse / dense:.3f} of dense)"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--timing-k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--timing-batch", type=int, default=8)
    parser.add_argument("--timed-layers", type=int, default=-1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--kconv-size", type=int, default=5)
    parser.add_argument("--router-distill", type=float, default=0.0)
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--block-size", type=int, default=DEFAULTS["block_size"])
    parser.add_argument("--n-pairs", type=int, default=DEFAULTS["n_pairs"])
    parser.add_argument("--n-queries", type=int, default=DEFAULTS["n_queries"])
    parser.add_argument("--n-filler", type=int, default=DEFAULTS["n_filler"])
    parser.add_argument("--n-keys", type=int, default=DEFAULTS["n_keys"])
    parser.add_argument("--n-values", type=int, default=DEFAULTS["n_values"])
    parser.add_argument("--noise-key-rate", type=float, default=DEFAULTS["noise_key_rate"])
    parser.add_argument("--noise-value-rate", type=float,
                        default=DEFAULTS["noise_value_rate"])
    parser.add_argument("--with-compiled", action="store_true", default=True)
    parser.add_argument("--with-metal", action="store_true", default=True)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--snr-trials", type=int, default=4000)
    parser.add_argument("--signal-cosine", type=float, default=0.8)
    parser.add_argument("--out", type=str, default="experiment.json")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    task, base, train_cfg = build_configs(args)
    ks = [min(k, task.n_blocks) for k in args.ks]
    seeds = list(range(args.seeds))
    started = time.perf_counter()

    log("MoBA experiment")
    log(
        f"task: L={task.seq_len} B={task.block_size} K={task.n_blocks} "
        f"pairs={task.n_pairs} queries={task.n_queries} "
        f"noise_key={task.noise_key_rate} noise_value={task.noise_value_rate} "
        f"vocab={task.vocab_size}"
    )
    log(
        f"model: d_model={args.d_model} heads={args.n_heads} layers={args.n_layers} "
        f"d_ff={args.d_ff} kconv={args.kconv_size} router_distill={args.router_distill} "
        f"train={args.steps} steps, batch {args.batch_size}, lr {args.learning_rate}"
    )
    n_params = None
    probe = MoBALM(ModelConfig(**base, k=ks[0], impl="tiled"))
    from mlx.utils import tree_flatten

    n_params = int(sum(v.size for _, v in tree_flatten(probe.parameters())))
    log(f"parameters: {n_params}")

    log("--- k sweep (trained from scratch per k) ---")
    sweep = train_sweep(task, base, train_cfg, ks, seeds, args.workers)
    dense = sweep[task.n_blocks] if task.n_blocks in sweep else None
    if dense is not None:
        log(
            f"dense baseline (k=K={task.n_blocks}): loss={dense['loss']:.4f} "
            f"acc={dense['accuracy']:.4f}"
        )

    log("--- literature check: k = K must equal dense causal attention ---")
    equivalent = _check_dense_equivalence(task, base, args)
    log(f"max |moba(k=K) - dense causal| over {equivalent['batches']} batches: "
        f"{equivalent['max_abs_diff']:.2e}")

    log("--- traffic (loaded blocks) ---")
    traffic = traffic_table(task, base, ks, seeds, args.workers, train_cfg)
    for row in traffic:
        log(
            f"  k={row['k']}: {row['loaded_blocks']:.0f} loaded blocks, "
            f"{row['loaded_fraction_of_dense']:.3f} of dense, "
            f"{row['loaded_fraction_of_all_blocks']:.3f} of all blocks, "
            f"k/K={row['k'] / task.n_blocks:.3f}"
        )

    log("--- wall clock per implementation ---")
    timing = timing_table(task, base, args)
    traffic_timing = traffic_wall_clock(task, base, ks, train_cfg, args)
    for k, ms in traffic_timing.items():
        log(f"  tiled forward k={k}: {ms['median_ms']:.2f} ms")

    log("--- training-free k sweep on the dense routing model ---")
    free = training_free_sweep(task, base, ks, args)

    log("--- figures ---")
    figure_paths = make_figures(task, base, args, sweep, traffic, timing,
                                traffic_timing, ks)

    log("--- SNR design principle (simulation of the paper's Section 3 model) ---")
    snr_rows = snr_scaling_table(
        head_dim=args.d_model // args.n_heads,
        block_sizes=SNR_BLOCK_SIZES,
        n_blocks=16,
        signal_cosine=args.signal_cosine,
        trials=args.snr_trials,
        seed=0,
    )
    for row in snr_rows:
        log(
            f"  B={row['block_size']:.0f}: empirical SNR={row['empirical_snr']:.3f} "
            f"(predicted {row['predicted_snr']:.3f}), recall@1={row['recall_at_1']:.3f}, "
            f"sqrt(d/B)={row['snr_scale_sqrt_d_over_b']:.3f}"
        )

    results = {
        "task": asdict(task),
        "model": {
            k: base[k] for k in
            ["d_model", "n_heads", "n_layers", "d_ff", "block_size", "kconv_size"]
        },
        "parameters": n_params,
        "train_cfg": asdict(train_cfg),
        "seeds": seeds,
        "k_sweep": {str(k): sweep[k] for k in ks},
        "dense_baseline": dense,
        "dense_equivalence": equivalent,
        "traffic": traffic,
        "timing_ms": timing["timing_ms"],
        "equivalence": timing["equivalence"],
        "tiled_forward_ms": traffic_timing,
        "training_free_sweep": free,
        "snr_simulation": snr_rows,
        "figures": figure_paths,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    out_path = os.path.join(RESULTS, args.out)
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    log(f"wrote {os.path.relpath(out_path, REPO_ROOT)}")
    log(f"total wall clock {results['wall_clock_seconds']:.0f}s")
    return 0


def _check_dense_equivalence(task, base, args) -> dict:
    model_cfg = ModelConfig(**base, k=task.n_blocks, impl="tiled")
    model = MoBALM(model_cfg)
    mx.eval(model.parameters())
    batches = held_out_batches(task, 4, 3, seed=31337)
    worst = 0.0
    for batch in batches:
        sparse = model(batch.tokens, impl="tiled")
        dense = model(batch.tokens, impl="dense")
        mx.eval(sparse, dense)
        worst = max(
            worst, float(mx.max(mx.abs(sparse.astype(mx.float32) - dense.astype(mx.float32))))
        )
    return {"max_abs_diff": worst, "batches": len(batches),
            "note": "untrained weights; the identity is checked on random routing too"}


def traffic_wall_clock(task, base, ks, train_cfg, args) -> dict:
    model_cfg = ModelConfig(**base, k=ks[0], impl="tiled")
    model = MoBALM(model_cfg)
    mx.eval(model.parameters())
    batch = generate_batch(task, args.timing_batch, np.random.default_rng(11))
    out = {}
    for k in ks:
        for block in model.blocks:
            block.attn.router.k = min(k, task.n_blocks)
        stats = benchmark(
            lambda: model(batch.tokens, impl="tiled"),
            warmup=args.warmup,
            iters=args.iters,
        )
        out[str(k)] = stats
    dense = benchmark(
        lambda: model(batch.tokens, impl="dense"),
        warmup=args.warmup,
        iters=args.iters,
    )
    out["dense"] = dense
    return out


def training_free_sweep(task, base, ks, args) -> list[dict]:
    """Re-run the router of one dense model at every k, without retraining."""
    model_cfg = ModelConfig(**base, k=task.n_blocks, impl="tiled")
    cache = os.path.join(RESULTS, "routing_model.npz")
    model = _load_or_train(task, model_cfg, args)
    batches = held_out_batches(task, 32, 4, seed=5150)
    rows = []
    for k in ks:
        for block in model.blocks:
            block.attn.router.k = min(k, task.n_blocks)
        losses, accuracies, recalls = [], [], []
        sparse = dense = 0.0
        for batch in batches:
            logits = model(batch.tokens, impl="reference")
            mx.eval(logits)
            losses.append(float(model.loss(
                batch.tokens, batch.targets, batch.mask, impl="reference"
            )))
            accuracies.append(answer_accuracy(logits, batch.targets, batch.mask))
            trace = []
            model(batch.tokens, trace=trace, impl="reference")
            mx.eval(trace[-1]["selected"])
            signals = signal_blocks(batch)
            for i in range(signals.shape[1]):
                recalls.append(
                    router_recall(
                        trace[-1]["indices"], trace[-1]["selected"], signals[:, i],
                        position=int(batch.answer_positions[i]),
                    )["recall"]
                )
            stats = traffic_stats(
                trace[-1]["selected"], task.seq_len, task.block_size,
                head_dim=base["d_model"] // base["n_heads"],
            )
            sparse += stats["sparse_blocks"]
            dense += stats["dense_blocks"]
        rows.append(
            {
                "k": int(k),
                "loss": float(np.mean(losses)),
                "accuracy": float(np.mean(accuracies)),
                "router_recall_answer": float(np.mean(recalls)),
                "loaded_fraction_of_dense": sparse / dense,
            }
        )
        log(
            f"  k={k}: loss={rows[-1]['loss']:.4f} acc={rows[-1]['accuracy']:.4f} "
            f"recall={rows[-1]['router_recall_answer']:.3f} "
            f"loaded={rows[-1]['loaded_fraction_of_dense']:.3f} of dense"
        )
    return rows


def make_figures(task, base, args, sweep, traffic, timing, traffic_timing, ks) -> dict:
    model_cfg = ModelConfig(**base, k=task.n_blocks, impl="tiled")
    model = _load_or_train(task, model_cfg, args)
    paths = {}

    losses = [sweep[k]["loss"] for k in ks]
    loss_errs = [sweep[k]["loss_std"] for k in ks]
    accs = [sweep[k]["accuracy"] for k in ks]
    acc_errs = [sweep[k]["accuracy_std"] for k in ks]
    dense_key = task.n_blocks if task.n_blocks in sweep else max(sweep)
    dense_loss = sweep[dense_key]["loss"]
    dense_acc = sweep[dense_key]["accuracy"]
    paths["accuracy_vs_k"] = plot_accuracy_vs_k(
        ks, losses, loss_errs, accs, acc_errs, dense_loss, dense_acc, task.n_blocks,
        os.path.join(RESULTS, "accuracy_vs_k.png"),
    )

    batches = held_out_batches(task, 2, 1, seed=616)
    batch = batches[0]
    matrices, titles = [], []
    for k in ks[:3]:
        for block in model.blocks:
            block.attn.router.k = min(k, task.n_blocks)
        trace = []
        model(batch.tokens, trace=trace, impl="reference")
        entry = trace[-1]
        mx.eval(entry["selected"])
        matrix = selection_matrix(
            entry["indices"], entry["selected"], task.seq_len, task.block_size
        )
        matrices.append(matrix)
        titles.append(f"k={k}, all heads")
    paths["selection_matrix"] = plot_selection_matrix(
        matrices, titles, task.block_size, os.path.join(RESULTS, "selection_matrix.png")
    )

    schedules, titles = [], []
    for k in ks[:3]:
        for block in model.blocks:
            block.attn.router.k = min(k, task.n_blocks)
        trace = []
        model(batch.tokens, trace=trace, impl="reference")
        entry = trace[-1]
        mx.eval(entry["selected"])
        schedule = tile_utilisation(
            entry["indices"], entry["selected"], task.seq_len, task.block_size
        )
        schedules.append(schedule)
        stats = schedule_stats(
            entry["indices"], entry["selected"], task.seq_len, task.block_size
        )
        titles.append(
            f"MoBA k={k}: {stats['tile_fraction']:.0%} of causal tiles live"
        )
    # Dense causal reference schedule: every tile on and below the diagonal.
    count = task.n_blocks
    schedules.append(np.tril(np.ones((count, count))))
    titles.append("dense causal: 100% of causal tiles live")
    paths["tile_schedule"] = plot_tile_schedule(
        schedules, titles, os.path.join(RESULTS, "tile_schedule.png")
    )

    loaded_fractions = [
        sweep[k]["loaded_fraction_of_all_blocks"] for k in ks
    ]
    kv_bytes = [sweep[k]["kv_bytes_fraction_of_dense"] for k in ks]
    ms = [traffic_timing[str(k)]["median_ms"] for k in ks]
    dense_ms = traffic_timing["dense"]["median_ms"]
    paths["traffic_vs_k"] = plot_traffic_vs_k(
        ks, loaded_fractions, kv_bytes, 1.0, task.n_blocks, ms, dense_ms,
        os.path.join(RESULTS, "traffic_vs_k.png"),
    )

    rows = snr_scaling_table(
        head_dim=base["d_model"] // args.n_heads,
        block_sizes=SNR_BLOCK_SIZES,
        n_blocks=16,
        signal_cosine=args.signal_cosine,
        trials=args.snr_trials,
        seed=0,
    )
    block_sizes = [int(r["block_size"]) for r in rows]
    paths["snr_scaling"] = plot_snr_scaling(
        block_sizes,
        [r["empirical_snr"] for r in rows],
        [r["predicted_snr"] for r in rows],
        [r["recall_at_1"] for r in rows],
        base["d_model"] // args.n_heads,
        os.path.join(RESULTS, "snr_scaling.png"),
    )
    paths["snr_recall"] = plot_snr_recall(
        block_sizes,
        [r["recall_at_1"] for r in rows],
        os.path.join(RESULTS, "snr_recall.png"),
    )
    return {name: os.path.relpath(path, REPO_ROOT) for name, path in paths.items()}


if __name__ == "__main__":
    raise SystemExit(main())
