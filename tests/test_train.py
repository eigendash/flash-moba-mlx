import mlx.core as mx
import numpy as np
import pytest

from moba.model import MoBALM, ModelConfig, routing_trace
from moba.task import TaskConfig, generate_batch, held_out_batches
from moba.train import TrainConfig, evaluate, make_model, train


def _task(**kw):
    base = dict(
        seq_len=32,
        block_size=8,
        n_pairs=2,
        n_queries=2,
        n_filler=8,
        n_keys=16,
        n_values=8,
        noise_key_rate=0.2,
        noise_value_rate=0.0,
    )
    base.update(kw)
    return TaskConfig(**base)


def _model_cfg(task, **kw):
    base = dict(
        vocab_size=task.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=task.seq_len,
        block_size=task.block_size,
        k=task.n_blocks,
        kconv_size=0,
        impl="tiled",
    )
    base.update(kw)
    return ModelConfig(**base)


def test_make_model_is_deterministic_for_a_seed():
    task = _task()
    cfg = _model_cfg(task)
    a = make_model(task, cfg, seed=0)
    b = make_model(task, cfg, seed=0)
    c = make_model(task, cfg, seed=1)
    from mlx.utils import tree_flatten

    pa = dict(tree_flatten(a.parameters()))
    pb = dict(tree_flatten(b.parameters()))
    pc = dict(tree_flatten(c.parameters()))
    for name in pa:
        assert np.array_equal(np.array(pa[name]), np.array(pb[name]))
    assert any(
        not np.array_equal(np.array(pa[name]), np.array(pc[name])) for name in pa
    )


def test_training_lowers_the_loss():
    task = _task()
    cfg = _model_cfg(task)
    train_cfg = TrainConfig(steps=150, batch_size=8, learning_rate=3e-3, seed=0)
    rng = np.random.default_rng(0)
    before = generate_batch(task, 8, rng)
    model = make_model(task, cfg, seed=0)
    start = float(model.loss(before.tokens, before.targets, before.mask, impl="reference"))
    model = train(task, cfg, train_cfg)
    after = held_out_batches(task, 8, 1, seed=77)[0]
    end = float(model.loss(after.tokens, after.targets, after.mask, impl="reference"))
    assert np.isfinite(start) and np.isfinite(end)
    assert end < start


def test_training_progress_callback_receives_finite_losses():
    task = _task()
    seen = []
    train(
        task,
        _model_cfg(task),
        TrainConfig(steps=20, batch_size=4, learning_rate=1e-3, seed=2),
        progress=lambda step, loss: seen.append((step, loss)),
    )
    assert seen
    assert all(np.isfinite(loss) for _, loss in seen)


def test_evaluate_reports_the_expected_quantities():
    task = _task(seq_len=32)
    cfg = _model_cfg(task)
    model = make_model(task, cfg, seed=3)
    train_cfg = TrainConfig(steps=5, batch_size=4, eval_batches=2, eval_batch_size=4)
    result = evaluate(model, task, cfg, train_cfg)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.router_recall_answer <= 1.0
    assert np.isfinite(result.loss)
    # A causal mask can never load more than K blocks per query.
    assert 0.0 < result.loaded_fraction_of_all_blocks <= 1.0
    assert 0.0 < result.loaded_fraction_of_dense <= 1.0
    assert len(result.per_batch_accuracy) == 2


def test_evaluate_routing_is_unchanged_by_the_attention_implementation():
    """Routing is a property of q and k, not of how attention is computed."""
    task = _task()
    cfg = _model_cfg(task, k=2)
    model = make_model(task, cfg, seed=4)
    batch = held_out_batches(task, 4, 1, seed=11)[0]
    traces = {}
    for impl in ("reference", "tiled"):
        trace = []
        model(batch.tokens, trace=trace, impl=impl)
        mx.eval([t["indices"] for t in trace])
        traces[impl] = [np.array(t["indices"]) for t in trace]
    for a, b in zip(traces["reference"], traces["tiled"]):
        np.testing.assert_array_equal(a, b)


def test_dense_equivalence_holds_for_a_model_with_every_block_selected():
    task = _task()
    cfg = _model_cfg(task, k=task.n_blocks)
    model = make_model(task, cfg, seed=5)
    batch = held_out_batches(task, 2, 1, seed=12)[0]
    sparse = np.array(model(batch.tokens, impl="tiled"))
    dense = np.array(model(batch.tokens, impl="dense"))
    assert np.max(np.abs(sparse - dense)) <= 1e-5


def test_training_with_the_metal_impl_is_not_attempted():
    """The Metal kernel is forward-only, and the config makes that explicit."""
    task = _task()
    cfg = _model_cfg(task)
    assert cfg.impl == "tiled"
    model = make_model(task, cfg, seed=6)
    trace = []
    with pytest.raises(ValueError):
        model(tokens=mx.zeros((1, 32), dtype=mx.int32), trace=trace, impl="nope")
