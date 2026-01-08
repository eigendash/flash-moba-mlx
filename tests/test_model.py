import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from moba.attention import dense_attention
from moba.model import (
    MoBALM,
    ModelConfig,
    answer_accuracy,
    parameter_count,
    routing_trace,
)


def _cfg(**kw):
    base = dict(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=32,
        block_size=8,
        k=2,
    )
    base.update(kw)
    return ModelConfig(**base)


def test_config_validates_head_split_and_block_alignment():
    with pytest.raises(ValueError):
        _cfg(d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        _cfg(max_seq_len=30, block_size=8)
    with pytest.raises(ValueError):
        _cfg(k=0)


def test_model_shapes_and_dtypes():
    mx.random.seed(0)
    model = MoBALM(_cfg())
    tokens = mx.random.randint(0, 32, (3, 32))
    logits = model(tokens)
    assert logits.shape == (3, 32, 32)
    assert bool(mx.isfinite(logits).all())


def test_model_rejects_sequences_longer_than_max_seq_len():
    model = MoBALM(_cfg())
    with pytest.raises(ValueError):
        model(mx.zeros((1, 64), dtype=mx.int32))


def test_every_attention_layer_produces_a_routing_decision():
    mx.random.seed(1)
    cfg = _cfg(n_layers=3)
    model = MoBALM(cfg)
    tokens = mx.random.randint(0, 32, (2, 32))
    trace = routing_trace(model, tokens)
    assert len(trace) == cfg.n_layers
    for entry in trace:
        indices = np.array(entry["indices"])
        selected = np.array(entry["selected"])
        assert indices.shape == (2, cfg.n_heads, 32, cfg.k)
        assert selected.shape == indices.shape
        # A query only has t // B + 1 candidate blocks, so the first block of
        # queries can fill a single slot no matter how large k is.
        counts = selected.sum(-1)
        expected = np.minimum(cfg.k, np.arange(32) // cfg.block_size + 1)
        assert (counts == expected[None, None, :]).all()


def test_k_equals_the_block_count_makes_the_model_dense():
    """The headline reduction, at model level: MoBA with k = K is dense attention."""
    mx.random.seed(2)
    length, block_size = 32, 8
    count = length // block_size
    sparse = MoBALM(_cfg(k=count, impl="tiled"))
    tokens = mx.random.randint(0, 32, (2, length))
    got = np.array(sparse(tokens, impl="tiled"))
    # Swap in reference attention over every block and compare.
    ref = np.array(sparse(tokens, impl="reference"))
    dense = np.array(sparse(tokens, impl="dense"))
    assert np.max(np.abs(ref - dense)) <= 1e-5
    assert np.max(np.abs(got - dense)) <= 1e-5


def test_a_single_layer_model_matches_manual_block_sparse_attention():
    """End to end against hand-assembled attention, with a one-layer model."""
    mx.random.seed(3)
    cfg = _cfg(n_layers=1, k=2)
    model = MoBALM(cfg)
    tokens = mx.random.randint(0, 32, (1, 16))
    trace = []
    got = np.array(model(tokens, trace=trace, impl="reference"))
    entry = trace[0]
    # Rebuild the same forward pass outside the model.
    x = model.token_emb(tokens) + model.pos_emb(mx.arange(16))[None]
    block = model.blocks[0]
    h = block.norm1(x)
    attn = block.attn
    q = attn._heads(attn.q_proj(h))
    k = attn._heads(attn.k_proj(h))
    v = attn._heads(attn.v_proj(h))
    out = dense_attention(q, k, v)
    from moba.attention import moba_reference_attention

    out = moba_reference_attention(
        q, k, v, entry["indices"], entry["selected"], cfg.block_size
    )
    merged = mx.transpose(out, (0, 2, 1, 3)).reshape(1, 16, cfg.d_model)
    h = x + attn.o_proj(merged)
    h = h + block.mlp(block.norm2(h))
    want = np.array(model.lm_head(model.norm(h)))
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_loss_is_the_answer_position_cross_entropy():
    mx.random.seed(4)
    model = MoBALM(_cfg())
    tokens = mx.random.randint(0, 32, (2, 16))
    targets = mx.random.randint(0, 32, (2, 16))
    mask = mx.zeros((2, 16), dtype=mx.float32)
    mask = mask.at[:, -1].add(1.0)
    full = model.loss(tokens, targets)
    masked = model.loss(tokens, targets, mask=mask)
    logits = model(tokens)
    log_probs = np.array(nn.log_softmax(logits, -1))
    rows = np.arange(2)
    want = -log_probs[rows, -1, np.array(targets)[:, -1]].mean()
    np.testing.assert_allclose(float(masked), want, atol=1e-5)
    assert not np.isclose(float(full), float(masked))


def test_answer_accuracy_is_one_for_the_argmax():
    logits = mx.zeros((2, 4, 6))
    logits = logits.at[:, :, 3].add(1.0)
    targets = mx.full((2, 4), 3, dtype=mx.int32)
    assert answer_accuracy(logits, targets) == pytest.approx(1.0)
    targets = mx.full((2, 4), 2, dtype=mx.int32)
    assert answer_accuracy(logits, targets) == pytest.approx(0.0)
    mask = mx.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    mixed = mx.array([[2, 2, 2, 3], [2, 2, 2, 2]], dtype=mx.int32)
    assert answer_accuracy(logits, mixed, mask) == pytest.approx(1.0)


def test_gradients_are_finite_through_the_whole_model():
    mx.random.seed(5)
    model = MoBALM(_cfg())
    tokens = mx.random.randint(0, 32, (2, 16))
    targets = mx.random.randint(0, 32, (2, 16))
    mask = mx.zeros((2, 16), dtype=mx.float32).at[:, -1].add(1.0)
    grads = mx.grad(lambda m: m.loss(tokens, targets, mask=mask))(model)
    flat = tree_flatten(grads)
    assert flat, "no gradients returned"
    for _, g in flat:
        assert bool(mx.isfinite(g).all()), "non-finite gradient"
    assert any(float(mx.max(mx.abs(g))) > 0 for _, g in flat)


def test_parameter_count_is_in_the_toy_regime():
    model = MoBALM(
        ModelConfig(
            vocab_size=81,
            d_model=64,
            n_heads=4,
            n_layers=3,
            d_ff=256,
            max_seq_len=256,
            block_size=16,
            k=2,
        )
    )
    count = parameter_count(model)
    assert 100_000 < count < 600_000


def test_metal_path_is_available_as_a_model_implementation():
    if not mx.metal.is_available():
        pytest.skip("Metal is not available")
    mx.random.seed(6)
    model = MoBALM(_cfg())
    tokens = mx.random.randint(0, 32, (1, 16))
    got = np.array(model(tokens, impl="metal"))
    want = np.array(model(tokens, impl="reference"))
    assert np.max(np.abs(got - want)) <= 1e-5


def test_key_convolution_changes_the_routing_but_not_the_shapes():
    mx.random.seed(7)
    plain = MoBALM(_cfg(kconv_size=0))
    convolved = MoBALM(_cfg(kconv_size=5))
    tokens = mx.random.randint(0, 32, (1, 32))
    a = routing_trace(plain, tokens)[0]
    b = routing_trace(convolved, tokens)[0]
    assert np.array(b["scores"]).shape == np.array(a["scores"]).shape
    assert not np.allclose(np.array(a["scores"]), np.array(b["scores"]))
