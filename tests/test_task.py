import mlx.core as mx
import numpy as np
import pytest

from moba.task import (
    TaskConfig,
    generate_batch,
    held_out_batches,
    signal_block,
    signal_blocks,
)


def _cfg(**kw):
    base = dict(
        seq_len=64,
        block_size=8,
        n_pairs=3,
        n_queries=2,
        n_filler=8,
        n_keys=16,
        n_values=16,
        noise_key_rate=0.4,
        noise_value_rate=0.0,
    )
    base.update(kw)
    return TaskConfig(**base)


def test_config_rejects_impossible_layouts():
    with pytest.raises(ValueError):
        _cfg(seq_len=65)
    with pytest.raises(ValueError):
        _cfg(seq_len=24, block_size=8, n_pairs=2)  # only 3 blocks
    with pytest.raises(ValueError):
        _cfg(n_keys=2, n_pairs=3)
    with pytest.raises(ValueError):
        _cfg(n_values=2, n_pairs=3)
    with pytest.raises(ValueError):
        _cfg(n_queries=0)
    with pytest.raises(ValueError):
        _cfg(n_queries=4, n_pairs=3)
    with pytest.raises(ValueError):
        _cfg(n_queries=8, n_pairs=8, block_size=8)  # query region will not fit


def test_vocabulary_ranges_are_disjoint_and_cover_the_vocab():
    cfg = _cfg()
    f0, f1 = cfg.filler_ids
    k0, k1 = cfg.key_ids
    v0, v1 = cfg.value_ids
    assert f1 == k0 and k1 == v0
    assert v1 == cfg.vocab_size
    assert f1 - f0 == cfg.n_filler
    assert k1 - k0 == cfg.n_keys
    assert v1 - v0 == cfg.n_values


def test_batch_shapes_and_position_dtypes():
    cfg = _cfg()
    batch = generate_batch(cfg, 4, np.random.default_rng(0))
    assert batch.tokens.shape == (4, cfg.seq_len)
    assert batch.targets.shape == (4, cfg.seq_len)
    assert batch.mask.shape == (4, cfg.seq_len)
    assert batch.tokens.dtype == mx.int32
    assert batch.targets.dtype == mx.int32
    assert len(batch) == 4
    assert batch.n_queries == cfg.n_queries
    assert batch.answer_position == cfg.seq_len - 1


def test_answer_positions_are_the_last_positions_of_the_sequence():
    cfg = _cfg()
    positions = cfg.answer_positions
    assert positions.shape == (cfg.n_queries,)
    assert positions[-1] == cfg.seq_len - 1
    assert np.all(np.diff(positions) == 1)


def test_only_the_query_positions_are_scored():
    cfg = _cfg()
    batch = generate_batch(cfg, 4, np.random.default_rng(1))
    mask = np.array(batch.mask)
    for position in cfg.answer_positions:
        assert mask[:, position].tolist() == [1.0] * 4
    assert mask.sum() == 4 * cfg.n_queries


def test_targets_are_the_next_token_except_at_the_query_positions():
    cfg = _cfg()
    batch = generate_batch(cfg, 3, np.random.default_rng(2))
    tokens = np.array(batch.tokens)
    targets = np.array(batch.targets)
    keep = np.ones(cfg.seq_len - 1, bool)
    keep[cfg.answer_positions[cfg.answer_positions < cfg.seq_len - 1]] = False
    np.testing.assert_array_equal(
        targets[:, :-1][:, keep], tokens[:, 1:][:, keep]
    )
    np.testing.assert_array_equal(targets[:, cfg.answer_positions], batch.answer)


def test_each_bound_value_sits_right_after_its_key():
    cfg = _cfg()
    batch = generate_batch(cfg, 8, np.random.default_rng(3))
    tokens = np.array(batch.tokens)
    for b in range(len(batch)):
        for p in range(cfg.n_pairs):
            start = int(batch.pair_blocks[b, p]) * cfg.block_size
            value = int(tokens[b, start + 1])
            low, high = cfg.value_ids
            assert low <= value < high


def test_each_query_key_is_bound_to_exactly_one_value():
    """The task is unambiguous: no binding key is reused as noise."""
    cfg = _cfg()
    batch = generate_batch(cfg, 8, np.random.default_rng(4))
    tokens = np.array(batch.tokens)
    blocks = signal_blocks(batch)
    for b in range(len(batch)):
        for i in range(cfg.n_queries):
            key = int(batch.query_key[b, i])
            assert int((tokens[b] == key).sum()) == 2
            start = int(blocks[b, i]) * cfg.block_size
            assert tokens[b, start] == key
            assert tokens[b, start + 1] == batch.answer[b, i]
            assert tokens[b, int(cfg.answer_positions[i])] == key


def test_pair_blocks_are_distinct_and_exclude_the_query_block():
    cfg = _cfg()
    batch = generate_batch(cfg, 16, np.random.default_rng(6))
    for b in range(len(batch)):
        blocks = batch.pair_blocks[b]
        assert len(set(blocks.tolist())) == cfg.n_pairs
        assert blocks.min() >= 1
        assert blocks.max() <= cfg.n_blocks - 2


def test_signal_blocks_are_inside_the_sequence_and_causal():
    cfg = _cfg()
    batch = generate_batch(cfg, 8, np.random.default_rng(7))
    blocks = signal_blocks(batch)
    assert blocks.shape == (8, cfg.n_queries)
    assert (blocks >= 1).all()
    assert (blocks < cfg.n_blocks - 1).all()
    assert (((blocks + 1) * cfg.block_size - 1) < cfg.seq_len - 1).all()
    np.testing.assert_array_equal(signal_block(batch), blocks[:, 0])


def test_generation_is_deterministic_for_a_seed():
    cfg = _cfg()
    a = generate_batch(cfg, 4, np.random.default_rng(8))
    b = generate_batch(cfg, 4, np.random.default_rng(8))
    np.testing.assert_array_equal(np.array(a.tokens), np.array(b.tokens))
    np.testing.assert_array_equal(np.array(a.targets), np.array(b.targets))
    np.testing.assert_array_equal(a.pair_blocks, b.pair_blocks)


def test_two_seeds_give_different_sequences():
    cfg = _cfg()
    a = generate_batch(cfg, 4, np.random.default_rng(9))
    b = generate_batch(cfg, 4, np.random.default_rng(10))
    assert not np.array_equal(np.array(a.tokens), np.array(b.tokens))


def test_held_out_batches_are_stable_and_distinct():
    cfg = _cfg()
    first = held_out_batches(cfg, 4, 3, seed=11)
    second = held_out_batches(cfg, 4, 3, seed=11)
    assert len(first) == 3
    for a, b in zip(first, second):
        np.testing.assert_array_equal(np.array(a.tokens), np.array(b.tokens))
    assert not np.array_equal(
        np.array(first[0].tokens), np.array(first[1].tokens)
    )


def test_noise_rates_control_how_many_key_and_value_tokens_appear():
    quiet = _cfg(noise_key_rate=0.0, noise_value_rate=0.0)
    keys_only = _cfg(noise_key_rate=0.9, noise_value_rate=0.0)
    values_only = _cfg(noise_key_rate=0.0, noise_value_rate=0.9)
    rng = np.random.default_rng(12)
    quiet_tokens = np.array(generate_batch(quiet, 8, rng).tokens)
    key_tokens = np.array(generate_batch(keys_only, 8, rng).tokens)
    value_tokens = np.array(generate_batch(values_only, 8, rng).tokens)
    low_key, high_key = quiet.key_ids

    def key_fraction(tokens):
        return ((tokens >= low_key) & (tokens < high_key)).mean()

    def value_fraction(tokens, cfg):
        low, high = cfg.value_ids
        return ((tokens >= low) & (tokens < high)).mean()

    assert key_fraction(key_tokens) > 0.6 > key_fraction(quiet_tokens)
    assert value_fraction(value_tokens, values_only) > 0.6
    assert value_fraction(quiet_tokens, quiet) < 0.2


def test_value_noise_removes_the_narrow_to_present_values_shortcut():
    """With enough value noise nearly every value type shows up in the filler."""
    clean = _cfg(noise_value_rate=0.0)
    noisy = _cfg(noise_value_rate=0.9)
    clean_tokens = np.array(generate_batch(clean, 8, np.random.default_rng(13)).tokens)
    noisy_tokens = np.array(generate_batch(noisy, 8, np.random.default_rng(13)).tokens)
    low, high = clean.value_ids
    clean_values = set(
        clean_tokens[(clean_tokens >= low) & (clean_tokens < high)].tolist()
    )
    noisy_values = set(
        noisy_tokens[(noisy_tokens >= low) & (noisy_tokens < high)].tolist()
    )
    assert len(noisy_values) > len(clean_values)
    assert len(noisy_values) == clean.n_values


def test_every_batch_element_is_a_valid_token_id():
    cfg = _cfg()
    batch = generate_batch(cfg, 4, np.random.default_rng(14))
    for array in (np.array(batch.tokens), np.array(batch.targets)):
        assert array.min() >= 0
        assert array.max() < cfg.vocab_size
