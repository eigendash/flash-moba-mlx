import mlx.core as mx
import numpy as np
import pytest

from moba.task import (
    TaskConfig,
    generate_batch,
    held_out_batches,
    signal_block,
)


def _cfg(**kw):
    base = dict(seq_len=64, block_size=8, n_pairs=2, n_filler=8, n_keys=16, n_values=16)
    base.update(kw)
    return TaskConfig(**base)


def test_config_rejects_impossible_layouts():
    with pytest.raises(ValueError):
        _cfg(seq_len=65)
    with pytest.raises(ValueError):
        _cfg(seq_len=24, block_size=8, n_pairs=2)  # only 3 blocks
    with pytest.raises(ValueError):
        _cfg(n_keys=2, n_pairs=2)
    with pytest.raises(ValueError):
        _cfg(n_values=1, n_pairs=2)


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
    assert batch.answer_position == cfg.seq_len - 1


def test_only_the_final_position_is_scored():
    cfg = _cfg()
    batch = generate_batch(cfg, 4, np.random.default_rng(1))
    mask = np.array(batch.mask)
    assert mask[:, -1].tolist() == [1.0] * 4
    assert mask[:, :-1].sum() == 0.0


def test_targets_are_the_next_token_except_at_the_answer_position():
    cfg = _cfg()
    batch = generate_batch(cfg, 3, np.random.default_rng(2))
    tokens = np.array(batch.tokens)
    targets = np.array(batch.targets)
    np.testing.assert_array_equal(targets[:, :-1], tokens[:, 1:])
    np.testing.assert_array_equal(targets[:, -1], batch.answer)


def test_every_bound_value_appears_exactly_once_in_the_sequence():
    """Every pair value is unique, so the answer cannot be copied from nearby."""
    cfg = _cfg()
    batch = generate_batch(cfg, 8, np.random.default_rng(3))
    tokens = np.array(batch.tokens)
    for b in range(len(batch)):
        for p in range(cfg.n_pairs):
            start = int(batch.pair_blocks[b, p]) * cfg.block_size
            value = int(tokens[b, start + 1])
            assert int((tokens[b] == value).sum()) == 1
        low, high = cfg.value_ids
        present = tokens[b][(tokens[b] >= low) & (tokens[b] < high)]
        assert len(set(present.tolist())) == len(present.tolist())


def test_the_final_token_is_the_query_key_of_an_existing_binding():
    cfg = _cfg()
    batch = generate_batch(cfg, 6, np.random.default_rng(4))
    tokens = np.array(batch.tokens)
    for b in range(len(batch)):
        assert tokens[b, -1] == batch.query_key[b]
        block, pair = int(signal_block(batch, b)[0]), int(batch.target_pair[b])
        assert block == int(batch.pair_blocks[b, pair])
        start = block * cfg.block_size
        assert tokens[b, start] == batch.query_key[b]
        assert tokens[b, start + 1] == batch.answer[b]


def test_pair_blocks_are_distinct_and_exclude_the_query_block():
    cfg = _cfg()
    batch = generate_batch(cfg, 16, np.random.default_rng(5))
    for b in range(len(batch)):
        blocks = batch.pair_blocks[b]
        assert len(set(blocks.tolist())) == cfg.n_pairs
        assert blocks.min() >= 1
        assert blocks.max() <= cfg.n_blocks - 2


def test_signal_blocks_are_inside_the_sequence_and_causal():
    cfg = _cfg()
    batch = generate_batch(cfg, 8, np.random.default_rng(6))
    blocks = signal_block(batch)
    assert blocks.shape == (8,)
    assert (blocks >= 1).all()
    assert (blocks < cfg.n_blocks - 1).all()
    # The binding must be fully in the past relative to the query token.
    assert (((blocks + 1) * cfg.block_size - 1) < cfg.seq_len - 1).all()


def test_generation_is_deterministic_for_a_seed():
    cfg = _cfg()
    a = generate_batch(cfg, 4, np.random.default_rng(7))
    b = generate_batch(cfg, 4, np.random.default_rng(7))
    np.testing.assert_array_equal(np.array(a.tokens), np.array(b.tokens))
    np.testing.assert_array_equal(np.array(a.targets), np.array(b.targets))
    np.testing.assert_array_equal(a.pair_blocks, b.pair_blocks)


def test_two_seeds_give_different_sequences():
    cfg = _cfg()
    a = generate_batch(cfg, 4, np.random.default_rng(8))
    b = generate_batch(cfg, 4, np.random.default_rng(9))
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


def test_noise_key_rate_controls_how_many_keys_are_scattered():
    cfg_quiet = _cfg(noise_key_rate=0.0)
    cfg_loud = _cfg(noise_key_rate=0.9)
    quiet = np.array(generate_batch(cfg_quiet, 8, np.random.default_rng(10)).tokens)
    loud = np.array(generate_batch(cfg_loud, 8, np.random.default_rng(10)).tokens)
    low, high = cfg_quiet.key_ids

    def key_fraction(tokens):
        in_range = (tokens >= low) & (tokens < high)
        return in_range.mean()

    assert key_fraction(quiet) < key_fraction(loud)
    assert key_fraction(loud) > 0.5


def test_every_batch_element_is_a_valid_token_id():
    cfg = _cfg()
    batch = generate_batch(cfg, 4, np.random.default_rng(12))
    tokens = np.array(batch.tokens)
    assert tokens.min() >= 0
    assert tokens.max() < cfg.vocab_size
    targets = np.array(batch.targets)
    assert targets.min() >= 0
    assert targets.max() < cfg.vocab_size
