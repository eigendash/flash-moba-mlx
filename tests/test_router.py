import mlx.core as mx
import numpy as np
import pytest

from moba.blocks import block_valid_mask, causal_block_scores
from moba.router import (
    Router,
    block_selection_mask,
    token_selection_mask,
    topk_blocks,
)


def _numpy_topk(scores, k, valid):
    """Reference: rank the legal blocks by score, ties broken by lower index."""
    order = []
    for t in range(scores.shape[0]):
        cand = [j for j in range(scores.shape[1]) if valid[t, j]]
        cand.sort(key=lambda j: (-float(scores[t, j]), j))
        order.append(cand[:k])
    return order


def test_topk_selects_the_highest_scoring_blocks():
    rng = np.random.default_rng(0)
    scores = mx.array(rng.normal(size=(5, 8)).astype(np.float32))
    valid = mx.array(np.ones((5, 8), bool))
    idx, sel = topk_blocks(scores, 3, valid)
    got = np.array(idx)
    want = _numpy_topk(np.array(scores), 3, np.ones((5, 8), bool))
    assert sel.all()
    for t in range(5):
        assert sorted(got[t].tolist()) == sorted(want[t])


def test_topk_returns_scores_in_descending_order():
    rng = np.random.default_rng(1)
    scores = mx.array(rng.normal(size=(4, 16)).astype(np.float32))
    valid = mx.array(np.ones((4, 16), bool))
    idx, _ = topk_blocks(scores, 5, valid)
    picked = np.take_along_axis(np.array(scores), np.array(idx), axis=-1)
    for row in picked:
        assert np.all(np.diff(row) <= 1e-6)


def test_topk_marks_slots_unselected_when_there_are_too_few_candidates():
    scores = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.float32)
    valid = mx.array([[[True, False, False, False]]])
    idx, sel = topk_blocks(scores, 3, valid)
    assert sel.tolist() == [[[True, False, False]]]
    assert int(idx[0, 0, 0]) == 0


def test_topk_pads_when_k_exceeds_the_block_count():
    scores = mx.arange(6, dtype=mx.float32).reshape(1, 2, 3)
    idx, sel = topk_blocks(scores, 5)
    assert idx.shape == (1, 2, 5)
    assert sel.tolist() == [[[True] * 3 + [False] * 2] * 2]
    assert np.array(idx)[..., :3].max() < 3


def test_topk_is_deterministic_even_under_ties():
    tied = mx.array([[1.0, 1.0, 1.0, 0.5]])
    first, sel_first = topk_blocks(tied, 2)
    for _ in range(5):
        again, sel_again = topk_blocks(tied, 2)
        assert np.array_equal(np.array(first), np.array(again))
        assert np.array_equal(np.array(sel_first), np.array(sel_again))


def test_topk_rejects_k_below_one():
    with pytest.raises(ValueError):
        topk_blocks(mx.zeros((2, 4)), 0)
    with pytest.raises(ValueError):
        Router(k=0, block_size=4)


def test_router_never_selects_a_future_only_block():
    rng = np.random.default_rng(2)
    length, block_size, k = 24, 4, 2
    q = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    router = Router(k=k, block_size=block_size)
    idx, sel, _ = router(q, keys)
    valid = np.array(block_valid_mask(length, block_size))
    for t in range(length):
        for slot in range(k):
            if bool(sel[0, t, slot]):
                j = int(idx[0, t, slot])
                assert valid[t, j], (t, slot, j)
                assert j <= t // block_size


def test_router_selection_is_reproducible():
    rng = np.random.default_rng(3)
    q = mx.array(rng.normal(size=(2, 32, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(2, 32, 8)).astype(np.float32))
    router = Router(k=3, block_size=8)
    a_idx, a_sel, _ = router(q, keys)
    b_idx, b_sel, _ = router(q, keys)
    assert np.array_equal(np.array(a_idx), np.array(b_idx))
    assert np.array_equal(np.array(a_sel), np.array(b_sel))


def test_router_scores_match_the_block_statistics_module():
    rng = np.random.default_rng(4)
    q = mx.array(rng.normal(size=(1, 16, 5)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, 16, 5)).astype(np.float32))
    router = Router(k=2, block_size=4)
    got = np.array(router.scores(q, keys))
    want = np.array(causal_block_scores(q, keys, 4))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_router_k_otherwise_only_changes_how_many_blocks_are_kept():
    rng = np.random.default_rng(5)
    q = mx.array(rng.normal(size=(1, 32, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, 32, 8)).astype(np.float32))
    one, sel_one, scores = Router(k=1, block_size=8)(q, keys)
    four, _, _ = Router(k=4, block_size=8)(q, keys)
    # The k=1 pick is the head of the k=4 list.
    assert np.array_equal(np.array(one)[..., 0], np.array(four)[..., 0])
    # With k = K every candidate block is selected, so `sel` is exactly the
    # causal candidate mask.
    count = 32 // 8
    full_idx, full_sel, _ = Router(k=count, block_size=8)(q, keys)
    valid = np.array(block_valid_mask(32, 8))
    got = np.zeros_like(valid)
    for t in range(32):
        for slot in range(count):
            if bool(full_sel[0, t, slot]):
                got[t, int(full_idx[0, t, slot])] = True
    assert np.array_equal(got, valid)


def test_block_selection_mask_round_trips_the_indices():
    idx = mx.array([[[2, 0], [1, 3]]], dtype=mx.int32)
    sel = mx.array([[[True, True], [True, False]]])
    mask = np.array(block_selection_mask(idx, sel, 4))
    assert mask.shape == (1, 2, 4)
    assert mask[0, 0].tolist() == [True, False, True, False]
    # The unselected slot pointed at block 3 must not light it up.
    assert mask[0, 1].tolist() == [False, True, False, False]


def test_block_selection_mask_is_not_clobbered_by_padding():
    # Slot 0 picks block 0; slots 1 and 2 are padding that must be ignored.
    idx = mx.array([[[0, 0, 0]]], dtype=mx.int32)
    sel = mx.array([[[True, False, False]]])
    mask = np.array(block_selection_mask(idx, sel, 3))
    assert mask[0, 0].tolist() == [True, False, False]


def test_token_selection_mask_is_causal_and_block_aligned():
    # Tokens 0-3 select blocks 0 and 1; tokens 4-7 select block 1 only.
    idx = np.zeros((1, 8, 2), np.int32)
    sel = np.zeros((1, 8, 2), bool)
    idx[0, :4, 0], idx[0, :4, 1] = 1, 0
    sel[0, :4] = True
    idx[0, 4:, 0], sel[0, 4:, 0] = 1, True
    mask = np.array(
        token_selection_mask(mx.array(idx), mx.array(sel), 8, 4)
    )
    assert mask.shape == (1, 8, 8)
    for t in range(4):
        assert mask[0, t].tolist() == [True] * (t + 1) + [False] * (7 - t)
    # Tokens 4-7 select block 1 only, so they never see tokens 0-3.
    for t in range(4, 8):
        want = [False] * 4 + [True] * (t - 3) + [False] * (7 - t)
        assert mask[0, t].tolist() == want


def test_token_selection_mask_drops_the_causal_future():
    idx = mx.array([[[2]]], dtype=mx.int32)
    sel = mx.array([[[True]]])
    mask = np.array(token_selection_mask(idx, sel, 12, 4))
    # Selecting block 2 cannot expose a future token of block 2.
    for t in range(8, 12):
        assert mask[0, t, t + 1 :].sum() == 0
        assert mask[0, t, 8 : t + 1].all()
