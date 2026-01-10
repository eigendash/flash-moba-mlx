import math

import numpy as np
import pytest

from moba.snr import (
    measure_router_snr,
    normal_cdf,
    predicted_failure_rate,
    predicted_snr,
    sample_router_problem,
    snr_scaling_table,
)


def test_predicted_snr_follows_the_paper_formula():
    # Eq. 3: SNR = Delta_mu_eff sqrt(d / 2B).
    assert predicted_snr(1.0, 16, 8) == pytest.approx(math.sqrt(16 / 16))
    assert predicted_snr(1.0, 16, 16) == pytest.approx(math.sqrt(16 / 32))
    # Halving the block size multiplies the SNR by sqrt(2).
    assert predicted_snr(1.0, 16, 8) / predicted_snr(1.0, 16, 16) == pytest.approx(
        math.sqrt(2)
    )


def test_normal_cdf_and_failure_rate_bounds():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(-10.0) < 1e-6
    assert predicted_failure_rate(0.0) == pytest.approx(0.5)
    assert predicted_failure_rate(3.0) < 0.01


def test_sampled_problem_has_unit_vectors():
    rng = np.random.default_rng(0)
    query, centroids, signal_block = sample_router_problem(
        rng, head_dim=16, block_size=8, n_blocks=6, signal_cosine=0.5
    )
    assert query.shape == (16,)
    assert centroids.shape == (6, 16)
    assert np.linalg.norm(query) == pytest.approx(1.0)
    assert 0 <= signal_block < 6


def test_measured_snr_is_positive_when_the_signal_is_separable():
    row = measure_router_snr(
        head_dim=16, block_size=4, n_blocks=16, signal_cosine=0.9, trials=1500, seed=1
    )
    assert row["empirical_snr"] > 0.5
    assert 0.0 <= row["recall_at_1"] <= 1.0
    assert row["recall_at_1"] > 1.0 / 16


def test_recall_and_snr_are_worse_with_larger_blocks():
    """The paper's first design principle: smaller B routes better."""
    rows = snr_scaling_table(
        head_dim=16,
        block_sizes=[4, 16, 64],
        n_blocks=16,
        signal_cosine=0.8,
        trials=2000,
        seed=2,
    )
    snrs = [row["empirical_snr"] for row in rows]
    recalls = [row["recall_at_1"] for row in rows]
    assert snrs[0] > snrs[1] > snrs[2]
    assert recalls[0] > recalls[1] > recalls[2]


def test_empirical_snr_tracks_the_closed_form_within_a_tolerance():
    rows = snr_scaling_table(
        head_dim=16,
        block_sizes=[4, 8, 16, 32, 64],
        n_blocks=16,
        signal_cosine=0.8,
        trials=3000,
        seed=3,
    )
    for row in rows:
        assert row["empirical_snr"] == pytest.approx(row["predicted_snr"], abs=0.15)
    # Delta_mu_eff should be a property of the keys, not of the block size.
    delta_mus = [row["delta_mu_eff_from_mean"] for row in rows]
    assert np.std(delta_mus) < 0.05


def test_empirical_snr_is_linear_in_sqrt_d_over_b():
    rows = snr_scaling_table(
        head_dim=16,
        block_sizes=[8, 32],
        n_blocks=16,
        signal_cosine=0.8,
        trials=3000,
        seed=4,
    )
    # sqrt(d/B) doubles from B=32 to B=8, so the SNR should roughly double.
    ratio = rows[0]["empirical_snr"] / rows[1]["empirical_snr"]
    assert ratio == pytest.approx(2.0, rel=0.25)


def test_more_signal_clustering_raises_the_snr():
    weak = measure_router_snr(16, 8, 16, signal_cosine=0.3, trials=1500, seed=5)
    strong = measure_router_snr(16, 8, 16, signal_cosine=0.9, trials=1500, seed=5)
    assert strong["empirical_snr"] > weak["empirical_snr"]
    assert strong["recall_at_1"] > weak["recall_at_1"]
