"""The paper's signal-to-noise-ratio model, and a simulation of it.

Section 3 of arXiv:2511.11571 models block selection as a needle-in-a-haystack
problem.  A query ``q`` scores block ``j`` by its centroid
``k~_j = (1/B) sum_{s in block j} k_s``, and the paper asks whether the score of
the block containing the signal key beats the score of a pure-noise block.  With
``D = s_{j*} - s_j`` it derives

    E[D]     = Delta_mu_eff / B                                   (Eq. 1)
    Var(D)   ~ 2 / (d B)   for normalized vectors                 (Eq. 2)
    SNR      = E[D] / sqrt(Var(D)) = Delta_mu_eff * sqrt(d / 2B)  (Eq. 3)

and ``p_fail = Phi(-SNR)``.  What is implemented here is exactly that algebra plus
a simulation which draws the keys the paper's assumptions describe, so the scaling
in ``d`` and ``B`` can be checked against the closed form.  It is a simulation of
the paper's *model*, not a measurement of the paper's model.
"""

from __future__ import annotations

import math

import numpy as np


def predicted_snr(delta_mu: float, head_dim: int, block_size: int) -> float:
    """``Delta_mu_eff * sqrt(d / (2 B))`` -- Eq. 3."""
    return delta_mu * math.sqrt(head_dim / (2.0 * block_size))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def predicted_failure_rate(snr: float) -> float:
    """``Phi(-SNR)``: the chance a single noise block outranks the signal block."""
    return normal_cdf(-snr)


def sample_router_problem(
    rng: np.random.Generator,
    head_dim: int,
    block_size: int,
    n_blocks: int,
    signal_cosine: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """One draw of the paper's setup.

    Returns ``(query, block_centroids, signal_block)``.  The signal key has
    expected cosine ``signal_cosine`` with the query and every other key is
    independent noise; both are unit vectors, which is the case Eq. 2 is derived
    for.
    """
    query = rng.normal(size=head_dim)
    query /= np.linalg.norm(query)
    keys = rng.normal(size=(n_blocks, block_size, head_dim))
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    signal_block = int(rng.integers(0, n_blocks))
    noise = rng.normal(size=head_dim)
    noise -= noise @ query * query
    noise /= np.linalg.norm(noise)
    signal_key = signal_cosine * query + math.sqrt(
        max(0.0, 1.0 - signal_cosine**2)
    ) * noise
    keys[signal_block, 0] = signal_key / np.linalg.norm(signal_key)
    centroids = keys.mean(axis=1)
    return query, centroids, signal_block


def measure_router_snr(
    head_dim: int,
    block_size: int,
    n_blocks: int,
    signal_cosine: float = 0.35,
    trials: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Empirical SNR, failure rate and recall@1 of the centroid router."""
    rng = np.random.default_rng(seed)
    differences = []
    failures = 0
    for _ in range(trials):
        query, centroids, signal_block = sample_router_problem(
            rng, head_dim, block_size, n_blocks, signal_cosine
        )
        scores = centroids @ query
        noise_blocks = np.delete(np.arange(n_blocks), signal_block)
        # Eq. 1 and Eq. 2 are about one pure-noise block, so D is measured
        # against a single random one; the max over all of them is what the
        # failure rate is about, and mixing the two is what made a first version
        # of this function report a negative SNR for a positive effect.
        differences.append(scores[signal_block] - scores[rng.choice(noise_blocks)])
        if np.max(scores[noise_blocks]) > scores[signal_block]:
            failures += 1
    differences = np.asarray(differences)
    return {
        "head_dim": float(head_dim),
        "block_size": float(block_size),
        "n_blocks": float(n_blocks),
        "signal_cosine": float(signal_cosine),
        "trials": float(trials),
        "empirical_mean_D": float(differences.mean()),
        "empirical_std_D": float(differences.std(ddof=1)),
        "empirical_snr": float(differences.mean() / differences.std(ddof=1)),
        "measured_failure_rate": failures / trials,
        "recall_at_1": 1.0 - failures / trials,
    }


def snr_scaling_table(
    head_dim: int,
    block_sizes: list[int],
    n_blocks: int,
    signal_cosine: float = 0.35,
    trials: int = 4000,
    seed: int = 0,
) -> list[dict[str, float]]:
    """One simulated row per block size, with the predicted SNR alongside.

    ``Delta_mu_eff`` is taken to be the empirical mean of ``D`` times ``B``, which
    is Eq. 1 inverted; feeding that single number back into Eq. 3 means the
    prediction is tested on its *scaling* in ``B`` rather than on the value of a
    free parameter.
    """
    rows = []
    for block_size in block_sizes:
        row = measure_router_snr(
            head_dim, block_size, n_blocks, signal_cosine, trials, seed
        )
        delta_mu_eff = row["empirical_mean_D"] * block_size
        row["delta_mu_eff_from_mean"] = float(delta_mu_eff)
        row["predicted_snr"] = predicted_snr(delta_mu_eff, head_dim, block_size)
        row["predicted_failure_rate"] = predicted_failure_rate(row["predicted_snr"])
        row["snr_scale_sqrt_d_over_b"] = math.sqrt(head_dim / block_size)
        rows.append(row)
    return rows
