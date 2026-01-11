"""Matplotlib figures for the MoBA experiment.

Three figures, each of which the README points at:

* ``accuracy_vs_k.png`` -- held-out loss and answer accuracy against ``k``, with
  the dense baseline as a horizontal reference;
* ``selection_matrix.png`` -- the (query position, key block) mask the router
  produced, one panel per head, so the block-diagonal causal structure and the
  spread of the selections are both visible;
* ``tile_schedule.png`` -- the logical query-block x key-block schedule: how much
  of each tile the kernel would actually execute, next to the loaded-block
  fraction as a function of ``k`` with the ``k / K`` line drawn on top.

The last one is the tile-schedule visualisation: it is the grid the paper's kernel
iterates (Algorithm 1, the outer loop over logical query and key blocks), coloured
by occupancy, and it is what makes the difference between a dense causal schedule
and a block-sparse one concrete.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_accuracy_vs_k(
    ks: list[int],
    losses: list[float],
    loss_errs: list[float],
    accuracies: list[float],
    accuracy_errs: list[float],
    dense_loss: float,
    dense_accuracy: float,
    n_blocks: int,
    path: str,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].errorbar(ks, losses, yerr=loss_errs, marker="o", capsize=3)
    axes[0].axhline(dense_loss, ls="--", c="grey", label="dense baseline")
    axes[0].set_xlabel("k (blocks attended per query)")
    axes[0].set_ylabel("held-out answer loss")
    axes[0].set_title("Loss vs k")
    axes[0].legend()

    axes[1].errorbar(ks, accuracies, yerr=accuracy_errs, marker="o", capsize=3)
    axes[1].axhline(dense_accuracy, ls="--", c="grey", label="dense baseline")
    axes[1].set_xlabel("k (blocks attended per query)")
    axes[1].set_ylabel("held-out answer accuracy")
    axes[1].set_title(f"Accuracy vs k (K = {n_blocks} blocks)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    return _save(fig, path)


def plot_selection_matrix(
    matrices: list[np.ndarray],
    titles: list[str],
    block_size: int,
    path: str,
) -> str:
    """One panel per head: which queries selected which key blocks."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.3), squeeze=False)
    rows = matrices[0].shape[0]
    n_blocks = matrices[0].shape[1]
    for ax, matrix, title in zip(axes[0], matrices, titles):
        ax.imshow(
            matrix.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"query position (L={rows})")
        ax.set_ylabel(f"key block (B={block_size})")
        # Block boundaries on the query axis.
        for boundary in range(block_size, rows, block_size):
            ax.axvline(boundary - 0.5, color="white", lw=0.4, alpha=0.5)
        ax.set_yticks(range(n_blocks))
    return _save(fig, path)


def plot_tile_schedule(schedules: list[np.ndarray], titles: list[str], path: str) -> str:
    """The query-block x key-block schedule the kernel would iterate."""
    n = len(schedules)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    for ax, schedule, title in zip(axes[0], schedules, titles):
        count = schedule.shape[0]
        image = ax.imshow(
            schedule,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("key block j")
        ax.set_ylabel("query block i")
        ax.set_xticks(range(count))
        ax.set_yticks(range(count))
        for i in range(count):
            for j in range(count):
                if schedule[i, j] > 0:
                    ax.text(
                        j, i, f"{schedule[i, j]:.2f}".lstrip("0"),
                        ha="center", va="center", fontsize=5.5,
                        color="white" if schedule[i, j] < 0.6 else "black",
                    )
        fig.colorbar(image, ax=ax, fraction=0.046, label="share of queries using the tile")
    return _save(fig, path)


def plot_traffic_vs_k(
    ks: list[int],
    loaded_fractions: list[float],
    kv_bytes: list[float],
    dense_bytes: float,
    n_blocks: int,
    wall_clock_ms: list[float],
    dense_ms: float,
    path: str,
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ideal = [k / n_blocks for k in ks]

    axes[0].plot(ks, loaded_fractions, marker="o", label="measured")
    axes[0].plot(ks, ideal, ls=":", marker="x", label="k / K")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("loaded blocks / all blocks")
    axes[0].set_title("Block loads track k / K")
    axes[0].legend()

    axes[1].plot(ks, [b / dense_bytes for b in kv_bytes], marker="o", label="sparse")
    axes[1].axhline(1.0, ls="--", c="grey", label="dense causal")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("KV bytes / dense KV bytes")
    axes[1].set_title("KV traffic fraction of dense")
    axes[1].legend()

    axes[2].plot(ks, wall_clock_ms, marker="o", label="tiled forward")
    axes[2].axhline(dense_ms, ls="--", c="grey", label="dense attention")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("ms per forward")
    axes[2].set_title("Wall clock (this machine)")
    axes[2].legend()
    return _save(fig, path)


def plot_snr_scaling(
    block_sizes: list[int],
    empirical_snr: list[float],
    predicted_snr: list[float],
    recall: list[float],
    head_dim: int,
    path: str,
) -> str:
    """The paper's Eq. 3: SNR should fall like sqrt(d / B) as blocks grow."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].plot(block_sizes, empirical_snr, marker="o", label="measured SNR of D")
    axes[0].plot(block_sizes, predicted_snr, ls="--", marker="x",
                 label=r"predicted $\Delta\mu_{eff}\sqrt{d/2B}$")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(block_sizes)
    axes[0].set_xticklabels([str(b) for b in block_sizes])
    axes[0].set_xlabel("block size B")
    axes[0].set_ylabel("router SNR")
    axes[0].set_title(f"SNR vs block size (d = {head_dim})")
    axes[0].legend()

    scale = [(head_dim / b) ** 0.5 for b in block_sizes]
    axes[1].plot(scale, empirical_snr, marker="o", label="measured")
    axes[1].set_xlabel(r"$\sqrt{d/B}$")
    axes[1].set_ylabel("router SNR")
    axes[1].set_title("SNR is linear in sqrt(d/B)")
    axes[1].legend()
    return _save(fig, path)


def plot_snr_recall(block_sizes: list[int], recall: list[float], path: str) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(block_sizes, recall, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes)
    ax.set_xticklabels([str(b) for b in block_sizes])
    ax.set_xlabel("block size B")
    ax.set_ylabel("recall@1 of the signal block")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Smaller blocks retrieve better")
    return _save(fig, path)
