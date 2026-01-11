import os

import numpy as np
import pytest

from moba.plots import (
    plot_accuracy_vs_k,
    plot_selection_matrix,
    plot_snr_recall,
    plot_snr_scaling,
    plot_tile_schedule,
    plot_traffic_vs_k,
)


def _nonempty(path):
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000


def test_accuracy_figure_is_written(tmp_path):
    path = str(tmp_path / "accuracy.png")
    out = plot_accuracy_vs_k(
        ks=[1, 2, 4, 8],
        losses=[2.1, 1.4, 1.1, 1.0],
        loss_errs=[0.1, 0.1, 0.05, 0.02],
        accuracies=[0.2, 0.4, 0.5, 0.55],
        accuracy_errs=[0.01, 0.02, 0.03, 0.01],
        dense_loss=1.0,
        dense_accuracy=0.55,
        n_blocks=8,
        path=path,
    )
    assert out == path
    _nonempty(path)


def test_selection_matrix_figure_handles_one_panel(tmp_path):
    path = str(tmp_path / "selection.png")
    matrix = np.tril(np.ones((16, 4), np.float32))
    out = plot_selection_matrix([matrix], ["k=1"], block_size=4, path=path)
    assert out == path
    _nonempty(path)


def test_tile_schedule_figure_annotates_live_tiles(tmp_path):
    path = str(tmp_path / "schedule.png")
    schedule = np.eye(4, dtype=np.float32)
    out = plot_tile_schedule([schedule, np.tril(np.ones((4, 4), np.float32))],
                             ["sparse", "dense"], path=path)
    assert out == path
    _nonempty(path)


def test_traffic_figure_is_written(tmp_path):
    path = str(tmp_path / "traffic.png")
    out = plot_traffic_vs_k(
        ks=[1, 2, 4, 8],
        loaded_fractions=[0.125, 0.25, 0.5, 0.5625],
        kv_bytes=[0.22, 0.44, 0.83, 1.0],
        dense_bytes=1.0,
        n_blocks=8,
        wall_clock_ms=[1.0, 1.2, 1.4, 1.3],
        dense_ms=1.1,
        path=path,
    )
    assert out == path
    _nonempty(path)


def test_snr_figures_are_written(tmp_path):
    blocks = [4, 8, 16, 32, 64]
    empirical = [1.25, 0.83, 0.60, 0.43, 0.29]
    predicted = [1.14, 0.80, 0.59, 0.41, 0.29]
    recall = [0.45, 0.29, 0.21, 0.15, 0.12]
    first = plot_snr_scaling(blocks, empirical, predicted, recall, 16,
                             str(tmp_path / "snr.png"))
    second = plot_snr_recall(blocks, recall, str(tmp_path / "recall.png"))
    _nonempty(first)
    _nonempty(second)
