# Source: ported from Older_Imp.ipynb cell 6 (first DLGN plot helpers only)
# Minimal preserved plots. matplotlib is imported lazily; plotting is optional.
# Sweep/report generators from the notebook are NOT included here.
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_hist_gates(
    net,
    save_path=None,
    title: str = 'Distribution of Gates (Sorted)',
) -> None:
    """Plot a horizontal bar chart of gate usage counts.

    net: flat array of hard gate IDs (int).
    save_path: if provided, save the figure instead of displaying it.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise RuntimeError('matplotlib is required for plotting gate histograms') from exc

    gate_types = [
        'FALSE', 'AND', 'A AND (NOT B)', 'A', '(NOT A) AND B', 'B',
        'XOR', 'OR', 'NOR', 'XNOR', 'NOT B', 'A OR (NOT B)',
        'NOT A', '(NOT A) OR B', 'NAND', 'TRUE',
    ]

    net = np.asarray(net).reshape(-1)
    gate_counts = np.bincount(net, minlength=len(gate_types))
    sorted_indices = np.argsort(gate_counts)
    sorted_counts = gate_counts[sorted_indices]
    sorted_gate_types = [gate_types[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.barh(sorted_gate_types, sorted_counts, alpha=0.7, edgecolor='white', linewidth=1)
    ax.set_xlabel('# Gates', fontsize=12, fontweight='bold')
    ax.set_ylabel('Type of gate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_facecolor('#f8f9fa')
    fig.set_facecolor('white')
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path)
        print(f'Saved histogram to {save_path.resolve()}')
        plt.close(fig)
    else:
        plt.show()


def plot_training_progress(
    loss_train,
    loss_test,
    compute_every: int,
    save_path=None,
    title: str = 'Training Progress - Soft vs Hard Gates',
) -> None:
    """Plot soft and hard loss curves over training steps.

    loss_train: per-step soft losses.
    loss_test: per-eval-cadence hard losses.
    compute_every: eval cadence (steps between evaluations).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('matplotlib is required for plotting training curves') from exc

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.plot(loss_train, linewidth=2, label='Soft Gates Loss', alpha=0.9)

    test_x = np.arange(0, compute_every * len(loss_test), compute_every)
    ax.plot(test_x, loss_test, linestyle='--', linewidth=2, label='Hard Gates Loss', alpha=0.9)

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path)
        print(f'Saved training curve to {save_path.resolve()}')
        plt.close(fig)
    else:
        plt.show()
