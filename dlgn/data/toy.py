# Source: ported from Older_Imp.ipynb cell 4
# Toy datasets used for smoke tests and quick experiments.
from __future__ import annotations

import numpy as np


def make_xor_arrays(repeats: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) NumPy arrays for XOR classification."""
    x = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    return np.tile(x, (repeats, 1)), np.tile(y, repeats)


def make_parity_arrays(
    num_samples: int,
    num_bits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) NumPy arrays for parity classification."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(num_samples, num_bits), dtype=np.int32).astype(np.float32)
    y = (x.sum(axis=-1) % 2).astype(np.int32)
    return x, y
