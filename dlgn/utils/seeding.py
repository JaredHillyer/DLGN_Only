# New file — coordinate seeds across Python, NumPy, and Torch where needed.
# Keep this module minimal; do not invent a randomness framework.
from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int, *, seed_torch: bool = False) -> None:
    """Set seeds for Python random and NumPy, and optionally PyTorch.

    Torch import is opt-in because the JAX-only toy path should remain usable even in
    environments where the optional Torch stack is unavailable or unhealthy.
    """
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        import torch
        torch.manual_seed(seed)
