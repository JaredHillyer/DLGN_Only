from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

### Batch conversion utilities (torch tensors or numpy arrays → JAX arrays) and Tree I/O ###

def batch_to_jax(batch) -> tuple[jax.Array, jax.Array]:
    """Convert a (x, y) batch from any loader into flat JAX float32/int32 arrays."""
    x, y = batch

    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if hasattr(y, 'detach'):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)

    x = x.reshape(x.shape[0], -1).astype(np.float32)
    y = y.reshape(-1).astype(np.int32)
    return jnp.asarray(x), jnp.asarray(y)

def tree_to_numpy(tree):
    """Recursively convert a JAX pytree to NumPy arrays (for pickling)."""
    return jax.tree_util.tree_map(
        lambda x: None if x is None else np.asarray(jax.device_get(x)),
        tree,
    )

def tree_to_jax(tree):
    """Recursively convert a NumPy pytree back to JAX arrays (after loading)."""
    return jax.tree_util.tree_map(
        lambda x: None if x is None else jnp.asarray(x),
        tree,
    )

### Paths ###
def make_run_name(config: dict, timestamp: str | None = None) -> str:
    """Generate a readable run name from config fields.

    Pattern: YYYYMMDD-HHMMSS_<dataset>_<logic_family>_<architecture>_s<seed>
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    dataset = config.get('dataset', 'unknown')
    lf = config.get('logic_family', 'unk')
    arch = config.get('architecture', 'unk')
    seed = config.get('seed', 0)
    return f'{timestamp}_{dataset}_{lf}_{arch}_s{seed}'

def make_run_dir(base_output_dir: str | Path, run_name: str) -> Path:
    """Create and return a run directory with standard subdirectories."""
    run_dir = Path(base_output_dir) / 'runs' / run_name
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    return run_dir

## Seeds to use across Python, NumPy, and Torch ###
# Keep this module minimal; do not invent a randomness framework.
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


### JAX persistent compilation cache (D24) ###
# Cache compiled XLA programs to disk so re-running the same config skips
# the JIT compile wait. Numerics-preserving per JAX contract.

_DEFAULT_JIT_CACHE_DIR = '.jax_cache'


def setup_jit_cache(cache_dir: str | Path = _DEFAULT_JIT_CACHE_DIR,
                    enabled: bool = True) -> str | None:
    """Enable JAX persistent compilation cache (D24).

    Args:
        cache_dir: directory to store compiled XLA programs. Created if
            missing. Relative paths resolve against the current working
            directory; pass an absolute path if you want a single shared
            cache across runs from different cwd.
        enabled: when False, this is a no-op and returns None — used to
            implement --no-jit-cache for benchmarking the cold-compile
            path.

    Returns:
        The resolved cache directory path as a string when enabled, else
        None.

    Notes:
        - Safe to call multiple times in the same process.
        - The cache key includes JAX/XLA version + traced signature, so
          a JAX upgrade silently invalidates older entries (intended).
        - Concurrent runs writing to the same dir are safe per JAX docs.
    """
    if not enabled:
        return None
    resolved = Path(cache_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', str(resolved))
    # Cache anything that takes >= 1s to compile. Keeps the cache from
    # being polluted by trivial micro-jits.
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)
    return str(resolved)

