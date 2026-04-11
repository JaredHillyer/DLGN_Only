# Source: ported from Older_Imp.ipynb cell 6 (_tree_to_numpy, _tree_to_jax)
# JAX pytree serialization helpers for checkpoint save/load.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


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
