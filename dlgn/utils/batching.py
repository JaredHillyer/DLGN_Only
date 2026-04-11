# Source: ported from Older_Imp.ipynb cell 4
# Batch conversion utilities (torch tensors or numpy arrays → JAX arrays).
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


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
