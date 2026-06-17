# Source: ported from Older_Imp.ipynb cell 2
# GroupSum output head. Do not replace with mean pooling or learned linear.
from __future__ import annotations

import jax
import jax.numpy as jnp


def group_sum_head(x: jax.Array, class_count: int, sum_tau: float) -> jax.Array:
    """Sum neurons in groups, one group per class, scaled by sum_tau.

    Divisibility guard: num_neurons must divide evenly by class_count.
    Temperature sum_tau controls output sharpness — do not remove.
    """
    if x.shape[-1] % class_count != 0:
        raise ValueError(
            'num_neurons must be divisible by class_count for GroupSum: '
            f'{x.shape[-1]=} {class_count=}'
        )
    group_size = x.shape[-1] // class_count
    x = x.reshape(x.shape[0], class_count, group_size)
    return x.sum(axis=-1) / jnp.maximum(sum_tau, 1e-6)
