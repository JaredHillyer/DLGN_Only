# Source: ported from Older_Imp.ipynb cell 5 (DLGN loss_f only)
# DLGN loss function. Training loss is soft; aux hard loss is for monitoring.
# Do not confuse with DLCA loss_f in Latest_Patch.ipynb — those are different functions.
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from dlgn.models.network import forward_logits
from dlgn.types import LogicFamily, WirePair


def loss_f(
    params: list[jax.Array],
    wires: list[WirePair],
    batch_x: jax.Array,
    batch_y: jax.Array,
    key: jax.Array,
    architecture: str,
    gumb_tau: float,
    dirichlet_concentration: float,
    class_count: int,
    sum_tau: float,
    logic_family: LogicFamily,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute soft training loss and hard evaluation loss.

    Returns (soft_loss, {'hard': hard_loss}).
    Training optimizes soft_loss; hard_loss is aux for monitoring only.
    """
    logits_soft = forward_logits(
        params, wires, batch_x, True, key,
        architecture, gumb_tau, dirichlet_concentration,
        class_count, sum_tau, logic_family,
    )
    logits_hard = forward_logits(
        params, wires, batch_x, False, key,
        architecture, gumb_tau, dirichlet_concentration,
        class_count, sum_tau, logic_family,
    )
    soft_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_soft, batch_y
    ).mean()
    hard_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_hard, batch_y
    ).mean()
    return soft_loss, {'hard': hard_loss}
