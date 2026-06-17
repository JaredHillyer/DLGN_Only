# Source: ported from Older_Imp.ipynb cell 5
# JIT-compiled train step and evaluation helpers.
# make_train_step and eval_batch must remain JIT-compiled.
# evaluate_loader key construction preserves notebook's deterministic eval seeding.
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from dlgn.models.network import forward_logits
from dlgn.training.losses import loss_f
from dlgn.training.state import TrainState
from dlgn.types import LogicFamily, WirePair
from dlgn.utils import batch_to_jax


def make_train_step(tx: optax.GradientTransformation):
    """Return a JIT-compiled train step function bound to optimizer tx."""

    @partial(jax.jit, static_argnames=('architecture', 'class_count', 'logic_family'))
    def train_step(
        state: TrainState,
        batch_x: jax.Array,
        batch_y: jax.Array,
        wires: list[WirePair],
        architecture: str,
        gumb_tau: float,
        dirichlet_concentration: float,
        class_count: int,
        sum_tau: float,
        logic_family: LogicFamily,
    ) -> tuple[TrainState, jax.Array, dict[str, jax.Array]]:
        key, subkey = jax.random.split(state.key)

        def apply_loss(params):
            return loss_f(
                params, wires, batch_x, batch_y, subkey,
                architecture, gumb_tau, dirichlet_concentration,
                class_count, sum_tau, logic_family,
            )

        (loss, aux), grads = jax.value_and_grad(apply_loss, has_aux=True)(state.params)
        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        state = state.replace(params=params, opt_state=opt_state, key=key)
        return state, loss, aux

    return train_step


@partial(jax.jit, static_argnames=('architecture', 'class_count', 'logic_family'))
def eval_batch(
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
) -> dict[str, jax.Array]:
    """JIT-compiled per-batch evaluation metrics (sums, not averages)."""
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
    return {
        'soft_loss_sum': optax.softmax_cross_entropy_with_integer_labels(
            logits_soft, batch_y
        ).sum(),
        'hard_loss_sum': optax.softmax_cross_entropy_with_integer_labels(
            logits_hard, batch_y
        ).sum(),
        'soft_correct': (jnp.argmax(logits_soft, axis=-1) == batch_y).sum(),
        'hard_correct': (jnp.argmax(logits_hard, axis=-1) == batch_y).sum(),
        'count': batch_y.shape[0],
    }


def evaluate_loader(
    params: list[jax.Array],
    wires: list[WirePair],
    loader,
    config: dict,
    class_count: int,
) -> dict[str, float] | None:
    """Evaluate over a full loader. Returns None if loader is None.

    Eval keys are deterministic: seed + 50_000 + batch_index (notebook convention).
    """
    if loader is None:
        return None

    totals = {
        'soft_loss_sum': 0.0,
        'hard_loss_sum': 0.0,
        'soft_correct': 0.0,
        'hard_correct': 0.0,
        'count': 0.0,
    }

    for batch_index, batch in enumerate(loader):
        batch_x, batch_y = batch_to_jax(batch)
        metrics = eval_batch(
            params,
            wires,
            batch_x,
            batch_y,
            jax.random.PRNGKey(config['seed'] + 50_000 + batch_index),
            config['architecture'],
            config['gumb_tau'],
            config['dirichlet_concentration'],
            class_count,
            config['sum_tau'],
            config['logic_family'],
        )
        for k, v in metrics.items():
            totals[k] += float(v)

    count = max(totals['count'], 1.0)
    soft_acc = totals['soft_correct'] / count
    hard_acc = totals['hard_correct'] / count
    return {
        'soft_loss': totals['soft_loss_sum'] / count,
        'hard_loss': totals['hard_loss_sum'] / count,
        'soft_acc': soft_acc,
        'hard_acc': hard_acc,
        # Soft-to-hard discretization gap; positive means the soft forward
        # is more accurate than the hard one (B-perf-9 in
        # dlgn_decision_log.md). Equal to soft_acc - hard_acc by definition.
        'soft_hard_gap': soft_acc - hard_acc,
    }
