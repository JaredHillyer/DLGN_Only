# Source: ported from Older_Imp.ipynb cell 5
# High-level training orchestration. Preserves history keys and return contract.
# History keys: steps, train_soft_loss, train_hard_loss, valid_hard_acc, test_hard_acc
from __future__ import annotations

import jax
import jax.numpy as jnp

from dlgn.config import validate_logic_config
from dlgn.data.loaders import load_dataset, cycle_loader
from dlgn.data.registry import input_dim_of_dataset, num_classes_of_dataset
from dlgn.models.initialization import init_logic_gate_network
from dlgn.training.optim import create_optimizer
from dlgn.training.state import TrainState
from dlgn.training.steps import make_train_step, evaluate_loader
from dlgn.types import WirePair
from dlgn.utils.batching import batch_to_jax


def _empty_history() -> dict[str, list]:
    """Return a fresh history dict with the canonical notebook keys."""
    return {
        'steps': [],
        'train_soft_loss': [],
        'train_hard_loss': [],
        'valid_hard_acc': [],
        'test_hard_acc': [],
    }


def train_model(
    config: dict,
    initial_state: TrainState | None = None,
    initial_wires: list[WirePair] | None = None,
    history: dict | None = None,
    start_step: int | None = None,
):
    """Train a DLGN model from config (or resume from an existing state).

    Returns (state, wires, history, dataset_info, validation_loader, test_loader).
    dataset_info contains at least: dataset, input_dim, class_count, logic_family.
    """
    validate_logic_config(config)
    train_loader, validation_loader, test_loader = load_dataset(config)
    input_dim = input_dim_of_dataset(config['dataset'])
    class_count = num_classes_of_dataset(config['dataset'])

    if config['num_neurons'] % class_count != 0:
        raise ValueError(
            'num_neurons must be divisible by the number of classes for GroupSum: '
            f"{config['num_neurons']=} {class_count=}"
        )

    tx = create_optimizer(config)
    train_step = make_train_step(tx)

    if initial_state is None:
        key = jax.random.PRNGKey(config['seed'])
        key, init_key = jax.random.split(key)
        params, wires = init_logic_gate_network(
            input_dim=input_dim,
            num_neurons=config['num_neurons'],
            num_layers=config['num_layers'],
            connections=config['connections'],
            key=init_key,
            logic_family=config['logic_family'],
        )
        state = TrainState(params=params, opt_state=tx.init(params), key=key)
    else:
        if initial_wires is None:
            raise ValueError(
                'initial_wires must be provided when resuming from an existing state'
            )
        state = initial_state
        wires = initial_wires

    if history is None:
        history = _empty_history()
    else:
        history = {k: list(v) for k, v in history.items()}
        for k in _empty_history():
            history.setdefault(k, [])

    if start_step is None:
        start_step = 0 if not history['steps'] else int(history['steps'][-1]) + 1

    loader_iter = cycle_loader(train_loader)
    for step in range(start_step, config['num_steps']):
        batch = next(loader_iter)
        batch_x, batch_y = batch_to_jax(batch)
        state, soft_loss, aux = train_step(
            state,
            batch_x,
            batch_y,
            wires,
            config['architecture'],
            config['gumb_tau'],
            config['dirichlet_concentration'],
            class_count,
            config['sum_tau'],
            config['logic_family'],
        )

        history['steps'].append(step)
        history['train_soft_loss'].append(float(soft_loss))
        history['train_hard_loss'].append(float(aux['hard']))

        should_eval = (
            step % config['eval_every'] == 0
            or step == config['num_steps'] - 1
        )
        if should_eval:
            valid_metrics = evaluate_loader(
                state.params, wires, validation_loader, config, class_count
            )
            test_metrics = evaluate_loader(
                state.params, wires, test_loader, config, class_count
            )

            history['valid_hard_acc'].append(
                None if valid_metrics is None else valid_metrics['hard_acc']
            )
            history['test_hard_acc'].append(
                None if test_metrics is None else test_metrics['hard_acc']
            )

            valid_msg = 'n/a' if valid_metrics is None else f"{valid_metrics['hard_acc']:.4f}"
            test_msg = 'n/a' if test_metrics is None else f"{test_metrics['hard_acc']:.4f}"
            print(
                f'step={step:05d} '
                f'soft_loss={float(soft_loss):.6f} '
                f'hard_loss={float(aux["hard"]):.6f} '
                f'valid_hard_acc={valid_msg} '
                f'test_hard_acc={test_msg}'
            )

    dataset_info = {
        'dataset': config['dataset'],
        'input_dim': input_dim,
        'class_count': class_count,
        'logic_family': config['logic_family'],
    }
    return state, wires, history, dataset_info, validation_loader, test_loader
