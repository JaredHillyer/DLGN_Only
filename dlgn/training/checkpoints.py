# Source: ported from Older_Imp.ipynb cell 6
# Checkpoint save/load/resume. Payload fields must not be removed or renamed.
# Required payload keys: step, params, opt_state, key, wires, config, history,
#                        dataset_info, extra
from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dlgn.training.state import TrainState
from dlgn.types import WirePair
from dlgn.utils.tree_io import tree_to_numpy, tree_to_jax


def save_checkpoint(
    path,
    state: TrainState,
    wires: list[WirePair],
    config: dict,
    history: dict | None = None,
    dataset_info: dict | None = None,
    step: int | None = None,
    extra: dict | None = None,
) -> None:
    """Serialize a training checkpoint to disk.

    The checkpoint payload always contains the canonical training fields.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if step is None:
        if history is not None and history.get('steps'):
            step = int(history['steps'][-1])
        else:
            step = 0

    payload = {
        'step': step,
        'params': tree_to_numpy(state.params),
        'opt_state': tree_to_numpy(state.opt_state),
        'key': np.asarray(jax.device_get(state.key)),
        'wires': tree_to_numpy(wires),
        'config': config,
        'history': history,
        'dataset_info': dataset_info,
        'extra': {} if extra is None else extra,
    }

    with path.open('wb') as f:
        pickle.dump(payload, f)

    print(f'Saved checkpoint to {path.resolve()}')


def load_checkpoint(path) -> dict:
    """Load a checkpoint and reconstruct TrainState and wires.

    Returns the raw payload dict with an additional 'state' key containing
    the reconstructed TrainState.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path.resolve()}')

    with path.open('rb') as f:
        payload = pickle.load(f)

    # Validate required fields
    required_keys = {'step', 'params', 'opt_state', 'key', 'wires', 'config'}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(
            f'Checkpoint at {path} is missing required fields: {missing}'
        )

    state = TrainState(
        params=tree_to_jax(payload['params']),
        opt_state=tree_to_jax(payload['opt_state']),
        key=jnp.asarray(payload['key']),
    )
    wires = tree_to_jax(payload['wires'])

    payload['state'] = state
    payload['wires'] = wires
    return payload


def resume_training_from_checkpoint(path, config_override: dict | None = None):
    """Load a checkpoint and resume training from the saved step + 1.

    config_override keys are merged on top of the saved config.
    Any override should be intentional and visible in the run's config snapshot.
    """
    from dlgn.training.loop import train_model

    checkpoint = load_checkpoint(path)
    merged_config = dict(checkpoint['config'])
    if config_override is not None:
        merged_config.update(config_override)

    return train_model(
        merged_config,
        initial_state=checkpoint['state'],
        initial_wires=checkpoint['wires'],
        history=checkpoint.get('history'),
        start_step=int(checkpoint.get('step', -1)) + 1,
    )
