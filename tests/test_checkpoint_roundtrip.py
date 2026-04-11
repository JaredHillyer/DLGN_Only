# Guards checkpoint save/load contract.
# Payload fields: step, params, opt_state, key, wires, config, history, dataset_info, extra
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dlgn.models.initialization import init_logic_gate_network
from dlgn.training.optim import create_optimizer
from dlgn.training.state import TrainState
from dlgn.training.checkpoints import save_checkpoint, load_checkpoint


def _make_state_and_wires():
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    config = {
        'learning_rate': 0.01, 'weight_decay': 1e-4, 'clip_value': 1.0,
    }
    params, wires = init_logic_gate_network(
        input_dim=8, num_neurons=16, num_layers=2,
        connections='unique', key=init_key, logic_family='full',
    )
    tx = create_optimizer(config)
    state = TrainState(params=params, opt_state=tx.init(params), key=key)
    return state, wires


def test_checkpoint_roundtrip_payload_keys():
    state, wires = _make_state_and_wires()
    config = {'dataset': 'toy_xor', 'logic_family': 'full', 'architecture': 'softmax'}
    history = {'steps': [0, 1], 'train_soft_loss': [0.7, 0.5],
               'train_hard_loss': [0.8, 0.6], 'valid_hard_acc': [None],
               'test_hard_acc': [0.75]}
    dataset_info = {'dataset': 'toy_xor', 'input_dim': 2, 'class_count': 2,
                    'logic_family': 'full'}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'test.pkl'
        save_checkpoint(path, state, wires, config, history, dataset_info,
                        step=1, extra={'test_key': 42})

        payload = load_checkpoint(path)

    required = {'step', 'params', 'opt_state', 'key', 'wires', 'config',
                'history', 'dataset_info', 'extra'}
    for k in required:
        assert k in payload, f'Missing key: {k}'

    assert payload['step'] == 1
    assert payload['config'] == config
    assert payload['dataset_info'] == dataset_info
    assert payload['extra'] == {'test_key': 42}


def test_checkpoint_params_survive_roundtrip():
    state, wires = _make_state_and_wires()
    config = {'dataset': 'toy_xor', 'logic_family': 'full', 'architecture': 'softmax'}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'test.pkl'
        save_checkpoint(path, state, wires, config)
        payload = load_checkpoint(path)

    restored_state = payload['state']
    for orig, loaded in zip(state.params, restored_state.params):
        np.testing.assert_array_almost_equal(
            np.asarray(orig), np.asarray(loaded), decimal=6
        )


def test_checkpoint_wires_survive_roundtrip():
    state, wires = _make_state_and_wires()
    config = {'dataset': 'toy_xor', 'logic_family': 'full', 'architecture': 'softmax'}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'test.pkl'
        save_checkpoint(path, state, wires, config)
        payload = load_checkpoint(path)

    restored_wires = payload['wires']
    for (a, b), (ra, rb) in zip(wires, restored_wires):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(ra))
        np.testing.assert_array_equal(np.asarray(b), np.asarray(rb))


def test_load_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        load_checkpoint('/tmp/does_not_exist_dlgn_test.pkl')
