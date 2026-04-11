# End-to-end DLGN smoke test: short XOR run should reach correct hard predictions.
# This test guards the full forward-pass stack: init → train → hard eval.
import jax
import jax.numpy as jnp
import numpy as np
import pytest


CONFIG = {
    'dataset': 'toy_xor',
    'seed': 42,
    'batch_size': 32,
    'valid_set_size': 0.0,
    'num_workers': 0,
    'num_steps': 400,
    'eval_every': 200,
    'learning_rate': 0.03,
    'weight_decay': 1e-4,
    'clip_value': 1.0,
    'connections': 'random',
    'logic_family': 'full',
    'architecture': 'softmax',
    'gumb_tau': 1.0,
    'dirichlet_concentration': 1.0,
    'sum_tau': 1.0,
    'num_neurons': 4,
    'num_layers': 2,
    'toy_num_bits': 2,
    'data_roots': {
        'block': './blocks_dataset', 'uci': './data-uci',
        'mnist': './data-mnist', 'cifar': './data-cifar',
    },
}


def test_xor_hard_predictions_correct():
    """After a short training run, the DLGN should solve XOR with hard gates."""
    from dlgn.training.loop import train_model
    from dlgn.models.network import forward_logits

    state, wires, history, dataset_info, _, _ = train_model(CONFIG)

    xor_x = jnp.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    xor_y = np.array([0, 1, 1, 0])

    logits = forward_logits(
        state.params, wires, xor_x, False,
        jax.random.PRNGKey(0),
        CONFIG['architecture'], CONFIG['gumb_tau'], CONFIG['dirichlet_concentration'],
        dataset_info['class_count'], CONFIG['sum_tau'], CONFIG['logic_family'],
    )
    preds = np.asarray(jnp.argmax(logits, axis=-1))
    assert (preds == xor_y).all(), (
        f'XOR not solved: predictions={preds.tolist()} expected={xor_y.tolist()}'
    )


def test_xor_history_keys_present():
    """Training history must contain all canonical notebook keys."""
    from dlgn.training.loop import train_model

    _, _, history, _, _, _ = train_model({**CONFIG, 'num_steps': 10, 'eval_every': 5})

    required_keys = {'steps', 'train_soft_loss', 'train_hard_loss',
                     'valid_hard_acc', 'test_hard_acc'}
    for k in required_keys:
        assert k in history, f'Missing history key: {k}'
        assert isinstance(history[k], list), f'{k} is not a list'


def test_xor_dataset_info_fields():
    """dataset_info must contain the required fields."""
    from dlgn.training.loop import train_model

    _, _, _, dataset_info, _, _ = train_model({**CONFIG, 'num_steps': 5, 'eval_every': 5})

    for field in ('dataset', 'input_dim', 'class_count', 'logic_family'):
        assert field in dataset_info, f'Missing dataset_info field: {field}'

    assert dataset_info['dataset'] == 'toy_xor'
    assert dataset_info['input_dim'] == 2
    assert dataset_info['class_count'] == 2
    assert dataset_info['logic_family'] == 'full'
