"""Smoke test: run a short toy_xor DLGN experiment and verify it solves XOR.

Usage:
    python scripts/smoke_test_xor.py --output-dir outputs/smoke

Exit code 0 on success, 1 if hard predictions fail to solve XOR.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import jax
import numpy as np


def main(argv=None) -> int:
    from dlgn.config import validate_logic_config
    from dlgn.training.loop import train_model
    from dlgn.training.checkpoints import save_checkpoint
    from dlgn.models.network import forward_logits
    from dlgn.utils.paths import make_run_dir
    from dlgn.utils.batching import batch_to_jax

    parser = argparse.ArgumentParser(description='DLGN XOR smoke test')
    parser.add_argument('--output-dir', default='outputs/smoke', dest='output_dir')
    parser.add_argument('--num-steps', type=int, default=300, dest='num_steps')
    args = parser.parse_args(argv)

    config = {
        'dataset': 'toy_xor',
        'seed': 42,
        'batch_size': 32,
        'valid_set_size': 0.0,
        'num_workers': 0,
        'num_steps': args.num_steps,
        'eval_every': 100,
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
        'data_roots': {'uci': './data-uci', 'mnist': './data-mnist',
                       'cifar': './data-cifar', 'block': './blocks_dataset'},
    }
    validate_logic_config(config)

    print('Running XOR smoke test ...')
    state, wires, history, dataset_info, _, test_loader = train_model(config)

    # Evaluate hard predictions on the 4 XOR patterns
    xor_x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
    xor_y = np.array([0, 1, 1, 0], dtype=np.int32)
    import jax.numpy as jnp
    x_jax = jnp.asarray(xor_x)
    logits = forward_logits(
        state.params, wires, x_jax, False,
        jax.random.PRNGKey(0),
        config['architecture'], config['gumb_tau'], config['dirichlet_concentration'],
        dataset_info['class_count'], config['sum_tau'], config['logic_family'],
    )
    preds = np.asarray(jnp.argmax(logits, axis=-1))
    correct = (preds == xor_y).all()

    print(f'XOR predictions: {preds.tolist()}  expected: {xor_y.tolist()}')
    print(f'Hard XOR solved: {correct}')

    # Save a small checkpoint for inspection
    run_dir = make_run_dir(args.output_dir, 'xor_smoke')
    save_checkpoint(
        run_dir / 'checkpoints' / 'final.pkl',
        state, wires, config, history, dataset_info,
    )
    print(f'Checkpoint saved to {run_dir.resolve()}')

    if not correct:
        print('SMOKE TEST FAILED: XOR not solved with hard gates.', file=sys.stderr)
        return 1

    print('SMOKE TEST PASSED.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
