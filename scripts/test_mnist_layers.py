"""Smoke-test regular and convolutional DLGN layers on MNIST.

This script exercises two paths against a real MNIST loader:

1. A regular DLGN classifier on flattened images.
2. A convolutional DLGN feature extractor followed by a regular DLGN head.

The convolutional path is intentionally explicit here because the active package
does not yet expose a first-class conv training loop.

Usage:
    python scripts/test_mnist_layers.py
    python scripts/test_mnist_layers.py --dataset mnist20x20 --model both
    python scripts/test_mnist_layers.py --train-steps 10 --storage-root dataset_storage
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
import jax.numpy as jnp
import numpy as np
import optax


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Smoke-test regular and convolutional DLGN models on MNIST'
    )
    parser.add_argument(
        '--dataset',
        default='mnist',
        choices=['mnist', 'mnist20x20', 'mnist_bin', 'mnist20x20_bin'],
        help='MNIST variant to load',
    )
    parser.add_argument(
        '--model',
        default='both',
        choices=['regular', 'conv', 'both'],
        help='Which model path to run',
    )
    parser.add_argument(
        '--storage-root',
        default=str(ROOT / 'dataset_storage'),
        help='Root directory for dataset files',
    )
    parser.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    parser.add_argument('--train-steps', type=int, default=5, dest='train_steps')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--regular-neurons', type=int, default=40, dest='regular_neurons')
    parser.add_argument('--regular-layers', type=int, default=2, dest='regular_layers')
    parser.add_argument(
        '--regular-logic-family',
        default='full',
        choices=['full', 'light'],
        dest='regular_logic_family',
    )
    parser.add_argument(
        '--regular-architecture',
        default='softmax',
        dest='regular_architecture',
        help='Decoder architecture for the regular model',
    )

    parser.add_argument('--conv-channels', type=int, default=8, dest='conv_channels')
    parser.add_argument('--conv-depth', type=int, default=2, dest='conv_depth')
    parser.add_argument('--conv-kernel-size', type=int, default=3, dest='conv_kernel_size')
    parser.add_argument('--conv-stride', type=int, default=1, dest='conv_stride')
    parser.add_argument('--pool-size', type=int, default=2, dest='pool_size')
    parser.add_argument('--conv-head-neurons', type=int, default=40, dest='conv_head_neurons')
    parser.add_argument('--conv-head-layers', type=int, default=2, dest='conv_head_layers')

    parser.add_argument('--learning-rate', type=float, default=0.01, dest='learning_rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, dest='weight_decay')
    parser.add_argument('--clip-value', type=float, default=1.0, dest='clip_value')
    parser.add_argument('--sum-tau', type=float, default=1.0, dest='sum_tau')
    parser.add_argument('--gumb-tau', type=float, default=1.0, dest='gumb_tau')
    parser.add_argument(
        '--dirichlet-concentration',
        type=float,
        default=1.0,
        dest='dirichlet_concentration',
    )
    return parser


def build_data_config(args: argparse.Namespace) -> dict:
    storage_root = Path(args.storage_root)
    return {
        'dataset': args.dataset,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'valid_set_size': 0.0,
        'num_workers': 0,
        'data_roots': {
            'uci': str(storage_root / 'uci'),
            'mnist': str(storage_root / 'mnist'),
            'cifar': str(storage_root / 'cifar'),
            'block': str(storage_root / 'block'),
        },
    }


def batch_to_image_jax(batch) -> tuple[jax.Array, jax.Array]:
    """Convert a loader batch to NHWC images and int32 labels."""
    x, y = batch

    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if hasattr(y, 'detach'):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)

    if x.ndim == 4 and x.shape[1] in (1, 3):
        x = np.transpose(x, (0, 2, 3, 1))
    elif x.ndim == 3:
        x = x[..., None]

    x = x.astype(np.float32)
    y = y.reshape(-1).astype(np.int32)
    return jnp.asarray(x), jnp.asarray(y)


def compute_metrics(
    logits_soft: jax.Array,
    logits_hard: jax.Array,
    labels: jax.Array,
) -> dict[str, float]:
    return {
        'soft_loss': float(
            optax.softmax_cross_entropy_with_integer_labels(logits_soft, labels).mean()
        ),
        'hard_loss': float(
            optax.softmax_cross_entropy_with_integer_labels(logits_hard, labels).mean()
        ),
        'soft_acc': float((jnp.argmax(logits_soft, axis=-1) == labels).mean()),
        'hard_acc': float((jnp.argmax(logits_hard, axis=-1) == labels).mean()),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"soft_loss={metrics['soft_loss']:.4f} "
        f"hard_loss={metrics['hard_loss']:.4f} "
        f"soft_acc={metrics['soft_acc']:.4f} "
        f"hard_acc={metrics['hard_acc']:.4f}"
    )


def run_regular_path(
    args: argparse.Namespace,
    train_loader,
    test_loader,
    class_count: int,
) -> dict[str, float]:
    from dlgn.config import validate_logic_config
    from dlgn.data.loaders import cycle_loader
    from dlgn.models.initialization import init_logic_gate_network
    from dlgn.models.network import forward_logits
    from dlgn.training.optim import create_optimizer
    from dlgn.training.state import TrainState
    from dlgn.training.steps import make_train_step
    from dlgn.utils.batching import batch_to_jax

    config = {
        'dataset': args.dataset,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'valid_set_size': 0.0,
        'num_workers': 0,
        'num_steps': args.train_steps,
        'eval_every': args.train_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'clip_value': args.clip_value,
        'connections': 'random',
        'logic_family': args.regular_logic_family,
        'architecture': args.regular_architecture,
        'gumb_tau': args.gumb_tau,
        'dirichlet_concentration': args.dirichlet_concentration,
        'sum_tau': args.sum_tau,
        'num_neurons': args.regular_neurons,
        'num_layers': args.regular_layers,
    }
    validate_logic_config(config)

    train_batch_x, _ = batch_to_jax(next(iter(train_loader)))
    input_dim = int(train_batch_x.shape[-1])

    key = jax.random.PRNGKey(args.seed)
    key, init_key, eval_key = jax.random.split(key, 3)
    params, wires = init_logic_gate_network(
        input_dim=input_dim,
        num_neurons=args.regular_neurons,
        num_layers=args.regular_layers,
        connections='random',
        key=init_key,
        logic_family=args.regular_logic_family,
    )

    tx = create_optimizer(config)
    train_step = make_train_step(tx)
    state = TrainState(params=params, opt_state=tx.init(params), key=key)
    train_iter = cycle_loader(train_loader)

    test_batch = next(iter(test_loader))
    test_x, test_y = batch_to_jax(test_batch)

    logits_soft = forward_logits(
        state.params,
        wires,
        test_x,
        True,
        eval_key,
        args.regular_architecture,
        args.gumb_tau,
        args.dirichlet_concentration,
        class_count,
        args.sum_tau,
        args.regular_logic_family,
    )
    logits_hard = forward_logits(
        state.params,
        wires,
        test_x,
        False,
        eval_key,
        args.regular_architecture,
        args.gumb_tau,
        args.dirichlet_concentration,
        class_count,
        args.sum_tau,
        args.regular_logic_family,
    )
    before = compute_metrics(logits_soft, logits_hard, test_y)

    last_train_loss = None
    for _ in range(args.train_steps):
        batch_x, batch_y = batch_to_jax(next(train_iter))
        state, loss, _ = train_step(
            state,
            batch_x,
            batch_y,
            wires,
            args.regular_architecture,
            args.gumb_tau,
            args.dirichlet_concentration,
            class_count,
            args.sum_tau,
            args.regular_logic_family,
        )
        last_train_loss = float(loss)

    eval_key = jax.random.PRNGKey(args.seed + 10_000)
    logits_soft = forward_logits(
        state.params,
        wires,
        test_x,
        True,
        eval_key,
        args.regular_architecture,
        args.gumb_tau,
        args.dirichlet_concentration,
        class_count,
        args.sum_tau,
        args.regular_logic_family,
    )
    logits_hard = forward_logits(
        state.params,
        wires,
        test_x,
        False,
        eval_key,
        args.regular_architecture,
        args.gumb_tau,
        args.dirichlet_concentration,
        class_count,
        args.sum_tau,
        args.regular_logic_family,
    )
    after = compute_metrics(logits_soft, logits_hard, test_y)

    print('\n[regular] flattened DLGN on MNIST')
    print(f'input_dim={input_dim} neurons={args.regular_neurons} layers={args.regular_layers}')
    print(f'before: {format_metrics(before)}')
    print(f'after:  {format_metrics(after)}')
    if last_train_loss is not None:
        print(f'last_train_soft_loss={last_train_loss:.4f}')
    return after


def conv_features(
    conv_params,
    conv_wires,
    images: jax.Array,
    training: bool,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    pool_size: int,
) -> jax.Array:
    from dlgn.models.conv import or_pool, run_conv_gate_layer

    x = run_conv_gate_layer(
        conv_params,
        conv_wires,
        images,
        training,
        kernel_size=kernel_size,
        stride=stride,
    )
    if pool_size > 1:
        x = or_pool(
            x,
            kernel_size=(pool_size, pool_size),
            stride=(pool_size, pool_size),
        )
    return x


def conv_forward_logits(
    params: dict,
    wires: dict,
    images: jax.Array,
    training: bool,
    key: jax.Array,
    class_count: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    pool_size: int,
    sum_tau: float,
) -> jax.Array:
    from dlgn.models.network import forward_logits

    conv_key, head_key = jax.random.split(key)
    feats = conv_features(
        params['conv'],
        wires['conv'],
        images,
        training,
        kernel_size,
        stride,
        pool_size,
    )
    flat = feats.reshape(feats.shape[0], -1)
    return forward_logits(
        params['head'],
        wires['head'],
        flat,
        training,
        head_key,
        'softmax',
        1.0,
        1.0,
        class_count,
        sum_tau,
        'full',
    )


def make_conv_train_step(
    tx: optax.GradientTransformation,
    *,
    class_count: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    pool_size: int,
    sum_tau: float,
):
    from dlgn.training.state import TrainState

    def train_step(
        state: TrainState,
        images: jax.Array,
        labels: jax.Array,
        wires: dict,
    ) -> tuple[TrainState, jax.Array, dict[str, jax.Array]]:
        key, subkey = jax.random.split(state.key)

        def apply_loss(params):
            logits_soft = conv_forward_logits(
                params,
                wires,
                images,
                True,
                subkey,
                class_count,
                kernel_size,
                stride,
                pool_size,
                sum_tau,
            )
            logits_hard = conv_forward_logits(
                params,
                wires,
                images,
                False,
                subkey,
                class_count,
                kernel_size,
                stride,
                pool_size,
                sum_tau,
            )
            soft_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits_soft,
                labels,
            ).mean()
            hard_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits_hard,
                labels,
            ).mean()
            return soft_loss, {'hard': hard_loss}

        (loss, aux), grads = jax.value_and_grad(apply_loss, has_aux=True)(state.params)
        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        state = state.replace(params=params, opt_state=opt_state, key=key)
        return state, loss, aux

    return train_step


def run_conv_path(
    args: argparse.Namespace,
    train_loader,
    test_loader,
    class_count: int,
) -> dict[str, float]:
    from dlgn.data.loaders import cycle_loader
    from dlgn.models.conv import init_conv_gate_layer
    from dlgn.models.initialization import init_logic_gate_network
    from dlgn.training.optim import create_optimizer
    from dlgn.training.state import TrainState

    kernel_size = (args.conv_kernel_size, args.conv_kernel_size)
    stride = (args.conv_stride, args.conv_stride)

    config = {
        'dataset': args.dataset,
        'seed': args.seed,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'clip_value': args.clip_value,
    }
    tx = create_optimizer(config)

    train_images, _ = batch_to_image_jax(next(iter(train_loader)))
    in_channels = int(train_images.shape[-1])

    key = jax.random.PRNGKey(args.seed + 1_000)
    key, conv_key, head_key = jax.random.split(key, 3)
    conv_params, conv_wires = init_conv_gate_layer(
        conv_key,
        in_channels=in_channels,
        out_channels=args.conv_channels,
        kernel_size=kernel_size,
        depth=args.conv_depth,
        connection_type='random',
        logic_family='full',
    )

    sample_feats = conv_features(
        conv_params,
        conv_wires,
        train_images,
        False,
        kernel_size,
        stride,
        args.pool_size,
    )
    head_input_dim = int(np.prod(sample_feats.shape[1:]))
    head_params, head_wires = init_logic_gate_network(
        input_dim=head_input_dim,
        num_neurons=args.conv_head_neurons,
        num_layers=args.conv_head_layers,
        connections='random',
        key=head_key,
        logic_family='full',
    )

    params = {'conv': conv_params, 'head': head_params}
    wires = {'conv': conv_wires, 'head': head_wires}
    state = TrainState(params=params, opt_state=tx.init(params), key=key)
    train_step = make_conv_train_step(
        tx,
        class_count=class_count,
        kernel_size=kernel_size,
        stride=stride,
        pool_size=args.pool_size,
        sum_tau=args.sum_tau,
    )
    train_iter = cycle_loader(train_loader)

    test_images, test_y = batch_to_image_jax(next(iter(test_loader)))
    eval_key = jax.random.PRNGKey(args.seed + 20_000)
    logits_soft = conv_forward_logits(
        state.params,
        wires,
        test_images,
        True,
        eval_key,
        class_count,
        kernel_size,
        stride,
        args.pool_size,
        args.sum_tau,
    )
    logits_hard = conv_forward_logits(
        state.params,
        wires,
        test_images,
        False,
        eval_key,
        class_count,
        kernel_size,
        stride,
        args.pool_size,
        args.sum_tau,
    )
    before = compute_metrics(logits_soft, logits_hard, test_y)

    last_train_loss = None
    for _ in range(args.train_steps):
        batch_x, batch_y = batch_to_image_jax(next(train_iter))
        state, loss, _ = train_step(state, batch_x, batch_y, wires)
        last_train_loss = float(loss)

    eval_key = jax.random.PRNGKey(args.seed + 30_000)
    logits_soft = conv_forward_logits(
        state.params,
        wires,
        test_images,
        True,
        eval_key,
        class_count,
        kernel_size,
        stride,
        args.pool_size,
        args.sum_tau,
    )
    logits_hard = conv_forward_logits(
        state.params,
        wires,
        test_images,
        False,
        eval_key,
        class_count,
        kernel_size,
        stride,
        args.pool_size,
        args.sum_tau,
    )
    after = compute_metrics(logits_soft, logits_hard, test_y)

    print('\n[conv] conv DLGN features + regular DLGN head on MNIST')
    print(
        'input_shape='
        f'{tuple(train_images.shape[1:])} '
        f'conv_channels={args.conv_channels} '
        f'conv_depth={args.conv_depth} '
        f'pooled_feature_shape={tuple(sample_feats.shape[1:])} '
        f'head_input_dim={head_input_dim}'
    )
    print(f'before: {format_metrics(before)}')
    print(f'after:  {format_metrics(after)}')
    if last_train_loss is not None:
        print(f'last_train_soft_loss={last_train_loss:.4f}')
    return after


def main(argv=None) -> int:
    from dlgn.data.loaders import load_dataset
    from dlgn.data.registry import num_classes_of_dataset
    from dlgn.utils.seeding import seed_all

    parser = build_parser()
    args = parser.parse_args(argv)

    storage_root = Path(args.storage_root).resolve()
    print(f'Dataset storage root: {storage_root}')

    data_config = build_data_config(args)
    seed_all(args.seed, seed_torch=True)
    train_loader, _, test_loader = load_dataset(data_config)
    class_count = num_classes_of_dataset(args.dataset)

    if args.regular_neurons % class_count != 0:
        parser.error(
            '--regular-neurons must be divisible by the class count '
            f'({class_count}) for GroupSum'
        )
    if args.conv_head_neurons % class_count != 0:
        parser.error(
            '--conv-head-neurons must be divisible by the class count '
            f'({class_count}) for GroupSum'
        )

    print(f'Dataset: {args.dataset}  class_count={class_count}  batch_size={args.batch_size}')
    print(f'Train steps per path: {args.train_steps}')

    if args.model in ('regular', 'both'):
        run_regular_path(args, train_loader, test_loader, class_count)

    if args.model in ('conv', 'both'):
        run_conv_path(args, train_loader, test_loader, class_count)

    print('\nMNIST layer smoke test complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
