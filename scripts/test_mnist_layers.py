"""Smoke-test regular and convolutional DLGN layers on MNIST.

Exercises two paths against a real MNIST loader:
1. A regular (flat) DLGN classifier.
2. A convolutional DLGN feature extractor + regular DLGN head.

Usage:
    python scripts/test_mnist_layers.py
    python scripts/test_mnist_layers.py --dataset mnist20x20 --model both
    python scripts/test_mnist_layers.py --train-steps 10
    python scripts/test_mnist_layers.py --preset realistic
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------------------------------------------------------------------------
# Helpers to avoid repeating forward_logits' 11-arg signature everywhere
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadConfig:
    """Bundles the decoder/head args that forward_logits needs beyond params/wires/x."""
    architecture: str = 'softmax'
    logic_family: str = 'full'
    class_count: int = 10
    sum_tau: float = 1.0
    gumb_tau: float = 1.0
    dirichlet_concentration: float = 1.0


def head_forward(params, wires, x, training, key, cfg: HeadConfig):
    """Thin wrapper around forward_logits that unpacks a HeadConfig."""
    from dlgn.models.network import forward_logits
    return forward_logits(
        params, wires, x, training, key,
        cfg.architecture, cfg.gumb_tau, cfg.dirichlet_concentration,
        cfg.class_count, cfg.sum_tau, cfg.logic_family,
    )


def evaluate(params, wires, x, labels, key, cfg: HeadConfig) -> dict[str, float]:
    """Run soft + hard forward, return loss and accuracy metrics."""
    logits_soft = head_forward(params, wires, x, True, key, cfg)
    logits_hard = head_forward(params, wires, x, False, key, cfg)
    return {
        'soft_loss': float(optax.softmax_cross_entropy_with_integer_labels(logits_soft, labels).mean()),
        'hard_loss': float(optax.softmax_cross_entropy_with_integer_labels(logits_hard, labels).mean()),
        'soft_acc': float((jnp.argmax(logits_soft, -1) == labels).mean()),
        'hard_acc': float((jnp.argmax(logits_hard, -1) == labels).mean()),
    }


def fmt(metrics: dict[str, float]) -> str:
    return ' '.join(f'{k}={v:.4f}' for k, v in metrics.items())


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def to_flat_jax(batch):
    """Loader batch → (flat float32, int32 labels). For the regular path."""
    from dlgn.utils.batching import batch_to_jax
    return batch_to_jax(batch)


def to_image_jax(batch):
    """Loader batch → (NHWC float32, int32 labels). For the conv path."""
    x, y = batch
    x = x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    y = y.detach().cpu().numpy() if hasattr(y, 'detach') else np.asarray(y)

    if x.ndim == 4 and x.shape[1] in (1, 3):
        x = np.transpose(x, (0, 2, 3, 1))  # NCHW → NHWC
    elif x.ndim == 3:
        x = x[..., None]

    return jnp.asarray(x.astype(np.float32)), jnp.asarray(y.reshape(-1).astype(np.int32))


# ---------------------------------------------------------------------------
# Regular (flat) DLGN path
# ---------------------------------------------------------------------------

def run_regular_path(args, train_loader, test_loader, class_count):
    from dlgn.data.loaders import cycle_loader
    from dlgn.models.initialization import init_logic_gate_network
    from dlgn.training.optim import create_optimizer
    from dlgn.training.state import TrainState
    from dlgn.training.steps import make_train_step

    cfg = HeadConfig(
        architecture=args.regular_architecture,
        logic_family=args.regular_logic_family,
        class_count=class_count,
        sum_tau=args.sum_tau,
        gumb_tau=args.gumb_tau,
        dirichlet_concentration=args.dirichlet_concentration,
    )

    # --- Init model ---
    sample_x, _ = to_flat_jax(next(iter(train_loader)))
    input_dim = sample_x.shape[-1]

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params, wires = init_logic_gate_network(
        input_dim=input_dim,
        num_neurons=args.regular_neurons,
        num_layers=args.regular_layers,
        connections='random',
        key=init_key,
        logic_family=cfg.logic_family,
    )

    # --- Optimizer ---
    opt_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'clip_value': args.clip_value,
    }
    tx = create_optimizer(opt_config)
    state = TrainState(params=params, opt_state=tx.init(params), key=key)
    train_step = make_train_step(tx)

    # --- Eval before training ---
    test_x, test_y = to_flat_jax(next(iter(test_loader)))
    eval_key = jax.random.PRNGKey(args.seed + 10_000)
    before = evaluate(state.params, wires, test_x, test_y, eval_key, cfg)

    # --- Train ---
    train_iter = cycle_loader(train_loader)
    for _ in range(args.train_steps):
        batch_x, batch_y = to_flat_jax(next(train_iter))
        state, loss, _ = train_step(
            state, batch_x, batch_y, wires,
            cfg.architecture, cfg.gumb_tau, cfg.dirichlet_concentration,
            cfg.class_count, cfg.sum_tau, cfg.logic_family,
        )

    # --- Eval after training ---
    eval_key = jax.random.PRNGKey(args.seed + 20_000)
    after = evaluate(state.params, wires, test_x, test_y, eval_key, cfg)

    print('\n[regular] flattened DLGN on MNIST')
    print(f'  input_dim={input_dim}  neurons={args.regular_neurons}  layers={args.regular_layers}')
    print(f'  before: {fmt(before)}')
    print(f'  after:  {fmt(after)}')
    print(f'  last_train_loss={float(loss):.4f}')
    return after


# ---------------------------------------------------------------------------
# Convolutional DLGN path
# ---------------------------------------------------------------------------

def conv_extract(conv_params, conv_wires, images, training, key,
                 kernel_size, stride, pool_size,
                 conv_architecture='softmax', gumb_tau=1.0,
                 dirichlet_concentration=1.0, conv_logic_family='full'):
    """Conv layer → or-pool → feature maps."""
    from dlgn.models.conv import run_conv_gate_layer, or_pool
    x = run_conv_gate_layer(conv_params, conv_wires, images, training,
                            key=key, kernel_size=kernel_size, stride=stride,
                            architecture=conv_architecture,
                            gumb_tau=gumb_tau,
                            dirichlet_concentration=dirichlet_concentration,
                            logic_family=conv_logic_family)
    if pool_size > 1:
        x = or_pool(x, kernel_size=(pool_size, pool_size))
    return x


def conv_forward(params, wires, images, training, key, cfg: HeadConfig,
                 kernel_size, stride, pool_size,
                 conv_architecture='softmax', conv_logic_family='full'):
    """Full conv pipeline: extract features → flatten → head → class logits."""
    key, conv_key = jax.random.split(key)
    feats = conv_extract(params['conv'], wires['conv'], images, training,
                         conv_key, kernel_size, stride, pool_size,
                         conv_architecture=conv_architecture,
                         gumb_tau=cfg.gumb_tau,
                         dirichlet_concentration=cfg.dirichlet_concentration,
                         conv_logic_family=conv_logic_family)
    flat = feats.reshape(feats.shape[0], -1)
    return head_forward(params['head'], wires['head'], flat, training, key, cfg)


def conv_evaluate(params, wires, images, labels, key, cfg, kernel_size, stride, pool_size,
                  conv_architecture='softmax', conv_logic_family='full'):
    """Soft + hard eval for the conv model."""
    key1, key2 = jax.random.split(key)
    logits_soft = conv_forward(params, wires, images, True, key1, cfg,
                               kernel_size, stride, pool_size,
                               conv_architecture, conv_logic_family)
    logits_hard = conv_forward(params, wires, images, False, key2, cfg,
                               kernel_size, stride, pool_size,
                               conv_architecture, conv_logic_family)
    return {
        'soft_loss': float(optax.softmax_cross_entropy_with_integer_labels(logits_soft, labels).mean()),
        'hard_loss': float(optax.softmax_cross_entropy_with_integer_labels(logits_hard, labels).mean()),
        'soft_acc': float((jnp.argmax(logits_soft, -1) == labels).mean()),
        'hard_acc': float((jnp.argmax(logits_hard, -1) == labels).mean()),
    }


def run_conv_path(args, train_loader, test_loader, class_count):
    from dlgn.data.loaders import cycle_loader
    from dlgn.models.conv import init_conv_gate_layer
    from dlgn.models.initialization import init_logic_gate_network
    from dlgn.training.optim import create_optimizer
    from dlgn.training.state import TrainState

    kernel_size = (args.conv_kernel_size, args.conv_kernel_size)
    stride = (args.conv_stride, args.conv_stride)

    # Conv and head can use different logic families/architectures
    conv_logic = args.conv_logic_family
    conv_arch = args.conv_architecture
    head_logic = args.conv_head_logic_family
    head_arch = args.conv_head_architecture

    cfg = HeadConfig(
        architecture=head_arch,
        logic_family=head_logic,
        class_count=class_count,
        sum_tau=args.sum_tau,
        gumb_tau=args.gumb_tau,
        dirichlet_concentration=args.dirichlet_concentration,
    )

    # --- Init conv layer ---
    sample_images, _ = to_image_jax(next(iter(train_loader)))
    in_channels = sample_images.shape[-1]

    key = jax.random.PRNGKey(args.seed + 1_000)
    key, conv_key, head_key, probe_key = jax.random.split(key, 4)
    conv_params, conv_wires = init_conv_gate_layer(
        conv_key,
        in_channels=in_channels,
        out_channels=args.conv_channels,
        kernel_size=kernel_size,
        depth=args.conv_depth,
        connection_type='random',
        logic_family=conv_logic,
    )

    # --- Init head (sized from a probe forward pass) ---
    probe_feats = conv_extract(conv_params, conv_wires, sample_images, False,
                               probe_key, kernel_size, stride, args.pool_size,
                               conv_architecture=conv_arch,
                               conv_logic_family=conv_logic)
    head_input_dim = int(np.prod(probe_feats.shape[1:]))

    head_params, head_wires = init_logic_gate_network(
        input_dim=head_input_dim,
        num_neurons=args.conv_head_neurons,
        num_layers=args.conv_head_layers,
        connections='random',
        key=head_key,
        logic_family=head_logic,
    )

    params = {'conv': conv_params, 'head': head_params}
    wires = {'conv': conv_wires, 'head': head_wires}

    # --- Optimizer ---
    tx = create_optimizer({
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'clip_value': args.clip_value,
    })
    state = TrainState(params=params, opt_state=tx.init(params), key=key)

    # --- Eval before training ---
    test_images, test_y = to_image_jax(next(iter(test_loader)))
    eval_key = jax.random.PRNGKey(args.seed + 20_000)
    before = conv_evaluate(state.params, wires, test_images, test_y, eval_key,
                           cfg, kernel_size, stride, args.pool_size,
                           conv_arch, conv_logic)

    # --- Train ---
    train_iter = cycle_loader(train_loader)
    for _ in range(args.train_steps):
        batch_images, batch_y = to_image_jax(next(train_iter))
        key, subkey = jax.random.split(state.key)

        def loss_fn(p):
            logits_s = conv_forward(p, wires, batch_images, True, subkey, cfg,
                                    kernel_size, stride, args.pool_size,
                                    conv_arch, conv_logic)
            logits_h = conv_forward(p, wires, batch_images, False, subkey, cfg,
                                    kernel_size, stride, args.pool_size,
                                    conv_arch, conv_logic)
            soft = optax.softmax_cross_entropy_with_integer_labels(logits_s, batch_y).mean()
            hard = optax.softmax_cross_entropy_with_integer_labels(logits_h, batch_y).mean()
            return soft, {'hard': hard}

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        state = state.replace(params=new_params, opt_state=opt_state, key=key)

    # --- Eval after training ---
    eval_key = jax.random.PRNGKey(args.seed + 30_000)
    after = conv_evaluate(state.params, wires, test_images, test_y, eval_key,
                          cfg, kernel_size, stride, args.pool_size,
                          conv_arch, conv_logic)

    print(f'\n[conv] conv DLGN ({conv_logic}) + head ({head_logic}) on MNIST')
    print(f'  input={tuple(sample_images.shape[1:])}  conv_out={args.conv_channels}  '
          f'depth={args.conv_depth}  pooled={tuple(probe_feats.shape[1:])}  '
          f'head_in={head_input_dim}')
    print(f'  before: {fmt(before)}')
    print(f'  after:  {fmt(after)}')
    print(f'  last_train_loss={float(loss):.4f}')
    return after


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PRESETS = {
    'smoke': {},
    'realistic': {
        'regular_neurons': 1020,
        'regular_layers': 8,
        'conv_channels': 32,
        'conv_depth': 2,
        'conv_kernel_size': 3,
        'conv_stride': 1,
        'pool_size': 2,
        'conv_head_neurons': 1020,
        'conv_head_layers': 6,
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Smoke-test DLGN layers on MNIST')

    p.add_argument('--preset', default='smoke', choices=PRESETS.keys())
    p.add_argument('--dataset', default='mnist',
                   choices=['mnist', 'mnist20x20', 'mnist_bin', 'mnist20x20_bin'])
    p.add_argument('--model', default='both', choices=['regular', 'conv', 'both'])
    p.add_argument('--storage-root', default=str(ROOT / 'dataset_storage'), dest='storage_root')
    p.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    p.add_argument('--train-steps', type=int, default=5, dest='train_steps')
    p.add_argument('--seed', type=int, default=0)

    g = p.add_argument_group('regular model')
    g.add_argument('--regular-neurons', type=int, default=40, dest='regular_neurons')
    g.add_argument('--regular-layers', type=int, default=2, dest='regular_layers')
    g.add_argument('--regular-logic-family', default='full', choices=['full', 'light'],
                   dest='regular_logic_family')
    g.add_argument('--regular-architecture', default='softmax', dest='regular_architecture')

    g = p.add_argument_group('conv model')
    g.add_argument('--conv-channels', type=int, default=8, dest='conv_channels')
    g.add_argument('--conv-depth', type=int, default=2, dest='conv_depth')
    g.add_argument('--conv-kernel-size', type=int, default=3, dest='conv_kernel_size')
    g.add_argument('--conv-stride', type=int, default=1, dest='conv_stride')
    g.add_argument('--pool-size', type=int, default=2, dest='pool_size')
    g.add_argument('--conv-logic-family', default='full', choices=['full', 'light'],
                   dest='conv_logic_family')
    g.add_argument('--conv-architecture', default='softmax', dest='conv_architecture')
    g.add_argument('--conv-head-neurons', type=int, default=40, dest='conv_head_neurons')
    g.add_argument('--conv-head-layers', type=int, default=2, dest='conv_head_layers')
    g.add_argument('--conv-head-logic-family', default='full', choices=['full', 'light'],
                   dest='conv_head_logic_family')
    g.add_argument('--conv-head-architecture', default='softmax',
                   dest='conv_head_architecture')

    g = p.add_argument_group('optimizer / decoder')
    g.add_argument('--learning-rate', type=float, default=0.01, dest='learning_rate')
    g.add_argument('--weight-decay', type=float, default=1e-4, dest='weight_decay')
    g.add_argument('--clip-value', type=float, default=1.0, dest='clip_value')
    g.add_argument('--sum-tau', type=float, default=1.0, dest='sum_tau')
    g.add_argument('--gumb-tau', type=float, default=1.0, dest='gumb_tau')
    g.add_argument('--dirichlet-concentration', type=float, default=1.0,
                   dest='dirichlet_concentration')
    return p


def apply_preset(args, cli_argv):
    """Apply preset defaults for any field not explicitly set on the CLI."""
    preset = PRESETS.get(args.preset, {})
    cli_flags = set(cli_argv)
    flag_name = lambda attr: '--' + attr.replace('_', '-')
    for attr, value in preset.items():
        if flag_name(attr) not in cli_flags:
            setattr(args, attr, value)


def main(argv=None) -> int:
    from dlgn.data.loaders import load_dataset
    from dlgn.data.registry import num_classes_of_dataset
    from dlgn.utils.seeding import seed_all

    if argv is None:
        argv = sys.argv[1:]

    args = build_parser().parse_args(argv)
    apply_preset(args, argv)

    storage_root = Path(args.storage_root).resolve()
    seed_all(args.seed, seed_torch=True)

    data_config = {
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

    train_loader, _, test_loader = load_dataset(data_config)
    class_count = num_classes_of_dataset(args.dataset)

    for name, attr in [('regular-neurons', 'regular_neurons'),
                       ('conv-head-neurons', 'conv_head_neurons')]:
        if getattr(args, attr) % class_count != 0:
            print(f'Error: --{name} must be divisible by class_count ({class_count})')
            return 1

    print(f'Dataset: {args.dataset}  classes={class_count}  '
          f'batch={args.batch_size}  steps={args.train_steps}')

    if args.model in ('regular', 'both'):
        run_regular_path(args, train_loader, test_loader, class_count)

    if args.model in ('conv', 'both'):
        run_conv_path(args, train_loader, test_loader, class_count)

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# python scripts/test_mnist_layers.py --model conv \
#   --conv-logic-family full --conv-architecture softmax \
#   --conv-head-logic-family light --conv-head-architecture light_sigmoid

# # All-light
# python scripts/test_mnist_layers.py --model conv \
#   --conv-logic-family light --conv-architecture light_sigmoid \
#   --conv-head-logic-family light --conv-head-architecture light_sigmoid