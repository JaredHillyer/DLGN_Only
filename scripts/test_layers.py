"""Train and evaluate DLGN models on MNIST.

Supports two model types:
1. A regular (flat) DLGN classifier.
2. A deep convolutional DLGN: multiple conv blocks → flatten → FC stack → GroupSum.

Usage:
    python scripts/test_mnist_layers.py --preset smoke
    python scripts/test_mnist_layers.py --preset small --model conv
    python scripts/test_mnist_layers.py --preset medium --model conv

Overnight runs:
    nohup python scripts/test_mnist_layers.py --preset small --model conv > small.log 2>&1 &
    nohup python scripts/test_mnist_layers.py --preset medium --model conv > medium.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
# Helpers
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
    soft_acc = float((jnp.argmax(logits_soft, -1) == labels).mean())
    hard_acc = float((jnp.argmax(logits_hard, -1) == labels).mean())
    return {
        'soft_loss': float(optax.softmax_cross_entropy_with_integer_labels(logits_soft, labels).mean()),
        'hard_loss': float(optax.softmax_cross_entropy_with_integer_labels(logits_hard, labels).mean()),
        'soft_acc': soft_acc,
        'hard_acc': hard_acc,
        'soft_hard_gap': soft_acc - hard_acc,  # B-perf-9
    }


def fmt(metrics: dict[str, float]) -> str:
    return ' '.join(f'{k}={v:.4f}' for k, v in metrics.items())


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def to_flat_jax(batch):
    """Loader batch → (flat float32, int32 labels). For the regular path."""
    from dlgn.utils import batch_to_jax
    return batch_to_jax(batch)


def to_image_jax(batch):
    """Loader batch → (NHWC float32, int32 labels). For the conv path."""
    x, y = batch
    x = x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    y = y.detach().cpu().numpy() if hasattr(y, 'detach') else np.asarray(y)

    if x.ndim == 4:
        x = np.transpose(x, (0, 2, 3, 1))  # NCHW → NHWC (any channel count)
    elif x.ndim == 3:
        x = x[..., None]

    return jnp.asarray(x.astype(np.float32)), jnp.asarray(y.reshape(-1).astype(np.int32))


# ---------------------------------------------------------------------------
# Regular (flat) DLGN path — unchanged
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
# Deep convolutional DLGN path
# ---------------------------------------------------------------------------

def init_fc_stack(key, layer_sizes, connection_type, logic_family):
    """Init FC DLGN layers with variable widths.

    Args:
        layer_sizes: [input_dim, hidden_1, hidden_2, ..., output_dim]
    Returns:
        (params, wires) — same list format as init_logic_gate_network.
    """
    from dlgn.models.initialization import init_gate_layer
    params, wires = [], []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, subkey = jax.random.split(key)
        p, w = init_gate_layer(subkey, int(in_dim), int(out_dim),
                               connection_type, logic_family)
        params.append(p)
        wires.append(w)
    return params, wires


# D25: remat policy lookup. dots saves matmul results so they don't get
# recomputed; nothing recomputes everything.
REMAT_POLICIES = {
    'dots':    jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    'nothing': jax.checkpoint_policies.nothing_saveable,
}


def deep_conv_forward(params, wires, images, training, key, cfg,
                      block_kernels, block_strides, block_paddings,
                      pool_size, pool_stride, pool_type='or',
                      remat_conv=False, remat_fc=False, remat_policy=None):
    """Conv blocks → flatten → FC head → class logits.

    pool_type selects the spatial reduction after each conv block:
        'or'  → or_pool   (probabilistic t-conorm; default)
        'max' → max_pool  (lax.reduce_window(lax.max))

    remat_conv / remat_fc independently wrap each per-block body and/or
    the FC head in ``jax.checkpoint`` so activations get recomputed on
    the backward pass instead of saved. Numerics are preserved up to
    float32 round-off (~1e-5 absolute): the forward output is
    mathematically equivalent but XLA may re-fuse ops inside the
    checkpointed region, producing LSB-level differences vs remat=off.
    RNG splits happen OUTSIDE the checkpointed region so the same
    conv_key reaches the block body on both forward and backward.
    """
    from dlgn.models.conv import run_conv_gate_layer, or_pool, max_pool

    if pool_type == 'or':
        pool_fn = or_pool
    elif pool_type == 'max':
        pool_fn = max_pool
    else:
        raise ValueError(f"pool_type must be 'or' or 'max', got {pool_type!r}")

    n_blocks = len(params['conv'])

    def block_body(block_x, block_params, block_wires, conv_key, ks, st, pad):
        """Per-block forward: pad → conv → pool. Pure of side effects."""
        if pad > 0:
            block_x = jnp.pad(block_x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        block_x = run_conv_gate_layer(
            block_params, block_wires, block_x, training, conv_key,
            kernel_size=ks, stride=st,
            architecture=cfg.architecture, logic_family=cfg.logic_family,
        )
        return pool_fn(block_x, kernel_size=(pool_size, pool_size),
                       stride=(pool_stride, pool_stride))

    if remat_conv:
        # Wrap the block body so backward recomputes the activations.
        # ks, st, pad (positions 4, 5, 6) are Python ints / tuples and must
        # be marked static for jax.checkpoint to avoid TracerBoolConversionError
        # on `if pad > 0:` inside the body.
        block_fn = jax.checkpoint(
            block_body, policy=remat_policy, static_argnums=(4, 5, 6),
        )
    else:
        block_fn = block_body

    x = images
    for i in range(n_blocks):
        # RNG split happens OUTSIDE the checkpointed region.
        key, conv_key = jax.random.split(key)
        ks = (block_kernels[i], block_kernels[i])
        st = (block_strides[i], block_strides[i])
        x = block_fn(x, params['conv'][i], wires['conv'][i], conv_key,
                     ks, st, block_paddings[i])

    flat = x.reshape(x.shape[0], -1)

    if remat_fc:
        # head_forward signature: (params, wires, x, training, key, cfg).
        # Pass training (Python bool) and cfg (HeadConfig dataclass) as
        # static args so jax.checkpoint can compile the head with them
        # baked in. Args 3 and 5 in the call below.
        head_fn = jax.checkpoint(
            head_forward, policy=remat_policy, static_argnums=(3, 5),
        )
        return head_fn(params['fc'], wires['fc'], flat, training, key, cfg)
    return head_forward(params['fc'], wires['fc'], flat, training, key, cfg)


def deep_conv_evaluate(params, wires, images, labels, key, cfg, **conv_kw):
    """Soft + hard eval for the deep conv model."""
    key1, key2 = jax.random.split(key)
    logits_soft = deep_conv_forward(params, wires, images, True, key1, cfg, **conv_kw)
    logits_hard = deep_conv_forward(params, wires, images, False, key2, cfg, **conv_kw)
    soft_acc = float((jnp.argmax(logits_soft, -1) == labels).mean())
    hard_acc = float((jnp.argmax(logits_hard, -1) == labels).mean())
    return {
        'soft_loss': float(optax.softmax_cross_entropy_with_integer_labels(logits_soft, labels).mean()),
        'hard_loss': float(optax.softmax_cross_entropy_with_integer_labels(logits_hard, labels).mean()),
        'soft_acc': soft_acc,
        'hard_acc': hard_acc,
        'soft_hard_gap': soft_acc - hard_acc,  # B-perf-9
    }


def make_deep_conv_step(tx, wires, cfg, conv_kw):
    """Return a JIT-compiled train step for the deep conv model."""

    @jax.jit
    def step(params, opt_state, images, labels, key):
        key, subkey = jax.random.split(key)

        def loss_fn(p):
            logits = deep_conv_forward(p, wires, images, True, subkey, cfg, **conv_kw)
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, key, loss

    return step


def run_deep_conv_path(args, train_loader, test_loader, class_count):
    """Multi-block conv + variable-width FC stack."""
    from dlgn.data.loaders import cycle_loader
    from dlgn.models.conv import init_conv_gate_layer, run_conv_gate_layer, or_pool, max_pool
    from dlgn.training.optim import create_optimizer

    logic_family = args.logic_family
    architecture = args.architecture

    cfg = HeadConfig(
        architecture=architecture,
        logic_family=logic_family,
        class_count=class_count,
        sum_tau=args.sum_tau,
        gumb_tau=args.gumb_tau,
        dirichlet_concentration=args.dirichlet_concentration,
    )

    # Parse block configs
    block_channels = args.conv_block_channels
    block_kernels = args.conv_block_kernel_sizes
    block_depths = args.conv_block_depths
    block_paddings = args.conv_block_paddings
    block_strides = args.conv_block_strides
    n_blocks = len(block_channels)

    pool_size = args.pool_size
    pool_stride = args.pool_stride
    pool_type = args.pool_type
    fc_sizes = args.fc_sizes

    if pool_type == 'or':
        probe_pool_fn = or_pool
    elif pool_type == 'max':
        probe_pool_fn = max_pool
    else:
        raise ValueError(f"--pool-type must be 'or' or 'max', got {pool_type!r}")

    # D25: remat plumbing. Translate the unified --remat / --remat-policy
    # CLI pair into two boolean toggles for deep_conv_forward.
    remat_arg = getattr(args, 'remat', 'off')
    remat_conv = remat_arg in ('conv', 'both')
    remat_fc = remat_arg in ('fc', 'both')
    remat_policy_str = getattr(args, 'remat_policy', 'dots')
    remat_policy = (REMAT_POLICIES[remat_policy_str]
                    if (remat_conv or remat_fc) else None)

    # Validate list lengths
    for name, lst in [('kernel_sizes', block_kernels), ('depths', block_depths),
                      ('paddings', block_paddings), ('strides', block_strides)]:
        if len(lst) != n_blocks:
            raise ValueError(
                f'conv_block_{name} has {len(lst)} entries, '
                f'expected {n_blocks} (matching conv_block_channels)'
            )

    # Validate GroupSum divisibility (post-D8: raise instead of returning
    # None; the caller did not check the return value so silent failures
    # would print "Done." after a misconfigured invocation).
    if fc_sizes[-1] % class_count != 0:
        raise ValueError(
            f'last fc_size ({fc_sizes[-1]}) must be divisible '
            f'by class_count ({class_count}) for GroupSum'
        )

    # --- Init conv blocks ---
    sample_images, _ = to_image_jax(next(iter(train_loader)))
    in_channels = sample_images.shape[-1]

    key = jax.random.PRNGKey(args.seed + 1_000)
    conv_params_list = []
    conv_wires_list = []
    cur_channels = in_channels

    for i in range(n_blocks):
        key, subkey = jax.random.split(key)
        ks = (block_kernels[i], block_kernels[i])
        cp, cw = init_conv_gate_layer(
            subkey,
            in_channels=cur_channels,
            out_channels=block_channels[i],
            kernel_size=ks,
            depth=block_depths[i],
            connection_type='random',
            logic_family=logic_family,
        )
        conv_params_list.append(cp)
        conv_wires_list.append(cw)
        cur_channels = block_channels[i]

    # --- Probe to get flatten dim and per-block pre-pool shapes ---
    key, probe_key = jax.random.split(key)
    x_probe = sample_images
    pre_pool_shapes = []  # (H, W, C) into pool, per block
    for i in range(n_blocks):
        pad = block_paddings[i]
        if pad > 0:
            x_probe = jnp.pad(x_probe, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        ks = (block_kernels[i], block_kernels[i])
        st = (block_strides[i], block_strides[i])
        x_probe = run_conv_gate_layer(
            conv_params_list[i], conv_wires_list[i], x_probe, False, probe_key,
            kernel_size=ks, stride=st,
            architecture=architecture, logic_family=logic_family,
        )
        pre_pool_shapes.append(x_probe.shape[1:])
        x_probe = probe_pool_fn(x_probe, kernel_size=(pool_size, pool_size),
                                stride=(pool_stride, pool_stride))

    flatten_dim = int(np.prod(x_probe.shape[1:]))
    layer_sizes = [flatten_dim] + list(fc_sizes)

    # --- Init FC stack ---
    key, fc_key = jax.random.split(key)
    fc_params, fc_wires = init_fc_stack(
        fc_key, layer_sizes, 'random', logic_family,
    )

    params = {'conv': conv_params_list, 'fc': fc_params}
    wires = {'conv': conv_wires_list, 'fc': fc_wires}

    conv_kw = dict(
        block_kernels=block_kernels,
        block_strides=block_strides,
        block_paddings=block_paddings,
        pool_size=pool_size,
        pool_stride=pool_stride,
        pool_type=pool_type,
        remat_conv=remat_conv,
        remat_fc=remat_fc,
        remat_policy=remat_policy,
    )

    # --- Print architecture summary ---
    dataset_label = getattr(args, 'dataset', 'mnist')
    tb = getattr(args, 'threshold_bits', None)
    print(f'\n[conv] deep conv DLGN ({logic_family}/{architecture}) on {dataset_label}')
    print(f'  input={tuple(sample_images.shape[1:])}  '
          f'threshold_bits={tb if tb is not None else "N/A"}')
    for i in range(n_blocks):
        print(f'  block {i}: ch={block_channels[i]}  k={block_kernels[i]}  '
              f'd={block_depths[i]}  pad={block_paddings[i]}  s={block_strides[i]}')
    print(f'  pool={pool_type} {pool_size}x{pool_size} stride={pool_stride}')
    remat_label = (f'{remat_arg}/{remat_policy_str}'
                   if remat_arg != 'off' else 'off')
    print(f'  remat={remat_label}')
    print(f'  flatten={flatten_dim}  fc={fc_sizes}')
    print(f'  sum_tau={cfg.sum_tau}  class_count={class_count}')

    # --- Optimizer ---
    tx = create_optimizer({
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'clip_value': args.clip_value,
    })
    opt_state = tx.init(params)

    # --- Eval before training ---
    test_images, test_y = to_image_jax(next(iter(test_loader)))
    eval_key = jax.random.PRNGKey(args.seed + 20_000)
    before = deep_conv_evaluate(params, wires, test_images, test_y, eval_key,
                                cfg, **conv_kw)
    print(f'  before: {fmt(before)}')

    # --- Build JIT step ---
    print('  compiling train step (this may take a while) ...', flush=True)
    t0 = time.time()
    train_step = make_deep_conv_step(tx, wires, cfg, conv_kw)
    # Warm up JIT with first batch
    train_iter = cycle_loader(train_loader)
    batch_images, batch_y = to_image_jax(next(train_iter))
    params, opt_state, key, loss = train_step(
        params, opt_state, batch_images, batch_y, key,
    )
    jax.block_until_ready(loss)
    compile_time = time.time() - t0
    print(f'  compiled in {compile_time:.1f}s  step_0_loss={float(loss):.4f}', flush=True)

    # --- Train ---
    eval_every = args.eval_every
    t_train = time.time()
    metrics = before

    if args.train_steps == 1:
        eval_key = jax.random.PRNGKey(args.seed + 30_001)
        metrics = deep_conv_evaluate(params, wires, test_images, test_y,
                                     eval_key, cfg, **conv_kw)
        print(f'  step {1:>5}/{args.train_steps}  '
              f'train_loss={float(loss):.4f}  {fmt(metrics)}  '
              f'[0s]', flush=True)

    for step in range(2, args.train_steps + 1):
        batch_images, batch_y = to_image_jax(next(train_iter))
        params, opt_state, key, loss = train_step(
            params, opt_state, batch_images, batch_y, key,
        )

        if step % eval_every == 0 or step == args.train_steps:
            jax.block_until_ready(loss)
            elapsed = time.time() - t_train
            eval_key = jax.random.PRNGKey(args.seed + 30_000 + step)
            metrics = deep_conv_evaluate(params, wires, test_images, test_y,
                                         eval_key, cfg, **conv_kw)
            print(f'  step {step:>5}/{args.train_steps}  '
                  f'train_loss={float(loss):.4f}  {fmt(metrics)}  '
                  f'[{elapsed:.0f}s]', flush=True)

    # --- Final summary ---
    total = time.time() - t0
    print(f'  total time: {total:.0f}s  (compile {compile_time:.1f}s)')
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PRESETS = {
    'smoke': {
        'conv_block_channels': [8],
        'conv_block_kernel_sizes': [3],
        'conv_block_depths': [2],
        'conv_block_paddings': [0],
        'conv_block_strides': [1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [40],
        'logic_family': 'full',
        'architecture': 'softmax',
        'train_steps': 5,
        'eval_every': 1,
    },

##############################################################
#NO FC DOUBLING
    'small': {
        'batch_size': 512,
        'train_steps': 5000,
        'learning_rate': 0.01,
        'weight_decay': 0.0,
        'clip_value': 1.0,
        'sum_tau': 6.5,
        'conv_block_channels': [16, 48, 144],
        'conv_block_kernel_sizes': [5, 3, 3],
        'conv_block_depths': [3, 3, 3],
        'conv_block_paddings': [0, 1, 1],
        'conv_block_strides': [1, 1, 1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [20480, 10240, 5120],
        'logic_family': 'full',
        'architecture': 'softmax',
        'eval_every': 100,
        'threshold_bits': 1,
    },

    'medium': {
        'batch_size': 256,
        'train_steps': 5000,
        'learning_rate': 0.01,
        'weight_decay': 0.0,
        'clip_value': 1.0,
        'sum_tau': 28,
        'conv_block_channels': [64, 192, 576],
        'conv_block_kernel_sizes': [5, 3, 3],
        'conv_block_depths': [3, 3, 3],
        'conv_block_paddings': [0, 1, 1],
        'conv_block_strides': [1, 1, 1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [81920, 40960, 20480],
        'logic_family': 'full',
        'architecture': 'softmax',
        'eval_every': 100,
        'threshold_bits': 1,
    },

    'large': {
        'batch_size': 128,
        'train_steps': 5000,
        'learning_rate': 0.01,
        'weight_decay': 0.0,
        'clip_value': 1.0,
        'sum_tau': 35,
        'conv_block_channels': [1024, 3072, 9216],
        'conv_block_kernel_sizes': [5, 3, 3],
        'conv_block_depths': [3, 3, 3],
        'conv_block_paddings': [0, 1, 1],
        'conv_block_strides': [1, 1, 1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [1310720, 655360, 327680],
        'logic_family': 'full',
        'architecture': 'softmax',
        'eval_every': 100,
        'threshold_bits': 1,
    },

##############################################################
#NO FC DOUBLING for base and large
    'small_cifar': {
        'dataset': 'cifar10',
        'batch_size': 128,
        'train_steps': 5000,
        'learning_rate': 0.02,
        'weight_decay': 0.002,
        'clip_value': 1.0,
        'sum_tau': 20,
        'conv_block_channels': [32, 128, 512, 1024],
        'conv_block_kernel_sizes': [3, 3, 3, 3],
        'conv_block_depths': [3, 3, 3, 3],
        'conv_block_paddings': [1, 1, 1, 1],
        'conv_block_strides': [1, 1, 1, 1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [40960, 20480, 10240],
        'logic_family': 'full',
        'architecture': 'softmax',
        'eval_every': 100,
        'threshold_bits': 2,
    },

    'medium_cifar': {
        'dataset': 'cifar10',
        'batch_size': 128,
        'train_steps': 5000,
        'learning_rate': 0.02,
        'weight_decay': 0.002,
        'clip_value': 1.0,
        'sum_tau': 40,
        'conv_block_channels': [256, 1024, 4096, 8192],
        'conv_block_kernel_sizes': [3, 3, 3, 3],
        'conv_block_depths': [3, 3, 3, 3],
        'conv_block_paddings': [1, 1, 1, 1],
        'conv_block_strides': [1, 1, 1, 1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [327680, 163840, 81920],
        'logic_family': 'full',
        'architecture': 'softmax',
        'eval_every': 100,
        'threshold_bits': 2,
    },

    'base_cifar': {
        'dataset': 'cifar10',
        'batch_size': 128,
        'train_steps': 5000,
        'learning_rate': 0.02,
        'weight_decay': 0.002,
        'clip_value': 1.0,
        'sum_tau': 280,
        'conv_block_channels': [512, 2048, 8192, 16384],
        'conv_block_kernel_sizes': [3, 3, 3, 3],
        'conv_block_depths': [3, 3, 3, 3],
        'conv_block_paddings': [1, 1, 1, 1],
        'conv_block_strides': [1, 1, 1, 1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [1310720, 655360, 327680], #Div 2 = [[655360, 327680, 163840]
        'logic_family': 'full',
        'architecture': 'softmax',
        'eval_every': 100,
        'threshold_bits': 5,
    },

    'large_cifar': {
        'dataset': 'cifar10',
        'batch_size': 128,
        'train_steps': 5000,
        'learning_rate': 0.02,
        'weight_decay': 0.002,
        'clip_value': 1.0,
        'sum_tau': 340,
        'conv_block_channels': [1024, 4096, 16384, 32768],
        'conv_block_kernel_sizes': [3, 3, 3, 3],
        'conv_block_depths': [3, 3, 3, 3],
        'conv_block_paddings': [1, 1, 1, 1],
        'conv_block_strides': [1, 1, 1, 1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [2621440, 1310720, 655360], #Div 2 = [1310720, 655360, 327690]
        'logic_family': 'full',
        'architecture': 'softmax',
        'eval_every': 100,
        'threshold_bits': 5,
    },

    'giant_cifar': {
        'dataset': 'cifar10',
        'batch_size': 128,
        'train_steps': 5000,
        'learning_rate': 0.02,
        'weight_decay': 0.001,
        'clip_value': 1.0,
        'sum_tau': 450,
        'conv_block_channels': [2560, 10240, 40960, 81920],
        'conv_block_kernel_sizes': [3, 3, 3, 3],
        'conv_block_depths': [3, 3, 3, 3],
        'conv_block_paddings': [1, 1, 1, 1],
        'conv_block_strides': [1, 1, 1, 1],
        'pool_size': 2,
        'pool_stride': 2,
        'fc_sizes': [3276800, 1638400, 819200],
        'logic_family': 'full',
        'architecture': 'softmax',
        'eval_every': 100,
        'threshold_bits': 5,
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train DLGN models on MNIST')

    p.add_argument('--preset', default='smoke', choices=PRESETS.keys())
    p.add_argument('--dataset', default='mnist',
                   choices=['mnist', 'mnist20x20', 'mnist_bin', 'mnist20x20_bin',
                            'cifar10', 'cifar-10-3-thresholds', 'cifar-10-31-thresholds'])
    p.add_argument('--model', default='both', choices=['regular', 'conv', 'both'])
    p.add_argument('--storage-root', default=str(ROOT / 'dataset_storage'),
                   dest='storage_root')
    p.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    p.add_argument('--train-steps', type=int, default=5, dest='train_steps')
    p.add_argument('--eval-every', type=int, default=1, dest='eval_every')
    p.add_argument('--seed', type=int, default=0)

    g = p.add_argument_group('regular model')
    g.add_argument('--regular-neurons', type=int, default=1200, dest='regular_neurons')
    g.add_argument('--regular-layers', type=int, default=5, dest='regular_layers')
    g.add_argument('--regular-logic-family', default='full', choices=['full', 'light'],
                   dest='regular_logic_family')
    g.add_argument('--regular-architecture', default='softmax',
                   dest='regular_architecture')

    g = p.add_argument_group('conv model')
    g.add_argument('--conv-block-channels', type=int, nargs='+', default=[8],
                   dest='conv_block_channels')
    g.add_argument('--conv-block-kernel-sizes', type=int, nargs='+', default=[3],
                   dest='conv_block_kernel_sizes')
    g.add_argument('--conv-block-depths', type=int, nargs='+', default=[2],
                   dest='conv_block_depths')
    g.add_argument('--conv-block-paddings', type=int, nargs='+', default=[0],
                   dest='conv_block_paddings')
    g.add_argument('--conv-block-strides', type=int, nargs='+', default=[1],
                   dest='conv_block_strides')
    g.add_argument('--pool-size', type=int, default=2, dest='pool_size')
    g.add_argument('--pool-stride', type=int, default=2, dest='pool_stride')
    g.add_argument('--pool-type', default='or', choices=['or', 'max'],
                   dest='pool_type',
                   help='Spatial pool after each conv block (default: or)')
    # D25: opt-in rematerialization. Default 'off' preserves baseline.
    g.add_argument('--remat', default='off',
                   choices=['off', 'conv', 'fc', 'both'],
                   dest='remat',
                   help='Activation rematerialization scope (default: off). '
                        'conv = wrap each conv block; fc = wrap the FC head; '
                        'both = wrap everything.')
    g.add_argument('--remat-policy', default='dots',
                   choices=['dots', 'nothing'],
                   dest='remat_policy',
                   help='jax.checkpoint policy: dots saves matmul results; '
                        'nothing recomputes everything (default: dots)')
    g.add_argument('--fc-sizes', type=int, nargs='+', default=[40],
                   dest='fc_sizes')
    g.add_argument('--logic-family', default='full', choices=['full', 'light'],
                   dest='logic_family')
    g.add_argument('--architecture', default='softmax', dest='architecture')

    g = p.add_argument_group('optimizer / decoder')
    g.add_argument('--learning-rate', type=float, default=0.01, dest='learning_rate')
    g.add_argument('--weight-decay', type=float, default=1e-4, dest='weight_decay')
    g.add_argument('--clip-value', type=float, default=1.0, dest='clip_value')
    g.add_argument('--sum-tau', type=float, default=1.0, dest='sum_tau')
    g.add_argument('--gumb-tau', type=float, default=1.0, dest='gumb_tau')
    g.add_argument('--dirichlet-concentration', type=float, default=1.0,
                   dest='dirichlet_concentration')
    g.add_argument('--threshold-bits', type=int, default=None,
                   dest='threshold_bits',
                   help='Number of binary thresholds per input channel '
                        '(CIFAR: 3ch × N = 3N input channels)')
    # D24: JAX persistent compilation cache. Default on; --no-jit-cache
    # disables for benchmarking the cold-compile path.
    g.add_argument('--no-jit-cache', action='store_true', dest='no_jit_cache',
                   help='Disable the JAX persistent compilation cache')
    g.add_argument('--jit-cache-dir', default='.jax_cache', dest='jit_cache_dir',
                   help='JAX compilation cache directory (default: ./.jax_cache)')
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
    from dlgn.data.loaders import load_dataset, num_classes_of_dataset
    from dlgn.utils import seed_all, setup_jit_cache

    if argv is None:
        argv = sys.argv[1:]

    args = build_parser().parse_args(argv)
    apply_preset(args, argv)

    storage_root = Path(args.storage_root).resolve()
    seed_all(args.seed, seed_torch=True)
    # D24: enable JAX persistent compile cache unless --no-jit-cache.
    setup_jit_cache(args.jit_cache_dir, enabled=not args.no_jit_cache)

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
    if getattr(args, 'threshold_bits', None) is not None:
        data_config['threshold_bits'] = args.threshold_bits

    train_loader, _, test_loader = load_dataset(data_config)
    class_count = num_classes_of_dataset(args.dataset)

    # Validate regular path divisibility
    if args.model in ('regular', 'both'):
        if args.regular_neurons % class_count != 0:
            print(f'Error: --regular-neurons ({args.regular_neurons}) must be '
                  f'divisible by class_count ({class_count})')
            return 1

    tb = getattr(args, 'threshold_bits', None)
    print(f'Dataset: {args.dataset}  classes={class_count}  '
          f'batch={args.batch_size}  steps={args.train_steps}  '
          f'preset={args.preset}'
          + (f'  threshold_bits={tb}' if tb is not None else ''))

    if args.model in ('regular', 'both'):
        run_regular_path(args, train_loader, test_loader, class_count)

    if args.model in ('conv', 'both'):
        run_deep_conv_path(args, train_loader, test_loader, class_count)

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
