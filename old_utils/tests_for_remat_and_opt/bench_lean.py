"""Benchmark harness for DLGN_Only_Lean.

See ../dlgn_benchmark_protocol.md for the full protocol. This harness implements
the smoke-scale baselines that Phase 3 captures:

    smoke_regular  : regular DLGN, MNIST, full/softmax, 64 neurons, 4 layers,
                     200 steps. 3 seeds.
    smoke_conv     : conv path, MNIST, smoke preset, 5 steps. 3 seeds.

Larger configs (MNIST small @ 5000 steps; small_cifar @ 5000 steps + ~170MB
download) are listed but require --include-slow because they take many
minutes.test_layers

Output:
    bench_results/<commit>/<bench_name>_seed<N>.jsonl
    bench_results/<commit>/summary.jsonl

Usage:
    cd DLGN_Only_Lean
    python scripts/bench_lean.py --bench smoke_regular
    python scripts/bench_lean.py --bench smoke_conv
    python scripts/bench_lean.py --bench all_smoke
    python scripts/bench_lean.py --bench small --include-slow
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax


PROTOCOL_VERSION = '1'


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
            text=True,
        ).strip()
        return out or 'unknown'
    except subprocess.CalledProcessError:
        return 'nogit'


def _count_params(params) -> int:
    """Recursive param-element-count over arbitrarily nested lists of arrays."""
    if isinstance(params, dict):
        return sum(_count_params(v) for v in params.values())
    if isinstance(params, (list, tuple)):
        return sum(_count_params(v) for v in params)
    if hasattr(params, 'shape'):
        return int(np.prod(params.shape))
    return 0


# ── Smoke #1: Regular DLGN MNIST ────────────────────────────────────────────


def bench_smoke_regular(seed: int, steady_steps: int = 200) -> dict:
    """Regular DLGN, full/softmax on MNIST. 64 neurons, 4 layers."""
    from dlgn.data.loaders import load_dataset, num_classes_of_dataset, input_dim_of_dataset
    from dlgn.models.initialization import init_logic_gate_network
    from dlgn.training.optim import create_optimizer
    from dlgn.training.state import TrainState
    from dlgn.training.steps import make_train_step, eval_batch
    from dlgn.utils import batch_to_jax, seed_all

    config = {
        'dataset': 'mnist',
        'seed': seed,
        'batch_size': 128,
        'valid_set_size': 0.0,
        'num_workers': 0,
        'logic_family': 'full',
        'architecture': 'softmax',
        'gumb_tau': 1.0,
        'dirichlet_concentration': 1.0,
        'sum_tau': 1.0,
        'learning_rate': 0.01,
        'weight_decay': 1e-4,
        'clip_value': 1.0,
        'data_roots': {
            'mnist': str(ROOT / 'dataset_storage' / 'mnist'),
            'uci': str(ROOT / 'dataset_storage' / 'uci'),
            'cifar': str(ROOT / 'dataset_storage' / 'cifar'),
        },
    }

    seed_all(seed, seed_torch=True)
    train_loader, _, test_loader = load_dataset(config)
    input_dim = input_dim_of_dataset('mnist')
    class_count = num_classes_of_dataset('mnist')
    # GroupSum requires num_neurons % class_count == 0. MNIST class_count=10,
    # so pick a small multiple of 10. 100 keeps the smoke fast (~200 steps
    # at 128 batch) while staying near the legacy 64-neuron default.
    num_neurons = 100
    num_layers = 4

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params, wires = init_logic_gate_network(
        input_dim=input_dim, num_neurons=num_neurons, num_layers=num_layers,
        connections='random', key=init_key, logic_family='full',
    )
    tx = create_optimizer(config)
    state = TrainState(params=params, opt_state=tx.init(params), key=key)
    train_step = make_train_step(tx)

    # Compile timer: one batch
    train_iter = iter(train_loader)
    batch_x, batch_y = batch_to_jax(next(train_iter))
    t0 = time.perf_counter()
    state, loss, aux = train_step(
        state, batch_x, batch_y, wires,
        'softmax', 1.0, 1.0, class_count, 1.0, 'full',
    )
    jax.block_until_ready(loss)
    compile_s = time.perf_counter() - t0

    # Warmup
    for _ in range(5):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch_x, batch_y = batch_to_jax(batch)
        state, loss, aux = train_step(
            state, batch_x, batch_y, wires,
            'softmax', 1.0, 1.0, class_count, 1.0, 'full',
        )
    jax.block_until_ready(loss)

    # Steady-state timer
    t0 = time.perf_counter()
    n_real_steps = 0
    for _ in range(steady_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch_x, batch_y = batch_to_jax(batch)
        state, loss, aux = train_step(
            state, batch_x, batch_y, wires,
            'softmax', 1.0, 1.0, class_count, 1.0, 'full',
        )
        n_real_steps += 1
    jax.block_until_ready(loss)
    steady_s = time.perf_counter() - t0
    steps_per_sec = n_real_steps / steady_s

    # One eval batch on the test set
    test_iter = iter(test_loader)
    test_batch_x, test_batch_y = batch_to_jax(next(test_iter))
    metrics = eval_batch(
        state.params, wires, test_batch_x, test_batch_y, jax.random.PRNGKey(seed + 50_000),
        'softmax', 1.0, 1.0, class_count, 1.0, 'full',
    )
    count = float(metrics['count'])
    soft_acc = float(metrics['soft_correct']) / count
    hard_acc = float(metrics['hard_correct']) / count
    soft_loss = float(metrics['soft_loss_sum']) / count
    hard_loss = float(metrics['hard_loss_sum']) / count

    return {
        'bench': 'smoke_regular',
        'seed': seed,
        'dataset': 'mnist',
        'effective_input_dim': input_dim,
        'image_shape_NHWC': None,
        'threshold_bits': None,
        'logic_family': 'full',
        'architecture': 'softmax',
        'wiring': 'random',
        'pool_type': None,
        'sum_tau': 1.0,
        'gumb_tau': 1.0,
        'clip_value': 1.0,
        'learning_rate': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 128,
        'num_steps': n_real_steps,
        'compile_s': compile_s,
        'steady_steps_per_sec': steps_per_sec,
        'train_soft_loss_last': float(loss),
        'train_hard_loss_last': float(aux['hard']),
        'soft_loss': soft_loss,
        'hard_loss': hard_loss,
        'soft_acc': soft_acc,
        'hard_acc': hard_acc,
        'soft_hard_gap': soft_acc - hard_acc,
        'parameter_count': _count_params(state.params),
        'wall_time_s': compile_s + steady_s,
        'commit_hash': _git_commit(),
        'protocol_version': PROTOCOL_VERSION,
    }


# ── Smoke #2: Conv smoke preset (delegates to primary script's main) ────────


def bench_smoke_conv(seed: int) -> dict:
    """Conv smoke preset, MNIST. 5 training steps. Times the JIT compile and steps."""
    from scripts.test_layers import main as primary_main

    # Capture the script's prints so we don't pollute the bench output.
    import io
    import contextlib
    buf = io.StringIO()

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        rc = primary_main([
            '--preset', 'smoke',
            '--model', 'conv',
            '--seed', str(seed),
            '--train-steps', '5',
            '--eval-every', '5',
            '--storage-root', str(ROOT / 'dataset_storage'),
        ])
    wall_s = time.perf_counter() - t0
    log = buf.getvalue()

    # Parse the last metric line for soft/hard acc.
    soft_acc = hard_acc = soft_loss = hard_loss = None
    compile_s = None
    for line in log.splitlines():
        if 'compiled in' in line:
            try:
                compile_s = float(line.split('compiled in')[1].split('s')[0].strip())
            except ValueError:
                pass
        if 'soft_acc=' in line and 'hard_acc=' in line:
            for tok in line.split():
                if tok.startswith('soft_acc='):
                    soft_acc = float(tok.split('=')[1])
                elif tok.startswith('hard_acc='):
                    hard_acc = float(tok.split('=')[1])
                elif tok.startswith('soft_loss='):
                    soft_loss = float(tok.split('=')[1])
                elif tok.startswith('hard_loss='):
                    hard_loss = float(tok.split('=')[1])

    return {
        'bench': 'smoke_conv',
        'seed': seed,
        'dataset': 'mnist',
        'effective_input_dim': None,  # conv path doesn't flatten input
        'image_shape_NHWC': '(28,28,1)',
        'threshold_bits': None,
        'logic_family': 'full',
        'architecture': 'softmax',
        'wiring': 'random',
        'pool_type': 'or',
        'sum_tau': 1.0,
        'gumb_tau': 1.0,
        'clip_value': 1.0,
        'learning_rate': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'num_steps': 5,
        'compile_s': compile_s,
        'steady_steps_per_sec': None,  # 5 steps too small for a clean number
        'soft_loss': soft_loss,
        'hard_loss': hard_loss,
        'soft_acc': soft_acc,
        'hard_acc': hard_acc,
        'soft_hard_gap': (soft_acc - hard_acc) if (soft_acc is not None and hard_acc is not None) else None,
        'wall_time_s': wall_s,
        'commit_hash': _git_commit(),
        'protocol_version': PROTOCOL_VERSION,
        '_rc': rc,
    }


# ── Driver ───────────────────────────────────────────────────────────────────


SMOKE_FNS = {
    'smoke_regular': bench_smoke_regular,
    'smoke_conv': bench_smoke_conv,
}

SLOW_BENCHES = {'small', 'small_cifar'}  # listed but require --include-slow


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')


def summarize(records: list[dict]) -> dict:
    """Median across seeds for the numeric fields."""
    numeric_keys = [k for k, v in records[0].items()
                    if isinstance(v, (int, float)) and v is not None
                    and k != 'seed']
    out = {
        'bench': records[0]['bench'],
        'dataset': records[0]['dataset'],
        'n_seeds': len(records),
        'commit_hash': records[0]['commit_hash'],
        'protocol_version': PROTOCOL_VERSION,
    }
    for k in numeric_keys:
        vals = [r[k] for r in records if r.get(k) is not None]
        if vals:
            out[f'{k}_median'] = float(np.median(vals))
            out[f'{k}_min'] = float(np.min(vals))
            out[f'{k}_max'] = float(np.max(vals))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description='Bench DLGN_Only_Lean smoke configs')
    p.add_argument('--bench', required=True,
                   choices=list(SMOKE_FNS.keys()) + ['all_smoke', 'small', 'small_cifar'])
    p.add_argument('--seeds', default='0,1,2', help='Comma-separated seeds (default 0,1,2)')
    p.add_argument('--steady-steps', type=int, default=200,
                   help='Steady-state step count for smoke_regular (default 200)')
    p.add_argument('--include-slow', action='store_true',
                   help='Allow slow configs (small, small_cifar)')
    p.add_argument('--output-dir', default=str(ROOT / 'bench_results'),
                   help='Output root for JSONL records')
    args = p.parse_args()

    if args.bench in SLOW_BENCHES and not args.include_slow:
        p.error(f'Bench {args.bench!r} is slow; pass --include-slow to confirm.')

    if args.bench in SLOW_BENCHES:
        p.error(f'Bench {args.bench!r} is not yet implemented in this harness; '
                'use the primary script directly with the preset name.')

    seeds = [int(s) for s in args.seeds.split(',')]
    benches = list(SMOKE_FNS.keys()) if args.bench == 'all_smoke' else [args.bench]

    commit = _git_commit()
    out_root = Path(args.output_dir) / commit

    for bench in benches:
        print(f'\n=== {bench} (seeds={seeds}) ===')
        fn = SMOKE_FNS[bench]
        records = []
        for seed in seeds:
            print(f'  seed={seed} ...', flush=True)
            if bench == 'smoke_regular':
                rec = fn(seed, steady_steps=args.steady_steps)
            else:
                rec = fn(seed)
            records.append(rec)
            print(f'    soft_acc={rec.get("soft_acc")} '
                  f'hard_acc={rec.get("hard_acc")} '
                  f'compile_s={rec.get("compile_s")} '
                  f'steady_steps_per_sec={rec.get("steady_steps_per_sec")}')

        write_jsonl(records, out_root / f'{bench}_per_seed.jsonl')
        summary = summarize(records)
        write_jsonl([summary], out_root / f'{bench}_summary.jsonl')

        print(f'  summary: '
              f'soft_acc_median={summary.get("soft_acc_median")} '
              f'hard_acc_median={summary.get("hard_acc_median")} '
              f'compile_s_median={summary.get("compile_s_median")} '
              f'steady_steps_per_sec_median={summary.get("steady_steps_per_sec_median")}')

    print(f'\nResults under: {out_root.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
