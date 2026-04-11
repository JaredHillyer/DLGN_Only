# New file — CLI entry point for training.
# Calls into package modules. Does not duplicate training logic.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Train a Differentiable Logic Gate Network'
    )

    # ── Required model args ──────────────────────────────────────────────────
    p.add_argument('--dataset', required=True,
                   help='Dataset name (e.g. toy_xor, adult, mnist)')
    p.add_argument('--logic-family', required=True, choices=['full', 'light'],
                   dest='logic_family')
    p.add_argument('--architecture', required=True,
                   help='Decoder architecture (softmax, gumbel, dirichlet, light_sigmoid, ...)')
    p.add_argument('--num-neurons', required=True, type=int, dest='num_neurons')
    p.add_argument('--num-layers', required=True, type=int, dest='num_layers')

    # ── Training args ────────────────────────────────────────────────────────
    p.add_argument('--num-steps', type=int, default=1000, dest='num_steps')
    p.add_argument('--eval-every', type=int, default=100, dest='eval_every')
    p.add_argument('--batch-size', type=int, default=128, dest='batch_size')
    p.add_argument('--learning-rate', type=float, default=0.01, dest='learning_rate')
    p.add_argument('--weight-decay', type=float, default=1e-4, dest='weight_decay')
    p.add_argument('--clip-value', type=float, default=1.0, dest='clip_value')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--connections', default='random', choices=['unique', 'random'])

    # ── Decoder/head args ────────────────────────────────────────────────────
    p.add_argument('--sum-tau', type=float, default=1.0, dest='sum_tau')
    p.add_argument('--gumb-tau', type=float, default=1.0, dest='gumb_tau')
    p.add_argument('--dirichlet-concentration', type=float, default=1.0,
                   dest='dirichlet_concentration')

    # ── I/O args ─────────────────────────────────────────────────────────────
    p.add_argument('--output-dir', default='outputs', dest='output_dir')
    p.add_argument('--run-name', default=None, dest='run_name',
                   help='Run name override (auto-generated if omitted)')
    p.add_argument('--save-every', type=int, default=None, dest='save_every',
                   help='Save latest.pkl every N steps (default: at eval events)')
    p.add_argument('--save-final', action=argparse.BooleanOptionalAction, default=True,
                   dest='save_final')

    # ── Data root args ───────────────────────────────────────────────────────
    p.add_argument('--data-root-uci', default='./data-uci', dest='data_root_uci')
    p.add_argument('--data-root-mnist', default='./data-mnist', dest='data_root_mnist')
    p.add_argument('--data-root-cifar', default='./data-cifar', dest='data_root_cifar')
    p.add_argument('--data-root-block', default='./blocks_dataset', dest='data_root_block')

    # ── Optional args ────────────────────────────────────────────────────────
    p.add_argument('--valid-set-size', type=float, default=0.1, dest='valid_set_size')
    p.add_argument('--num-workers', type=int, default=0, dest='num_workers')
    p.add_argument('--toy-num-bits', type=int, default=8, dest='toy_num_bits')
    p.add_argument('--config', default=None,
                   help='Path to a JSON preset config (CLI args override preset values)')
    p.add_argument('--notes', default='', help='Free-text notes stored in config.json')

    return p


def args_to_config(args: argparse.Namespace) -> dict:
    """Convert parsed CLI args to a notebook-compatible config dict."""
    config = {
        'dataset': args.dataset,
        'logic_family': args.logic_family,
        'architecture': args.architecture,
        'num_neurons': args.num_neurons,
        'num_layers': args.num_layers,
        'num_steps': args.num_steps,
        'eval_every': args.eval_every,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'clip_value': args.clip_value,
        'seed': args.seed,
        'connections': args.connections,
        'sum_tau': args.sum_tau,
        'gumb_tau': args.gumb_tau,
        'dirichlet_concentration': args.dirichlet_concentration,
        'valid_set_size': args.valid_set_size,
        'num_workers': args.num_workers,
        'toy_num_bits': args.toy_num_bits,
        'data_roots': {
            'uci': args.data_root_uci,
            'mnist': args.data_root_mnist,
            'cifar': args.data_root_cifar,
            'block': args.data_root_block,
        },
    }
    if args.notes:
        config['_notes'] = args.notes
    return config


def main(argv=None) -> None:
    from dlgn.config import validate_logic_config, save_config_snapshot
    from dlgn.training.loop import train_model
    from dlgn.training.checkpoints import save_checkpoint
    from dlgn.utils.paths import make_run_name, make_run_dir
    from dlgn.utils.seeding import seed_all

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.save_every is not None:
        parser.error(
            '--save-every is not implemented yet; the current CLI only writes '
            'checkpoints after training completes.'
        )

    # Load preset config if provided, then override with CLI args
    config: dict = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # CLI values override preset values
    cli_config = args_to_config(args)
    config.update({k: v for k, v in cli_config.items() if v is not None})

    # Validate before doing any work
    validate_logic_config(config)

    needs_torch = config['dataset'] not in ('toy_xor', 'toy_parity')
    seed_all(config['seed'], seed_torch=needs_torch)

    run_name = args.run_name or make_run_name(config)
    run_dir = make_run_dir(args.output_dir, run_name)
    print(f'Run directory: {run_dir.resolve()}')

    save_config_snapshot(config, run_dir)

    # ── Training loop ────────────────────────────────────────────────────────
    metrics_path = run_dir / 'metrics.jsonl'

    # Wrap train_model to intercept checkpoint saving at eval cadence
    # by running in a step-aware loop via the loop module directly.
    # For simplicity in first implementation: run full train_model, then save.
    state, wires, history, dataset_info, validation_loader, test_loader = train_model(config)

    # ── Post-training checkpoints ────────────────────────────────────────────
    final_step = int(history['steps'][-1]) if history['steps'] else config['num_steps'] - 1

    if args.save_final:
        save_checkpoint(
            run_dir / 'checkpoints' / 'final.pkl',
            state, wires, config, history, dataset_info, step=final_step,
        )
        # Also write latest
        save_checkpoint(
            run_dir / 'checkpoints' / 'latest.pkl',
            state, wires, config, history, dataset_info, step=final_step,
        )

    # ── Save history and metrics ─────────────────────────────────────────────
    import json as _json
    with open(run_dir / 'history.json', 'w') as f:
        _json.dump(history, f, indent=2, default=lambda x: None if x is None else float(x))

    # Write metrics.jsonl: one line per eval event
    with open(metrics_path, 'w') as f:
        eval_idx = 0
        for step_idx, step in enumerate(history['steps']):
            is_eval = (step % config['eval_every'] == 0) or (step == final_step)
            if not is_eval:
                continue

            record = {
                'step': step,
                'train_soft_loss': history['train_soft_loss'][step_idx],
                'train_hard_loss': history['train_hard_loss'][step_idx],
                'valid_hard_acc': (
                    history['valid_hard_acc'][eval_idx]
                    if eval_idx < len(history['valid_hard_acc'])
                    else None
                ),
                'test_hard_acc': (
                    history['test_hard_acc'][eval_idx]
                    if eval_idx < len(history['test_hard_acc'])
                    else None
                ),
            }
            f.write(_json.dumps(record) + '\n')
            eval_idx += 1

    # ── Save summary ─────────────────────────────────────────────────────────
    summary = {
        'run_name': run_name,
        'run_dir': str(run_dir.resolve()),
        'final_step': final_step,
        'dataset': config['dataset'],
        'logic_family': config['logic_family'],
        'architecture': config['architecture'],
        'final_test_hard_acc': history['test_hard_acc'][-1] if history['test_hard_acc'] else None,
        'final_valid_hard_acc': history['valid_hard_acc'][-1] if history['valid_hard_acc'] else None,
    }
    with open(run_dir / 'summary.json', 'w') as f:
        _json.dump(summary, f, indent=2, default=lambda x: None if x is None else float(x))

    print(f'\nRun complete. Summary saved to {run_dir / "summary.json"}')


if __name__ == '__main__':
    main()
