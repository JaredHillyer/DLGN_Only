# New file — CLI entry point for resuming from a checkpoint.
from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Resume a DLGN training run from a checkpoint'
    )
    p.add_argument('--checkpoint', required=True,
                   help='Path to checkpoint file (latest.pkl or final.pkl)')

    # Config override options (common useful overrides)
    p.add_argument('--num-steps', type=int, default=None, dest='num_steps',
                   help='Override num_steps (extend training beyond saved value)')
    p.add_argument('--eval-every', type=int, default=None, dest='eval_every')
    p.add_argument('--learning-rate', type=float, default=None, dest='learning_rate')
    p.add_argument('--output-dir', default='outputs', dest='output_dir')
    p.add_argument('--run-name', default=None, dest='run_name',
                   help='Run name for the resumed run directory')
    p.add_argument('--save-final', action=argparse.BooleanOptionalAction, default=True,
                   dest='save_final')

    # D24: JAX persistent compilation cache.
    p.add_argument('--no-jit-cache', action='store_true', dest='no_jit_cache',
                   help='Disable the JAX persistent compilation cache')
    p.add_argument('--jit-cache-dir', default='.jax_cache', dest='jit_cache_dir',
                   help='JAX compilation cache directory (default: ./.jax_cache)')

    return p


def main(argv=None) -> None:
    from dlgn.training.checkpoints import resume_training_from_checkpoint, save_checkpoint
    from dlgn.utils import make_run_name, make_run_dir, setup_jit_cache
    from dlgn.config import save_config_snapshot

    parser = build_parser()
    args = parser.parse_args(argv)
    # D24: enable JAX persistent compile cache unless --no-jit-cache.
    setup_jit_cache(args.jit_cache_dir, enabled=not args.no_jit_cache)

    # Build override dict from non-None CLI values
    overrides = {}
    if args.num_steps is not None:
        overrides['num_steps'] = args.num_steps
    if args.eval_every is not None:
        overrides['eval_every'] = args.eval_every
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate

    print(f'Resuming from: {args.checkpoint}')
    if overrides:
        print(f'Config overrides: {overrides}')

    state, wires, history, dataset_info, validation_loader, test_loader = \
        resume_training_from_checkpoint(args.checkpoint, config_override=overrides or None)

    # Save resumed run outputs
    from dlgn.training.checkpoints import load_checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    merged_config = dict(checkpoint['config'])
    merged_config.update(overrides)

    run_name = args.run_name or ('resume_' + make_run_name(merged_config))
    run_dir = make_run_dir(args.output_dir, run_name)
    save_config_snapshot(merged_config, run_dir)

    final_step = int(history['steps'][-1]) if history['steps'] else 0

    if args.save_final:
        save_checkpoint(
            run_dir / 'checkpoints' / 'final.pkl',
            state, wires, merged_config, history, dataset_info, step=final_step,
        )
        save_checkpoint(
            run_dir / 'checkpoints' / 'latest.pkl',
            state, wires, merged_config, history, dataset_info, step=final_step,
        )

    import json as _json
    with open(run_dir / 'history.json', 'w') as f:
        _json.dump(history, f, indent=2, default=lambda x: None if x is None else float(x))

    print(f'\nResume complete. Run dir: {run_dir.resolve()}')


if __name__ == '__main__':
    main()
