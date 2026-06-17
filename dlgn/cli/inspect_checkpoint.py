# New file — CLI entry point for inspecting a saved checkpoint without notebooks.
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Inspect a DLGN checkpoint file'
    )
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    return p


def main(argv=None) -> None:
    from dlgn.training.checkpoints import load_checkpoint

    parser = build_parser()
    args = parser.parse_args(argv)

    print(f'Checkpoint: {args.checkpoint}')
    payload = load_checkpoint(args.checkpoint)

    print(f'\n── Step ──')
    print(f'  step: {payload.get("step")}')

    print(f'\n── Dataset info ──')
    dataset_info = payload.get('dataset_info') or {}
    for k, v in dataset_info.items():
        print(f'  {k}: {v}')

    print(f'\n── Core config ──')
    config = payload.get('config') or {}
    core_keys = [
        'dataset', 'logic_family', 'architecture', 'num_neurons', 'num_layers',
        'num_steps', 'eval_every', 'batch_size', 'learning_rate', 'seed', 'connections',
    ]
    for k in core_keys:
        if k in config:
            print(f'  {k}: {config[k]}')

    print(f'\n── History ──')
    history = payload.get('history') or {}
    for k, v in history.items():
        if isinstance(v, list):
            length = len(v)
            last = v[-1] if v else None
            print(f'  {k}: len={length}  last={last}')
        else:
            print(f'  {k}: {v}')

    has_valid = bool(history.get('valid_hard_acc') and any(
        x is not None for x in history['valid_hard_acc']
    ))
    has_test = bool(history.get('test_hard_acc') and any(
        x is not None for x in history['test_hard_acc']
    ))
    print(f'\n  has validation history: {has_valid}')
    print(f'  has test history:       {has_test}')

    print(f'\n── Params ──')
    state = payload.get('state')
    if state is not None:
        for i, layer in enumerate(state.params):
            print(f'  layer {i}: shape={layer.shape}')

    print(f'\n── Extra ──')
    extra = payload.get('extra') or {}
    if extra:
        for k, v in extra.items():
            print(f'  {k}: {v}')
    else:
        print('  (empty)')


if __name__ == '__main__':
    main()
