"""Download all supported non-block datasets into a repo-level storage folder.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --storage-root dataset_storage
    python scripts/download_datasets.py --dataset adult --dataset mnist

This script intentionally stores data outside the `dlgn/` package tree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


DEFAULT_DATASETS = [
    'adult',
    'breast_cancer',
    'monk1',
    'monk2',
    'monk3',
    'mnist',
    'mnist20x20',
    'mnist_bin',
    'mnist20x20_bin',
    'cifar-10-3-thresholds',
    'cifar-10-31-thresholds',
]

SKIPPED_DATASETS = {
    'toy_xor': 'synthetic in-memory dataset; no files to download',
    'toy_parity': 'synthetic in-memory dataset; no files to download',
    'block': 'dataset source is missing in this workspace',
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Download DLGN datasets into dataset_storage/'
    )
    parser.add_argument(
        '--storage-root',
        default=str(ROOT / 'dataset_storage'),
        help='Directory where dataset files should be stored',
    )
    parser.add_argument(
        '--dataset',
        action='append',
        dest='datasets',
        default=None,
        help='Specific dataset to download; may be passed multiple times',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='Print supported download targets and exit',
    )
    return parser


def make_config(dataset: str, storage_root: Path) -> dict:
    return {
        'dataset': dataset,
        'seed': 0,
        'batch_size': 32,
        'valid_set_size': 0.0,
        'num_workers': 0,
        'data_roots': {
            'uci': str(storage_root / 'uci'),
            'mnist': str(storage_root / 'mnist'),
            'cifar': str(storage_root / 'cifar'),
            'block': str(storage_root / 'block'),
        },
    }


def main(argv=None) -> int:
    from dlgn.data.loaders import load_dataset
    from dlgn.data.registry import SUPPORTED_DATASETS

    parser = build_parser()
    args = parser.parse_args(argv)

    selected = args.datasets or list(DEFAULT_DATASETS)
    supported_downloads = set(DEFAULT_DATASETS)

    if args.list:
        print('Downloadable datasets:')
        for name in DEFAULT_DATASETS:
            print(f'  - {name}')
        print('\nSkipped by design:')
        for name, reason in SKIPPED_DATASETS.items():
            print(f'  - {name}: {reason}')
        extra = sorted(set(SUPPORTED_DATASETS) - supported_downloads - set(SKIPPED_DATASETS))
        if extra:
            print('\nRegistry datasets not in the default downloader:')
            for name in extra:
                print(f'  - {name}')
        return 0

    invalid = [name for name in selected if name not in SUPPORTED_DATASETS]
    if invalid:
        parser.error(f'Unknown dataset(s): {invalid}')

    skipped = [name for name in selected if name in SKIPPED_DATASETS]
    selected = [name for name in selected if name not in SKIPPED_DATASETS]

    storage_root = Path(args.storage_root).resolve()
    for subdir in ('uci', 'mnist', 'cifar', 'block'):
        (storage_root / subdir).mkdir(parents=True, exist_ok=True)

    print(f'Storage root: {storage_root}')
    if skipped:
        for name in skipped:
            print(f'Skipping {name}: {SKIPPED_DATASETS[name]}')

    failures: list[tuple[str, str]] = []
    for dataset in selected:
        print(f'\n=== Downloading {dataset} ===')
        config = make_config(dataset, storage_root)
        try:
            load_dataset(config)
        except Exception as exc:  # surface all failures cleanly in one pass
            failures.append((dataset, f'{type(exc).__name__}: {exc}'))
            print(f'FAILED: {dataset} -> {type(exc).__name__}: {exc}')
        else:
            print(f'OK: {dataset}')

    print('\nDownload summary:')
    if selected:
        print(f'  attempted: {len(selected)}')
    else:
        print('  attempted: 0')
    print(f'  skipped: {len(skipped)}')
    print(f'  failed: {len(failures)}')

    if failures:
        print('\nFailures:')
        for dataset, message in failures:
            print(f'  - {dataset}: {message}')
        print("\nIf imports are missing, install the data extras with: pip install -e '.[data]'")
        return 1

    print('\nAll requested dataset downloads completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
