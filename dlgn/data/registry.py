# Source: ported from Older_Imp.ipynb cell 4 (input_dim_of_dataset, num_classes_of_dataset)
# Canonical dataset name registry. Dataset names and metadata are the ground truth.
# Do not rename datasets; do not remove names without explicit user approval.
from __future__ import annotations

# ── Supported dataset names ──────────────────────────────────────────────────
# block is listed but load_dataset raises NotImplementedError for it
# (BlocksDataset source is still missing).
SUPPORTED_DATASETS = frozenset([
    'toy_xor',
    'toy_parity',
    'block',
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
])


def input_dim_of_dataset(dataset: str) -> int:
    """Return the flat input feature dimension for a dataset name."""
    dims = {
        'toy_xor': 2,
        'toy_parity': 6,
        'block': 16,
        'adult': 116,
        'breast_cancer': 51,
        'monk1': 17,
        'monk2': 17,
        'monk3': 17,
        'mnist': 784,
        'mnist20x20': 400,
        'mnist_bin': 784,
        'mnist20x20_bin': 400,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }
    if dataset not in dims:
        raise ValueError(
            f'Unknown dataset {dataset!r}. '
            f'Supported datasets: {sorted(SUPPORTED_DATASETS)}'
        )
    return dims[dataset]


def num_classes_of_dataset(dataset: str) -> int:
    """Return the number of output classes for a dataset name."""
    classes = {
        'toy_xor': 2,
        'toy_parity': 2,
        'block': 16,
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'mnist_bin': 2,
        'mnist20x20_bin': 2,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }
    if dataset not in classes:
        raise ValueError(
            f'Unknown dataset {dataset!r}. '
            f'Supported datasets: {sorted(SUPPORTED_DATASETS)}'
        )
    return classes[dataset]
