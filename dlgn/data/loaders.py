# Source: ported from Older_Imp.ipynb cell 4
# Loader orchestration. Replaces _require_module indirection with direct local imports.
# load_dataset is the single dispatch point for all datasets.
# Torch is imported lazily so the JAX-only toy path does not require the data extras
# or a working local Torch runtime.
# Source: ported from Older_Imp.ipynb cell 4 (input_dim_of_dataset, num_classes_of_dataset)
# Canonical dataset name registry. Dataset names and metadata are the ground truth.
# Do not rename datasets; do not remove names without explicit user approval.

from __future__ import annotations
import math
import numpy as np
import torch
import torchvision

# ── Supported dataset names ──────────────────────────────────────────────────
SUPPORTED_DATASETS = frozenset([
    'adult',
    'breast_cancer',
    'monk1',
    'monk2',
    'monk3',
    'mnist',
    'mnist20x20',
    'mnist_bin',
    'mnist20x20_bin',
    'cifar10',
    'cifar-10-3-thresholds',
    'cifar-10-31-thresholds',
])


# Datasets in the CIFAR-10 family share a fixed 3-channel, 32x32 spatial
# shape and the threshold-bit encoding multiplies that by the bit count.
_CIFAR10_FAMILY = frozenset({'cifar10', 'cifar-10-3-thresholds', 'cifar-10-31-thresholds'})


def input_dim_of_dataset(dataset: str, threshold_bits: int | None = None) -> int:
    """Return the flat input feature dimension for a dataset name.

    For datasets in the CIFAR-10 family, ``threshold_bits`` overrides the
    static registry value and returns ``3 * 32 * 32 * threshold_bits``. This
    matches the V2 trainer's behavior of running on `cifar-10-3-thresholds`
    metadata but with a 5-threshold effective encoding. See D15 in
    dlgn_decision_log.md.

    - Bare ``'cifar10'`` requires ``threshold_bits``; raises ``ValueError``
      if missing.
    - For non-CIFAR datasets, ``threshold_bits`` is silently ignored so
      upstream callers can pass it unconditionally.
    """
    dims = {
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
    if dataset in _CIFAR10_FAMILY:
        if threshold_bits is not None:
            return 3 * 32 * 32 * int(threshold_bits)
        # No override given: fall back to the static entry if it exists.
        if dataset in dims:
            return dims[dataset]
        # Bare 'cifar10' with no override is a loud error.
        raise ValueError(
            f"input_dim_of_dataset({dataset!r}) requires threshold_bits "
            f"because {dataset!r} has no static input-dim entry."
        )
    if dataset not in dims:
        raise ValueError(
            f'Unknown dataset {dataset!r}. '
            f'Supported datasets: {sorted(SUPPORTED_DATASETS)}'
        )
    return dims[dataset]


def num_classes_of_dataset(dataset: str) -> int:
    """Return the number of output classes for a dataset name."""
    classes = {
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'mnist_bin': 2,
        'mnist20x20_bin': 2,
        'cifar10': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }
    if dataset not in classes:
        raise ValueError(
            f'Unknown dataset {dataset!r}. '
            f'Supported datasets: {sorted(SUPPORTED_DATASETS)}'
        )
    return classes[dataset]



class ArrayDataLoader:
    """NumPy-backed data loader for toy/array datasets (no torch dependency)."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = np.arange(len(self.x))
        if self.shuffle:
            self.rng.shuffle(indices)
        step = self.batch_size
        stop = len(indices) if not self.drop_last else len(indices) - (len(indices) % step)
        for start in range(0, stop, step):
            batch_idx = indices[start:start + step]
            if len(batch_idx) < step and self.drop_last:
                continue
            yield self.x[batch_idx], self.y[batch_idx]

    def __len__(self):
        n = len(self.x)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def cycle_loader(loader):
    """Cycle infinitely over a loader for step-based training."""
    while True:
        for batch in loader:
            yield batch


def _make_torch_loaders_from_split(
    train_set,
    test_set,
    batch_size: int,
    valid_set_size: float,
    seed: int,
    num_workers: int,
):
    """Create train/validation/test DataLoaders with an optional validation split."""
    validation_loader = None
    if valid_set_size > 0:
        train_set_size = math.ceil((1 - valid_set_size) * len(train_set))
        valid_size = len(train_set) - train_set_size
        generator = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(
            train_set,
            [train_set_size, valid_size],
            generator=generator,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=num_workers,
        )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader, test_loader

def load_cifar_threshold(
    dataset: str,
    batch_size: int,
    valid_set_size: float,
    seed: int,
    data_root: str = './data-cifar',
    num_workers: int = 0,
    n_thresholds: int | None = None,
):
    """Load thresholded CIFAR-10 for DLGN.

    Args:
        n_thresholds: If provided, use this threshold count directly.
            Otherwise fall back to the dataset-name lookup.
    """
    if dataset == 'cifar-10-real-input':
        raise ValueError(
            'cifar-10-real-input is not supported: '
            'DLGN expects thresholded/binary-style inputs.'
        )

    if n_thresholds is None:
        threshold_map = {
            'cifar-10-3-thresholds': 3,
            'cifar-10-31-thresholds': 31,
        }
        if dataset not in threshold_map:
            raise ValueError(
                f'Unknown CIFAR dataset {dataset!r} and n_thresholds not provided. '
                f'Supported names: {sorted(threshold_map)}'
            )
        n_thresholds = threshold_map[dataset]
    transform_fn = lambda x: np.concatenate(
        [(x > (i + 1) / (n_thresholds + 1)).astype(np.float32) for i in range(n_thresholds)],
        axis=0,
    )

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: transform_fn(x.numpy())),
    ])

    train_set = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True, transform=transforms
    )
    test_set = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=transforms
    )

    return _make_torch_loaders_from_split(
        train_set, test_set, batch_size, valid_set_size, seed, num_workers
    )


def load_dataset(config: dict):
    """Dispatch to the correct dataset loader based on config['dataset'].

    Returns (train_loader, validation_loader, test_loader).
    validation_loader may be None if valid_set_size == 0.
    """
    dataset = config['dataset']
    batch_size = config['batch_size']
    valid_set_size = config.get('valid_set_size', 0.0)
    seed = config['seed']
    num_workers = config.get('num_workers', 0)
    data_roots = config.get('data_roots', {
        'uci': './data-uci',
        'mnist': './data-mnist',
        'cifar': './data-cifar',
    })

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f'Unknown dataset {dataset!r}. '
            f'Supported datasets: {sorted(SUPPORTED_DATASETS)}'
        )

    # ── UCI datasets ─────────────────────────────────────────────────────────
    if dataset == 'adult':
        from dlgn.data.uci import AdultDataset
        train_set = AdultDataset(data_roots['uci'], split='train', download=True, with_val=False)
        test_set = AdultDataset(data_roots['uci'], split='test', with_val=False)
        return _make_torch_loaders_from_split(train_set, test_set, batch_size,
                                              valid_set_size, seed, num_workers)

    if dataset == 'breast_cancer':
        from dlgn.data.uci import BreastCancerDataset
        train_set = BreastCancerDataset(data_roots['uci'], split='train', download=True,
                                        with_val=False)
        test_set = BreastCancerDataset(data_roots['uci'], split='test', with_val=False)
        return _make_torch_loaders_from_split(train_set, test_set, batch_size,
                                              valid_set_size, seed, num_workers)

    if dataset.startswith('monk'):
        from dlgn.data.uci import MONKsDataset
        style = int(dataset[4])
        train_set = MONKsDataset(data_roots['uci'], style, split='train', download=True,
                                 with_val=False)
        test_set = MONKsDataset(data_roots['uci'], style, split='test', download=True,
                                with_val=False)
        return _make_torch_loaders_from_split(train_set, test_set, batch_size,
                                              valid_set_size, seed, num_workers)

    # ── MNIST datasets ───────────────────────────────────────────────────────
    if dataset in ('mnist', 'mnist20x20'):
        from dlgn.data.mnist import MNIST
        remove_border = dataset == 'mnist20x20'
        train_set = MNIST(data_roots['mnist'], train=True, download=True,
                          remove_border=remove_border)
        test_set = MNIST(data_roots['mnist'], train=False, download=True,
                         remove_border=remove_border)
        return _make_torch_loaders_from_split(train_set, test_set, batch_size,
                                              valid_set_size, seed, num_workers)

    if dataset in ('mnist_bin', 'mnist20x20_bin'):
        from dlgn.data.mnist import MNIST
        remove_border = dataset == 'mnist20x20_bin'
        train_set = MNIST(data_roots['mnist'], train=True, download=True,
                          remove_border=remove_border)
        test_set = MNIST(data_roots['mnist'], train=False, download=True,
                         remove_border=remove_border)

        keep = torch.tensor([0, 1])

        def keep_only(ds, keep_labels):
            mask = torch.isin(ds.targets, keep_labels)
            ds.data = ds.data[mask]
            ds.targets = ds.targets[mask]
            ds.classes = [ds.classes[k] for k in keep_labels.tolist()]
            return ds

        train_set = keep_only(train_set, keep)
        test_set = keep_only(test_set, keep)
        return _make_torch_loaders_from_split(train_set, test_set, batch_size,
                                              valid_set_size, seed, num_workers)

    # ── CIFAR threshold datasets ─────────────────────────────────────────────
    if dataset == 'cifar10' or dataset.startswith('cifar-10'):
        return load_cifar_threshold(
            dataset=dataset,
            batch_size=batch_size,
            valid_set_size=valid_set_size,
            seed=seed,
            data_root=data_roots.get('cifar', './data-cifar'),
            num_workers=num_workers,
            n_thresholds=config.get('threshold_bits'),
        )

    # Should not reach here since we checked SUPPORTED_DATASETS above.
    raise NotImplementedError(f'Dataset {dataset!r} is in the registry but has no loader.')