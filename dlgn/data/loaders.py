# Source: ported from Older_Imp.ipynb cell 4
# Loader orchestration. Replaces _require_module indirection with direct local imports.
# load_dataset is the single dispatch point for all datasets.
# Torch is imported lazily so the JAX-only toy path does not require the data extras
# or a working local Torch runtime.
from __future__ import annotations

import math

import numpy as np

from dlgn.data.registry import SUPPORTED_DATASETS
from dlgn.data.toy import make_xor_arrays, make_parity_arrays


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
    import torch

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
        'block': './blocks_dataset',
        'uci': './data-uci',
        'mnist': './data-mnist',
        'cifar': './data-cifar',
    })

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f'Unknown dataset {dataset!r}. '
            f'Supported datasets: {sorted(SUPPORTED_DATASETS)}'
        )

    # ── Toy datasets (no torch) ──────────────────────────────────────────────
    if dataset == 'toy_xor':
        train_x, train_y = make_xor_arrays(repeats=1024)
        test_x, test_y = make_xor_arrays(repeats=64)
        train_loader = ArrayDataLoader(train_x, train_y, batch_size=batch_size,
                                       shuffle=True, drop_last=True, seed=seed)
        test_loader = ArrayDataLoader(test_x, test_y, batch_size=batch_size,
                                      shuffle=False, drop_last=False, seed=seed + 1)
        return train_loader, None, test_loader

    if dataset == 'toy_parity':
        num_bits = config.get('toy_num_bits', 6)
        train_x, train_y = make_parity_arrays(num_samples=4096, num_bits=num_bits, seed=seed)
        test_x, test_y = make_parity_arrays(num_samples=1024, num_bits=num_bits, seed=seed + 1)
        train_loader = ArrayDataLoader(train_x, train_y, batch_size=batch_size,
                                       shuffle=True, drop_last=True, seed=seed)
        test_loader = ArrayDataLoader(test_x, test_y, batch_size=batch_size,
                                      shuffle=False, drop_last=False, seed=seed + 1)
        return train_loader, None, test_loader

    # ── Block dataset ────────────────────────────────────────────────────────
    if dataset == 'block':
        from dlgn.data.block import load_block_dataset
        return load_block_dataset(data_roots['block'])  # raises NotImplementedError

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
        import torch
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
    if dataset.startswith('cifar-10'):
        from dlgn.data.cifar_threshold import load_cifar_threshold
        return load_cifar_threshold(
            dataset=dataset,
            batch_size=batch_size,
            valid_set_size=valid_set_size,
            seed=seed,
            data_root=data_roots.get('cifar', './data-cifar'),
            num_workers=num_workers,
        )

    # Should not reach here since we checked SUPPORTED_DATASETS above.
    raise NotImplementedError(f'Dataset {dataset!r} is in the registry but has no loader.')
