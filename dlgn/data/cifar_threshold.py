# Source: ported from Older_Imp.ipynb cell 4 (CIFAR branch)
# Thresholded CIFAR-10 loaders for DLGN experiments.
# Supports only threshold-encoded variants — cifar-10-real-input is NOT supported.
# External coupling to execution_setup.directories.DIR_DATA is removed.
from __future__ import annotations

import math

import numpy as np
import torch
import torchvision


def _make_torch_loaders_from_split(
    train_set,
    test_set,
    batch_size: int,
    valid_set_size: float,
    seed: int,
    num_workers: int,
):
    """Split train set optionally and create DataLoaders. Internal helper."""
    validation_loader = None
    if valid_set_size > 0:
        train_set_size = math.ceil((1 - valid_set_size) * len(train_set))
        valid_size = len(train_set) - train_set_size
        generator = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(
            train_set, [train_set_size, valid_size], generator=generator
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=batch_size, shuffle=False,
            pin_memory=True, drop_last=False, num_workers=num_workers,
        )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, drop_last=True, num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, drop_last=False, num_workers=num_workers,
    )
    return train_loader, validation_loader, test_loader


def load_cifar_threshold(
    dataset: str,
    batch_size: int,
    valid_set_size: float,
    seed: int,
    data_root: str = './data-cifar',
    num_workers: int = 0,
):
    """Load thresholded CIFAR-10 for DLGN.

    Supported dataset names:
      - 'cifar-10-3-thresholds'   (3 thresholds per channel)
      - 'cifar-10-31-thresholds'  (31 thresholds per channel)
    """
    if dataset == 'cifar-10-real-input':
        raise ValueError(
            'cifar-10-real-input is not supported: '
            'DLGN expects thresholded/binary-style inputs.'
        )

    threshold_map = {
        'cifar-10-3-thresholds': 3,
        'cifar-10-31-thresholds': 31,
    }
    if dataset not in threshold_map:
        raise ValueError(
            f'Unknown CIFAR dataset {dataset!r}. '
            f'Supported: {sorted(threshold_map)}'
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
