# Guards dataset name contract and metadata stability.
import pytest

from dlgn.data.registry import (
    input_dim_of_dataset,
    num_classes_of_dataset,
    SUPPORTED_DATASETS,
)


REQUIRED_DATASETS = [
    'toy_xor', 'toy_parity', 'block', 'adult', 'breast_cancer',
    'monk1', 'monk2', 'monk3', 'mnist', 'mnist20x20', 'mnist_bin',
    'mnist20x20_bin', 'cifar-10-3-thresholds', 'cifar-10-31-thresholds',
]

EXPECTED_DIMS = {
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

EXPECTED_CLASSES = {
    'toy_xor': 2, 'toy_parity': 2, 'block': 16, 'adult': 2,
    'breast_cancer': 2, 'monk1': 2, 'monk2': 2, 'monk3': 2,
    'mnist': 10, 'mnist20x20': 10, 'mnist_bin': 2, 'mnist20x20_bin': 2,
    'cifar-10-3-thresholds': 10, 'cifar-10-31-thresholds': 10,
}


def test_all_required_datasets_in_registry():
    for name in REQUIRED_DATASETS:
        assert name in SUPPORTED_DATASETS, f'{name!r} missing from SUPPORTED_DATASETS'


@pytest.mark.parametrize('dataset,expected', EXPECTED_DIMS.items())
def test_input_dim(dataset, expected):
    assert input_dim_of_dataset(dataset) == expected


@pytest.mark.parametrize('dataset,expected', EXPECTED_CLASSES.items())
def test_num_classes(dataset, expected):
    assert num_classes_of_dataset(dataset) == expected


def test_unknown_dataset_raises_value_error():
    with pytest.raises(ValueError, match='Unknown dataset'):
        input_dim_of_dataset('nonexistent_dataset')

    with pytest.raises(ValueError, match='Unknown dataset'):
        num_classes_of_dataset('nonexistent_dataset')
