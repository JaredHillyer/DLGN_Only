# Source: block dataset boundary — BlocksDataset source is missing from this workspace.
# This module exists so the dataset name does not silently disappear from the registry.
from __future__ import annotations


def load_block_dataset(data_root: str, **kwargs):
    """Placeholder that raises a clear error until BlocksDataset is recovered.

    TODO: Implement once take_datasets.block_datasets.BlocksDataset source is
    located or re-implemented.
    """
    raise NotImplementedError(
        "BlocksDataset is not available in this workspace. "
        "The source module 'take_datasets.block_datasets.BlocksDataset' is missing."
    )
