# Source: ported from Older_Imp.ipynb cell 5 (validate_logic_config)
# Config validation and snapshot helpers.
# Preserves all notebook config key names — do not rename them.
from __future__ import annotations

import json
from pathlib import Path


# ── Validation ───────────────────────────────────────────────────────────────

def validate_logic_config(config: dict) -> None:
    """Validate logic_family and architecture compatibility.

    Raises ValueError for unknown families or mismatched architectures.
    """
    logic_family = config['logic_family']
    architecture = config['architecture']

    valid_architectures = {
        'full': {'softmax', 'gumbel', 'dirichlet'},
        'light': {'light_sigmoid', 'light_st', 'light_sigmoid_st', 'sigmoid', 'sigmoid_st'},
    }

    if logic_family not in valid_architectures:
        raise ValueError(f"Unknown logic_family: {logic_family!r}")
    if architecture not in valid_architectures[logic_family]:
        raise ValueError(
            f"architecture={architecture!r} is not valid for logic_family={logic_family!r}. "
            f"Expected one of {sorted(valid_architectures[logic_family])}."
        )


# ── Default config ────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    'dataset': 'toy_xor',
    'seed': 0,
    'batch_size': 128,
    'valid_set_size': 0.1,
    'num_workers': 0,
    'num_steps': 1000,
    'eval_every': 100,
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
    'clip_value': 1.0,
    'connections': 'random',
    'logic_family': 'full',
    'architecture': 'softmax',
    'gumb_tau': 1.0,
    'dirichlet_concentration': 1.0,
    'sum_tau': 1.0,
    'num_neurons': 64,
    'num_layers': 4,
    'toy_num_bits': 8,
    'data_roots': {
        'block': './blocks_dataset',
        'uci': './data-uci',
        'mnist': './data-mnist',
        'cifar': './data-cifar',
    },
}


def build_config(overrides: dict) -> dict:
    """Return a config dict from DEFAULT_CONFIG with overrides applied."""
    config = dict(DEFAULT_CONFIG)
    # Merge data_roots separately so partial overrides work
    if 'data_roots' in overrides:
        merged_roots = dict(config['data_roots'])
        merged_roots.update(overrides.pop('data_roots'))
        config['data_roots'] = merged_roots
    config.update(overrides)
    return config


# ── Snapshot helpers ──────────────────────────────────────────────────────────

def save_config_snapshot(config: dict, run_dir: Path | str) -> None:
    """Write the config used for a run to run_dir/config.json."""
    path = Path(run_dir) / 'config.json'

    # data_roots may contain non-serializable Path objects — normalize
    serializable = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in config.items()
    }
    if 'data_roots' in serializable and isinstance(serializable['data_roots'], dict):
        serializable['data_roots'] = {
            k: str(v) for k, v in serializable['data_roots'].items()
        }

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_config_snapshot(run_dir: Path | str) -> dict:
    """Load config.json from a run directory."""
    path = Path(run_dir) / 'config.json'
    with open(path) as f:
        return json.load(f)
