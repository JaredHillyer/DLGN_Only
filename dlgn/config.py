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

    # Every architecture reachable from decode_training_gates is whitelisted
    # (see D3 in dlgn_decision_log.md). Adding a new dispatcher branch in
    # dlgn/models/decoders.py without adding it here will trip
    # validate_logic_config — that is intentional, so the validator stays
    # the single source of truth for what is a supported architecture.
    valid_architectures = {
        'full': {
            'softmax', 'gumbel', 'dirichlet',
            # Straight-through variants — one-hot forward, soft backward.
            'full_st', 'softmax_st',
        },
        'light': {
            'light_sigmoid', 'light_st', 'light_sigmoid_st', 'sigmoid', 'sigmoid_st',
            # Sinusoidal soft estimator + STE variant.
            'light_sin01', 'light_sin01_st',
            # Linear-clip soft estimator + STE variant.
            'light_linear', 'light_linear_st',
        },
    }

    if logic_family not in valid_architectures:
        raise ValueError(f"Unknown logic_family: {logic_family!r}")
    if architecture not in valid_architectures[logic_family]:
        raise ValueError(
            f"architecture={architecture!r} is not valid for logic_family={logic_family!r}. "
            f"Expected one of {sorted(valid_architectures[logic_family])}."
        )


# Note: DEFAULT_CONFIG and build_config were removed per D5 in
# dlgn_decision_log.md. The CLI requires explicit args, and train_model
# expects a fully-populated config dict from the caller. The old default
# pointed at 'toy_xor' which has no registered loader, so any caller of
# build_config(overrides={}) would immediately fail at load_dataset —
# the partial-config merger had no working code path.


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
