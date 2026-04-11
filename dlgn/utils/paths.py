# New file — run directory management for CLI experiments.
# Provides deterministic, human-readable run-name generation and output dir layout.
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def make_run_name(config: dict, timestamp: str | None = None) -> str:
    """Generate a readable run name from config fields.

    Pattern: YYYYMMDD-HHMMSS_<dataset>_<logic_family>_<architecture>_s<seed>
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    dataset = config.get('dataset', 'unknown')
    lf = config.get('logic_family', 'unk')
    arch = config.get('architecture', 'unk')
    seed = config.get('seed', 0)
    return f'{timestamp}_{dataset}_{lf}_{arch}_s{seed}'


def make_run_dir(base_output_dir: str | Path, run_name: str) -> Path:
    """Create and return a run directory with standard subdirectories."""
    run_dir = Path(base_output_dir) / 'runs' / run_name
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    return run_dir
