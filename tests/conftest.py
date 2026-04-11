"""Test-path bootstrap for running `pytest` from the repo root.

The review pass found that invoking the `pytest` entrypoint directly did not put the
repository root on `sys.path` in this environment, even though `python -m pytest`
worked. Keep this bootstrap minimal and local to tests.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

