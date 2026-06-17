# Source: ported from Older_Imp.ipynb cell 5
# TrainState — the only mutable container passed through the training loop.
# Field names must not change: params, opt_state, key.
# wires are kept separate from TrainState (notebook convention).
#
# Uses flax.struct.dataclass for .replace() compatibility with the notebook.
# If your flax version does not support flax.struct, see the note below.
#
# NOTE: flax.struct.dataclass was deprecated in flax >= 0.7 and removed in
# some versions. If you encounter ImportError or AttributeError on flax.struct,
# replace this with a standard frozen dataclass and update .replace() calls to
# use dataclasses.replace(state, ...) in the training code.
from __future__ import annotations

import flax
import jax
import optax


@flax.struct.dataclass
class TrainState:
    params: list[jax.Array]
    opt_state: optax.OptState
    key: jax.Array
