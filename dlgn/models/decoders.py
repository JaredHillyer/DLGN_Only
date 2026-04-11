# Source: ported from Older_Imp.ipynb cell 2
# Gate-decoding behavior for full and light logic families.
# Decoder semantics must be preserved exactly — do not silently remap names.
from __future__ import annotations

import jax
import jax.numpy as jnp

from dlgn.models.gates import NUMBER_OF_GATES, LIGHT_TRUTH_TABLE_SIZE
from dlgn.types import LogicFamily

# ── Logic-family detection ───────────────────────────────────────────────────

def logic_family_of_logits(logits: jax.Array) -> LogicFamily:
    """Infer logic family from the last dimension of a logit tensor."""
    if logits.shape[-1] == LIGHT_TRUTH_TABLE_SIZE:
        return 'light'
    if logits.shape[-1] == NUMBER_OF_GATES:
        return 'full'
    raise ValueError(f'Unsupported gate parameter size: {logits.shape[-1]}')


# ── Full-family decoders ─────────────────────────────────────────────────────

def decode_soft(logits: jax.Array) -> jax.Array:
    """Softmax over gate logits (full family, training)."""
    return jax.nn.softmax(logits, axis=-1)


def decode_gumbel_soft(
    logits: jax.Array,
    key: jax.Array,
    temperature: float = 1.0,
) -> jax.Array:
    """Gumbel-softmax (soft forward, used internally by decode_gumbel_st)."""
    temperature = jnp.maximum(temperature, 1e-6)
    noise = jax.random.gumbel(key, logits.shape)
    return jax.nn.softmax((logits + noise) / temperature, axis=-1)


def decode_gumbel_st(
    logits: jax.Array,
    key: jax.Array,
    temperature: float = 1.0,
) -> jax.Array:
    """Gumbel straight-through: one-hot forward, soft backward gradient."""
    y_soft = decode_gumbel_soft(logits, key, temperature)
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1], dtype=y_soft.dtype)
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)


def decode_dirichlet_sample(
    logits: jax.Array,
    key: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """Dirichlet sample from softplus-transformed logits."""
    alpha = jax.nn.softplus(logits) + eps
    return jax.random.dirichlet(key, alpha)


def decode_full_hard(logits: jax.Array) -> jax.Array:
    """One-hot hard decode for the full family (eval path)."""
    return jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1])


# ── Light-family decoders ────────────────────────────────────────────────────

def decode_light_soft(logits: jax.Array) -> jax.Array:
    """Sigmoid over truth-table logits (light family, training)."""
    return jax.nn.sigmoid(logits)


def decode_light_st(logits: jax.Array) -> jax.Array:
    """Light straight-through: binary forward, sigmoid backward gradient."""
    truth_table_soft = decode_light_soft(logits)
    truth_table_hard = (truth_table_soft >= 0.5).astype(truth_table_soft.dtype)
    return truth_table_soft + jax.lax.stop_gradient(truth_table_hard - truth_table_soft)


def decode_light_hard(logits: jax.Array) -> jax.Array:
    """Binary hard decode for the light family (eval path)."""
    return (decode_light_soft(logits) >= 0.5).astype(logits.dtype)


# ── Unified decode dispatch ──────────────────────────────────────────────────

def decode_training_gates(
    logits: jax.Array,
    key: jax.Array,
    architecture: str,
    gumb_tau: float,
    dirichlet_concentration: float,
    logic_family: LogicFamily,
) -> jax.Array:
    """Select the correct soft decoder for the given architecture and logic family.

    Note: dirichlet_concentration is accepted for API compatibility but the
    current notebook implementation delegates concentration to decode_dirichlet_sample
    via softplus; the parameter is unused here (matching notebook behavior).
    """
    del dirichlet_concentration  # matches notebook: concentration handled in softplus

    if logic_family == 'full':
        if architecture == 'softmax':
            return decode_soft(logits)
        if architecture == 'gumbel':
            return decode_gumbel_st(logits, key, gumb_tau)
        if architecture == 'dirichlet':
            return decode_dirichlet_sample(logits, key)
        raise ValueError(f'Unknown full-DLGN architecture: {architecture!r}')

    if logic_family == 'light':
        if architecture in ('light_sigmoid', 'sigmoid'):
            return decode_light_soft(logits)
        if architecture in ('light_st', 'light_sigmoid_st', 'sigmoid_st'):
            return decode_light_st(logits)
        raise ValueError(f'Unknown light-DLGN architecture: {architecture!r}')

    raise ValueError(f'Unknown logic family: {logic_family!r}')


def decode_hard(logits: jax.Array, logic_family: LogicFamily) -> jax.Array:
    """Select the correct hard decoder for the given logic family (eval path)."""
    if logic_family == 'full':
        return decode_full_hard(logits)
    if logic_family == 'light':
        return decode_light_hard(logits)
    raise ValueError(f'Unknown logic family: {logic_family!r}')
