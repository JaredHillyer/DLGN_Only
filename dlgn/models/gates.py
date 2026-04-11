# Source: ported from Older_Imp.ipynb cell 2
# Gate constants, truth-table helpers, and binary gate operations.
# Gate ID ordering is canonical — do not change it.
from __future__ import annotations

import jax
import jax.numpy as jnp

# ── Constants ────────────────────────────────────────────────────────────────

PASS_THROUGH_GATE = 3        # gate index for identity-A in the full family
DEFAULT_PASS_VALUE = 10.0    # initial logit bias toward pass-through
NUMBER_OF_GATES = 16         # full-family gate count — must stay 16
LIGHT_TRUTH_TABLE_SIZE = 4   # light-family gate count — must stay 4

TRUTH_TABLE_BIT_WEIGHTS = jnp.asarray([8, 4, 2, 1], dtype=jnp.int32)

# Ordered exactly as in the notebook — position == gate ID.
GATE_NAMES = [
    'FALSE',
    'AND',
    'A_AND_NOT_B',
    'A',
    'NOT_A_AND_B',
    'B',
    'XOR',
    'OR',
    'NOR',
    'XNOR',
    'NOT_B',
    'A_OR_NOT_B',
    'NOT_A',
    'NOT_A_OR_B',
    'NAND',
    'TRUE',
]

LIGHT_TRUTH_TABLE_NAMES = ['00', '01', '10', '11']

# ── Gate ID helpers ──────────────────────────────────────────────────────────

def truth_table_bits_to_gate_id(bits: jax.Array) -> jax.Array:
    """Convert a 4-bit truth-table row into the corresponding full-family gate ID."""
    bits = bits.astype(jnp.int32)
    return jnp.sum(bits * TRUTH_TABLE_BIT_WEIGHTS, axis=-1).astype(jnp.int32)


def hard_gate_ids_from_logits(logits: jax.Array) -> jax.Array:
    """Return hard gate IDs from logits for either logic family."""
    from dlgn.models.decoders import logic_family_of_logits, decode_light_hard
    logic_family = logic_family_of_logits(logits)
    if logic_family == 'light':
        return truth_table_bits_to_gate_id(decode_light_hard(logits))
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


# ── Binary gate operations ───────────────────────────────────────────────────

def bin_op_all_combinations(a: jax.Array, b: jax.Array) -> jax.Array:
    """Return a tensor of shape (*batch, 16) with all 16 binary gate outputs."""
    return jnp.stack(
        [
            jnp.zeros_like(a),          # 0  FALSE
            a * b,                       # 1  AND
            a - a * b,                   # 2  A AND NOT B
            a,                           # 3  A
            b - a * b,                   # 4  NOT A AND B
            b,                           # 5  B
            a + b - 2 * a * b,           # 6  XOR
            a + b - a * b,               # 7  OR
            1 - (a + b - a * b),         # 8  NOR
            1 - (a + b - 2 * a * b),     # 9  XNOR
            1 - b,                       # 10 NOT B
            1 - b + a * b,               # 11 A OR NOT B
            1 - a,                       # 12 NOT A
            1 - a + a * b,               # 13 NOT A OR B
            1 - a * b,                   # 14 NAND
            jnp.ones_like(a),            # 15 TRUE
        ],
        axis=-1,
    )


def bin_op_s(a: jax.Array, b: jax.Array, gate_weights: jax.Array) -> jax.Array:
    """Full-family soft gate op: weighted sum over all 16 gate outputs."""
    return jnp.sum(bin_op_all_combinations(a, b) * gate_weights, axis=-1)


def light_truth_table_basis(a: jax.Array, b: jax.Array) -> jax.Array:
    """4-element truth-table basis for the light family."""
    return jnp.stack(
        [
            (1.0 - a) * (1.0 - b),   # (0,0)
            (1.0 - a) * b,            # (0,1)
            a * (1.0 - b),            # (1,0)
            a * b,                    # (1,1)
        ],
        axis=-1,
    )


def bin_op_light(a: jax.Array, b: jax.Array, truth_table_weights: jax.Array) -> jax.Array:
    """Light-family soft gate op: weighted sum over truth-table basis."""
    return jnp.sum(light_truth_table_basis(a, b) * truth_table_weights, axis=-1)
