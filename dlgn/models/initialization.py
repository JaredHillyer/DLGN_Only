# Source: ported from Older_Imp.ipynb cell 2
# Network initialization and wiring. Preserves pass-through-biased init.
from __future__ import annotations

import jax
import jax.numpy as jnp

from dlgn.models.gates import (
    NUMBER_OF_GATES,
    LIGHT_TRUTH_TABLE_SIZE,
    PASS_THROUGH_GATE,
    DEFAULT_PASS_VALUE,
)
from dlgn.types import ConnectionType, LogicFamily, WirePair

# ── Wiring helpers ───────────────────────────────────────────────────────────

def get_unique_connections(in_dim: int, out_dim: int, key: jax.Array) -> WirePair:
    """Generate out_dim wire pairs that cover in_dim inputs with minimal overlap."""
    if out_dim * 2 < in_dim:
        raise ValueError(
            'unique connections require out_dim * 2 >= in_dim: '
            f'{out_dim=} {in_dim=}'
        )

    x = jnp.arange(in_dim)
    a = x[::2]
    b = x[1::2]
    m = min(a.shape[0], b.shape[0])
    a = a[:m]
    b = b[:m]

    if a.shape[0] < out_dim:
        a_ = x[1::2]
        b_ = x[2::2]
        m = min(a_.shape[0], b_.shape[0])
        a = jnp.concatenate([a, a_[:m]])
        b = jnp.concatenate([b, b_[:m]])

    offset = 2
    while out_dim > a.shape[0] and offset < in_dim:
        a = jnp.concatenate([a, x[:-offset]])
        b = jnp.concatenate([b, x[offset:]])
        offset += 1

    if a.shape[0] < out_dim:
        raise ValueError(
            f'Could not generate enough unique connections: {a.shape[0]} < {out_dim}'
        )

    a = a[:out_dim]
    b = b[:out_dim]
    perm = jax.random.permutation(key, out_dim)
    return a[perm], b[perm]


# ── Gate parameter initialization ───────────────────────────────────────────

def init_full_gates(
    n: int,
    num_gates: int = NUMBER_OF_GATES,
    pass_through_gate: int = PASS_THROUGH_GATE,
    default_pass_value: float = DEFAULT_PASS_VALUE,
) -> jax.Array:
    """Initialize full-family gate logits, biased toward pass-through gate A."""
    gates = jnp.zeros((n, num_gates))
    return gates.at[:, pass_through_gate].set(default_pass_value)


def init_light_gates(
    n: int,
    default_pass_value: float = DEFAULT_PASS_VALUE,
) -> jax.Array:
    """Initialize light-family gate logits, biased toward pass-through-like behavior."""
    gates = jnp.full((n, LIGHT_TRUTH_TABLE_SIZE), -default_pass_value)
    return gates.at[:, 2:].set(default_pass_value)


# ── Layer and network initialization ────────────────────────────────────────

def init_gate_layer(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    connection_type: ConnectionType,
    logic_family: LogicFamily,
) -> tuple[jax.Array, WirePair]:
    """Initialize one gate layer: returns (params, wires)."""
    if connection_type == 'random':
        key1, key2 = jax.random.split(key)
        c = jax.random.permutation(key2, 2 * out_dim) % in_dim
        c = jax.random.permutation(key1, in_dim)[c]
        c = c.reshape(2, out_dim)
        wires = (c[0], c[1])
    elif connection_type == 'unique':
        wires = get_unique_connections(in_dim, out_dim, key)
    else:
        raise ValueError(f'Unknown connection type: {connection_type!r}')

    if logic_family == 'full':
        gate_params = init_full_gates(out_dim)
    elif logic_family == 'light':
        gate_params = init_light_gates(out_dim)
    else:
        raise ValueError(f'Unknown logic family: {logic_family!r}')

    return gate_params, wires


def build_layer_sizes(input_dim: int, num_neurons: int, num_layers: int) -> list[int]:
    """Return list of layer sizes: [input_dim, num_neurons, ..., num_neurons]."""
    if num_layers < 1:
        raise ValueError('num_layers must be at least 1')
    return [input_dim] + [num_neurons] * num_layers


def init_logic_gate_network(
    input_dim: int,
    num_neurons: int,
    num_layers: int,
    connections: ConnectionType,
    key: jax.Array,
    logic_family: LogicFamily,
) -> tuple[list[jax.Array], list[WirePair]]:
    """Initialize a full DLGN: returns (params, wires) lists, one entry per layer."""
    layer_sizes = build_layer_sizes(input_dim, num_neurons, num_layers)
    params = []
    wires = []

    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, subkey = jax.random.split(key)
        gate_logits, gate_wires = init_gate_layer(
            subkey,
            int(in_dim),
            int(out_dim),
            connections,
            logic_family,
        )
        params.append(gate_logits)
        wires.append(gate_wires)

    return params, wires
