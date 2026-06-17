# Source: ported from Older_Imp.ipynb cell 2
# DLGN forward pass. Preserves per-layer key splitting and soft/hard decode split.
from __future__ import annotations

import jax
import jax.numpy as jnp

from dlgn.models.decoders import decode_training_gates, decode_hard
from dlgn.models.gates import bin_op_s, bin_op_light
from dlgn.models.heads import group_sum_head
from dlgn.types import LogicFamily, WirePair


def run_layer(
    logits: jax.Array,
    wires: WirePair,
    x: jax.Array,
    training: bool,
    key: jax.Array,
    architecture: str,
    gumb_tau: float,
    dirichlet_concentration: float,
    logic_family: LogicFamily,
) -> jax.Array:
    """Run one gate layer: wire inputs, decode gates, apply binary operation."""
    a = x[..., wires[0]]
    b = x[..., wires[1]]
    gate_weights = (
        decode_training_gates(
            logits,
            key,
            architecture,
            gumb_tau,
            dirichlet_concentration,
            logic_family,
        )
        if training
        else decode_hard(logits, logic_family, architecture)
    )

    if logic_family == 'full':
        return bin_op_s(a, b, gate_weights)
    if logic_family == 'light':
        return bin_op_light(a, b, gate_weights)
    raise ValueError(f'Unknown logic family: {logic_family!r}')


def run_logic_gate_network(
    params: list[jax.Array],
    wires: list[WirePair],
    x: jax.Array,
    training: bool,
    key: jax.Array,
    architecture: str,
    gumb_tau: float,
    dirichlet_concentration: float,
    logic_family: LogicFamily,
) -> jax.Array:
    """Run all gate layers. Splits one key per layer to preserve RNG threading."""
    layer_keys = jax.random.split(key, len(params))
    for logits, layer_wires, layer_key in zip(params, wires, layer_keys):
        x = run_layer(
            logits,
            layer_wires,
            x,
            training,
            layer_key,
            architecture,
            gumb_tau,
            dirichlet_concentration,
            logic_family,
        )
    return x


def forward_logits(
    params: list[jax.Array],
    wires: list[WirePair],
    x: jax.Array,
    training: bool,
    key: jax.Array,
    architecture: str,
    gumb_tau: float,
    dirichlet_concentration: float,
    class_count: int,
    sum_tau: float,
    logic_family: LogicFamily,
) -> jax.Array:
    """Full forward pass: gate network then GroupSum head, returns class logits."""
    features = run_logic_gate_network(
        params,
        wires,
        x,
        training,
        key,
        architecture,
        gumb_tau,
        dirichlet_concentration,
        logic_family,
    )
    return group_sum_head(features, class_count, sum_tau)
