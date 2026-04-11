# Guards full=16 and light=4 gate parameter widths.
# These widths must not silently change during any refactor.
import jax
import jax.numpy as jnp
import pytest

from dlgn.models.initialization import init_full_gates, init_light_gates
from dlgn.models.gates import NUMBER_OF_GATES, LIGHT_TRUTH_TABLE_SIZE


def test_full_gate_width():
    params = init_full_gates(n=8)
    assert params.shape == (8, NUMBER_OF_GATES), (
        f'Expected (8, {NUMBER_OF_GATES}), got {params.shape}'
    )
    assert NUMBER_OF_GATES == 16


def test_light_gate_width():
    params = init_light_gates(n=8)
    assert params.shape == (8, LIGHT_TRUTH_TABLE_SIZE), (
        f'Expected (8, {LIGHT_TRUTH_TABLE_SIZE}), got {params.shape}'
    )
    assert LIGHT_TRUTH_TABLE_SIZE == 4


def test_full_init_pass_through_bias():
    """Full gates should be biased toward PASS_THROUGH_GATE (index 3)."""
    from dlgn.models.gates import PASS_THROUGH_GATE, DEFAULT_PASS_VALUE
    params = init_full_gates(n=4)
    assert float(params[0, PASS_THROUGH_GATE]) == DEFAULT_PASS_VALUE
    assert float(params[0, 0]) == 0.0  # all other logits zero


def test_light_init_pass_through_like():
    """Light gates should be biased toward truth table bits [10] and [11] (indices 2,3)."""
    from dlgn.models.gates import DEFAULT_PASS_VALUE
    params = init_light_gates(n=4)
    assert float(params[0, 2]) == DEFAULT_PASS_VALUE
    assert float(params[0, 3]) == DEFAULT_PASS_VALUE
    assert float(params[0, 0]) == -DEFAULT_PASS_VALUE


def test_init_network_shapes():
    from dlgn.models.initialization import init_logic_gate_network
    key = jax.random.PRNGKey(0)
    params, wires = init_logic_gate_network(
        input_dim=8, num_neurons=16, num_layers=3,
        connections='unique', key=key, logic_family='full',
    )
    assert len(params) == 3
    assert len(wires) == 3
    assert params[0].shape == (16, NUMBER_OF_GATES)
    assert wires[0][0].shape == (16,)
    assert wires[0][1].shape == (16,)
