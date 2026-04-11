# Guards a wiring constraint that explains why the XOR smoke path uses
# `connections="random"` instead of `connections="unique"`.
#
# For toy_xor, input_dim=2 and the smoke config uses num_neurons=4. The notebook's
# unique-wiring generator cannot construct four distinct wire pairs from two inputs,
# so `unique` is structurally invalid for that shape rather than a semantic choice.
import jax
import pytest

from dlgn.models.initialization import init_logic_gate_network


def test_unique_connections_invalid_for_xor_smoke_shape():
    with pytest.raises(ValueError, match='unique connections|Could not generate enough unique connections'):
        init_logic_gate_network(
            input_dim=2,
            num_neurons=4,
            num_layers=2,
            connections='unique',
            key=jax.random.PRNGKey(0),
            logic_family='full',
        )

