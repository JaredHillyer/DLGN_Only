# Guards GroupSum divisibility guard and output shape/temperature semantics.
import jax.numpy as jnp
import pytest

from dlgn.models.heads import group_sum_head


def test_group_sum_shape():
    x = jnp.ones((4, 20))  # batch=4, neurons=20
    out = group_sum_head(x, class_count=4, sum_tau=1.0)
    assert out.shape == (4, 4)


def test_group_sum_divisibility_guard():
    x = jnp.ones((4, 10))  # 10 neurons, 3 classes → not divisible
    with pytest.raises(ValueError, match='divisible'):
        group_sum_head(x, class_count=3, sum_tau=1.0)


def test_group_sum_values():
    """Each class group sums to group_size / sum_tau."""
    x = jnp.ones((1, 8))   # 8 neurons, 2 classes → group_size=4
    out = group_sum_head(x, class_count=2, sum_tau=1.0)
    assert out.shape == (1, 2)
    assert float(out[0, 0]) == pytest.approx(4.0)
    assert float(out[0, 1]) == pytest.approx(4.0)


def test_group_sum_temperature():
    """sum_tau scales output inversely."""
    x = jnp.ones((1, 4))
    out_1 = group_sum_head(x, class_count=2, sum_tau=1.0)
    out_2 = group_sum_head(x, class_count=2, sum_tau=2.0)
    assert float(out_1[0, 0]) == pytest.approx(2.0)
    assert float(out_2[0, 0]) == pytest.approx(1.0)


def test_group_sum_zero_tau_clamped():
    """sum_tau=0 should not cause division by zero (clamped to 1e-6)."""
    x = jnp.ones((1, 4))
    out = group_sum_head(x, class_count=2, sum_tau=0.0)
    assert jnp.isfinite(out).all()
