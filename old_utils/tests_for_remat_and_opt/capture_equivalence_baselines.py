"""One-shot baseline capture for tests/test_equivalence.py.

Runs each numerical baseline expression on the current commit and prints
literal jnp.array(...) values. The output is pasted into tests/test_equivalence.py
to pin behavior.

Re-run only when a Track-A decision intentionally rewrites a test
(see dlgn_decision_log.md for which decision invalidates which test).

Usage:
    cd DLGN_Only_Lean
    python scripts/capture_equivalence_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ── Import the package surface we're pinning ─────────────────────────────────
from dlgn.models.gates import bin_op_s, bin_op_light
from dlgn.models.heads import group_sum_head
from dlgn.models.conv import or_pool, max_pool
from dlgn.models.initialization import init_logic_gate_network
from dlgn.training.state import TrainState
from dlgn.training.steps import make_train_step, eval_batch

# Force float32 deterministic mode
jax.config.update('jax_enable_x64', False)


def fmt_array(arr, name: str, *, precision: int = 12) -> str:
    """Return a literal jnp.array(...) string for embedding in tests."""
    arr_np = np.asarray(jax.device_get(arr))
    if arr_np.dtype == np.float32:
        # repr emits full precision; we want compact but bit-equal.
        # np.array2string with precision controls display only; we need
        # the actual literal that reconstructs the same float32 bits.
        # Use repr of a list-of-floats so float() literals round-trip via
        # struct/IEEE-754 — works for float32 because 9 significant digits
        # uniquely determine a float32 value.
        s = np.array2string(
            arr_np,
            separator=', ',
            threshold=np.inf,
            max_line_width=80,
            precision=9,
            floatmode='maxprec_equal',
        )
    else:
        s = np.array2string(
            arr_np,
            separator=', ',
            threshold=np.inf,
            max_line_width=80,
        )
    s = s.replace('\n', '\n    ')
    dtype = f'jnp.{arr_np.dtype}'
    return f'EXPECTED_{name} = jnp.array(\n    {s},\n    dtype={dtype},\n)'


# ── T1: bin_op_s ─────────────────────────────────────────────────────────────

def capture_T1():
    a = jnp.linspace(0.0, 1.0, 7, dtype=jnp.float32)
    b = jnp.linspace(0.0, 1.0, 7, dtype=jnp.float32)
    logits = (jnp.arange(7 * 16, dtype=jnp.float32).reshape(7, 16) / 50.0)
    gate_weights = jax.nn.softmax(logits, axis=-1)
    return bin_op_s(a, b, gate_weights)


# ── T2: bin_op_light ─────────────────────────────────────────────────────────

def capture_T2():
    a = jnp.linspace(0.0, 1.0, 7, dtype=jnp.float32)
    b = jnp.linspace(0.0, 1.0, 7, dtype=jnp.float32)
    truth_table_logits = jnp.array([
        [-1.0, -0.5,  0.5,  1.0],
        [-2.0,  0.0,  0.0,  2.0],
        [ 0.5, -0.5,  0.5, -0.5],
        [ 1.0,  1.0, -1.0, -1.0],
        [-1.0,  0.0,  1.0,  2.0],
        [ 0.0,  0.5,  1.0,  1.5],
        [-2.0, -1.0,  1.0,  2.0],
    ], dtype=jnp.float32)
    weights = jax.nn.sigmoid(truth_table_logits)
    return bin_op_light(a, b, weights)


# ── T3: group_sum_head ───────────────────────────────────────────────────────

def capture_T3():
    x = (jnp.arange(2 * 40, dtype=jnp.float32).reshape(2, 40) / 100.0)
    return group_sum_head(x, 10, 4.0)


def capture_T3_clamp():
    x = (jnp.arange(2 * 40, dtype=jnp.float32).reshape(2, 40) / 100.0)
    # sum_tau == 0.0 should clamp to 1e-6 (scaling by ~1e6)
    return group_sum_head(x, 10, 0.0)


# ── T4a: or_pool ─────────────────────────────────────────────────────────────

def capture_T4a():
    x = jax.random.uniform(jax.random.PRNGKey(7), (2, 4, 4, 3), dtype=jnp.float32)
    return or_pool(x, kernel_size=(2, 2))


# ── T5: max_pool ─────────────────────────────────────────────────────────────

def capture_T5():
    x = jax.random.uniform(jax.random.PRNGKey(11), (2, 4, 4, 3), dtype=jnp.float32)
    return max_pool(x, kernel_size=(2, 2))


# ── T6 / T7: one JIT train step + one JIT eval batch ─────────────────────────

def _setup_T6_T7():
    rng = jax.random.PRNGKey(0)
    x = jax.random.uniform(rng, (8, 16), dtype=jnp.float32)
    y = jnp.arange(8) % 2
    init_key = jax.random.PRNGKey(1)
    params, wires = init_logic_gate_network(
        input_dim=16, num_neurons=4, num_layers=2,
        connections='random', key=init_key, logic_family='full',
    )
    tx = optax.chain(
        optax.clip(1.0),
        optax.adamw(learning_rate=0.01, b1=0.9, b2=0.99, weight_decay=1e-4),
    )
    state = TrainState(params=params, opt_state=tx.init(params), key=jax.random.PRNGKey(2))
    return x, y, params, wires, tx, state


def capture_T6():
    """Return (loss, aux_hard, params0_row0_after_step)."""
    x, y, params, wires, tx, state = _setup_T6_T7()
    step = make_train_step(tx)
    state2, loss, aux = step(
        state, x, y, wires,
        'softmax', 1.0, 1.0,
        2, 1.0, 'full',
    )
    return float(loss), float(aux['hard']), state2.params[0][0]


def capture_T7():
    """Return the 5-scalar eval dict."""
    x, y, params, wires, tx, state = _setup_T6_T7()
    metrics = eval_batch(
        state.params, wires, x, y, jax.random.PRNGKey(99),
        'softmax', 1.0, 1.0,
        2, 1.0, 'full',
    )
    return {k: float(v) for k, v in metrics.items()}


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> int:
    print('# Auto-generated — paste into tests/test_equivalence.py')
    print('# Captured on commit: $(git rev-parse HEAD)')
    print('# Re-run: python scripts/capture_equivalence_baselines.py')
    print()

    print('# T1')
    t1 = capture_T1()
    print(fmt_array(t1, 'T1'))
    print()

    print('# T2')
    t2 = capture_T2()
    print(fmt_array(t2, 'T2'))
    print()

    print('# T3')
    t3 = capture_T3()
    print(fmt_array(t3, 'T3'))
    print()

    print('# T3 clamp probe (sum_tau == 0.0 → floors to 1e-6 → output ~1e6 scale)')
    t3c = capture_T3_clamp()
    print(f'# T3 clamp max abs = {float(jnp.max(jnp.abs(t3c))):.6e}')
    print(f'# T3 clamp is_finite = {bool(jnp.all(jnp.isfinite(t3c)))}')
    print()

    print('# T4a')
    t4a = capture_T4a()
    print(fmt_array(t4a, 'T4a'))
    print()

    print('# T5')
    t5 = capture_T5()
    print(fmt_array(t5, 'T5'))
    print()

    print('# T6: one JIT train step')
    loss, hard, p0 = capture_T6()
    # repr for floats so the literal round-trips bit-exactly when read back.
    print(f'EXPECTED_T6_LOSS = {loss!r}')
    print(f'EXPECTED_T6_HARD = {hard!r}')
    print(fmt_array(p0, 'T6_PARAMS0_ROW0'))
    print()

    print('# T7: one JIT eval batch')
    m = capture_T7()
    print('EXPECTED_T7 = {')
    for k, v in m.items():
        print(f'    {k!r}: {v!r},')
    print('}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
