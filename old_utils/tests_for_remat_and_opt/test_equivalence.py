"""Equivalence tests for DLGN_Only_Lean.

See ../dlgn_equivalence_tests.md for the test plan and the D-N decision
each test is tied to. Literal baselines were captured by
``scripts/capture_equivalence_baselines.py`` on the Phase-3 baseline
commit. Do not edit literals by hand; re-run the capture script if a
Track-A decision intentionally rewrites a test.

Run with:
    cd DLGN_Only_Lean
    pytest tests/test_equivalence.py -v
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
import pytest


# ── Package surface under test ──────────────────────────────────────────────
from dlgn.models.gates import bin_op_s, bin_op_light
from dlgn.models.heads import group_sum_head
from dlgn.models.conv import or_pool, max_pool
from dlgn.models.initialization import init_logic_gate_network
from dlgn.training.state import TrainState
from dlgn.training.steps import make_train_step, eval_batch, evaluate_loader
from dlgn.config import validate_logic_config
from dlgn.data.loaders import input_dim_of_dataset, num_classes_of_dataset


# ── Baseline literals captured by capture_equivalence_baselines.py ──────────

EXPECTED_T1 = jnp.array(
    [0.53991485, 0.53202248, 0.52495939, 0.51872563, 0.51332104, 0.50874579,
     0.50499976],
    dtype=jnp.float32,
)

EXPECTED_T2 = jnp.array(
    [0.26894143, 0.24613528, 0.54081976, 0.50000000, 0.69491631, 0.76963758,
     0.88079703],
    dtype=jnp.float32,
)

EXPECTED_T3 = jnp.array(
    [[0.014999999, 0.055000000, 0.094999999, 0.135000005, 0.174999997, 0.215000004,
      0.254999995, 0.295000017, 0.334999979, 0.375000000],
     [0.415000021, 0.454999983, 0.495000005, 0.534999967, 0.574999988, 0.615000010,
      0.654999971, 0.694999993, 0.735000014, 0.774999976]],
    dtype=jnp.float32,
)

EXPECTED_T4a = jnp.array(
    [[[[0.97212058, 0.99606335, 0.99580407],
       [0.80805475, 0.99386376, 0.99692672]],

      [[0.90638709, 0.83197212, 0.98992372],
       [0.99514377, 0.66480494, 0.99948257]]],


     [[[0.86300087, 0.98381376, 0.96794939],
       [0.96165270, 0.99251699, 0.98512995]],

      [[0.89519018, 0.99978203, 0.99886847],
       [0.95362824, 0.99759692, 0.98648667]]]],
    dtype=jnp.float32,
)

EXPECTED_T5 = jnp.array(
    [[[[0.91248631, 0.90480244, 0.72743881],
       [0.99391675, 0.92408872, 0.55310678]],

      [[0.60078478, 0.57393026, 0.91193676],
       [0.90335631, 0.96149886, 0.94273579]]],


     [[[0.69350958, 0.76120758, 0.84286225],
       [0.81209850, 0.96771276, 0.76254761]],

      [[0.81029439, 0.77285099, 0.99235904],
       [0.67652988, 0.53424609, 0.62664378]]]],
    dtype=jnp.float32,
)

EXPECTED_T6_LOSS = 0.5705485343933105
EXPECTED_T6_HARD = 0.570489764213562
EXPECTED_T6_PARAMS0_ROW0 = jnp.array(
    [-9.9852365e-03, -9.9377371e-03, -9.9806795e-03,  1.0009989e+01,
     -9.9877995e-03, -9.9669881e-03, -9.9848444e-03, -9.9300900e-03,
     -9.9857561e-03, -9.9460287e-03, -9.9815577e-03, -9.6058222e-03,
     -9.9881552e-03, -9.9694747e-03, -9.9853892e-03, -9.9403746e-03],
    dtype=jnp.float32,
)

EXPECTED_T7 = {
    'count': 8.0,
    'hard_correct': 5.0,
    'hard_loss_sum': 4.563918113708496,
    'soft_correct': 5.0,
    'soft_loss_sum': 4.564388275146484,
}


# ── T0. PREFLIGHT smoke imports (D18 regression test) ────────────────────────


def test_T0_smoke_imports():
    """All entry-point modules must import cleanly post-D18."""
    import dlgn  # noqa
    import dlgn.config  # noqa
    import dlgn.utils  # noqa
    import dlgn.data.loaders  # noqa
    import dlgn.training.loop  # noqa
    import dlgn.training.steps  # noqa
    import dlgn.training.checkpoints  # noqa
    import dlgn.models.network  # noqa
    import dlgn.models.conv  # noqa
    import dlgn.cli.train  # noqa
    import dlgn.cli.resume  # noqa
    import dlgn.cli.inspect_checkpoint  # noqa
    from dlgn.utils import seed_all, batch_to_jax, make_run_name, make_run_dir, tree_to_numpy, tree_to_jax  # noqa
    from dlgn.data.loaders import (
        SUPPORTED_DATASETS, input_dim_of_dataset, num_classes_of_dataset,
        load_dataset, load_cifar_threshold, cycle_loader, ArrayDataLoader,
    )  # noqa
    # Defensive: seed_all is NOT importable from data.loaders
    with pytest.raises(ImportError):
        from dlgn.data.loaders import seed_all  # noqa


# ── T1. bin_op_s pinned output ───────────────────────────────────────────────


def test_T1_bin_op_s():
    a = jnp.linspace(0.0, 1.0, 7, dtype=jnp.float32)
    b = jnp.linspace(0.0, 1.0, 7, dtype=jnp.float32)
    logits = (jnp.arange(7 * 16, dtype=jnp.float32).reshape(7, 16) / 50.0)
    gate_weights = jax.nn.softmax(logits, axis=-1)
    out = bin_op_s(a, b, gate_weights)
    assert out.shape == EXPECTED_T1.shape
    assert jnp.allclose(out, EXPECTED_T1, atol=1e-7, rtol=0), \
        f'bin_op_s mismatch: {out} vs {EXPECTED_T1}'


# ── T2. bin_op_light pinned output ───────────────────────────────────────────


def test_T2_bin_op_light():
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
    out = bin_op_light(a, b, weights)
    assert jnp.allclose(out, EXPECTED_T2, atol=1e-7, rtol=0), \
        f'bin_op_light mismatch: {out} vs {EXPECTED_T2}'


# ── T3. group_sum_head pinned output + clamp behavior ────────────────────────


def test_T3_group_sum_head():
    x = (jnp.arange(2 * 40, dtype=jnp.float32).reshape(2, 40) / 100.0)
    out = group_sum_head(x, 10, 4.0)
    assert out.shape == EXPECTED_T3.shape
    assert jnp.allclose(out, EXPECTED_T3, atol=1e-7, rtol=0)


def test_T3_group_sum_head_sum_tau_zero_clamp():
    """sum_tau == 0 floors to 1e-6 → output ~1e6 scaled (benchmark protocol §5)."""
    x = (jnp.arange(2 * 40, dtype=jnp.float32).reshape(2, 40) / 100.0)
    out = group_sum_head(x, 10, 0.0)
    assert jnp.all(jnp.isfinite(out))
    # max should be ~3.1e6 — definitely > 1e5
    assert float(jnp.max(jnp.abs(out))) > 1e5


# ── T4a. or_pool pinned output (post-D1/D21 signature) ───────────────────────


def test_T4a_or_pool():
    x = jax.random.uniform(jax.random.PRNGKey(7), (2, 4, 4, 3), dtype=jnp.float32)
    out = or_pool(x, kernel_size=(2, 2))
    assert out.shape == EXPECTED_T4a.shape
    assert jnp.allclose(out, EXPECTED_T4a, atol=1e-7, rtol=0)


# ── T5. max_pool pinned output (post-D2/D22 signature) ───────────────────────


def test_T5_max_pool():
    x = jax.random.uniform(jax.random.PRNGKey(11), (2, 4, 4, 3), dtype=jnp.float32)
    out = max_pool(x, kernel_size=(2, 2))
    assert out.shape == EXPECTED_T5.shape
    assert jnp.allclose(out, EXPECTED_T5, atol=1e-7, rtol=0)


# ── T6 / T7. One JIT train step + one JIT eval batch on a tiny network ──────


def _setup_tiny_network():
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


def test_T6_one_jit_train_step():
    x, y, params, wires, tx, state = _setup_tiny_network()
    step = make_train_step(tx)
    state2, loss, aux = step(
        state, x, y, wires,
        'softmax', 1.0, 1.0,
        2, 1.0, 'full',
    )
    assert np.isclose(float(loss), EXPECTED_T6_LOSS, atol=1e-5), \
        f'loss drift: {float(loss)} vs {EXPECTED_T6_LOSS}'
    assert np.isclose(float(aux['hard']), EXPECTED_T6_HARD, atol=1e-5), \
        f'hard drift: {float(aux["hard"])} vs {EXPECTED_T6_HARD}'
    row0 = state2.params[0][0]
    assert jnp.allclose(row0, EXPECTED_T6_PARAMS0_ROW0, atol=1e-5), \
        f'param drift: {row0} vs {EXPECTED_T6_PARAMS0_ROW0}'


def test_T7_one_jit_eval_batch():
    x, y, params, wires, tx, state = _setup_tiny_network()
    metrics = eval_batch(
        state.params, wires, x, y, jax.random.PRNGKey(99),
        'softmax', 1.0, 1.0,
        2, 1.0, 'full',
    )
    for k, expected_v in EXPECTED_T7.items():
        actual_v = float(metrics[k])
        assert np.isclose(actual_v, expected_v, atol=1e-6), \
            f'eval_batch[{k!r}] drift: {actual_v} vs {expected_v}'


# ── T8. Validator accepts all dispatcher-reachable architectures (post-D3) ──


@pytest.mark.parametrize('family,arch', [
    ('light', 'light_sin01'),
    ('light', 'light_sin01_st'),
    ('light', 'light_linear'),
    ('light', 'light_linear_st'),
    ('full',  'full_st'),
    ('full',  'softmax_st'),
])
def test_T8_validator_accepts_dispatcher_reachable(family, arch):
    """POST-D3: validator accepts every name decode_training_gates dispatches."""
    # Should not raise.
    validate_logic_config({'logic_family': family, 'architecture': arch})


@pytest.mark.parametrize('family,arch', [
    ('light', 'light_sin01'),
    ('light', 'light_sin01_st'),
    ('light', 'light_linear'),
    ('light', 'light_linear_st'),
    ('full',  'full_st'),
    ('full',  'softmax_st'),
])
def test_T8_post_D3_forward_pass_is_finite(family, arch):
    """POST-D3: a tiny forward through each newly-accepted architecture is finite."""
    from dlgn.models.network import forward_logits
    from dlgn.models.initialization import init_logic_gate_network
    init_key = jax.random.PRNGKey(0)
    params, wires = init_logic_gate_network(
        input_dim=8, num_neurons=4, num_layers=2,
        connections='random', key=init_key, logic_family=family,
    )
    x = jax.random.uniform(jax.random.PRNGKey(1), (3, 8), dtype=jnp.float32)
    logits = forward_logits(
        params, wires, x, True, jax.random.PRNGKey(2),
        arch, 1.0, 1.0, class_count=2, sum_tau=1.0, logic_family=family,
    )
    assert logits.shape == (3, 2)
    assert bool(jnp.all(jnp.isfinite(logits))), \
        f'{family}/{arch} forward not finite: {logits}'


@pytest.mark.parametrize('family,arch', [
    ('full', 'unknown_arch'),
    ('light', 'unknown_light'),
])
def test_T8_validator_still_rejects_unknown(family, arch):
    """POST-D3: validator still rejects names the dispatcher doesn't reach."""
    with pytest.raises(ValueError):
        validate_logic_config({'logic_family': family, 'architecture': arch})


# ── T9. input_dim_of_dataset — post-D15 form ────────────────────────────────


def test_T9_input_dim_post_D15_unchanged_for_non_cifar():
    """POST-D15: non-CIFAR datasets unchanged. threshold_bits silently ignored."""
    assert input_dim_of_dataset('mnist') == 784
    assert input_dim_of_dataset('mnist20x20') == 400
    assert input_dim_of_dataset('mnist_bin') == 784
    assert input_dim_of_dataset('adult') == 116
    # threshold_bits silently ignored for non-cifar names
    assert input_dim_of_dataset('mnist', threshold_bits=5) == 784


def test_T9_input_dim_post_D15_static_cifar_unchanged():
    """POST-D15: cifar-10-N-thresholds without override matches static entries."""
    assert input_dim_of_dataset('cifar-10-3-thresholds') == 9216
    assert input_dim_of_dataset('cifar-10-31-thresholds') == 95232


def test_T9_input_dim_post_D15_threshold_bits_override_wins():
    """POST-D15: threshold_bits override wins for cifar family."""
    assert input_dim_of_dataset('cifar10', threshold_bits=1) == 3 * 32 * 32 * 1
    assert input_dim_of_dataset('cifar10', threshold_bits=2) == 3 * 32 * 32 * 2
    assert input_dim_of_dataset('cifar10', threshold_bits=5) == 15360
    # Override beats the static entry too.
    assert input_dim_of_dataset('cifar-10-3-thresholds', threshold_bits=5) == 15360
    assert input_dim_of_dataset('cifar-10-31-thresholds', threshold_bits=5) == 15360


def test_T9_input_dim_post_D15_bare_cifar10_requires_threshold_bits():
    """POST-D15: bare cifar10 with no override raises loudly."""
    with pytest.raises(ValueError, match='threshold_bits'):
        input_dim_of_dataset('cifar10')
    with pytest.raises(ValueError, match='threshold_bits'):
        input_dim_of_dataset('cifar10', threshold_bits=None)


def test_T9_input_dim_post_D15_num_classes_unchanged():
    """POST-D15: num_classes_of_dataset behavior is unchanged."""
    assert num_classes_of_dataset('cifar10') == 10
    assert num_classes_of_dataset('cifar-10-3-thresholds') == 10
    assert num_classes_of_dataset('mnist') == 10


# ── T10. DEFAULT_CONFIG / build_config (post-D5) ─────────────────────────────


def test_T10_default_config_removed_post_D5():
    """POST-D5: importing DEFAULT_CONFIG or build_config raises ImportError."""
    with pytest.raises(ImportError):
        from dlgn.config import DEFAULT_CONFIG  # noqa
    with pytest.raises(ImportError):
        from dlgn.config import build_config  # noqa


def test_T10_post_D5_other_config_helpers_preserved():
    """POST-D5: the helpers that the CLI does use are still present."""
    from dlgn.config import (
        validate_logic_config, save_config_snapshot, load_config_snapshot,
    )
    assert callable(validate_logic_config)
    assert callable(save_config_snapshot)
    assert callable(load_config_snapshot)


# ── T11 / T12. Pool comparison (post-D6 — --pool-type flag) ─────────────────


def test_T11_pool_post_D6_flag_and_imports():
    """POST-D6: both pools are importable; --pool-type flag exists; default 'or'."""
    primary = Path(__file__).resolve().parents[1] / 'scripts' / 'test_layers.py'
    text = primary.read_text()
    # Both pools imported in the conv forward
    assert 'from dlgn.models.conv import run_conv_gate_layer, or_pool, max_pool' in text
    # CLI flag exists with both choices and 'or' default
    assert "'--pool-type'" in text
    assert "choices=['or', 'max']" in text
    assert "default='or'" in text  # preserves baseline


def test_T12_pool_or_and_max_produce_different_outputs():
    """POST-D6: deep_conv_forward with pool_type='or' vs 'max' gives different logits."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from importlib import import_module
    # The script is named with a dash-free import path
    mod = import_module('test_layers')
    from dlgn.models.conv import init_conv_gate_layer
    from dlgn.models.initialization import init_gate_layer

    # Tiny conv config
    key = jax.random.PRNGKey(0)
    key, ck = jax.random.split(key)
    conv_params, conv_wires = init_conv_gate_layer(
        ck, in_channels=1, out_channels=4, kernel_size=(3, 3),
        depth=2, connection_type='random', logic_family='full',
    )
    images = jax.random.uniform(jax.random.PRNGKey(11), (2, 8, 8, 1), dtype=jnp.float32)

    # Probe to get flatten dim under each pool
    from dlgn.models.conv import run_conv_gate_layer, or_pool, max_pool
    x_probe = run_conv_gate_layer(
        conv_params, conv_wires, images, False, jax.random.PRNGKey(2),
        kernel_size=(3, 3), stride=(1, 1),
        architecture='softmax', logic_family='full',
    )
    x_or = or_pool(x_probe, kernel_size=(2, 2), stride=(2, 2))
    x_max = max_pool(x_probe, kernel_size=(2, 2), stride=(2, 2))
    flatten_dim = int(np.prod(x_or.shape[1:]))

    # FC stack so the GroupSum head is well-defined
    key, fck = jax.random.split(key)
    fc_params, fc_wires = init_gate_layer(
        fck, in_dim=flatten_dim, out_dim=20,
        connection_type='random', logic_family='full',
    )
    params = {'conv': [conv_params], 'fc': [fc_params]}
    wires = {'conv': [conv_wires], 'fc': [fc_wires]}

    cfg = mod.HeadConfig(architecture='softmax', logic_family='full',
                         class_count=10, sum_tau=1.0)

    common = dict(
        block_kernels=[3], block_strides=[1], block_paddings=[0],
        pool_size=2, pool_stride=2,
    )
    key, fwd_key = jax.random.split(key)
    logits_or = mod.deep_conv_forward(
        params, wires, images, True, fwd_key, cfg, pool_type='or', **common,
    )
    logits_max = mod.deep_conv_forward(
        params, wires, images, True, fwd_key, cfg, pool_type='max', **common,
    )
    # Both produce finite, same-shape outputs
    assert logits_or.shape == logits_max.shape == (2, 10)
    assert bool(jnp.all(jnp.isfinite(logits_or)))
    assert bool(jnp.all(jnp.isfinite(logits_max)))
    # Outputs disagree (different pool semantics)
    assert not bool(jnp.allclose(logits_or, logits_max)), \
        'OR and MAX pool produced identical logits — pool_type branch likely broken'


def test_T12b_pool_type_invalid_raises():
    """POST-D6: deep_conv_forward rejects unknown pool_type."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from importlib import import_module
    mod = import_module('test_layers')
    with pytest.raises(ValueError, match="pool_type"):
        mod.deep_conv_forward(
            {'conv': [], 'fc': []}, {'conv': [], 'fc': []},
            jnp.zeros((1, 8, 8, 1)), True, jax.random.PRNGKey(0),
            mod.HeadConfig(class_count=10),
            block_kernels=[], block_strides=[], block_paddings=[],
            pool_size=2, pool_stride=2, pool_type='bogus',
        )


# ── T13. Conv divisibility guard raises (post-D8) ────────────────────────────


def test_T13_conv_divisibility_raises_post_D8_text():
    """POST-D8: the guard raises ValueError instead of returning None."""
    primary = Path(__file__).resolve().parents[1] / 'scripts' / 'test_layers.py'
    text = primary.read_text()
    assert 'if fc_sizes[-1] % class_count != 0:' in text
    guard_section = text.split('if fc_sizes[-1] % class_count != 0:')[1][:300]
    assert 'raise ValueError' in guard_section
    assert 'return None' not in guard_section


def test_T13_conv_divisibility_runtime_raises_post_D8():
    """POST-D8: invoking run_deep_conv_path with a bad fc_sizes raises."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from importlib import import_module
    mod = import_module('test_layers')

    class Args:
        seed = 0
        logic_family = 'full'
        architecture = 'softmax'
        sum_tau = 1.0
        gumb_tau = 1.0
        dirichlet_concentration = 1.0
        # 1 conv block, then fc_sizes ending in 7 which doesn't divide 10
        conv_block_channels = [8]
        conv_block_kernel_sizes = [3]
        conv_block_depths = [2]
        conv_block_paddings = [0]
        conv_block_strides = [1]
        pool_size = 2
        pool_stride = 2
        pool_type = 'or'
        fc_sizes = [7]  # 7 % 10 != 0 → must raise
        learning_rate = 0.01
        weight_decay = 1e-4
        clip_value = 1.0
        eval_every = 1
        train_steps = 1
        dataset = 'mnist'
        threshold_bits = None

    class FakeLoader:
        def __iter__(self):
            # one MNIST-shaped batch — only need shape, not real data
            yield (np.zeros((4, 1, 28, 28), dtype=np.float32),
                   np.zeros(4, dtype=np.int64))

    with pytest.raises(ValueError, match='divisible'):
        mod.run_deep_conv_path(Args(), FakeLoader(), FakeLoader(), class_count=10)


# ── T14. evaluate_loader returns the 4 base keys; no soft_hard_gap yet ───────


def test_T14_evaluate_loader_keys_post_b_perf_9():
    """POST-B-perf-9: evaluate_loader returns 5 keys; soft_hard_gap = soft_acc - hard_acc."""
    from dlgn.training.steps import evaluate_loader
    x_batch = np.linspace(0.0, 1.0, 8 * 16).reshape(8, 16).astype(np.float32)
    y_batch = (np.arange(8) % 2).astype(np.int32)
    class FakeLoader:
        def __iter__(self):
            yield (x_batch, y_batch)
    _, _, params, wires, _, state = _setup_tiny_network()
    config = {
        'seed': 0, 'architecture': 'softmax', 'gumb_tau': 1.0,
        'dirichlet_concentration': 1.0, 'sum_tau': 1.0, 'logic_family': 'full',
    }
    metrics = evaluate_loader(state.params, wires, FakeLoader(), config, class_count=2)
    assert set(metrics.keys()) == {
        'soft_loss', 'hard_loss', 'soft_acc', 'hard_acc', 'soft_hard_gap',
    }
    # New key is exact arithmetic of two existing keys.
    assert np.isclose(metrics['soft_hard_gap'],
                      metrics['soft_acc'] - metrics['hard_acc'],
                      atol=1e-12)


def test_T14b_conv_evaluate_has_soft_hard_gap():
    """POST-B-perf-9: deep_conv_evaluate and evaluate also expose soft_hard_gap."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from importlib import import_module
    mod = import_module('test_layers')
    # evaluate() (regular path)
    _, _, params, wires, _, state = _setup_tiny_network()
    cfg = mod.HeadConfig(architecture='softmax', logic_family='full',
                         class_count=2, sum_tau=1.0)
    x_batch = jnp.asarray(np.linspace(0.0, 1.0, 8 * 16).reshape(8, 16).astype(np.float32))
    y_batch = jnp.asarray((np.arange(8) % 2).astype(np.int32))
    out = mod.evaluate(state.params, wires, x_batch, y_batch,
                       jax.random.PRNGKey(0), cfg)
    assert 'soft_hard_gap' in out
    assert np.isclose(out['soft_hard_gap'], out['soft_acc'] - out['hard_acc'],
                      atol=1e-12)


# ── T17. Remat is numerics-preserving (D25) ─────────────────────────────────


@pytest.mark.parametrize('remat_conv,remat_fc,policy_str', [
    (False, False, 'dots'),    # off — reference
    (True,  False, 'dots'),    # conv only
    (False, True,  'dots'),    # fc only
    (True,  True,  'dots'),    # both
    (True,  True,  'nothing'), # both, aggressive policy
])
def test_T17_remat_modes_produce_identical_logits(remat_conv, remat_fc, policy_str):
    """D25: jax.checkpoint is numerics-preserving. All remat modes match remat=off."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from importlib import import_module
    mod = import_module('test_layers')
    from dlgn.models.conv import init_conv_gate_layer
    from dlgn.models.initialization import init_gate_layer
    from dlgn.models.conv import run_conv_gate_layer, or_pool

    # Tiny 1-block conv config — small enough to run fast, big enough that
    # remat would actually do something on a real model.
    key = jax.random.PRNGKey(0)
    key, ck = jax.random.split(key)
    conv_params, conv_wires = init_conv_gate_layer(
        ck, in_channels=1, out_channels=4, kernel_size=(3, 3),
        depth=2, connection_type='random', logic_family='full',
    )
    images = jax.random.uniform(jax.random.PRNGKey(11), (2, 8, 8, 1), dtype=jnp.float32)

    # Probe for flatten_dim
    x_probe = run_conv_gate_layer(
        conv_params, conv_wires, images, False, jax.random.PRNGKey(2),
        kernel_size=(3, 3), stride=(1, 1),
        architecture='softmax', logic_family='full',
    )
    x_pooled = or_pool(x_probe, kernel_size=(2, 2), stride=(2, 2))
    flatten_dim = int(np.prod(x_pooled.shape[1:]))

    key, fck = jax.random.split(key)
    fc_params, fc_wires = init_gate_layer(
        fck, in_dim=flatten_dim, out_dim=20,
        connection_type='random', logic_family='full',
    )
    params = {'conv': [conv_params], 'fc': [fc_params]}
    wires = {'conv': [conv_wires], 'fc': [fc_wires]}
    cfg = mod.HeadConfig(architecture='softmax', logic_family='full',
                         class_count=10, sum_tau=1.0)

    common = dict(
        block_kernels=[3], block_strides=[1], block_paddings=[0],
        pool_size=2, pool_stride=2, pool_type='or',
    )
    key, fwd_key = jax.random.split(key)

    # Reference: remat=off
    logits_ref = mod.deep_conv_forward(
        params, wires, images, True, fwd_key, cfg,
        remat_conv=False, remat_fc=False, remat_policy=None,
        **common,
    )

    # Mode under test
    policy = mod.REMAT_POLICIES[policy_str] if (remat_conv or remat_fc) else None
    logits_test = mod.deep_conv_forward(
        params, wires, images, True, fwd_key, cfg,
        remat_conv=remat_conv, remat_fc=remat_fc, remat_policy=policy,
        **common,
    )

    # jax.checkpoint is mathematically equivalent but XLA may re-fuse ops
    # inside the checkpointed region, producing float32 LSB drift (~1e-6).
    # Use allclose with a tight tolerance — anything looser means a real
    # numerical bug, not round-off.
    max_abs_diff = float(jnp.max(jnp.abs(logits_ref - logits_test)))
    assert bool(jnp.allclose(logits_ref, logits_test, atol=1e-5, rtol=0)), \
        f'remat(conv={remat_conv}, fc={remat_fc}, policy={policy_str}) ' \
        f'changed logits beyond round-off: max abs diff = {max_abs_diff}'


def test_T17_remat_modes_grad_identical():
    """D25: gradients under jax.checkpoint match the un-checkpointed gradients."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from importlib import import_module
    mod = import_module('test_layers')
    from dlgn.models.conv import init_conv_gate_layer
    from dlgn.models.initialization import init_gate_layer

    key = jax.random.PRNGKey(0)
    key, ck = jax.random.split(key)
    conv_params, conv_wires = init_conv_gate_layer(
        ck, in_channels=1, out_channels=4, kernel_size=(3, 3),
        depth=2, connection_type='random', logic_family='full',
    )
    images = jax.random.uniform(jax.random.PRNGKey(11), (2, 8, 8, 1), dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    # Probe + init FC
    from dlgn.models.conv import run_conv_gate_layer, or_pool
    x_probe = run_conv_gate_layer(
        conv_params, conv_wires, images, False, jax.random.PRNGKey(2),
        kernel_size=(3, 3), stride=(1, 1),
        architecture='softmax', logic_family='full',
    )
    flatten_dim = int(np.prod(or_pool(x_probe, (2, 2), (2, 2)).shape[1:]))
    key, fck = jax.random.split(key)
    fc_params, fc_wires = init_gate_layer(
        fck, in_dim=flatten_dim, out_dim=20,
        connection_type='random', logic_family='full',
    )
    params = {'conv': [conv_params], 'fc': [fc_params]}
    wires = {'conv': [conv_wires], 'fc': [fc_wires]}
    cfg = mod.HeadConfig(architecture='softmax', logic_family='full',
                         class_count=10, sum_tau=1.0)
    common = dict(
        block_kernels=[3], block_strides=[1], block_paddings=[0],
        pool_size=2, pool_stride=2, pool_type='or',
    )
    key, fwd_key = jax.random.split(key)

    def loss_fn(p, remat_conv, remat_fc, policy):
        logits = mod.deep_conv_forward(
            p, wires, images, True, fwd_key, cfg,
            remat_conv=remat_conv, remat_fc=remat_fc, remat_policy=policy,
            **common,
        )
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grad_ref = jax.grad(loss_fn)(params, False, False, None)
    grad_both = jax.grad(loss_fn)(params, True, True,
                                  mod.REMAT_POLICIES['dots'])
    # The gradient pytree mirrors params: dict of lists of (nested) arrays.
    # Walk it leaf-by-leaf and check each leaf matches within round-off.
    abs_diffs = jax.tree_util.tree_map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))),
        grad_ref, grad_both,
    )
    flat_diffs = jax.tree_util.tree_leaves(abs_diffs)
    max_overall = max(flat_diffs)
    assert max_overall < 1e-5, \
        f'remat=both changed gradients beyond round-off: max abs diff = {max_overall}'


# ── T16. setup_jit_cache (D24) ──────────────────────────────────────────────


def test_T16_setup_jit_cache_enabled():
    """D24: setup_jit_cache(enabled=True) creates the dir and sets JAX config."""
    import tempfile
    from dlgn.utils import setup_jit_cache
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / 'subdir' / 'cache'
        result = setup_jit_cache(cache_dir, enabled=True)
        assert result == str(cache_dir.resolve())
        assert cache_dir.is_dir()
        # JAX config should now point at this dir.
        assert jax.config.values.get('jax_compilation_cache_dir') == str(cache_dir.resolve())


def test_T16_setup_jit_cache_disabled_noop():
    """D24: setup_jit_cache(enabled=False) returns None and creates nothing."""
    import tempfile
    from dlgn.utils import setup_jit_cache
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / 'should_not_exist'
        result = setup_jit_cache(cache_dir, enabled=False)
        assert result is None
        assert not cache_dir.exists()


def test_T16_setup_jit_cache_idempotent():
    """D24: calling setup_jit_cache twice is safe."""
    import tempfile
    from dlgn.utils import setup_jit_cache
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / 'cache'
        a = setup_jit_cache(cache_dir, enabled=True)
        b = setup_jit_cache(cache_dir, enabled=True)
        assert a == b == str(cache_dir.resolve())


# ── T15. JIT cache size — eval_batch is reused, not re-traced (diagnostic) ──


def test_T15_eval_batch_jit_cache_size():
    x, y, params, wires, tx, state = _setup_tiny_network()
    # Two calls with identical static args + same shapes → cache size should stay 1.
    _ = eval_batch(state.params, wires, x, y, jax.random.PRNGKey(0),
                   'softmax', 1.0, 1.0, 2, 1.0, 'full')
    _ = eval_batch(state.params, wires, x, y, jax.random.PRNGKey(1),
                   'softmax', 1.0, 1.0, 2, 1.0, 'full')
    # eval_batch._cache_size() is the JIT cache count for this concrete signature.
    # In JAX 0.4+ the attribute is _cache_size on the PjitFunction.
    cache_size = eval_batch._cache_size()
    assert cache_size == 1, f'eval_batch retraced: cache_size={cache_size}'
