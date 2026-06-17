# Source: ported from Older_Imp.ipynb cell 6
# Gate inspection helpers. gate_analysis is included — it stays DLGN-only here.
from __future__ import annotations

import jax
import numpy as np

from dlgn.models.gates import (
    GATE_NAMES,
    NUMBER_OF_GATES,
    hard_gate_ids_from_logits,
)
from dlgn.models.decoders import (
    logic_family_of_logits,
    decode_light_soft,
    decode_light_hard,
)
from dlgn.types import WirePair


def logic_argmax_row(layer_logits: jax.Array) -> np.ndarray:
    """Return hard gate IDs for one layer as a NumPy int32 array."""
    return np.asarray(jax.device_get(hard_gate_ids_from_logits(layer_logits)), dtype=np.int32)


def gate_usage_summary(params: list[jax.Array]) -> list[tuple[str, int]]:
    """Aggregate hard gate counts across all layers."""
    import jax.numpy as jnp
    gate_ids = jnp.concatenate([hard_gate_ids_from_logits(layer).reshape(-1) for layer in params])
    counts = jnp.bincount(gate_ids, length=NUMBER_OF_GATES)
    return [(GATE_NAMES[i], int(counts[i])) for i in range(NUMBER_OF_GATES) if int(counts[i]) > 0]


def print_gate_usage(params: list[jax.Array]) -> None:
    """Print a human-readable gate usage table."""
    print('Gate usage after hard decoding:')
    for gate_name, count in gate_usage_summary(params):
        print(f'  {gate_name:>14}: {count}')


def show_logiclayer_info(
    layer,
    wires: WirePair | None = None,
    n: int | None = None,
    topk: int = 1,
    layer_name: str | None = None,
) -> None:
    """Print per-neuron gate info for one layer."""
    if wires is None:
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise ValueError(
                'Pass either (layer_logits, layer_wires) as a tuple or supply wires explicitly'
            )
        layer_logits, layer_wires = layer
    else:
        layer_logits, layer_wires = layer, wires

    W = np.asarray(jax.device_get(layer_logits))
    out_dim = W.shape[0]
    rows = out_dim if n is None else min(n, out_dim)
    prefix = '' if layer_name is None else f'{layer_name} '

    a_idx = np.asarray(jax.device_get(layer_wires[0]))
    b_idx = np.asarray(jax.device_get(layer_wires[1]))

    if logic_family_of_logits(layer_logits) == 'light':
        soft_table = np.asarray(jax.device_get(decode_light_soft(layer_logits)))
        hard_table = np.asarray(jax.device_get(decode_light_hard(layer_logits))).astype(np.int32)
        gate_ids = np.asarray(jax.device_get(hard_gate_ids_from_logits(layer_logits))).astype(np.int32)
        vals, counts = np.unique(gate_ids, return_counts=True)
        print(f"{prefix}op histogram:", ' '.join(f"{GATE_NAMES[int(v)]}:{int(c)}" for v, c in zip(vals, counts)))
        for i in range(rows):
            pair_str = f' pair=({int(a_idx[i])},{int(b_idx[i])})'
            gate_id = int(gate_ids[i])
            print(
                f'  neuron {i:3d}:{pair_str} '
                f'gate={gate_id}:{GATE_NAMES[gate_id]} '
                f'truth_table={hard_table[i].tolist()} '
                f'probs={[float(x) for x in soft_table[i].tolist()]}'
            )
        return

    topk = max(1, min(topk, W.shape[1]))
    topi = np.argsort(W, axis=1)[:, -topk:][:, ::-1]
    topv = np.take_along_axis(W, topi, axis=1)
    vals, counts = np.unique(W.argmax(axis=1), return_counts=True)
    print(f"{prefix}op histogram:", ' '.join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts)))
    for i in range(rows):
        pair_str = f' pair=({int(a_idx[i])},{int(b_idx[i])})'
        print(
            f'  neuron {i:3d}:{pair_str} '
            f'ops={topi[i].tolist()} '
            f'logits={[float(x) for x in topv[i].tolist()]}'
        )


def gate_analysis(
    train_state,
    wires: list[WirePair],
    name: str = 'model',
    topk: int = 1,
    exclude_passthrough: bool = False,
) -> None:
    """Print per-layer gate info and optionally save gate histograms."""
    from .plots import plot_hist_gates

    params = train_state.params
    all_gate_ids = []

    for layer_idx, (layer_logits, layer_wires) in enumerate(zip(params, wires)):
        gate_ids = logic_argmax_row(layer_logits)
        all_gate_ids.append(gate_ids)

        show_logiclayer_info(
            layer_logits,
            layer_wires,
            topk=topk,
            layer_name=f'layer {layer_idx}',
        )

        plot_ids = gate_ids
        if exclude_passthrough:
            plot_ids = plot_ids[(plot_ids != 3) & (plot_ids != 5)]

        if plot_ids.size > 0:
            save_path = None if not name else f'{name}_layer_{layer_idx}_gates.svg'
            plot_hist_gates(
                plot_ids,
                save_path=save_path,
                title=f'{name} Layer {layer_idx} Gate Distribution',
            )

    flat_gates = np.concatenate(all_gate_ids, axis=0)
    if exclude_passthrough:
        flat_gates = flat_gates[(flat_gates != 3) & (flat_gates != 5)]

    if flat_gates.size > 0:
        save_path = None if not name else f'{name}_all_gates.svg'
        plot_hist_gates(
            flat_gates, save_path=save_path, title=f'{name} All Layers Gate Distribution'
        )
