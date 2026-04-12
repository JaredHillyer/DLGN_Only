"""Convolutional Differentiable Logic Gate Network (DLGN) layers in JAX.

Pure-functional style (no Flax modules). Init functions return (params, wires),
run functions consume them — same pattern as the DLCA notebook.

Based on:
- Petersen et al., "Convolutional Differentiable Logic Gate Networks" (arXiv:2411.04732)
- DLCA notebook: google-research/self-organising-systems diffLogic_CA
"""

import jax
import jax.numpy as jnp
from jax import lax

from dlgn.models.initialization import init_gate_layer
from dlgn.models.network import run_layer
from dlgn.types import ConnectionType, LogicFamily


# ---------------------------------------------------------------------------
# Initialization: one conv layer = out_channels independent gate trees
# ---------------------------------------------------------------------------

def init_conv_gate_layer(
    key,
    in_channels,
    out_channels,
    kernel_size=(3, 3),
    depth=2,
    connection_type: ConnectionType = 'random',
    logic_family: LogicFamily = 'full',
):
    """Initialize a single convolutional DLGN layer.

    Each of the `out_channels` output channels gets its own binary gate tree
    of depth `depth`, wired into a flattened patch of size
    in_channels * kh * kw.

    Args:
        key: PRNG key.
        in_channels: number of input channels (C).
        out_channels: number of output channels / tree kernels (n).
        kernel_size: spatial receptive field (kh, kw).
        depth: depth of each binary gate tree.
            - leaf inputs per tree = 2^depth
            - gates per tree      = 2^depth - 1
            - layers in tree      = depth
        connection_type: how to wire leaf inputs to the patch.
        logic_family: which gate family to use for initialization.

    Returns:
        params: list of length out_channels, where params[k] is a list of
                depth gate-logit arrays for kernel k.
        wires:  list of length out_channels, where wires[k] is a list of
                depth (wa, wb) index-pair tuples for kernel k.
    """
    kh, kw = kernel_size
    patch_dim = in_channels * kh * kw
    n_leaves = 2 ** depth

    all_logits = []  # all_logits[k][layer_i] = logit array
    all_wires = []   # all_wires[k][layer_i]  = (wa, wb)

    for k in range(out_channels):
        key, subkey = jax.random.split(key)
        tree_logits = []
        tree_wires = []
        layer_in = patch_dim

        for layer_i in range(depth):
            subkey, layer_key = jax.random.split(subkey)
            layer_out = n_leaves // (2 ** (layer_i + 1))

            # init_gate_layer returns (gate_logits, [indices_a, indices_b])
            logits, wires = init_gate_layer(
                layer_key, layer_in, layer_out,
                connection_type, logic_family,
            )
            tree_logits.append(logits)
            tree_wires.append(wires)
            layer_in = layer_out

        all_logits.append(tree_logits)
        all_wires.append(tree_wires)

    return all_logits, all_wires


# ---------------------------------------------------------------------------
# Forward pass: extract patches → apply tree kernels at every position
# ---------------------------------------------------------------------------

def run_conv_gate_layer(
    params,
    wires,
    x,
    training,
    key,
    kernel_size=(3, 3),
    stride=(1, 1),
    architecture='softmax',
    gumb_tau=1.0,
    dirichlet_concentration=1.0,
    logic_family: LogicFamily = 'full',
):
    """Run a convolutional DLGN layer.

    Args:
        params: list[list[array]] — params[k][layer_i] gate logits for kernel k.
        wires:  list[list[(wa,wb)]] — wires[k][layer_i] index pairs for kernel k.
        x: input (B, H, W, C), values in [0, 1].
        training: bool — soft vs hard decoding.
        key: PRNG key for stochastic decoders (gumbel, dirichlet).
        kernel_size: (kh, kw) must match what was used in init.
        stride: (sh, sw) convolution stride.
        architecture: decoder architecture (softmax, gumbel, light_sigmoid, etc.).
        gumb_tau: Gumbel-softmax temperature.
        dirichlet_concentration: Dirichlet concentration parameter.
        logic_family: 'full' (16 gates) or 'light' (4-element truth table).

    Returns:
        (B, out_h, out_w, out_channels)
    """
    B, H, W, C = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    out_channels = len(params)
    depth = len(params[0])
    patch_dim = C * kh * kw

    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1

    # --- Extract patches ---
    patches = lax.conv_general_dilated_patches(
        x, (kh, kw), (sh, sw),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )  # (B, out_h, out_w, patch_dim)

    # Split keys: one per kernel × layer
    all_keys = jax.random.split(key, out_channels * depth)

    # --- Apply each kernel's gate tree to every patch ---
    def apply_tree(patch_flat, k):
        """Run kernel k's gate tree on one flattened patch → scalar."""
        z = patch_flat
        for layer_i in range(depth):
            layer_key = all_keys[k * depth + layer_i]
            z = run_layer(
                params[k][layer_i],
                wires[k][layer_i],
                z,
                training,
                layer_key,
                architecture,
                gumb_tau,
                dirichlet_concentration,
                logic_family,
            )
        return z.squeeze(-1)  # root gate output (scalar)

    def apply_all_kernels(patch_flat):
        """Run all kernel trees on one patch → (out_channels,)."""
        return jnp.array([apply_tree(patch_flat, k)
                          for k in range(out_channels)])

    # vmap over batch × spatial positions
    flat = patches.reshape(B * out_h * out_w, patch_dim)
    out = jax.vmap(apply_all_kernels)(flat)  # (B*oh*ow, out_channels)
    return out.reshape(B, out_h, out_w, out_channels)


# ---------------------------------------------------------------------------
# Or-pooling: max-pool as t-conorm relaxation of logical OR
# ---------------------------------------------------------------------------

def or_pool(x, kernel_size=(2, 2), stride=None):
    """Or-pooling via max (t-conorm relaxation of logical OR).

    For [0,1] activations, max(a,b) is the maximum t-conorm of a ∨ b.
    Stride defaults to kernel_size (non-overlapping).

    Args:
        x: (B, H, W, C) values in [0, 1].
        kernel_size: pooling window (ph, pw).
        stride: pooling stride; defaults to kernel_size.

    Returns:
        (B, out_h, out_w, C)
    """
    ph, pw = kernel_size
    sh, sw = stride if stride is not None else kernel_size

    return lax.reduce_window(
        x,
        init_value=-jnp.inf,
        computation=lax.max,
        window_dimensions=(1, ph, pw, 1),
        window_strides=(1, sh, sw, 1),
        padding='VALID',
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    key = jax.random.key(0)
    x = jax.random.uniform(key, (2, 8, 8, 3))  # batch=2, 8×8, 3 channels

    # --- Conv layer (full family) ---
    key, init_key, run_key = jax.random.split(key, 3)
    params, wires = init_conv_gate_layer(
        init_key,
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 3),
        depth=2,
        logic_family='full',
    )
    y = run_conv_gate_layer(params, wires, x, training=True, key=run_key,
                            kernel_size=(3, 3), stride=(1, 1),
                            logic_family='full')
    print(f'ConvDLGN (full)   input {x.shape} -> output {y.shape}')

    # --- Conv layer (light family) ---
    key, init_key, run_key = jax.random.split(key, 3)
    params_l, wires_l = init_conv_gate_layer(
        init_key,
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 3),
        depth=2,
        logic_family='light',
    )
    y_l = run_conv_gate_layer(params_l, wires_l, x, training=True, key=run_key,
                              kernel_size=(3, 3), stride=(1, 1),
                              architecture='light_sigmoid',
                              logic_family='light')
    print(f'ConvDLGN (light)  input {x.shape} -> output {y_l.shape}')

    # --- Or-pool ---
    y2 = or_pool(y, kernel_size=(2, 2))
    print(f'OrPool            input {y.shape} -> output {y2.shape}')
