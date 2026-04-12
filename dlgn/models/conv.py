"""Convolutional and perception layers for Differentiable Logic Gate Networks.

Pure-functional style: init functions return (params, wires),
run functions consume them.

Architecture hierarchy:
    _init_tree_kernels / _run_tree_kernels   ← shared core
        ├── init_conv_gate_layer / run_conv_gate_layer  ← image convolution
        ├── init_perceive_layer  / run_perceive          ← patch perception
        └── or_pool                                      ← pooling

Based on:
- Petersen et al., "Convolutional Differentiable Logic Gate Networks" (arXiv:2411.04732)
- DLCA notebook: google-research/self-organising-systems diffLogic_CA
"""

import jax
import jax.numpy as jnp
from jax import lax

from dlgn.models.initialization import init_gate_layer
from dlgn.models.gates import bin_op_all_combinations


# ---------------------------------------------------------------------------
# Gate forward (softmax / one-hot decode applied to all 16 binary ops)
# ---------------------------------------------------------------------------

def run_gate_layer(a, b, logits, training):
    """Apply learned gates between paired inputs a, b.

    Args:
        a, b: (..., n_gates)
        logits: (n_gates, 16)
        training: bool — soft (softmax) vs hard (argmax one-hot)
    Returns:
        (..., n_gates)
    """
    combos = bin_op_all_combinations(a, b)       # (..., n_gates, 16)
    weights = jax.lax.cond(
        training,
        lambda w: jax.nn.softmax(w, axis=-1),
        lambda w: jax.nn.one_hot(jnp.argmax(w, axis=-1), 16),
        logits,
    )
    return jnp.sum(combos * weights, axis=-1)


# ---------------------------------------------------------------------------
# Shared core: N independent gate trees wired into a flat input vector
# ---------------------------------------------------------------------------

def _init_tree_kernels(key, input_dim, n_kernels, depth, connection_type, logic_family):
    """Initialize n_kernels independent gate trees over input_dim inputs.

    Each kernel is a binary tree of depth `depth`:
        - leaf inputs = 2^depth  (sampled from input_dim via wiring)
        - gates       = 2^depth - 1
        - layers      = depth

    Returns:
        params: params[k][layer_i] = gate logit array
        wires:  wires[k][layer_i]  = (wa, wb) index arrays
    """
    n_leaves = 2 ** depth
    all_logits = []
    all_wires = []

    for k in range(n_kernels):
        key, subkey = jax.random.split(key)
        tree_logits = []
        tree_wires = []
        layer_in = input_dim

        for layer_i in range(depth):
            subkey, layer_key = jax.random.split(subkey)
            layer_out = n_leaves // (2 ** (layer_i + 1))
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


def _run_tree_kernels(params, wires, flat, training):
    """Run all kernel trees on flat input vectors.

    Args:
        params: params[k][layer_i] — gate logits
        wires:  wires[k][layer_i]  — (wa, wb) index pairs
        flat:   (N, input_dim) — batch of flat vectors
        training: bool
    Returns:
        (N, n_kernels)
    """
    n_kernels = len(params)
    depth = len(params[0])

    def apply_tree(patch_flat, k):
        z = patch_flat
        for layer_i in range(depth):
            wa, wb = wires[k][layer_i]
            g = params[k][layer_i]
            z = run_gate_layer(z[wa], z[wb], g, training)
        return z.squeeze(-1)

    def apply_all_kernels(patch_flat):
        return jnp.array([apply_tree(patch_flat, k) for k in range(n_kernels)])

    return jax.vmap(apply_all_kernels)(flat)


# ---------------------------------------------------------------------------
# Perceive layer: pre-extracted patches → kernel trees → feature vector
# ---------------------------------------------------------------------------

def init_perceive_layer(
    key,
    patch_dim,
    n_kernels,
    depth=2,
    connection_type='random',
    logic_family='full',
):
    """Initialize a perception layer.

    Each of the n_kernels kernels is a gate tree wired into a
    flattened patch of size patch_dim.

    For a CA-style perceive with a 3×3 Moore neighborhood and C channels:
        patch_dim = 9 * C

    Args:
        key: PRNG key.
        patch_dim: size of the flattened input vector per patch.
        n_kernels: number of independent kernel trees (output features).
        depth: tree depth (leaf inputs = 2^depth).
        connection_type: 'random' or 'unique'.
        logic_family: 'full' or 'light'.

    Returns:
        params[k][layer_i], wires[k][layer_i]
    """
    return _init_tree_kernels(key, patch_dim, n_kernels, depth,
                              connection_type, logic_family)


def run_perceive(params, wires, x, training):
    """Apply perception kernels to patch(es).

    Replaces the DLCA-style perceive that transposes, duplicates across
    kernels, and rearranges with einops. Instead:
        flatten patch → each kernel selects from flat pool → tree outputs

    Args:
        params: params[k][layer_i] from init_perceive_layer.
        wires:  wires[k][layer_i] from init_perceive_layer.
        x: patch input. Accepts:
            (patch_size, channels)         — single patch
            (batch, patch_size, channels)  — batch of patches
            (batch, patch_dim)             — already flattened
        training: bool — soft vs hard gate decoding.

    Returns:
        (n_kernels,) for single patch, or (batch, n_kernels) for batch.
    """
    squeezed = False

    if x.ndim == 2:
        # (patch_size, channels) — single patch, flatten and add batch dim
        flat = x.reshape(1, -1)
        squeezed = True
    elif x.ndim == 3:
        # (batch, patch_size, channels) → (batch, patch_dim)
        flat = x.reshape(x.shape[0], -1)
    else:
        # (batch, patch_dim) — already flat
        flat = x

    out = _run_tree_kernels(params, wires, flat, training)

    if squeezed:
        return out.squeeze(0)   # (n_kernels,)
    return out                  # (batch, n_kernels)


# ---------------------------------------------------------------------------
# Conv layer: full image → patch extraction → kernel trees → spatial output
# ---------------------------------------------------------------------------

def init_conv_gate_layer(
    key,
    in_channels,
    out_channels,
    kernel_size=(3, 3),
    depth=2,
    connection_type='random',
    logic_family='full',
):
    """Initialize a convolutional DLGN layer.

    Each output channel gets a gate tree wired into patches of size
    in_channels * kh * kw.

    Returns:
        params[k][layer_i], wires[k][layer_i]
    """
    kh, kw = kernel_size
    patch_dim = in_channels * kh * kw
    return _init_tree_kernels(key, patch_dim, out_channels, depth,
                              connection_type, logic_family)


def run_conv_gate_layer(params, wires, x, training, kernel_size=(3, 3), stride=(1, 1)):
    """Run a convolutional DLGN layer.

    Extracts patches from the image, runs kernel trees at every
    spatial position (weight sharing), returns spatial feature maps.

    Args:
        params, wires: from init_conv_gate_layer.
        x: (B, H, W, C), values in [0, 1].
        training: bool.
        kernel_size: (kh, kw) — must match init.
        stride: (sh, sw).

    Returns:
        (B, out_h, out_w, out_channels)
    """
    B, H, W, C = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    out_channels = len(params)
    patch_dim = C * kh * kw

    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1

    patches = lax.conv_general_dilated_patches(
        x, (kh, kw), (sh, sw),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )  # (B, out_h, out_w, patch_dim)

    flat = patches.reshape(B * out_h * out_w, patch_dim)
    out = _run_tree_kernels(params, wires, flat, training)
    return out.reshape(B, out_h, out_w, out_channels)


# ---------------------------------------------------------------------------
# Or-pooling: max as t-conorm relaxation of logical OR
# ---------------------------------------------------------------------------

def or_pool(x, kernel_size=(2, 2), stride=None):
    """Or-pooling via max over a spatial window.

    For [0,1] activations, max(a,b) is the t-conorm of a ∨ b.
    Stride defaults to kernel_size (non-overlapping).

    Args:
        x: (B, H, W, C).
        kernel_size: (ph, pw).
        stride: defaults to kernel_size.
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
    key = jax.random.PRNGKey(0)

    # --- Perceive layer (patch input, like DLCA) ---
    key, k1 = jax.random.split(key)
    patch = jax.random.uniform(k1, (4, 9, 3))  # batch=4, 9 neighbors, 3 channels

    key, k2 = jax.random.split(key)
    p_params, p_wires = init_perceive_layer(
        k2, patch_dim=9 * 3, n_kernels=8, depth=2,
    )
    feats = run_perceive(p_params, p_wires, patch, training=True)
    print(f'Perceive  input {patch.shape} -> output {feats.shape}')
    # expect (4, 8)

    # --- Conv layer (image input) ---
    x = jax.random.uniform(key, (2, 8, 8, 3))

    key, k3 = jax.random.split(key)
    c_params, c_wires = init_conv_gate_layer(
        k3, in_channels=3, out_channels=4,
        kernel_size=(3, 3), depth=2,
    )
    y = run_conv_gate_layer(c_params, c_wires, x, training=True,
                            kernel_size=(3, 3), stride=(1, 1))
    print(f'ConvDLGN  input {x.shape} -> output {y.shape}')
    # expect (2, 6, 6, 4)

    # --- Or-pool ---
    y2 = or_pool(y, kernel_size=(2, 2))
    print(f'OrPool    input {y.shape} -> output {y2.shape}')
    # expect (2, 3, 3, 4)

    # --- Verify conv and perceive agree on the same patch ---
    one_patch = x[0, 0:3, 0:3, :]          # (3, 3, 3) from image
    one_patch_flat = one_patch.reshape(1, -1)  # (1, 27)

    conv_at_00 = y[0, 0, 0, :]
    perceive_at_00 = run_perceive(c_params, c_wires, one_patch_flat, training=True)

    match = jnp.allclose(conv_at_00, perceive_at_00, atol=1e-5)
    print(f'\nConv[0,0,0] vs Perceive(same patch): '
          f'{"MATCH" if match else "MISMATCH"}')
    if not match:
        print(f'  conv:     {conv_at_00}')
        print(f'  perceive: {perceive_at_00}')