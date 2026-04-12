import jax
import jax.numpy as jnp
import jax.random as random

@partial(jax.jit, static_argnums=(1,2))
def get_grid_patches(grid, patch_size, channel_dim, periodic):
  # Given a grid NxN it creates a list of (patch_size x patch_size, channel_dim) patches.

  pad_size = (patch_size - 1) // 2

  # Two possible modes exist: with or without periodic boundary conditions.
  padded_grid = jax.lax.cond(
      periodic,
      lambda g: jnp.pad(
          g, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="wrap"
      ),
      lambda g: jnp.pad(
          g,
          ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
          mode="constant",
          constant_values=0,
      ),
      grid,
  )
  padded_grid = jnp.expand_dims(padded_grid, axis=0)
  patches = conv_general_dilated_patches(
      padded_grid,
      filter_shape=(patch_size, patch_size),
      window_strides=(1, 1),
      padding="VALID",
      dimension_numbers=("NHWC", "OIHW", "NHWC"),
  )[0]

  # Rearrange to have (list, patch_size x patch_size, channel_dim)
  patches = rearrange(patches, "x y (c f) -> (x y) f c", c=channel_dim)
  return patches

def main():
   

if(__name__=="__main__"):
    main()




def run_layer(logits, wires, x, training):
  a = x[..., wires[0]]
  b = x[..., wires[1]]
  logits = jax.lax.cond(training, decode_soft, decode_hard, logits)
  out = bin_op_s(a, b, logits)
  return out


def run_update(params, wires, patches, training):
  for g, c in zip(params, wires):
    x = run_layer(g, c, , training)
  return x




# Current:
# (batch, patch_positions, channels)
# → transpose to channel-first
# → duplicate across kernels
# → run layered logic transforms
# → flatten features
# → concatenate original center cell

# New:
# (batch, patch_positions, channels)
# → flatten to (batch, patch_dim)
# → each kernel samples/selects from that flat candidate pool
# → run tree / logic stack
# → produce per-kernel outputs
# → optionally pool / hand off onward

# code:
# def run_perceive(params, wires, x, training):
#   """Applies a perception layer to a patch.

#   Args:
#       params: List of kernel parameters for each layer.
#       wires: List of wire configurations for each layer.
#       x: Input patch, shape [batch_size, patch_size, channel_size].
#       training: Boolean indicating training mode.

#   Returns:
#       Output feature vector, shape [batch_size, channel_size].
#   """

#   # Apply each layer using vmap for kernel parallelism.
#   run_layer_map = jax.vmap(run_layer, in_axes=(0, None, 0, None))
#   x_prev = x
#   x = x.T  # [channel_size, batch_size, patch_size]

#   """
#     Duplicate 'x' to create n_kernels copies for the first layer, which all share the same input.
#     Subsequent layers receive unique inputs.
#     """
#   x = jnp.repeat(
#       x[None, ...], params[0].shape[0], axis=0
#   )  # [n_kernels, channel_size, batch_size, patch_size]

#   # Iterate through layers, applying kernels and wire configurations.
#   for g, c in zip(params, wires):
#     x = run_layer_map(g, c, x, training)

#   x = rearrange(
#       x, 'k c s -> (c s k)'
#   )  # [channel_size * patch_size * n_kernels]

#   return jnp.concatenate(
#       [x_prev[4, :], x], axis=-1
#   )  # Concatenate the original input.