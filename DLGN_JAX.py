


@dataclass(frozen=True)
class LogicGateNetwork:
  """A differentiable logic gate network with per-layer logits and wiring."""

  params: list[jax.Array]
  wires: list[WirePair]






def decode_soft(weights: jax.Array) -> jax.Array:
  """Computes gate probabilities with a softmax over gate logits."""
  return jax.nn.softmax(weights, axis=-1)


def decode_hard(weights: jax.Array) -> jax.Array:
  """Selects the most likely gate as a one-hot vector."""
  return jax.nn.one_hot(jnp.argmax(weights, axis=-1), NUMBER_OF_GATES)





def init_gate_layer(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    connection_type: ConnectionType = "random",
) -> tuple[jax.Array, WirePair]:
  """Initializes one differentiable logic gate layer."""
  if connection_type == "random":
    key1, key2 = jax.random.split(key)
    c = jax.random.permutation(key2, 2 * out_dim) % in_dim
    c = jax.random.permutation(key1, in_dim)[c]
    c = c.reshape(2, out_dim)
    wires = (c[0, :], c[1, :])
  elif connection_type == "unique":
    wires = get_unique_connections(in_dim, out_dim, key)
  else:
    raise ValueError(
        f"Connection type {connection_type!r} not implemented. "
        "Supported values are 'random' and 'unique'."
    )

  return init_gates(out_dim), wires


def init_logic_gate_network(
    layer_sizes: Sequence[int],
    key: jax.Array,
    connections: ConnectionType | Sequence[ConnectionType] = "random",
) -> LogicGateNetwork:
  """Initializes a multi-layer differentiable logic gate network."""
  if len(layer_sizes) < 2:
    raise ValueError("layer_sizes must contain at least an input and output size")

  num_layers = len(layer_sizes) - 1
  if isinstance(connections, str):
    connection_types = [connections] * num_layers
  else:
    connection_types = list(connections)

  if len(connection_types) != num_layers:
    raise ValueError(
        "connections must provide exactly one entry per layer: "
        f"expected {num_layers}, got {len(connection_types)}"
    )

  params: list[jax.Array] = []
  wires: list[WirePair] = []

  for in_dim, out_dim, connection_type in zip(
      layer_sizes[:-1], layer_sizes[1:], connection_types
  ):
    key, subkey = jax.random.split(key)
    gate_logits, gate_wires = init_gate_layer(
        subkey, int(in_dim), int(out_dim), connection_type
    )
    params.append(gate_logits)
    wires.append(gate_wires)

  return LogicGateNetwork(params=params, wires=wires)


def run_layer(logits: jax.Array, wires: WirePair, x: jax.Array, training: bool,) -> jax.Array:
  """Runs one logic gate layer on input x."""
  a = x[..., wires[0]]
  b = x[..., wires[1]]
  gate_weights = jax.lax.cond(training, decode_soft, decode_hard, logits)
  return bin_op_s(a, b, gate_weights)


def run_logic_gate_network(
    network: LogicGateNetwork,
    x: jax.Array,
    training: bool = True,
) -> jax.Array:
  """Runs all layers of a differentiable logic gate network."""
  for logits, wires in zip(network.params, network.wires):
    x = run_layer(logits, wires, x, training)
  return x


__all__ = [
    "ConnectionType",
    "DEFAULT_PASS_VALUE",
    "LogicGateNetwork",
    "NUMBER_OF_GATES",
    "PASS_THROUGH_GATE",
    "bin_op_all_combinations",
    "bin_op_s",
    "decode_hard",
    "decode_soft",
    "get_unique_connections",
    "init_gate_layer",
    "init_gates",
    "init_logic_gate_network",
    "run_layer",
    "run_logic_gate_network",
]

def get_parser():
    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--dataset', type=str, choices=[
        'block', 'adult', 'breast_cancer',
        'monk1', 'monk2', 'monk3',
        'mnist', 'mnist20x20',
        'mnist_bin', 'mnist20x20_bin',
        'cifar-10-3-thresholds',
        'cifar-10-31-thresholds',
    ], required=True, help='the dataset to use')

    parser.add_argument('--sum-tau', '-st', type=float, default=10, help='the group-sum temperature tau')
    parser.add_argument('--gumb-tau', '-gt', type=float, default=1, help='the softmax or gumbel-softmax temperature tau')
    
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                             'implementation intended for helping with the understanding.')

    parser.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                                     'additional eval step.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument('--connections', '-co', type=str,
                        default='fully',
                        choices=['random', 'unique', 'fully'])
    
    parser.add_argument('--architecture', '-a', type=str, default='softmax', choices=['gumbel', 'dirichlet', 'softmax'])
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)

    parser.add_argument('--grad-factor', type=float, default=1.)
    
    parser.add_argument('--checkpoint', action='store_true', help='Enable checkpoints (start and end).')
    
    return parser

def main():
    

if __name__ == "__main__":
    main()