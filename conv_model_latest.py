from __future__ import annotations

import math
import random as rand

import flax
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import matplotlib.pyplot as plt


def input_shape_of_dataset(dataset: str) -> tuple[int, int, int]:
    if dataset == "mnist":
        return (28, 28, 1)
    raise NotImplementedError(f"Unsupported dataset: {dataset}")


def num_classes_of_dataset(dataset: str) -> int:
    if dataset == "mnist":
        return 10
    raise NotImplementedError(f"Unsupported dataset: {dataset}")


def kernel_dim_change(
    input_dimension: int,
    padding: int,
    kernel_dimension: int,
    stride: int,
) -> int:
    output_dimension = (input_dimension + 2 * padding - kernel_dimension) // stride + 1
    if output_dimension <= 0:
        raise ValueError(
            "Invalid layer geometry: "
            f"input={input_dimension}, padding={padding}, kernel={kernel_dimension}, stride={stride}"
        )
    return output_dimension


def init_layer(weight_shape: tuple[int, ...], layer_type: str, key: jax.Array) -> jax.Array:
    if layer_type not in ("Conv", "Full", "Output"):
        raise ValueError(f"Layer type {layer_type} does not have trainable weights")
    initializer = jax.nn.initializers.he_normal()
    return initializer(key, weight_shape, dtype=jnp.float32)


def init_bias(bias_shape: tuple[int, ...], layer_type: str) -> jax.Array:
    if layer_type not in ("Conv", "Full", "Output"):
        raise ValueError(f"Layer type {layer_type} does not have trainable bias")
    return jnp.zeros(bias_shape, dtype=jnp.float32)


def _flattened_size(activation_shape: tuple[int, ...] | int) -> int:
    if isinstance(activation_shape, int):
        return activation_shape
    return int(np.prod(activation_shape))


def init_network(hyperparameters: dict, key: jax.Array) -> dict[str, tuple[jax.Array, ...]]:
    layer_specs = hyperparameters["layers"]
    layer_types = hyperparameters["layer_types"]

    if len(layer_specs) != len(layer_types):
        raise ValueError("layers and layer_types must have the same length")
    if not layer_types or layer_types[0] != "Input":
        raise ValueError("The first layer must be an Input layer")

    params_layers: list[jax.Array] = []
    params_bias: list[jax.Array] = []

    activation_shape: tuple[int, ...] | int = layer_specs[0]

    for layer_spec, layer_type in zip(layer_specs[1:], layer_types[1:]):
        if layer_type == "Conv":
            if not isinstance(activation_shape, tuple) or len(activation_shape) != 3:
                raise ValueError(f"Conv expects an HWC activation shape, got {activation_shape}")

            kernel_h, kernel_w, out_channels = layer_spec
            in_height, in_width, in_channels = activation_shape

            weight_shape = (kernel_h, kernel_w, in_channels, out_channels)
            bias_shape = (out_channels,)
            activation_shape = (
                kernel_dim_change(in_height, hyperparameters["conv_pad"], kernel_h, hyperparameters["conv_stride"]),
                kernel_dim_change(in_width, hyperparameters["conv_pad"], kernel_w, hyperparameters["conv_stride"]),
                out_channels,
            )

            key, weight_key = random.split(key)
            params_layers.append(init_layer(weight_shape, layer_type, weight_key))
            params_bias.append(init_bias(bias_shape, layer_type))

        elif layer_type == "Max":
            if not isinstance(activation_shape, tuple) or len(activation_shape) != 3:
                raise ValueError(f"MaxPool expects an HWC activation shape, got {activation_shape}")

            pool_h, pool_w = layer_spec
            in_height, in_width, channels = activation_shape
            activation_shape = (
                kernel_dim_change(in_height, hyperparameters["pool_pad"], pool_h, hyperparameters["pool_stride"]),
                kernel_dim_change(in_width, hyperparameters["pool_pad"], pool_w, hyperparameters["pool_stride"]),
                channels,
            )

        elif layer_type in ("Full", "Output"):
            out_features = int(layer_spec)
            in_features = _flattened_size(activation_shape)

            weight_shape = (in_features, out_features)
            bias_shape = (out_features,)
            activation_shape = out_features

            key, weight_key = random.split(key)
            params_layers.append(init_layer(weight_shape, layer_type, weight_key))
            params_bias.append(init_bias(bias_shape, layer_type))

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    return {
        "layers": tuple(params_layers),
        "bias": tuple(params_bias),
    }


def print_parameter_shapes(params: dict[str, tuple[jax.Array, ...]], hyperparameters: dict) -> None:
    print("Parameter shapes:")
    param_index = 0
    for layer_spec, layer_type in zip(hyperparameters["layers"], hyperparameters["layer_types"]):
        if layer_type in ("Conv", "Full", "Output"):
            weights = params["layers"][param_index]
            bias = params["bias"][param_index]
            print(
                f"  {layer_type:>6} spec={layer_spec} "
                f"weight={tuple(weights.shape)} bias={tuple(bias.shape)}"
            )
            param_index += 1
        else:
            print(f"  {layer_type:>6} spec={layer_spec}")


def _make_torch_loaders_from_split(
    train_set,
    test_set,
    batch_size: int,
    valid_set_size: float,
    seed: int,
    num_workers: int,
):
    import torch

    validation_loader = None
    if valid_set_size > 0:
        train_set_size = math.ceil((1 - valid_set_size) * len(train_set))
        valid_size = len(train_set) - train_set_size
        generator = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(
            train_set,
            [train_set_size, valid_size],
            generator=generator,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=num_workers,
        )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader, test_loader


def load_dataset(config: dict):
    from torchvision import datasets, transforms

    dataset = config["dataset"]
    batch_size = config["batch_size"]
    valid_set_size = config.get("valid_set_size", 0.0)
    seed = config["seed"]
    num_workers = config.get("num_workers", 0)
    data_roots = config.get("data_roots", {"mnist": "./data-mnist"})

    if dataset != "mnist":
        raise NotImplementedError(f"Unsupported dataset: {dataset}")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root=data_roots["mnist"],
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_roots["mnist"],
        train=False,
        download=True,
        transform=transform,
    )
    
    X_train = train_set.data.numpy()
    Y_train = train_set.targets.numpy()
    X_test = test_set.data.numpy()
    Y_test = test_set.targets.numpy()
    
    print("\nX_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape, "\n")
    
    index = rand.randrange(0, Y_test.shape[0])
    plt.imshow(X_train[index], cmap="gray")
    plt.title(f"Label: {Y_train[index]}")
    plt.axis("off")
    plt.savefig("training_image.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    X_train_prep = X_train.reshape(-1, 28, 28, 1)
    X_test_prep = X_test.reshape(-1, 28, 28, 1)
    print("\nX_test reshape:", X_train_prep.shape)
    print("Y_test reshape:", X_test_prep.shape, "\n")
    
    print("\nX_train pre-float32 conversion:", X_train.dtype)
    print("X_test pre-float32 conversion:", X_test.dtype)
    X_train_prep = X_train_prep.astype(np.float32)
    X_test_prep = X_test_prep.astype(np.float32)
    print("\nX_train post-float32 conversion:", X_train_prep.dtype)
    print("X_test post-float32 conversion:", X_test_prep.dtype, "\n")
    
    print("\nX_train slice before normalization", X_train_prep[index][14])
    print("X_test slice before normalization", X_test_prep[index][14])
    X_train_prep = X_train_prep / 255.0
    X_test_prep = X_test_prep / 255.0
    print("\nX_train slice after normalization", X_train_prep[index][14])
    print("X_test slice after normalization", X_test_prep[index][14], "\n")
    
    plt.imshow(X_train_prep[index], cmap="gray")
    plt.title(f"Label: {Y_train[index]}")
    plt.axis("off")
    plt.savefig("training_image_normalized.png", dpi=300, bbox_inches="tight")
    plt.show()

    return _make_torch_loaders_from_split(
        train_set,
        test_set,
        batch_size,
        valid_set_size,
        seed,
        num_workers,
    )


def batch_to_jax(batch) -> tuple[jax.Array, jax.Array]:
    x, y = batch

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)

    x = x.astype(np.float32)
    y = y.reshape(-1).astype(np.int32)

    if x.ndim == 2:
        x = x[None, ..., None]
    elif x.ndim == 3:
        x = x[..., None]
    elif x.ndim == 4 and x.shape[1] in (1, 3):
        x = np.transpose(x, (0, 2, 3, 1))
    elif x.ndim == 4 and x.shape[-1] in (1, 3):
        pass
    else:
        raise ValueError(f"Unexpected image batch shape: {x.shape}")

    return jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.int32)


@flax.struct.dataclass
class TrainState:
    params: dict[str, tuple[jax.Array, ...]]
    opt_state: optax.OptState
    key: jax.Array


def apply_activation(x: jax.Array, activation_type: str) -> jax.Array:
    if activation_type == "ReLU":
        return jax.nn.relu(x)
    if activation_type == "Sigmoid":
        return jax.nn.sigmoid(x)
    if activation_type == "Tanh":
        return jnp.tanh(x)
    if activation_type == "Softmax":
        return jax.nn.softmax(x, axis=-1)
    if activation_type in ("Linear", "Logits"):
        return x
    raise ValueError(f"Unknown activation_type: {activation_type}")


def conv_forward(
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    stride: int,
    padding: int,
) -> jax.Array:
    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + b


def maxpool_forward(
    x: jax.Array,
    pool_h: int,
    pool_w: int,
    stride: int,
    padding: int,
) -> jax.Array:
    return jax.lax.reduce_window(
        operand=x,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, pool_h, pool_w, 1),
        window_strides=(1, stride, stride, 1),
        padding=((0, 0), (padding, padding), (padding, padding), (0, 0)),
    )


def dense_forward(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    return x @ w + b


def forward_logits(
    params: dict[str, tuple[jax.Array, ...]],
    x: jax.Array,
    hyperparameters: dict,
) -> jax.Array:
    layer_specs = hyperparameters["layers"]
    layer_types = hyperparameters["layer_types"]
    hidden_activation = hyperparameters["activation_type"]

    if x.ndim == 3:
        x = x[None, ...]
    if x.ndim != 4:
        raise ValueError(f"forward_logits expects NHWC input, got shape {x.shape}")

    param_index = 0

    for layer_spec, layer_type in zip(layer_specs, layer_types):
        if layer_type == "Input":
            continue

        if layer_type == "Conv":
            weights = params["layers"][param_index]
            bias = params["bias"][param_index]
            x = conv_forward(
                x,
                weights,
                bias,
                hyperparameters["conv_stride"],
                hyperparameters["conv_pad"],
            )
            x = apply_activation(x, hidden_activation)
            param_index += 1
            continue

        if layer_type == "Max":
            pool_h, pool_w = layer_spec
            x = maxpool_forward(
                x,
                pool_h,
                pool_w,
                hyperparameters["pool_stride"],
                hyperparameters["pool_pad"],
            )
            continue

        if layer_type in ("Full", "Output"):
            if x.ndim > 2:
                x = x.reshape((x.shape[0], -1))

            weights = params["layers"][param_index]
            bias = params["bias"][param_index]
            x = dense_forward(x, weights, bias)
            if layer_type == "Full":
                x = apply_activation(x, hidden_activation)
            param_index += 1
            continue

        raise ValueError(f"Unsupported layer type: {layer_type}")

    if param_index != len(params["layers"]):
        raise ValueError("Parameter tree and layer spec are out of sync")

    return x


def forward(
    params: dict[str, tuple[jax.Array, ...]],
    x: jax.Array,
    hyperparameters: dict,
) -> jax.Array:
    logits = forward_logits(params, x, hyperparameters)
    output_activation = hyperparameters.get("output_activation", "Linear")

    if output_activation == "ArgMax":
        return jnp.argmax(logits, axis=-1)
    return apply_activation(logits, output_activation)


def loss_f(
    params: dict[str, tuple[jax.Array, ...]],
    batch_x: jax.Array,
    batch_y: jax.Array,
    hyperparameters: dict,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = forward_logits(params, batch_x, hyperparameters)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    acc = (jnp.argmax(logits, axis=-1) == batch_y).astype(jnp.float32).mean()
    return loss, {"acc": acc}


def make_train_step(tx: optax.GradientTransformation, hyperparameters: dict):
    @jax.jit
    def train_step(
        state: TrainState,
        batch_x: jax.Array,
        batch_y: jax.Array,
    ) -> tuple[TrainState, jax.Array, dict[str, jax.Array]]:
        def apply_loss(params):
            return loss_f(params, batch_x, batch_y, hyperparameters)

        (loss, aux), grads = jax.value_and_grad(apply_loss, has_aux=True)(state.params)
        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        state = state.replace(params=params, opt_state=opt_state)
        return state, loss, aux

    return train_step


def make_eval_batch(hyperparameters: dict):
    @jax.jit
    def eval_batch(
        params: dict[str, tuple[jax.Array, ...]],
        batch_x: jax.Array,
        batch_y: jax.Array,
    ) -> dict[str, jax.Array]:
        logits = forward_logits(params, batch_x, hyperparameters)
        return {
            "loss_sum": optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).sum(),
            "correct": (jnp.argmax(logits, axis=-1) == batch_y).sum(),
            "count": jnp.asarray(batch_y.shape[0], dtype=jnp.float32),
        }

    return eval_batch


def create_optimizer(config: dict) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip(config["clip_value"]),
        optax.adamw(
            learning_rate=config["lr"],
            b1=config.get("b1", 0.9),
            b2=config.get("b2", 0.99),
            weight_decay=config["weight_decay"],
        ),
    )


def evaluate_loader(
    params: dict[str, tuple[jax.Array, ...]],
    loader,
    eval_batch_fn,
) -> dict[str, float] | None:
    if loader is None:
        return None

    totals = {"loss_sum": 0.0, "correct": 0.0, "count": 0.0}

    for batch in loader:
        batch_x, batch_y = batch_to_jax(batch)
        metrics = eval_batch_fn(params, batch_x, batch_y)
        for key, value in metrics.items():
            totals[key] += float(value)

    count = max(totals["count"], 1.0)
    return {
        "loss": totals["loss_sum"] / count,
        "acc": totals["correct"] / count,
    }


def format_metrics(metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return "n/a"
    return f"loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}"


def main() -> None:
    dataset = "mnist"
    hyperparameters = {
        "seed": 0,
        "dataset": dataset,
        "activation_type": "ReLU",
        "output_activation": "Linear",
        "layer_types": [
            "Input",
            "Conv",
            "Max",
            "Conv",
            "Max",
            "Full",
            "Full",
            "Full",
            "Full",
            "Output",
        ],
        "layers": [
            input_shape_of_dataset(dataset),
            (5, 5, 6),
            (2, 2),
            (5, 5, 12),
            (2, 2),
            128,
            128,
            128,
            128,
            num_classes_of_dataset(dataset),
        ],
        "conv_pad": 1,
        "conv_stride": 1,
        "pool_pad": 0,
        "pool_stride": 2,
        "lr": 1e-3,
        "clip_value": 1.0,
        "b1": 0.9,
        "b2": 0.99,
        "weight_decay": 1e-4,
        "batch_size": 5000,
        "valid_set_size": 0.1,
        "num_workers": 0,
        "epochs": 10,
    }

    print(f"jax {jax.__version__}")
    print(f"flax {flax.__version__}")
    print(f"optax {optax.__version__}\n")

    key = random.key(hyperparameters["seed"])
    tx = create_optimizer(hyperparameters)

    key, init_key = random.split(key)
    params = init_network(hyperparameters, init_key)
    print_parameter_shapes(params, hyperparameters)

    state = TrainState(params=params, opt_state=tx.init(params), key=key)

    train_loader, valid_loader, test_loader = load_dataset(hyperparameters)
    train_step_fn = make_train_step(tx, hyperparameters)
    eval_batch_fn = make_eval_batch(hyperparameters)

    epochs = []

    loss_train = []
    loss_valid = []
    loss_test = []

    accuracy_train = []
    accuracy_valid = []
    accuracy_test = []

    for epoch in range(1, hyperparameters["epochs"] + 1):
        batch_losses: list[float] = []
        batch_accs: list[float] = []

        for batch in train_loader:
            batch_x, batch_y = batch_to_jax(batch)
            state, loss, aux = train_step_fn(state, batch_x, batch_y)
            batch_losses.append(float(loss))
            batch_accs.append(float(aux["acc"]))

        train_metrics = {
            "loss": float(np.mean(batch_losses)),
            "acc": float(np.mean(batch_accs)),
        }
        valid_metrics = evaluate_loader(state.params, valid_loader, eval_batch_fn)
        test_metrics = evaluate_loader(state.params, test_loader, eval_batch_fn)

        epochs.append(epoch)

        loss_train.append(train_metrics["loss"])
        loss_valid.append(None if valid_metrics is None else valid_metrics["loss"])
        loss_test.append(None if test_metrics is None else test_metrics["loss"])

        accuracy_train.append(train_metrics["acc"])
        accuracy_valid.append(None if valid_metrics is None else valid_metrics["acc"])
        accuracy_test.append(None if test_metrics is None else test_metrics["acc"])

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"valid={format_metrics(valid_metrics)} "
            f"test={format_metrics(test_metrics)}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_train, label="Training Loss")
    plt.plot(epochs, loss_valid, label="Validation Loss")
    plt.plot(epochs, loss_test, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_image.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy_train, label="Training Accuracy")
    plt.plot(epochs, accuracy_valid, label="Validation Accuracy")
    plt.plot(epochs, accuracy_test, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_image.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
    
# python3 conv_model_latest.py
# jax 0.9.0.1
# flax 0.12.4
# optax 0.2.6

# Parameter shapes:
#    Input spec=(28, 28, 1)
#     Conv spec=(5, 5, 6) weight=(5, 5, 1, 6) bias=(6,)
#      Max spec=(2, 2)
#     Conv spec=(5, 5, 12) weight=(5, 5, 6, 12) bias=(12,)
#      Max spec=(2, 2)
#     Full spec=128 weight=(300, 128) bias=(128,)
#     Full spec=128 weight=(128, 128) bias=(128,)
#     Full spec=128 weight=(128, 128) bias=(128,)
#     Full spec=128 weight=(128, 128) bias=(128,)
#   Output spec=10 weight=(128, 10) bias=(10,)

# X_train shape: (60000, 28, 28)
# Y_train shape: (60000,)
# X_test shape: (10000, 28, 28)
# Y_test shape: (10000,) 


# X_test reshape: (60000, 28, 28, 1)
# Y_test reshape: (10000, 28, 28, 1) 


# X_train pre-float32 conversion: uint8
# X_test pre-float32 conversion: uint8

# X_train post-float32 conversion: float32
# X_test post-float32 conversion: float32 


# X_train slice before normalization [[  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [ 55.]
#  [184.]
#  [253.]
#  [253.]
#  [252.]
#  [125.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]]
# X_test slice before normalization [[  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [ 59.]
#  [253.]
#  [253.]
#  [213.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [156.]
#  [253.]
#  [253.]
#  [116.]
#  [  0.]
#  [  0.]
#  [  0.]
#  [  0.]]

# X_train slice after normalization [[0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.21568628]
#  [0.72156864]
#  [0.99215686]
#  [0.99215686]
#  [0.9882353 ]
#  [0.49019608]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]]
# X_test slice after normalization [[0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.23137255]
#  [0.99215686]
#  [0.99215686]
#  [0.8352941 ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.6117647 ]
#  [0.99215686]
#  [0.99215686]
#  [0.45490196]
#  [0.        ]
#  [0.        ]
#  [0.        ]
#  [0.        ]] 

# epoch=01 train_loss=1.9883 train_acc=0.3800 valid=loss=1.3374 acc=0.6695 test=loss=1.3219 acc=0.6790
# epoch=02 train_loss=0.8733 train_acc=0.7561 valid=loss=0.5391 acc=0.8227 test=loss=0.5114 acc=0.8375
# epoch=03 train_loss=0.4450 train_acc=0.8588 valid=loss=0.3671 acc=0.8802 test=loss=0.3575 acc=0.8900
# epoch=04 train_loss=0.3250 train_acc=0.9002 valid=loss=0.2748 acc=0.9108 test=loss=0.2682 acc=0.9198
# epoch=05 train_loss=0.2561 train_acc=0.9214 valid=loss=0.2242 acc=0.9298 test=loss=0.2166 acc=0.9360
# epoch=06 train_loss=0.2116 train_acc=0.9352 valid=loss=0.1873 acc=0.9423 test=loss=0.1849 acc=0.9448
# epoch=07 train_loss=0.1802 train_acc=0.9454 valid=loss=0.1629 acc=0.9493 test=loss=0.1617 acc=0.9507
# epoch=08 train_loss=0.1571 train_acc=0.9516 valid=loss=0.1472 acc=0.9575 test=loss=0.1467 acc=0.9545
# epoch=09 train_loss=0.1412 train_acc=0.9572 valid=loss=0.1343 acc=0.9595 test=loss=0.1300 acc=0.9577
# epoch=10 train_loss=0.1280 train_acc=0.9605 valid=loss=0.1192 acc=0.9655 test=loss=0.1188 acc=0.9622