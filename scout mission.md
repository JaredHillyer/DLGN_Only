# scout mission

This is a neutral scouting note for the repository. The goal is to map what is here, how the code families relate, and where they differ. It is not a recommendation memo, migration plan, or bug report, though obvious gaps are called out as observations.

`NCA_audit_report.md` already contains a detailed deep dive on the NCA reference repositories. This document uses that as prior context and keeps the focus broader: the whole repo layout, active code versus reference/provenance, and the bridges between DLGN, DLCA, and NCA work.

## quick read

The repo is organized around a newer standalone JAX DLGN package plus several imported or archived NCA/DLCA references.

- `dlgn/`, `scripts/`, `configs/`, and `tests/` are the most package-like, actively structured code.
- `conv_model_latest.py`, `Conv_to_Alter_PREMADE_KERNElS_NEEDED/`, and `difflogic_ca.py` are active or transitional experiments, not fully integrated package code.
- `WBC-NCA-main/`, `aNCA-main/`, `hNCA-main/`, and `NCA-WSS-main/` are reference NCA repositories. They share a continuous NCA backbone but differ mainly in classifier heads, segmentation use, and hierarchy/attention variants.
- `Growing-Neural-Cellular-Automata-*` directories are broader NCA reference material for growth, tasks, particles, image-to-image, and visual demos.
- `archive/` preserves notebooks and older dataset/DLGN code that appear to be provenance for the current package.
- `logs/`, `outputs/`, weights, images, gifs, and result files are context and evidence, not core source.

## relationship map

| Area | Role | Framework | Main entrypoints | Connects to | Status |
|---|---|---|---|---|---|
| `dlgn/` | Standalone Differentiable Logic Gate Network package | JAX, Optax, optional Torch data loaders | `dlgn.cli.train`, `dlgn.training.loop.train_model`, `dlgn.models.network.forward_logits` | Uses `configs/`, `scripts/`, `tests/`; draws from archived DLGN notebooks | Active package core |
| `scripts/` | User-facing runners and experiments | Python, JAX, Optax, Torch for data | `train_dlgn.py`, `resume_dlgn.py`, `smoke_test_xor.py`, `test_mnist_layers.py`, `download_datasets.py` | Wraps `dlgn/`; `test_mnist_layers.py` uses `dlgn.models.conv` for conv DLGN | Active runner layer |
| `configs/presets/` | JSON presets for flat DLGN runs | JSON | `xor_smoke.json`, `mnist_light.json`, `adult_full.json` | Consumed by CLI/config flows | Active config examples |
| `tests/` | Regression coverage for package contracts | Pytest, JAX, NumPy | Dataset registry, gates, GroupSum, checkpoints, XOR smoke | Anchors behavior of `dlgn/` | Active verification |
| `conv_model_latest.py` | Loose CNN-on-MNIST JAX experiment | JAX, Flax, Optax, Torchvision | Script-level `main()` | Conceptually adjacent to conv DLGN, but not imported by package | Active/untracked scratch experiment |
| `Conv_to_Alter_PREMADE_KERNElS_NEEDED/` | Earlier conv DLGN/perception staging area | JAX | `conv.py`, `Get_Conv_Working.ipynb`, `config.py` | Similar to `dlgn/models/conv.py`; appears to precede package integration | Transitional/provenance |
| `difflogic_ca.py` and `diffLogic_CA.ipynb` | Google DLCA notebook export/reference | JAX, Flax, Optax, notebook utilities | Notebook-style cells, not a package entrypoint | Conceptual source for differentiable logic cellular automata | Reference/provenance |
| `WBC-NCA-main/` | Original NCA classifier reference | PyTorch | `train_NCA_generalize.py`, `src/models/NCA.py`, `src/agents/Agent.py` | Compared in `NCA_audit_report.md`; baseline continuous NCA classifier | Reference repo |
| `aNCA-main/` | Attention-pooling NCA classifier reference | PyTorch | `src/train.py`, `src/eval.py`, `src/NCA.py` | Extends WBC-style NCA with many heads | Reference repo |
| `hNCA-main/` | Hierarchical NCA classifier reference | PyTorch | `src/train.py`, `src/hNCA.py`, `results/` | Stacks two NCA backbones; shares patterns with aNCA | Reference repo plus results |
| `NCA-WSS-main/` | Weakly supervised segmentation reference | PyTorch | `train_five_fold_weak.py`, `evaluate_model.py` | Reuses NCA hidden maps for PCA/Otsu segmentation | Reference repo, incomplete source |
| `Growing-Neural-Cellular-Automata-master/` | PyTorch reproduction of Growing NCA | PyTorch, Pygame | `training.ipynb`, `main_pygame_dl.py`, `lib/CAModel.py` | General NCA background and demo material | Reference |
| `Growing-Neural-Cellular-Automata-Pytorch-master/` | Broader NCA task experiments | PyTorch | `CA_tasks/`, `CA_Basic/`, `CA_Particles_V3/`, `CA_Img2Img/` | Shows growth, matrix/task, particle, and image-to-image CA variants | Reference/provenance |
| `archive/` | Older notebooks and dataset code | Mixed JAX/Python notebooks | `archive/notebooks/*`, `archive/data_set_code_old/*` | Source/provenance for `dlgn/` modules and dataset rewrites | Historical context |
| `logs/`, `outputs/`, result files, weights, media | Run artifacts and visual/model evidence | Text, pickle/checkpoint, images/gifs/videos | Logs and generated outputs | Help interpret experiments but are not source | Generated/reference context |

## active dlgn package

The `dlgn/` package is the clearest active implementation. It is a JAX implementation of Differentiable Logic Gate Networks with a small package boundary, CLIs, dataset loaders, tests, analysis helpers, and checkpointing.

Core structure:

- `dlgn/models/gates.py` defines the canonical 16 full-family binary gates and the 4-bit light truth-table family.
- `dlgn/models/decoders.py` turns trainable logits into soft, straight-through, Dirichlet, or hard gate weights.
- `dlgn/models/initialization.py` builds gate layers and wiring. It biases initialization toward pass-through behavior, mirroring the "start near identity" idea from continuous NCA but expressed in gate logits.
- `dlgn/models/network.py` wires gate layers into a flat feed-forward DLGN and sends the final features to `group_sum_head`.
- `dlgn/models/conv.py` adds convolutional/perception-style DLGN layers: extract image patches, run shared gate trees over the flattened patch, then optionally OR-pool via max.
- `dlgn/training/` owns loss, optimizer, train/eval steps, checkpoints, and the high-level training loop.
- `dlgn/data/` owns the dataset registry and loaders for toy XOR/parity, UCI-style datasets, MNIST variants, and thresholded CIFAR.
- `dlgn/analysis/` contains gate inspection and plotting helpers.
- `dlgn/cli/` provides train/resume/checkpoint-inspection entrypoints exposed by `pyproject.toml`.

The package has a fairly deliberate split: `models/` contains pure-ish JAX forward/init logic, `training/` contains step orchestration, `data/` handles non-JAX input plumbing, and `scripts/` are convenience wrappers or larger experiments.

## scripts and presets

The root `scripts/` directory has two different flavors.

- Package wrappers: `train_dlgn.py` and `resume_dlgn.py` mostly delegate to `dlgn.cli`.
- Operational scripts: `download_datasets.py` prepares storage, and `smoke_test_xor.py` runs a small end-to-end XOR training job and saves a checkpoint.
- Experiment script: `test_mnist_layers.py` compares a regular flattened DLGN with a deeper convolutional DLGN on MNIST/CIFAR-like inputs. It has its own presets and training loop rather than being only a thin CLI wrapper.

The JSON files under `configs/presets/` are simpler flat-DLGN presets. They line up with the package config schema rather than the larger conv experiment script.

## active and loose experiments

`conv_model_latest.py` is currently untracked, so it should be treated carefully as live local work. It is a conventional JAX/Flax/Optax CNN-style MNIST experiment, not a DLGN package module. It has its own data loading, network initialization, convolution, pooling, dense layers, optimizer, evaluation, and plotting. It looks useful as a comparison or scaffold for image-training mechanics, but it is not wired into `dlgn/`.

`Conv_to_Alter_PREMADE_KERNElS_NEEDED/conv.py` closely resembles `dlgn/models/conv.py`, but the package version has a more vectorized `_run_tree_kernels` implementation. This directory reads like a staging/provenance area for the conv DLGN layer, with a notebook and config sketch kept nearby.

`difflogic_ca.py` is a Python export of the Google `diffLogic_CA.ipynb` notebook. It contains notebook/Colab setup, visualization utilities, DLCA constants, and experiment code. As exported Python, it is reference material rather than an importable package module.

`patch_calc.py`, `wait.py`, and `yo.txt` are small loose root files. From a scouting perspective, they are peripheral compared with `dlgn/`, the conv experiments, and the NCA references.

## nca reference repos

The four microscopy NCA repos form a related PyTorch family. The existing `NCA_audit_report.md` gives the deep architecture read, so this section only summarizes their relationship.

- `WBC-NCA-main/` is the original white-blood-cell classifier reference. Its key architecture is a continuous NCA update loop followed by global max pooling and a small MLP head.
- `NCA-WSS-main/` uses NCA classification features for weakly supervised segmentation. The source is incomplete because its `src/` package is missing, but its train/eval scripts still show the segmentation pipeline: hidden features, PCA, Otsu thresholding, and IoU evaluation.
- `hNCA-main/` is a hierarchical follow-up that stacks two NCA backbones. It includes `results/` with trained checkpoints/logs for fold-1 runs.
- `aNCA-main/` is the broadest of the four. It keeps the NCA backbone but experiments with attention and pooling heads, including the hierarchical head.

These are continuous-valued PyTorch NCAs. They are conceptually close to the DLCA direction because they update a spatial state over repeated steps, but their learned update is usually a continuous perception-plus-MLP block rather than a Boolean/differentiable-logic gate circuit.

## growing nca references and archive

`Growing-Neural-Cellular-Automata-master/` is an unofficial PyTorch reproduction of the original Growing NCA work, with a model, training notebook, and Pygame demo.

`Growing-Neural-Cellular-Automata-Pytorch-master/` is larger and more varied. It contains:

- `CA_Basic/` for growth/image target experiments.
- `CA_tasks/` for computational tasks such as copying and matrix multiplication.
- `CA_Particles_V3/` and deprecated particle directories for learned particle/physics-style CA.
- `CA_Img2Img/` for image-to-image CA work.

The `archive/` directory preserves prior notebooks and older dataset code. The comments in `dlgn/` show that many package modules were ported from `Older_Imp.ipynb` and related notebooks. The archive is therefore best read as provenance for design decisions, not as the primary code to run.

## major differences

### dlgn gate network vs dlca notebook

The current `dlgn/` package is mostly a flat or convolutional differentiable logic gate network. A layer takes input features, selects wire pairs, decodes trainable gate logits, applies binary gate operations, and repeats through a stack. Classification uses a `GroupSum` head.

The DLCA notebook/export is cellular automata oriented. It thinks in terms of a spatial state, local perception, repeated CA update steps, and visualization of evolving hidden channels. Its code is notebook-first and experiment-first.

The bridge is `dlgn/models/conv.py`: it adapts DLGN gate trees to image patches and perception-like kernels. That makes it closer to cellular automata code, but it is still a feed-forward conv/pool stack in the current scripts rather than a full recurrent CA unroll with persistent state.

### flat dlgn vs convolutional dlgn

Flat DLGN:

- Inputs are vectors such as XOR bits, one-hot/tabular UCI features, or flattened MNIST.
- `init_logic_gate_network` builds a list of gate layers with fixed widths.
- `forward_logits` runs the layers and applies `group_sum_head`.
- The main `train_model` path uses registry metadata like input dimension and class count.

Convolutional DLGN:

- Inputs are images in NHWC form.
- `init_conv_gate_layer` creates gate-tree kernels over flattened local patches.
- `run_conv_gate_layer` extracts patches using JAX convolution patch extraction, runs shared gate trees at every spatial location, and returns feature maps.
- `or_pool` uses max pooling as a relaxed logical OR.
- The main training path for conv DLGN currently lives in `scripts/test_mnist_layers.py`, not in the general `dlgn.cli.train` path.

### continuous nca vs boolean/differentiable logic direction

The PyTorch NCA references use continuous state updates:

- A state tensor holds input channels plus hidden channels.
- A perception step uses depthwise 3x3 convolutions and identity features.
- A per-cell MLP predicts an update.
- A stochastic fire mask updates some cells per step.
- The classifier head reads the final hidden state.

The DLGN/DLCA direction replaces or complements the continuous MLP update with differentiable logic:

- Gate logits represent soft or hard Boolean functions.
- Wire pairs define which prior features feed each gate.
- Full-family gates use a 16-function truth table; light-family gates use a smaller truth-table parameterization.
- Soft training and hard evaluation are explicit modes.

The shared idea is repeated local computation over structured state. The main difference is the update rule: continuous learned filters/MLPs in NCA versus differentiable Boolean gate circuits in DLGN/DLCA.

### dataset loading

`dlgn/data/loaders.py` is the active dataset dispatch layer. It supports toy arrays without Torch and uses Torch/Torchvision lazily for real datasets. The registry in `dlgn/data/registry.py` is the ground truth for flat model dimensions and class counts.

Older dataset code remains in `archive/data_set_code_old/`. Current modules mention minimal rewrites and bug fixes, especially in UCI handling. The package has tests around dataset metadata and an AdultDataset bucket fix, which suggests dataset stability has mattered during cleanup.

CIFAR is split between thresholded/binary-style loading and conv experiments. The loader supports CIFAR through threshold transforms, while the general flat training path depends on registry dimensions. The conv script can infer image input shape from sampled data instead of using the flat registry in the same way.

### training loops

The package training loop in `dlgn/training/loop.py` is step-based:

- load dataset
- initialize params/wires
- create optimizer
- cycle batches
- run a JIT train step
- evaluate hard/soft metrics at intervals
- return state, wires, history, dataset info, and loaders

The conv experiment in `scripts/test_mnist_layers.py` has its own training loop because its parameter structure is a nested conv-plus-FC tree and it needs image-specific probing to determine flatten sizes.

The NCA reference repos use PyTorch training patterns: modules, optimizers, schedulers, epochs/folds, and model-specific train/eval scripts.

### checkpointing and outputs

The active package has checkpoint helpers in `dlgn/training/checkpoints.py`. Payloads preserve params, optimizer state, PRNG key, wires, config, history, dataset info, and extra metadata. The tests explicitly guard round-trip behavior.

The CLI writes run directories under `outputs/`, including config snapshots, history, metrics, summary, and final/latest checkpoints. The current `--save-every` CLI option is present but intentionally errors as not implemented.

The reference repos include their own results/checkpoints/logs, especially under `hNCA-main/results/` and various model-weight files. Those artifacts are useful evidence but are not part of the active package's checkpoint format.

### heads and losses

Flat DLGN uses `GroupSum`: output features are divided into class groups, summed, and temperature-scaled. Training uses softmax cross-entropy with integer labels.

The NCA references use pooling-plus-MLP heads, direct hidden-channel heads, attention heads, conv heads, or hierarchical heads depending on the repo. Their losses include BCE, cross entropy, and focal loss depending on the project.

This means "classification head" means different things across the repo:

- In `dlgn/`, it is a non-learned grouping/summing rule over final gate features.
- In WBC/aNCA/hNCA, it is usually a learned MLP or attention/pooling structure over a spatial hidden state.
- In NCA-WSS, the hidden state itself is also used as a segmentation signal after classification.

## known gaps and rough edges

These are observations, not prioritized tasks.

- `NCA-WSS-main/` references a missing `src/` package, so it is not runnable as shipped.
- `dlgn/data/block.py` is a boundary stub; the original BlocksDataset source is missing.
- `dlgn.cli.train` exposes `--save-every`, but the option currently exits with an "not implemented yet" parser error.
- Bare `cifar10` appears in the dataset registry class list and conv script choices, but the flat registry does not define a bare `cifar10` input dimension. Thresholded CIFAR names do have dimensions.
- `conv_model_latest.py` is untracked and separate from package code. It should be treated as live local work rather than ignored or overwritten.
- `Conv_to_Alter_PREMADE_KERNElS_NEEDED/config.py` reads like a sketch rather than a polished importable config file.
- `difflogic_ca.py` is an exported notebook and includes notebook/Colab idioms, so it should not be treated like a normal Python module without cleanup.
- Several large assets, model weights, gifs, videos, notebooks, and logs are valuable context but are not easily comparable through source-only scouting.

## neutral orientation notes

The repo's center of gravity appears to be moving from scattered notebooks and reference repos toward a cleaner JAX package for DLGN experiments. The strongest current package boundary is `dlgn/`. The strongest conceptual bridge toward DLCA is `dlgn/models/conv.py`, because it brings logic gates into local image-patch computation.

The NCA reference repos are not redundant copies of the active DLGN code. They answer a different question: how continuous neural cellular automata classify or segment microscopy images. They are useful for architectural comparison, especially state layout, local perception, update masking, pooling heads, and hidden-state interpretability.

The archive and loose experiments matter because they explain how the repo got here. The active package is cleaner than the notebooks, but the notebooks and scratch files still contain design history that may not be fully represented in packaged modules.
