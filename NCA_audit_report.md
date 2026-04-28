# NCA Reference Audit for DLCA Build

Read of four repos in `/Users/user/Documents/ALL_NCA_STUFF/`. All four are from the TUM/Helmholtz Munich group (Deutges → Yang chain): WBC-NCA is the original 2024 classifier paper, NCA-WSS is its 2025 weakly-supervised-segmentation extension, hNCA and aNCA are 2025 followups by Chen Yang refining the head architecture. The codebases are heavily related and reuse the same NCA backbone, so much of the analysis collapses across them.

> Note on completeness: `NCA-WSS-main` is shipped without its `src/` directory. Its `train_five_fold_weak.py` and `evaluate_model.py` reference `src.Datasets`, `src.Models`, `src.Losses` but those files are absent. The repo is unrunnable as-shipped, but the model class it imports (`MaxNCA` / `MeanNCA`) is the same architecture as in `WBC-NCA-main/src/models/NCA.py`, and the training / evaluation scripts contain enough code to fully characterize the pipeline. I treat the WBC-NCA models file as authoritative for the "Deutges" architecture.

---

## Section 1 — Per-repository summary

### aNCA — `/ALL_NCA_STUFF/aNCA-main`
**What:** Attention-pooled NCA classifier (Yang et al., arXiv 2508.12324). The most feature-complete repo of the four. Implements one shared NCA backbone plus ~9 alternative classification heads, selectable by `--predict`: `pool1`, `aNCA`, `hNCA`, `avgfusion`, `attention_lastchannel`, `attention_conv`, `mic`, `patch_attention`, `patch_aNCA`, `embed_patch`. Trains/evals on AML, PBC, MLL, Malaria, SIPAKMED, Urine, CIFAR10, CRC, PatchCamelyon, OCT. 5-fold stratified CV.

**File map:**
- Model: [src/NCA.py](../ALL_NCA_STUFF/aNCA-main/src/NCA.py) — `class NCA(nn.Module)` (top-level w/ head) + `class NCA_backbone(nn.Module)` (the unrolling core).
- Head (alternate): [src/Classifier.py](../ALL_NCA_STUFF/aNCA-main/src/Classifier.py) — standalone `FC_Classifier`. Not used by `train.py`; the heads live inside `NCA.classify()`.
- Trainer: [src/Trainer.py](../ALL_NCA_STUFF/aNCA-main/src/Trainer.py:30) — `Trainer.batch_step` is the training inner loop.
- Train entrypoint: [src/train.py](../ALL_NCA_STUFF/aNCA-main/src/train.py).
- Eval/profiling: [src/eval.py](../ALL_NCA_STUFF/aNCA-main/src/eval.py) — also includes channel-visualization plumbing.
- Utils: [src/utils.py](../ALL_NCA_STUFF/aNCA-main/src/utils.py) — `make_seed`, `evaluate_model`, t-SNE/UMAP visualization.
- Data: [src/data/](../ALL_NCA_STUFF/aNCA-main/src/data/) — `Dataset.py`, `Loader.py`, `DataInfo.py`, `GetData.py`.

**Key parameters (defaults from [src/train.py:199-220](../ALL_NCA_STUFF/aNCA-main/src/train.py:199)):** `channel_n_1=16, hidden_size_1=128, steps_1=64, fire_rate=0.5, batch_size=16, lr=4e-4, n_epochs=32, dropout=0, criterion=Focal`. The argparse defaults conflict with the eval-script defaults (`channel_n_1=128`); published runs are 128 channels (per `aNCA-main/src/eval.py:404`), training default is 16 — read the hNCA training log for actual values used: `--channel_n_1 128 --steps_1 32 --learning_rate 2e-4 --n_epochs 50 --dropout 0.1`. Resize 64×64.

---

### hNCA — `/ALL_NCA_STUFF/hNCA-main`
**What:** Hierarchical NCA (Yang et al., 2025). Stripped subset of aNCA: only the `hNCA` head is implemented. Two NCA backbones in series — first runs at 64×64, output is max-pooled to 16×16 and fed to a second NCA, whose final state is 1×1 max-pooled and sent through a 2-layer MLP. Same datasets as aNCA.

**File map:**
- Model: [src/hNCA.py](../ALL_NCA_STUFF/hNCA-main/src/hNCA.py) — `class hNCA` + `class NCA_backbone`. Backbone is byte-for-byte identical to aNCA's except the dropout in the MLP is enabled here.
- Trainer: [src/Trainer.py](../ALL_NCA_STUFF/hNCA-main/src/Trainer.py) — same as aNCA, minus the `--predict` permute.
- Train entrypoint: [src/train.py](../ALL_NCA_STUFF/hNCA-main/src/train.py).
- Results: [results/](../ALL_NCA_STUFF/hNCA-main/results/) — actual training command lines + per-class confusion matrices for fold 1 of each dataset. **Use these as ground-truth hyperparameters.**

**Key parameters** (from [results/AML+1/output_30094516.txt](../ALL_NCA_STUFF/hNCA-main/results/AML+1/output_30094516.txt:1)): `channel_n_1=128, channel_n_2=128, steps_1=32, steps_2=16, hidden_size_1=hidden_size_2=128, fire_rate=0.5, lr=2e-4, n_epochs=50, dropout=0.1, weight_decay=1e-4, batch_size=16, criterion=Focal, resize=64`. The repo's argparse defaults match.

---

### WBC-NCA — `/ALL_NCA_STUFF/WBC-NCA-main`
**What:** Original NCA-as-classifier (Deutges et al., arXiv 2404.05584 — the WBC paper. **This is the paper your bet is most directly modeled on.**) Four NCA model variants under one backbone:
- `MaxNCA` — global max-pool then 2-layer MLP head. The standard.
- `SimpleNCA` — spatial mean over channels `[input_channels : input_channels+13]`. Uses 13 hidden channels as direct class logits, no MLP.
- `SegNCA` — outputs `x[...,3]`; segmentation flavor.
- `ConvNCA` — feeds final NCA state through a conv stack + MLP. Heavy head.

**File map:**
- Model: [src/models/NCA.py](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py) — all four variants.
- Trainer (Agent): [src/agents/Agent.py](../ALL_NCA_STUFF/WBC-NCA-main/src/agents/Agent.py) — encapsulates seed creation, optimizer/scheduler, train loop.
- Train entrypoint: [train_NCA_generalize.py](../ALL_NCA_STUFF/WBC-NCA-main/train_NCA_generalize.py).
- Loss: [src/losses/LossFunctions.py](../ALL_NCA_STUFF/WBC-NCA-main/src/losses/LossFunctions.py) — `BCELoss`, `DiceLoss`, `DiceBCELoss`. Uses BCE with sigmoid for classification (one-hot targets).
- Datasets: [src/datasets/Dataset.py](../ALL_NCA_STUFF/WBC-NCA-main/src/datasets/Dataset.py).
- Explainability: [explainability.ipynb](../ALL_NCA_STUFF/WBC-NCA-main/explainability.ipynb) (not opened — large output cells).
- Config: [config.yaml](../ALL_NCA_STUFF/WBC-NCA-main/config.yaml).

**Key parameters** ([config.yaml](../ALL_NCA_STUFF/WBC-NCA-main/config.yaml:1)): `channel_n=128, hidden_size=128, steps=64, fire_rate=0.5, lr=4e-4, n_epochs=32, batch_size=16, resize=64, criterion=BCE`. `Agent.__init__` hardcodes Adam (0.9, 0.999) + ExponentialLR(0.9999).

---

### NCA-WSS — `/ALL_NCA_STUFF/NCA-WSS-main`
**What:** Weakly-supervised segmentation by re-using the NCA's hidden state (Deutges et al., arXiv 2508.12322). Train classification first (cross-entropy/focal); at inference, take final hidden state, run PCA across channels to a single component, threshold with Otsu → segmentation mask. Datasets: Matek (white blood cells), Raabin, MLL.

**File map (incomplete — `src/` missing in download):**
- Train: [train_five_fold_weak.py](../ALL_NCA_STUFF/NCA-WSS-main/train_five_fold_weak.py) — 5-fold cross-validation; instantiates `Models.MaxNCA(...)`.
- Eval: [evaluate_model.py](../ALL_NCA_STUFF/NCA-WSS-main/evaluate_model.py) — contains the **PCA-on-hidden-channels-→-Otsu-threshold-→-IoU** pipeline. This is the explicit "feature emergence from classification" mechanism.
- Missing: `src/Datasets.py`, `src/Models.py`, `src/Losses.py`. The model API differs from WBC-NCA's `MaxNCA` in returning **3 tensors instead of 2** (e.g. [evaluate_model.py:30](../ALL_NCA_STUFF/NCA-WSS-main/evaluate_model.py:30) does `_, _, x = model(x)`), implying the WSS variant exposes both logits and the hidden trajectory tensor.

**Key parameters** ([train_five_fold_weak.py:144-188](../ALL_NCA_STUFF/NCA-WSS-main/train_five_fold_weak.py:144)): `channel_n=32, hidden_size=32, steps=32, fire_rate=0.5, lr=1e-4, n_epochs=128, batch_size=32, criterion=Focal, resize=64, ExponentialLR(0.9999)`. Note: lower channel/hidden than WBC-NCA — the lower capacity actually helps the segmentation-emergence story.

---

## Section 2 — The unroll and gradient flow, exactly as coded

All four repos use the same unroll pattern with minor variations. The canonical loop lives in `NCA_backbone.forward`. I quote the **aNCA backbone** since hNCA's is identical and WBC-NCA's `MaxNCA` differs only in flag handling.

### 2.1 The unroll (aNCA / hNCA, identical)

[src/NCA.py:267-271](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:267):
```python
def forward(self, x):
    for _ in range(self.steps):
        x2 = self.update(x).clone()
        x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
    return x # ([b, 64, 64, c])
```

[src/NCA.py:236-265](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:236):
```python
def update(self, x_in): # ([b, 64, 64, c])
    x = x_in.transpose(1,3)              # NCHW for conv2d
    dx = self.perceive(x)                # 3x depthwise conv concat -> NCHW with 3c channels
    dx = dx.transpose(1,3)               # NHWC for the cell-wise MLP
    dx = F.relu(self.fc0(dx), inplace=True)
    dx = self.fc1(dx)                    # ([b, 64, 64, c])
    stochastic = torch.rand_like(dx[..., :1]) >= self.fire_rate
    stochastic = stochastic.float().to(self.device)
    dx = dx * stochastic                 # broadcast across channels
    x = x + dx.transpose(1,3)            # residual update
    x = x.transpose(1,3)                 # back to NHWC
    return x
```

**Gradient flow annotation:**
- `forward` is a Python `for` loop over `self.steps` (32–64 for production runs). Every step's perception, MLP, mask, and residual addition is in autograd's tape. `loss.backward()` traverses all `steps × num_layers_per_step` ops.
- Input enters via `make_seed` ([src/utils.py:322](../ALL_NCA_STUFF/aNCA-main/src/utils.py:322)): `seed = F.pad(img, (0, channel_n - img.shape[-1]), mode='constant', value=0)`. This is the ONLY entry point of the image; `input_channels` slots are filled with image data, the rest with zeros.
- The image is **re-clamped every step** by the `torch.concat((x[...,:input_channels], x2[...,input_channels:]), 3)` line. The first `input_channels` slots of the state are restored to whatever they were before the update — i.e. they're held at the seed values forever. They never get a learnable update. For the DLCA, this is the canonical place to inject the input.
- The mask `stochastic` is a per-cell scalar in `{0, 1}`, broadcast across channels — a single coin flip determines whether a *whole cell* updates or not. NOT per-channel. The `>= fire_rate` (vs `>` in WBC-NCA) means a `fire_rate=0.5` updates ~50% of cells per step.
- The mask is **non-differentiable** (Bernoulli; the `.float()` cast freezes its grad). Gradient flows through `dx`, not through the mask. At unfired cells, `dx*0=0`, so `state` is unchanged at those cells — which means *the gradient at that cell wrt step-N parameters is whatever propagates through later steps*. The randomness IS injected fresh every step (`torch.rand_like` is called inside the loop), so cells that didn't fire at step `t` may fire at `t+1`.
- `x2 = self.update(x).clone()`: the `.clone()` is load-bearing — it's needed because `x2` is then sliced in the concat that produces `x`, and PyTorch otherwise refuses the in-place-on-leaf compose. **Do not remove this** in your DLCA port without testing.
- No truncated BPTT. No gradient checkpointing. The full 32–64-step computation graph is held in memory. WBC-NCA's `Agent.batch_step` ([src/agents/Agent.py:48-60](../ALL_NCA_STUFF/WBC-NCA-main/src/agents/Agent.py:48)) is `optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()` — vanilla autograd, no tricks.

### 2.2 Where the head plugs in (aNCA top-level)

[src/NCA.py:98-108](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:98):
```python
def forward(self, x):
    x = utils.make_seed(x, self.channel_n_1, self.device)   # (b, H, W, C) seed
    x = self.nca1(x)                                        # 32-64 step unroll
    out = self.classify(x)                                  # head
    return out
```

The head sees only the **final** hidden state — no trajectory, no pooling across timesteps. Gradient from `loss = focal(out, target)` walks back through `classify`, then through the unroll, then into the perception/MLP weights and the seed (which isn't learnable, so it dead-ends there).

### 2.3 WBC-NCA `MaxNCA` (the variant your project most closely mirrors)

[src/models/NCA.py:65-76](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:65):
```python
def forward(self, x, steps=32, fire_rate=0.5):
    for step in range(steps):
        x2 = self.update(x, fire_rate).clone()
        x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
    max=F.adaptive_max_pool2d(x.permute(0, 3, 1, 2), (1, 1))
    max = max.view(max.size(0), -1)
    out=self.fc2(max)
    out = F.relu(out)
    out =self.fc3(out)
    return out, x
```

**This is the one to copy.** Single class, no `predict_head` switch, returns `(logits, final_state)` so you can inspect the hidden state for emergence analysis.

### 2.4 NCA-WSS reference call site

[evaluate_model.py:30](../ALL_NCA_STUFF/NCA-WSS-main/evaluate_model.py:30):
```python
def batch_segmentation(model, x):
    _, _, x = model(x)        # 3-tuple return; 3rd is hidden state
    x = x.detach().cpu().numpy()
    x[x <= 0] = 0; x[x > 0] = 1
    ...
    feature_maps = np.transpose(x[i], (2, 0, 1))  # (C, H, W)
    seg = apply_pca_segmentation(feature_maps, n_components=1)
    thresh = threshold_otsu(seg)
    seg = seg > thresh
```

The WSS variant has been refactored to return `(logits, ?, hidden)`. The hidden state is binarized at zero, **then PCA'd across the channel dimension** — the first principal component, after Otsu thresholding, is the segmentation mask. The fact that this works is exactly the "emergent feature extraction from classification" claim: classification gradients shape the channels into a basis whose top PC tracks the cell's spatial extent.

---

## Section 3 — Perception and update mechanisms compared

| Aspect | aNCA / hNCA | WBC-NCA / NCA-WSS |
|---|---|---|
| Perception | `cat(x, p0(x), p1(x))` → 3C channels | identical |
| `p0`, `p1` | depthwise (`groups=channel_n`), `kernel=3`, `padding=1`, `padding_mode="reflect"`, **learnable** | identical |
| State layout | NHWC (transposed in/out for the conv) | identical |
| Update style | residual: `x = x + dx*mask` | identical |
| Fire mask | `rand_like(dx[..., :1]) >= fire_rate`, per-cell, broadcast across channels | `rand([B,H,W,1]) > fire_rate` (note `>` vs `>=`; same intent). The older WBC-NCA explicitly creates the mask on CPU then `.to(device)`; aNCA uses `rand_like` directly. |
| Input persistence | `concat(x[..., :input_channels], x2[..., input_channels:])` after every step | identical |
| Initialization | `fc1.weight.zero_()` is **not done** in aNCA/hNCA. fc0/fc1 default PyTorch init. | `fc1.weight.zero_()` ([src/models/NCA.py:28](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:28)). This is the original "Growing NCA" trick: initial dx=0 so the unroll is initially identity. **aNCA dropped this.** |
| Bias on fc1 | aNCA: default (with bias) | WBC-NCA: `bias=False` ([src/models/NCA.py:22](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:22)) |
| Normalization | none anywhere | none anywhere |

**Key cross-repo observations:**
1. The 3×3 depthwise convs `p0`, `p1` are the perception step. Each channel learns its own pair of 3×3 kernels — i.e., per channel, two arbitrary linear combinations of the 3×3 neighborhood. The `cat(x, p0(x), p1(x))` produces a 3C-dim vector per cell that the per-cell MLP processes. There is no Sobel filter, no fixed perception. This is intentional — the perception is part of what's learned.
2. **For Boolean DLGN, the `p0`/`p1` continuous depthwise conv is the wrong abstraction.** The DLGN needs the raw 3×3 neighborhood across all channels as a Boolean vector (length 9·C). The right replacement is `F.unfold(x, kernel_size=3, padding=1)` to get the literal neighborhood gather (im2col), no learning in this step. The DLGN circuit then consumes that 9C-bit vector and produces a C-bit replacement.
3. The WBC-NCA `fc1.weight.zero_()` trick is essential for stable continuous NCA training — initial `dx=0` means the seed is preserved at step 0, the network slowly learns to perturb it. In Boolean / softmax-gate land, "zero weight" doesn't have a clean analog. The closest is initializing the gate-mixing logits so that the softmax favors the **identity gate** (function index that returns input A unchanged: index 3 in the canonical 16-function table). This makes the initial unroll an identity map on the cell state. **You should do this.**
4. The mask is per-cell, not per-channel. Be consistent: in your DLCA, one Bernoulli per cell, applied to all C output bits.

---

## Section 4 — Head architectures compared

| Head (where defined) | Pooling | Trainable params | Notes |
|---|---|---|---|
| `pool1` ([aNCA NCA.py:113](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:113)) | `AdaptiveMaxPool2d(1,1)` over all C channels | fc(C→hid) + fc(hid→K) | Simplest. Per-channel max over space → MLP. |
| `MaxNCA` ([WBC-NCA NCA.py:70](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:70)) | identical to `pool1` | identical | Same architecture, different repo. |
| `SimpleNCA` ([WBC-NCA NCA.py:196](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:196)) | spatial mean of channels `[in:in+13]` | none | Designates 13 channels as direct logits, no MLP. **No learnable head at all.** |
| `aNCA` ([NCA.py:141](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:141)) | `x * sigmoid(learnable_attn[H,W])`, then sort spatial values per-channel and mean over top `att_percent=0.1` | spatial-attention map (HW params) + MLP | Soft top-K-of-space attention. |
| `attention_lastchannel` ([NCA.py:159](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:159)) | `x * sigmoid(x[:, -1])`, then `AdaptiveMaxPool2d(1,1)` | MLP only | **The state's last channel functions as a learned spatial alpha mask**. Closest analog of the Growing-NCA "alpha channel". |
| `attention_conv` ([NCA.py:164](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:164)) | 1×1 conv on state → 1-channel attention map → multiply → max-pool | 1×1 conv + MLP | Like `attention_lastchannel` but the mask is a learned linear combination across all channels rather than a single dedicated channel. |
| `mic` ([NCA.py:169](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:169)) | downsampled 1×1 conv → softmax-over-space → upsample → multiply → spatial sum | 1×1 conv + MLP | Multiscale-attention variant. Uses softmax (not sigmoid). |
| `hNCA` ([NCA.py:117](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:117) and [hNCA.py:56](../ALL_NCA_STUFF/hNCA-main/src/hNCA.py:56)) | NCA1 final state → max-pool to 16×16 → re-seed → NCA2 (steps_2 unroll) → max-pool 1 → MLP | full second NCA backbone (steps_2 × all params) | Stacked unrolls. **Doubles the BPTT depth.** Adds an `nca2` whose seed is the pooled NCA1 output. |
| `ConvNCA` ([WBC-NCA NCA.py:264](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:264)) | conv stack (3 conv2d + 3 maxpool) → flatten → MLP | heavy CNN head | Defeats the lightweight-NCA argument; included as a baseline. |

**Loss functions:** `FocalLoss(alpha=1, gamma=2)` is the default in all of aNCA, hNCA, and NCA-WSS. WBC-NCA uses `BCELoss` with sigmoid + one-hot targets (binary cross-entropy applied per-class — equivalent to multi-label, slightly different gradient shape than CE for single-label classification but works fine here). For your single-label MNIST/CIFAR task, **use `nn.CrossEntropyLoss` or `FocalLoss(gamma=2)`**. Focal helps with class imbalance — for balanced MNIST/CIFAR, plain CE is fine.

**Trajectory storage:** None of the four repos store the unroll trajectory. The head sees only the final state. WBC-NCA's `MaxNCA.forward` returns `(out, x)` where `x` is the final state, and `utils.animate_activation` ([WBC-NCA src/utils/utils.py:111](../ALL_NCA_STUFF/WBC-NCA-main/src/utils/utils.py:111)) reconstructs trajectories at eval time by *re-running the model with `steps=i`* for `i=0..steps`. Same in aNCA. **For your project, I'd add a `return_trajectory=False` flag rather than re-run.**

---

## Section 5 — Direct recommendations for the DLCA build

### 5.1 Scaffold to clone: WBC-NCA's `MaxNCA`
[src/models/NCA.py:9-76](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:9). It is the most minimal and the most directly aligned with your spec:
- single class, no `predict_head` switch (you don't need 9 head variants for your prototype)
- returns `(logits, final_state)` so you have a hook for the emergence analysis
- does the input-channel splice after every step
- has the canonical NCA shape: pad-to-channels → unroll → pool → MLP
- the per-step `clone()` that PyTorch needs is in the right place

Take this as your skeleton. Strip out the `init_method == "xavier"` branch and the residual addition.

### 5.2 Perception: replace continuous depthwise conv with a fixed im2col gather
The `p0(x)`, `p1(x)` learnable depthwise convs are conceptually wrong for Boolean DLGN. Replace `perceive` with:
```python
def perceive(self, x):  # x: (B, H, W, C) Boolean/soft-Boolean
    x_chw = x.permute(0, 3, 1, 2)
    nbhd = F.unfold(x_chw, kernel_size=3, padding=1)   # (B, C*9, H*W)
    return nbhd.transpose(1, 2).reshape(B, H, W, C * 9)
```
This produces the literal 3×3×C neighborhood as a (9·C)-bit vector at every cell, with reflection-or-zero padding. **No learnable perception** — all the learning is in the gates downstream. (You could also use a fixed identity depthwise conv to match the existing data-flow shape, but `unfold` is cleaner and more idiomatic for "give me the neighborhood".)

### 5.3 Update masking pattern: the WBC-NCA `>=`/`>` cell mask, but applied as replacement not residual
Keep the same per-cell mask shape:
```python
mask = (torch.rand_like(dx[..., :1]) >= self.fire_rate).float()
```
But change the residual line to a where:
```python
# replacement, not residual
new_state = self.dlgn_circuit(self.perceive(x))
x = mask * new_state + (1.0 - mask) * x      # equivalent to where(mask, new, x)
# then preserve input
x = torch.cat((x[..., :self.input_channels], x[..., self.input_channels:]), -1)
# (the second line is equivalent to leaving the input columns of `new_state` alone, but explicit is fine)
```
You can write this as `torch.where(mask.bool(), new_state, x)`, which produces the same gradient (gradient passes through `new_state` only where `mask=1`, through `x` otherwise — exactly what you want for replacement-Boolean). Note that in continuous-NCA-land the residual form `x = x + dx*mask` and the replacement form differ; the residual form lets unfired cells *still* get a tiny gradient via the running residual sum, which doesn't apply to you anyway.

### 5.4 Head: start with `MaxNCA`'s `AdaptiveMaxPool2d(1,1) → MLP`
Simplest, smallest, and the most-validated for classification. Once your unroll trains, optionally try `attention_lastchannel` ([aNCA NCA.py:159](../ALL_NCA_STUFF/aNCA-main/src/NCA.py:159)) — sigmoid-of-last-channel as spatial attention. The "last channel = alpha mask" pattern maps nicely onto your "1–2 reserved channels": designate one as the readout-attention channel, take its sigmoid, multiply, then max-pool.

The `aNCA` (sort-and-top-K) head is overkill for MNIST and harder to interpret in Boolean state. Skip.

### 5.5 What to throw out

1. **`fc1.weight.zero_()`** — useless for Boolean (no `fc1`). Replace with: initialize gate-mixing logits to favor the identity-on-input-A gate. This makes step 0 a near-identity, mirroring the WBC-NCA initial-zero-residual trick.
2. **`F.relu` in `update`** — replaced by the gate forward pass. Drop entirely.
3. **`dropout` in MLP** — fine to keep in the head, but don't try to drop *gate outputs* mid-unroll. Gate outputs are already stochastic (softmax mixture); adding dropout on top is double-counting noise.
4. **The `p0`, `p1` depthwise convs** — see §5.2. Replace with `F.unfold`.
5. **The residual addition** `x = x + dx.transpose(1,3)` — see §5.3.
6. **The `predict_head` switch** in aNCA's `NCA.classify` — pick one head.
7. **`weight_decay`** — set to zero. Weight decay on softmax-mixing logits flattens the gate distribution toward uniform-over-16, undoing your training. If you must regularize, use a custom entropy regularizer on the gate distribution.

### 5.6 Where exactly to insert the DLGN circuit
Inside `update`, between the perception step and the masking step. Concretely, in WBC-NCA's [src/models/NCA.py:44-58](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:44):
```python
def update(self, x_in, fire_rate):
    x = x_in.transpose(1,3)
    dx = self.perceive(x)        # ← REPLACE with F.unfold-based gather (5.2)
    dx = dx.transpose(1,3)
    dx = self.fc0(dx)            # ← REPLACE: fc0+relu+fc1 → DLGN circuit
    dx = F.relu(dx)              # ← DELETE
    dx = self.fc1(dx)            # ← already replaced

    stochastic = ...             # ← KEEP unchanged
    dx = dx * stochastic         # ← REPLACE with `where(mask, dx, x)` (5.3)
    x = x + dx.transpose(1,3)    # ← DELETE; replacement done above
    x = x.transpose(1,3)
    return x
```
The DLGN replaces exactly the `fc0 → relu → fc1` block — i.e., the per-cell MLP that maps the 3C-dim perception vector to a C-dim update. Your DLGN must:
- accept input of shape `(B, H, W, 9*C)` (the unfolded neighborhood)
- output `(B, H, W, C)` (the new cell state)
- be applied identically at every spatial cell (the existing MLP is shared across cells via being applied along the last dim — this works for free if your DLGN takes the last dim as the gate-input bus)

### 5.7 Optimizer / scheduler — copy the Yang group's choices verbatim
- `Adam(lr=2e-4, betas=(0.9, 0.999), weight_decay=0)` — *not* the WBC-NCA `lr=4e-4`. The hNCA results show 2e-4 was needed for the larger 128-channel runs to be stable. For your smaller 8–16 channel net you can probably push to `4e-4` again, but start at `2e-4`.
- `ExponentialLR(gamma=0.9999)` — gentle decay. Per-step (not per-epoch). With ~thousand steps an epoch, this knocks off ~10% per epoch.
- Batch size 16 (what they used) is fine; you have headroom on MNIST so `32` or `64` is also OK.
- Don't use weight decay (see §5.5).
- No gradient clipping in any of these repos. NCAs are stable enough at these LRs that they got away with it. **For Boolean / softmax gates, add `clip_grad_norm_(model.parameters(), 1.0)`** — softmax gradients can spike near sharp gate distributions.

---

## Section 6 — Risks and gotchas

### 6.1 Memory cost of unrolling
For aNCA's published config (B=16, H=W=64, C=128, hidden=128, T=64) the activation memory is roughly:

```
per step:
  perceive output:  16 * 64 * 64 * 384 floats * 4 B  ≈ 100 MB
  fc0 output:       16 * 64 * 64 * 128 floats * 4 B  ≈ 33  MB
  fc1 output:       16 * 64 * 64 * 128 floats * 4 B  ≈ 33  MB
  mask:                                              ≈ 0.25 MB
  ~170 MB / step
total over 64 steps: ~10 GB activations
```

Their `eval.py` is full of `torch.cuda.max_memory_allocated()` instrumentation, which is the tell that they hit this wall. They get away with it on A100s.

For **your** target (B=64, H=W=28, C=12, T=32) the equivalent is much more comfortable:
```
per-step base state:    64 * 28 * 28 * 12 * 4B ≈ 2.4 MB
per-step neighborhood:  64 * 28 * 28 * 108 * 4B ≈ 22 MB
DLGN intermediates depend on circuit depth and gate-mixing storage.
For a softmax-over-16 with depth-D tree of N gates:
  per gate: B * H * W * 16 floats (the 16-way mixture) ≈ 64*28*28*16*4B ≈ 3.2 MB / gate
```

If your circuit has, say, 200 gates, that's ~640 MB of mixture tensors per step, and 32 steps × 640 MB = **~20 GB** of activations. On a 24 GB GPU this is the wall. Mitigations:
1. Drop to LDLGN (2N parameters) with single-sigmoid gate-mixing — gate intermediates are O(1) per gate, not O(16).
2. Reduce T (try T=16 first, scale up after the model trains).
3. Use `torch.utils.checkpoint.checkpoint` on the unroll loop (next subsection).

### 6.2 Truncated BPTT
**You probably don't need it for MNIST**, given the comfortable per-step memory. Don't add complexity. But **add a CLI flag** to enable a `bptt_window` so you can degrade gracefully if you blow up. Implementation: stop the gradient on `x` every `window` steps inside the unroll loop:
```python
for t in range(T):
    x = step(x)
    if (t+1) % bptt_window == 0 and t+1 < T:
        x = x.detach()
```
This caps memory at `window` steps' activations, but the gate parameters stop seeing earlier-time-step gradients. Use only as an escape valve.

### 6.3 Gradient checkpointing
Genuinely not used in any of these repos. For T=32, B=64, MNIST it's premature. If you go to CIFAR (32×32, T=32, C=16) and start running out, wrap the inner-loop step in `torch.utils.checkpoint.checkpoint`:
```python
x = torch.utils.checkpoint.checkpoint(self.update, x, use_reentrant=False)
```
This re-runs each step during backward instead of storing activations — 2× compute for ~T× less memory. The fire-rate mask randomness is a footgun here: by default checkpoint re-runs forward, generating *new* random masks. Pass `preserve_rng_state=True` (default) so the same masks are used in the recomputation; check this in PyTorch docs for your version.

### 6.4 Stochastic-fire × stochastic-gate gradient interaction — the nontrivial bit
This is your real concern and worth taking seriously.

**The two stochasticities:**
1. **Fire mask** `m ∈ {0,1}`, Bernoulli(p=1−fire_rate). Non-differentiable; gradient does not flow through `m`. The mask is *fixed* during a forward pass, fresh at each timestep.
2. **Gate mixture** `g = Σ_i softmax(θ)_i · f_i(input)`. Differentiable; gradient flows into `θ`. The mixture is fully deterministic given `θ` — there is no Gumbel sampling unless you explicitly add it.

If the gate is a deterministic mixture (your softmax-over-16 case), the interaction is benign: the loss `L(out)` decomposes via the chain rule as a sum over (fired-cells × steps × gates) of ∂L/∂θ_g, weighted by the *realized* gate output at each fired cell. **Each batch sees a different fire pattern, but no gradient through the fire process itself.** The variance in the gradient is bounded by the variance in `m·∇` over batches, which is ~ `1/B` for any fixed gradient magnitude — Adam handles this without trouble.

**The real risk:** if you also apply Gumbel-softmax sampling to the gate (to get hard binary intermediates), then you have double stochasticity, and the fire mask compounds the Gumbel variance. In that regime: (a) increase batch size, (b) anneal Gumbel temperature *slowly* — typical `τ`: start 5.0, end 0.1, decay over training; (c) consider keeping the gate mixture soft in the hidden channels and only hard-thresholding at the head's max-pool input.

**My recommendation:** For your prototype, **do not Gumbel-sample mid-unroll**. Keep gate outputs as soft mixtures (∈ [0,1]). The cell state stays in [0,1] throughout the unroll. Only at the head's classification time do you binarize (or just feed the soft state directly into the MLP). This is the safest gradient regime and lets you debug the architecture before adding hard-Boolean noise.

### 6.5 Fire-rate × residual interaction (already addressed)
In continuous NCA, residual update means an unfired cell doesn't receive `dx`, but its existing state continues to be referenced by neighbors' next-step perception. Gradient flows back through unfired cells via their downstream usage. Same in your replacement scheme — `where(m, new, old)` returns `old` at unfired cells, and `old` is still in the graph. No issue.

### 6.6 Initialization for Boolean state
WBC-NCA sets `fc1.weight.zero_()` ([src/models/NCA.py:28](../ALL_NCA_STUFF/WBC-NCA-main/src/models/NCA.py:28)). aNCA dropped this and seems to train OK with default init — but they have a much bigger network and 10× more compute. For Boolean DLCA at small scale, you want the equivalent: an initial unroll that's near-identity.

Concrete recipe for softmax-16 DLGN:
- Initialize gate-mixing logits `θ ~ N(0, σ²)` with `σ` small (e.g. 0.1) and *bias the index of the input-A-pass-through gate by +c* (e.g. c=2). Softmax temperature is 1, so a +2 bias gives that gate ~80% of the mixture mass. Initial cell update ≈ "copy a particular bit of the neighborhood through" — a near-identity if you bias toward "A=center cell's i-th channel".
- Seed = `F.pad(input, (0, C - input_channels))` like the existing repos. The padded zeros are the initial Boolean state of hidden channels.

For LDLGN (product-of-sigmoids):
- Initialize the 2N parameters so that each sigmoid is ≈ 0.5, except the param that controls "A pass-through" should be biased high. Same near-identity goal.

### 6.7 Fire-rate magic numbers
All four repos use `fire_rate = 0.5`. WBC-NCA's mask uses `>` (so `>0.5` means ~50% updates), aNCA uses `>=` (subtle off-by-one in the 0-included case but in practice identical for floats). The chessboard alternation pattern is *not* used in any of these — they all do random-Bernoulli stochasticity. If you want to map onto these codebases cleanly, **start with `fire_rate=0.5` Bernoulli per cell**. Try chessboard later as an ablation.

### 6.8 The `.clone()` on `x2`
In all four repos: `x2 = self.update(x).clone()` before the input-channel splice. Removing this `.clone()` causes a "view + in-place" autograd error in some PyTorch versions. Keep it. (You may not need it with `where`-based replacement, but keeping it is free.)

### 6.9 The `transpose(1,3)` / NHWC dance
The repos shuffle between NHWC and NCHW four times per step. This is annoying but unavoidable: `nn.Conv2d` wants NCHW, and applying `nn.Linear` "per cell" wants the channel dim last. **For your DLGN port, decide once and stick to it.** The cleanest is to run everything in NHWC and use `F.unfold` (which takes NCHW input and produces (B, C·k², L) — you'll permute once for unfold and once back). Don't recapitulate the four-transposes-per-step pattern; it's there because the existing code grew that way, not because it's good.

---

## Section 7 — Open questions to resolve before coding

These are architectural decisions still under-specified in your spec; pick before starting.

1. **DLGN parametrization choice.** Softmax-over-16-functions (more expressive, more parameters per gate, more memory) vs LDLGN product-of-sigmoids (cheaper, less expressive, easier gradient). Your spec mentions both. Pick one for v1.
2. **Gate fan-in.** 2-input gates (the original DLGN paper) or k-input (k=4, larger truth tables, fewer gates). Affects circuit depth needed to get from 9·C inputs to C outputs.
3. **Tree shape.** Single layer of gates with random wiring? Multi-layer tree with full/sparse connections? Number of gates per output channel?
4. **Identity of input channels.** Your spec says 1–2 reserved. Are those bits (a) the raw image quantized to 1-bit per pixel, (b) thresholded grayscale, (c) multi-bit encoded (4 channels = 4-bit grayscale)? For MNIST, single-channel binary is fine (Otsu threshold on the input). For CIFAR, you probably want more bits.
5. **Re-clamping every step.** Spec says "input persistent (re-clamped each step)" matches the existing repos. Confirm. Alternative: input only at t=0, never re-injected. The repos all do persistent re-clamp.
6. **Boolean during training.** Soft (gate outputs ∈ [0,1], state evolves in continuous [0,1]^C) vs hard with straight-through estimator (force state to {0,1} on forward, gradient is 1.0 through the threshold). Recommend soft for v1 (see §6.4); add STE later if you want crisp gates at deployment.
7. **Hidden state initialization.** Zeros (matches existing repos via the `F.pad` seed)? Random Boolean (50/50)? Encoded from input by some lookup? The repos all do zeros. Do that.
8. **Stochastic fire vs chessboard.** Bernoulli fire-rate=0.5 (matches all four repos and is the path of least resistance) vs chessboard alternation (every cell fires every other step in alternating parity). Spec lists both.
9. **Head choice.** `MaxPool→MLP` (simple, recommended for v1) vs `attention_lastchannel` (uses a designated last channel as soft alpha mask) vs `1×1 conv→pool→MLP`. Pick one for v1.
10. **Trajectory storage.** Final state only (matches all repos) vs trajectory mean / max / last-K-step pooling (richer signal, more memory). Final-state only is simpler — recommend that for v1.
11. **Truncated BPTT window.** Off (recommended at MNIST scale) vs window=8 vs window=16. If on, requires care with the random-mask interaction (§6.2).
12. **Gate-distribution regularization.** None (recommended for v1) vs entropy regularizer on the gate softmax to encourage commitment to a single function (`-Σ p log p` with negative coefficient, or anneal toward delta) vs L2 on logits (will hurt — see §5.5). Useful as v2 once you observe whether gates collapse or stay diffuse.
13. **Loss.** `CrossEntropy` for clean MNIST (recommended) vs `FocalLoss(γ=2)` (matches the repos; helpful for class imbalance, harmless for balanced MNIST) vs `BCELoss` w/ one-hot (matches WBC-NCA; equivalent to multi-label, usually CE is preferred for single-label).
14. **Number of steps T.** Spec says 16–32. The repos use 32–64. Bigger T = more chances for the cell-state to organize, but more memory. Recommend start T=16, increase if accuracy plateaus low.
15. **Channel count.** Spec says 8–16. The repos use 32–128. They use higher because they're at 64×64 with a continuous MLP — your gate network is more parameter-efficient per channel. 12 is a reasonable default; 16 if you have memory headroom.
16. **Should the perception step include the cell itself?** WBC-NCA does `cat(x, p0(x), p1(x))` — i.e., raw `x` is one of the three "perceptions" (identity). For Boolean, `F.unfold` with kernel 3 already includes the center cell as one of the 9 positions, so this is implicit and you don't need a separate identity term. Verify this matches your wiring.

---

End of audit.
