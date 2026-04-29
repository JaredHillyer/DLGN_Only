# NCA at MarrLab, the Broader NCA Landscape, and the DLCA Bridge

A technical survey for someone building Differentiable Logic Cellular Automata. The framing throughout is **system identification**: inferring local update rules from observed spatiotemporal data such that, when iterated, the rules reproduce the data. Foundational pre-2020 CA theory (von Neumann, Conway, Wolfram, Langton) is cited only where it actively shapes a current method. Venue status is flagged for every artifact: **[peer-reviewed]**, **[preprint]**, **[blog/demo]**, **[workshop]**.

---

## Section 1 — MarrLab's NCA work

Carsten Marr leads the Institute of AI for Health at Helmholtz Munich and (since July 2025) holds a W3 Professorship in AI in Cell Therapy and Hematology at LMU University Hospital ([helmholtz-munich.de/en/aih/carsten-marr](https://www.helmholtz-munich.de/en/aih/carsten-marr)). The lab's NCA line is co-led with **Ario Sadafi** (also Helmholtz Munich + TUM CAMP under Nassir Navab) and is staffed by **Michael Deutges**, **Chen Yang**, and **Raheleh Salehi**. **Daniel M. Lang** is a Helmholtz Munich collaborator under Julia Schnabel; he is not strictly MarrLab personnel but works at the same institute and on closely adjacent NCA medical-imaging problems. The treatment of "TeNCA" below reflects that adjacency.

### 1.1 WBC-NCA — Deutges, Sadafi, Navab, Marr (MICCAI 2024)

- **Title / venue:** *Neural Cellular Automata for Lightweight, Robust and Explainable Classification of White Blood Cell Images*. **[peer-reviewed]** MICCAI 2024 LNCS 15003, pp. 693–702 ([doi.org/10.1007/978-3-031-72384-1_65](https://doi.org/10.1007/978-3-031-72384-1_65)). arXiv [2404.05584](https://arxiv.org/abs/2404.05584). Repo [github.com/marrlab/WBC-NCA](https://github.com/marrlab/WBC-NCA).
- **Problem domain / data:** Single-cell classification of WBCs in peripheral-blood-smear microscopy. Three datasets used (Matek-19, Acevedo-20, INT-20), with cross-domain transfer central to the evaluation.
- **NCA architecture:** Image-classification NCA (this is, per the authors, "to the best of our knowledge, the first application of NCA for image classification" — the architectural novelty over prior work is that classification is read from NCA features, not produced by an external head over conv features). Each image pixel is a cell; perception by hard-coded Sobel/identity kernels; an MLP update rule; iterative updates; the highest 10% of pixel activations per channel are mean-pooled into a low-dimensional embedding which is fed to a small fully-connected classifier (this pooling rule is reused across the lab's later papers).
- **Specific contribution:** First NCA-based classifier (concurrent with but distinct from Med-NCA-style segmentation), with explicit emphasis on parameter efficiency (orders-of-magnitude smaller than ResNet/EfficientNet baselines) and out-of-domain robustness when train/test datasets come from different acquisition pipelines.
- **Reported strengths:** Competitive accuracy at a fraction of the parameters; large robustness gain under domain shift; pixel-level saliency arises naturally from the iterative dynamics, providing inherent explainability rather than post-hoc CAM.
- **Limitations:** Does not match the absolute-best in-domain accuracy of larger backbones; convergence sensitive to the number of iterations and to image scale; classification performance per the follow-on aNCA work was below CNN baselines on harder microscopy datasets.
- **Compute footprint:** Paper reports the model is "significantly smaller in terms of parameters" than baselines; specific parameter count not pulled exactly here but on the order of ~10^4–10^5 [inference]. Training was on standard single-GPU consumer hardware.
- **Reproducibility audit:** Code at [github.com/marrlab/WBC-NCA](https://github.com/marrlab/WBC-NCA). README is brief but training scripts and configs are present. Datasets are public (Matek-19, Acevedo-20). Weights are not obviously released; retraining from scratch is the expected path. Rerun feasibility: medium [inference].

### 1.2 Hierarchical NCA — Yang, Deutges, Navab, Sadafi, Marr (IPMI 2025)

- **Title / venue:** *Hierarchical Neural Cellular Automata for Lightweight Microscopy Image Classification*. **[peer-reviewed]** IPMI 2025, LNCS 15829 ([doi.org/10.1007/978-3-031-96628-6_2](https://doi.org/10.1007/978-3-031-96628-6_2)). Yang and Deutges share first authorship; Marr and Sadafi share last.
- **Architecture:** Stacks NCAs at multiple resolutions, propagating learned features across scales rather than running a single NCA at the input resolution. Conceptually related to (but distinct from) Pande & Grattarola's hierarchical NCA — the IPMI 2025 paper is a microscopy-classification instantiation rather than a generative one.
- **Contribution over WBC-NCA:** Improves both accuracy and parameter efficiency on microscopy classification by allowing global context to flow through coarse-scale NCA layers before fine-scale refinement. Aligned in spirit with Med-NCA's two-step downscale → patch trick ([arXiv 2302.03473](https://arxiv.org/abs/2302.03473)) but with the second stage itself being NCA.
- **Reproducibility audit:** No standalone repo surfaced; the proceedings volume is the canonical reference. **Not found:** a public code release dedicated to this paper. Reuses the WBC-NCA pipeline conceptually.

### 1.3 aNCA — Attention-pooled NCA (arXiv 2508.12324, 2025)

- **Title / venue:** *Attention Pooling Enhances NCA-based Classification of Microscopy Images*. **[preprint]** arXiv [2508.12324](https://arxiv.org/abs/2508.12324). Repo [github.com/marrlab/aNCA](https://github.com/marrlab/aNCA).
- **Architecture:** Replaces the "top-10% mean per channel" pooling of WBC-NCA with a learned attention-pooling head over the final NCA state. Otherwise the NCA stack matches WBC-NCA / hierarchical NCA conventions (Sobel + identity perception; small MLP update; ~10²–10³ steps).
- **Specific contribution:** Closes the in-domain accuracy gap to CNN baselines on multiple microscopy datasets while keeping the lightweight footprint of NCAs; ablations show pooling is the main bottleneck rather than the CA dynamics.
- **Reproducibility audit:** README provides conda env (`env_nca.yml`), `train.py` with `--mode train --predict aNCA --output ... --train_set ... --fold ...`. Datasets are external. **Rerun feasibility: high** assuming you have a microscopy dataset in the expected layout.

### 1.4 NCA-WSS — Deutges, Yang, Salehi, Navab, Marr, Sadafi (MICCAI 2025)

- **Title / venue:** *Neural Cellular Automata for Weakly Supervised Segmentation of White Blood Cells*. **[peer-reviewed]** MICCAI 2025, LNCS volume in [Springer chapter 10.1007/978-3-032-13961-0_29](https://doi.org/10.1007/978-3-032-13961-0_29). arXiv [2508.12322](https://arxiv.org/abs/2508.12322). Repo [github.com/marrlab/NCA-WSS](https://github.com/marrlab/NCA-WSS) (linked from the paper).
- **Architecture:** Train an NCA classifier with image-level labels only (the WBC-NCA recipe); then extract segmentation masks from intermediate NCA feature maps via principal-component analysis applied per-channel. No retraining with pixel labels.
- **Contribution:** Weakly supervised segmentation entirely without CAM/Grad-CAM. Treats the NCA's iterative refinement of feature channels as an implicit localization signal — i.e., "what the cells decided to attend to" — and demonstrates significant outperformance vs. existing WSSS baselines on three WBC datasets, including cross-domain.
- **Strengths:** No retraining; PCA on activations yields interpretable channel-level structure aligning with cytoplasm/nucleus/background. Cross-dataset generalization of the segmentation masks is competitive.
- **Limitations:** Segmentation quality bounded by classifier feature quality; the PCA step is unsupervised and has no notion of part semantics, so labels must be assigned to PCA components manually or via simple heuristics.
- **Reproducibility audit:** Code published at github.com/marrlab/NCA-WSS per the paper. Datasets identical to WBC-NCA.

### 1.5 TeNCA — Lang, Osuala, Spieker, Lekadir, Braren, Schnabel (MICCAI 2025)

Lang's affiliation is Helmholtz Munich + TUM (Schnabel group). MarrLab-adjacent rather than MarrLab proper, but the work is the most explicit example of NCA *as a temporal forward model* in the broader Helmholtz Munich orbit.

- **Title / venue:** *Temporal Neural Cellular Automata: Application to modeling of contrast enhancement in breast MRI*. **[peer-reviewed]** MICCAI 2025, LNCS 15963 pp. 604–614 ([Springer chapter](https://doi.org/10.1007/978-3-032-04965-0_57)). arXiv [2506.18720](https://arxiv.org/abs/2506.18720). Repo [github.com/LangDaniel/TeNCA](https://github.com/LangDaniel/TeNCA).
- **Problem:** Synthesize post-contrast breast-MRI sequences from a pre-contrast acquisition. The ground-truth sequence is temporally **sparse and non-uniformly sampled**, which is exactly the regime where standard Mordvintsev-style NCAs struggle (they are trained on unit-step rollouts).
- **Architecture:** Standard NCA backbone with two extensions: (i) **adaptive loss computation** — the loss is evaluated only at acquired time points, with the NCA's iteration count rescaled to match the physical time interval; (ii) the iterative depth is interpreted as physical time, conditioning the model toward physiologically plausible trajectories.
- **Contribution:** First MICCAI-track demonstration that NCAs beat state-of-the-art image-to-image baselines (U-Net + batch-norm trained on MAE) on a *temporal-evolution-consistency* metric, not just per-frame fidelity. This is exactly the system-identification problem framed against unevenly sampled imaging data.
- **Reproducibility:** Code public; preprocessing scripts provided. Dataset is institutional and not freely redistributable (typical for breast MRI), so external reproduction requires substituting a different DCE-MRI cohort.

### 1.6 The MarrLab GitHub organization — repo-level audit

Direct browsing of the [marrlab](https://github.com/marrlab) org via the API was blocked in this environment, but search hits surface the following NCA-relevant repos:

| Repo | Paper | State |
|---|---|---|
| [marrlab/WBC-NCA](https://github.com/marrlab/WBC-NCA) | Deutges 2024 | Released |
| [marrlab/aNCA](https://github.com/marrlab/aNCA) | aNCA preprint 2025 | Active |
| [marrlab/NCA-WSS](https://github.com/marrlab/NCA-WSS) | Deutges 2025 | Released |

**Not found:** a public marrlab repo for the IPMI 2025 hierarchical-NCA paper, or any NCA repo flagged "in-progress / unpublished" beyond these three. If such repos exist they are private. [inference] Given the cadence (MICCAI 2024, IPMI 2025, MICCAI 2025, aNCA preprint), expect a continuation paper in the 2026 cycle, plausibly a 3D / temporal variant or a foundation-model distillation; this is a guess, not a finding.

### 1.7 The lab's NCA thesis

Across these five projects, the consistent claim is: **NCAs are the right backbone for clinical microscopy and DCE imaging because they trade absolute accuracy for (parameters, OOD-robustness, native interpretability, temporal-consistency) — and clinicians need the second basket.** The contributions stack along that axis: first show NCA classification works at all (WBC-NCA), then improve in-domain accuracy via attention pooling (aNCA) and via hierarchy (Yang IPMI 2025), then exploit the iterative dynamics for label-free segmentation (NCA-WSS), then port to the sparse-temporal regime (TeNCA).

This is **not** the Schumacher-style "system identification of biological dynamics" agenda. The lab is using NCAs as **lightweight, interpretable image models**, not as PDE-discovery tools. That distinction matters for Section 4: the DLCA replacement story is more compelling for the lab's own line of work (where discrete logic + edge deployment is a real win) than for, say, Sottoriva's MNCA (where intrinsic stochasticity is the contribution).

---

## Section 2 — Broader NCA landscape

### 2.1 Citation lineage

The current NCA tree branches from Mordvintsev, Randazzo, Niklasson & Levin's *Growing Neural Cellular Automata* (Distill, 2020, [doi 10.23915/distill.00023](https://distill.pub/2020/growing-ca/)). The Distill series itself produced four follow-ons before the field diversified: *Self-classifying MNIST Digits* ([10.23915/distill.00027.002](https://distill.pub/selforg/2020/mnist/)), *Self-Organising Textures* ([10.23915/distill.00027.003](https://distill.pub/selforg/2021/textures/)), *Adversarial Reprogramming of NCA* ([10.23915/distill.00027.004](https://distill.pub/selforg/2021/adversarial/)), and Niklasson, Mordvintsev, Randazzo's *Asynchronicity in Neural Cellular Automata* (ALIFE 2021, [doi 10.1162/isal_a_00461](https://doi.org/10.1162/isal_a_00461)). The asynchronicity paper is conceptually decisive — it establishes the framing of NCA training as **system identification of a PDE under specified boundary conditions**, which is the framing that all of Schumacher, Sottoriva, Béna and DLCA later inherit.

From there, four somewhat-independent threads:

- **Texture / shape generation** (Mordvintsev–Niklasson, Risi group): µNCA, MeshNCA, Variational NCA, Goal-Guided NCA, EngramNCA.
- **Medical imaging** (Mukhopadhyay group at TU Darmstadt and MarrLab/Schnabel at Helmholtz Munich): Med-NCA, M3D-NCA, Diff-NCA / FourierDiff-NCA, MedSegDiffNCA, NCAdapt, WBC-NCA, NCA-WSS, TeNCA.
- **Mechanistic biology / PDE discovery** (Schumacher Edinburgh, Sottoriva Human Technopole): Richardson 2024 PLOS Comp Bio, MNCA 2025.
- **Universality and discrete computation** (Béna/Faldor/Cully/Goodman Imperial; Mordvintsev/Miotti/Niklasson/Randazzo Google → DLCA; Stovold; Xu/Miikkulainen).

### 2.2 Group table

| Group / PI | Institution | Methodological angle | Domain | Overlap with MarrLab | Key paper |
|---|---|---|---|---|---|
| Mordvintsev / Niklasson / Randazzo / Levin | Google Paradigms of Intelligence; Tufts (Levin) | Foundational NCA, async, isotropic, texture, **DLCA** | Generative / morphogenesis / programmable matter | Architectural ancestor of every MarrLab paper | Mordvintsev 2020 [Distill](https://distill.pub/2020/growing-ca/); Niklasson 2021 [ALIFE async](https://doi.org/10.1162/isal_a_00461); **Miotti 2025 [arXiv 2506.04912](https://arxiv.org/abs/2506.04912)** |
| Risi group | IT University of Copenhagen | Goal-conditioned, hyperNCA, RL-NCA, controllable self-organization | ALife, robotics, generative | Methodological cross-pollination via async + control | Sudhakaran, Najarro, Risi 2022 [arXiv 2205.06806](https://arxiv.org/abs/2205.06806); Najarro et al. HyperNCA 2022 |
| Mukhopadhyay (MEC Lab) | TU Darmstadt | NCA segmentation, 3D NCA, NCA + diffusion, continual NCA | Medical imaging | Direct competitor / collaborator on medical NCA | Kalkhof 2023 Med-NCA [arXiv 2302.03473](https://arxiv.org/abs/2302.03473); M3D-NCA 2023; Diff-NCA / FourierDiff-NCA [Nature npj 2025](https://www.nature.com/articles/s44335-025-00026-4) |
| Schumacher group | University of Edinburgh | NCA as PDE solver / system-identifier; Turing patterns; symmetry constraints | Mathematical biology / morphogenesis | Methodologically the closest external lab to "NCAs as mechanistic models" framing | Richardson, Antal, Blythe, Schumacher 2024 [PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011589); arXiv [2310.14809](https://arxiv.org/abs/2310.14809); repo [AlexDR1998/NCA](https://github.com/AlexDR1998/NCA) and JAX successor `Differentiable-Patterning` |
| Sottoriva group | Human Technopole, Milan | Mixture of NCAs (MNCA): probabilistic rules + intrinsic noise | Cancer growth, tissue heterogeneity, microscopy seg. | Closest to MarrLab on biological microscopy; complementary on stochasticity | Milite, Caravagna, Sottoriva 2025 [arXiv 2506.20486](https://arxiv.org/abs/2506.20486), ICML GenBio workshop |
| Béna / Faldor / Cully / Goodman | Imperial College London | Universal-computation NCA, immutable substrate as "hardware" channel | Computational primitives, MNIST-in-CA, GNN-NCA hybrid | Methodological — the universality angle; CAX library ecosystem | Béna 2025 [arXiv 2505.13058](https://arxiv.org/abs/2505.13058), GECCO '25 Companion. CAX accelerated lib: Faldor & Cully 2024 |
| Stovold | Lancaster University Leipzig | Signal response, identity layers for stability, ALife semantics | Artificial morphogenesis | Methodological — stability and signaling | [arXiv 2305.12971](https://arxiv.org/abs/2305.12971) (ALIFE 2023); [arXiv 2508.06389](https://arxiv.org/abs/2508.06389) (ALIFE 2025) |
| Miikkulainen group | UT Austin | Few-shot abstract reasoning, gradient-trained NCAs on grid puzzles | ARC-AGI | Methodological — task-conditioned NCA | Xu & Miikkulainen 2025 [arXiv 2506.15746](https://arxiv.org/abs/2506.15746), IEEE SSCI / ALIFE 2025 |
| Nichele group (Guichard, Reimers, Kvalsund, Lepperød) | Østfold UC / U Oslo / Simula | EngramNCA + ARC-NCA: hidden memory states for developmental ARC | ARC-AGI | Concurrent ARC-NCA approach to Xu/Miikkulainen | [arXiv 2505.08778](https://arxiv.org/abs/2505.08778), ALIFE 2025; [arXiv 2504.11855](https://arxiv.org/abs/2504.11855) for EngramNCA |
| Magnus Petersen (FIAS Frankfurt) | Goethe Univ. Frankfurt | Multi-neighborhood NCA with structured-noise seeds | Texture / generative | Tangential | [arXiv 2311.16123](https://arxiv.org/abs/2311.16123) **[preprint]** |
| Grattarola, Gala, Quaeghebeur | TU Eindhoven / independent | Graph NCA, E(n)-equivariant Graph NCA | Graph dynamics, point clouds | Methodological — relaxes lattice assumption | Grattarola 2021 *Learning Graph Cellular Automata*; Gala et al. *E(n)-equivariant GNCA* TMLR 2024 ([OpenReview](https://openreview.net/forum?id=7PNJzAxkij)) |
| Tesfaldet / Pal / Nowrouzezahrai | McGill | Attention-based NCA (ViTCA-style local self-attention update rule) | Image recovery | Architectural cross-cite in MarrLab papers | Tesfaldet et al. NeurIPS 2022 |
| Petersen (Felix), Stanford | Stanford | Differentiable Logic Gate Networks (the DLGN substrate underneath DLCA) | Hardware-efficient classification | Methodological prerequisite for DLCA | NeurIPS 2022 [DLGN](https://papers.neurips.cc/paper_files/paper/2022/hash/0d3496dd0cec77a999c98d35003203ca-Abstract-Conference.html); NeurIPS 2024 [Convolutional DLGN](https://proceedings.neurips.cc/paper_files/paper/2024/hash/db988b089d8d97d0f159c15ed0be6a71-Abstract-Conference.html) |

### 2.3 Most-cited and most-underrated

**Most-cited (by counts visible in Semantic Scholar / Google Scholar as of April 2026, [inference] from secondary aggregator hits since I did not pull citation counts directly):**

1. Mordvintsev et al. 2020 — *Growing Neural Cellular Automata* (Distill). The canonical reference; cited by essentially every paper in this report.
2. Randazzo et al. 2020 — *Self-classifying MNIST Digits* (Distill). The classification analog of Mordvintsev 2020.
3. Niklasson et al. 2021 — *Asynchronicity in Neural Cellular Automata* (ALIFE 2021). Establishes the PDE/system-id framing and async-update protocol used widely thereafter.
4. Kalkhof et al. 2023 — *Med-NCA* (IPMI 2023). The medical-imaging gateway.
5. Sandler et al. 2020 — *Image Segmentation via Cellular Automata* (arXiv 2008.04965). Predates Med-NCA and is the architectural template Med-NCA refines.

**Most-underrated [inference, based on citation density relative to methodological importance]:**

1. Niklasson, Mordvintsev, Randazzo 2021 — *Asynchronicity in Neural Cellular Automata*. Almost everyone cites Mordvintsev 2020 and forgets the async paper that actually says "this is system identification of a PDE."
2. Richardson, Antal, Blythe, Schumacher 2024 — explicitly compares NCAs to SINDy and EDMD. The closest published statement of the system-identification framing the user works in.
3. Grattarola, Livi, Alippi 2021 — *Learning Graph Cellular Automata*. Decouples the lattice assumption; the natural prerequisite for any DLCA-on-graph work.
4. Stovold 2023/2025 — signaling and identity. Identifies a real failure mode of the original NCA (boundary instability) and offers a one-channel fix; under-cited because ALIFE.
5. Tesfaldet et al. 2022 — *Attention-based NCA* (NeurIPS). The single best demonstration that the perception kernel can be learned, and the architectural ancestor of MarrLab's aNCA.

---

## Section 3 — DLCA / DiffLogic CA

### 3.1 Anchor paper

Miotti, Niklasson, Randazzo, Mordvintsev (2025), *Differentiable Logic Cellular Automata: From Game of Life to Pattern Generation*. **[peer-reviewed]** ALIFE 2025 ([doi 10.1162/ISAL.a.882](https://direct.mit.edu/isal/proceedings/isal2025/37/54/134069)) and **[preprint]** arXiv [2506.04912](https://arxiv.org/abs/2506.04912). Code in the diffLogic_CA.ipynb notebook at [github.com/google-research/self-organising-systems](https://github.com/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb). Project page: [google-research.github.io/self-organising-systems/difflogic-ca](https://google-research.github.io/self-organising-systems/difflogic-ca/) **[blog/demo]**.

DLCA replaces the perception-Sobel + MLP-update of a standard NCA with two stacks of **differentiable logic gates** (Petersen et al. 2022/2024). Each cell maintains discrete bits in registers (described as "gray" and "orange" in the project page); during training each gate is a softmax over the 16 binary 2-input logic operations, relaxed to operate in [0,1]; at inference each gate snaps to its argmax operation and the entire network executes as a recurrent boolean circuit with discrete cell states.

### 3.2 Prerequisite literature

- **Petersen et al. 2022 — *Deep Differentiable Logic Gate Networks* [peer-reviewed], NeurIPS 2022.** [PDF](https://papers.neurips.cc/paper_files/paper/2022/file/0d3496dd0cec77a999c98d35003203ca-Paper-Conference.pdf), arXiv [2210.08277](https://arxiv.org/abs/2210.08277). Establishes the gate-relaxation trick: each gate maintains a probability distribution over the 16 two-input boolean operations, gradient-trained, then crystallized. Reports >1 million MNIST inferences/s on a single CPU core.
- **Petersen et al. 2024 — *Convolutional Differentiable Logic Gate Networks* [peer-reviewed], NeurIPS 2024.** [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/db988b089d8d97d0f159c15ed0be6a71-Paper-Conference.pdf), arXiv [2411.04732](https://arxiv.org/abs/2411.04732). Adds convolutional kernel structure; crucially, demonstrates **<10 ns per CIFAR-10 image on a Xilinx XC7Z045 FPGA** with placement diagrams, and shows that splitting the gate graph into k/8 grouped sub-circuits avoids routing congestion without hurting accuracy. This is the FPGA paper DLCA cites as "inference measured in nanoseconds."
- **Mordvintsev et al. 2020 + Niklasson et al. 2021 (async).** Standard NCA architecture and the async-update protocol that DLCA inherits (the DLCA paper explicitly explores async updates following Niklasson 2021a).
- **Wolfram-class universality.** Cited as motivation: the design choice to keep cell state strictly discrete and the update strictly boolean is what makes it possible to talk about Wolfram-style classification of the learned circuit at all.

### 3.3 Demonstrated tasks

The arXiv version reports four:

1. **Conway's Game of Life rules learned end-to-end.** Confirmed: the learned circuit (Figure 4 in the paper) reproduces GoL for 20+ steps. Architecture: 16 perception kernels, three perception layers of [8, 4, 2] gates each, update network of 16 layers — 10 layers of 256 gates followed by [128, 64, 32, 16, 8, 8] gates.
2. **Checkerboard patterns** with resilience to noise and damage — both synchronous and asynchronous variants succeed.
3. **Lizard shape growth** (the equivalent of Mordvintsev's lizard from Growing NCA).
4. **Multi-color pattern generation** — 64-dimensional cell state, four perception kernels each with a 3-layer [8, 4, 2] structure.

The project page additionally walks through these demos interactively. **Not found:** any DLCA demonstration on MNIST classification, on textures, on 3D structures, or on biological / medical data, as of April 2026.

### 3.4 Training stability and failure modes

The paper documents:

- **Asynchronous updates work for simple patterns but degrade for complex ones.** Explicit quote from the arXiv: "For simpler patterns, we observe success with both synchronous and asynchronous updates"; the multi-color case is reported only with synchronous updates.
- **Soft-to-discrete inference snap.** The model is trained on continuous gate distributions and crystallized at inference. The paper measures error as the sum of absolute differences between the target and reconstructed pattern; it reports successful crystallization for the demonstrated tasks but does **not** quantify the train→inference accuracy gap as a sweep over circuit depth or gate count. This is the most important under-reported number for a follower [inference].
- **Convergence sensitivity.** The paper notes that using kernels with multiple bits of output per channel (instead of one) "improves convergence in some cases" — a form of architectural regularization with no quantitative ablation in the public manuscript.
- **Self-healing regime.** A widely-quoted result from the project page is a 6-gate self-healing circuit; the news write-up at [DeepNewz](https://deepnewz.com/ai-modeling/google-introduces-differentiable-logic-cellular-automata-replicating-conway-s-7817a7f7) **[blog/demo]** highlights this; it is genuine but the formal paper emphasizes more complex networks for the lizard / multi-color tasks.

### 3.5 Hardware angle

This is the most significant gap in the literature.

- **DLGN ancestor on FPGA.** Petersen 2024 demonstrates convolutional DLGN inference on Xilinx Artix-7 / XC7Z045 FPGAs at <10 ns per CIFAR-10 image with full LUT placement.
- **Recurrent DLGN.** Bührer et al. 2025 *Recurrent Deep Differentiable Logic Gate Networks* (arXiv [2508.06097](https://arxiv.org/abs/2508.06097)) **[preprint]** introduces sequential DLGNs (flip-flops, latches) reportedly delivering RNN/Transformer-comparable WMT'14 translation with ~20,000× fewer logic ops. This is directly relevant: DLCA *is* a recurrent DLGN (the cell update is recurrent across time and shared across space), but the two papers do not yet cite each other, and no joint FPGA implementation has appeared.
- **Light DLGN.** *Light Differentiable Logic Gate Networks* (arXiv [2510.03250](https://arxiv.org/abs/2510.03250), 2025) **[preprint]** reparametrizes DLGNs for 4× memory reduction and 8.5× faster training — also relevant but again not specifically wired to DLCA.
- **LILogic Net** (arXiv [2511.12340](https://arxiv.org/abs/2511.12340), 2025) **[preprint]** addresses learnable connectivity in LGNs for hardware deployment.
- **DLCA itself on FPGA / ASIC.** **Not found.** The DLCA paper writes "could naturally map to FPGA or other specialized hardware" and cites Petersen 2022 for nanosecond inference, but **no FPGA synthesis of a trained DLCA has been published as of April 2026** in the sources surveyed. The classical FPGA-CA literature (Tsoutsouras et al. 2021 *Large-scale Cellular Automata on FPGAs* — TRETS [doi 10.1145/3423185](https://dl.acm.org/doi/10.1145/3423185), and earlier CAM-8) has not yet been bridged to learned DLCA circuits. **No published gate count, no published power profile, no LUT-utilization measurement for a trained DLCA.** This is the single most exploitable gap for a hardware-leaning DLCA chapter.

### 3.6 Non-Google follow-on DLCA work

**Not found** as of April 2026. DLCA appears in third-party listings (e.g., MECLabTUDA's awesome-nca catalog, the Hacker News thread [item 43286161](https://news.ycombinator.com/item?id=43286161)) and is cited by survey-adjacent recurrent-DLGN papers, but no independent group has yet published a DLCA training run, a DLCA application, or a DLCA hardware deployment. There are non-DLCA recurrent-DLGN papers (Bührer 2025) and non-recurrent DLGN-FPGA work (Bacellar et al., cited inside Petersen 2024), but neither closes the cellular-recurrent + boolean + hardware loop.

### 3.7 NCA properties tested vs. untested in DLCA

| NCA property | Tested in DLCA? |
|---|---|
| Regenerative robustness (mask-and-regrow) | **Yes** — checkerboard pattern under noise and damage |
| Synchronous training | **Yes** — all four demonstrations |
| Asynchronous updates | **Partially** — works for simple patterns, not all multi-color cases |
| Stochastic dynamics | **No** — DLCA is deterministic at inference; the only randomness source is async-update sampling. No Sottoriva-style intrinsic-noise experiment. |
| Scale generalization (train on 64×64, infer on 128×128) | **Implicitly yes** for some pattern tasks (the lizard and checkerboard appear scale-permissive in the demos), but not benchmarked. |
| Domain shift / OOD | **No** — there is no DLCA equivalent of the WBC-NCA cross-dataset evaluation. |
| Conditional generation / goals (à la Sudhakaran 2022) | **No** |
| Graph / non-lattice topology | **No** — DLCA is fixed-grid 2D Moore neighborhood |
| Continuous PDE identification (à la Schumacher) | **No** — discrete cell states preclude direct PDE recovery |
| Hidden-memory channels (à la EngramNCA) | **No** |
| Hierarchical / multi-scale (à la Yang 2025 IPMI) | **No** |

---

## Section 4 — Intersection and gap analysis

The substantive question for someone working on DLCA is: where does an NCA → DLCA replacement plausibly land, and where will it fail?

| Project / branch | DLCA precedent? | Parameter-budget fit | Discrete-inference fit | Known blockers | Nearest published analog |
|---|---|---|---|---|---|
| **MarrLab WBC-NCA** (image classification, OOD-robust) | None | Excellent — WBC-NCA already targets compactness; DLCA is even smaller per active path | Mixed — classification softmax is real-valued, would need a bit-counting head as in DLGN | Need to map 24-bit RGB cell input into a discrete state (precision loss); top-10% activation pooling has no clean discrete analog | DLGN on CIFAR-10 (Petersen 2024); MarrLab aNCA |
| **MarrLab aNCA** (attention-pooled classifier) | None | Good | Poor — attention's softmax-over-floats is incompatible with strict discrete inference | The pooling head is the contribution; DLCA-ifying it dilutes the contribution | None published |
| **MarrLab NCA-WSS** (PCA over feature channels) | None | Good | Poor — PCA assumes continuous channels with covariance structure | Discrete state space breaks PCA; would need a discrete clustering proxy | None published |
| **MarrLab TeNCA** (sparse temporal MRI) | None | Good in principle | Poor in practice — DCE intensity values are real-valued and physiology demands smooth interpolation | Quantizing MRI intensities to bits while preserving contrast is non-trivial | None published; closest analogy is recurrent-DLGN (Bührer 2025) |
| **Schumacher PDE identification** | None | Inadequate — PDEs require continuous gradients to recover diffusion coefficients | Very poor — the system being identified is continuous; discrete state precludes recovering Gray-Scott parameters | Fundamental mismatch: SINDy-class equation discovery is *about* continuous coefficients | None — this is arguably outside DLCA's reachable scope |
| **Sottoriva MNCA** (stochastic biological growth) | None | Good | Mixed — a stochastic DLCA would need a probabilistic gate, which is a research project | Stochasticity in DLCA exists only via async sampling (Bernoulli per cell), not via state noise | None published |
| **Béna UNCA** (universal continuous CA) | None — but **the closest conceptual cousin** | Not directly comparable | Excellent — discrete = universal computation by definition | None obvious; the two papers are complementary — one continuous, one discrete | Béna 2025 GECCO Companion |
| **Risi GoalNCA / EngramNCA** (controllable / memory) | None | Good | Mixed — the goal vector or hidden memory needs a discrete encoding | Memory across iterations in DLCA is whatever bits the cell carries; capacity must be designed | None published |
| **Stovold ALIFE 2023/2025** (signaling, identity) | None | Good | Reasonable — identity is naturally discrete (a small bit-tag) | Stability hacks designed for continuous NCAs may not port literally | None published |
| **Xu / Miikkulainen ARC-NCA** | None | **Best fit of any branch** — ARC grids are already discrete colors | **Excellent** — ARC is intrinsically a discrete-state, grid-rule problem | Memory states (EngramNCA) need to be expressible in bits; few-shot meta-learning over circuits is research | None published |
| **Grattarola / Gala graph-NCA** | None | TBD | Clean conceptually if graph DLGN existed | No graph DLGN architecture has been published; would need to invent perception over node neighborhoods in pure logic | None published |
| **Mukhopadhyay Med-NCA / M3D-NCA** (medical seg) | None | Good | Poor — segmentation softmax is continuous | Output decoding | None published |

**Key gap statements:**

- **No published DLCA work on medical imaging or biological data.** WBC-NCA → DLCA is the most concretely tractable PhD-chapter project: data is already at low-resolution, classification heads are discrete, and the FPGA-deployment story would be genuine (point-of-care leukemia screening as a hardware demo).
- **No published DLCA work on system identification of observed dynamics.** DLCA's discrete state space is **not** a drop-in for Schumacher's PDE-recovery agenda; the right analogy is "learn the *symbolic* update rule of a discrete process" — close to learning Wolfram-class rules from data, which is a live but unaddressed problem.
- **No published parameter / interpretability / robustness comparison** between trained NCAs and trained DLCAs on equivalent tasks. Game of Life is solved by both, but with very different architecture sizes and very different interpretability profiles (the DLCA circuit is literally a circuit; the NCA's MLP is not). A clean head-to-head benchmark across regenerative robustness, scale generalization, async tolerance and parameter count does not exist.
- **Known carry-over from NCA → DLCA:** regenerative behavior (checkerboard demo); per-cell locality (definitionally preserved); async tolerance for simple patterns.
- **Known gaps in the NCA → DLCA translation:** continuous PDE recovery; stochastic dynamics from intrinsic noise; smooth interpolation between targets; soft attention; differentiable hierarchy; PCA-style feature analysis.
- **Published negative results at the intersection:** None found. The closest is the DLCA paper's own observation that async updates fail for the multi-color task — but this is reported in passing without quantification or ablation.

---

## Section 5 — Open questions

Format: **Question → why it matters → what's needed.** Bias toward 6-month / single-PhD-chapter scope.

1. **What is the smallest DLCA circuit that fully reproduces Conway's GoL?** → The current circuit is 16 perception kernels × ~6,000 gates; minimal-gate circuits would set a hardware lower bound. → A pruning sweep with structural retraining; some hours of A100 time.
2. **Quantify the soft→discrete crystallization gap.** → DLCA training accuracy ≠ inference accuracy and the paper does not report the gap as a curve over circuit depth. → Reproducible benchmark across all four DLCA tasks; tabulate gap vs. gate count.
3. **Does a DLCA replicate WBC-NCA's OOD-robustness?** → If yes, you have a sub-100 µW point-of-care leukemia detector; if no, the medical NCA story may not transfer. → The Matek-19 / Acevedo-20 / INT-20 datasets, the WBC-NCA codebase, and ~2 weeks of training; a discrete-friendly classification head needs to be designed.
4. **DLCA on ARC-AGI.** → ARC's discrete colors and grid structure align with DLCA's primitives more cleanly than with continuous NCAs; combining EngramNCA-style hidden bits with DLCA gates is plausible. → ARC public eval set; CAX (Faldor & Cully 2024) as the JAX harness.
5. **Synthesize a trained DLCA to FPGA and report gate count, LUT utilization and power.** → This is the "programmable matter" claim's actual test. → Xilinx Vivado/Vitis flow; Petersen 2024's k/8-grouping trick; a Pynq board.
6. **Does intrinsic stochasticity emerge from a probabilistic-gate DLCA?** → Bridges DLCA to Sottoriva's MNCA agenda for cancer-growth modeling. → A reformulation where each gate samples from its 16-op categorical distribution at inference, not just at training; KL-divergence loss against observed stochastic trajectories.
7. **Can DLCA learn the rule of a non-trivial elementary CA (Rule 110, Rule 30) from observation?** → Direct test of the "learn a Wolfram-class rule from data" framing; cleaner than GoL because 1D and provably universal (Rule 110). → Days of compute; obvious benchmark for a system-identification chapter.
8. **What is the asynchronous-update phase diagram of DLCA?** → The paper notes async fails for complex patterns but does not characterize when. → Sweep update probability p ∈ (0, 1] across all four tasks; produce the empirical phase boundary.
9. **DLCA on graph topology.** → Combining Grattarola's GNCA with DLGN gates would lift the lattice constraint; useful for Sottoriva-style tumor-graph or epidemiology-graph models. → Architectural design (perception over node neighborhoods using boolean primitives is non-obvious); likely 6 months.
10. **Hidden-memory DLCA (EngramNCA-style) — does it improve circuit reachability?** → EngramNCA showed memory bits help on ARC; the analog in DLCA is private register channels never used in perception. → Modify the open-source DLCA notebook; benchmark on lizard + multi-color.
11. **Identity layers (Stovold 2025) for DLCA stability.** → Stovold's identity tag improves NCA boundary stability; in DLCA the analog is one or two reserved bits per cell. → A small direct port and an ALIFE-format paper.
12. **Does soft-DLCA → discrete-DLCA crystallization preserve regenerative behavior under damage?** → Standard NCA regeneration is partially due to noise tolerance from continuous interpolation; with strict booleans, regeneration may collapse. → Damage-and-recover protocol applied at multiple snap thresholds.
13. **Hardware-aware DLCA training.** → Add gate-count and LUT-routing penalties to the loss directly, à la Petersen's grouped-convolution strategy. → Reuses Petersen 2024's penalty machinery; estimate weeks.
14. **DLCA for sparse temporal sequence modeling (TeNCA-equivalent).** → Can a discrete-state recurrent circuit model contrast-enhancement dynamics? Likely not at full intensity precision, but maybe at a quantized tier. → Public DCE-MRI dataset (DUKE-Breast-Cancer-MRI); compare against TeNCA.
15. **Symbolic extraction and Wolfram-class classification of trained DLCAs.** → A trained DLCA is *literally a finite-state circuit*; it should be amenable to symbolic minimization (Espresso, ABC) and to the Wolfram-class taxonomy. Nobody has reported this. → Logic-synthesis tooling + behavioral classification; ~1 month for GoL, longer for the lizard.

---

## Section 6 — Reading order

| # | Paper | Why read it now |
|---|---|---|
| 1 | Mordvintsev et al. 2020, *Growing Neural Cellular Automata* — Distill **[blog/demo, peer-edited]** [link](https://distill.pub/2020/growing-ca/) | The architectural ground truth. Read the interactive figures, not just the text. |
| 2 | Niklasson, Mordvintsev, Randazzo 2021, *Asynchronicity in Neural Cellular Automata* — ALIFE 2021 **[peer-reviewed]** [doi](https://doi.org/10.1162/isal_a_00461) | Establishes the system-identification framing and async protocol. Most under-cited foundational paper. |
| 3 | Petersen et al. 2022, *Deep Differentiable Logic Gate Networks* — NeurIPS 2022 **[peer-reviewed]** [PDF](https://papers.neurips.cc/paper_files/paper/2022/file/0d3496dd0cec77a999c98d35003203ca-Paper-Conference.pdf) | The gate-relaxation trick. Read for the math, not the MNIST numbers. |
| 4 | Petersen et al. 2024, *Convolutional Differentiable Logic Gate Networks* — NeurIPS 2024 **[peer-reviewed]** [arXiv 2411.04732](https://arxiv.org/abs/2411.04732) | The FPGA paper. The placement diagrams and the k/8 grouping trick are the practical hardware blueprint. |
| 5 | Miotti, Niklasson, Randazzo, Mordvintsev 2025, *DiffLogic CA* — ALIFE 2025 **[peer-reviewed]** + arXiv [2506.04912](https://arxiv.org/abs/2506.04912) | The bridge. Read alongside the [project page](https://google-research.github.io/self-organising-systems/difflogic-ca/) and the Colab. |
| 6 | Richardson, Antal, Blythe, Schumacher 2024, PLOS Comp Bio **[peer-reviewed]** [link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011589) | Most explicit articulation of NCA-as-system-identification, with a direct SINDy / EDMD comparison. |
| 7 | Kalkhof, González, Mukhopadhyay 2023, *Med-NCA* — IPMI 2023 **[peer-reviewed]** [arXiv 2302.03473](https://arxiv.org/abs/2302.03473) | The medical-imaging gateway and the architectural template MarrLab refines. |
| 8 | Deutges, Sadafi, Navab, Marr 2024, *WBC-NCA* — MICCAI 2024 **[peer-reviewed]** [arXiv 2404.05584](https://arxiv.org/abs/2404.05584) | First MarrLab paper; classification recipe. Read with the [WBC-NCA repo](https://github.com/marrlab/WBC-NCA). |
| 9 | Deutges et al. 2025, *NCA-WSS* — MICCAI 2025 **[peer-reviewed]** [arXiv 2508.12322](https://arxiv.org/abs/2508.12322) | Shows how to read structure out of NCA features without retraining; will inform any DLCA-segmentation attempt. |
| 10 | Lang et al. 2025, *TeNCA* — MICCAI 2025 **[peer-reviewed]** [arXiv 2506.18720](https://arxiv.org/abs/2506.18720) | The sparse-temporal extension; the closest MarrLab-orbit paper to a forward-model use of NCAs. |
| 11 | Milite, Caravagna, Sottoriva 2025, *Mixture of NCAs* — ICML GenBio workshop **[workshop]** [arXiv 2506.20486](https://arxiv.org/abs/2506.20486) | Stochastic NCAs for biology. Read for what DLCA *cannot* yet do. |
| 12 | Béna, Faldor, Goodman, Cully 2025, *Path to Universal NCA* — GECCO '25 Companion **[peer-reviewed]** [arXiv 2505.13058](https://arxiv.org/abs/2505.13058) | The continuous-universality counterpart to DLCA's discrete-universality. The "immutable substrate" idea is the cleanest way to think about hardware-vs-software channels. |
| 13 | Xu, Miikkulainen 2025, *Neural Cellular Automata for ARC-AGI* — IEEE SSCI / ALIFE 2025 **[peer-reviewed]** [arXiv 2506.15746](https://arxiv.org/abs/2506.15746) | Shows what NCAs can and can't do on intrinsically discrete tasks. Direct relevance to DLCA's reachable problem class. |
| 14 | Grattarola, Livi, Alippi 2021, *Learning Graph Cellular Automata* — NeurIPS 2021 **[peer-reviewed]** | The non-lattice extension. Read if you want to push DLCA beyond a 2D Moore grid. |
| 15 | Bührer et al. 2025, *Recurrent Deep Differentiable Logic Gate Networks* — **[preprint]** [arXiv 2508.06097](https://arxiv.org/abs/2508.06097) | Independent recurrent-DLGN line, not yet wired to DLCA. Read for what a hardware-deployable recurrent boolean network looks like in 2025. |

---

## Section 7 — Known unknowns

Queries run that returned little or nothing useful, and topics where the literature is genuinely silent. Each entry is a candidate research opening.

- **DLCA on FPGA / ASIC.** Searched: `DiffLogic CA FPGA hardware synthesis`, `differentiable logic cellular automata FPGA 2025 2026`, `DLCA gate count power LUT`. Result: Petersen 2024 has FPGA placement; Bührer 2025 has recurrent FPGA-relevant DLGNs; **no paper has actually synthesized a trained DLCA to FPGA or reported its gate count, LUT count, or power**. This is the single most exploitable gap.
- **DLCA on medical imaging.** Searched: `DiffLogic CA medical OR biology 2025 2026`, `differentiable logic cellular automata segmentation`. Nothing. The MarrLab medical-NCA line and DLCA are entirely disjoint citation graphs as of April 2026. A WBC-DLCA preprint would be unprecedented.
- **DLCA for system identification of PDEs.** Searched in the context of Schumacher 2024 and Niklasson 2021's PDE framing: there is no published attempt to use DLCA to recover a PDE rule — and a strong a-priori reason this would fail (continuous coefficients vs. discrete state). The narrower question — "can DLCA recover a discrete update rule observed in data, e.g., a learned Rule 110 or a learned Margolus-block rule?" — is open and tractable.
- **DLCA + stochastic dynamics.** No published probabilistic-gate variant. The DLCA paper's own asynchrony is the only randomness. A "Mixture of DLCAs" or noisy-gate DLCA would be a clean contribution.
- **DLCA + graph / non-lattice.** No public work. Combines two unsolved sub-problems (perception over irregular neighborhoods in pure logic, and graph-DLGN itself).
- **Quantitative train-vs-inference accuracy gap for DLCA.** The arXiv reports successful crystallization but not a sweep over network depth, gate count, or task complexity. **Anyone reproducing DLCA can publish this as a stand-alone benchmark.**
- **DLCA scaling laws.** No data on how DLCA accuracy scales with circuit width or depth, the way Petersen 2022/2024 plot for DLGN.
- **MarrLab in-progress code.** The marrlab GitHub org could not be browsed directly in this environment. The three NCA repos surfaced via search are the published ones; whether internal repos exist for hierarchical-NCA (Yang IPMI 2025) or for follow-on work is unclear. **Not found:** any MarrLab-affiliated DLCA experiment.
- **Symbolic minimization / Espresso-class analysis of trained DLCA circuits.** Searched: `DiffLogic CA symbolic minimization Espresso ABC`. Nothing. This is striking because the trained circuit is a literal boolean function; standard logic synthesis should apply directly.
- **Wolfram-class behavioral classification of learned DLCA rules.** Searched: `DiffLogic CA Wolfram class IV`. Nothing. The DLCA paper invokes universality as motivation but does not classify its learned rules behaviorally.
- **DLCA on 3D / volumetric data.** Searched: `DiffLogic CA 3D volumetric`. Nothing. Med-NCA already has a 3D variant (M3D-NCA); the DLCA version is wide open.
- **Conditional / goal-conditioned DLCA.** No published GoalDLCA — the analog of Sudhakaran et al. 2022.
- **DLCA + diffusion or generative use beyond fixed targets.** No published variational or diffusion-DLCA; Diff-NCA and FourierDiff-NCA exist for continuous NCAs but have no logic-gate counterpart.
- **Empirical comparison: trained NCA vs. trained DLCA on the *same* task.** No paper benchmarks both on the same lizard, the same WBC dataset, or the same PDE. Even on Game of Life — where both succeed — there is no head-to-head report on convergence time, parameter count, robustness profile, or interpretability.
- **Async phase diagram for DLCA.** Mentioned in passing in the paper, never plotted.

The pattern across these "not founds" is that **DLCA is exactly one paper deep in 2026**. Nearly every direction a follower would want to push — medical, biological, graph, 3D, hardware, stochastic, conditional, symbolic-extracted, scaled — is unoccupied or barely sketched. The combinatorial space of "an NCA paper that exists × `DLCA-ify it`" is mostly an empty grid, and the DLCA / WBC-NCA cell, the DLCA / FPGA cell, and the DLCA / Rule 110 cell stand out as having both the cleanest fit and the highest near-term payoff.