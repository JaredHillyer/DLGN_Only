# Papers and Resources

This repository is best read as a research workspace around three connected
questions:

- How far can the active JAX Differentiable Logic Gate Network (DLGN) package be
  pushed toward image and cellular-automata style computation?
- What should be borrowed from Differentiable Logic Cellular Automata (DLCA) and
  the Google self-organising-systems line?
- Which Neural Cellular Automata (NCA) ideas from medical imaging, weak
  supervision, hierarchy, attention, and explainability are useful bridges for
  future DLCA work?

The short version: start with the scout report to orient yourself, use the NCA
audit for architecture-level details, then use the MarrLab/DLCA research reports
and papers to decide which code family answers which research question.

## Fast Orientation

| Resource | Role in the repo | Best use |
|---|---|---|
| [scout mission.md](<scout mission.md>) | Broad repo map across active code, reference repos, archives, papers, and artifacts. | First read. It explains what is active package code versus provenance or reference material. |
| [NCA_audit_report.md](NCA_audit_report.md) | Deep technical audit of the MarrLab-style NCA repositories. | Use when porting NCA ideas into DLCA or comparing update rules, heads, pooling, masks, and training loops. |
| [deep-research-report.md](deep-research-report.md) | Curated research synthesis on MarrLab, broader NCA work, DLCA, and system-identification gaps. | Use for literature framing and research direction. |
| [compass_artifact_wf-305d4cc4-0f43-40ef-8a57-961a9905d1bf_text_markdown.md](compass_artifact_wf-305d4cc4-0f43-40ef-8a57-961a9905d1bf_text_markdown.md) | Longer generated survey artifact with citations and broader landscape notes. | Use as a source-rich companion to the cleaned research report. |
| [Neural Cellular Automata at MarrLab and the DLCA Bridge.docx](<Neural Cellular Automata at MarrLab and the DLCA Bridge.docx>) | Document version of the MarrLab/DLCA bridge report. | Use when a shareable document artifact is more convenient than Markdown. |
| [Deep_R.pdf](Deep_R.pdf) | PDF companion artifact for generated research material. Text extraction did not recover useful content locally. | Treat as a generated/image-style companion, not the primary source of truth. |
| [README.md](README.md) | Early top-level project note about separating DLGN from DLCA and building toward convolutional DLGN. | Use as intent/provenance for the repo cleanup direction. |

## Local Papers and Documents

| Local file | Paper or document | Connects to code | Research value |
|---|---|---|---|
| [WBA.pdf](WBA.pdf) | *Neural Cellular Automata for Lightweight, Robust and Explainable Classification of White Blood Cell Images* | [WBC-NCA-main/](WBC-NCA-main/) | Baseline medical NCA classifier: image-as-cell-grid, recurrent local update, hidden-state features, max-pooling head, and explainability hooks. |
| [Attention Pooling Enhances NCA-based Classification of Microscopy Images.pdf](<Attention Pooling Enhances NCA-based Classification of Microscopy Images.pdf>) | *Attention Pooling Enhances NCA-based Classification of Microscopy Images* | [aNCA-main/](aNCA-main/) | Shows how attention/pooling improves NCA microscopy classification without abandoning the NCA backbone. |
| [WeaklySupervisedSegmentationofWhiteBloodCells.pdf](WeaklySupervisedSegmentationofWhiteBloodCells.pdf) | *Neural Cellular Automata for Weakly Supervised Segmentation of White Blood Cells* | [NCA-WSS-main/](NCA-WSS-main/) and [WBC-NCA-main/](WBC-NCA-main/) | Motivates using classifier hidden states as segmentation evidence. Important for latent-state interpretability and weak supervision. |
| [ExplainableAIidentifiesdiagnosticcells ofgeneticAMLsubtypes.pdf](<ExplainableAIidentifiesdiagnosticcells ofgeneticAMLsubtypes.pdf>) | *Explainable AI identifies diagnostic cells of genetic AML subtypes* | No direct NCA implementation; contextual link to [WBC-NCA-main/explainability.ipynb](WBC-NCA-main/explainability.ipynb) | Biomedical explainability context for why transparent cell-level decisions matter in hematology. |
| [deep-research-report.md](deep-research-report.md) | *Neural Cellular Automata at MarrLab and the DLCA Bridge* | All NCA/DLCA areas listed below | Best single narrative for the literature-to-code bridge and current research openings. |
| [Deep_R.pdf](Deep_R.pdf) | Generated research PDF companion | Same conceptual scope as the generated reports | Useful as a preserved artifact, but prefer Markdown/docx for searchable detail. |

## Literature-to-Code Bridge

### Active DLGN and DLCA Direction

| Code area | What it is | Research connection |
|---|---|---|
| [dlgn/](dlgn/) | Active JAX DLGN package with models, training, data, analysis, utilities, and CLIs. | The main implementation boundary for differentiable logic gate experiments. |
| [dlgn/models/gates.py](dlgn/models/gates.py) | Canonical full-family and light-family logic gate definitions. | Connects directly to DLGN and DLCA papers because gate semantics, truth-table order, and hard/soft decoding matter. |
| [dlgn/models/conv.py](dlgn/models/conv.py) | Convolutional/perception-style DLGN layers over image patches. | The strongest current code bridge toward DLCA-style local spatial computation. |
| [dlgn/models/network.py](dlgn/models/network.py) | Flat DLGN forward pass and GroupSum classifier head. | The core package path for non-cellular DLGN classification. |
| [scripts/](scripts/) | Thin CLI wrappers plus experiment scripts. | Operational layer for running or comparing DLGN experiments. |
| [configs/presets/](configs/presets/) | Example JSON presets for flat DLGN runs. | Reproducible starting points for XOR, MNIST-light, and Adult experiments. |
| [tests/](tests/) | Regression coverage for package contracts. | Helps separate stable package behavior from notebooks and reference repos. |
| [difflogic_ca.py](difflogic_ca.py) and [diffLogic_CA.ipynb](diffLogic_CA.ipynb) | Google DLCA notebook/export reference. | DLCA provenance: local recurrent update, differentiable logic gates, soft training, hard discrete inference. |
| [Conv_to_Alter_PREMADE_KERNElS_NEEDED/](Conv_to_Alter_PREMADE_KERNElS_NEEDED/) | Transitional convolutional DLGN staging area. | Shows the path from loose experiments toward [dlgn/models/conv.py](dlgn/models/conv.py). |
| [conv_model_latest.py](conv_model_latest.py) | Loose JAX/Flax image-classification experiment. | Useful as live local comparison code for image training mechanics, but not package-integrated. |

### MarrLab and Medical NCA References

| Code area | What it is | Research connection |
|---|---|---|
| [WBC-NCA-main/](WBC-NCA-main/) | Original white-blood-cell NCA classifier reference. | Best minimal continuous-NCA skeleton for classification and hidden-state inspection. |
| [WBC-NCA-main/src/models/NCA.py](WBC-NCA-main/src/models/NCA.py) | NCA model variants including `MaxNCA`, `SimpleNCA`, `SegNCA`, and `ConvNCA`. | The most direct code reference for recurrent state updates, input-channel persistence, fire masks, and pooling heads. |
| [WBC-NCA-main/explainability.ipynb](WBC-NCA-main/explainability.ipynb) | Explainability notebook. | Use with the WBC paper and AML explainability paper when thinking about interpretable cell-level predictions. |
| [aNCA-main/](aNCA-main/) | Attention-pooling NCA classifier reference. | Shows how NCA backbones can be paired with richer pooling/attention heads across microscopy datasets. |
| [aNCA-main/src/NCA.py](aNCA-main/src/NCA.py) | aNCA backbone and head variants. | Useful for comparing pooling, attention, hierarchy-like heads, and hidden-state readout choices. |
| [hNCA-main/](hNCA-main/) | Hierarchical NCA classifier reference with fold-1 results/checkpoints. | Shows the multiresolution move: one NCA stage feeds another at a coarser scale. |
| [hNCA-main/results/](hNCA-main/results/) | Logs, losses, figures, and trained checkpoints. | Evidence for concrete hNCA training settings and outcomes. |
| [NCA-WSS-main/](NCA-WSS-main/) | Weakly supervised segmentation reference. | Important for the idea that classification-trained NCA hidden maps can become segmentation signals. |
| [NCA-WSS-main/evaluate_model.py](NCA-WSS-main/evaluate_model.py) | Evaluation and PCA/Otsu segmentation pipeline. | Best local code pointer for feature-map-to-mask extraction, even though the repo is incomplete as shipped. |

### Growing NCA, General CA, and Provenance

| Code area | What it is | Research connection |
|---|---|---|
| [Growing-Neural-Cellular-Automata-master/](Growing-Neural-Cellular-Automata-master/) | PyTorch reproduction of Growing NCA with demo/training material. | General NCA intuition: growth, regeneration, local rules, and visualization. |
| [Growing-Neural-Cellular-Automata-Pytorch-master/](Growing-Neural-Cellular-Automata-Pytorch-master/) | Broader CA/NCA experiments: growth, tasks, particles, image-to-image, and videos. | Useful for seeing how CA-style local update ideas appear outside medical classification. |
| [Growing-Neural-Cellular-Automata-Pytorch-master/CA_tasks/](Growing-Neural-Cellular-Automata-Pytorch-master/CA_tasks/) | CA computational task experiments. | Relevant to DLCA as local computation rather than image modeling alone. |
| [Growing-Neural-Cellular-Automata-Pytorch-master/CA_Particles_V3/](Growing-Neural-Cellular-Automata-Pytorch-master/CA_Particles_V3/) | Particle/physics-style CA work. | Background for learned local dynamics and possible system-identification analogs. |
| [archive/notebooks/](archive/notebooks/) | Older notebooks for DLGN/DLCA attempts and wavelet/provenance work. | Historical design context. Use after reading active package code. |
| [archive/data_set_code_old/](archive/data_set_code_old/) | Older dataset loading code. | Provenance for current dataset handling in [dlgn/data/](dlgn/data/). |
| [logs/](logs/) and [outputs/](outputs/) | Local run evidence and generated outputs. | Useful for reconstructing experiments, but not primary source code or literature. |

## Recommended Research Workflow

1. Read [scout mission.md](<scout mission.md>) to understand the whole repo shape.
2. Read [NCA_audit_report.md](NCA_audit_report.md) if you need exact NCA architecture details.
3. Read [deep-research-report.md](deep-research-report.md) for the broader MarrLab, NCA, DLGN, and DLCA landscape.
4. Inspect the active DLGN package: [dlgn/models/gates.py](dlgn/models/gates.py), [dlgn/models/network.py](dlgn/models/network.py), and [dlgn/models/conv.py](dlgn/models/conv.py).
5. Compare against the DLCA provenance in [difflogic_ca.py](difflogic_ca.py) and [diffLogic_CA.ipynb](diffLogic_CA.ipynb).
6. Use [WBC-NCA-main/](WBC-NCA-main/) as the first medical-NCA code reference, then branch to [aNCA-main/](aNCA-main/), [hNCA-main/](hNCA-main/), or [NCA-WSS-main/](NCA-WSS-main/) depending on whether the question is attention, hierarchy, or weak segmentation.
7. Use the Growing NCA directories and [archive/](archive/) only after the active and reference code paths are clear.

## Research Gaps Exposed by This Repo

- DLCA has not yet been demonstrated locally or in the surveyed literature on
  medical microscopy, hematology, MRI, or other biomedical data. The MarrLab
  NCA references motivate the application surface, but they do not prove the
  DLCA transfer.
- Continuous-valued medical images need an input/state encoding strategy before
  discrete logic-cellular inference can be evaluated fairly.
- NCA-WSS motivates weakly supervised segmentation from hidden states, but a
  discrete-state DLCA analog would need a replacement for continuous PCA-style
  feature-map analysis.
- Schumacher-style system identification is the cleanest conceptual target for
  local-rule learning, but current local code is closer to classification and
  pattern-generation scaffolding than trajectory-loss benchmarks.
- Hardware/export is a major opportunity: DLGN is hardware-aligned, but this repo
  does not yet contain a trained DLCA-to-FPGA/ASIC export path or equivalence
  checks.
- Hierarchical DLCA is an open bridge suggested by hNCA and the DLCA limitations:
  the active code has convolutional/perception DLGN pieces, but not a packaged
  recurrent multiscale logic-CA model.

## Caveats

- The imported NCA repositories are reference material, not a single integrated
  package. Their dependencies, data layouts, and checkpoints differ.
- [NCA-WSS-main/](NCA-WSS-main/) is incomplete as shipped because its `src/`
  package is absent; its train/eval scripts still document the segmentation
  idea and feature-extraction pipeline.
- [difflogic_ca.py](difflogic_ca.py) is an exported notebook, not a polished
  package module.
- [conv_model_latest.py](conv_model_latest.py) is live loose work and should not
  be treated as an established package entrypoint.
- Generated research artifacts are helpful for navigation, but paper claims
  should be checked against the local PDFs or the linked primary sources before
  citation.
