# VinPRO — WP2: Computer Vision for Grapevine Structure Estimation

[![ASP](https://img.shields.io/badge/Alta%20Scuola%20Politecnica-VinPRO-8B0000)](https://www.asp-poli.it/)
[![PIC4SeR](https://img.shields.io/badge/Lab-PIC4SeR%20PoliTo%2FPoliMi-003399)](https://pic4ser.polito.it/)
[![Baseline](https://img.shields.io/badge/Baseline-ViNet%20(Gentilhomme%20et%20al.%202023)-blue)](https://doi.org/10.1016/j.compag.2023.107736)
[![Dataset](https://img.shields.io/badge/Dataset-3D2cut%20Single%20Guyot-green)](https://doi.org/10.34777/azf6-tm83)

> Part of the [VinPRO project](../README.md) — *Vineyard Pruning with Robots through Collaborative Learning in Virtual Reality*

This repository contains the **Computer Vision pipeline (WP2)** of the **VinPRO** project, developed within [Alta Scuola Politecnica](https://www.asp-poli.it/) (ASP), the joint honours programme of [Politecnico di Torino](https://www.polito.it/) and [Politecnico di Milano](https://www.polimi.it/).

Given a single RGB image of a grapevine, this module predicts a directed graph encoding every node's location, type, and branch category. That graph is then consumed by a pruning policy algorithm and passed on to WP3 for robotic execution on a **Kinova Gen3 Lite** arm at **[PIC4SeR](https://pic4ser.polito.it/)** (Interdepartmental Centre for Service Robotics, PoliTo / PoliMi).

---

## The VinPRO Project

VinPRO aims to design and implement a robotic system capable of autonomously pruning grapevines. The project integrates four interconnected Work Packages:

| WP | Focus | Description |
|---|---|---|
| **WP1** | Gripper Design | Custom 3D-printed PA6 end-effector with electric shears and Arduino Nano control for the Kinova Gen3 Lite arm |
| **WP2** | Computer Vision | Deep learning pipeline for vine node detection, branch classification, and structure estimation **(this repo)** |
| **WP3** | Control & Planning | ROS 2 Humble / MoveIt 2 motion planning, pixel-to-3D conversion via RealSense D435i + TF2, and cut execution |
| **WP4** | Virtual Reality | Unity (URP) + ROS2 vineyard simulation for testing and pruning strategy validation |

The output of WP2 — a directed graph representing the plant structure — feeds into a **pruning policy algorithm** that selects cutting points, which are then passed to WP3 for robotic execution.

**Partners:** [PIC4SeR](https://pic4ser.polito.it/) (PoliTo/PoliMi), YANMAR R&D Europe, Cantina 366 (Aglié, TO)

**WP2 leads:** Vincenzo Avantaggiato (MSc Computer Engineering, PoliTo), Alberto Eusebio (MSc Computer Science and Engineering, PoliMi)

**Full team:** Vincenzo Avantaggiato, Alberto Eusebio, Riccardo Ghianni, Francesco Risso, Faik Tahirović, Eleonora Troilo, Riccardo Vallino, Lorenzo Vignoli

**Academic Tutors:** Marcello Chiaberge (PoliTo), Luca Bascetta (PoliMi), Mauro Martini, Marco Ambrosio, Alessandro Navone, Brenno Tuberga, Luigi Mazzara (DET, PoliTo)

**External Tutor:** Marta Niccolini (YANMAR R&D Europe Srl)

---

## Approach

Our computer vision pipeline is a re-implementation of **ViNet** (Gentilhomme et al., 2023), adapted for our robotic pruning workflow. The pipeline was designed to balance detection accuracy with computational efficiency so that it could run under field conditions on limited hardware.

> **Baseline paper:**
> *Towards smart pruning: ViNet, a deep-learning approach for grapevine structure estimation*
> Theophile Gentilhomme, Michael Villamizar, Jerome Corre, Jean-Marc Odobez
> Computers and Electronics in Agriculture, 207, 107736 (2023)
> [https://doi.org/10.1016/j.compag.2023.107736](https://doi.org/10.1016/j.compag.2023.107736)

The pipeline consists of two main steps:

1. **Detection** — A Stacked Hourglass Network (SHG) predicts node confidence heatmaps and branch vector fields (Part Affinity Fields) from a single RGB image.
2. **Association** — Detected nodes are connected into a tree structure via shortest-path optimization on a resistivity graph derived from the predicted vector fields.

```
RGB Image → [Stacked Hourglass Network] → Heatmaps + Vector Fields
                                                ↓
                                         [Node Extraction]
                                          (peak detection)
                                                ↓
                                     [Resistivity Graph + Dijkstra]
                                      R = (1 − A) × ‖v_cp‖
                                                ↓
                                     Directed Graph (plant structure)
                                                ↓
                                     [Pruning Policy] → Cutting Points → WP3
```

For each input image, the system produces a directed graph whose nodes encode their location, type (root crown, branch node, growing tip, pruning cut), and branch category (trunk, courson, cane, shoot, lateral shoot). Each node has **at most one parent**. This graph is then passed to the pruning policy algorithm, which identifies *partition nodes* (nodes with multiple predecessors), excludes redundant cuts, and selects midpoints between structural nodes as practical cutting locations. The cane branch is preserved to ensure next year's yield.

---

## Repository Structure

```
Stacked-Hourglass-Network/
├── README.md
├── requirements.txt
├── pyproject.toml
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation with AllNodeMetric (Hungarian matching)
├── predict.py               # Single-image inference & visualization
├── vinet/
│   ├── __init__.py
│   ├── config.py            # Constants: NODE_TYPES, BRANCH_TYPES, POSSIBLE_PARENTS,
│   │                        #   NUM_OUTPUT_CHANNELS=30, default hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # VineDataset — loads 3D2cut, generates M1 (σ=40) and
│   │   │                    #   M2 (σ=15) heatmap/vector-field pairs
│   │   ├── transforms.py    # Albumentations augmentation pipelines
│   │   └── encoding.py      # Gaussian blob and limb vector field generation
│   ├── model/
│   │   ├── __init__.py
│   │   ├── hourglass.py     # ResidualUnit, TripleResidualBlock, FeatureExtractor,
│   │   │                    #   HourglassModule (5-level), StackedHourglassNetwork
│   │   └── lightning_module.py  # HourglassLightningModule — MSE loss at both stages,
│   │                        #   Adam optimizer, StepLR (×0.9 every 5000 steps)
│   └── inference/
│       ├── __init__.py
│       ├── node_extraction.py   # Peak detection on predicted heatmaps
│       ├── association.py       # Resistivity graph construction + Dijkstra tree estimation
│       └── visualization.py     # Color-coded node/edge overlay utilities
├── models/
│   ├── model.pt             # Primary trained checkpoint
│   ├── model_1.pt           # Additional checkpoint
│   └── model_2.pt           # Additional checkpoint
├── notebooks/
│   ├── vinnet-paper-replica.ipynb   # Main development notebook
│   ├── vinnet2.ipynb                # Extended experiments
│   ├── final_result.ipynb           # Final evaluation results and visualizations
│   ├── output.ipynb
│   ├── output1.ipynb
│   ├── output2.ipynb
│   └── output3.ipynb
├── outputs/
│   ├── graph.adjlist                # Predicted plant graph (NetworkX adjacency list)
│   ├── tree_structure.adjlist       # Estimated tree structure
│   └── tree_structure_{0-9}.adjlist # Per-image outputs for 10 test samples
└── scripts/
    └── download_dataset.sh  # Helper to fetch the 3D2cut dataset
```

---

## Installation

```bash
git clone --recurse-submodules https://github.com/<your-username>/VinPRO.git
cd VinPRO/Stacked-Hourglass-Network

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

We use the **3D2cut Single Guyot** dataset published alongside the ViNet paper:
[https://doi.org/10.34777/azf6-tm83](https://doi.org/10.34777/azf6-tm83) (CC BY-NC-SA license)

The dataset contains **1,513 single-plant grapevine images** captured across three French vineyards with blue or white artificial backgrounds. Each image is fully annotated with node locations, node types, branch types, and parent–child dependencies.

| Split | Images | Sets |
|-------|--------|------|
| Train + Validation | 1,255 | 00–05 (train), 06 (val) |
| Independent Test | 258 | 07 |

Expected layout:
```
<DATA_ROOT>/
├── 01-TrainAndValidationSet/
│   ├── image001.jpg
│   ├── image001_annotation.json
│   └── ...
└── 02-IndependentTestSet/
    ├── image001.jpg
    ├── image001_annotation.json
    └── ...
```

Download helper:
```bash
bash scripts/download_dataset.sh /path/to/destination/
```

---

## Usage

### Training

```bash
python train.py --data_path /path/to/3D2cut_Single_Guyot/ \
                --max_epochs 300 \
                --batch_size 1 \
                --lr 1e-3 \
                --hourglass_channels 256 \
                --gpus 1
```

Checkpoints are saved under `lightning_logs/` (top-3 by `val_loss`). The final weights are exported to `vinet_final.pt`.

### Evaluation

```bash
python evaluate.py --data_path /path/to/3D2cut_Single_Guyot/ \
                   --checkpoint path/to/best_checkpoint.ckpt
```

Results are cached in `--cache_dir` (default `eval_cache/`) to speed up re-runs. Use `--no_cache` to force re-computation.

### Single-Image Inference

```bash
python predict.py --image path/to/vine_image.jpg \
                  --checkpoint path/to/model.pt \
                  --output output_prediction.png
```

The script resizes the input to 1024×1024, runs a forward pass, extracts nodes, builds the resistivity graph, estimates the tree, and saves the color-coded overlay.

---

## Architecture Details

### Stacked Hourglass Network

| Component | Details |
|-----------|---------|
| **Feature Extractor** | 7×7 conv (stride 2) → MaxPool → Triple residual block → 1×1 channel match |
| **Hourglass Module** | 5-level encoder-decoder with skip connections and triple residual blocks; gradient checkpointing enabled |
| **Stacked HG** | 2 hourglass modules in sequence; stage-1 predictions are merged back into the stage-2 input feature map |
| **Normalization** | Instance Normalization throughout (BatchNorm is unsuitable at batch size 1) |
| **Output** | 30 channels per stage: 20 node heatmaps (5 branch types × 4 node types) + 10 vector field maps (5 branch types × 2 directional components) |
| **Output resolution** | H/4 × W/4 (quarter of input) |
| **Reference config** | `2HG-256` — ~13.2M parameters, K=256 channels |

### Node and Branch Taxonomy

**Node types** (4):

| ID | Type | Description |
|----|------|-------------|
| 0 | `rootCrown` | Root attachment point at the trunk base |
| 1 | `branchNode` | Junction between two or more branches |
| 2 | `growingTip` | Apical tip of a shoot |
| 3 | `pruningCut` | Point identified by the pruning policy as a cut location |

**Branch types** (5):

| ID | Type | Description |
|----|------|-------------|
| 0 | `mainTrunk` | Primary vertical trunk |
| 1 | `courson` | Short spur supporting fruiting canes |
| 2 | `cane` | One-year-old fruiting cane (preserved after pruning) |
| 3 | `shoot` | Current-season growth |
| 4 | `lateralShoot` | Secondary shoot off a primary shoot |

### Training Objective

The model is trained with **MSE loss applied at both hourglass stages** simultaneously. Two ground-truth heatmap scales are used:

- **M1** (σ = 40 px) — coarse, used to guide stage-1
- **M2** (σ = 15 px) — fine, used to guide stage-2

The `POSSIBLE_PARENTS` compatibility matrix enforces that only anatomically valid parent–child branch type pairs are considered during graph construction.

### Resistivity Graph

The association step builds a directed graph where the edge weight between a candidate child node *c* and parent node *p* is:

```
R(c → p) = (1 − A) × ‖v_cp‖
```

where:
- **A** is the mean cosine alignment of the predicted vector field along the line segment from *c* to *p*
- **‖v_cp‖** is the Euclidean distance between the two nodes

Lower resistivity indicates higher confidence that a direct branch connection exists. Dijkstra's shortest path from each detected node to the root crown then recovers the full directed tree.

---

## Results

### Baseline (Gentilhomme et al., 2023) — 3D2cut Single Guyot test set

**AllNodeMetric** (τ_d = 5):

| Metric | Value |
|--------|-------|
| Precision | 0.95 |
| Recall | 0.90 |
| F-Score | 0.92 |

**CoursonMetric** (τ_d = 5):

| Metric | Value |
|--------|-------|
| Precision | 0.76 |
| Recall | 0.74 |
| F-Score | 0.75 |

Per-category and overall metrics for our implementation are available in `notebooks/final_result.ipynb`. The `evaluate.py` script uses the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) for optimal node-to-ground-truth matching.

---

## Pruning Policy

The pruning-policy module consumes the directed graph and returns a set of cutting-point pixel coordinates. Each node carries its pixel location, predecessor list, successor list, and type.

**Algorithm:**

1. Find all **partition nodes** — nodes with more than one predecessor. These mark branch divisions where a cut decision must be made.
2. Remove partition nodes whose successors already contain another partition node. Cutting closer to the trunk makes the more distal cut on the same branch redundant.
3. For each surviving partition node, place the cut at the **midpoint between the node and its predecessor** (not directly at the junction), so the cut lands on a practical branch segment.
4. Preserve **one cane** (one-year-old shoot bearing next season's fruit) regardless of the policy output.

The resulting pixel coordinates are passed to WP3.

---

## Integration with Other Work Packages

The directed graph and cutting points produced by this pipeline are consumed downstream:

- **WP3 (Control)** — Cutting point pixel coordinates are lifted to 3D space via an **Intel RealSense D435i** depth channel and a TF2 transform into the robot base frame. A **MoveIt Task Constructor** pipeline (ROS 2 Humble / MoveIt 2) builds modular motion stages (approach → orient → cut → retract), runs inverse kinematics, and previews the plan in RViz before execution. Final actuation is triggered via `ros2_control` → **Arduino Nano** → electric shear.

Integration testing was conducted at **[PIC4SeR](https://pic4ser.polito.it/)** (Interdepartmental Centre for Service Robotics, Politecnico di Torino / Politecnico di Milano) using a physical grapevine mock-up. A field visit to **Cantina 366** (Aglié, TO) validated the pruning rules encoded in the policy under real vineyard conditions.

> **Integration status:** The final report states that project expectations were "mostly satisfied," but explicitly notes that *"the connection between them is not completed"* — the subsystems work substantially at the component level but are not yet fully integrated end-to-end.

---

## Citation

If you use this code or build upon our work, please cite the baseline paper:

```bibtex
@article{gentilhomme2023towards,
  title={Towards smart pruning: {ViNet}, a deep-learning approach for grapevine structure estimation},
  author={Gentilhomme, Theophile and Villamizar, Michael and Corre, Jerome and Odobez, Jean-Marc},
  journal={Computers and Electronics in Agriculture},
  volume={207},
  pages={107736},
  year={2023},
  publisher={Elsevier}
}
```

---

## Acknowledgments

This work was developed as part of the **VinPRO** project within [Alta Scuola Politecnica](https://www.asp-poli.it/), the joint honours programme of Politecnico di Torino and Politecnico di Milano.

We gratefully acknowledge:

- **[PIC4SeR](https://pic4ser.polito.it/)** — Interdepartmental Centre for Service Robotics (PoliTo/PoliMi) — for providing the Kinova Gen3 Lite robotic arm, mobile rover, 3D-printing facilities, and lab space for integration testing and validation.
- **YANMAR R&D Europe Srl** — for technological and commercial support under the YANMAR Smart Agriculture programme.
- **Cantina 366** (Aglié, TO) — for sharing expert pruning knowledge and providing field access. The owner's hands-on guidance ("*a bottle costs around €4/L and must cover all expenses*") directly shaped the pruning policy design.
- **3D2cut SA** and the authors of ViNet — for releasing the annotated dataset and the method that served as our baseline.

The computer vision pipeline in this repository is based on the ViNet method by Gentilhomme et al. (2023). All credit for the original architecture design and dataset goes to the original authors.
