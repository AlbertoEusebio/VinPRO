# VinPRO — WP2: Computer Vision for Grapevine Structure Estimation

[![ASP](https://img.shields.io/badge/Alta%20Scuola%20Politecnica-VinPRO-8B0000)](https://www.asp-poli.it/)
[![Baseline](https://img.shields.io/badge/Baseline-ViNet%20(Gentilhomme%20et%20al.%202023)-blue)](https://doi.org/10.1016/j.compag.2023.107736)
[![Dataset](https://img.shields.io/badge/Dataset-3D2cut%20Single%20Guyot-green)](https://doi.org/10.34777/azf6-tm83)

This repository contains the **Computer Vision pipeline (WP2)** of the **VinPRO** project — *Vineyard Pruning with Robots through Collaborative Learning in Virtual Reality* — developed within [Alta Scuola Politecnica](https://www.asp-poli.it/) (ASP), the joint honours programme of Politecnico di Torino and Politecnico di Milano.

## The VinPRO Project

VinPRO aims to design and implement a robotic system capable of autonomously pruning grapevines. The project integrates four interconnected Work Packages:

| WP | Focus | Description |
|---|---|---|
| **WP1** | Gripper Design | Custom 3D-printed end-effector with electric shears for the Kinova Gen3 Lite arm |
| **WP2** | Computer Vision | Deep learning pipeline for vine node detection, branch classification, and structure estimation **(this repo)** |
| **WP3** | Control & Planning | ROS 2 / MoveIt 2 motion planning, pixel-to-3D conversion, and cut execution |
| **WP4** | Virtual Reality | Unity-based vineyard simulation for training data generation and pruning strategy validation |

The output of WP2 — a directed graph representing the plant structure — feeds into a **pruning policy algorithm** that selects cutting points, which are then passed to WP3 for robotic execution.

**Partners:** [PIC4SeR](https://pic4ser.polito.it/) (PoliTo/PoliMi), YANMAR R&D Europe, Cantina 366 (Aglié, TO)

**Team:** Vincenzo Avantaggiato, Alberto Eusebio, Riccardo Ghianni, Francesco Risso, Faik Tahirović, Eleonora Troilo, Riccardo Vallino, Lorenzo Vignoli

**Academic Tutors:** Marcello Chiaberge (PoliTo), Luca Bascetta (PoliMi), Mauro Martini, Marco Ambrosio, Alessandro Navone, Brenno Tuberga, Luigi Mazzara

## WP2: Approach

Our computer vision pipeline is based on the method proposed by **ViNet** (Gentilhomme et al., 2023), which we used as a baseline and re-implemented with adaptations for our robotic pruning workflow.

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
                                                ↓
                                     [Resistivity Graph + Dijkstra]
                                                ↓
                                     Directed Graph (plant structure)
                                                ↓
                                     [Pruning Policy] → Cutting Points → WP3
```

For each input image, the system produces a directed graph whose nodes encode their location, type (root crown, branch node, growing tip, pruning cut), and branch category (trunk, courson, cane, shoot, lateral shoot). Each node has at most one parent. This graph is then passed to the pruning policy algorithm, which identifies partition nodes and selects midpoints between structural nodes as practical cutting locations.

## Repository Structure

```
vinpro-wp2/
├── README.md
├── requirements.txt
├── pyproject.toml
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation with AllNodeMetric
├── predict.py               # Single-image inference & visualization
├── vinet/
│   ├── __init__.py
│   ├── config.py            # Constants, node/branch types, hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # VineDataset (PyTorch Dataset)
│   │   ├── transforms.py    # Albumentations augmentation pipelines
│   │   └── encoding.py      # Heatmap & vector field generation
│   ├── model/
│   │   ├── __init__.py
│   │   ├── hourglass.py     # Stacked Hourglass Network
│   │   └── lightning_module.py  # PyTorch Lightning wrapper
│   └── inference/
│       ├── __init__.py
│       ├── node_extraction.py   # Node coordinate extraction from heatmaps
│       ├── association.py       # Resistivity graph & tree structure estimation
│       └── visualization.py     # Plotting utilities
└── scripts/
    └── download_dataset.sh  # Helper to fetch the 3D2cut dataset
```

## Installation

```bash
git clone https://github.com/<your-username>/vinpro-wp2.git
cd vinpro-wp2

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

We use the **3D2cut Single Guyot** dataset published alongside the ViNet paper: [https://doi.org/10.34777/azf6-tm83](https://doi.org/10.34777/azf6-tm83) (CC BY-NC-SA license).

The dataset contains 1,513 single-plant grapevine images captured across three French vineyards with blue or white artificial backgrounds. Each image is fully annotated with node locations, node types, branch types, and parent–child dependencies. The train/test split is 1,255 / 258 images.

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

### Evaluation

```bash
python evaluate.py --data_path /path/to/3D2cut_Single_Guyot/ \
                   --checkpoint path/to/best_checkpoint.ckpt
```

### Single-Image Inference

```bash
python predict.py --image path/to/vine_image.jpg \
                  --checkpoint path/to/model.pt \
                  --output output_prediction.png
```

## Architecture Details

| Component | Description |
|---|---|
| **Feature Extractor** | 7×7 conv (stride 2) → MaxPool → Triple residual block → 1×1 channel match |
| **Hourglass Module** | 5-level encoder-decoder with skip connections and triple residual blocks |
| **Stacked HG** | 2 hourglass modules; stage-1 predictions merged into stage-2 input |
| **Output** | 30 channels: 20 node heatmaps (5 branch × 4 node types) + 10 vector field maps (5 branch × 2 components) |
| **Reference model** | `2HG-256`: ~13.2M parameters, K=256 channels, Instance Normalization |

## Baseline Results (from Gentilhomme et al., 2023)

**AllNodeMetric** (τ_d = 5):

| Metric | Value |
|---|---|
| Precision | 0.95 |
| Recall | 0.90 |
| F-Score | 0.92 |

**CoursonMetric** (τ_d = 5):

| Metric | Value |
|---|---|
| Precision | 0.76 |
| Recall | 0.74 |
| F-Score | 0.75 |

## Integration with Other Work Packages

The directed graph produced by this pipeline is consumed downstream:

- **Pruning Policy** — An algorithm traverses the graph to identify *partition nodes* (nodes with multiple predecessors), excludes redundant cuts, and selects midpoints between structural nodes as cutting locations. The cane branch is preserved for next year's yield.
- **WP3 (Control)** — Cutting point pixel coordinates are converted to 3D space via an Intel RealSense D435i depth camera, transformed into the robot's base frame using TF2, and passed to MoveIt 2's Task Constructor for collision-free trajectory planning on the Kinova Gen3 Lite arm.

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

## Acknowledgments

This work was developed as part of the **VinPRO** project within [Alta Scuola Politecnica](https://www.asp-poli.it/), the joint honours programme of Politecnico di Torino and Politecnico di Milano.

We gratefully acknowledge the support of:
- **[PIC4SeR](https://pic4ser.polito.it/)** — Interdepartmental Centre for Service Robotics (PoliTo/PoliMi), for providing the robotic platform, lab facilities, and technical supervision
- **YANMAR R&D Europe** — for technological and commercial support
- **Cantina 366** (Aglié, TO) — for sharing pruning expertise and providing access to their vineyard for field testing
- **3D2cut SA** and the authors of ViNet — for releasing the annotated dataset and the method that served as our baseline

The computer vision pipeline in this repository is based on the ViNet method by Gentilhomme et al. (2023). All credit for the original architecture design and dataset goes to the original authors.
