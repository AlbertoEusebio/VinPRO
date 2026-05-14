# WP2 — Computer Vision for Grapevine Structure Estimation

> Deep learning pipeline that turns a single RGB image into a directed graph of vine structure and outputs precision pruning coordinates.

[![ASP](https://img.shields.io/badge/Alta%20Scuola%20Politecnica-VinPRO-8B0000)](https://www.asp-poli.it/)
[![PIC4SeR](https://img.shields.io/badge/Lab-PIC4SeR%20PoliTo%2FPoliMi-003399)](https://pic4ser.polito.it/)
[![Baseline](https://img.shields.io/badge/Baseline-ViNet%20(Gentilhomme%20et%20al.%202023)-blue)](https://doi.org/10.1016/j.compag.2023.107736)
[![Dataset](https://img.shields.io/badge/Dataset-3D2cut%20Single%20Guyot-green)](https://doi.org/10.34777/azf6-tm83)

Part of the [VinPRO project](../README.md) · Developed at [PIC4SeR](https://pic4ser.polito.it/), Politecnico di Torino / Politecnico di Milano

**WP2 leads:** Vincenzo Avantaggiato (MSc Computer Engineering, PoliTo) · Alberto Eusebio (MSc Computer Science and Engineering, PoliMi)

## What this module does

Given one RGB image of a grapevine, this module:

1. Runs a **Stacked Hourglass Network** to detect node locations and branch directions
2. Builds a **resistivity graph** from the predictions
3. Recovers the **directed vine tree** via Dijkstra shortest paths
4. Applies a **pruning policy** to select cutting points
5. Publishes pixel coordinates to the [WP3 control stack](../wp3-control/)

The output is a directed graph where every node carries its image coordinates, node type, and branch category. That graph feeds a pruning policy that selects cut locations, which WP3 then lifts to 3D and executes with the robot arm.

## Pipeline

```
RGB Image (1024x1024 input)
        |
        v
Stacked Hourglass Network (2 stages, K=256)
        |
        |-- Heatmaps    (5 branch types x 4 node types = 20 channels, 256x256)
        +-- Vec. fields (5 branch types x 2 components = 10 channels, 256x256)
                |
                v
        Peak detection  ->  node coordinates
                |
                v
        Resistivity graph
        R(c->p) = (1 - A) x ||v_cp||
                |
                v
        Dijkstra tree estimation
                |
                v
        Pruning policy
        (partition nodes -> midpoint cuts, preserve one cane)
                |
                v
        Pixel cutting coordinates  ->  WP3
```

## Node and Branch Taxonomy

**Node types:**

| ID | Name | Description |
|----|------|-------------|
| 0 | `rootCrown` | Root attachment at trunk base |
| 1 | `branchNode` | Junction between branches |
| 2 | `growingTip` | Apical shoot tip |
| 3 | `pruningCut` | Annotated cut location |

**Branch types:**

| ID | Name | Description |
|----|------|-------------|
| 0 | `mainTrunk` | Primary vertical trunk |
| 1 | `courson` | Short spur supporting canes |
| 2 | `cane` | One-year-old fruiting cane (preserved after pruning) |
| 3 | `shoot` | Current-season growth |
| 4 | `lateralShoot` | Secondary shoot off a primary shoot |

## Installation

```bash
git clone --recurse-submodules https://github.com/<org>/VinPRO.git
cd VinPRO/wp2-computer-vision

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

**3D2cut Single Guyot** — published alongside the ViNet paper.

| Property | Value |
|----------|-------|
| DOI | [10.34777/azf6-tm83](https://doi.org/10.34777/azf6-tm83) |
| License | CC BY-NC-SA |
| Images | 1,513 single-plant grapevine photos |
| Vineyards | 3 French vineyards, blue/white artificial backgrounds |
| Annotations | Node locations, node types, branch types, parent-child links |
| Train split | 1,255 images (sets 00-05 train, set 06 val) |
| Test split | 258 images (set 07) |

Download:

```bash
bash scripts/download_dataset.sh /path/to/destination/
```

Expected layout:

```
<DATA_ROOT>/
|-- 01-TrainAndValidationSet/
|   |-- image001.jpg
|   |-- image001_annotation.json
|   +-- ...
+-- 02-IndependentTestSet/
    |-- image001.jpg
    |-- image001_annotation.json
    +-- ...
```

## Quick Start

**Train:**

```bash
python train.py --data_path /path/to/3D2cut/ \
                --max_epochs 300 \
                --batch_size 1 \
                --lr 1e-3 \
                --hourglass_channels 256 \
                --gpus 1
```

Checkpoints are saved to `lightning_logs/` (top-3 by `val_loss`). Final weights are exported to `vinet_final.pt`.

**Evaluate:**

```bash
python evaluate.py --data_path /path/to/3D2cut/ \
                   --checkpoint models/model.pt
```

Results are cached in `eval_cache/` for fast re-runs. Use `--no_cache` to force recomputation.

**Single-image inference:**

```bash
python predict.py --image path/to/vine.jpg \
                  --checkpoint models/model.pt \
                  --output result.png
```

## Architecture

| Component | Detail |
|-----------|--------|
| Feature extractor | 7x7 conv (stride 2) -> MaxPool -> Triple residual block -> 1x1 channel match |
| Hourglass module | 5-level encoder-decoder with skip connections and triple residual blocks; gradient checkpointing |
| Stacked HG | 2 modules in sequence; stage-1 predictions merged back into stage-2 input |
| Normalization | Instance Normalization (batch size = 1) |
| Output | 30 channels at H/4 x W/4: 20 heatmaps + 10 vector field maps |
| Reference config | `2HG-256` · ~13.2 M parameters · K = 256 channels |

**Training targets — two heatmap scales:**

| Stage | Sigma | Purpose |
|-------|-------|---------|
| Stage 1 | 40 px | Coarse guidance |
| Stage 2 | 15 px | Fine localization |

Loss: MSE applied at both stages simultaneously. Optimizer: Adam with StepLR (x0.9 every 5,000 steps).

## Pruning Policy

The pruning policy turns the vine graph into concrete cutting coordinates:

1. Find **partition nodes** — nodes where `in_degree > 1` (multiple branches split toward the tips)
2. Remove redundant cuts — if a partition node's parent is also a partition node, cutting at the parent already handles the subtree
3. Place each cut at the **midpoint** between the partition node and its parent
4. **Preserve one cane** — the one-year-old shoot that carries next season's fruit is not cut

The `POSSIBLE_PARENTS` compatibility matrix in `vinet/config.py` ensures only anatomically valid parent-child branch-type pairs are considered during graph construction.

## Resistivity Formula

```
R(c -> p) = (1 - A) x ||v_cp||
```

**A** is the mean cosine alignment of the predicted branch vector field along the child-to-parent segment. **||v_cp||** is the Euclidean distance between the two nodes. Lower resistivity indicates higher confidence of a branch connection.

## Coordinate System

The network inputs are resized to **1024 x 1024** (square). Heatmap outputs are at **256 x 256** (H/4 x W/4). Node coordinates are in 256 x 256 space. The [WP3 bridge node](../wp3-control/src/vinpro_perception/vinpro_perception/inference_node.py) scales them back to the sensor resolution before depth lookup:

```
u_sensor = u_256 x sensor_W / 256
v_sensor = v_256 x sensor_H / 256
```

## Results

Evaluated on the 3D2cut Single Guyot independent test set (258 images).

**AllNodeMetric** (distance threshold 5 px):

| Metric | ViNet baseline (Gentilhomme et al., 2023) |
|--------|------------------------------------------|
| Precision | 0.95 |
| Recall | 0.90 |
| F-Score | 0.92 |

**CoursonMetric** (distance threshold 5 px):

| Metric | ViNet baseline |
|--------|---------------|
| Precision | 0.76 |
| Recall | 0.74 |
| F-Score | 0.75 |

Full per-category results for our implementation: `notebooks/final_result.ipynb`

## Repository Structure

```
wp2-computer-vision/
|-- vinet/
|   |-- config.py            # NODE_TYPES, BRANCH_TYPES, POSSIBLE_PARENTS, hyperparameters
|   |-- data/
|   |   |-- dataset.py       # VineDataset: loads 3D2cut, generates M1/M2 heatmap pairs
|   |   |-- transforms.py    # Albumentations augmentation pipelines
|   |   +-- encoding.py      # Gaussian blob and limb vector field generation
|   |-- model/
|   |   |-- hourglass.py     # StackedHourglassNetwork, HourglassModule, residual blocks
|   |   +-- lightning_module.py  # PyTorch Lightning wrapper with MSE loss
|   +-- inference/
|       |-- node_extraction.py   # Peak detection on predicted heatmaps
|       |-- association.py       # Resistivity graph + Dijkstra tree estimation
|       +-- visualization.py     # Color-coded overlay utilities
|-- train.py
|-- evaluate.py
|-- predict.py
|-- models/                  # Trained checkpoints (.pt)
|-- notebooks/               # Experiment and result notebooks
|-- outputs/                 # Predicted graph adjacency lists
+-- scripts/
    +-- download_dataset.sh
```

## Integration with WP3

WP2 outputs pixel coordinates of cutting points. The WP3 bridge node (`vinpro_perception`) subscribes to the camera stream, runs this pipeline, and publishes coordinates on `/pixel_coordinates`. The downstream camera node lifts them to 3D using the RealSense D435i depth channel.

For branch-orientation-aware shear alignment (planned), the branch vector field at each cut pixel can be extracted and published alongside the coordinates — see `PointsArray.msg` in `wp3-control/src/custom_msgs/`.

## Citation

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

This pipeline re-implements and extends the **ViNet** method by Gentilhomme et al. (2023). All credit for the original architecture design and dataset goes to the original authors.

Developed as part of **[VinPRO](../README.md)** within [Alta Scuola Politecnica](https://www.asp-poli.it/), with support from [PIC4SeR](https://pic4ser.polito.it/), YANMAR R&D Europe, and Cantina 366 (Aglié, TO).
