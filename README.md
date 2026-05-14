# VinPRO — Robotic Vineyard Pruning with Deep Learning and Virtual Reality

> An integrated platform that detects vine structure from a single RGB image and autonomously executes precision pruning cuts with a robotic arm.

[![Alta Scuola Politecnica](https://img.shields.io/badge/Alta%20Scuola%20Politecnica-Multidisciplinary%20Project-8B0000)](https://www.asp-poli.it/)
[![PIC4SeR](https://img.shields.io/badge/Lab-PIC4SeR%20PoliTo%2FPoliMi-003399)](https://pic4ser.polito.it/)
[![YANMAR](https://img.shields.io/badge/Industry-YANMAR%20R%26D%20Europe-FF6600)](https://www.yanmar.com/eu/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is VinPRO?

VinPRO is a 1.5-year multidisciplinary project developed within **[Alta Scuola Politecnica](https://www.asp-poli.it/)** — the joint honours programme of [Politecnico di Torino](https://www.polito.it/) and [Politecnico di Milano](https://www.polimi.it/).

The system takes a live RGB-D image of a grapevine, builds a graph of its branch structure using a deep neural network, computes where to cut, and directs a **Kinova Gen3 Lite** robotic arm to execute each cut with a custom electric shear — all autonomously.

**Who is it for?** Vineyard owners and agricultural robotics researchers interested in selective, knowledge-preserving automated pruning. **Why does it matter?** Vineyard pruning accounts for up to 25% of annual labor costs in fruit production and relies on a shrinking pool of skilled workers. VinPRO encodes expert pruning rules directly into the robot's decision policy, preserving quality while reducing manual effort.

## Demo

<video controls src="https://github.com/AlbertoEusebio/VinPRO/blob/main/assets/videos/presentation.mp4" width="100%" title="Presentation"></video>

*Integration testing at [PIC4SeR](https://pic4ser.polito.it/) (Turin), demonstrating the full pipeline from image capture to robotic cut on a physical grapevine mock-up.*

## Key Features

| Feature | Status |
|---------|--------|
| Stacked Hourglass Network for node detection and branch classification | Implemented |
| Resistivity-graph association + Dijkstra tree estimation | Implemented |
| Pruning policy: partition-node selection, cane preservation | Implemented |
| ROS 2 / MoveIt Task Constructor motion planning | Implemented |
| Intel RealSense D435i depth-based 2D→3D coordinate lifting | Implemented |
| Custom electric shear with Arduino Nano ros2_control interface | Implemented |
| Procedural grapevine generator (Blender Python) | Implemented |
| Unity virtual vineyard with interactive VR pruning | Implemented |
| Branch-orientation-aware shear alignment | Planned |
| End-to-end field deployment | In progress |

## System Architecture

```
Camera (RealSense D435i)
        │  RGB stream
        ▼
┌─────────────────────────────────────────────┐
│  wp2-computer-vision  (Python / PyTorch)     │
│  Stacked Hourglass Network (2×, 256 ch.)    │
│  → 20 node heatmaps + 10 branch vector maps │
│  → Peak detection → node coordinates        │
│  → Resistivity graph + Dijkstra tree        │
│  → Pruning policy → cutting pixels          │
└─────────────────────┬───────────────────────┘
                      │  /pixel_coordinates
                      ▼
┌─────────────────────────────────────────────┐
│  wp3-control  (C++ / ROS 2 Humble)          │
│  Depth lookup → camera-frame 3D point       │
│  TF2 transform → base_link frame            │
│  MoveIt Task Constructor pipeline           │
│    Connect → Approach → Move → Cut →        │
│    Open → Retreat → Home                    │
│  ros2_control → Arduino Nano → shear        │
└─────────────────────┬───────────────────────┘
                      ▼
              Kinova Gen3 Lite arm
              Custom electric shear
```

`wp4-virtual-reality` runs alongside: a procedurally generated Unity vineyard with full ROS 2 integration provides a risk-free testing and operator training environment.

## Repository Structure

```
VinPRO/
├── wp2-computer-vision/          # Deep learning vision pipeline (WP2)
│   ├── vinet/                    # Core package: model, data, inference
│   ├── train.py / evaluate.py / predict.py
│   ├── models/                   # Trained checkpoints
│   └── notebooks/                # Experiment and result notebooks
│
├── wp3-control/                  # ROS 2 control stack (WP3)
│   └── src/
│       ├── custom_msgs/          # Shared message definitions
│       ├── vinpro_perception/    # WP2 bridge node (Python)
│       ├── vinpro_camera/        # Depth → 3D transform (C++)
│       ├── vinpro_transform/     # TF2 + MTC launcher (C++)
│       ├── vinpro_mtc/           # MoveIt Task Constructor pipeline (C++)
│       ├── vinpro_arduino/       # Hardware interface for the shear (C++)
│       ├── vinpro_description/   # URDF + MoveIt config
│       └── vinpro_bringup/       # Launch files
│
├── wp4-virtual-reality/          # Blender generator + Unity simulation (WP4)
│   ├── main.py                   # Procedural vine generator (Blender Python)
│   └── VinPRO cutting demonstrator/  # Unity URP project
│
└── assets/
    ├── VinPRO_final_report.pdf
    ├── VinPRO_poster.pdf
    └── videos/presentation.mp4
```

> **WP1 (Gripper Design)** — the custom 3D-printed end-effector and electric shear hardware are described in the [final report](assets/VinPRO_final_report.pdf). The URDF model is in `wp3-control/src/vinpro_description/urdf/vinpro_eef.urdf.xacro`.

## Hardware

| Component | Specification |
|-----------|--------------|
| Robotic arm | Kinova Gen3 Lite — 6 DOF, mm-scale repeatability |
| Mobile base | Autonomous rover (PIC4SeR, PoliTo) |
| RGB-D camera | Intel RealSense D435i |
| End-effector | 3D-printed polymer housing + electric pruning shear |
| Microcontroller | Arduino Nano — unlock / cut / lock cycle |
| Robot OS | ROS 2 Humble, MoveIt 2, MoveIt Task Constructor, ros2_control |

## Installation

### WP2 — Computer Vision

```bash
git clone --recurse-submodules https://github.com/<org>/VinPRO.git
cd VinPRO/wp2-computer-vision

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### WP3 — Control Stack

Requires **ROS 2 Humble**, **MoveIt 2**, **MoveIt Task Constructor**, and the [Kinova ROS 2 driver](https://github.com/Kinovarobotics/ros2_kortex).

```bash
cd VinPRO/wp3-control
colcon build --symlink-install
source install/setup.bash
```

## Quick Start

**Download the dataset:**

```bash
bash wp2-computer-vision/scripts/download_dataset.sh /data/3d2cut/
```

**Train the vision model:**

```bash
cd wp2-computer-vision
python train.py --data_path /data/3d2cut/ --max_epochs 300 --gpus 1
```

**Run inference on a single image:**

```bash
python predict.py --image path/to/vine.jpg \
                  --checkpoint models/model.pt \
                  --output result.png
```

**Launch the full robot system:**

```bash
ros2 launch vinpro_bringup full_system.launch.py \
    model_checkpoint:=/abs/path/to/model.pt
```

**Simulation only (no hardware required):**

```bash
ros2 launch vinpro_bringup sim_only.launch.py \
    model_checkpoint:=/abs/path/to/model.pt

# Inject a test cutting point
ros2 topic pub /pixel_coordinates std_msgs/msg/Int32MultiArray \
    "data: [320, 240]" --once
```

## Usage Examples

**Evaluate the vision pipeline:**

```bash
cd wp2-computer-vision
python evaluate.py --data_path /data/3d2cut/ \
                   --checkpoint models/model.pt
```

**View predicted graph structure (notebook):**

Open `wp2-computer-vision/notebooks/final_result.ipynb`.

**Hardware-in-the-loop test:**

```bash
ros2 launch vinpro_bringup hardware.launch.py \
    model_checkpoint:=/abs/path/to/model.pt \
    serial_port:=/dev/ttyACM0
```

**Check RViz cutting-point markers:**
Subscribe to `/cutting_point_markers` (red spheres at each cut location in `base_link` frame).

## Results

Vision pipeline evaluated on the **3D2cut Single Guyot** independent test set ([DOI: 10.34777/azf6-tm83](https://doi.org/10.34777/azf6-tm83), 258 images):

**AllNodeMetric** (distance threshold τ_d = 5 px):

| Metric | ViNet baseline¹ |
|--------|----------------|
| Precision | 0.95 |
| Recall | 0.90 |
| F-Score | 0.92 |

**CoursonMetric** (τ_d = 5 px):

| Metric | ViNet baseline¹ |
|--------|----------------|
| Precision | 0.76 |
| Recall | 0.74 |
| F-Score | 0.75 |

¹ From Gentilhomme et al., 2023. Full per-category results for our implementation: `wp2-computer-vision/notebooks/final_result.ipynb`.

## Current Status

The four work packages are independently functional and have been validated in lab conditions at **[PIC4SeR](https://pic4ser.polito.it/)**:

- **WP2** — vision pipeline produces accurate directed graphs on the 3D2cut dataset
- **WP3** — control stack successfully plans and executes cuts on a physical mock-up
- **WP4** — VR environment supports interactive pruning demonstrations with ROS 2 feedback
- **WP1** — custom end-effector operates reliably with the Kinova arm

End-to-end field deployment on natural vineyard conditions is ongoing. The next development priorities are shear-orientation alignment from WP2 branch vectors and extended field robustness testing.

## Contributing

This project was developed as an academic prototype. Issues and pull requests are welcome. For major changes please open an issue first to discuss scope.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use or build on this work, please cite the ViNet baseline method:

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

Developed within **[Alta Scuola Politecnica](https://www.asp-poli.it/)**, the joint excellence programme of Politecnico di Torino and Politecnico di Milano.

**Team:** Vincenzo Avantaggiato, Alberto Eusebio (WP2) · Riccardo Ghianni, Faik Tahirović (WP3) · Eleonora Troilo, Riccardo Vallino, Lorenzo Vignoli (WP1) · Francesco Risso (WP4)

**Principal Tutor:** Prof. Marcello Chiaberge (DET, PoliTo)

**Academic Tutors:** Luca Bascetta (DEIB, PoliMi), Mauro Martini, Marco Ambrosio, Alessandro Navone, Brenno Tuberga, Luigi Mazzara (DET, PoliTo)

**External Tutor:** Marta Niccolini (YANMAR R&D Europe Srl)

We thank:
- **[PIC4SeR](https://pic4ser.polito.it/)** for the robotic platform, lab facilities, and technical supervision
- **YANMAR R&D Europe Srl** for industry guidance under the YANMAR Smart Agriculture programme
- **Cantina 366** (Aglié, TO) for domain expertise and vineyard access
- **3D2cut SA** and the ViNet authors for the dataset and baseline method
