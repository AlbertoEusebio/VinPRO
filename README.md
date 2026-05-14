# VinPRO — Vineyard Pruning with Robots through Collaborative Learning in Virtual Reality

[![Alta Scuola Politecnica](https://img.shields.io/badge/Alta%20Scuola%20Politecnica-Multidisciplinary%20Project-8B0000)](https://www.asp-poli.it/)
[![PIC4SeR](https://img.shields.io/badge/Lab-PIC4SeR%20PoliTo%2FPoliMi-003399)](https://pic4ser.polito.it/)
[![Partner](https://img.shields.io/badge/Industry-YANMAR%20R%26D%20Europe-FF6600)](https://www.yanmar.com/eu/)
[![Vineyard](https://img.shields.io/badge/Field%20Partner-Cantina%20366%20Agli%C3%A9%20TO-darkgreen)]()

VinPRO is a 1 year and 6 months **[Alta Scuola Politecnica](https://www.asp-poli.it/)** multidisciplinary project that developed a proof-of-concept robotic vineyard pruning system. It integrates a custom pruning end-effector, a deep learning computer vision pipeline, a motion planning control stack, and a Virtual Reality training environment. From perceiving a vine in a single RGB image to physically executing a precision cut with a robotic arm.

The project was developed in collaboration with **[PIC4SeR](https://pic4ser.polito.it/)** (Interdepartmental Centre for Service Robotics, Politecnico di Torino / Politecnico di Milano), **YANMAR R&D Europe Srl**, and **Cantina 366** (Aglié, TO).

---

## Demo

<video src="assets/videos/presentation.mp4" controls width="100%"></video>

> Recorded at **[PIC4SeR](https://pic4ser.polito.it/)** (Turin) during final integration testing on a grapevine mock-up, showing the complete pipeline from image capture to robotic cut.

---

## Motivation

Vineyard pruning is a high-skill, high-cost, seasonal task. It accounts for up to **25% of annual labor costs** in fruit production and must be completed within a narrow seasonal window. Hand pruning represents **75% of yearly vineyard labor demand**; full mechanization can reduce this by 50–90% but at the cost of selectivity and cut quality. Italy — the world's largest wine producer at ≈49 million hectoliters/year — saw its number of wineries fall by one-third between 2010 and 2020, sharpening the labor shortage.

Existing automated solutions are either too expensive, require vineyard restructuring, or sacrifice pruning precision. The *Bumblebee* end-to-end robotic system reportedly achieves 87% pruning accuracy but requires 213 seconds per vine — too slow for practical deployment. VinPRO's goal is to move beyond these coarse approaches by **encoding expert viticultural knowledge into an autonomous robot** that is cost-effective for small and medium producers.

Pruning rules were contributed directly by the owner of **Cantina 366** (Aglié, TO): he demonstrated hand techniques, explained how strategy varies across plant types, and emphasized that production costs must remain below the price point of ≈€4/L. This domain knowledge is encoded in the pruning policy algorithm (see [WP2](#wp2--computer-vision)).

---

## Partners and Institutions

| Role | Organisation |
|------|-------------|
| **Academic host** | [Alta Scuola Politecnica](https://www.asp-poli.it/) — joint honours programme of Politecnico di Torino and Politecnico di Milano |
| **Research lab** | [PIC4SeR](https://pic4ser.polito.it/) — Interdepartmental Centre for Service Robotics (PoliTo / PoliMi). Provided the Kinova Gen3 Lite arm, mobile rover, 3D-printing facilities, and lab space. Hosted all integration testing. |
| **Industry partner** | **YANMAR R&D Europe Srl** — technological and commercial support under the YANMAR Smart Agriculture programme. External tutor: *Marta Niccolini* (Robotics Group Leader). |
| **Domain expert** | **Cantina 366** (Aglié, TO) — Italian winery; contributed pruning expertise and field access. The owner personally defined the pruning rules encoded in the policy algorithm. |
| **Baseline method** | **3D2cut SA** and the ViNet team — authors of the ViNet method and the 3D2cut Single Guyot dataset used to train the vision pipeline. |

---

## Team

| Name | Background | Work Package |
|------|-----------|-------------|
| Vincenzo Avantaggiato | MSc Computer Engineering, Politecnico di Torino | WP2 — Computer Vision |
| Alberto Eusebio | MSc Computer Science and Engineering, Politecnico di Milano | WP2 — Computer Vision |
| Riccardo Ghianni | MSc Quantum Engineering, Politecnico di Torino | WP3 — Control & Planning |
| Faik Tahirović | MSc Automation and Control Engineering, Politecnico di Milano | WP3 — Control & Planning |
| Eleonora Troilo | MSc Mechanical Engineering, Politecnico di Torino | WP1 — Gripper Design |
| Riccardo Vallino | MSc Aerospace Engineering, Politecnico di Torino | WP1 — Gripper Design |
| Lorenzo Vignoli | MSc Mechanical Engineering, Politecnico di Milano | WP1 — Gripper Design |
| Francesco Risso | MSc Computer Engineering, Politecnico di Torino | WP4 — Virtual Reality |

**Principal Academic Tutor:** Prof. Marcello Chiaberge, Dipartimento di Elettronica e Telecomunicazioni, Politecnico di Torino

**Academic Tutors:** Luca Bascetta (DEIB, PoliMi), Mauro Martini, Marco Ambrosio, Alessandro Navone, Brenno Tuberga, Luigi Mazzara (all DET, PoliTo)

**External Tutor:** Marta Niccolini, YANMAR R&D Europe Srl

---

## System Overview

The four Work Packages are tightly coupled: WP2 perception quality directly affects WP3 motion planning, WP1 mechanical reliability determines whether planned cuts succeed, and WP4 depends on data from all three to build a realistic simulation.

```
RGB Image (Intel RealSense D435i)
        │
        ▼
┌──────────────────────────────────────┐
│  WP2 — Computer Vision               │
│  Stacked Hourglass Network (2×)      │
│  → 20 node heatmaps + 10 vec. fields │
│  → Peak detection → node coords      │
│  → Resistivity graph + Dijkstra      │
│  → Directed plant graph              │
│  → Pruning policy (partition nodes)  │
│  → Cutting point (pixel coords)      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  WP3 — Control & Planning            │
│  Camera subscriber → camera frame    │
│  TF2 transform → robot base frame    │
│  MoveIt Task Constructor stages      │
│  ros2_control + scissors interface   │
│  Collision-free trajectory           │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  WP1 — Gripper & End-Effector        │
│  Kinova Gen3 Lite 6-DOF arm          │
│  Custom 3D-printed polymer housing   │
│  Electric shear + Arduino Nano       │
└──────────────────────────────────────┘
```

**WP4 (Virtual Reality)** runs in parallel: a Unity (URP) vineyard with full ROS 2 integration serves as a risk-free testing ground and operator training environment.

---

## Work Packages

### WP1 — Gripper Design

The Kinova Gen3 Lite's original gripper could not be removed, so the team designed a custom polymer housing — a mechanical "glove" — that encloses the existing gripper and integrates the vision system, cutting tool, actuator, and control electronics. The camera is positioned on the upper part of the housing to maximize plant visibility and minimize blade occlusion.

Two actuation concepts were evaluated:

| Option | Assessment |
|--------|-----------|
| **Pneumatic** | Required compressor, tank, solenoid valve, end-stroke sensor — rejected as too heavy and bulky |
| **Electric** | Reuses components from a commercial electric shear (bypass shears, reduction gear, motor, actuator board) — selected for its compact footprint |

An **Arduino Nano** manages the unlock → cut → lock cycle and receives commands from the ROS 2 control stack via `ros2_control`.

All mechanical components were fabricated at **PIC4SeR** (Turin).

---

### WP2 — Computer Vision

> **Repository:** [`Stacked-Hourglass-Network/`](Stacked-Hourglass-Network/README.md)

The vision pipeline is a re-implementation of **ViNet** (Gentilhomme et al., 2023) adapted for the robotic pruning workflow. Given a single RGB image, it produces a directed graph encoding every vine node's location, type, and branch category.

**Detection step** — A Stacked Hourglass Network predicts:
- 20 node heatmaps (5 branch types × 4 node types)
- 10 vector field maps (5 branch types × 2 directional components)

**Association step** — A sparse resistivity graph is built among anatomically compatible node pairs. Edge weights follow:

```
R(c → p) = (1 − A) × ‖v_cp‖
```

where A is the mean cosine alignment of the predicted vector field along the child-to-parent segment. Dijkstra's shortest path from each node to the root crown yields the directed tree (each node has at most one parent).

**Pruning policy** — The policy takes the plant graph and:
1. Identifies **partition nodes** — nodes with more than one predecessor, marking branch divisions.
2. Eliminates partition nodes whose successors already include another partition node (cutting closer to the trunk makes the distal cut redundant).
3. Places the actual cut at the **midpoint between a partition node and its predecessor**, ensuring the cut lands on a branch segment rather than at a junction.
4. Preserves **one cane** (one-year-old shoot) to ensure next season's yield.

> **Baseline paper:**
> *Towards smart pruning: ViNet, a deep-learning approach for grapevine structure estimation*
> Theophile Gentilhomme, Michael Villamizar, Jerome Corre, Jean-Marc Odobez
> *Computers and Electronics in Agriculture*, 207, 107736 (2023)
> [https://doi.org/10.1016/j.compag.2023.107736](https://doi.org/10.1016/j.compag.2023.107736)

Training data: **3D2cut Single Guyot** — 1,513 annotated grapevine images, 1,255 train / 258 test
([https://doi.org/10.34777/azf6-tm83](https://doi.org/10.34777/azf6-tm83), CC BY-NC-SA)

---

### WP3 — Control & Planning

The control system is implemented in **ROS 2 Humble** with **MoveIt 2**, **MoveIt Task Constructor**, and **ros2_control**. The ROS 2 architecture consists of:

1. **Camera subscriber node** — converts pixel/depth pruning-point coordinates into camera-frame 3D positions.
2. **Transformer / MTC initializer** — transforms coordinates into the robot base frame via **TF2** and launches task execution.
3. **MTC pipeline constructor** — builds modular motion stages (approach, orient, cut, retract) and runs inverse kinematics. Stages can be previewed in RViz before real deployment.
4. **Scissors hardware interface** — bridges `ros2_control` commands to the **Arduino Nano**, which triggers the electric shear.

**Hardware used:** Kinova Gen3 Lite (6 DOF, millimeter-scale repeatability, joint torque sensing, native ROS 2 driver), Intel RealSense D435i (RGB-D + point cloud), LiDAR (environment mapping / obstacle avoidance).

---

### WP4 — Virtual Reality Simulation

> **Repository:** [`Plant-generate-cut/`](Plant-generate-cut/)

**Blender (procedural generation)** — `Plant-generate-cut/main.py` implements a recursive rule-based algorithm that assembles vine components (trunk, branches, buds, leaves, grape bunches) according to botanical growth constraints. GJK-based collision detection prevents unrealistic intersections. The algorithm checks orientation, branch length, and bifurcation angles, then exports FBX assets.

**Unity (real-time simulation)** — FBX assets are imported into a Unity (URP) scene. Rather than cutting arbitrary meshes at runtime (too computationally expensive), the team predefined **cutting planes** in the 3D models — including both agronomically correct and incorrect cut positions. During simulation, triggering a cut separates the vine at the nearest cutting plane, applies rigid-body physics to the detached fragment, and gives the operator immediate pruning feedback. The environment integrates with **ROS 2** for closed-loop testing of the full control stack.

---

## Repository Structure

```
VinPRO/
├── README.md                           # This file
├── .gitmodules                         # Submodule declarations
│
├── Stacked-Hourglass-Network/          # WP2: Computer Vision pipeline
│   ├── README.md
│   ├── train.py / evaluate.py / predict.py
│   ├── vinet/                          # Core ML package
│   │   ├── config.py                   # Node/branch types, hyperparameters
│   │   ├── data/                       # VineDataset, transforms, encoding
│   │   ├── model/                      # StackedHourglassNetwork, Lightning module
│   │   └── inference/                  # Node extraction, resistivity graph, viz
│   ├── models/                         # Trained checkpoints (.pt)
│   ├── notebooks/                      # Experiment and result notebooks
│   └── outputs/                        # Predicted graph adjacency lists
│
├── Plant-generate-cut/                 # WP4: Blender + Unity simulation (submodule)
│   ├── main.py                         # Procedural vine generator (Blender Python)
│   └── VinPRO cutting demonstrator/    # Unity URP project
│       └── Assets/Scripts/
│           ├── Plants/PlantGenerator.cs
│           ├── Plants/PlantPart.cs
│           ├── Plants/PlantCutTrigger.cs
│           └── Cutter.cs
│
└── assets/
    ├── VinPRO_final_report.pdf         # Full 38-page academic final report
    ├── VinPRO_poster.pdf               # Project poster / executive summary
    └── videos/
        └── presentation.mp4            # Integration demo video (PIC4SeR lab)
```

---

## Hardware Platform

| Component | Specification |
|-----------|--------------|
| Robotic arm | Kinova Gen3 Lite — 6 DOF, joint torque sensing, mm-scale repeatability |
| Mobile base | Autonomous rover (PIC4SeR, PoliTo) |
| RGB-D camera | Intel RealSense D435i |
| Environment sensor | LiDAR |
| End-effector housing | 3D-printed polymer "glove" (fabricated at PIC4SeR) |
| Actuator | Electric pruning shear + compact reduction gear + electric motor |
| Microcontroller | Arduino Nano (unlock / cut / lock cycle) |
| Robot OS | ROS 2 Humble + MoveIt 2 + MoveIt Task Constructor + ros2_control |

---

## Results

WP2 vision pipeline, evaluated on the 3D2cut Single Guyot independent test set (258 images):

**AllNodeMetric** (τ_d = 5):

| Metric | Baseline ViNet (Gentilhomme et al., 2023) | VinPRO |
|--------|-------------------------------------------|--------|
| Precision | 0.95 | — |
| Recall | 0.90 | — |
| F-Score | 0.92 | — |

**CoursonMetric** (τ_d = 5):

| Metric | Baseline ViNet | VinPRO |
|--------|---------------|--------|
| Precision | 0.76 | — |
| Recall | 0.74 | — |
| F-Score | 0.75 | — |

Per-category and full results: `Stacked-Hourglass-Network/notebooks/final_result.ipynb`

---

## Acknowledgments

This project was developed within **[Alta Scuola Politecnica](https://www.asp-poli.it/)**, the joint excellence programme of Politecnico di Torino and Politecnico di Milano.

We gratefully acknowledge:

- **[PIC4SeR](https://pic4ser.polito.it/)** — Interdepartmental Centre for Service Robotics (PoliTo / PoliMi) — for the robotic platform, mobile rover, 3D-printing facilities, and lab space for all integration and validation testing.
- **YANMAR R&D Europe Srl** — for technological and commercial guidance under the YANMAR Smart Agriculture programme.
- **Cantina 366** (Aglié, TO) — for sharing expert pruning knowledge, hosting a field visit, and providing the domain rules encoded in the pruning policy.
- **3D2cut SA** and the authors of ViNet — for releasing the annotated dataset and the method that served as the WP2 baseline.

---

## Citation

If you build on this work, please cite the ViNet baseline:

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

## Selected References

1. Navone, A., Martini, M., and Chiaberge, M., "Autonomous Robotic Pruning in Orchards and Vineyards: a Review," May 2024.
2. Allegro, G. et al., "Effects of Mechanical Winter Pruning on Vine Performances and Management Costs in a Trebbiano Romagnolo Vineyard: A Five-Year Study," *Horticulturae*, 2022.
3. Gentilhomme, T. et al., "Towards smart pruning: ViNet," *Computers and Electronics in Agriculture*, 207, 107736 (2023).
4. Chen, Z. et al., "Grapevine Branch Recognition and Pruning Point Localization Technology Based on Image Processing," *Applied Sciences*, 2022.
5. Zhang, J. et al., "Branch detection for apple trees trained in fruiting wall architecture using depth features and R-CNN," *Computers and Electronics in Agriculture*.
6. Borrenpohl, D. and Karkee, M., "Automated pruning decisions in dormant sweet cherry canopies using instance segmentation," *Computers and Electronics in Agriculture*.
