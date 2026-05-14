"""
Visualization utilities for ViNet predictions.

Provides functions to:
    - Recover heatmaps and vector fields from the stacked output tensor
    - Plot node heatmaps per (branch_type, node_type)
    - Plot vector field x/y components and magnitudes
    - Overlay the inferred tree structure on the original image
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torchvision.transforms import v2

from ..config import NODE_TYPES, BRANCH_TYPES, NUM_NODE_TYPES, NUM_BRANCH_TYPES


# ── Color Maps ────────────────────────────────────────────────────────────────

NODE_COLOR_MAP = {
    "mainTrunk": "red",
    "courson": "blue",
    "cane": "green",
    "lateralShoot": "orange",
    "shoot": "magenta",
    "rootCrown": "black",
}


# ── Tensor Utilities ──────────────────────────────────────────────────────────

def recover_heatmaps_vector_fields(
    M: torch.Tensor,
    resize: tuple[int, int] = (256, 256),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the stacked output tensor M into heatmaps and vector fields.

    Args:
        M: Stacked tensor of shape (C, H, W) or (B, C, H, W).
        resize: Spatial dimensions (H, W) for reshaping.

    Returns:
        heatmaps: (B?, N_bt, N_nt, H, W)
        vector_fields: (B?, N_bt, 2, H, W)
    """
    h, w = resize
    n_hm = NUM_BRANCH_TYPES * NUM_NODE_TYPES
    n_vf = NUM_BRANCH_TYPES * 2

    if M.dim() == 4:
        B = M.size(0)
        heatmaps = M[:, :n_hm].view(B, NUM_BRANCH_TYPES, NUM_NODE_TYPES, h, w)
        vector_fields = M[:, -n_vf:].view(B, NUM_BRANCH_TYPES, 2, h, w)
    else:
        heatmaps = M[:n_hm].view(NUM_BRANCH_TYPES, NUM_NODE_TYPES, h, w)
        vector_fields = M[-n_vf:].view(NUM_BRANCH_TYPES, 2, h, w)

    return heatmaps, vector_fields


# ── Plotting Functions ────────────────────────────────────────────────────────

def plot_heatmaps(
    heatmaps: torch.Tensor,
    image: torch.Tensor = None,
    figsize: tuple = (20, 20),
) -> None:
    """
    Plot node heatmaps in a (branch_type × node_type) grid.

    Args:
        heatmaps: Tensor of shape (N_bt, N_nt, H, W).
        image: Optional image tensor (C, H, W) to overlay.
        figsize: Figure size.
    """
    fig, axes = plt.subplots(NUM_BRANCH_TYPES, NUM_NODE_TYPES, figsize=figsize)
    for bname, bt in BRANCH_TYPES.items():
        for nname, nt in NODE_TYPES.items():
            ax = axes[bt, nt]
            if image is not None:
                img = v2.Resize(size=heatmaps.shape[-1])(image)
                ax.imshow(img.permute(1, 2, 0))
            hm = heatmaps[bt, nt]
            if isinstance(hm, torch.Tensor):
                hm = hm.numpy()
            ax.imshow(hm, cmap="hot", alpha=0.7 if image is not None else 1)
            ax.set_title(f"{bname} / {nname}")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_vector_fields(
    vector_fields: torch.Tensor,
    image: torch.Tensor = None,
    figsize: tuple = (15, 20),
) -> None:
    """
    Plot vector field x, y components and magnitude per branch type.

    Args:
        vector_fields: Tensor of shape (N_bt, 2, H, W).
        image: Optional image tensor (C, H, W) to overlay.
        figsize: Figure size.
    """
    fig, axes = plt.subplots(NUM_BRANCH_TYPES, 3, figsize=figsize)
    for bname, bt in BRANCH_TYPES.items():
        for i in range(2):
            ax = axes[bt, i]
            if image is not None:
                img = v2.Resize(size=vector_fields.shape[-1])(image)
                ax.imshow(img.permute(1, 2, 0))
            vf = vector_fields[bt, i]
            if isinstance(vf, torch.Tensor):
                vf = vf.numpy()
            ax.imshow(vf, cmap="viridis", alpha=0.7 if image is not None else 1)
            ax.set_title(f"{bname} / {'xy'[i]}")
            ax.axis("off")

        # Magnitude
        ax = axes[bt, 2]
        if image is not None:
            ax.imshow(img.permute(1, 2, 0))
        vx = vector_fields[bt, 0]
        vy = vector_fields[bt, 1]
        if isinstance(vx, torch.Tensor):
            vx, vy = vx.numpy(), vy.numpy()
        mag = np.sqrt(vx**2 + vy**2)
        ax.imshow(mag, cmap="viridis", alpha=0.9)
        ax.set_title(f"{bname} / magnitude")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_graph_on_image(
    graph: nx.DiGraph,
    image: np.ndarray,
    draw_resistivity: bool = False,
    figsize: tuple = (10, 10),
) -> None:
    """
    Overlay the inferred tree structure on the original image.

    Nodes are colored by branch type; edges follow the parent's color.

    Args:
        graph: Tree structure as a directed graph.
        image: Image array of shape (H, W, 3).
        draw_resistivity: Whether to annotate edges with resistivity values.
        figsize: Figure size.
    """
    image = cv2.resize(image, (256, 256)) if image.shape[:2] != (256, 256) else image

    pos = {(node, nt): (node[0], node[1]) for (node, nt) in graph.nodes}

    node_colors = []
    for (node, node_type) in graph.nodes:
        main_type = node_type[0]
        if node_type[1] == "rootCrown":
            main_type = "rootCrown"
        node_colors.append(NODE_COLOR_MAP.get(main_type, "gray"))

    edge_colors = [
        NODE_COLOR_MAP.get(v[1][0], "gray") for (u, v) in graph.edges
    ]

    plt.figure(figsize=figsize)
    plt.imshow(image, alpha=0.3)
    plt.axis("off")

    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, alpha=0.7, arrows=True)

    if draw_resistivity:
        edge_labels = nx.get_edge_attributes(graph, "weight")
        formatted = {edge: f"{w:.2f}" for edge, w in edge_labels.items()}
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=formatted, font_size=8, font_color="green"
        )

    plt.title("Predicted Grapevine Structure")
    plt.show()
