"""
Single-image inference script for ViNet.

Loads a trained model, runs inference on a single vine image,
extracts the tree structure, and saves/displays the result.

Usage:
    python predict.py --image path/to/vine.jpg \
                      --checkpoint path/to/model.pt \
                      --output prediction.png
"""

import argparse

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from vinet.config import (
    NODE_TYPES,
    BRANCH_TYPES,
    NUM_OUTPUT_CHANNELS,
    DEFAULT_RESIZE,
    DEFAULT_FRONT_CHANNELS,
    DEFAULT_HOURGLASS_CHANNELS,
    POSSIBLE_PARENTS,
)
from vinet.model import StackedHourglassNetwork
from vinet.inference import (
    extract_node_coordinates,
    construct_resistivity_graph,
    grapevine_structure_estimation,
    recover_heatmaps_vector_fields,
)


NODE_COLOR_MAP = {
    "mainTrunk": (255, 0, 0),
    "courson": (0, 0, 255),
    "cane": (0, 200, 200),
    "shoot": (255, 0, 255),
    "lateralShoot": (255, 165, 0),
    "rootCrown": (0, 0, 0),
}


def preprocess_image(
    image_path: str,
    crop_size: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image for inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)

    tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return tensor, image_resized


def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass and return heatmaps and vector fields."""
    model.eval()
    with torch.no_grad():
        _, output2 = model(image_tensor)

    heatmaps, vector_fields = recover_heatmaps_vector_fields(
        output2[0].cpu(), resize=DEFAULT_RESIZE
    )
    return heatmaps, vector_fields


def extract_all_nodes(heatmaps: torch.Tensor) -> dict:
    """Extract nodes from all (branch_type, node_type) heatmap channels."""
    total_nodes = {}
    for bname, bt in BRANCH_TYPES.items():
        for nname, nt in NODE_TYPES.items():
            hm = heatmaps[bt, nt].numpy()
            coords = extract_node_coordinates(hm)
            total_nodes[(bname, nname)] = coords
    return total_nodes


def build_tree(
    total_nodes: dict,
    vector_fields: torch.Tensor,
) -> nx.DiGraph:
    """Construct the resistivity graph and estimate the tree structure."""
    vf_np = vector_fields.numpy()

    graph = construct_resistivity_graph(
        total_nodes,
        branch_types=BRANCH_TYPES,
        vector_fields=vf_np,
        possible_parents=POSSIBLE_PARENTS,
        radius=np.inf,
    )

    # Find root crown
    root_coords = total_nodes.get(("mainTrunk", "rootCrown"), [(0, 0)])[0]
    root_node = (root_coords, ("mainTrunk", "rootCrown"))

    tree = grapevine_structure_estimation(graph, root_node)
    return tree


def draw_prediction(
    image: np.ndarray,
    tree: nx.DiGraph,
    total_nodes: dict,
    output_path: str = None,
) -> None:
    """Draw the predicted tree structure overlaid on the image."""
    display = cv2.resize(image.copy(), (256, 256))

    # Draw edges
    for (child, child_type), (parent, parent_type) in tree.edges:
        color = NODE_COLOR_MAP.get(parent_type[0], (128, 128, 128))
        pt1 = (int(child[0]), int(child[1]))
        pt2 = (int(parent[0]), int(parent[1]))
        cv2.line(display, pt1, pt2, color, 1, cv2.LINE_AA)

    # Draw nodes
    for (node, node_type) in tree.nodes:
        main_type = node_type[0]
        if node_type[1] == "rootCrown":
            main_type = "rootCrown"
        color = NODE_COLOR_MAP.get(main_type, (128, 128, 128))
        center = (int(node[0]), int(node[1]))
        cv2.circle(display, center, 3, color, -1, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(cv2.resize(image, (256, 256)))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(display)
    axes[1].set_title("Predicted Structure")
    axes[1].axis("off")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=np.array(c) / 255, label=name)
        for name, c in NODE_COLOR_MAP.items()
    ]
    axes[1].legend(handles=legend_elements, loc="lower left", fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved prediction to {output_path}")
    else:
        plt.show()

    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="ViNet single-image inference")
    parser.add_argument("--image", type=str, required=True, help="Path to vine image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--front_channels", type=int, default=DEFAULT_FRONT_CHANNELS)
    parser.add_argument("--hourglass_channels", type=int, default=DEFAULT_HOURGLASS_CHANNELS)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = StackedHourglassNetwork(
        in_channels=3,
        front_channels=args.front_channels,
        hourglass_channels=args.hourglass_channels,
        num_output_channels=NUM_OUTPUT_CHANNELS,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Preprocess image
    print(f"Processing {args.image}...")
    image_tensor, image_np = preprocess_image(args.image, device=device)

    # Inference
    heatmaps, vector_fields = run_inference(model, image_tensor, device)

    # Node extraction
    total_nodes = extract_all_nodes(heatmaps)
    n_nodes = sum(
        len([c for c in coords if c != (0, 0)])
        for coords in total_nodes.values()
    )
    print(f"Detected {n_nodes} nodes")

    # Tree structure estimation
    tree = build_tree(total_nodes, vector_fields)
    print(f"Tree structure: {tree.number_of_nodes()} nodes, {tree.number_of_edges()} edges")

    # Draw and save
    output_path = args.output or args.image.rsplit(".", 1)[0] + "_prediction.png"
    draw_prediction(image_np, tree, total_nodes, output_path)


if __name__ == "__main__":
    main()
