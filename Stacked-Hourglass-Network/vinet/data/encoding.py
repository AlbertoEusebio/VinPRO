"""
Ground-truth encoding utilities for ViNet.

Generates node heatmaps (Gaussian blobs) and branch vector fields
(Part Affinity Fields) from annotated vine structure data.

References:
    - Eq. 1 in Gentilhomme et al. (2023): Node heatmap definition
    - Eq. 2 in Gentilhomme et al. (2023): Affinity field definition
    - Cao et al. (2019): Part Affinity Fields for pose estimation
"""

import json
import numpy as np


# ── Annotation Parsing ────────────────────────────────────────────────────────

def load_annotation(json_file: str) -> dict:
    """Load a JSON annotation file for a vine image."""
    with open(json_file, "r") as f:
        return json.load(f)


def parse_features(annotation: dict) -> tuple[list, list]:
    """
    Extract vine features (nodes and branches) from the annotation JSON.

    Returns:
        nodes: List of dicts with keys: id, coordinates, type, branch_label
        branches: List of dicts with keys: parent_id, child_id, branch_label
    """
    vine_images = annotation.get("VineImage", None)
    if vine_images is None or len(vine_images) == 0:
        raise KeyError("'VineImage' key not found or empty in the JSON file.")

    features = vine_images[0].get("VineFeature", None)
    if features is None:
        raise KeyError("'VineFeature' key not found under 'VineImage'.")

    nodes = []
    branches = []

    for feature_list in features:
        for feature in feature_list:
            if (
                "FeatureType" in feature
                and feature["FeatureType"]
                and "ParentID" in feature
                and feature["BranchLabel"]
            ):
                branch = (
                    "mainTrunk"
                    if feature["BranchLabel"] == "root"
                    else feature["BranchLabel"]
                )
                branches.append(
                    {
                        "parent_id": feature["ParentID"],
                        "child_id": feature["FeatureID"],
                        "branch_label": branch,
                    }
                )
                nodes.append(
                    {
                        "id": feature["FeatureID"],
                        "coordinates": feature["FeatureCoordinates"],
                        "type": feature["FeatureType"],
                        "branch_label": branch,
                    }
                )
    return nodes, branches


def convert_nodes(
    nodes: list, branches: list, branch_types: dict, node_types: dict
) -> list:
    """Convert parsed nodes into (coordinates, node_type_idx, branch_type_idx) tuples."""
    new_nodes = []
    for n in nodes:
        if n["type"] not in node_types or n["branch_label"] not in branch_types:
            continue
        new_nodes.append(
            (n["coordinates"], node_types[n["type"]], branch_types[n["branch_label"]])
        )
    return new_nodes


# ── Vector Field Generation ───────────────────────────────────────────────────

def _get_branch_couples(branches: list, nodes: list) -> dict:
    """Group branch segments as (parent_coords, child_coords) pairs by branch type."""
    couples = {
        "mainTrunk": [],
        "courson": [],
        "cane": [],
        "shoot": [],
        "lateralShoot": [],
    }
    for b in branches:
        parent_node = next((n for n in nodes if n["id"] == b["parent_id"]), None)
        child_node = next((n for n in nodes if n["id"] == b["child_id"]), None)
        if parent_node is None or child_node is None:
            continue
        if b["branch_label"] not in couples:
            continue
        couples[b["branch_label"]].append(
            (parent_node["coordinates"], child_node["coordinates"])
        )
    return couples


def generate_vector_field(
    image_size: tuple,
    field_size: tuple,
    branches: list,
    limb_width: float,
) -> np.ndarray:
    """
    Generate a vector field (Part Affinity Field) for a set of branch segments.

    For each branch segment between two nodes, the unit direction vector is
    assigned to all pixels within `limb_width` of the line segment (Eq. 2).

    Args:
        image_size: Original image dimensions (H, W).
        field_size: Target field resolution (H_field, W_field).
        branches: List of [(x1, y1), (x2, y2)] segment pairs.
        limb_width: Half-width of the vector field band around each segment.

    Returns:
        Vector field of shape (H_field, W_field, 2) with x, y components.
    """
    height, width = image_size
    H, W = field_size
    vector_field = np.zeros((H, W, 2), dtype=np.float32)

    for (x1, y1), (x2, y2) in branches:
        # Scale to field resolution
        x1_s = x1 / width * W
        y1_s = y1 / height * H
        x2_s = x2 / width * W
        y2_s = y2 / height * H

        dx, dy = x2_s - x1_s, y2_s - y1_s
        seg_len_sq = dx**2 + dy**2
        if seg_len_sq == 0:
            continue
        seg_len = np.sqrt(seg_len_sq)
        unit_vec = (dx / seg_len, dy / seg_len)

        # Bounding box around the segment
        x_min = max(0, int(min(x1_s, x2_s) - limb_width))
        x_max = min(W, int(max(x1_s, x2_s) + limb_width))
        y_min = max(0, int(min(y1_s, y2_s) - limb_width))
        y_max = min(H, int(max(y1_s, y2_s) + limb_width))

        x_grid, y_grid = np.meshgrid(
            np.arange(x_min, x_max), np.arange(y_min, y_max)
        )

        # Project each pixel onto the segment, clamp to [0, 1]
        t = ((x_grid - x1_s) * dx + (y_grid - y1_s) * dy) / seg_len_sq
        t_clamped = np.clip(t, 0, 1)

        closest_x = x1_s + t_clamped * dx
        closest_y = y1_s + t_clamped * dy
        distances = np.sqrt((x_grid - closest_x) ** 2 + (y_grid - closest_y) ** 2)

        mask = distances <= limb_width
        vector_field[y_min:y_max, x_min:x_max, 0][mask] = unit_vec[0]
        vector_field[y_min:y_max, x_min:x_max, 1][mask] = unit_vec[1]

    return vector_field


def get_vector_fields(
    image_size: tuple,
    field_size: tuple,
    nodes: list,
    branches: list,
    limb_width: float = 3,
) -> dict:
    """Generate vector fields for all branch types."""
    couples = _get_branch_couples(branches, nodes)
    vector_fields = {}
    for branch_type, segments in couples.items():
        vector_fields[branch_type] = generate_vector_field(
            image_size, field_size, segments, limb_width
        )
    return vector_fields


# ── Heatmap Generation ────────────────────────────────────────────────────────

def generate_node_heatmaps(
    image_size: tuple,
    nodes: list,
    sigma: float,
    num_node_types: int,
    num_branch_types: int,
    new_size: tuple,
) -> np.ndarray:
    """
    Generate Gaussian heatmaps for each (branch_type, node_type) combination.

    Each node produces a Gaussian blob centered at its location (Eq. 1).

    Args:
        image_size: Original image dimensions (H, W).
        nodes: List of (coordinates, node_type_idx, branch_type_idx) tuples.
        sigma: Gaussian standard deviation.
        num_node_types: Number of node categories.
        num_branch_types: Number of branch categories.
        new_size: Target heatmap resolution (H_new, W_new).

    Returns:
        Heatmaps of shape (num_branch_types, num_node_types, H_new, W_new).
    """
    height, width = image_size
    new_height, new_width = new_size
    heatmaps = np.zeros(
        (num_branch_types, num_node_types, new_height, new_width), dtype=np.float32
    )
    scale_x = new_width / width
    scale_y = new_height / height

    for (x, y), node_type, branch_type in nodes:
        x_s = int(x * scale_x)
        y_s = int(y * scale_y)

        # Bounding box for efficiency; use 5σ radius for the exponential-decay
        # formula (Eq. 1) which decays slower than a standard Gaussian.
        x_min = max(0, int(x_s - 5 * sigma))
        x_max = min(new_width, int(x_s + 5 * sigma))
        y_min = max(0, int(y_s - 5 * sigma))
        y_max = min(new_height, int(y_s + 5 * sigma))

        if x_max <= x_min or y_max <= y_min:
            continue

        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        # Eq. 1: M*_c(p) = max_j exp(-||p - x_j|| / σ²)
        # Numerator is L2 distance (not squared); denominator is σ² (not 2σ²).
        gaussian = np.exp(
            -np.sqrt((x_grid - x_s) ** 2 + (y_grid - y_s) ** 2) / (sigma ** 2)
        )
        heatmaps[branch_type, node_type, y_min:y_max, x_min:x_max] += gaussian

    np.clip(heatmaps, 0, 1, out=heatmaps)
    return heatmaps
