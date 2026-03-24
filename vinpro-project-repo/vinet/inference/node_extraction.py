"""
Node coordinate extraction from predicted heatmaps (Section 2.3.3).

Pipeline:
    1. Normalize prediction map using local maxima
    2. Threshold to obtain binary map
    3. Connected component analysis to find blob centers
    4. Return (x, y) coordinates of detected nodes
"""

import numpy as np
from scipy.ndimage import maximum_filter, label, center_of_mass

from ..config import DEFAULT_TAU_N, DEFAULT_TAU_M, DEFAULT_ALPHA_LM


def extract_node_coordinates(
    prediction_map: np.ndarray,
    tau_n: float = DEFAULT_TAU_N,
    tau_m: float = DEFAULT_TAU_M,
    alpha_lm: float = DEFAULT_ALPHA_LM,
) -> list[tuple[int, int]]:
    """
    Extract 2D node coordinates from a predicted heatmap.

    Identifies local maxima within sufficiently large blobs, following
    the procedure in Section 2.3.3 of the paper.

    Args:
        prediction_map: Predicted heatmap of shape (H, W).
        tau_n: Minimum confidence threshold (default: 0.5).
        tau_m: Local maxima threshold (default: 0.97).
        alpha_lm: Neighborhood distance factor (default: 0.1).

    Returns:
        List of (x, y) coordinates of detected nodes.
    """
    H, W = prediction_map.shape
    d = int(alpha_lm * W)  # Neighborhood distance

    # Step 1: Compute local maximum map M^max_c
    local_max = maximum_filter(prediction_map, size=d, mode="constant")
    local_max_map = np.where(prediction_map >= local_max, prediction_map, 0)

    # Step 2: Normalize by local maxima (filter out low-confidence regions)
    local_max_map = np.where(prediction_map >= tau_n, local_max_map, 0)
    max_val = np.max(local_max_map)
    if max_val > 0:
        normalized_map = local_max_map / max_val
    else:
        return [(0, 0)]

    # Step 3: Threshold for binary map
    binary_map = (normalized_map >= tau_m).astype(np.uint8)

    # Step 4: Connected component analysis → extract blob centers
    labeled_map, num_features = label(binary_map)
    coordinates = []
    for i in range(1, num_features + 1):
        coords = center_of_mass(binary_map, labeled_map, i)
        coordinates.append((int(coords[1]), int(coords[0])))  # (x, y) format

    if len(coordinates) == 0:
        coordinates = [(0, 0)]

    return coordinates
