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
    d = int(alpha_lm * W)  # Neighborhood distance d = α_lm * W_h

    # Step 1: For each pixel, compute the local maximum within distance d.
    local_max = maximum_filter(prediction_map, size=d, mode="constant")

    # Step 2: Where prediction < τ_n, set local_max to 1 so that
    # normalized value stays below τ_m (low-confidence pixels won't be detected).
    local_max_safe = np.where(prediction_map >= tau_n, local_max, 1.0)

    # Step 3: Per-pixel normalization: M_c = M_c / M_c^max (Section 2.3.3).
    normalized_map = prediction_map / local_max_safe

    # Step 4: Threshold for binary map
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
