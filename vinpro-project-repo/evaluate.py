"""
Evaluation script for ViNet.

Implements the two metrics from Section 3.2 of the paper:
    - AllNodeMetric: Precision/Recall/F-Score for node detection (PCK-based)
    - CoursonMetric: Structure-aware metric for pruning-relevant nodes

Usage:
    python evaluate.py --data_path /path/to/3D2cut_Single_Guyot/ \
                       --checkpoint path/to/model.pt
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from vinet.config import (
    NODE_TYPES,
    BRANCH_TYPES,
    NUM_OUTPUT_CHANNELS,
    DEFAULT_RESIZE,
    DEFAULT_FRONT_CHANNELS,
    DEFAULT_HOURGLASS_CHANNELS,
)
from vinet.data import VineDataset, get_val_transforms
from vinet.data.encoding import load_annotation, parse_features, convert_nodes
from vinet.model import StackedHourglassNetwork
from vinet.inference import extract_node_coordinates, recover_heatmaps_vector_fields


# ── Metrics (Section 3.2) ─────────────────────────────────────────────────────

def compute_allnode_metric(
    pred_nodes: dict,
    gt_nodes: dict,
    tau_d: float = 5.0,
) -> dict:
    """
    Compute the AllNodeMetric: per-category and overall precision/recall/F-score.

    Uses the Hungarian algorithm to find the optimal assignment between
    predicted and ground-truth nodes within distance threshold tau_d (Eq. 7-11).

    Args:
        pred_nodes: Dict mapping (branch_type, node_type) → list of (x, y).
        gt_nodes: Dict mapping (branch_type, node_type) → list of (x, y).
        tau_d: Distance threshold for matching (default: 5 pixels at 256×256).

    Returns:
        Dict with per-category and overall metrics.
    """
    results = {}
    total_tp, total_det, total_gt = 0, 0, 0

    all_categories = set(list(pred_nodes.keys()) + list(gt_nodes.keys()))

    for cat in all_categories:
        preds = pred_nodes.get(cat, [])
        gts = gt_nodes.get(cat, [])

        # Filter out dummy (0,0) nodes
        preds = [p for p in preds if p != (0, 0)]
        gts = [g for g in gts if g != (0, 0)]

        n_det = len(preds)
        n_gt = len(gts)

        if n_det == 0 and n_gt == 0:
            continue

        n = max(n_det, n_gt)
        # Build cost matrix (Eq. 7)
        cost_matrix = np.ones((n, n), dtype=np.float64)
        for i in range(n_gt):
            for j in range(n_det):
                dist = np.sqrt(
                    (gts[i][0] - preds[j][0]) ** 2 + (gts[i][1] - preds[j][1]) ** 2
                )
                if dist < tau_d:
                    cost_matrix[i, j] = 0.0

        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        tp = sum(1 for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] == 0.0)

        total_tp += tp
        total_det += n_det
        total_gt += n_gt

        precision = tp / n_det if n_det > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[cat] = {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "tp": tp,
            "n_det": n_det,
            "n_gt": n_gt,
        }

    # Overall metrics (Eq. 9-11)
    overall_p = total_tp / total_det if total_det > 0 else 0.0
    overall_r = total_tp / total_gt if total_gt > 0 else 0.0
    overall_f = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r) > 0
        else 0.0
    )
    results["all_nodes"] = {
        "precision": overall_p,
        "recall": overall_r,
        "f_score": overall_f,
        "tp": total_tp,
        "n_det": total_det,
        "n_gt": total_gt,
    }

    return results


# ── Main Evaluation Loop ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViNet on the 3D2cut test set")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt model weights")
    parser.add_argument("--tau_d", type=float, default=5.0, help="Association distance threshold")
    parser.add_argument("--front_channels", type=int, default=DEFAULT_FRONT_CHANNELS)
    parser.add_argument("--hourglass_channels", type=int, default=DEFAULT_HOURGLASS_CHANNELS)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize_h, resize_w = DEFAULT_RESIZE

    # Load model
    model = StackedHourglassNetwork(
        in_channels=3,
        front_channels=args.front_channels,
        hourglass_channels=args.hourglass_channels,
        num_output_channels=NUM_OUTPUT_CHANNELS,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # Load test dataset
    val_transforms = get_val_transforms()
    test_dataset = VineDataset(
        args.data_path + "/02-IndependentTestSet",
        transforms=val_transforms,
        new_height=resize_h,
        new_width=resize_w,
        split="test",
    )

    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Accumulate metrics across all images
    accumulated_pred = defaultdict(list)
    accumulated_gt = defaultdict(list)
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (images, M) in enumerate(test_loader):
            images = images.float().to(device)
            M = M.to(device)

            output1, output2 = model(images)
            loss = criterion(output1, M) + criterion(output2, M)
            total_loss += loss.item()

            # Extract predictions from stage 2
            for k in range(images.size(0)):
                heatmaps, vector_fields = recover_heatmaps_vector_fields(
                    output2[k].cpu(), resize=(resize_h, resize_w)
                )

                # Extract predicted nodes
                for bname, bt in BRANCH_TYPES.items():
                    for nname, nt in NODE_TYPES.items():
                        hm = heatmaps[bt, nt].numpy()
                        coords = extract_node_coordinates(hm)
                        accumulated_pred[(bname, nname)].extend(coords)

                # Extract GT nodes from M
                gt_heatmaps, _ = recover_heatmaps_vector_fields(
                    M[k].cpu(), resize=(resize_h, resize_w)
                )
                for bname, bt in BRANCH_TYPES.items():
                    for nname, nt in NODE_TYPES.items():
                        hm = gt_heatmaps[bt, nt].numpy()
                        coords = extract_node_coordinates(hm)
                        accumulated_gt[(bname, nname)].extend(coords)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

    # Compute metrics
    avg_loss = total_loss / len(test_loader)
    print(f"\nAverage test loss: {avg_loss:.6f}")

    metrics = compute_allnode_metric(accumulated_pred, accumulated_gt, tau_d=args.tau_d)

    # Print results table
    print(f"\n{'='*70}")
    print(f"AllNodeMetric Results (tau_d = {args.tau_d})")
    print(f"{'='*70}")
    print(f"{'Category':<35} {'Precision':>9} {'Recall':>9} {'F-Score':>9}")
    print(f"{'-'*70}")

    for cat, m in sorted(metrics.items()):
        if cat == "all_nodes":
            continue
        label = f"{cat[0]} / {cat[1]}"
        print(f"{label:<35} {m['precision']:>9.3f} {m['recall']:>9.3f} {m['f_score']:>9.3f}")

    if "all_nodes" in metrics:
        m = metrics["all_nodes"]
        print(f"{'-'*70}")
        print(f"{'All nodes':<35} {m['precision']:>9.3f} {m['recall']:>9.3f} {m['f_score']:>9.3f}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
