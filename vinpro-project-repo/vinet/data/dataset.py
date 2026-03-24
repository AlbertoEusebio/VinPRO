"""
PyTorch Dataset for the 3D2cut Single Guyot grapevine dataset.

Each sample returns:
    - image: Tensor of shape (3, H, W)
    - M: Stacked ground-truth tensor of shape (N_h, H_map, W_map)
         containing node heatmaps and branch vector fields.
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import (
    NODE_TYPES,
    BRANCH_TYPES,
    NUM_NODE_TYPES,
    NUM_BRANCH_TYPES,
    DEFAULT_SIGMA_HEATMAP,
    DEFAULT_LIMB_WIDTH,
)
from .encoding import (
    load_annotation,
    parse_features,
    convert_nodes,
    generate_node_heatmaps,
    get_vector_fields,
)


class VineDataset(Dataset):
    """
    Custom dataset for vine images and annotations from the 3D2cut dataset.

    The dataset folder is expected to contain image files (.jpg/.jpeg) and
    corresponding annotation files (*_annotation.json).

    Args:
        data_dir: Path to the dataset directory.
        transforms: Albumentations transform pipeline.
        new_height: Target heatmap/vector field height.
        new_width: Target heatmap/vector field width.
        split: One of 'train', 'val', or 'test'.
        sigma: Gaussian spread for node heatmaps.
        limb_width: Width of vector field around branch segments.
    """

    def __init__(
        self,
        data_dir: str,
        transforms=None,
        new_height: int = 256,
        new_width: int = 256,
        split: str = "train",
        sigma: float = DEFAULT_SIGMA_HEATMAP,
        limb_width: float = DEFAULT_LIMB_WIDTH,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.new_height = new_height
        self.new_width = new_width
        self.sigma = sigma
        self.limb_width = limb_width
        self.node_types = NODE_TYPES
        self.branch_types = BRANCH_TYPES

        # Filter image files by split
        all_images = [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".jpg") or f.endswith(".jpeg")
        ]

        if split == "train":
            self.image_files = [
                f for f in all_images if any(f"Set0{x}" in f for x in range(0, 6))
            ]
        elif split == "val":
            self.image_files = [
                f for f in all_images if any(f"Set0{x}" in f for x in range(6, 7))
            ]
        elif split == "test":
            self.image_files = all_images
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_file = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        annotation_path = os.path.join(
            self.data_dir,
            img_file.replace(".jpg", "_annotation.json").replace(
                ".jpeg", "_annotation.json"
            ),
        )

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and parse annotation
        annotation = load_annotation(annotation_path)
        nodes, branches = parse_features(annotation)

        # Prepare keypoints for augmentation
        keypoints = np.array(
            [[node["coordinates"][0], node["coordinates"][1]] for node in nodes]
        )

        # Apply augmentations
        if self.transforms:
            blob = self.transforms(image=image, keypoints=keypoints)
            image = blob["image"]
            keypoints = blob["keypoints"]

        real_height, real_width, _ = image.shape

        # Update node coordinates after augmentation
        for i, node in enumerate(nodes):
            node["coordinates"] = keypoints[i]

        # Convert image to tensor (C, H, W)
        image_tensor = torch.tensor(image).permute(2, 0, 1)

        # Generate vector fields for all branch types
        vector_fields = get_vector_fields(
            (real_height, real_width),
            (self.new_height, self.new_width),
            nodes,
            branches,
            self.limb_width,
        )

        # Generate node heatmaps
        new_nodes = convert_nodes(nodes, branches, self.branch_types, self.node_types)
        heatmaps = generate_node_heatmaps(
            (real_height, real_width),
            new_nodes,
            self.sigma,
            NUM_NODE_TYPES,
            NUM_BRANCH_TYPES,
            (self.new_height, self.new_width),
        )

        # Stack vector fields: (N_branch_types, 2, H, W) → (2*N_bt, H, W)
        vf_array = np.array(
            [vf.transpose(2, 0, 1) for vf in vector_fields.values()]
        )
        vf_tensor = torch.tensor(vf_array).view(
            2 * NUM_BRANCH_TYPES, self.new_height, self.new_width
        )

        # Stack heatmaps: (N_bt, N_nt, H, W) → (N_bt*N_nt, H, W)
        hm_tensor = torch.tensor(heatmaps).view(
            NUM_BRANCH_TYPES * NUM_NODE_TYPES, self.new_height, self.new_width
        )

        # Concatenate: [heatmaps | vector_fields]
        M = torch.vstack([hm_tensor, vf_tensor])

        return image_tensor, M
