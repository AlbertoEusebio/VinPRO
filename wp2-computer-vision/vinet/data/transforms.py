"""
Data augmentation pipelines for ViNet training and evaluation.

Training augmentations follow the paper:
    - Random rescale to [1100, 1300] on smallest dimension
    - Random crop to 1024×1024
    - Random rotation in [-15°, 15°]
    - Random horizontal flip with p=0.5
"""

import cv2
import albumentations as A


def get_train_transforms(crop_size: int = 1024) -> A.Compose:
    """Training augmentation pipeline with geometric transforms."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=1300, interpolation=cv2.INTER_CUBIC),
            A.LongestMaxSize(max_size=2000, interpolation=cv2.INTER_CUBIC),
            A.RandomCrop(width=crop_size, height=crop_size),
            A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_transforms(crop_size: int = 1024) -> A.Compose:
    """Validation/test transform: deterministic resize only."""
    return A.Compose(
        [
            A.Resize(width=crop_size, height=crop_size, interpolation=cv2.INTER_CUBIC),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
