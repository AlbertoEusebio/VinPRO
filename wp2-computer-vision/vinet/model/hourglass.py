"""
Stacked Hourglass Network (SHG) for grapevine structure estimation.

Architecture follows Newell et al. (2016) with modifications:
    - Instance normalization instead of batch normalization (batch size = 1)
    - Residual blocks with 1×1 → 3×3 → 1×1 convolutions
    - 5-level encoder-decoder with skip connections
    - 2 stacked hourglasses (refinement)
    - Gradient checkpointing for memory efficiency

Reference model (2HG-256): 13.2M parameters
    - Front features: 64 channels
    - Hourglass channels: K = 256
    - Output: 30 channels (20 heatmaps + 10 vector field components)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ── Building Blocks ───────────────────────────────────────────────────────────

class ResidualUnit(nn.Module):
    """
    Single residual unit: Conv(1×1) → IN → ReLU → Conv(3×3) → IN → ReLU → Conv(1×1) → IN + skip.

    Uses Instance Normalization instead of BatchNorm since ViNet trains with
    batch size 1 (Section 3.1 of the paper).
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(num_channels, affine=False)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.in2 = nn.InstanceNorm2d(num_channels, affine=False)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(num_channels, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        out = F.relu(self.in1(self.conv1(x)), inplace=True)
        out = F.relu(self.in2(self.conv2(out)), inplace=True)
        out = self.in3(self.conv3(out))
        out = F.relu(skip + out, inplace=True)
        return out


class TripleResidualBlock(nn.Module):
    """Block B_i composed of three consecutive residual units (R1, R2, R3)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.ru1 = ResidualUnit(num_channels)
        self.ru2 = ResidualUnit(num_channels)
        self.ru3 = ResidualUnit(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ru3(self.ru2(self.ru1(x)))


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Front-end feature extraction module.

    Reduces spatial resolution by 4× via a 7×7 strided convolution + maxpool,
    then refines features with a triple residual block and a 1×1 channel
    projection to match the hourglass channel dimension.

    Input:  (B, 3, H, W)
    Output: (B, hourglass_channels, H/4, W/4)
    """

    def __init__(
        self,
        in_channels: int = 3,
        front_channels: int = 64,
        hourglass_channels: int = 256,
    ):
        super().__init__()
        self.conv7 = nn.Conv2d(
            in_channels, front_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.in7 = nn.InstanceNorm2d(front_channels, affine=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.triple_res = TripleResidualBlock(front_channels)
        self.match_conv = nn.Conv2d(
            front_channels, hourglass_channels, kernel_size=1, bias=False
        )
        self.match_in = nn.InstanceNorm2d(hourglass_channels, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in7(self.conv7(x)), inplace=True)
        x = self.pool(x)
        x = self.triple_res(x)
        x = F.relu(self.match_in(self.match_conv(x)), inplace=True)
        return x


# ── Hourglass Module ──────────────────────────────────────────────────────────

class HourglassModule(nn.Module):
    """
    Single hourglass module with 5-level encoder-decoder.

    Down-path: B1 → pool → B2 → pool → B3 → pool → B4 → pool → B5
    Up-path:   upsample + S4 → upsample + S3 → upsample + S2 → upsample + S1 → B6

    Skip connections (S_i) add features from the down-path at matching resolutions.
    Uses gradient checkpointing to reduce memory during training.
    """

    def __init__(self, num_channels: int, depth: int = 5):
        super().__init__()
        # Down-path blocks
        self.down1 = TripleResidualBlock(num_channels)
        self.down2 = TripleResidualBlock(num_channels)
        self.down3 = TripleResidualBlock(num_channels)
        self.down4 = TripleResidualBlock(num_channels)
        self.down5 = TripleResidualBlock(num_channels)  # bottleneck

        # Skip blocks
        self.skip1 = TripleResidualBlock(num_channels)
        self.skip2 = TripleResidualBlock(num_channels)
        self.skip3 = TripleResidualBlock(num_channels)
        self.skip4 = TripleResidualBlock(num_channels)

        # Upsample layers
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Final aggregation block (B6)
        self.b6 = TripleResidualBlock(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (down-path)
        d1 = cp.checkpoint(self.down1, x, use_reentrant=False)
        out = F.max_pool2d(d1, 2, 2)

        d2 = cp.checkpoint(self.down2, out, use_reentrant=False)
        out = F.max_pool2d(d2, 2, 2)

        d3 = cp.checkpoint(self.down3, out, use_reentrant=False)
        out = F.max_pool2d(d3, 2, 2)

        d4 = cp.checkpoint(self.down4, out, use_reentrant=False)
        out = F.max_pool2d(d4, 2, 2)

        d5 = cp.checkpoint(self.down5, out, use_reentrant=False)

        # Decoder (up-path) with skip connections
        out = self.up4(d5) + cp.checkpoint(self.skip4, d4, use_reentrant=False)
        out = self.up3(out) + cp.checkpoint(self.skip3, d3, use_reentrant=False)
        out = self.up2(out) + cp.checkpoint(self.skip2, d2, use_reentrant=False)
        out = self.up1(out) + cp.checkpoint(self.skip1, d1, use_reentrant=False)

        out = cp.checkpoint(self.b6, out, use_reentrant=False)
        return out


# ── Stacked Hourglass Network ─────────────────────────────────────────────────

class StackedHourglassNetwork(nn.Module):
    """
    Two-stage Stacked Hourglass Network for ViNet.

    Stage 1 produces initial predictions, which are merged back with
    intermediate features and fed to Stage 2 for refinement (Section 2.3.1).

    Both stages are supervised during training (Eq. 3).

    Args:
        in_channels: Input image channels (default: 3 for RGB).
        front_channels: Channels in the feature extractor (default: 64).
        hourglass_channels: Channels throughout hourglass blocks (K, default: 256).
        num_output_channels: Number of output heatmap/vector field channels (default: 30).

    Returns:
        pred_stage1: (B, num_output_channels, H/4, W/4) — first stage predictions
        pred_stage2: (B, num_output_channels, H/4, W/4) — refined predictions
    """

    def __init__(
        self,
        in_channels: int = 3,
        front_channels: int = 64,
        hourglass_channels: int = 256,
        num_output_channels: int = 30,
    ):
        super().__init__()

        # Feature extractor (front module)
        self.feature_extractor = FeatureExtractor(
            in_channels=in_channels,
            front_channels=front_channels,
            hourglass_channels=hourglass_channels,
        )

        # Hourglass #1
        self.hg1 = HourglassModule(hourglass_channels)
        self.hg1_post = TripleResidualBlock(hourglass_channels)
        self.out_stage1 = nn.Conv2d(
            hourglass_channels, num_output_channels, kernel_size=1, bias=True
        )

        # Merge stage 1 → stage 2
        self.merge_feat1 = nn.Conv2d(
            hourglass_channels, hourglass_channels, kernel_size=1, bias=False
        )
        self.merge_pred1 = nn.Conv2d(
            num_output_channels, hourglass_channels, kernel_size=1, bias=False
        )

        # Hourglass #2
        self.hg2 = HourglassModule(hourglass_channels)
        self.hg2_post = TripleResidualBlock(hourglass_channels)
        self.out_stage2 = nn.Conv2d(
            hourglass_channels, num_output_channels, kernel_size=1, bias=True
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract features at 1/4 resolution
        features = self.feature_extractor(x)

        # Stage 1
        y1 = self.hg1(features)
        y1 = self.hg1_post(y1)
        pred_stage1 = self.out_stage1(y1)

        # Merge for stage 2
        feat1 = self.merge_feat1(y1) + self.merge_pred1(pred_stage1)

        # Stage 2 (refined)
        y2 = self.hg2(feat1)
        y2 = self.hg2_post(y2)
        pred_stage2 = self.out_stage2(y2)

        return pred_stage1, pred_stage2
