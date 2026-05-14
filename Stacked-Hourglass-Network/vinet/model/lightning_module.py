"""
PyTorch Lightning module wrapping the Stacked Hourglass Network.

Implements the two-stage loss from Eq. 3:
    L = L_1 + L_2
where L_s is the MSE loss between predicted and ground-truth heatmaps
at hourglass stage s.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from ..config import DEFAULT_LR


class HourglassLightningModule(pl.LightningModule):
    """
    Lightning wrapper for training and evaluating the ViNet model.

    Args:
        model: A StackedHourglassNetwork instance.
        lr: Learning rate for Adam optimizer.
        loss_fn: Loss function (default: MSELoss as in the paper).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = DEFAULT_LR,
        loss_fn: nn.Module = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        # Paper Eq. 3 sums over all pixels and channels, so use 'sum' reduction.
        self.criterion = loss_fn or nn.MSELoss(reduction="sum")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, M1, M2 = batch
        out1, out2 = self.forward(images.float())
        loss1 = self.criterion(out1, M1)
        loss2 = self.criterion(out2, M2)
        loss = loss1 + loss2

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_loss_stage1", loss1, on_step=False, on_epoch=True)
        self.log(f"{stage}_loss_stage2", loss2, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """Adam optimizer with StepLR: decay 0.9× every 5000 iterations (Section 3.1)."""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
