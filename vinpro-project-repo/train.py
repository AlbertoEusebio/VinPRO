"""
Training script for ViNet.

Usage:
    python train.py --data_path /path/to/3D2cut_Single_Guyot/ \
                    --max_epochs 300 --batch_size 1 --lr 1e-3
"""

import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from vinet.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_RESIZE,
    DEFAULT_FRONT_CHANNELS,
    DEFAULT_HOURGLASS_CHANNELS,
    NUM_OUTPUT_CHANNELS,
)
from vinet.data import VineDataset, get_train_transforms, get_val_transforms
from vinet.model import StackedHourglassNetwork, HourglassLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViNet on the 3D2cut dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Root path to 3D2cut dataset")
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--front_channels", type=int, default=DEFAULT_FRONT_CHANNELS)
    parser.add_argument("--hourglass_channels", type=int, default=DEFAULT_HOURGLASS_CHANNELS)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1 if torch.cuda.is_available() else 0)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    resize_h, resize_w = DEFAULT_RESIZE

    # Data
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_dataset = VineDataset(
        args.data_path + "/01-TrainAndValidationSet",
        transforms=train_transforms,
        new_height=resize_h, new_width=resize_w,
        split="train",
    )
    val_dataset = VineDataset(
        args.data_path + "/01-TrainAndValidationSet",
        transforms=val_transforms,
        new_height=resize_h, new_width=resize_w,
        split="val",
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    num_workers = args.num_workers or (18 if torch.cuda.is_available() else 0)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers,
    )

    # Model
    model = StackedHourglassNetwork(
        in_channels=3,
        front_channels=args.front_channels,
        hourglass_channels=args.hourglass_channels,
        num_output_channels=NUM_OUTPUT_CHANNELS,
    )
    lightning_model = HourglassLightningModule(model, lr=args.lr)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="vinet-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=20, mode="min")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        callbacks=[checkpoint_callback, early_stop],
    )

    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=args.checkpoint)

    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")

    # Save final model weights
    torch.save(model.state_dict(), "vinet_final.pt")
    print("Saved final model weights to vinet_final.pt")


if __name__ == "__main__":
    main()
