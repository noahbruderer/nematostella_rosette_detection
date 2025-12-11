#!/usr/bin/env python3
"""
Train Rosette Detection Model
=============================

This script trains an Attention U-Net model for rosette detection using
pre-processed H5 files. It is a professional-grade training script featuring:
- Dynamic configuration from YAML files.
- Dynamic data augmentation pipelines.
- Early stopping to prevent overfitting and save time.
- Rich, real-time logging with TensorBoard, including image predictions.
"""

# 1. First, import all necessary modules
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import ML/DL and data handling libraries
import albumentations as A
import h5py
import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import yaml
from albumentations.pytorch import ToTensorV2
from skimage.segmentation import find_boundaries
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# ==============================================================================
# Helper Functions
# ==============================================================================


def load_samples_from_manifest(manifest_path, data_dir):
    """Load train/val samples from a JSON split manifest."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    train_samples = manifest.get("train_sample_list", [])
    val_samples = manifest.get("val_sample_list", [])
    train_files = [
        str(Path(data_dir) / s / f"processed_data_{s}.h5") for s in train_samples
    ]
    val_files = [
        str(Path(data_dir) / s / f"processed_data_{s}.h5") for s in val_samples
    ]
    return train_files, val_files


def save_training_plots(history, approach_name, save_dir):
    """Save plots of training and validation metrics."""
    # (This function remains unchanged)
    save_dir = Path(save_dir)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    (
        plt.xlabel("Epoch"),
        plt.ylabel("Loss"),
        plt.title("Loss"),
        plt.legend(),
        plt.grid(True),
    )
    plt.subplot(1, 3, 2)
    plt.plot(history.get("val_iou", []), label="Validation IoU")
    (
        plt.xlabel("Epoch"),
        plt.ylabel("IoU"),
        plt.title("IoU"),
        plt.legend(),
        plt.grid(True),
    )
    plt.subplot(1, 3, 3)
    plt.plot(history.get("val_f1", []), label="Validation F1")
    (
        plt.xlabel("Epoch"),
        plt.ylabel("F1 Score"),
        plt.title("F1 Score"),
        plt.legend(),
        plt.grid(True),
    )
    plt.suptitle(
        f"Training Metrics for {approach_name.replace('_', ' ').title()}", fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = save_dir / "training_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
# Data Augmentation Pipelines (Now Dynamic)
# ==============================================================================


def get_train_transforms(patch_size=512, aug_config=None):
    """Define a robust augmentation pipeline driven by a config dictionary."""
    if aug_config is None:
        aug_config = {}

    transforms = [
        A.HorizontalFlip(p=aug_config.get("horizontal_flip", 0.5)),
        A.VerticalFlip(p=aug_config.get("vertical_flip", 0.5)),
        A.RandomRotate90(p=aug_config.get("rotation", 0.5)),
        A.ElasticTransform(
            p=aug_config.get("elastic_transform", 0.25),
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
        ),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            p=aug_config.get("affine_transform", 0.5),
        ),
        A.CoarseDropout(
            max_objects=5,
            max_height=40,
            max_width=40,
            min_height=10,
            min_width=10,
            p=aug_config.get("coarse_dropout", 0.2),
        ),
        ToTensorV2(),
    ]
    # Filter out transforms with probability 0
    active_transforms = [t for t in transforms if not (hasattr(t, "p") and t.p == 0)]
    print(f"Active augmentations: {[t.__class__.__name__ for t in active_transforms]}")
    return A.Compose(active_transforms)


def get_val_transforms(patch_size=512):
    """Validation pipeline is static."""
    return A.Compose([ToTensorV2()])


# ==============================================================================
# Dataset Class (Now Accepts Augmentation Config)
# ==============================================================================


class RosetteDataset(Dataset):
    def __init__(
        self,
        h5_files,
        patch_size=512,
        mode="train",
        use_intensity=False,
        aug_config=None,  # Accept aug_config
    ):
        self.patch_size = patch_size
        self.mode = mode
        self.use_intensity = use_intensity
        self.h5_files = [Path(f) for f in h5_files]

        if self.mode == "train":
            self.transform = get_train_transforms(patch_size, aug_config)
        else:
            self.transform = get_val_transforms(patch_size)

        print(f"Initialized {self.mode} dataset with {len(self.h5_files)} files.")

    def __len__(self):
        return len(self.h5_files)

    def _smart_crop(self, X, y):
        h, w = X.shape[1:]
        crop_size = self.patch_size
        rosette_positions = np.argwhere(y[0] > 0)
        do_random_crop = (len(rosette_positions) == 0) or (np.random.random() < 0.3)

        if do_random_crop:
            top = np.random.randint(0, max(1, h - crop_size))
            left = np.random.randint(0, max(1, w - crop_size))
        else:
            center_idx = np.random.randint(0, len(rosette_positions))
            center_y, center_x = rosette_positions[center_idx]
            top = max(0, min(center_y - crop_size // 2, h - crop_size))
            left = max(0, min(center_x - crop_size // 2, w - crop_size))

        X_crop = X[:, top : top + crop_size, left : left + crop_size]
        y_crop = y[:, top : top + crop_size, left : left + crop_size]

        pad_h = max(0, crop_size - X_crop.shape[1])
        pad_w = max(0, crop_size - X_crop.shape[2])
        if pad_h > 0 or pad_w > 0:
            X_crop = np.pad(X_crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            y_crop = np.pad(y_crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        return X_crop, y_crop

    def __getitem__(self, idx):
        # (This method remains largely unchanged)
        with h5py.File(self.h5_files[idx], "r") as f:
            cell_instance_mask = np.squeeze(f["segmentation_outlines"][:])
            rosette_instance_mask = np.squeeze(f["rosettes_binary"][:])
            raw_image = (
                np.squeeze(f["raw_image"][:])
                if "raw_image" in f
                else np.zeros_like(cell_instance_mask)
            )

        y = (rosette_instance_mask > 0).astype(np.float32)[None, ...]
        cell_boundaries = find_boundaries(cell_instance_mask, mode="thick").astype(
            np.float32
        )
        cell_mask_binary = (cell_instance_mask > 0).astype(np.float32)

        if self.use_intensity:
            raw_image_norm = raw_image.astype(np.float32)
            if raw_image_norm.max() > 1.0:
                raw_image_norm /= raw_image_norm.max() + 1e-8
            X = np.stack([cell_boundaries, cell_mask_binary, raw_image_norm], axis=0)
        else:
            X = np.stack([cell_boundaries, cell_mask_binary], axis=0)

        if X.shape[1] > self.patch_size or X.shape[2] > self.patch_size:
            X, y = self._smart_crop(X, y)

        X, y = X.transpose(1, 2, 0), y.transpose(1, 2, 0)
        augmented = self.transform(image=X, mask=y)
        X, y = augmented["image"], augmented["mask"]

        if y.ndim == 3:
            y = y.permute(2, 0, 1)
        elif y.ndim == 2:
            y = y.unsqueeze(0)
        return X, y


# ==============================================================================
# Attention UNet Model (Unchanged)
# ==============================================================================
# The model architecture (AttentionGate, AttentionUNet) is solid and remains the same.
# For brevity, it's omitted here. Please copy it from your previous script.
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, n_channels=2):
        super(AttentionUNet, self).__init__()
        # Encoder
        self.enc1 = self._conv_block(n_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        # Bridge
        self.bridge = self._conv_block(512, 1024, pool=False)
        # Attention Gates
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512, pool=False)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256, pool=False)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128, pool=False)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64, pool=False)
        # Final layer
        self.final = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())

    def _conv_block(self, in_ch, out_ch, pool=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        # --- THIS IS THE FULLY CORRECTED METHOD ---
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bridge
        b = self.bridge(e4)

        # Decoder Path with Attention and RESIZING
        d4 = self.upconv4(b)
        e4_att = self.att4(g=d4, x=e4)
        # --- The F.interpolate call is critical to match the sizes ---
        if e4_att.shape[2:] != d4.shape[2:]:
            e4_att = F.interpolate(
                e4_att, size=d4.shape[2:], mode="bilinear", align_corners=False
            )
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        e3_att = self.att3(g=d3, x=e3)
        if e3_att.shape[2:] != d3.shape[2:]:
            e3_att = F.interpolate(
                e3_att, size=d3.shape[2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        e2_att = self.att2(g=d2, x=e2)
        if e2_att.shape[2:] != d2.shape[2:]:
            e2_att = F.interpolate(
                e2_att, size=d2.shape[2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        e1_att = self.att1(g=d1, x=e1)
        if e1_att.shape[2:] != d1.shape[2:]:
            e1_att = F.interpolate(
                e1_att, size=d1.shape[2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ==============================================================================
# Loss, Metrics, and Training Loop (Upgraded)
# ==============================================================================


class CombinedLoss(nn.Module):
    # (This class remains unchanged)
    def __init__(self, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight, self.bce = dice_weight, nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean()
        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(pred, target, threshold=0.5):
    # (This function remains unchanged)
    pred_b = (pred > threshold).float()
    tp = (pred_b * target).sum((2, 3))
    fp = (pred_b * (1 - target)).sum((2, 3))
    fn = ((1 - pred_b) * target).sum((2, 3))
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return {"iou": iou.mean().item(), "f1": f1.mean().item()}


def log_validation_images(writer, model, val_loader, epoch, device):
    """Logs a batch of validation images, masks, and predictions to TensorBoard."""
    model.eval()
    with torch.no_grad():
        # Get one batch of data
        try:
            X, y = next(iter(val_loader))
            X, y = X.to(device), y.to(device)

            # Get model prediction
            pred = model(X)

            # We use the geometric channels for visualization
            # Input channel 0 is boundaries, channel 1 is cell mask
            input_image_geom = X[:, :1, :, :].repeat(
                1, 3, 1, 1
            )  # Repeat channel to make it RGB

            # Create a 3-channel overlay: GT=Green, Pred=Red. Yellow=Overlap
            overlay = torch.cat(
                [
                    pred,  # Red channel for prediction
                    y,  # Green channel for ground truth
                    torch.zeros_like(pred),  # Blue channel is empty
                ],
                dim=1,
            )

            # Combine into a grid
            grid = torchvision.utils.make_grid(
                torch.cat([input_image_geom, overlay], dim=0), nrow=X.size(0)
            )
            writer.add_image("Validation/Predictions_vs_GroundTruth", grid, epoch)
        except StopIteration:
            print("Validation loader is empty, skipping image logging.")
    model.train()  # Set model back to training mode


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    lr,
    save_dir,
    approach_name,
    patience,  # Added for early stopping
    create_timestamped_dir,
):
    """The main training loop with early stopping and TensorBoard logging."""
    save_dir = Path(save_dir)
    if create_timestamped_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir / f"run_{approach_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize TensorBoard Writer ---
    writer = SummaryWriter(log_dir=str(save_dir))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    print(f"Using device: {device}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = CombinedLoss(dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # Switched to AdamW
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=patience // 2
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss, val_metrics = 0.0, {"iou": 0.0, "f1": 0.0}
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                val_loss += criterion(output, y).item()
                metrics = calculate_metrics(output, y)
                val_metrics["iou"] += metrics["iou"]
                val_metrics["f1"] += metrics["f1"]
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_metrics["iou"] / len(val_loader)
        avg_val_f1 = val_metrics["f1"] / len(val_loader)
        history.update(
            {
                "val_loss": [avg_val_loss],
                "val_iou": [avg_val_iou],
                "val_f1": [avg_val_f1],
            }
        )

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val F1: {avg_val_f1:.4f}"
        )

        # --- TensorBoard Logging ---
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        writer.add_scalar("IoU/validation", avg_val_iou, epoch)
        writer.add_scalar("F1_Score/validation", avg_val_f1, epoch)
        writer.add_scalar("Misc/learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if epoch % 5 == 0:  # Log images every 5 epochs to save space and time
            log_validation_images(writer, model, val_loader, epoch, device)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print("  -> New best model saved.")
            patience_counter = 0
        else:
            patience_counter += 1

        save_training_plots(history, approach_name, save_dir)

        if patience_counter >= patience:
            print(
                f"\nEarly stopping triggered after {patience} epochs with no improvement."
            )
            break

    writer.close()  # Close the TensorBoard writer
    print(f"\nTraining finished. Best model and logs saved in {save_dir}")
    return history, save_dir


# ==============================================================================
# Main Execution Block
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train geometric rosette detection model using Attention UNet"
    )
    parser.add_argument(
        "--split-manifest", help="Path to a JSON file defining train/val splits"
    )
    parser.add_argument("--data-dir", help="Directory where H5 files are located")
    parser.add_argument("--output-dir", default="./training_outputs")
    parser.add_argument("--config", help="Path to a YAML configuration file")
    parser.add_argument(
        "--experiment-name", help="Name of the experiment to run from the config file"
    )
    parser.add_argument(
        "--log-file", help="Path to a completion log file for Snakemake"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't create a timestamped output subdirectory",
    )
    args = parser.parse_args()

    # --- Load configuration from YAML ---
    if not (args.config and args.experiment_name):
        raise ValueError("Both --config and --experiment-name must be provided.")

    print(
        f"Loading experiment '{args.experiment_name}' from config file: {args.config}"
    )
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    defaults = config.get("default_training", {})
    exp_list = config.get("experiments", [])
    exp_config = next(
        (exp for exp in exp_list if exp["name"] == args.experiment_name), None
    )
    if exp_config is None:
        raise ValueError(
            f"Experiment '{args.experiment_name}' not found in {args.config}"
        )

    # Combine defaults and experiment-specific settings
    params = {**defaults, **exp_config}

    np.random.seed(params.get("seed", 42))
    torch.manual_seed(params.get("seed", 42))

    # --- Determine Train/Validation File Lists ---
    print(f"Loading train/val splits from manifest: {args.split_manifest}")
    train_files, val_files = load_samples_from_manifest(
        args.split_manifest, args.data_dir
    )
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    # --- Create Datasets and DataLoaders ---
    use_intensity = params.get("use_intensity", False)
    aug_config = params.get("augmentation", {})

    train_dataset = RosetteDataset(
        h5_files=train_files,
        patch_size=params.get("patch_size", 512),
        mode="train",
        use_intensity=use_intensity,
        aug_config=aug_config,
    )
    val_dataset = RosetteDataset(
        h5_files=val_files,
        patch_size=params.get("patch_size", 512),
        mode="val",
        use_intensity=use_intensity,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.get("batch_size", 4),
        shuffle=True,
        num_workers=params.get("num_workers", 4),
        pin_memory=params.get("pin_memory", True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.get("batch_size", 4),
        shuffle=False,
        num_workers=params.get("num_workers", 4),
        pin_memory=params.get("pin_memory", True),
    )

    # --- Initialize and Train Model ---
    n_channels = 3 if use_intensity else 2
    model = AttentionUNet(n_channels=n_channels)
    approach_name = "attention_intensity" if use_intensity else "attention_geometric"
    print(
        f"\nStarting training for '{approach_name}' approach with {n_channels} input channels."
    )

    history, save_dir = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=params.get("num_epochs", 100),
        lr=params.get("learning_rate", 1e-3),
        save_dir=args.output_dir,
        approach_name=approach_name,
        patience=params.get("patience", 15),
        create_timestamped_dir=not args.no_timestamp,
    )

    # --- Finalize ---
    final_config = {
        **params,
        "approach": approach_name,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    with open(Path(save_dir) / "config.json", "w") as f:
        json.dump(final_config, f, indent=4)

    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_file, "w") as f:
            f.write(
                f"Training completed for {approach_name}.\nModel saved to: {save_dir}\n"
            )

    print(f"\n✅ Training complete. Outputs are in: {save_dir}")


if __name__ == "__main__":
    main()
