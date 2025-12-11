#!/usr/bin/env python3
"""
Find Optimal Threshold for Rosette Model
========================================

This script loads a trained model, runs it on the validation dataset,
and iterates through all possible thresholds (0.0 - 1.0) to find the
one that maximizes the F1-score.

It generates:
1. A Precision-Recall (PR) curve.
2. An F1-Score vs. Threshold plot.
3. The optimal threshold value printed to the console.
"""

import argparse
import json
from pathlib import Path

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from skimage.segmentation import find_boundaries
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---
# Copied from train_model.py to make this script self-contained
# ---

# ==============================================================================
# Data Augmentation (Validation only)
# ==============================================================================


def get_val_transforms(patch_size=512):
    """Define the pipeline for the validation set (only tensor conversion)."""
    return A.Compose([ToTensorV2()])


# ==============================================================================
# Dataset Class
# ==============================================================================


class RosetteDataset(Dataset):
    def __init__(
        self,
        h5_files,
        patch_size=512,
        mode="val",  # Hardcoded to 'val' for this script
        use_intensity=False,
        aug_config=None,
    ):
        self.patch_size = patch_size
        self.mode = mode
        self.use_intensity = use_intensity
        self.h5_files = [Path(f) for f in h5_files if Path(f).exists()]

        if not self.h5_files:
            print("Warning: No valid H5 files found from the provided list.")

        self.transform = get_val_transforms(patch_size)
        print(f"Initialized {self.mode} dataset with {len(self.h5_files)} files.")
        if use_intensity:
            print("Running in 3-channel (Geometric + Intensity) mode.")
        else:
            print("Running in 2-channel (Geometric-Only) mode.")

    def __len__(self):
        return len(self.h5_files)

    def _smart_crop(self, X, y):
        # For validation, we just take a center crop
        h, w = X.shape[1:]
        crop_size = self.patch_size

        if h < crop_size or w < crop_size:
            # Pad if image is smaller than patch size
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            X = np.pad(X, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            y = np.pad(y, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            h, w = X.shape[1:]

        top = (h - crop_size) // 2
        left = (w - crop_size) // 2

        X_crop = X[:, top : top + crop_size, left : left + crop_size]
        y_crop = y[:, top : top + crop_size, left : left + crop_size]

        return X_crop, y_crop

    def __getitem__(self, idx):
        try:
            with h5py.File(self.h5_files[idx], "r") as f:
                # This matches the h5 structure from your data prep script
                # Check for 'data' key first
                if "data" in f:
                    data_stack = f["data"][:]
                    cell_instance_mask = data_stack[0]
                    rosette_binary_mask = data_stack[1]
                    raw_image = data_stack[3]
                # Fallback to older keys if 'data' key doesn't exist
                elif "segmentation_outlines" in f:
                    cell_instance_mask = np.squeeze(f["segmentation_outlines"][:])
                    rosette_binary_mask = np.squeeze(f["rosettes_binary"][:])
                    raw_image = (
                        np.squeeze(f["raw_image"][:])
                        if "raw_image" in f
                        else np.zeros_like(cell_instance_mask)
                    )
                else:
                    raise KeyError(
                        f"Could not find 'data' or 'segmentation_outlines' in {self.h5_files[idx]}"
                    )

        except Exception as e:
            print(f"Error loading H5 file {self.h5_files[idx]}: {e}")
            # Return dummy data to prevent crash
            X = np.zeros(
                (3 if self.use_intensity else 2, self.patch_size, self.patch_size),
                dtype=np.float32,
            )
            y = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)
            y = y.transpose(1, 2, 0)  # (H, W, C)
            augmented = self.transform(image=X, mask=y)
            return augmented["image"], augmented["mask"].permute(2, 0, 1)

        y = (rosette_binary_mask > 0).astype(np.float32)[
            None, ...
        ]  # Ensure target is binary
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

        # Use center crop for validation
        if X.shape[1] > self.patch_size or X.shape[2] > self.patch_size:
            X, y = self._smart_crop(X, y)
        elif X.shape[1] < self.patch_size or X.shape[2] < self.patch_size:
            # Pad if image is smaller than patch size
            pad_h = max(0, self.patch_size - X.shape[1])
            pad_w = max(0, self.patch_size - X.shape[2])
            X = np.pad(X, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            y = np.pad(y, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

        X, y = X.transpose(1, 2, 0), y.transpose(1, 2, 0)
        augmented = self.transform(image=X, mask=y)
        X, y = augmented["image"], augmented["mask"]

        if y.ndim == 3:
            y = y.permute(2, 0, 1)
        elif y.ndim == 2:
            y = y.unsqueeze(0)
        return X, y


# ==============================================================================
# Model Architecture
# ==============================================================================


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
        self.enc1 = self._conv_block(n_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.bridge = self._conv_block(512, 1024, pool=False)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512, pool=False)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256, pool=False)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128, pool=False)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64, pool=False)
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
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bridge(e4)
        d4 = self.upconv4(b)
        e4_att = self.att4(g=d4, x=e4)
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
# Core Script Functions
# ==============================================================================


#
# --- THIS IS THE NEW FUNCTION I ADDED ---
#
def load_samples_from_manifest(manifest_path, data_dir):
    """Load train/val samples from a JSON split manifest."""
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        print(f"Error: Split manifest not found at {manifest_path}")
        return [], []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {manifest_path}")
        return [], []

    train_samples = manifest.get("train_sample_list", [])
    val_samples = manifest.get("val_sample_list", [])

    # Check if data_dir is correct
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_dir():
        print(f"Error: Data directory not found at {data_dir}")
        return [], []

    # Construct file paths and check for existence
    def get_files(samples):
        file_list = []
        for s in samples:
            # Check for both old and new H5 naming conventions
            old_path = data_dir_path / s / f"processed_data_{s}.h5"
            new_path = (
                data_dir_path / s / f"{s}_processed_data.h5"
            )  # From your data_prep script

            if old_path.exists():
                file_list.append(str(old_path))
            elif new_path.exists():
                file_list.append(str(new_path))
            else:
                print(
                    f"Warning: Could not find H5 file for sample {s} at {old_path} or {new_path}"
                )
        return file_list

    train_files = get_files(train_samples)
    val_files = get_files(val_samples)

    if not val_files:
        print("Warning: No validation files were found. Check manifest and data_dir.")

    return train_files, val_files


#
# --- END OF NEW FUNCTION ---
#


def load_model_and_config(model_path, config_path):
    """Loads the model and its configuration."""
    with open(config_path, "r") as f:
        config = json.load(f)

    use_intensity = config.get("use_intensity", False)
    n_channels = 3 if use_intensity else 2

    model = AttentionUNet(n_channels=n_channels)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"Warning: could not load state dict directly ({e}).")
        print("This is common if the model was saved as DataParallel.")
        print("Trying to load by stripping 'module.' prefix...")
        from collections import OrderedDict

        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(
            new_state_dict, strict=False
        )  # Use strict=False for flexibility

    model.to(device)
    model.eval()

    print(f"Model loaded onto {device}.")
    return model, config, device


def get_all_predictions(model, val_loader, device):
    """Run model on all validation data and return flattened predictions/gts."""
    all_preds = []
    all_gts = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Validating"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_preds.append(pred.cpu())
            all_gts.append(y.cpu())

    # Concatenate all batches
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_gts_tensor = torch.cat(all_gts, dim=0)

    # Flatten for metric calculation
    preds_flat = all_preds_tensor.flatten()
    gts_flat = all_gts_tensor.flatten()

    print(f"Collected {len(preds_flat)} total pixels for analysis.")
    return preds_flat, gts_flat


def calculate_metrics_at_thresholds(preds_flat, gts_flat, num_steps=100):
    """Iterate through thresholds and calculate P, R, and F1."""
    thresholds = np.linspace(0.01, 0.99, num_steps)
    results = []

    # Ensure gts is float for calculations
    gts_flat = gts_flat.float()

    print("Calculating metrics for each threshold...")
    for t in tqdm(thresholds, desc="Optimizing Threshold"):
        pred_binary = (preds_flat > t).float()

        tp = (pred_binary * gts_flat).sum()
        fp = (pred_binary * (1 - gts_flat)).sum()
        fn = ((1 - pred_binary) * gts_flat).sum()

        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)

        results.append(
            {
                "threshold": t,
                "precision": precision.item(),
                "recall": recall.item(),
                "f1": f1.item(),
            }
        )

    return pd.DataFrame(results)


def plot_results(df, best, output_dir):
    """Plot PR curve and F1 vs. Threshold."""
    print("Generating plots...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    plt.plot(df["recall"], df["precision"], marker=".", label="Precision-Recall")
    plt.scatter(
        best["recall"],
        best["precision"],
        marker="*",
        color="red",
        s=200,
        label=f"Best F1: {best['f1']:.3f}\n@ Thresh: {best['threshold']:.2f}",
        zorder=10,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    print(f"Saved PR curve to {output_dir / 'precision_recall_curve.png'}")

    # Plot 2: F1-Score vs. Threshold
    plt.figure(figsize=(10, 8))
    plt.plot(df["threshold"], df["f1"], label="F1-Score")
    plt.axvline(
        best["threshold"],
        color="red",
        linestyle="--",
        label=f"Optimal Threshold: {best['threshold']:.2f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title("F1-Score vs. Prediction Threshold")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(output_dir / "f1_vs_threshold.png", dpi=150)
    print(f"Saved F1 vs. Threshold plot to {output_dir / 'f1_vs_threshold.png'}")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal threshold for a trained rosette model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained 'best_model.pth' file.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the 'config.json' file from the same training run.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        required=True,
        help="Path to the 'split_manifest.json' file used for training.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the root directory containing the H5 training data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the output plots.",
    )
    args = parser.parse_args()

    # 1. Load Model and Config
    model, config, device = load_model_and_config(args.model_path, args.config_path)

    # 2. Load Validation Data
    _, val_files = load_samples_from_manifest(args.split_manifest, args.data_dir)
    if not val_files:
        print("Error: No validation files found. Check paths and manifest.")
        return

    val_dataset = RosetteDataset(
        h5_files=val_files,
        patch_size=config.get("patch_size", 512),
        mode="val",
        use_intensity=config.get("use_intensity", False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),  # Use training batch size
        shuffle=False,
        num_workers=max(1, config.get("num_workers", 2)),  # Use fewer workers
    )

    if len(val_loader) == 0:
        print(
            "Error: Validation DataLoader is empty, though files were found. Check H5 file contents or paths."
        )
        return

    # 3. Get Predictions
    preds_flat, gts_flat = get_all_predictions(model, val_loader, device)

    # 4. Calculate Metrics
    metrics_df = calculate_metrics_at_thresholds(preds_flat, gts_flat, num_steps=100)

    # Save metrics CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "threshold_metrics.csv", index=False)
    print(f"Saved all metrics to {output_dir / 'threshold_metrics.csv'}")

    # 5. Find Best Threshold
    best_f1_row = metrics_df.loc[metrics_df["f1"].idxmax()]

    print("\n" + "=" * 50)
    print("Optimal Threshold Found!")
    print(f"  Threshold: {best_f1_row['threshold']:.3f}")
    print(f"  F1-Score:  {best_f1_row['f1']:.4f}")
    print(f"  Precision: {best_f1_row['precision']:.4f}")
    print(f"  Recall:    {best_f1_row['recall']:.4f}")
    print("=" * 50)

    # 6. Plot Results
    plot_results(metrics_df, best_f1_row, output_dir)


if __name__ == "__main__":
    main()


### How to Run (Same as before)

# Now that the script is fixed, you can run the *exact same command* you ran last time.

# 1.  **Change Directory:**
#     ```bash
#     cd modules/model_training
#     ```

# 2.  **Run the script:**
#     ```bash
# poetry run python scripts/optimize_threshold.py \
#     --model-path ../../results_old/model_training_old/champion_hard_augmentation/best_model.pth \
#     --config-path ../../results_old/model_training_worked/champion_hard_augmentation/config.json \
#     --split-manifest ../../results_old/model_training_worked/split_manifest_champion_hard_augmentation.json \
#     --data-dir ../../results/training_data_preparation \
#     --output-dir ../../results_old/model_training_old/champion_hard_augmentation/threshold_analysis
