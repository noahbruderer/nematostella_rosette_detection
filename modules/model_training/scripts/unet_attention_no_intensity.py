# 1. First, import all necessary modules
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import the augmentation library
import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from skimage.segmentation import find_boundaries
from torch.utils.data import DataLoader, Dataset

# ==============================================================================
# Data Augmentation Pipelines (Same as before)
# ==============================================================================


def get_train_transforms(patch_size=512):
    """Define a robust GEOMETRIC augmentation pipeline for segmentation maps."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(
                p=0.25, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
            ),
            A.Affine(
                scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5
            ),
            A.CoarseDropout(
                max_objects=5,
                max_height=40,
                max_width=40,
                min_height=10,
                min_width=10,
                p=0.2,
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms(patch_size=512):
    """Define the pipeline for the validation set (only tensor conversion)."""
    return A.Compose([ToTensorV2()])


# ==============================================================================
# Dataset Class (Same as geometric-only version)
# ==============================================================================


class RosetteDataset(Dataset):
    def __init__(self, data_dir, patch_size=512, mode="train", use_intensity=False):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.mode = mode
        self.use_intensity = use_intensity

        if self.mode == "train":
            self.transform = get_train_transforms(patch_size)
            print("Using training augmentations.")
        else:
            self.transform = get_val_transforms(patch_size)
            print("Using validation transforms (no augmentation).")

        self.h5_files = list(self.data_dir.glob("**/*_processed_data.h5"))
        print(f"Found {len(self.h5_files)} H5 files")
        self.sample_groups = self._group_by_sample()

    def _group_by_sample(self):
        """Group files by sample ID to ensure proper train/val split."""
        sample_groups = {}
        for file in self.h5_files:
            sample_id = file.parent.name
            if sample_id not in sample_groups:
                sample_groups[sample_id] = []
            sample_groups[sample_id].append(file)
        return sample_groups

    def __len__(self):
        return len(self.h5_files)

    def _smart_crop(self, X, y):
        """Improved cropping strategy with balanced sampling."""
        h, w = X.shape[1:]
        crop_size = self.patch_size
        rosette_mask = y[0]
        rosette_positions = np.argwhere(rosette_mask > 0)
        do_random_crop = (len(rosette_positions) == 0) or (np.random.random() < 0.3)

        if do_random_crop:
            top = np.random.randint(0, max(1, h - crop_size + 1))
            left = np.random.randint(0, max(1, w - crop_size + 1))
        else:
            center_idx = np.random.randint(0, len(rosette_positions))
            center_y, center_x = rosette_positions[center_idx]
            top = max(0, min(center_y - crop_size // 2, h - crop_size))
            left = max(0, min(center_x - crop_size // 2, w - crop_size))

        X_crop = X[:, top : top + crop_size, left : left + crop_size]
        y_crop = y[:, top : top + crop_size, left : left + crop_size]

        if X_crop.shape[1:] != (crop_size, crop_size):
            pad_h = max(0, crop_size - X_crop.shape[1])
            pad_w = max(0, crop_size - X_crop.shape[2])
            X_crop = np.pad(X_crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            y_crop = np.pad(y_crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        return X_crop, y_crop

    def __getitem__(self, idx):
        h5_path = self.h5_files[idx]

        with h5py.File(h5_path, "r") as f:
            data_stack = f["data"][:]

            cell_instance_mask = data_stack[0]
            rosette_binary_mask = data_stack[1]  # Target
            rosette_instance_mask = data_stack[2]
            raw_image = data_stack[3]

        # Create input features based on use_intensity flag
        cell_boundaries = find_boundaries(cell_instance_mask, mode="thick").astype(
            np.float32
        )
        cell_mask_binary = (cell_instance_mask > 0).astype(np.float32)

        if self.use_intensity:
            # 3-channel version with raw image
            raw_image_norm = raw_image.astype(np.float32)
            if raw_image_norm.max() > 1.0:
                raw_image_norm = raw_image_norm / raw_image_norm.max()

            X = np.stack(
                [
                    cell_boundaries,
                    cell_mask_binary,
                    raw_image_norm,
                ],
                axis=0,
            )
        else:
            # 2-channel version (geometric only)
            X = np.stack(
                [
                    cell_boundaries,
                    cell_mask_binary,
                ],
                axis=0,
            )

        y = rosette_binary_mask.astype(np.float32)[None, ...]

        if X.shape[1] > self.patch_size:
            X, y = self._smart_crop(X, y)

        X = X.transpose(1, 2, 0)
        y = y.transpose(1, 2, 0)

        if self.transform:
            augmented = self.transform(image=X, mask=y)
            X = augmented["image"]
            y = augmented["mask"]

            if y.ndim == 3 and y.shape[0] == 1:
                pass
            elif y.ndim == 3 and y.shape[-1] == 1:
                y = y.permute(2, 0, 1)
            elif y.ndim == 2:
                y = y.unsqueeze(0)

        return X, y


# ==============================================================================
# Attention Gate Module
# ==============================================================================


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate
        F_g: Number of channels in gating signal (from decoder)
        F_l: Number of channels in skip connection (from encoder)
        F_int: Number of channels in intermediate layer
        """
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: gating signal from coarser scale (decoder feature)
        x: feature map from encoder (skip connection)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Ensure spatial dimensions match
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(
                g1, size=x1.size()[2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# ==============================================================================
# Attention UNet Model
# ==============================================================================


# ==============================================================================
# Attention UNet Model (Corrected)
# ==============================================================================


class AttentionUNet(nn.Module):
    def __init__(self, n_channels=2):
        super(AttentionUNet, self).__init__()

        # Encoder (same as original UNet)
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

        # Decoder with attention
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
        input_size = x.size()[2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bridge
        b = self.bridge(e4)

        # Decoder with attention
        d4 = self.upconv4(b)
        e4_att = self.att4(d4, e4)
        # Upsample e4_att to match d4's size before concatenation
        e4_att = F.interpolate(
            e4_att, size=d4.size()[2:], mode="bilinear", align_corners=False
        )
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        e3_att = self.att3(d3, e3)
        # Upsample e3_att to match d3's size before concatenation
        e3_att = F.interpolate(
            e3_att, size=d3.size()[2:], mode="bilinear", align_corners=False
        )
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        e2_att = self.att2(d2, e2)
        # Upsample e2_att to match d2's size before concatenation
        e2_att = F.interpolate(
            e2_att, size=d2.size()[2:], mode="bilinear", align_corners=False
        )
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        e1_att = self.att1(d1, e1)
        # Upsample e1_att to match d1's size before concatenation
        e1_att = F.interpolate(
            e1_att, size=d1.size()[2:], mode="bilinear", align_corners=False
        )
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        output = self.final(d1)

        # Ensure output matches input size
        if output.size()[2:] != input_size:
            output = F.interpolate(
                output, size=input_size, mode="bilinear", align_corners=False
            )

        return output


# ==============================================================================
# Loss, Metrics, and Training Loop (Same as before)
# ==============================================================================


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()
        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    intersection = (pred_binary * target_binary).sum(dim=(2, 3))
    union = (pred_binary + target_binary).gt(0).float().sum(dim=(2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    tp = (pred_binary * target_binary).sum(dim=(2, 3))
    fp = (pred_binary * (1 - target_binary)).sum(dim=(2, 3))
    fn = ((1 - pred_binary) * target_binary).sum(dim=(2, 3))
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return {
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    save_dir="outputs",
    approach_name="attention",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(save_dir) / f"run_{approach_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = CombinedLoss(dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_f1": [],
        "val_f1": [],
    }

    config = {
        "approach": approach_name,
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss, epoch_train_metrics = 0, {"iou": 0, "f1": 0}
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            metrics = calculate_metrics(output, y)
            epoch_train_metrics["iou"] += metrics["iou"]
            epoch_train_metrics["f1"] += metrics["f1"]

        epoch_train_loss /= len(train_loader)
        epoch_train_metrics["iou"] /= len(train_loader)
        epoch_train_metrics["f1"] /= len(train_loader)
        history["train_loss"].append(epoch_train_loss)
        history["train_iou"].append(epoch_train_metrics["iou"])
        history["train_f1"].append(epoch_train_metrics["f1"])

        model.eval()
        epoch_val_loss, epoch_val_metrics = 0, {"iou": 0, "f1": 0}
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                output = model(X)
                epoch_val_loss += criterion(output, y).item()
                metrics = calculate_metrics(output, y)
                epoch_val_metrics["iou"] += metrics["iou"]
                epoch_val_metrics["f1"] += metrics["f1"]

        epoch_val_loss /= len(val_loader)
        epoch_val_metrics["iou"] /= len(val_loader)
        epoch_val_metrics["f1"] /= len(val_loader)
        history["val_loss"].append(epoch_val_loss)
        history["val_iou"].append(epoch_val_metrics["iou"])
        history["val_f1"].append(epoch_val_metrics["f1"])

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(
                f"Epoch {epoch + 1}: New best model saved with validation loss: {best_val_loss:.4f}"
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f} | "
            f"Train IoU: {epoch_train_metrics['iou']:.4f}, Val IoU: {epoch_val_metrics['iou']:.4f} | "
            f"Train F1: {epoch_train_metrics['f1']:.4f}, Val F1: {epoch_val_metrics['f1']:.4f}"
        )

        # Plot and save metrics
        plt.figure(figsize=(18, 5))
        plt.subplot(131)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.legend(), plt.grid(True)
        plt.title(f"Loss vs. Epochs ({approach_name.title()})")
        plt.subplot(132)
        plt.plot(history["train_iou"], label="Train IoU")
        plt.plot(history["val_iou"], label="Val IoU")
        plt.xlabel("Epoch"), plt.ylabel("IoU"), plt.legend(), plt.grid(True)
        plt.title(f"IoU vs. Epochs ({approach_name.title()})")
        plt.subplot(133)
        plt.plot(history["train_f1"], label="Train F1 Score")
        plt.plot(history["val_f1"], label="Val F1 Score")
        plt.xlabel("Epoch"), plt.ylabel("F1 Score"), plt.legend(), plt.grid(True)
        plt.title(f"F1 Score vs. Epochs ({approach_name.title()})")
        plt.tight_layout()
        plt.savefig(save_dir / "metrics.png")
        plt.close()

    print(f"Training finished ({approach_name.title()} approach).")
    return history, save_dir


# ==============================================================================
# Main Execution Block
# ==============================================================================

# Old main execution code removed - now handled by main() function
    print()

    # Create dataset and split train/val
    print("Grouping samples for train/val split...")
    temp_dataset = RosetteDataset(data_dir, use_intensity=USE_INTENSITY)
    samples = list(temp_dataset.sample_groups.keys())

    if len(samples) == 0:
        print("No samples found! Check your data directory path.")
        exit(1)

    np.random.shuffle(samples)
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_files = [f for s in train_samples for f in temp_dataset.sample_groups[s]]
    val_files = [f for s in val_samples for f in temp_dataset.sample_groups[s]]

    # Create datasets
    print("\nInitializing training dataset...")
    train_dataset = RosetteDataset(data_dir, mode="train", use_intensity=USE_INTENSITY)
    train_dataset.h5_files = train_files

    print("\nInitializing validation dataset...")
    val_dataset = RosetteDataset(data_dir, mode="val", use_intensity=USE_INTENSITY)
    val_dataset.h5_files = val_files

    print(f"\nTotal samples: {len(samples)}")
    print(f"Training samples (files): {len(train_files)}")
    print(f"Validation samples (files): {len(val_files)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=False
    )

    # Initialize Attention UNet
    model = AttentionUNet(n_channels=n_channels)

    print(f"\nModel architecture: Attention UNet with {n_channels} input channels")

    # Train the model
    history, save_dir = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        save_dir=output_dir,
        approach_name=approach_name,
    )

    print(f"\nTraining completed. Outputs saved to: {save_dir}")
    print(f"This was the {approach_name.upper().replace('_', ' ')} approach")
    
    return save_dir


def main():
    """Main function with argument parsing for Snakemake integration."""
    parser = argparse.ArgumentParser(
        description="Train geometric rosette detection model using Attention UNet"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--data-dir", 
        default="/Users/noahbruderer/local_work_files/rosette_paper/training_data",
        help="Directory containing training data H5 files"
    )
    parser.add_argument(
        "--output-dir", 
        default="./training_outputs_attention_geometric_only",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (optional)"
    )
    
    # Training parameters
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size for training")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    
    # Model parameters
    parser.add_argument("--use-intensity", action="store_true", help="Use intensity data (3-channel vs 2-channel)")
    
    args = parser.parse_args()
    
    # Set up training parameters from arguments
    data_dir = args.data_dir
    USE_INTENSITY = args.use_intensity
    
    approach_name = (
        "attention_with_intensity" if USE_INTENSITY else "attention_geometric_only"
    )
    output_dir = args.output_dir
    n_channels = 3 if USE_INTENSITY else 2

    print(f"=== {approach_name.upper().replace('_', ' ')} APPROACH ===")
    print(f"This version uses Attention UNet with {n_channels} input channels")
    if USE_INTENSITY:
        print("Input features: Cell boundaries + Cell masks + Raw images + ATTENTION")
    else:
        print(
            "Input features: Cell boundaries + Cell masks (geometric-only) + ATTENTION"
        )

    print("\nGrouping samples for train/val split...")

    # Split data into training and validation sets by sample
    train_dataset = RosetteDataset(
        data_dir, patch_size=args.patch_size, mode="train", use_intensity=USE_INTENSITY
    )
    val_dataset = RosetteDataset(
        data_dir, patch_size=args.patch_size, mode="val", use_intensity=USE_INTENSITY
    )

    # Create data loaders with reduced num_workers for better stability
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False
    )

    # Initialize Attention UNet
    model = AttentionUNet(n_channels=n_channels)

    print(f"\nModel architecture: Attention UNet with {n_channels} input channels")

    # Train the model
    history, save_dir = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        save_dir=output_dir,
        approach_name=approach_name,
    )

    print(f"\nTraining completed. Outputs saved to: {save_dir}")
    print(f"This was the {approach_name.upper().replace('_', ' ')} approach")


if __name__ == "__main__":
    main()
