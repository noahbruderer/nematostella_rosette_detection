#!/usr/bin/env python3
"""
Rosette Detection Inference Script
=================================

This script performs inference on new segmentation data using a trained rosette detection model.
It processes segmentation masks and raw intensity images to predict rosette locations.

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from plotly.subplots import make_subplots
from skimage.segmentation import find_boundaries
from tqdm import tqdm

# Import your morphosnaker utilities if available
try:
    from morphosnaker import utils

    MORPHOSNAKER_AVAILABLE = True
except ImportError:
    print("Warning: morphosnaker not available, using fallback image loading")
    MORPHOSNAKER_AVAILABLE = False

# ==============================================================================
# Model Definition (Same as Training)
# ==============================================================================


class PretrainedSegmentationModel(nn.Module):
    def __init__(
        self, architecture="UNet", encoder_name="resnet34", in_channels=3, classes=1
    ):
        super().__init__()

        # Handle legacy architecture names
        if architecture == "Pretrained Segmentation Model":
            print("Warning: Using legacy architecture name, defaulting to UNet")
            architecture = "UNet"

        # Choose architecture
        if architecture == "UNet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture == "UnetPlusPlus":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture == "FPN":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture == "PSPNet":
            self.model = smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture == "DeepLabV3Plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        else:
            print(f"Warning: Unknown architecture '{architecture}', defaulting to UNet")
            architecture = "UNet"
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )

        self.sigmoid = nn.Sigmoid()

        # If we have non-RGB channels, adapt the first layer
        if in_channels != 3:
            self._adapt_input_channels(in_channels)

    def _adapt_input_channels(self, in_channels):
        """Adapt the first convolutional layer for non-RGB inputs"""
        # Get the first conv layer
        first_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                first_conv_name = name
                break

        if first_conv is not None:
            # Create new conv layer with the right number of input channels
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

            # Initialize new weights
            with torch.no_grad():
                if in_channels < 3:
                    # For 2 channels, use first 2 channels of pretrained weights
                    new_conv.weight[:, :in_channels] = first_conv.weight[
                        :, :in_channels
                    ]
                else:
                    # For more channels, repeat the RGB weights
                    new_conv.weight[:, :3] = first_conv.weight
                    for i in range(3, in_channels):
                        new_conv.weight[:, i : i + 1] = first_conv.weight[
                            :, i % 3 : i % 3 + 1
                        ]

                if first_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)

            # Replace the first conv layer
            module_names = first_conv_name.split(".")
            parent = self.model
            for name in module_names[:-1]:
                parent = getattr(parent, name)
            setattr(parent, module_names[-1], new_conv)

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x


# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================


class ImageLoader:
    """Handles loading of different image formats."""

    def __init__(self):
        self.morphosnaker_available = MORPHOSNAKER_AVAILABLE
        if self.morphosnaker_available:
            self.image_processor = utils.ImageProcessor()

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load an image using available methods."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if self.morphosnaker_available:
            try:
                # Try morphosnaker first (handles 5D format)
                image = self.image_processor.load(str(path), input_dims="XY")
                return image
            except Exception as e:
                print(f"Morphosnaker loading failed for {path}, trying fallback: {e}")

        # Fallback to basic loading methods
        try:
            from skimage import io

            image = io.imread(path)

            # Convert to 5D format if needed (matching training pipeline)
            if image.ndim == 2:
                image = image[None, None, None, :, :]  # Add C, T, Z dimensions
            elif image.ndim == 3:
                if image.shape[0] < 10:  # Assume channels first
                    image = image[None, None, :, :, :]  # Add T, Z dimensions
                else:  # Assume spatial dimensions
                    image = image[None, None, None, :, :]  # Add C, T, Z dimensions

            return image

        except ImportError:
            # Try with PIL
            from PIL import Image

            image = np.array(Image.open(path))

            # Convert to 5D format
            if image.ndim == 2:
                image = image[None, None, None, :, :]
            elif image.ndim == 3:
                image = image[None, None, None, :, :]

            return image


def get_inference_transforms(patch_size: int = 512) -> A.Compose:
    """Get transforms for inference (no augmentation)."""
    return A.Compose([ToTensorV2()])


def discover_inference_files(data_dir: Path) -> List[Dict[str, Path]]:
    """
    Discover paired intensity and segmentation files.

    Returns:
        List of dictionaries with 'intensity', 'segmentation', and 'sample_id' keys
    """
    data_dir = Path(data_dir)
    file_pairs = []

    # Find all .tif files
    all_tif_files = list(data_dir.glob("*.tif"))

    # Group files by sample
    samples = {}
    for tif_file in all_tif_files:
        name = tif_file.name

        # Skip files that are clearly processed versions
        if any(
            suffix in name.lower() for suffix in ["_pred", "_float", "_seg", "_0001"]
        ):
            continue

        # This should be a base intensity file
        sample_id = tif_file.stem
        samples[sample_id] = {"intensity": tif_file}

    # Now find corresponding segmentation files
    for sample_id, info in samples.items():
        # Look for segmentation file patterns
        seg_patterns = [
            f"{sample_id}_*seg*cur.tif",
            f"{sample_id}_seg_cur.tif",
            f"{sample_id}_seg.tif",
        ]

        seg_file = None
        for pattern in seg_patterns:
            matches = list(data_dir.glob(pattern))
            if matches:
                seg_file = matches[0]  # Take first match
                break

        if seg_file:
            file_pairs.append(
                {
                    "sample_id": sample_id,
                    "intensity": info["intensity"],
                    "segmentation": seg_file,
                }
            )
        else:
            print(f"Warning: No segmentation file found for {sample_id}")

    return file_pairs


def prepare_input_data(
    intensity_path: Path,
    segmentation_path: Path,
    use_intensity: bool = True,
    patch_size: int = 512,
) -> Tuple[np.ndarray, Dict]:
    """
    Prepare input data exactly like the training pipeline.

    Args:
        intensity_path: Path to raw intensity image
        segmentation_path: Path to segmentation mask
        use_intensity: Whether to include intensity data (should match training)
        patch_size: Patch size for processing

    Returns:
        Tuple of (input_array, metadata)
    """
    loader = ImageLoader()

    # Load images
    raw_image = loader.load_image(intensity_path)
    cell_mask = loader.load_image(segmentation_path)

    # Extract 2D slices (same as training)
    if raw_image.ndim > 2:
        raw_image_2d = raw_image[0, 0, 0]
    else:
        raw_image_2d = raw_image

    if cell_mask.ndim > 2:
        cell_mask_2d = cell_mask[0, 0, 0]
    else:
        cell_mask_2d = cell_mask

    # Create input channels exactly like training
    cell_boundaries = find_boundaries(cell_mask_2d, mode="thick").astype(np.float32)
    cell_mask_binary = (cell_mask_2d > 0).astype(np.float32)

    if use_intensity:
        # Normalize intensity
        raw_image_norm = raw_image_2d.astype(np.float32)
        if raw_image_norm.max() > 1.0:
            raw_image_norm = raw_image_norm / raw_image_norm.max()

        # Stack channels: boundaries, binary mask, intensity
        input_array = np.stack(
            [cell_boundaries, cell_mask_binary, raw_image_norm], axis=0
        )
    else:
        # Stack channels: boundaries, binary mask only
        input_array = np.stack([cell_boundaries, cell_mask_binary], axis=0)

    # Metadata
    metadata = {
        "original_shape": input_array.shape[1:],
        "intensity_path": str(intensity_path),
        "segmentation_path": str(segmentation_path),
        "use_intensity": use_intensity,
        "n_channels": input_array.shape[0],
    }

    return input_array, metadata


def sliding_window_inference(
    model: nn.Module,
    input_array: np.ndarray,
    patch_size: int = 512,
    overlap: int = 128,
    device: torch.device = None,
) -> np.ndarray:
    """
    Perform sliding window inference on large images.

    Args:
        model: Trained model
        input_array: Input data array (C, H, W)
        patch_size: Size of patches for inference
        overlap: Overlap between patches
        device: Device to run inference on

    Returns:
        Full-size prediction array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    C, H, W = input_array.shape

    # If image is smaller than patch_size, process directly
    if H <= patch_size and W <= patch_size:
        # Pad to patch_size if needed
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)

        if pad_h > 0 or pad_w > 0:
            input_padded = np.pad(
                input_array, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
            )
        else:
            input_padded = input_array

        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(input_padded).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        # Remove padding and batch dimension
        output_np = output.squeeze(0).squeeze(0).cpu().numpy()
        if pad_h > 0 or pad_w > 0:
            output_np = output_np[:H, :W]

        return output_np

    # Initialize output array
    prediction = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    # Calculate step size
    step = patch_size - overlap

    # Sliding window
    for y in range(0, H - patch_size + 1, step):
        for x in range(0, W - patch_size + 1, step):
            # Extract patch
            patch = input_array[:, y : y + patch_size, x : x + patch_size]

            # Convert to tensor
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                patch_pred = model(patch_tensor)

            # Add to output
            patch_pred_np = patch_pred.squeeze(0).squeeze(0).cpu().numpy()
            prediction[y : y + patch_size, x : x + patch_size] += patch_pred_np
            counts[y : y + patch_size, x : x + patch_size] += 1

    # Handle edge cases
    if H > patch_size:
        # Bottom edge
        y = H - patch_size
        for x in range(0, W - patch_size + 1, step):
            patch = input_array[:, y : y + patch_size, x : x + patch_size]
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                patch_pred = model(patch_tensor)

            patch_pred_np = patch_pred.squeeze(0).squeeze(0).cpu().numpy()
            prediction[y : y + patch_size, x : x + patch_size] += patch_pred_np
            counts[y : y + patch_size, x : x + patch_size] += 1

    if W > patch_size:
        # Right edge
        x = W - patch_size
        for y in range(0, H - patch_size + 1, step):
            patch = input_array[:, y : y + patch_size, x : x + patch_size]
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                patch_pred = model(patch_tensor)

            patch_pred_np = patch_pred.squeeze(0).squeeze(0).cpu().numpy()
            prediction[y : y + patch_size, x : x + patch_size] += patch_pred_np
            counts[y : y + patch_size, x : x + patch_size] += 1

    # Bottom-right corner
    if H > patch_size and W > patch_size:
        y, x = H - patch_size, W - patch_size
        patch = input_array[:, y : y + patch_size, x : x + patch_size]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            patch_pred = model(patch_tensor)

        patch_pred_np = patch_pred.squeeze(0).squeeze(0).cpu().numpy()
        prediction[y : y + patch_size, x : x + patch_size] += patch_pred_np
        counts[y : y + patch_size, x : x + patch_size] += 1

    # Average overlapping predictions
    prediction = np.divide(
        prediction, counts, out=np.zeros_like(prediction), where=counts != 0
    )

    return prediction


# ==============================================================================
# Post-processing and Analysis
# ==============================================================================


def postprocess_predictions(
    prediction: np.ndarray, threshold: float = 0.5, min_size: int = 10
) -> np.ndarray:
    """
    Post-process model predictions to create binary masks.

    Args:
        prediction: Raw model output (0-1)
        threshold: Threshold for binarization
        min_size: Minimum size of detected objects

    Returns:
        Binary mask of detected rosettes
    """
    from skimage.morphology import remove_small_objects

    # Threshold
    binary_mask = prediction > threshold

    # Remove small objects
    binary_mask = remove_small_objects(binary_mask, min_size=min_size)

    return binary_mask.astype(np.uint8)


def analyze_predictions(
    prediction: np.ndarray,
    binary_mask: np.ndarray,
    cell_mask: np.ndarray,
    sample_id: str,
) -> Dict:
    """
    Analyze predictions and compute statistics.

    Args:
        prediction: Raw prediction array
        binary_mask: Thresholded binary mask
        cell_mask: Original cell segmentation
        sample_id: Sample identifier

    Returns:
        Dictionary with analysis results
    """
    from skimage.measure import label, regionprops

    # Label connected components
    labeled_rosettes = label(binary_mask)

    # Basic statistics
    n_rosettes = np.max(labeled_rosettes)
    total_rosette_area = np.sum(binary_mask)
    mean_confidence = np.mean(prediction[binary_mask > 0]) if np.any(binary_mask) else 0

    # Cell-level analysis
    n_total_cells = len(np.unique(cell_mask)) - 1  # Subtract background

    # Find cells that overlap with rosettes
    rosette_cells = set()
    for region in regionprops(labeled_rosettes):
        region_mask = labeled_rosettes == region.label
        overlapping_cells = np.unique(cell_mask[region_mask])
        rosette_cells.update(overlapping_cells[overlapping_cells > 0])

    n_rosette_cells = len(rosette_cells)

    # Rosette properties
    rosette_props = []
    for region in regionprops(labeled_rosettes):
        rosette_props.append(
            {
                "area": region.area,
                "centroid": region.centroid,
                "eccentricity": region.eccentricity,
                "major_axis_length": region.major_axis_length,
                "minor_axis_length": region.minor_axis_length,
            }
        )

    return {
        "sample_id": sample_id,
        "n_rosettes": n_rosettes,
        "total_rosette_area": total_rosette_area,
        "mean_confidence": mean_confidence,
        "n_total_cells": n_total_cells,
        "n_rosette_cells": n_rosette_cells,
        "rosette_fraction": n_rosette_cells / n_total_cells if n_total_cells > 0 else 0,
        "rosette_properties": rosette_props,
    }


# ==============================================================================
# Visualization
# ==============================================================================


def create_inference_visualization(
    input_channels: np.ndarray,
    prediction: np.ndarray,
    binary_mask: np.ndarray,
    sample_id: str,
    output_path: Path,
    use_intensity: bool = True,
):
    """Create comprehensive visualization of inference results."""

    # Prepare channel names
    if use_intensity:
        channel_names = ["Cell Boundaries", "Cell Mask", "Raw Intensity"]
    else:
        channel_names = ["Cell Boundaries", "Cell Mask"]

    n_input_channels = input_channels.shape[0]

    # Create subplot layout
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            *channel_names[:n_input_channels],
            "Raw Prediction",
            "Binary Rosettes",
            "Overlay",
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    # Plot input channels
    for i in range(min(n_input_channels, 3)):
        row = 1 if i < 3 else 2
        col = (i % 3) + 1

        fig.add_trace(
            go.Heatmap(
                z=input_channels[i],
                colorscale="gray" if "Boundaries" in channel_names[i] else "viridis",
                showscale=False,
                name=channel_names[i],
            ),
            row=row,
            col=col,
        )

    # Raw prediction
    fig.add_trace(
        go.Heatmap(
            z=prediction,
            colorscale="hot",
            showscale=True,
            colorbar=dict(title="Confidence", x=0.68, len=0.4),
            name="Raw Prediction",
        ),
        row=2,
        col=1,
    )

    # Binary mask
    fig.add_trace(
        go.Heatmap(
            z=binary_mask,
            colorscale=[[0, "white"], [1, "red"]],
            showscale=False,
            name="Binary Rosettes",
        ),
        row=2,
        col=2,
    )

    # Overlay
    # Use cell mask as base
    base_layer = input_channels[1] if n_input_channels > 1 else input_channels[0]

    fig.add_trace(
        go.Heatmap(
            z=base_layer, colorscale="gray", showscale=False, opacity=0.7, name="Cells"
        ),
        row=2,
        col=3,
    )

    # Add rosettes overlay
    fig.add_trace(
        go.Heatmap(
            z=binary_mask,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,0,0,0.8)"]],
            showscale=False,
            name="Rosettes",
        ),
        row=2,
        col=3,
    )

    # Update layout
    fig.update_layout(
        title=f"Inference Results: {sample_id}",
        height=800,
        width=1200,
        showlegend=False,
    )

    # Fix aspect ratios
    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
            fig.update_yaxes(scaleanchor="x", scaleratio=1, row=row, col=col)

    # Save
    fig.write_html(str(output_path))


def save_inference_results(
    prediction: np.ndarray,
    binary_mask: np.ndarray,
    analysis: Dict,
    output_dir: Path,
    sample_id: str,
):
    """Save inference results in multiple formats."""

    # Save prediction as numpy array
    np.save(output_dir / f"{sample_id}_prediction.npy", prediction)

    # Save binary mask as TIF
    from skimage import io

    io.imsave(
        output_dir / f"{sample_id}_rosettes.tif", (binary_mask * 255).astype(np.uint8)
    )

    # Save analysis as JSON
    # Convert numpy types to Python types for JSON serialization
    analysis_json = analysis.copy()
    for key, value in analysis_json.items():
        if isinstance(value, np.ndarray):
            analysis_json[key] = value.tolist()
        elif isinstance(value, (np.int32, np.int64)):
            analysis_json[key] = int(value)
        elif isinstance(value, (np.float32, np.float64)):
            analysis_json[key] = float(value)

    # Handle rosette properties
    if "rosette_properties" in analysis_json:
        for prop in analysis_json["rosette_properties"]:
            for k, v in prop.items():
                if isinstance(v, (np.int32, np.int64)):
                    prop[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    prop[k] = float(v)
                elif isinstance(v, tuple):
                    prop[k] = list(v)

    with open(output_dir / f"{sample_id}_analysis.json", "w") as f:
        json.dump(analysis_json, f, indent=2)


# ==============================================================================
# Main Inference Pipeline
# ==============================================================================


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging for inference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rosette_inference")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(output_dir / "inference.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_trained_model(
    model_path: Path, config_path: Optional[Path] = None
) -> Tuple[nn.Module, Dict]:
    """Load trained model and configuration."""

    # Load configuration
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # Try to find config in same directory as model
        model_dir = model_path.parent
        possible_configs = list(model_dir.glob("*config*.json"))
        if possible_configs:
            with open(possible_configs[0], "r") as f:
                config = json.load(f)
        else:
            # Default configuration
            print("Warning: No configuration file found, using defaults")
            config = {
                "approach": "pretrained_unet_resnet34_with_intensity",
            }

    print(f"Loaded config: {config}")

    # Parse configuration - prioritize the approach name over individual fields
    architecture = "UNet"
    encoder = "resnet34"
    n_channels = 3
    use_intensity = True

    # First, try to parse from approach name (most reliable)
    if "approach" in config:
        approach = config["approach"].lower()
        print(f"Parsing approach: {approach}")

        # Parse architecture
        if "unetplusplus" in approach or "unet++" in approach:
            architecture = "UnetPlusPlus"
        elif "fpn" in approach:
            architecture = "FPN"
        elif "pspnet" in approach:
            architecture = "PSPNet"
        elif "deeplabv3plus" in approach or "deeplabv3+" in approach:
            architecture = "DeepLabV3Plus"
        elif "unet" in approach:
            architecture = "UNet"

        # Parse encoder
        if "efficientnet" in approach:
            if "efficientnet-b0" in approach:
                encoder = "efficientnet-b0"
            elif "efficientnet-b1" in approach:
                encoder = "efficientnet-b1"
            elif "efficientnet-b2" in approach:
                encoder = "efficientnet-b2"
            elif "efficientnet-b3" in approach:
                encoder = "efficientnet-b3"
            elif "efficientnet-b4" in approach:
                encoder = "efficientnet-b4"
            else:
                encoder = "efficientnet-b3"  # default
        elif "resnet50" in approach:
            encoder = "resnet50"
        elif "resnet101" in approach:
            encoder = "resnet101"
        elif "resnet152" in approach:
            encoder = "resnet152"
        elif "resnext50" in approach:
            encoder = "resnext50_32x4d"
        elif "resnext101" in approach:
            encoder = "resnext101_32x8d"
        elif "resnet34" in approach or "resnet" in approach:
            encoder = "resnet34"

        # Parse intensity usage
        use_intensity = "with_intensity" in approach
        n_channels = 3 if use_intensity else 2

    print(
        f"After parsing approach - Architecture: {architecture}, Encoder: {encoder}, Channels: {n_channels}"
    )

    # Check explicit config values and only use them if they're sensible
    if "architecture" in config:
        config_arch = config["architecture"]
        print(f"Config architecture field: '{config_arch}'")
        if config_arch not in ["Pretrained Segmentation Model", None, ""]:
            print(f"Using config architecture: {config_arch}")
            architecture = config_arch
        else:
            print(f"Ignoring generic config architecture: '{config_arch}'")

    if "encoder" in config and config["encoder"]:
        print(f"Using config encoder: {config['encoder']}")
        encoder = config["encoder"]

    if "n_channels" in config and config["n_channels"]:
        print(f"Using config n_channels: {config['n_channels']}")
        n_channels = config["n_channels"]

    if "use_intensity" in config and isinstance(config["use_intensity"], bool):
        print(f"Using config use_intensity: {config['use_intensity']}")
        use_intensity = config["use_intensity"]
        n_channels = 3 if use_intensity else 2

    print("Final model config:")
    print(f"  Architecture: {architecture}")
    print(f"  Encoder: {encoder}")
    print(f"  Channels: {n_channels}")
    print(f"  Use intensity: {use_intensity}")

    # Create model
    model = PretrainedSegmentationModel(
        architecture=architecture,
        encoder_name=encoder,
        in_channels=n_channels,
        classes=1,
    )

    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model_config = {
        "architecture": architecture,
        "encoder": encoder,
        "n_channels": n_channels,
        "use_intensity": use_intensity,
    }

    return model, model_config


def run_inference(
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    threshold: float = 0.5,
    patch_size: int = 512,
    overlap: int = 128,
    min_rosette_size: int = 10,
):
    """Run inference on all samples in data directory."""

    # Setup
    logger = setup_logging(output_dir)
    logger.info("Starting rosette detection inference")

    # Load model
    logger.info(f"Loading model from {model_path}")
    model, model_config = load_trained_model(model_path, config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Model config: {model_config}")

    # Discover files
    logger.info(f"Discovering files in {data_dir}")
    file_pairs = discover_inference_files(data_dir)

    if not file_pairs:
        logger.error("No valid file pairs found!")
        return

    logger.info(f"Found {len(file_pairs)} samples to process")

    # Process each sample
    all_results = []

    for file_info in tqdm(file_pairs, desc="Processing samples"):
        sample_id = file_info["sample_id"]
        intensity_path = file_info["intensity"]
        segmentation_path = file_info["segmentation"]

        logger.info(f"Processing {sample_id}")

        try:
            # Prepare input data
            input_array, metadata = prepare_input_data(
                intensity_path,
                segmentation_path,
                use_intensity=model_config["use_intensity"],
                patch_size=patch_size,
            )

            logger.info(f"Input shape: {input_array.shape}")

            # Run inference
            prediction = sliding_window_inference(
                model,
                input_array,
                patch_size=patch_size,
                overlap=overlap,
                device=device,
            )

            # Post-process
            binary_mask = postprocess_predictions(
                prediction, threshold=threshold, min_size=min_rosette_size
            )

            # Load original cell mask for analysis
            loader = ImageLoader()
            cell_mask_full = loader.load_image(segmentation_path)
            if cell_mask_full.ndim > 2:
                cell_mask = cell_mask_full[0, 0, 0]
            else:
                cell_mask = cell_mask_full

            # Analyze results
            analysis = analyze_predictions(
                prediction, binary_mask, cell_mask, sample_id
            )

            # Create output directory for this sample
            sample_output_dir = output_dir / sample_id
            sample_output_dir.mkdir(exist_ok=True)

            # Save results
            save_inference_results(
                prediction, binary_mask, analysis, sample_output_dir, sample_id
            )

            # Create visualization
            create_inference_visualization(
                input_array,
                prediction,
                binary_mask,
                sample_id,
                sample_output_dir / f"{sample_id}_inference_results.html",
                use_intensity=model_config["use_intensity"],
            )

            all_results.append(analysis)
            logger.info(
                f"Completed {sample_id}: {analysis['n_rosettes']} rosettes detected"
            )

        except Exception as e:
            logger.error(f"Error processing {sample_id}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    # Create summary report
    create_summary_report(all_results, output_dir, model_config)

    logger.info(f"Inference complete! Results saved to {output_dir}")


def create_summary_report(results: List[Dict], output_dir: Path, model_config: Dict):
    """Create a summary report of all inference results."""

    if not results:
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save detailed results
    df.to_csv(output_dir / "inference_summary.csv", index=False)

    # Create summary statistics
    summary_stats = {
        "total_samples": len(results),
        "total_rosettes": df["n_rosettes"].sum(),
        "mean_rosettes_per_sample": df["n_rosettes"].mean(),
        "std_rosettes_per_sample": df["n_rosettes"].std(),
        "mean_rosette_fraction": df["rosette_fraction"].mean(),
        "samples_with_rosettes": (df["n_rosettes"] > 0).sum(),
        "model_config": model_config,
    }

    # Save summary
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)

    # Create summary visualization
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Rosettes per Sample",
            "Rosette Fraction Distribution",
            "Total Rosette Area",
            "Sample Overview",
        ],
    )

    # Histogram of rosettes per sample
    fig.add_trace(
        go.Histogram(x=df["n_rosettes"], nbinsx=20, name="Rosettes per Sample"),
        row=1,
        col=1,
    )

    # Rosette fraction distribution
    fig.add_trace(
        go.Histogram(x=df["rosette_fraction"], nbinsx=20, name="Rosette Fraction"),
        row=1,
        col=2,
    )

    # Total rosette area
    fig.add_trace(
        go.Histogram(x=df["total_rosette_area"], nbinsx=20, name="Total Rosette Area"),
        row=2,
        col=1,
    )

    # Scatter plot: rosettes vs cells
    fig.add_trace(
        go.Scatter(
            x=df["n_total_cells"],
            y=df["n_rosettes"],
            mode="markers",
            text=df["sample_id"],
            name="Rosettes vs Total Cells",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(title="Inference Summary Report", height=800, showlegend=False)

    fig.update_xaxes(title="Number of Rosettes", row=1, col=1)
    fig.update_xaxes(title="Rosette Fraction", row=1, col=2)
    fig.update_xaxes(title="Total Rosette Area", row=2, col=1)
    fig.update_xaxes(title="Total Cells", row=2, col=2)
    fig.update_yaxes(title="Count", row=1, col=1)
    fig.update_yaxes(title="Count", row=1, col=2)
    fig.update_yaxes(title="Count", row=2, col=1)
    fig.update_yaxes(title="Number of Rosettes", row=2, col=2)

    fig.write_html(output_dir / "summary_report.html")

    print("\nSummary Statistics:")
    print(f"Total samples processed: {summary_stats['total_samples']}")
    print(f"Total rosettes detected: {summary_stats['total_rosettes']}")
    print(
        f"Mean rosettes per sample: {summary_stats['mean_rosettes_per_sample']:.2f} ± {summary_stats['std_rosettes_per_sample']:.2f}"
    )
    print(
        f"Samples with rosettes: {summary_stats['samples_with_rosettes']}/{summary_stats['total_samples']}"
    )
    print(f"Mean rosette fraction: {summary_stats['mean_rosette_fraction']:.3f}")


# ==============================================================================
# Command Line Interface
# ==============================================================================


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Rosette Detection Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python inference.py --data_dir /path/to/data --model_path model.pth --output_dir results

  # With custom parameters
  python inference.py --data_dir /path/to/data --model_path model.pth --output_dir results --threshold 0.3 --patch_size 1024

  # With specific config
  python inference.py --data_dir /path/to/data --model_path model.pth --config_path config.json --output_dir results
        """,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing intensity (.tif) and segmentation (*seg*cur.tif) files",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (.pth file)",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to model configuration file (optional)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for rosette detection (default: 0.5)",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Patch size for sliding window inference (default: 512)",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Overlap between patches (default: 128)",
    )

    parser.add_argument(
        "--min_rosette_size",
        type=int,
        default=10,
        help="Minimum size of detected rosettes in pixels (default: 10)",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Convert paths
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config_path) if args.config_path else None

    # Validate paths
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return

    if not model_path.exists():
        print(f"Error: Model file does not exist: {model_path}")
        return

    if config_path and not config_path.exists():
        print(f"Warning: Config file does not exist: {config_path}")
        config_path = None

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    try:
        run_inference(
            data_dir=data_dir,
            model_path=model_path,
            output_dir=output_dir,
            config_path=config_path,
            threshold=args.threshold,
            patch_size=args.patch_size,
            overlap=args.overlap,
            min_rosette_size=args.min_rosette_size,
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
