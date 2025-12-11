#!/usr/bin/env python3
"""
Cellpose cell segmentation script for Snakemake pipeline.
Compatible with Cellpose 3.x
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from skimage import measure

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Segment cells using Cellpose 3.x")

    # Required arguments
    parser.add_argument(
        "--input-image", type=str, required=True, help="Path to input image file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save outputs"
    )
    parser.add_argument(
        "--sample-name", type=str, required=True, help="Sample name for output files"
    )

    # Model parameters
    parser.add_argument(
        "--model-type",
        type=str,
        default="cyto3",
        help="Cellpose model type (e.g., 'cyto', 'cyto2', 'cyto3', 'nuclei')",
    )
    parser.add_argument(
        "--custom-model-path", type=str, default="", help="Path to custom trained model"
    )

    # Segmentation parameters
    parser.add_argument(
        "--diameter",
        type=str,
        default="None",
        help="Expected cell diameter in pixels (None for auto-detect)",
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="Flow error threshold (default: 0.4)",
    )
    parser.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Cell probability threshold (default: 0.0)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs=2,
        default=[0, 0],
        help="Channels to use [cytoplasm, nucleus]. Use [0,0] for grayscale",
    )
    parser.add_argument(
        "--min-size", type=int, default=15, help="Minimum cell size in pixels"
    )

    # Hardware
    parser.add_argument(
        "--use-gpu",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use GPU if available",
    )

    # Output options
    parser.add_argument(
        "--save-masks",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Save segmentation masks",
    )
    parser.add_argument(
        "--save-outlines",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Save outline visualization",
    )
    parser.add_argument(
        "--save-statistics",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Save cell statistics",
    )
    parser.add_argument(
        "--save-qc",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Save QC visualization",
    )

    return parser.parse_args()


def load_and_prepare_image(image_path: str) -> np.ndarray:
    """
    Load and prepare image for segmentation.

    Args:
        image_path: Path to image file

    Returns:
        Prepared image as numpy array
    """
    print(f"Loading image: {image_path}")
    image = tifffile.imread(image_path)

    print(
        f"Loaded image: {image.shape}, dtype: {image.dtype}, range: [{image.min():.3f}, {image.max():.3f}]"
    )

    # Handle multi-channel images
    if image.ndim == 3:
        if image.shape[0] <= 4:
            # Channels-first format
            print(f"Detected channels-first format with {image.shape[0]} channels")
            image = image[0]  # Take first channel
        else:
            # Channels-last format or z-stack
            print("Detected channels-last format or z-stack")
            image = image[:, :, 0]  # Take first channel

    # Normalize to float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    print(f"Prepared image: {image.shape}, dtype: {image.dtype}")

    return image


def segment_image(
    image: np.ndarray,
    model_type: str = "cyto3",
    custom_model_path: str = "",
    diameter: float = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    channels: list = [0, 0],
    min_size: int = 15,
    use_gpu: bool = True,
) -> tuple:
    """
    Segment cells using Cellpose 3.x

    Args:
        image: Input image
        model_type: Cellpose model type
        custom_model_path: Path to custom model
        diameter: Expected cell diameter
        flow_threshold: Flow error threshold
        cellprob_threshold: Cell probability threshold
        channels: Channel configuration
        min_size: Minimum cell size
        use_gpu: Use GPU if available

    Returns:
        Tuple of (masks, flows, styles, diameters)
    """
    print("\n" + "=" * 60)
    print("CELLPOSE SEGMENTATION")
    print("=" * 60)

    # Load model
    if custom_model_path and os.path.exists(custom_model_path):
        print(f"Loading CUSTOM model from: {custom_model_path}")
        print(f"Using base architecture: {model_type}")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=custom_model_path)
    else:
        print(f"Loading built-in model: {model_type}")
        model = models.Cellpose(gpu=use_gpu, model_type=model_type)

    print(f"Cellpose model initialized. GPU enabled: {use_gpu}")

    # Run segmentation
    print("Running Cellpose segmentation...")
    import time

    start_time = time.time()

    # CellposeModel returns 3 values, Cellpose returns 4 values
    result = model.eval(
        image,
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )

    # Handle different return signatures
    if len(result) == 3:
        masks, flows, styles = result
        diameters = diameter  # Use input diameter or None
    else:
        masks, flows, styles, diameters = result

    elapsed = time.time() - start_time
    print(f"Segmentation completed in {elapsed:.2f} seconds")

    num_cells = len(np.unique(masks)) - 1  # Subtract background
    print(f"Detected {num_cells} cells")
    print(f"Used diameter: {diameters}")

    return masks, flows, styles, diameters


def compute_cell_statistics(masks: np.ndarray, image: np.ndarray) -> pd.DataFrame:
    """
    Compute statistics for each segmented cell.

    Args:
        masks: Segmentation mask array
        image: Original image

    Returns:
        DataFrame with cell statistics
    """
    print("Computing cell statistics...")

    regions = measure.regionprops(masks, intensity_image=image)

    stats = []
    for region in regions:
        stats.append(
            {
                "cell_id": region.label,
                "area": region.area,
                "perimeter": region.perimeter,
                "eccentricity": region.eccentricity,
                "solidity": region.solidity,
                "mean_intensity": region.mean_intensity,
                "max_intensity": region.max_intensity,
                "min_intensity": region.min_intensity,
                "centroid_y": region.centroid[0],
                "centroid_x": region.centroid[1],
                "bbox_min_y": region.bbox[0],
                "bbox_min_x": region.bbox[1],
                "bbox_max_y": region.bbox[2],
                "bbox_max_x": region.bbox[3],
            }
        )

    df = pd.DataFrame(stats)
    print(f"Computed statistics for {len(df)} cells")

    return df


def create_outline_visualization(
    image: np.ndarray, masks: np.ndarray, output_path: str
) -> None:
    """
    Create and save outline visualization.

    Args:
        image: Original image
        masks: Segmentation masks
        output_path: Path to save visualization
    """
    print("Creating outline visualization...")

    from skimage import segmentation

    # Normalize image for display
    img_display = image.copy()
    img_display = (img_display - img_display.min()) / (
        img_display.max() - img_display.min()
    )

    # Find boundaries
    boundaries = segmentation.find_boundaries(masks, mode="outer")

    # Create RGB image
    rgb_image = np.stack([img_display] * 3, axis=-1)

    # Overlay boundaries in red
    rgb_image[boundaries, 0] = 1.0  # Red channel
    rgb_image[boundaries, 1] = 0.0  # Green channel
    rgb_image[boundaries, 2] = 0.0  # Blue channel

    # Save
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title(f"Cell Outlines (n={len(np.unique(masks)) - 1} cells)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved outline visualization: {output_path}")


def create_qc_visualization(
    image: np.ndarray,
    masks: np.ndarray,
    stats_df: pd.DataFrame,
    output_path: str,
    diameter: float = None,
) -> None:
    """
    Create comprehensive QC visualization.

    Args:
        image: Original image
        masks: Segmentation masks
        stats_df: Cell statistics DataFrame
        output_path: Path to save visualization
        diameter: Used diameter value
    """
    print("Creating QC visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Normalize image
    img_display = (image - image.min()) / (image.max() - image.min())

    # 1. Original image
    axes[0, 0].imshow(img_display, cmap="gray")
    axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 0].axis("off")

    # 2. Segmentation masks (colored)
    axes[0, 1].imshow(masks, cmap="tab20", interpolation="nearest")
    axes[0, 1].set_title(
        f"Segmentation ({len(np.unique(masks)) - 1} cells)", fontsize=14
    )
    axes[0, 1].axis("off")

    # 3. Overlay
    from skimage import segmentation

    boundaries = segmentation.find_boundaries(masks, mode="outer")
    rgb_overlay = np.stack([img_display] * 3, axis=-1)
    rgb_overlay[boundaries, 0] = 1.0
    rgb_overlay[boundaries, 1] = 0.0
    rgb_overlay[boundaries, 2] = 0.0
    axes[1, 0].imshow(rgb_overlay)
    axes[1, 0].set_title("Overlay", fontsize=14)
    axes[1, 0].axis("off")

    # 4. Statistics summary
    axes[1, 1].axis("off")

    # Create summary text
    diameter_str = (
        f"{diameter:.1f}" if diameter and diameter != "None" else "Auto-detected"
    )

    summary_text = [
        "Segmentation Statistics",
        "=" * 30,
        f"Total cells: {len(stats_df)}",
        f"Diameter used: {diameter_str}",
        "",
        "Cell Area Statistics:",
        f"  Mean: {stats_df['area'].mean():.1f} px²",
        f"  Median: {stats_df['area'].median():.1f} px²",
        f"  Std: {stats_df['area'].std():.1f} px²",
        f"  Min: {stats_df['area'].min():.1f} px²",
        f"  Max: {stats_df['area'].max():.1f} px²",
        "",
        "Intensity Statistics:",
        f"  Mean: {stats_df['mean_intensity'].mean():.1f}",
        f"  Median: {stats_df['mean_intensity'].median():.1f}",
        "",
        "Morphology:",
        f"  Mean eccentricity: {stats_df['eccentricity'].mean():.3f}",
        f"  Mean solidity: {stats_df['solidity'].mean():.3f}",
    ]

    axes[1, 1].text(
        0.1,
        0.9,
        "\n".join(summary_text),
        transform=axes[1, 1].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved QC visualization: {output_path}")


def main():
    """Main execution function."""
    args = parse_arguments()

    # Convert string booleans
    use_gpu = args.use_gpu.lower() == "true"
    save_masks = args.save_masks.lower() == "true"
    save_outlines = args.save_outlines.lower() == "true"
    save_statistics = args.save_statistics.lower() == "true"
    save_qc = args.save_qc.lower() == "true"

    # Convert diameter to proper type
    if args.diameter.lower() in ["none", "null", ""]:
        diameter = None
    else:
        diameter = float(args.diameter)

    print("=" * 70)
    print(f"CELLPOSE SEGMENTATION - Sample: {args.sample_name}")
    print("=" * 70)

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load image
        image = load_and_prepare_image(args.input_image)

        # Run segmentation
        masks, flows, styles, diameters = segment_image(
            image=image,
            model_type=args.model_type,
            custom_model_path=args.custom_model_path,
            diameter=diameter,  # Use converted diameter
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            channels=args.channels,
            min_size=args.min_size,
            use_gpu=use_gpu,
        )

        # Compute statistics
        stats_df = compute_cell_statistics(masks, image)

        # Save outputs
        if save_masks:
            masks_path = os.path.join(args.output_dir, f"{args.sample_name}_masks.tif")
            tifffile.imwrite(masks_path, masks.astype(np.uint16))
            print(f"Saved masks: {masks_path}")

        if save_statistics:
            stats_path = os.path.join(args.output_dir, f"{args.sample_name}_stats.csv")
            stats_df.to_csv(stats_path, index=False)
            print(f"Saved statistics: {stats_path}")

        if save_outlines:
            outlines_path = os.path.join(
                args.output_dir, f"{args.sample_name}_outlines.png"
            )
            create_outline_visualization(image, masks, outlines_path)

        if save_qc:
            qc_path = os.path.join(args.output_dir, f"{args.sample_name}_qc.png")
            create_qc_visualization(image, masks, stats_df, qc_path, diameters)

        print("\n" + "=" * 70)
        print("✓ SEGMENTATION COMPLETED SUCCESSFULLY")
        print("=" * 70)

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"ERROR: Segmentation failed for {args.sample_name}")
        print(f"Error details: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
