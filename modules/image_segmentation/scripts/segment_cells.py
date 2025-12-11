#!/usr/bin/env python3
"""
Cell segmentation script using Cellpose for the image_segmentation module.

This script provides a command-line interface for running Cellpose segmentation
with configurable parameters and output formats compatible with the Snakemake pipeline.
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from cellpose.utils import masks_to_outlines
from PIL import Image
from skimage import io as skio
from skimage import measure


def load_image(image_path):
    """Load image from file with multiple format support."""
    try:
        # Try tifffile first for TIFF images
        if image_path.lower().endswith((".tif", ".tiff")):
            img = tifffile.imread(image_path)
        else:
            # Use skimage for other formats
            img = skio.imread(image_path)

        # Ensure image is in the right format
        if img.dtype == np.uint16:
            img = img.astype(np.float32)
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        print(
            f"Loaded image: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]"
        )

        # Check image dimensions and log problematic ones
        if len(img.shape) != 2:
            debug_log_path = Path(
                "/Users/noahbruderer/local_work_files/rosette_paper/results/problematic_images.txt"
            )
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(debug_log_path, "a") as f:
                f.write(
                    f"{image_path} - Shape: {img.shape} - Expected 2D but got {len(img.shape)}D\n"
                )

            raise ValueError(f"Invalid shape {img.shape} for image data")

        return img

    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")


def setup_cellpose_model(model_type, custom_model_path="", use_gpu=True):
    """Initialize Cellpose model."""
    try:
        if custom_model_path:
            print(f"Loading CUSTOM model from: {custom_model_path}")
            print(f"Using base architecture: {model_type}")

            if not os.path.exists(custom_model_path):
                raise FileNotFoundError(f"Custom model not found: {custom_model_path}")

            model = models.CellposeModel(
                pretrained_model=custom_model_path,
                model_type=model_type,  # <-- This is the crucial part
                gpu=use_gpu,
            )
        else:
            print(f"Loading BUILT-IN model: {model_type}")
            model = models.CellposeModel(model_type=model_type, gpu=use_gpu)

        print(f"Cellpose model initialized. GPU enabled: {model.gpu}")
        return model

    except Exception as e:
        print(f"Warning: Could not initialize GPU model ({e}). Falling back to CPU.")
        # Fallback to CPU
        if custom_model_path:
            model = models.CellposeModel(
                pretrained_model=custom_model_path, model_type=model_type, gpu=False
            )
        else:
            model = models.CellposeModel(model_type=model_type, gpu=False)
        return model


def run_segmentation(
    model,
    image,
    diameter,
    flow_threshold,
    cellprob_threshold,
    channels,
    min_size,
    normalize=True,
):
    """Run Cellpose segmentation on image."""
    print("Running Cellpose segmentation...")
    start_time = time.time()

    try:
        result = model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            normalize=normalize,
        )

        # Handle different return formats from Cellpose versions
        if len(result) == 4:
            masks, flows, styles, diams = result
        elif len(result) == 3:
            masks, flows, styles = result
            diams = diameter  # Use input diameter as fallback
        else:
            raise ValueError(
                f"Unexpected number of return values from model.eval(): {len(result)}"
            )

        end_time = time.time()
        processing_time = end_time - start_time

        n_cells = len(np.unique(masks)) - 1  # Subtract 1 for background
        print(f"Segmentation completed in {processing_time:.2f} seconds")
        print(f"Detected {n_cells} cells")
        print(f"Used diameter: {diams}")

        return masks, flows, styles, diams, processing_time

    except Exception as e:
        raise RuntimeError(f"Segmentation failed: {e}")


def compute_cell_statistics(masks):
    """Compute basic statistics for segmented cells."""
    print("Computing cell statistics...")

    props = measure.regionprops(masks)

    stats = []
    for prop in props:
        stats.append(
            {
                "cell_id": prop.label,
                "area": prop.area,
                "perimeter": prop.perimeter,
                "centroid_x": prop.centroid[1],
                "centroid_y": prop.centroid[0],
                "eccentricity": prop.eccentricity,
                "solidity": prop.solidity,
                "extent": prop.extent,
                "bbox_min_row": prop.bbox[0],
                "bbox_min_col": prop.bbox[1],
                "bbox_max_row": prop.bbox[2],
                "bbox_max_col": prop.bbox[3],
            }
        )

    return pd.DataFrame(stats)


def create_qc_visualization(
    image, masks, sample_name, processing_time, n_cells, diameter_used
):
    """Create quality control visualization."""
    print("Creating QC visualization...")

    # Get outlines
    outlines = masks_to_outlines(masks)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Segmentation masks
    axes[1].imshow(masks, cmap="viridis")
    axes[1].set_title(f"Segmentation Masks\n{n_cells} cells detected")
    axes[1].axis("off")

    # Overlay outlines on original
    axes[2].imshow(image, cmap="gray", alpha=0.7)
    axes[2].imshow(outlines, cmap="Reds", alpha=0.8)
    axes[2].set_title(f"Cell Outlines\nDiameter: {diameter_used:.1f}px")
    axes[2].axis("off")

    # Add processing info
    fig.suptitle(
        f"Sample: {sample_name} | Processing time: {processing_time:.2f}s",
        fontsize=12,
        y=0.95,
    )

    plt.tight_layout()
    return fig


def save_outputs(
    masks,
    outlines,
    stats_df,
    qc_fig,
    output_dir,
    sample_name,
    save_masks=True,
    save_outlines=True,
    save_statistics=True,
    save_qc=True,
):
    """Save all outputs in specified formats."""
    print("Saving outputs...")

    output_files = {}

    if save_masks:
        # Save masks as TIFF
        mask_path = output_dir / f"{sample_name}_masks.tif"
        tifffile.imwrite(mask_path, masks.astype(np.uint16))
        output_files["masks"] = mask_path
        print(f"Saved masks: {mask_path}")

    if save_outlines:
        # Save outlines as PNG
        outline_path = output_dir / f"{sample_name}_outlines.png"
        # Convert outlines to 8-bit for PNG
        outline_img = (outlines * 255).astype(np.uint8)
        Image.fromarray(outline_img).save(outline_path)
        output_files["outlines"] = outline_path
        print(f"Saved outlines: {outline_path}")

    if save_statistics:
        # Save statistics as CSV
        stats_path = output_dir / f"{sample_name}_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        output_files["statistics"] = stats_path
        print(f"Saved statistics: {stats_path}")

    if save_qc:
        # Save QC visualization
        qc_path = output_dir / f"{sample_name}_qc.png"
        qc_fig.savefig(qc_path, dpi=150, bbox_inches="tight")
        plt.close(qc_fig)
        output_files["qc"] = qc_path
        print(f"Saved QC visualization: {qc_path}")

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Run Cellpose segmentation on microscopy images"
    )

    # Input/Output
    parser.add_argument("--input-image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--sample-name", required=True, help="Sample name for outputs")

    # Model parameters
    parser.add_argument("--model-type", default="cyto2", help="Cellpose model type")
    parser.add_argument("--custom-model-path", default="", help="Path to custom model")

    # Segmentation parameters
    parser.add_argument(
        "--diameter", type=float, default=None, help="Expected cell diameter"
    )
    parser.add_argument(
        "--flow-threshold", type=float, default=0.4, help="Flow threshold"
    )
    parser.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Cell probability threshold",
    )
    parser.add_argument(
        "--channels", type=int, nargs=2, default=[0, 0], help="Channels [cyto, nuclei]"
    )
    parser.add_argument("--min-size", type=int, default=15, help="Minimum cell size")

    # Processing options
    parser.add_argument(
        "--use-gpu",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Use GPU if available",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Normalize images",
    )

    # Output options
    parser.add_argument(
        "--save-masks",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Save segmentation masks",
    )
    parser.add_argument(
        "--save-outlines",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Save cell outlines",
    )
    parser.add_argument(
        "--save-statistics",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Save cell statistics",
    )
    parser.add_argument(
        "--save-qc",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Save QC visualization",
    )

    args = parser.parse_args()

    # Convert string booleans to actual booleans
    args.use_gpu = args.use_gpu.lower() == "true"
    args.normalize = args.normalize.lower() == "true"
    args.save_masks = args.save_masks.lower() == "true"
    args.save_outlines = args.save_outlines.lower() == "true"
    args.save_statistics = args.save_statistics.lower() == "true"
    args.save_qc = args.save_qc.lower() == "true"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CELLPOSE SEGMENTATION - Sample: {args.sample_name}")
    print("=" * 60)

    try:
        # Load image
        image = load_image(args.input_image)

        # Setup model
        model = setup_cellpose_model(
            args.model_type, args.custom_model_path, args.use_gpu
        )
        current_diameter = args.diameter

        if args.custom_model_path:
            current_diameter = 30  # ✅ Let Cellpose auto-detect
        else:
            current_diameter = args.diameter
        # Run segmentation
        masks, flows, styles, diams, processing_time = run_segmentation(
            model,
            image,
            current_diameter,
            args.flow_threshold,
            args.cellprob_threshold,
            args.channels,
            args.min_size,
            args.normalize,
        )

        # Get outlines for visualization
        outlines = masks_to_outlines(masks)

        # Compute statistics
        stats_df = compute_cell_statistics(masks)
        n_cells = len(stats_df)

        # Create QC visualization
        qc_fig = create_qc_visualization(
            image, masks, args.sample_name, processing_time, n_cells, diams
        )

        # Save outputs
        output_files = save_outputs(
            masks,
            outlines,
            stats_df,
            qc_fig,
            output_dir,
            args.sample_name,
            args.save_masks,
            args.save_outlines,
            args.save_statistics,
            args.save_qc,
        )

        print("\n" + "=" * 60)
        print("SEGMENTATION COMPLETED SUCCESSFULLY")
        print(f"Sample: {args.sample_name}")
        print(f"Cells detected: {n_cells}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nERROR: Segmentation failed for {args.sample_name}")
        print(f"Error details: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
