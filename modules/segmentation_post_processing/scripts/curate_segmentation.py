#!/usr/bin/env python3
"""
Segmentation curation script for the segmentation_post_processing module.

This script provides a command-line interface for running segmentation curation
using the existing SegmentationCurator class, adapted for Snakemake pipeline integration.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage import io as skio

# Add the post_raw_segmentation_curration to the path so we can import
sys.path.append(str(Path(__file__).parent.parent / "post_raw_segmentation_curration"))
from segmentation_curator import SegmentationCurator


def load_mask(mask_path):
    """Load segmentation mask from file with multiple format support."""
    try:
        # Try tifffile first for TIFF images
        if mask_path.lower().endswith((".tif", ".tiff")):
            mask = tifffile.imread(mask_path)
        else:
            # Use skimage for other formats
            mask = skio.imread(mask_path)

        print(
            f"Loaded mask: {mask.shape}, dtype: {mask.dtype}, unique values: {len(np.unique(mask))}"
        )
        return mask

    except Exception as e:
        raise ValueError(f"Could not load mask {mask_path}: {e}")


def save_curated_mask(mask, output_path):
    """Save curated mask as TIFF file."""
    try:
        # Ensure mask is in the right format for saving
        if mask.dtype != np.uint16:
            # Convert to uint16 for better storage efficiency
            mask = mask.astype(np.uint16)

        tifffile.imwrite(output_path, mask)
        print(f"Saved curated mask: {output_path}")

    except Exception as e:
        raise ValueError(f"Could not save curated mask to {output_path}: {e}")


def compute_curation_statistics(original_mask, curated_mask):
    """Compute statistics about the curation process."""
    original_cells = len(np.unique(original_mask)) - 1  # Subtract 1 for background
    curated_cells = len(np.unique(curated_mask)) - 1

    stats = {
        "original_cell_count": original_cells,
        "curated_cell_count": curated_cells,
        "cells_changed": curated_cells - original_cells,
        "change_percentage": ((curated_cells - original_cells) / original_cells * 100)
        if original_cells > 0
        else 0,
    }

    return stats


def create_qc_visualization(
    original_mask, curated_mask, sample_name, stats, processing_time
):
    """Create quality control visualization comparing original and curated masks."""
    print("Creating QC visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original mask
    axes[0].imshow(original_mask, cmap="viridis")
    axes[0].set_title(f"Original Segmentation\n{stats['original_cell_count']} cells")
    axes[0].axis("off")

    # Curated mask
    axes[1].imshow(curated_mask, cmap="viridis")
    axes[1].set_title(f"Curated Segmentation\n{stats['curated_cell_count']} cells")
    axes[1].axis("off")

    # Difference visualization
    original_binary = (original_mask > 0).astype(np.int32)
    curated_binary = (curated_mask > 0).astype(np.int32)
    diff_mask = curated_binary - original_binary
    axes[2].imshow(diff_mask, cmap="RdBu", vmin=-1, vmax=1)
    axes[2].set_title(
        f"Changes\n{stats['cells_changed']:+d} cells ({stats['change_percentage']:+.1f}%)"
    )
    axes[2].axis("off")

    # Add processing info with better positioning
    fig.suptitle(
        f"Sample: {sample_name} | Processing time: {processing_time:.2f}s",
        fontsize=12,
        y=0.98,
    )

    # Adjust layout to prevent title overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Leave more space at the top for the title
    return fig


def save_outputs(
    curated_mask,
    stats,
    qc_fig,
    output_dir,
    sample_name,
    save_statistics=True,
    save_qc_images=True,
    create_visualizations=True,
):
    """Save all outputs in specified formats."""
    print("Saving outputs...")

    output_files = {}

    # Save curated mask
    mask_path = output_dir / f"{sample_name}_masks_curated.tif"
    save_curated_mask(curated_mask, mask_path)
    output_files["curated_mask"] = mask_path

    if save_statistics:
        # Save statistics as CSV
        stats_path = output_dir / f"{sample_name}_curation_stats.csv"
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(stats_path, index=False)
        output_files["statistics"] = stats_path
        print(f"Saved statistics: {stats_path}")

    if save_qc_images:
        # Save QC visualization
        qc_path = output_dir / f"{sample_name}_curation_qc.png"
        qc_fig.savefig(qc_path, dpi=150, bbox_inches="tight")
        plt.close(qc_fig)
        output_files["qc"] = qc_path
        print(f"Saved QC visualization: {qc_path}")

    if create_visualizations:
        # The curator saves multiple HTML visualizations
        raw_vis_path = output_dir / f"raw_mask_vis_{sample_name}.html"
        processed_vis_path = output_dir / f"processed_mask_vis_{sample_name}.html"
        comparison_vis_path = (
            output_dir / f"cell_area_distribution_comparison_{sample_name}.html"
        )

        output_files["raw_visualization"] = raw_vis_path
        output_files["processed_visualization"] = processed_vis_path
        output_files["comparison_visualization"] = comparison_vis_path

        print("HTML visualizations saved:")
        print(f"  Raw mask: {raw_vis_path}")
        print(f"  Processed mask: {processed_vis_path}")
        print(f"  Area comparison: {comparison_vis_path}")

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Run segmentation curation using SegmentationCurator"
    )

    # Input/Output
    parser.add_argument(
        "--input-mask", required=True, help="Path to input segmentation mask"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--sample-name", required=True, help="Sample name for outputs")

    # Curation parameters
    parser.add_argument(
        "--min-cell-size",
        type=int,
        default=50,
        help="Minimum size threshold for creating new cells",
    )
    parser.add_argument(
        "--max-small-cell-size",
        type=int,
        default=50,
        help="Maximum size for highlighting small cells",
    )

    # Processing options
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--create-visualizations",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Create HTML visualizations",
    )
    parser.add_argument(
        "--save-statistics",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Save curation statistics",
    )
    parser.add_argument(
        "--save-qc-images",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Save QC images",
    )

    args = parser.parse_args()

    # Convert string booleans to actual booleans
    args.verbose = args.verbose.lower() == "true"
    args.create_visualizations = args.create_visualizations.lower() == "true"
    args.save_statistics = args.save_statistics.lower() == "true"
    args.save_qc_images = args.save_qc_images.lower() == "true"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SEGMENTATION CURATION - Sample: {args.sample_name}")
    print("=" * 60)

    try:
        start_time = time.time()

        # Load original mask
        original_mask = load_mask(args.input_mask)

        # Initialize curator
        curator = SegmentationCurator(
            min_cell_size=args.min_cell_size,
            max_small_cell_size=args.max_small_cell_size,
            verbose=args.verbose,
        )

        # Run curation - the curator's existing methods handle the curation logic
        if args.verbose:
            print("Running segmentation curation...")

        # Run the complete curation pipeline
        curation_result = curator.curate_segmentation(
            original_mask,
            output_dir=output_dir,
            sample_name=args.sample_name,
            save_visualizations=args.create_visualizations,
        )

        # Extract the processed mask
        curated_mask = curation_result["processed_mask"]

        # Ensure curated_mask is a proper numpy array
        curated_mask = np.array(curated_mask)
        if args.verbose:
            print(
                f"Curated mask shape: {curated_mask.shape}, dtype: {curated_mask.dtype}"
            )
            print(f"Curation result keys: {list(curation_result.keys())}")

        end_time = time.time()
        processing_time = end_time - start_time

        # Compute statistics
        stats = compute_curation_statistics(original_mask, curated_mask)
        stats["processing_time_seconds"] = processing_time

        # Create QC visualization
        qc_fig = create_qc_visualization(
            original_mask, curated_mask, args.sample_name, stats, processing_time
        )

        # Save outputs
        output_files = save_outputs(
            curated_mask,
            stats,
            qc_fig,
            output_dir,
            args.sample_name,
            args.save_statistics,
            args.save_qc_images,
            args.create_visualizations,
        )

        print("\\n" + "=" * 60)
        print("SEGMENTATION CURATION COMPLETED SUCCESSFULLY")
        print(f"Sample: {args.sample_name}")
        print(f"Original cells: {stats['original_cell_count']}")
        print(f"Curated cells: {stats['curated_cell_count']}")
        print(
            f"Change: {stats['cells_changed']:+d} cells ({stats['change_percentage']:+.1f}%)"
        )
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\\nERROR: Curation failed for {args.sample_name}")
        print(f"Error details: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
