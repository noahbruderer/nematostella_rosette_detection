#!/usr/bin/env python3
"""
Universal Rosette Quantification Script
=======================================

Quantifies rosettes and cell properties from curated masks for various
experiment types (Time Series, Inhibitors). It accepts metadata via
command-line arguments and writes it directly into the output CSV.

Author: Noah Bruderer
Date: 2025
"""

import argparse
import os

import numpy as np
import pandas as pd
import tifffile
from helper_functions import (
    highlight_cells_with_rosettes_with_boundaries,
    identify_rosette_cells,
    label_cells_from_boundaries,
    measure_cell_properties_with_rosettes_and_neighbors,
    neigh_graph_pipeline,
)


def load_image(image_path):
    """Load image, ensuring it is returned as a 2D array."""
    try:
        img = tifffile.imread(image_path)
        if img.ndim > 2:
            img = np.squeeze(img)
            if img.ndim > 2:
                raise ValueError(f"Image has {img.ndim} dimensions; expecting 2D.")
        return img
    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")


def quantify_sample_rosettes(
    segmentation_path,
    rosette_mask_path,
    output_dir,
    sample_name,
    replicate,
    timepoint,
    animal,
    image,  # <-- ADDED 'image'
    inhibitor,
    is_control,
    min_rosette_size,
    max_rosette_size,
    normalization_factor,
):
    os.makedirs(output_dir, exist_ok=True)
    segmentation_mask = load_image(segmentation_path)
    rosette_mask = load_image(rosette_mask_path)
    labeled_rosettes = label_cells_from_boundaries(rosette_mask.astype(np.uint8))
    rosette_dict = identify_rosette_cells(segmentation_mask, labeled_rosettes)
    highlight_cells_with_rosettes_with_boundaries(
        segmentation_mask, rosette_dict, output_dir, sample_name
    )
    neigh_G = neigh_graph_pipeline(
        segmentation_mask, rosette_dict, output_dir, sample_name
    )
    cell_properties_df = measure_cell_properties_with_rosettes_and_neighbors(
        sample_name,
        output_dir,
        segmentation_mask,
        rosette_dict,
        neigh_G,
        intensity_image=None,
    )

    if not cell_properties_df.empty:
        cell_properties_df["replicate"] = replicate
        cell_properties_df["timepoint"] = timepoint.split("_")[0]
        cell_properties_df["animal"] = animal
        cell_properties_df["image"] = image  # <-- ADDED THIS LINE

        if inhibitor is not None and inhibitor != "None":
            cell_properties_df["inhibitor"] = inhibitor
            cell_properties_df["is_control"] = is_control == "True"

        output_path = os.path.join(output_dir, f"cell_properties_{sample_name}.csv")
        cell_properties_df.to_csv(output_path, index=False)
        print(f"✅ Final cell properties with all metadata saved to {output_path}")

    total_cells = len(cell_properties_df)
    total_rosettes = cell_properties_df["rosette_id"].nunique()
    rosettes_per_norm_factor = (
        (total_rosettes / total_cells) * normalization_factor if total_cells > 0 else 0
    )
    quantification_results = {
        "sample": sample_name,
        "total_cells": total_cells,
        "total_rosettes": total_rosettes,
        f"rosettes_per_{normalization_factor}_cells": rosettes_per_norm_factor,
    }
    results_df = pd.DataFrame([quantification_results])
    results_path = os.path.join(output_dir, f"quantification_{sample_name}.csv")
    results_df.to_csv(results_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Universal Rosette Quantification Script"
    )
    # Input/Output paths
    parser.add_argument(
        "--segmentation",
        type=str,
        required=True,
        help="Path to cell segmentation mask file",
    )
    parser.add_argument(
        "--rosettes", type=str, required=True, help="Path to curated rosette mask file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument("--sample", type=str, required=True, help="Sample name")
    # Core Metadata (Required for all experiments)
    parser.add_argument(
        "--replicate", type=str, required=True, help="Replicate identifier (e.g., R1)"
    )
    parser.add_argument(
        "--timepoint",
        type=str,
        required=True,
        help="Timepoint identifier (e.g., T0 or T0_seg)",
    )
    parser.add_argument("--animal", type=str, required=True, help="Animal identifier")
    parser.add_argument(
        "--image", type=str, required=True, help="Image identifier"
    )  # <-- ADDED THIS ARGUMENT
    # Inhibitor Metadata (Optional)
    parser.add_argument(
        "--inhibitor", type=str, default=None, help="Inhibitor name (if applicable)"
    )
    parser.add_argument(
        "--is-control",
        type=str,
        default="False",
        help="Whether sample is a control ('True'/'False')",
    )
    # Quantification Parameters
    parser.add_argument(
        "--min-rosette-size",
        type=int,
        default=10,
        help="Minimum rosette size in pixels",
    )
    parser.add_argument(
        "--max-rosette-size",
        type=int,
        default=1000,
        help="Maximum rosette size in pixels",
    )
    parser.add_argument(
        "--normalization-factor",
        type=int,
        default=1000,
        help="Normalization factor for rosette counts",
    )
    args = parser.parse_args()

    try:
        quantify_sample_rosettes(
            segmentation_path=args.segmentation,
            rosette_mask_path=args.rosettes,
            output_dir=args.output,
            sample_name=args.sample,
            replicate=args.replicate,
            timepoint=args.timepoint,
            animal=args.animal,
            image=args.image,  # <-- PASSING THE ARGUMENT
            inhibitor=args.inhibitor,
            is_control=args.is_control,
            min_rosette_size=args.min_rosette_size,
            max_rosette_size=args.max_rosette_size,
            normalization_factor=args.normalization_factor,
        )
        print(f"✅ Successfully processed sample: {args.sample}")
        return 0
    except Exception as e:
        print(f"❌ Failed to process sample: {args.sample}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
