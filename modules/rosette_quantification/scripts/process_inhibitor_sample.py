#!/usr/bin/env python3
"""
Rosette Inhibitor Sample Processing Script
==========================================

Processes rosette inhibitor experiment samples with metadata extraction.
Adapted for Snakemake pipeline integration.

Author: Noah Bruderer
Date: 2025
"""

import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Import the existing helper functions
from helper_functions import (
    identify_rosette_cells,
    label_cells_from_boundaries,
    measure_cell_properties_with_rosettes_and_neighbors,
    neigh_graph_pipeline,
)


def load_image(image_path):
    """Load image, ensuring it is returned as a 2D or 3D array."""
    try:
        img = tifffile.imread(image_path)
        
        if img.ndim > 3:
            # If more than 3D, take the first slice along extra dimensions
            while img.ndim > 3:
                img = img[0]
        
        print(f"Loaded image: {img.shape}, dtype: {img.dtype}")
        return img
        
    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")


def parse_metadata_from_path(sample_dir):
    """
    Parse metadata from the folder name.
    Examples:
    - R1_T5_ZVAD_dmso_1 (replicate 1, timepoint 5, inhibitor ZVAD, control, animal 1, image 0)
    - R1_T5_ZVAD_dmso_2_0001 (replicate 1, timepoint 5, inhibitor ZVAD, control, animal 2, image 1)
    """
    # Split by underscore
    parts = sample_dir.split("_")

    # Extract information
    replicate = None
    timepoint = None
    inhibitor_name = None
    is_control = False
    animal_id = None
    image_id = 0  # Default to 0 if not specified

    for i, part in enumerate(parts):
        if part.startswith("R"):
            replicate = int(part[1:])
        elif part.startswith("T"):
            timepoint = int(part[1:])
        elif part.lower() == "dmso":
            is_control = True
        elif i == 2 and not part.startswith(("R", "T")):  # Assuming inhibitor is in position 2
            inhibitor_name = part

    # Find animal_id and image_id
    if len(parts) >= 5 and parts[-2].isdigit() and parts[-1].isdigit():
        animal_id = int(parts[-2])
        image_id = int(parts[-1])
    # If only one number at the end, it's the animal_id
    elif parts[-1].isdigit():
        animal_id = int(parts[-1])

    # Check for DMSO in any position
    if "dmso" in [p.lower() for p in parts]:
        is_control = True

    return {
        "Replicate": replicate,
        "Timepoint": timepoint,
        "Inhibitor": inhibitor_name,
        "IsControl": is_control,
        "Animal": animal_id,
        "Image": image_id,
    }


def process_inhibitor_sample(
    segmentation_path,
    rosette_curated_path,
    output_dir,
    sample_name,
    raw_image_path=None,
    rosette_predicted_path=None
):
    """
    Process a single inhibitor experiment sample.
    
    Args:
        segmentation_path: Path to segmentation mask file
        rosette_curated_path: Path to curated rosette mask file
        output_dir: Output directory for results
        sample_name: Name of the sample
        raw_image_path: Optional path to raw image
        rosette_predicted_path: Optional path to predicted rosettes
    
    Returns:
        dict: Processing results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    timing_summary = {}
    pipeline_start_time = time.time()
    
    print(f"Processing inhibitor sample: {sample_name}")
    print(f"Segmentation: {segmentation_path}")
    print(f"Rosette Curated: {rosette_curated_path}")
    
    try:
        # Step 1: Load segmentation and rosette images
        start_time = time.time()
        processed_mask = load_image(segmentation_path)
        rosette_curated = load_image(rosette_curated_path)
        timing_summary["Load Images"] = time.time() - start_time
        
        # Step 2: Label rosette cells from the rosette mask
        start_time = time.time()
        labeled_cells_rosettes = label_cells_from_boundaries(rosette_curated)
        timing_summary["Label Rosette Instances"] = time.time() - start_time
        
        # Step 3: Identify rosette cells in the processed mask using the rosette labels
        start_time = time.time()
        rosette_dict = identify_rosette_cells(processed_mask, labeled_cells_rosettes)
        timing_summary["Identify Rosette Cells"] = time.time() - start_time
        
        # Step 4: Generate neighborhood graph
        start_time = time.time()
        neigh_G = neigh_graph_pipeline(processed_mask, rosette_dict, output_dir, sample_name)
        timing_summary["Neighborhood Graph Pipeline"] = time.time() - start_time
        
        # Step 5: Measure cell properties
        start_time = time.time()
        
        # Handle different dimensionalities for cell properties measurement
        mask_2d = processed_mask
        if processed_mask.ndim > 2:
            mask_2d = processed_mask[0, 0, 0] if processed_mask.ndim == 5 else processed_mask[0]
        
        cell_props = measure_cell_properties_with_rosettes_and_neighbors(
            sample_name,
            output_dir,
            mask_2d,
            rosette_dict,
            neigh_G,
            intensity_image=None,
        )
        timing_summary["Cell Measurements"] = time.time() - start_time
        
        # Step 6: Add metadata and metrics
        start_time = time.time()
        if cell_props is not None and not cell_props.empty:
            # Add metadata columns
            metadata = parse_metadata_from_path(sample_name)
            for key, value in metadata.items():
                cell_props[key] = value

            # Calculate metrics
            total_cells = len(np.unique(mask_2d)) - 1  # Subtract background
            num_rosettes = len(rosette_dict)
            cells_in_rosettes = sum(len(cells) for cells in rosette_dict.values())
            rosettes_per_100_cells = (num_rosettes / total_cells * 100) if total_cells > 0 else 0
            rosette_percentage = (cells_in_rosettes / total_cells * 100) if total_cells > 0 else 0

            # Add metrics as columns
            cell_props["TotalCells"] = total_cells
            cell_props["NumRosettes"] = num_rosettes
            cell_props["CellsInRosettes"] = cells_in_rosettes
            cell_props["RosettePercentage"] = rosette_percentage
            cell_props["RosettesPerHundredCells"] = rosettes_per_100_cells

            # Add file paths to the CSV
            cell_props["segmentation_path"] = segmentation_path
            cell_props["rosette_curated_path"] = rosette_curated_path
            if raw_image_path:
                cell_props["raw_image_path"] = raw_image_path
            if rosette_predicted_path:
                cell_props["rosette_predicted_path"] = rosette_predicted_path

            # Save the updated properties
            enhanced_props_path = os.path.join(output_dir, f"{sample_name}_cell_properties_with_metadata.csv")
            cell_props.to_csv(enhanced_props_path, index=False)
            
            # Create summary results
            summary_results = {
                'sample': sample_name,
                'total_cells': total_cells,
                'num_rosettes': num_rosettes,
                'cells_in_rosettes': cells_in_rosettes,
                'rosette_percentage': rosette_percentage,
                'rosettes_per_100_cells': rosettes_per_100_cells,
                'processing_time_seconds': time.time() - pipeline_start_time,
                **metadata  # Include all metadata
            }
            
            # Save summary
            summary_df = pd.DataFrame([summary_results])
            summary_path = os.path.join(output_dir, f"{sample_name}_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
        timing_summary["Add Metadata and Metrics"] = time.time() - start_time
        
        # Save timing information
        timing_df = pd.DataFrame([{
            'sample': sample_name,
            'step': step,
            'time_seconds': time_val
        } for step, time_val in timing_summary.items()])
        timing_path = os.path.join(output_dir, f"{sample_name}_timing.csv")
        timing_df.to_csv(timing_path, index=False)
        
        print(f"\nInhibitor Sample Results for {sample_name}:")
        print(f"  Total cells: {total_cells}")
        print(f"  Number of rosettes: {num_rosettes}")
        print(f"  Cells in rosettes: {cells_in_rosettes}")
        print(f"  Rosette percentage: {rosette_percentage:.2f}%")
        print(f"  Rosettes per 100 cells: {rosettes_per_100_cells:.2f}")
        print(f"  Processing time: {time.time() - pipeline_start_time:.2f} seconds")
        
        return summary_results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing {sample_name}: {str(e)}\n{error_details}")
        
        # Save error information
        error_results = {
            'sample': sample_name,
            'total_cells': 0,
            'num_rosettes': 0,
            'cells_in_rosettes': 0,
            'rosette_percentage': 0,
            'rosettes_per_100_cells': 0,
            'processing_time_seconds': time.time() - pipeline_start_time,
            'error': str(e)
        }
        
        error_df = pd.DataFrame([error_results])
        error_path = os.path.join(output_dir, f"{sample_name}_summary.csv")
        error_df.to_csv(error_path, index=False)
        
        raise


def main():
    """Main function with command-line interface for Snakemake integration."""
    parser = argparse.ArgumentParser(
        description="Process inhibitor experiment samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python process_inhibitor_sample.py \
    --segmentation path/to/sample_segmentation.tif \
    --rosettes path/to/sample_rosettes_curated.tif \
    --output results/sample \
    --sample sample_name
        """
    )
    
    parser.add_argument("--segmentation", type=str, required=True, help="Path to segmentation mask file")
    parser.add_argument("--rosettes", type=str, required=True, help="Path to curated rosette mask file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--sample", type=str, required=True, help="Sample name")
    parser.add_argument("--raw-image", type=str, help="Optional path to raw image file")
    parser.add_argument("--predicted-rosettes", type=str, help="Optional path to predicted rosettes file")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.segmentation):
        print(f"Error: Segmentation file does not exist: {args.segmentation}")
        return 1
    if not os.path.exists(args.rosettes):
        print(f"Error: Rosette mask file does not exist: {args.rosettes}")
        return 1
    
    try:
        results = process_inhibitor_sample(
            segmentation_path=args.segmentation,
            rosette_curated_path=args.rosettes,
            output_dir=args.output,
            sample_name=args.sample,
            raw_image_path=args.raw_image,
            rosette_predicted_path=args.predicted_rosettes
        )
        print(f"✅ Successfully processed inhibitor sample: {args.sample}")
        return 0
    except Exception as e:
        print(f"❌ Failed to process inhibitor sample: {args.sample}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())