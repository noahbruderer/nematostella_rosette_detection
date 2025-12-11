#!/usr/bin/env python3
"""
Quick H5 Data Visualization Script (Updated)
============================================

This script loads and visualizes processed H5 data, showing each
dataset individually.

Usage:
    python h5_visualizer_updated.py
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb


def load_and_visualize_h5(h5_path):
    """Load and visualize H5 data with individually named datasets."""
    print(f"Loading H5 file: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        print(f"Available datasets: {list(f.keys())}")

        # --- FIX: Load each dataset by its specific key ---
        raw_image = f["raw_data"][:]
        outlines = f["segmentation_outlines"][:]
        rosette_binary = f["rosettes_binary"][:]
        rosette_individual = f["rosettes_individual"][:]
        # --- END FIX ---

    # --- Print statistics for verification ---
    print("\nDataset Statistics:")
    print(
        f"Raw Image - Shape: {raw_image.shape}, Type: {raw_image.dtype}, Range: [{raw_image.min()}, {raw_image.max()}]"
    )
    print(
        f"Outlines - Shape: {outlines.shape}, Type: {outlines.dtype}, Unique values: {np.unique(outlines)}"
    )
    print(
        f"Rosette Binary - Shape: {rosette_binary.shape}, Type: {rosette_binary.dtype}, Rosette Pixels: {rosette_binary.sum()}"
    )
    print(
        f"Rosette Individual - Shape: {rosette_individual.shape}, Type: {rosette_individual.dtype}, Rosettes Found: {len(np.unique(rosette_individual)) - 1}"
    )

    # --- Create visualization in a 2x2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"H5 Data Visualization: {Path(h5_path).stem}", fontsize=16)

    # 1. Raw Data
    axes[0, 0].imshow(raw_image, cmap="gray")
    axes[0, 0].set_title("raw_data")
    axes[0, 0].axis("off")

    # 2. Segmentation Outlines
    axes[0, 1].imshow(outlines, cmap="binary")
    axes[0, 1].set_title("segmentation_outlines")
    axes[0, 1].axis("off")

    # 3. Rosettes Binary
    axes[1, 0].imshow(rosette_binary, cmap="Reds")
    axes[1, 0].set_title("rosettes_binary")
    axes[1, 0].axis("off")

    # 4. Rosettes Individual (visualized with random colors)
    # Use label2rgb for better visualization of instance masks
    if rosette_individual.max() > 0:
        colored_rosettes = label2rgb(rosette_individual, bg_label=0, bg_color=(0, 0, 0))
        axes[1, 1].imshow(colored_rosettes)
    else:
        axes[1, 1].imshow(rosette_individual, cmap="gray")  # Show black if no rosettes

    axes[1, 1].set_title(
        f"rosettes_individual ({len(np.unique(rosette_individual)) - 1} rosettes)"
    )
    axes[1, 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle

    # Save the figure
    output_path = Path(h5_path).parent / f"{Path(h5_path).stem}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


def main():
    """Main function to run the visualization."""
    # --- IMPORTANT: UPDATE THIS PATH to your H5 file ---
    h5_path = "/Users/noahbruderer/local_work_files/rosette_paper/results/training_data_preparation/R2_T4_17_0001/processed_data_R2_T4_17_0001.h5"

    if not Path(h5_path).exists():
        print(f"ERROR: H5 file not found at {h5_path}")
        return

    try:
        load_and_visualize_h5(h5_path)
        print("✅ Visualization completed successfully!")
    except KeyError as e:
        print(f"❌ Error: A required dataset was not found in the H5 file: {e}")
        print(
            "Please ensure the H5 file was created with the new script and contains all four datasets."
        )
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
