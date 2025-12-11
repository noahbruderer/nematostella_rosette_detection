#!/usr/bin/env python3
"""
Script to plot cell outlines for a specific sample.
Loads segmentation data and creates a visualization showing only cell boundaries.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from skimage import segmentation


def load_sample_data(h5_file_path):
    """
    Load data from the processed h5 file.

    Args:
        h5_file_path (str): Path to the h5 file

    Returns:
        dict: Dictionary containing loaded data
    """
    h5_file_path = Path(h5_file_path)

    if not h5_file_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_file_path}")

    data = {}

    def explore_h5_structure(group, path=""):
        """Recursively explore h5 structure and load datasets"""
        for key in group.keys():
            full_path = f"{path}/{key}" if path else key
            item = group[key]

            if isinstance(item, h5py.Dataset):
                print(f"  - {full_path}: {item.shape} ({item.dtype})")
                data[full_path] = item[...]
            elif isinstance(item, h5py.Group):
                print(f"  - {full_path}: Group")
                explore_h5_structure(item, full_path)

    with h5py.File(h5_file_path, "r") as f:
        print(f"Exploring structure of {h5_file_path.name}:")
        explore_h5_structure(f)

    return data


def extract_mask_from_data(data):
    """
    Extract the segmentation mask from loaded data.

    Args:
        data (dict): Dictionary containing loaded datasets

    Returns:
        numpy.ndarray: 2D segmentation mask
    """
    # Try common mask field names (including path-based keys)
    possible_mask_keys = [
        "mask",
        "segmentation",
        "processed_mask",
        "labels",
        "cells",
        "segmentation/mask",
        "segmentation/processed_mask",
        "segmentation/labels",
        "processed_data/mask",
        "processed_data/segmentation",
    ]

    mask = None
    for key in possible_mask_keys:
        if key in data:
            mask = data[key]
            print(f"Found mask dataset: '{key}'")
            break

    # If no direct mask found, try datasets that might be masks
    if mask is None:
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                # Check if it looks like a segmentation mask (integer values)
                if np.issubdtype(value.dtype, np.integer):
                    mask = value
                    print(f"Using dataset '{key}' as mask (integer type)")
                    break

    # If still no mask, try any 2D+ array
    if mask is None:
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                mask = value
                print(f"Using dataset '{key}' as mask (fallback)")
                break

    if mask is None:
        print("Available datasets:")
        for key, value in data.items():
            print(f"  {key}: {type(value)} {getattr(value, 'shape', 'no shape')}")
        raise ValueError("Could not find a suitable mask dataset in the h5 file")

    # Extract 2D mask if it's multi-dimensional
    if mask.ndim > 2:
        # Take the first 2D slice
        mask = mask[0, 0, 0] if mask.ndim == 5 else mask[0]

    return mask


def plot_cell_outlines(mask, sample_name, save_path=None, show_plot=True):
    """
    Plot cell outlines arranged for A4 printing with two A5-sized squares.

    Args:
        mask (numpy.ndarray): 2D segmentation mask
        sample_name (str): Name of the sample for title
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    # Find cell boundaries
    boundaries = segmentation.find_boundaries(mask, mode="thick")

    # Create outline image with white background and black boundaries
    outline_image = np.ones_like(mask, dtype=float)
    outline_image[boundaries] = 0  # Black lines for boundaries

    # A4 size in inches (11.69 x 8.27 inches) - landscape orientation
    fig = plt.figure(figsize=(11.69, 8.27))

    # A5 dimensions: 5.83 x 8.27 inches
    # A4 can fit two A5s side by side: 5.83 * 2 = 11.66 inches (fits in 11.69)

    # Calculate positions for two A5-sized squares
    # Each A5 area should be 5.83/11.69 = 0.499 of the total width
    a5_width = 5.83 / 11.69  # 0.499
    a5_height = 8.27 / 8.27  # 1.0 (full height)

    # Position first A5 area (left side)
    ax1 = fig.add_axes([0.0, 0.0, a5_width, a5_height])
    ax1.imshow(outline_image, cmap="gray", interpolation="nearest")
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Add border
    for spine in ax1.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)

    # Position second A5 area (right side)
    ax2 = fig.add_axes([0.5, 0.0, a5_width, a5_height])
    ax2.imshow(outline_image, cmap="gray", interpolation="nearest")
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Add border
    for spine in ax2.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)

    if save_path:
        # Save at A4 dimensions with high DPI for crisp printing
        plt.savefig(
            save_path,
            dpi=600,
            bbox_inches=None,
            pad_inches=0,
            facecolor="white",
            edgecolor="none",
        )
        print(f"A4 double-layout saved to: {save_path} (600 DPI, 11.69x8.27 inches)")

    if show_plot:
        print("placeholder")
        # plt.show()

    return fig


def main():
    """Main function to run the cell outline plotting."""
    # Use the specific file path you provided
    h5_file_path = "/Users/noahbruderer/local_work_files/rosette_id/data_preprocessing/processed_data_2/R2_T5_17/R2_T5_17_processed_data.h5"
    sample_name = "R2_T0_5_0001"

    try:
        # Load data
        print(f"Loading data from: {h5_file_path}")
        data = load_sample_data(h5_file_path)

        # Extract mask
        mask = extract_mask_from_data(data)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask value range: {mask.min()} - {mask.max()}")
        print(f"Number of unique labels: {len(np.unique(mask))}")

        # Prepare save path
        save_path = Path(h5_file_path).parent / f"{sample_name}_cell_outlines.png"

        # Plot cell outlines
        fig = plot_cell_outlines(mask, sample_name, save_path=save_path, show_plot=True)

        print("Cell outline plotting completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
