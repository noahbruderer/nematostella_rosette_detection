#!/usr/bin/env python3
"""
Visualize training data from H5 files to inspect masks and data quality.
"""

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
from skimage.segmentation import find_boundaries


def load_sample_data(h5_path):
    """Load data from H5 file."""
    with h5py.File(h5_path, "r") as f:
        # Load the datasets
        cell_mask = np.squeeze(f["mask"][:])
        rosette_mask = np.squeeze(f["rosette_mask"][:])
        raw_image = np.squeeze(f["raw_image"][:]) if "raw_image" in f else np.zeros_like(cell_mask)
        
        # Get metadata
        sample_name = f.attrs.get("sample_name", "Unknown")
        num_cells = f.attrs.get("num_cells", 0)
        num_rosettes = f.attrs.get("num_rosettes", 0)
        num_patches = f.attrs.get("num_patches", 0)
        
        # Load patches if they exist
        patches = None
        patch_labels = None
        if "patches" in f:
            patches = f["patches"][:]
            patch_labels = f["patch_labels"][:]
        
        metadata = {
            "sample_name": sample_name,
            "num_cells": num_cells,
            "num_rosettes": num_rosettes,
            "num_patches": num_patches,
            "shape": cell_mask.shape
        }
        
    return cell_mask, rosette_mask, raw_image, patches, patch_labels, metadata


def visualize_full_image(cell_mask, rosette_mask, raw_image, metadata, save_path=None):
    """Create visualization of the full image with all masks."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Create binary masks
    rosette_binary = (rosette_mask > 0).astype(float)
    cell_binary = (cell_mask > 0).astype(float)
    cell_boundaries = find_boundaries(cell_mask, mode="thick").astype(float)
    
    # Plot raw image
    axes[0, 0].imshow(raw_image, cmap='gray')
    axes[0, 0].set_title(f'Raw Image\n{metadata["sample_name"]}')
    axes[0, 0].axis('off')
    
    # Plot cell instance mask
    axes[0, 1].imshow(cell_mask, cmap='nipy_spectral')
    axes[0, 1].set_title(f'Cell Instance Mask\n{metadata["num_cells"]} cells')
    axes[0, 1].axis('off')
    
    # Plot rosette instance mask
    axes[0, 2].imshow(rosette_mask, cmap='tab10')
    axes[0, 2].set_title(f'Rosette Instance Mask\n{metadata["num_rosettes"]} rosettes')
    axes[0, 2].axis('off')
    
    # Plot binary masks used for training
    axes[1, 0].imshow(cell_binary, cmap='gray')
    axes[1, 0].set_title('Cell Binary Mask\n(Training Input)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cell_boundaries, cmap='gray')
    axes[1, 1].set_title('Cell Boundaries\n(Training Input)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(rosette_binary, cmap='Reds')
    axes[1, 2].set_title('Rosette Binary Mask\n(Training Target)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_patches(patches, patch_labels, metadata, max_patches=16, save_path=None):
    """Visualize training patches with their labels."""
    if patches is None or len(patches) == 0:
        print("No patches found in this sample")
        return None
    
    n_patches = min(len(patches), max_patches)
    cols = 4
    rows = (n_patches + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Sample some patches (mix of positive and negative)
    positive_indices = np.where(patch_labels > 0)[0]
    negative_indices = np.where(patch_labels == 0)[0]
    
    # Try to get equal mix of positive and negative
    n_pos = min(len(positive_indices), n_patches // 2)
    n_neg = n_patches - n_pos
    
    selected_indices = []
    if n_pos > 0:
        selected_indices.extend(np.random.choice(positive_indices, n_pos, replace=False))
    if n_neg > 0 and len(negative_indices) > 0:
        selected_indices.extend(np.random.choice(negative_indices, min(n_neg, len(negative_indices)), replace=False))
    
    # Fill remaining with random patches
    while len(selected_indices) < n_patches:
        remaining = list(set(range(len(patches))) - set(selected_indices))
        if remaining:
            selected_indices.append(random.choice(remaining))
        else:
            break
    
    for i in range(rows * cols):
        row, col = i // cols, i % cols
        
        if i < len(selected_indices):
            idx = selected_indices[i]
            patch = patches[idx]
            label = patch_labels[idx]
            
            axes[row, col].imshow(patch, cmap='nipy_spectral')
            axes[row, col].set_title(f'Patch {idx}\n{"Rosette" if label else "Non-rosette"}', 
                                   color='red' if label else 'blue', fontweight='bold')
        else:
            axes[row, col].axis('off')
        
        axes[row, col].axis('off')
    
    plt.suptitle(f'Training Patches from {metadata["sample_name"]}\n'
                 f'Total: {len(patches)} patches, '
                 f'Positive: {np.sum(patch_labels)}, '
                 f'Negative: {len(patch_labels) - np.sum(patch_labels)}', 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved patch visualization to {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize training data from H5 files')
    parser.add_argument('--data-dir', default='../../results/training_data_preparation',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--max-samples', type=int, default=5,
                       help='Maximum number of samples to visualize')
    parser.add_argument('--max-patches', type=int, default=16,
                       help='Maximum number of patches to show per sample')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find H5 files
    data_dir = Path(args.data_dir)
    h5_files = list(data_dir.glob("*/processed_data_*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in {data_dir}")
        return
    
    print(f"Found {len(h5_files)} H5 files")
    
    # Sample random files to visualize
    sample_files = random.sample(h5_files, min(len(h5_files), args.max_samples))
    
    for i, h5_path in enumerate(sample_files):
        print(f"\nProcessing {h5_path.name}...")
        
        try:
            cell_mask, rosette_mask, raw_image, patches, patch_labels, metadata = load_sample_data(h5_path)
            
            print(f"Sample: {metadata['sample_name']}")
            print(f"Shape: {metadata['shape']}")
            print(f"Cells: {metadata['num_cells']}")
            print(f"Rosettes: {metadata['num_rosettes']}")
            print(f"Patches: {metadata['num_patches']}")
            
            # Create full image visualization
            sample_name = metadata['sample_name']
            full_viz_path = output_dir / f"{sample_name}_full_image.png"
            fig1 = visualize_full_image(cell_mask, rosette_mask, raw_image, metadata, full_viz_path)
            plt.close(fig1)
            
            # Create patch visualization
            if patches is not None and len(patches) > 0:
                patch_viz_path = output_dir / f"{sample_name}_patches.png"
                fig2 = visualize_patches(patches, patch_labels, metadata, args.max_patches, patch_viz_path)
                if fig2:
                    plt.close(fig2)
            else:
                print(f"No patches found for {sample_name}")
                
        except Exception as e:
            print(f"Error processing {h5_path}: {e}")
            continue
    
    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()