import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage
from skimage.measure import label, regionprops


class SegmentationCurator:
    """
    A class to curate segmentation masks by handling small cells, background regions,
    and isolated cells through merging and creation operations.
    """

    def __init__(self, min_cell_size=50, max_small_cell_size=50, verbose=True):
        """
        Initialize the SegmentationCurator.

        Args:
            min_cell_size (int): Minimum size threshold for creating new cells from background regions
            max_small_cell_size (int): Maximum size for highlighting small cells in visualization
            verbose (bool): Whether to print processing information
        """
        self.min_cell_size = min_cell_size
        self.max_small_cell_size = max_small_cell_size
        self.verbose = verbose

    def _extract_2d_mask(self, mask):
        """
        Extract 2D mask from input (handles both 2D and 5D arrays).

        Args:
            mask: Input mask (2D or 5D numpy array)

        Returns:
            numpy.ndarray: 2D mask
        """
        if mask.ndim > 2:
            return mask[0, 0, 0]  # Extract first 2D slice from 5D
        return mask

    def highlight_small_cells(
        self, mask, output_dir=None, sample_name="sample", save_visualization=True
    ):
        """
        Visualize cells in a mask that are below the specified maximum size threshold.

        Args:
            mask: Input mask (2D or 5D numpy array)
            output_dir (str, optional): Directory to save visualization
            sample_name (str): Sample identifier for file naming
            save_visualization (bool): Whether to save the visualization

        Returns:
            dict: Statistics about the cells
        """
        mask_2d = self._extract_2d_mask(mask)

        # Calculate areas for each cell
        areas = []
        small_cells_mask = np.zeros_like(mask_2d)
        size_map = np.zeros_like(mask_2d, dtype=float)

        # Create background mask (where pixel value is 0)
        background_mask = (mask_2d == 0).astype(float)

        for val in np.unique(mask_2d):
            if val == 0:  # Skip background
                continue
            cell_mask = mask_2d == val
            area = np.sum(cell_mask)
            areas.append(area)

            # Mark cells below threshold
            if area <= self.max_small_cell_size:
                small_cells_mask[cell_mask] = 1

            # Create size map for visualization
            size_map[cell_mask] = area

        if save_visualization and output_dir:
            self._save_small_cells_visualization(
                size_map, small_cells_mask, background_mask, output_dir, sample_name
            )

        # Calculate statistics
        total_cells = len(areas)
        small_cells = np.sum(small_cells_mask > 0)
        stats = {
            "total_cells": total_cells,
            "small_cells": small_cells,
            "small_cell_percentage": (small_cells / total_cells * 100)
            if total_cells > 0
            else 0,
            "areas": areas,
        }

        if self.verbose:
            print(
                f"Sample {sample_name}: {total_cells} total cells, {small_cells} small cells ({stats['small_cell_percentage']:.1f}%)"
            )

        return stats

    def _save_small_cells_visualization(
        self, size_map, small_cells_mask, background_mask, output_dir, sample_name
    ):
        """Save the small cells visualization as HTML."""
        os.makedirs(output_dir, exist_ok=True)

        # Create figure with subplots
        fig = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "All Cells (colored by size)",
                f"Cells below {self.max_small_cell_size} pixels",
                "Combined Visualization",
                "Background Only",
            ),
            horizontal_spacing=0.08,
        )

        # Add size map (Plot 1)
        fig.add_trace(
            go.Heatmap(
                z=size_map,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Cell Size (pixels)",
                    x=0.19,
                    len=0.75,
                    thickness=15,
                    yanchor="middle",
                    y=0.5,
                    outlinewidth=1,
                    outlinecolor="black",
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Add small cells highlight (Plot 2)
        fig.add_trace(
            go.Heatmap(
                z=small_cells_mask,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,0,0,0.7)"]],
                showscale=False,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Add combined visualization (Plot 3)
        fig.add_trace(
            go.Heatmap(
                z=size_map, colorscale="Viridis", showscale=False, showlegend=False
            ),
            row=1,
            col=3,
        )

        # Add small cells overlay on combined plot
        fig.add_trace(
            go.Heatmap(
                z=small_cells_mask,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,0,0,0.7)"]],
                showscale=False,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        # Add background visualization (Plot 4)
        fig.add_trace(
            go.Heatmap(
                z=background_mask,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,0,0,0.7)"]],
                showscale=False,
                showlegend=False,
            ),
            row=1,
            col=4,
        )

        # Update layout
        fig.update_layout(
            title=f"Overview: {sample_name}\n(threshold: {self.max_small_cell_size} pixels)",
            height=500,
            width=2000,
            showlegend=False,
            margin=dict(t=100, b=50, l=50, r=100),
        )

        # Fix aspect ratios for all plots
        for col in [1, 2, 3, 4]:
            fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=col)
            fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=col)

        # Save figure as HTML
        fig.write_html(f"{output_dir}/raw_mask_vis_{sample_name}.html")

    def process_mask_edge_cases(
        self, mask, output_dir=None, sample_name="sample", save_visualization=True
    ):
        """
        Process mask edge cases: create new cells from large background regions,
        merge small background regions with neighboring cells, and remove isolated cells.

        Args:
            mask: Input mask (2D or 5D numpy array)
            output_dir (str, optional): Directory to save visualization
            sample_name (str): Sample identifier for file naming
            save_visualization (bool): Whether to save the visualization

        Returns:
            numpy.ndarray: Processed mask in same format as input
        """
        mask_2d = self._extract_2d_mask(mask).copy()
        original_mask = mask_2d.copy()

        # Initialize statistics and tracking arrays
        stats = {
            "new_cells_created": 0,
            "regions_merged": 0,
            "cells_removed": 0,
            "cells_before": len(np.unique(mask_2d)) - 1,
        }

        # Arrays to track changes
        new_cells_mask = np.zeros_like(mask_2d, dtype=bool)
        deleted_cells_mask = np.zeros_like(mask_2d, dtype=bool)
        merged_regions_mask = np.zeros_like(mask_2d, dtype=bool)

        def get_neighbors(region_mask):
            """Get neighboring cell IDs for a given region."""
            dilated = ndimage.binary_dilation(region_mask)
            neighbor_mask = dilated & ~region_mask
            neighbor_ids = np.unique(mask_2d[neighbor_mask])
            return [n for n in neighbor_ids if n != 0]

        # Step 1: Label background regions
        background_mask = mask_2d == 0
        background_regions = label(background_mask)
        next_cell_id = mask_2d.max() + 1

        # Dictionary to keep track of merged cell IDs
        merged_cell_ids = set()

        # Process background regions
        for region in regionprops(background_regions):
            region_mask = background_regions == region.label
            neighbors = get_neighbors(region_mask)

            # Skip if this is an edge background region (touches image boundary)
            if (
                region_mask[0, :].any()
                or region_mask[-1, :].any()
                or region_mask[:, 0].any()
                or region_mask[:, -1].any()
            ):
                continue

            if len(neighbors) > 0:
                if region.area >= self.min_cell_size:
                    # Create new cell
                    mask_2d[region_mask] = next_cell_id
                    new_cells_mask[region_mask] = True
                    next_cell_id += 1
                    stats["new_cells_created"] += 1
                else:
                    # Merge with largest neighbor
                    neighbor_sizes = [(n, np.sum(mask_2d == n)) for n in neighbors]
                    largest_neighbor = max(neighbor_sizes, key=lambda x: x[1])[0]
                    mask_2d[region_mask] = largest_neighbor
                    merged_regions_mask[region_mask] = True
                    # Mark the cell it's being merged with
                    merged_regions_mask[mask_2d == largest_neighbor] = True
                    merged_cell_ids.add(largest_neighbor)
                    stats["regions_merged"] += 1

        # Step 2: Process isolated cells (cells with no neighbors)
        cell_labels = np.unique(mask_2d)[1:]  # Exclude background (0)
        for cell_id in cell_labels:
            cell_mask = mask_2d == cell_id
            neighbors = get_neighbors(cell_mask)

            if len(neighbors) == 0:
                deleted_cells_mask[cell_mask] = True
                mask_2d[cell_mask] = 0
                stats["cells_removed"] += 1

        stats["cells_after"] = len(np.unique(mask_2d)) - 1

        if self.verbose:
            print(
                f"Sample {sample_name}: {stats['cells_before']} -> {stats['cells_after']} cells"
            )
            print(
                f"  Created: {stats['new_cells_created']}, Merged: {stats['regions_merged']}, Removed: {stats['cells_removed']}"
            )

        if save_visualization and output_dir:
            self._save_edge_cases_visualization(
                original_mask,
                mask_2d,
                new_cells_mask,
                merged_regions_mask,
                deleted_cells_mask,
                output_dir,
                sample_name,
            )

        # Return processed mask in same format as input
        if mask.ndim > 2:
            processed_mask = mask.copy()
            processed_mask[0, 0, 0] = mask_2d
            return processed_mask
        else:
            return mask_2d

    def _save_edge_cases_visualization(
        self,
        original_mask,
        processed_mask,
        new_cells_mask,
        merged_regions_mask,
        deleted_cells_mask,
        output_dir,
        sample_name,
    ):
        """Save the edge cases processing visualization as HTML."""
        os.makedirs(output_dir, exist_ok=True)

        # Create visualization
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Original Mask", "Processed Mask", "Changes Highlighted"),
            horizontal_spacing=0.05,
        )

        # Original mask
        fig.add_trace(
            go.Heatmap(
                z=original_mask,
                colorscale=[[0, "white"], [1, "gray"]],
                showscale=False,
                name="Original Cells",
            ),
            row=1,
            col=1,
        )

        # Processed mask
        fig.add_trace(
            go.Heatmap(
                z=processed_mask,
                colorscale=[[0, "white"], [1, "gray"]],
                showscale=False,
                name="Processed Cells",
            ),
            row=1,
            col=2,
        )

        # Base layer for changes visualization
        fig.add_trace(
            go.Heatmap(
                z=(processed_mask > 0).astype(float)
                * ~(new_cells_mask | merged_regions_mask | deleted_cells_mask),
                colorscale=[[0, "white"], [1, "lightgray"]],
                showscale=False,
                name="Unchanged Cells",
            ),
            row=1,
            col=3,
        )

        # Add new cells in green
        if np.any(new_cells_mask):
            fig.add_trace(
                go.Heatmap(
                    z=new_cells_mask.astype(float),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,255,0,0.7)"]],
                    showscale=False,
                    name="New Cells",
                ),
                row=1,
                col=3,
            )

        # Add merged regions in orange
        if np.any(merged_regions_mask):
            fig.add_trace(
                go.Heatmap(
                    z=merged_regions_mask.astype(float),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,165,0,0.7)"]],
                    showscale=False,
                    name="Merged Regions",
                ),
                row=1,
                col=3,
            )

        # Add deleted cells in red
        if np.any(deleted_cells_mask):
            fig.add_trace(
                go.Heatmap(
                    z=deleted_cells_mask.astype(float),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,0,0,0.7)"]],
                    showscale=False,
                    name="Deleted Cells",
                ),
                row=1,
                col=3,
            )

        # Update layout with legend
        fig.update_layout(
            title=f"Sample: {sample_name}\nMask Processing Results",
            height=600,
            width=1500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.8)",
            ),
        )

        # Fix aspect ratios and ranges
        for col in [1, 2, 3]:
            fig.update_xaxes(
                scaleanchor="y", scaleratio=1, row=1, col=col, constrain="domain"
            )
            fig.update_yaxes(
                scaleanchor="x", scaleratio=1, row=1, col=col, constrain="domain"
            )

        # Save figure as HTML
        fig.write_html(f"{output_dir}/processed_mask_vis_{sample_name}.html")

    def analyze_and_compare_masks(
        self,
        original_mask,
        processed_mask,
        output_dir=None,
        sample_name="sample",
        save_visualization=True,
    ):
        """
        Analyze the area distribution of cells in the original and processed masks.

        Args:
            original_mask: Original mask before processing
            processed_mask: Processed mask after curation
            output_dir (str, optional): Directory to save visualization
            sample_name (str): Sample identifier for file naming
            save_visualization (bool): Whether to save the visualization

        Returns:
            tuple: (original_areas, processed_areas) - numpy arrays of cell areas
        """
        # Extract 2D masks
        mask_2d_original = self._extract_2d_mask(original_mask)
        mask_2d_processed = self._extract_2d_mask(processed_mask)

        def get_cell_areas(mask):
            """Get areas of all cells in a mask."""
            areas = []
            for val in np.unique(mask):
                if val == 0:  # Skip background
                    continue
                cell_area = np.sum(mask == val)
                areas.append(cell_area)
            return np.array(areas)

        # Calculate areas
        original_areas = get_cell_areas(mask_2d_original)
        processed_areas = get_cell_areas(mask_2d_processed)

        if self.verbose:
            print(f"Sample {sample_name} area analysis:")
            print(
                f"  Original: {len(original_areas)} cells, mean area: {np.mean(original_areas):.1f}"
            )
            print(
                f"  Processed: {len(processed_areas)} cells, mean area: {np.mean(processed_areas):.1f}"
            )

        if save_visualization and output_dir:
            self._save_comparison_visualization(
                original_areas, processed_areas, output_dir, sample_name
            )

        return original_areas, processed_areas

    def _save_comparison_visualization(
        self, original_areas, processed_areas, output_dir, sample_name
    ):
        """Save the area comparison visualization as HTML."""
        os.makedirs(output_dir, exist_ok=True)

        # Create visualization
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Overlay of Area Distributions (Histogram)",
                "Box Plot Comparison of Cell Areas",
            ),
            vertical_spacing=0.2,
        )

        # Overlay histogram
        fig.add_trace(
            go.Histogram(
                x=original_areas,
                name="Original Mask",
                opacity=0.6,
                marker=dict(color="blue"),
                nbinsx=50,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=processed_areas,
                name="Processed Mask",
                opacity=0.6,
                marker=dict(color="red"),
                nbinsx=50,
            ),
            row=1,
            col=1,
        )

        # Box plot comparison
        fig.add_trace(
            go.Box(
                y=original_areas,
                name="Original Mask",
                marker=dict(color="blue"),
                boxpoints="outliers",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Box(
                y=processed_areas,
                name="Processed Mask",
                marker=dict(color="red"),
                boxpoints="outliers",
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=800,
            title=f"Sample: {sample_name}\nCell Area Distribution Analysis and Comparison",
            showlegend=True,
        )
        fig.update_xaxes(title="Cell Area", row=1, col=1)
        fig.update_yaxes(title="Frequency", row=1, col=1)
        fig.update_yaxes(title="Area (pixels)", row=2, col=1)

        # Save figure as HTML
        fig.write_html(
            f"{output_dir}/cell_area_distribution_comparison_{sample_name}.html"
        )

    def curate_segmentation(
        self, mask, output_dir=None, sample_name="sample", save_visualizations=True
    ):
        """
        Complete segmentation curation pipeline.

        Args:
            mask: Input segmentation mask (2D or 5D numpy array)
            output_dir (str, optional): Directory to save visualizations
            sample_name (str): Sample identifier for file naming
            save_visualizations (bool): Whether to save visualization plots

        Returns:
            dict: Dictionary containing processed mask and statistics
        """
        if self.verbose:
            print(f"Starting segmentation curation for {sample_name}")

        # Step 1: Analyze original mask
        original_stats = self.highlight_small_cells(
            mask, output_dir, sample_name, save_visualizations
        )

        # Step 2: Process edge cases (merge small regions, create new cells, remove isolated cells)
        processed_mask = self.process_mask_edge_cases(
            mask, output_dir, sample_name, save_visualizations
        )

        # Step 3: Compare original and processed masks
        original_areas, processed_areas = self.analyze_and_compare_masks(
            mask, processed_mask, output_dir, sample_name, save_visualizations
        )

        if self.verbose:
            print(f"Completed segmentation curation for {sample_name}")

        return {
            "processed_mask": processed_mask,
            "original_stats": original_stats,
            "original_areas": original_areas,
            "processed_areas": processed_areas,
            "sample_name": sample_name,
        }
