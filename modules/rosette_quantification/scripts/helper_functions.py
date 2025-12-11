#!/usr/bin/env python3
"""
Helper Functions for Rosette Quantification
==========================================

Contains all helper functions needed for the rosette quantification pipeline.
"""

import os

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops_table
from skimage.segmentation import find_boundaries


def highlight_cells_with_rosettes_with_boundaries(
    mask, rosette_dict, output_dir, sample
):
    """
    Visualize all cells colored by size, with a second plot highlighting rosette cells in red
    and adding black boundaries around all cells.

    Args:
        mask: Input mask (2D or 5D numpy array; if 5D, uses the first slice)
        rosette_dict: Dictionary mapping rosette IDs to sets of cell IDs
        output_dir: Directory to save visualization
        sample: Sample name

    Returns:
        dict: Statistics about the cells
    """
    # Handle 5D input
    if mask.ndim > 2:
        mask_2d = mask[0, 0, 0]  # Extract first 2D slice
    else:
        mask_2d = mask

    # Calculate areas for each cell
    size_map = np.zeros_like(mask_2d, dtype=float)
    for val in np.unique(mask_2d):
        if val == 0:  # Skip background
            continue
        cell_mask = mask_2d == val
        area = np.sum(cell_mask)
        size_map[cell_mask] = area

    # Create a binary mask for rosettes
    rosette_highlight = np.zeros_like(mask_2d, dtype=bool)
    for cells in rosette_dict.values():
        for cell_id in cells:
            rosette_highlight[mask_2d == cell_id] = True

    # Compute cell boundaries
    boundaries = find_boundaries(mask_2d, mode="outer")

    # Create the figure
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "All Cells (colored by size)",
            "All Cells with Rosettes Highlighted",
        ),
        horizontal_spacing=0.1,  # Adjust spacing between plots
    )

    # Plot 1: All Cells Colored by Size
    fig.add_trace(
        go.Heatmap(
            z=size_map,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Cell Size (pixels)",
                len=0.8,
                thickness=15,
                yanchor="middle",
                y=0.5,
                outlinewidth=1,
                outlinecolor="black",
            ),
            name="Cell Size",
        ),
        row=1,
        col=1,
    )

    # Add cell boundaries to Plot 1
    fig.add_trace(
        go.Heatmap(
            z=boundaries.astype(int),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "black"]],  # Transparent to black
            showscale=False,
            name="Boundaries",
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    # Plot 2: All Cells with Rosettes Highlighted
    fig.add_trace(
        go.Heatmap(z=size_map, colorscale="Viridis", showscale=False, name="Cell Size"),
        row=1,
        col=2,
    )

    # Add rosettes in red to Plot 2
    fig.add_trace(
        go.Heatmap(
            z=rosette_highlight.astype(int),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,0,0,0.7)"]],
            showscale=False,
            name="Rosette Highlight",
        ),
        row=1,
        col=2,
    )

    # Add cell boundaries to Plot 2
    fig.add_trace(
        go.Heatmap(
            z=boundaries.astype(int),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "black"]],  # Transparent to black
            showscale=False,
            name="Boundaries",
            opacity=0.8,
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=f"Sample: {sample}\n Cell Analysis with Rosettes Highlighted and Boundaries",
        height=600,
        width=1400,
        margin=dict(t=100, b=50, l=50, r=50),
        showlegend=False,
    )

    # Fix aspect ratios for both plots
    for col in [1, 2]:
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=col)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=col)

    # Show the figure
    # Save figure as HTML
    fig.write_html(f"{output_dir}/rosette_highlight_vis_{sample}.html")
    print(
        f"\033[92mRosette highlight visualization saved as {output_dir}/rosette_highlight_vis_{sample}.html\033[0m"
    )
    return


def identify_rosette_cells(mask_o, rosettes):
    """
    Identify all cells fully enclosed by the outer boundary of rosettes,
    ignoring any internal boundaries in the rosette mask.

    Parameters:
    -----------
    mask_o : numpy.ndarray
        Original mask with individual cell labels
    rosettes : numpy.ndarray
        Mask with rosette regions labeled with distinct IDs

    Returns:
    --------
    rosette_dict : dict
        Dictionary mapping rosette IDs to sets of cell IDs
    """
    # Handle different dimensionalities
    if mask_o.ndim > 2:
        mask = mask_o[0, 0, 0]  # Extract 2D slice
    else:
        mask = mask_o

    if rosettes.ndim > 2:
        rosettes = rosettes[0, 0, 0]  # Extract 2D slice

    rosette_dict = {}

    # Get unique rosette labels (excluding background 0)
    rosette_labels = np.unique(rosettes)
    rosette_labels = rosette_labels[rosette_labels != 0]

    # Iterate over each rosette boundary
    for rosette_id in rosette_labels:
        # Create binary mask for this rosette
        rosette_mask = rosettes == rosette_id

        # Fill any internal holes in the rosette mask
        filled_rosette = binary_fill_holes(rosette_mask)

        # Identify all cells overlapping the filled rosette mask
        cells_in_rosette = np.unique(mask[filled_rosette])
        cells_in_rosette = cells_in_rosette[cells_in_rosette != 0]  # Exclude background

        # Check if the rosette contains more than one cell
        if len(cells_in_rosette) > 1:
            # Add all enclosed cells to the dictionary
            rosette_dict[int(rosette_id)] = set(cells_in_rosette)

    # Debug output
    print(f"Found {len(rosette_dict)} distinct rosettes:")
    for rosette_id, cells in rosette_dict.items():
        print(f"Rosette {rosette_id}: {len(cells)} cells - {sorted(cells)}")

    return rosette_dict


def label_cells_from_boundaries(mask):
    """
    Relabel regions inside boundaries as unique cells and compute their properties.
    Uses an improved colormap and aesthetics for better visualization.

    Parameters:
    -----------
    mask : numpy.ndarray
        Input mask (binary or labeled)

    Returns:
    --------
    labeled_cells : numpy.ndarray
        2D array with unique labels for each cell region
    """
    # Handle different dimensionalities
    if mask.ndim > 2:
        mask_2d = mask[0, 0, 0]  # Extract 2D slice
    else:
        mask_2d = mask

    # If mask is binary, label connected components
    if len(np.unique(mask_2d)) == 2:  # Binary mask (0 and 1)
        labeled_cells = label(mask_2d > 0, background=0)
    else:
        # If already labeled, use find_boundaries to separate touching objects
        boundaries = find_boundaries(mask_2d, mode="outer")
        interior_mask = np.where(boundaries, 0, mask_2d)
        labeled_cells = label(interior_mask > 0, background=0)

    return labeled_cells


def measure_cell_properties_with_rosettes_and_neighbors(
    sample,
    output_dir,
    labeled_image,
    rosette_dict,
    neighborhood_network,
    intensity_image=None,
    min_cell_size=50,
    min_rosette_size=10,
    max_rosette_size=1000,
    normalization_factor=1000,
):
    """
    Measure geometric, intensity-based, and rosette properties of cells in a labeled image,
    and include neighborhood information.

    Parameters:
    ----------
    sample : str
        Sample name
    output_dir : str
        Output directory
    labeled_image : ndarray
        Labeled image where each cell has a unique integer label
    rosette_dict : dict
        Dictionary where keys are rosette IDs and values are sets of cell IDs
    neighborhood_network : networkx.Graph
        Network graph with neighborhood relationships
    intensity_image : ndarray, optional
        Intensity image to measure intensity-based properties (default is None)
    min_cell_size : int
        Minimum cell size in pixels
    min_rosette_size : int
        Minimum rosette size in pixels
    max_rosette_size : int
        Maximum rosette size in pixels
    normalization_factor : int
        Factor for normalization (rosettes per X cells)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing properties for each cell, including rosette and neighborhood information
    """
    # Handle different dimensionalities
    if labeled_image.ndim > 2:
        labeled_image = labeled_image[0, 0, 0]

    # Define the properties to extract
    properties = [
        "label",  # Cell ID
        "area",  # Area
        "perimeter",  # Perimeter
        "eccentricity",  # Eccentricity
        "orientation",  # Orientation of major axis
        "major_axis_length",  # Length of major axis
        "minor_axis_length",  # Length of minor axis
        "centroid",  # Centroid coordinates
        "bbox",  # Bounding box
        "extent",  # Area / bounding box area
        "solidity",  # Area / convex area
        "convex_area",  # Area of convex hull
        "equivalent_diameter",  # Diameter of equivalent circle
    ]

    # Add intensity-based properties if intensity image is provided
    if intensity_image is not None:
        properties.extend(
            [
                "mean_intensity",
                "max_intensity",
                "min_intensity",
            ]
        )

    # Measure properties using regionprops_table
    props = regionprops_table(
        labeled_image, intensity_image=intensity_image, properties=properties
    )

    # Convert to pandas DataFrame
    cell_properties = pd.DataFrame(props)
    cell_properties["sample"] = sample

    # Add rosette-related columns
    cell_properties["is_in_rosette"] = False  # Initialize with False
    cell_properties["rosette_id"] = None  # Initialize with None
    cell_properties["rosette_nbr"] = (
        None  # Initialize with None (will store rosette number)
    )
    cell_properties["neighbors"] = None  # Initialize neighbors column

    # Map rosette information and neighbors to each cell
    for rosette_nbr, (rosette_id, cells) in enumerate(rosette_dict.items(), start=1):
        cell_properties.loc[cell_properties["label"].isin(cells), "is_in_rosette"] = (
            True
        )
        cell_properties.loc[cell_properties["label"].isin(cells), "rosette_id"] = (
            rosette_id
        )
        cell_properties.loc[cell_properties["label"].isin(cells), "rosette_nbr"] = (
            rosette_nbr  # Add rosette number
        )

    # Ensure the neighbors column is of type 'object'
    cell_properties["neighbors"] = [[] for _ in range(len(cell_properties))]

    # Add neighbors to each cell if neighborhood network exists
    if neighborhood_network is not None:
        for cell_id in neighborhood_network.nodes:
            # Convert cell_id back to int for matching with DataFrame
            cell_id_int = int(cell_id) if isinstance(cell_id, str) else cell_id

            # Get the neighbors of the current cell (convert back to ints)
            neighbors = [
                int(n) if isinstance(n, str) else n
                for n in neighborhood_network.neighbors(cell_id)
            ]

            # Find the index of the cell in the DataFrame by matching its label
            cell_index = cell_properties.index[
                cell_properties["label"] == cell_id_int
            ].tolist()

            if cell_index:  # If a matching index is found
                # Assign the list of neighbors to the "neighbors" column
                cell_properties.at[cell_index[0], "neighbors"] = neighbors

    # Save the properties to a CSV file
    output_path = os.path.join(output_dir, f"cell_properties_{sample}.csv")
    cell_properties.to_csv(output_path, index=False)
    print(f"\033[92mCell properties saved to {output_path}\033[0m")

    return cell_properties


def neigh_graph_pipeline(processed_mask, rosette_dict, output_dir, sample):
    """
    Create neighborhood graph pipeline for cell network analysis.

    Parameters:
    -----------
    processed_mask : numpy.ndarray
        Processed mask with cell labels
    rosette_dict : dict
        Dictionary mapping rosette IDs to sets of cell IDs
    output_dir : str
        Output directory
    sample : str
        Sample name

    Returns:
    --------
    networkx.Graph
        NetworkX graph with rosette annotations
    """
    # Handle different dimensionalities
    if processed_mask.ndim > 2:
        mask_2d = processed_mask[0, 0, 0]
    else:
        mask_2d = processed_mask

    # Step 1: Extract the neighborhood network
    neighborhood_network = extract_neighborhood(mask_2d)

    # Step 2: Annotate the neighborhood with rosette data
    annotated_network = annotate_neighborhood_with_rosettes(
        neighborhood_network, rosette_dict
    )

    # Step 3: Build the NetworkX graph with rosette annotations
    G = build_networkx_graph_with_rosettes(annotated_network, sample, output_dir)
    return G


def extract_neighborhood(labeled_mask):
    """
    Extract neighborhood relationships from a labeled mask.

    Parameters:
    -----------
    labeled_mask : numpy.ndarray
        2D array with labeled cells

    Returns:
    --------
    networkx.Graph
        Graph with cell neighborhoods
    """
    from scipy.ndimage import binary_dilation

    G = nx.Graph()

    # Get unique labels (excluding background)
    labels = np.unique(labeled_mask)
    labels = labels[labels != 0]

    # Add nodes (convert to strings for GML compatibility)
    for label in labels:
        G.add_node(str(label))

    # Find neighbors by checking dilated boundaries
    for label in labels:
        # Create mask for current cell
        cell_mask = labeled_mask == label

        # Dilate the cell mask to find neighbors
        dilated = binary_dilation(cell_mask)

        # Find neighboring labels
        neighbor_labels = np.unique(labeled_mask[dilated])
        neighbor_labels = neighbor_labels[neighbor_labels != 0]  # Remove background
        neighbor_labels = neighbor_labels[neighbor_labels != label]  # Remove self

        # Add edges to neighbors (convert to strings for GML compatibility)
        for neighbor in neighbor_labels:
            G.add_edge(str(label), str(neighbor))

    return G


def annotate_neighborhood_with_rosettes(neighborhood_network, rosette_dict):
    """
    Annotate neighborhood network with rosette information.

    Parameters:
    -----------
    neighborhood_network : networkx.Graph
        Network graph with cell neighborhoods
    rosette_dict : dict
        Dictionary mapping rosette IDs to sets of cell IDs

    Returns:
    --------
    networkx.Graph
        Annotated network graph
    """
    # Create a copy of the network
    annotated_network = neighborhood_network.copy()

    # Add rosette annotations to nodes
    for node in annotated_network.nodes():
        # Check if this cell is part of any rosette (convert node back to int for comparison)
        node_int = int(node) if isinstance(node, str) else node
        is_rosette_cell = False
        rosette_id = None

        for r_id, cells in rosette_dict.items():
            if node_int in cells:
                is_rosette_cell = True
                rosette_id = r_id
                break

        # Add attributes (convert all values to strings for GML compatibility)
        annotated_network.nodes[node]["is_rosette"] = str(is_rosette_cell)
        annotated_network.nodes[node]["rosette_id"] = (
            str(rosette_id) if rosette_id is not None else "None"
        )

    return annotated_network


def build_networkx_graph_with_rosettes(annotated_network, sample, output_dir):
    """
    Build and save NetworkX graph with rosette annotations.

    Parameters:
    -----------
    annotated_network : networkx.Graph
        Annotated network graph
    sample : str
        Sample name
    output_dir : str
        Output directory

    Returns:
    --------
    networkx.Graph
        Final network graph
    """
    # Save graph structure
    output_path = os.path.join(output_dir, f"neighborhood_graph_{sample}.gml")
    nx.write_gml(annotated_network, output_path)
    print(f"\033[92mNeighborhood graph saved to {output_path}\033[0m")

    return annotated_network
