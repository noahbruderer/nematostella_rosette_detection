#!/usr/bin/env python3
"""
Script to organize microscopy files for Napari verification.
Recursively finds and organizes original, segmentation, and prediction files
into subdirectories for easy loading in Napari, regardless of directory depth.
"""

import shutil
from pathlib import Path


def find_all_tif_directories(root_path):
    """
    Recursively find all directories containing .tif files.

    Args:
        root_path (Path): Root directory to search

    Returns:
        list: List of Path objects containing .tif files
    """
    tif_dirs = []

    for path in root_path.rglob("*"):
        if path.is_dir():
            # Check if this directory contains .tif files
            tif_files = list(path.glob("*.tif"))
            if tif_files:
                tif_dirs.append(path)

    return tif_dirs


def get_relative_path(full_path, root_path):
    """Get relative path from root."""
    return full_path.relative_to(root_path)


def organize_files_flexible(source_dir, output_dir):
    """
    Organize files from any directory containing .tif files into Napari-friendly structure.

    Args:
        source_dir (Path): Source directory containing .tif files
        output_dir (Path): Output directory for this specific folder
    """

    # Get all .tif files in the directory
    files = list(source_dir.glob("*.tif"))

    if not files:
        return

    print(f"  Found {len(files)} .tif files")

    # Group files by their base identifier
    file_groups = {}

    for file in files:
        # Try different patterns to extract base identifier
        base_id = extract_base_identifier(file.stem)

        if base_id not in file_groups:
            file_groups[base_id] = []
        file_groups[base_id].append(file)

    # Process each group of files
    for base_id, group_files in file_groups.items():
        if len(group_files) < 2:  # Skip if only one file (probably not a complete set)
            continue

        # Create subdirectory for this group
        group_dir = output_dir / base_id
        group_dir.mkdir(parents=True, exist_ok=True)

        # Find and copy the three types of files we need
        original_file = None
        seg_cur_file = None
        pred_file = None

        for file in group_files:
            filename = file.name.lower()  # Case insensitive matching

            # Original file (no extension indicators, just the base + .tif)
            if is_original_file(file.name, base_id):
                original_file = file

            # Segmentation curated file (*seg*cur.tif)
            elif "seg" in filename and "cur" in filename and filename.endswith(".tif"):
                seg_cur_file = file

            # Prediction file (*pred.tif, but not *pred_float.tif)
            elif "pred.tif" in filename and "pred_float.tif" not in filename:
                pred_file = file

        # Copy files with descriptive names
        files_copied = 0
        if original_file:
            shutil.copy2(original_file, group_dir / f"01_original_{original_file.name}")
            print(f"    ✓ Copied original: {original_file.name}")
            files_copied += 1

        if seg_cur_file:
            shutil.copy2(
                seg_cur_file, group_dir / f"02_segmentation_{seg_cur_file.name}"
            )
            print(f"    ✓ Copied segmentation: {seg_cur_file.name}")
            files_copied += 1

        if pred_file:
            shutil.copy2(pred_file, group_dir / f"03_prediction_{pred_file.name}")
            print(f"    ✓ Copied prediction: {pred_file.name}")
            files_copied += 1

        if files_copied > 0:
            print(f"    📁 Created group: {group_dir}")
        else:
            # Remove empty directory
            group_dir.rmdir()


def extract_base_identifier(filename):
    """
    Extract base identifier from filename using various patterns.
    IMPORTANT: Keeps _0001, _0002 etc. as part of the identifier since these are different images!

    Args:
        filename (str): Filename without extension

    Returns:
        str: Base identifier
    """

    # Remove processing-related suffixes but KEEP position identifiers like _0001, _0002
    base = filename
    processing_suffixes = ["_pred_float", "_curation_vis", "_pred", "_seg_cur", "_seg"]

    # Apply suffixes in order of specificity (longest first)
    processing_suffixes.sort(key=len, reverse=True)

    for suffix in processing_suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    # For files that end with just '_cur' (after removing '_seg_cur')
    if base.endswith("_cur"):
        base = base[:-4]  # Remove '_cur'

    return base


def is_original_file(filename, base_id):
    """
    Check if a file is the original image file.

    Args:
        filename (str): Full filename
        base_id (str): Base identifier

    Returns:
        bool: True if this is an original file
    """

    # Remove .tif extension for comparison
    name_no_ext = filename.replace(".tif", "")

    # Original file should be exactly the base_id
    if name_no_ext == base_id:
        return True

    return False


def create_napari_structure_flexible(source_dir, output_dir):
    """
    Create organized directory structure for Napari verification.
    Works with any directory structure depth.

    Args:
        source_dir (str): Source directory containing the original structure
        output_dir (str): Output directory for organized files
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)

    print(f"Searching for directories with .tif files in {source_path}...")

    # Find all directories containing .tif files
    tif_directories = find_all_tif_directories(source_path)

    print(f"Found {len(tif_directories)} directories with .tif files")

    for tif_dir in tif_directories:
        # Get relative path from source to maintain structure
        rel_path = get_relative_path(tif_dir, source_path)

        print(f"\nProcessing {rel_path}...")

        # Create corresponding output directory
        output_tif_dir = output_path / rel_path

        # Organize files in this directory
        organize_files_flexible(tif_dir, output_tif_dir)


def main():
    """Main function to run the script."""

    # Configuration - modify these paths as needed
    source_directory = "~/Desktop/250707_R3_timeseries_inhibitors_CF_seg"
    output_directory = "~/Desktop/napari_verification_files_all"

    # Expand user directory
    source_dir = Path(source_directory).expanduser()
    output_dir = Path(output_directory).expanduser()

    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist!")
        print("Please modify the source_directory variable in the script.")
        return

    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print("\nStarting flexible file organization...")

    try:
        create_napari_structure_flexible(source_dir, output_dir)
        print("\n✅ File organization complete!")
        print(f"Organized files are available in: {output_dir}")
        print("\nTo use in Napari:")
        print("1. Navigate to any group subdirectory")
        print(
            "2. Select the available files (01_original, 02_segmentation, 03_prediction)"
        )
        print("3. Drag and drop them into Napari for verification")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
