#!/usr/bin/env python3
"""
Script to copy raw intensity TIF files to match the structure of segmented files.
This ensures all three file types (raw, segmented, curated) are in the same location.
"""

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def find_tif_files(directory, exclude_patterns=None):
    """
    Find all TIF files in a directory, optionally excluding files with certain patterns.

    Args:
        directory (Path): Directory to search
        exclude_patterns (list): List of patterns to exclude from filenames

    Returns:
        list: List of Path objects for found TIF files
    """
    if exclude_patterns is None:
        exclude_patterns = []

    tif_files = []

    # Search for TIF files recursively
    for pattern in ["*.tif", "*.tiff"]:
        for tif_path in directory.rglob(pattern):
            # Check if any exclude pattern is in the filename
            if not any(
                exclude_pattern in tif_path.name for exclude_pattern in exclude_patterns
            ):
                tif_files.append(tif_path)

    return sorted(tif_files)


def create_target_path(source_file, source_root, target_root, suffix=""):
    """
    Create the target path for a file, preserving directory structure.

    Args:
        source_file (Path): Source file path
        source_root (Path): Root of source directory
        target_root (Path): Root of target directory
        suffix (str): Optional suffix to add to directories

    Returns:
        Path: Target file path
    """
    # Get relative path from source root
    rel_path = source_file.relative_to(source_root)

    # If we need to add suffix to directories
    if suffix:
        # Split the relative path into parts
        path_parts = list(rel_path.parts)

        # Add suffix to all directory parts (not the filename)
        if len(path_parts) > 1:
            dir_parts = [f"{part}{suffix}" for part in path_parts[:-1]]
            new_rel_path = Path(*dir_parts) / path_parts[-1]
        else:
            new_rel_path = rel_path
    else:
        new_rel_path = rel_path

    # Create target path
    target_path = target_root / new_rel_path

    return target_path


def copy_raw_files(source_dir, target_dir, dry_run=False, verbose=True):
    """
    Copy raw intensity files from source to target directory.

    Args:
        source_dir (str/Path): Source directory containing raw files
        target_dir (str/Path): Target directory (the _seg directory)
        dry_run (bool): If True, show what would be copied without actually copying
        verbose (bool): Print detailed information

    Returns:
        dict: Statistics about the copy operation
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    if not target_path.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_path}")

    # Find raw TIF files (exclude _seg and _cur files)
    exclude_patterns = ["_seg", "_cur"]
    raw_files = find_tif_files(source_path, exclude_patterns)

    if verbose:
        print(f"Found {len(raw_files)} raw TIF files in {source_path}")

    if len(raw_files) == 0:
        print("No raw TIF files found to copy!")
        return {"copied": 0, "skipped": 0, "failed": 0}

    stats = {"copied": 0, "skipped": 0, "failed": 0}
    failed_files = []

    # Process each file
    for source_file in tqdm(raw_files, desc="Copying files", disable=not verbose):
        try:
            # Create target path with _seg suffix for directories
            target_file = create_target_path(
                source_file, source_path, target_path, "_seg"
            )

            if verbose:
                print(f"  {source_file.name} -> {target_file}")

            if dry_run:
                print(f"[DRY RUN] Would copy: {source_file} -> {target_file}")
                stats["copied"] += 1
                continue

            # Check if target already exists
            if target_file.exists():
                if verbose:
                    print(f"    Skipping (already exists): {target_file.name}")
                stats["skipped"] += 1
                continue

            # Create target directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file
            shutil.copy2(source_file, target_file)
            stats["copied"] += 1

            if verbose:
                print(f"    ✓ Copied: {target_file.name}")

        except Exception as e:
            if verbose:
                print(f"    ✗ Failed to copy {source_file.name}: {e}")
            stats["failed"] += 1
            failed_files.append(str(source_file))

    # Print summary
    print("\n" + "=" * 60)
    print("COPY OPERATION SUMMARY")
    print("=" * 60)
    print(f"Total files found: {len(raw_files)}")
    print(f"Successfully copied: {stats['copied']}")
    print(f"Skipped (already exist): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")

    if failed_files:
        print("\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Copy raw intensity TIF files to match segmented file structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy files from original directory to _seg directory
  python copy_raw_files.py /path/to/original /path/to/original_seg

  # Dry run to see what would be copied
  python copy_raw_files.py /path/to/original /path/to/original_seg --dry-run

  # Based on your workflow:
  python copy_raw_files.py \\
    "/Users/noahbruderer/Desktop/250707_R3_timeseries_inhibitors_CF" \\
    "/Users/noahbruderer/Desktop/250707_R3_timeseries_inhibitors_CF_seg"
        """,
    )

    parser.add_argument("source_dir", help="Source directory containing raw TIF files")

    parser.add_argument(
        "target_dir",
        help="Target directory (_seg directory) where files should be copied",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying files",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed information (default: True)",
    )

    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    # Handle verbose flag
    verbose = args.verbose and not args.quiet

    try:
        # Validate directories
        source_path = Path(args.source_dir)
        target_path = Path(args.target_dir)

        print(f"Source directory: {source_path}")
        print(f"Target directory: {target_path}")

        if args.dry_run:
            print("DRY RUN MODE - No files will actually be copied")

        print("-" * 60)

        # Perform the copy operation
        stats = copy_raw_files(
            source_dir=source_path,
            target_dir=target_path,
            dry_run=args.dry_run,
            verbose=verbose,
        )

        if not args.dry_run:
            print("\n✅ Copy operation completed successfully!")
        else:
            print(
                "\n✅ Dry run completed! Use without --dry-run to actually copy files."
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
