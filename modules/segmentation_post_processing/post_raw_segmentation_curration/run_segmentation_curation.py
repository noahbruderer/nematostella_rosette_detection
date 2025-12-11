#!/usr/bin/env python3
"""
Multithreaded script to run segmentation curation on all TIF files in directories and subdirectories.
Processes input files like 'T10_6_0001_seg.tif' and saves curated versions as 'T10_6_0001_seg_cur.tif'.
Uses multiple threads for faster processing of large datasets.
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from segmentation_curator import SegmentationCurator
from skimage import io
from tqdm import tqdm


def find_tif_files(root_dir, pattern_contains="_seg"):
    """
    Find all TIF files in root_dir and subdirectories that contain the pattern.

    Args:
        root_dir (str): Root directory to search
        pattern_contains (str): Pattern that filenames should contain

    Returns:
        list: List of Path objects for found TIF files
    """
    root_path = Path(root_dir)
    tif_files = []

    # Search for TIF files recursively
    for tif_path in root_path.rglob("*.tif"):
        if pattern_contains in tif_path.name:
            tif_files.append(tif_path)

    # Also search for TIFF files
    for tif_path in root_path.rglob("*.tiff"):
        if pattern_contains in tif_path.name:
            tif_files.append(tif_path)

    return sorted(tif_files)


def load_tif_image(file_path):
    """
    Load a TIF image and handle different formats.

    Args:
        file_path (Path): Path to the TIF file

    Returns:
        numpy.ndarray: Loaded image
    """
    try:
        # Try loading with skimage first
        image = io.imread(str(file_path))

        # If the image has multiple dimensions, we might need to reshape
        if image.ndim > 2:
            # If it's a 3D image with singleton dimensions, squeeze them
            if image.shape[0] == 1 or image.shape[-1] == 1:
                image = np.squeeze(image)
            # If it's still 3D, take the first channel
            if image.ndim == 3:
                image = (
                    image[:, :, 0]
                    if image.shape[2] < image.shape[0]
                    else image[0, :, :]
                )

        return image

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def save_tif_image(image, file_path):
    """
    Save an image as a TIF file.

    Args:
        image (numpy.ndarray): Image to save
        file_path (Path): Path where to save the image
    """
    try:
        # Ensure the image is in the right format (typically uint16 for segmentation masks)
        if image.dtype != np.uint16:
            # Convert to uint16 if it's not already
            if image.max() <= 255:
                image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint16)

        io.imsave(str(file_path), image, check_contrast=False)
        return True

    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


def process_single_file(
    file_path, curator_params, suffix="_cur", save_visualizations=False, verbose=False
):
    """
    Process a single TIF file through the segmentation curation pipeline.
    Creates a new curator instance per thread to avoid sharing issues.

    Args:
        file_path (Path): Path to the input TIF file
        curator_params (dict): Parameters for creating SegmentationCurator
        suffix (str): Suffix to add to output filename
        save_visualizations (bool): Whether to save visualization plots
        verbose (bool): Whether to print verbose output

    Returns:
        dict: Result dictionary with success status, paths, timing, etc.
    """
    start_time = time.time()
    thread_id = threading.current_thread().ident

    try:
        # Create a new curator instance for this thread
        curator = SegmentationCurator(**curator_params)

        # Load the image
        mask = load_tif_image(file_path)
        if mask is None:
            return {
                "success": False,
                "file_path": file_path,
                "output_path": None,
                "processing_time": time.time() - start_time,
                "error": "Failed to load image",
                "thread_id": thread_id,
            }

        # Generate sample name and output path
        sample_name = file_path.stem  # Filename without extension
        output_path = file_path.parent / f"{sample_name}{suffix}.tif"

        # Check if output already exists
        if output_path.exists():
            if verbose:
                print(
                    f"[Thread {thread_id}] Output already exists, skipping: {output_path.name}"
                )
            return {
                "success": True,
                "file_path": file_path,
                "output_path": output_path,
                "processing_time": 0,
                "skipped": True,
                "thread_id": thread_id,
            }

        # Create output directory for visualizations if needed
        vis_output_dir = None
        if save_visualizations:
            vis_output_dir = file_path.parent / f"{sample_name}_curation_vis"

        # Run curation
        result = curator.curate_segmentation(
            mask=mask,
            output_dir=vis_output_dir,
            sample_name=sample_name,
            save_visualizations=save_visualizations,
        )

        # Extract the processed mask
        processed_mask = result["processed_mask"]

        # If the processed mask is 5D, extract the 2D slice
        if processed_mask.ndim > 2:
            processed_mask_2d = processed_mask[0, 0, 0]
        else:
            processed_mask_2d = processed_mask

        # Save the curated mask
        save_success = save_tif_image(processed_mask_2d, output_path)

        processing_time = time.time() - start_time

        if save_success:
            return {
                "success": True,
                "file_path": file_path,
                "output_path": output_path,
                "processing_time": processing_time,
                "skipped": False,
                "thread_id": thread_id,
            }
        else:
            return {
                "success": False,
                "file_path": file_path,
                "output_path": None,
                "processing_time": processing_time,
                "error": "Failed to save image",
                "thread_id": thread_id,
            }

    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "success": False,
            "file_path": file_path,
            "output_path": None,
            "processing_time": processing_time,
            "error": str(e),
            "thread_id": thread_id,
        }


def process_files_multithreaded(
    tif_files,
    curator_params,
    suffix="_cur",
    save_visualizations=False,
    verbose=False,
    max_workers=None,
):
    """
    Process multiple TIF files using multiple threads.

    Args:
        tif_files (list): List of Path objects for TIF files
        curator_params (dict): Parameters for SegmentationCurator
        suffix (str): Suffix for output files
        save_visualizations (bool): Whether to save visualizations
        verbose (bool): Whether to print verbose output
        max_workers (int): Maximum number of worker threads

    Returns:
        tuple: (results_list, summary_stats)
    """
    results = []
    successful = 0
    failed = 0
    skipped = 0
    total_time = 0
    failed_files = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_file,
                file_path,
                curator_params,
                suffix,
                save_visualizations,
                verbose,
            ): file_path
            for file_path in tif_files
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(tif_files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)

                # Update counters
                if result["success"]:
                    if result.get("skipped", False):
                        skipped += 1
                    else:
                        successful += 1
                else:
                    failed += 1
                    failed_files.append(str(result["file_path"]))

                total_time += result["processing_time"]

                # Update progress bar
                pbar.update(1)
                if verbose:
                    status = "✓" if result["success"] else "✗"
                    thread_info = f"[T{result['thread_id']}]" if verbose else ""
                    pbar.set_postfix_str(
                        f"{status} {thread_info} {result['file_path'].name}"
                    )

    summary_stats = {
        "total_files": len(tif_files),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "total_time": total_time,
        "failed_files": failed_files,
    }

    return results, summary_stats


def main():
    parser = argparse.ArgumentParser(
        description="Curate segmentation masks in TIF files using multiple threads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_segmentation_curation.py /path/to/data
  python run_segmentation_curation.py /path/to/data --threads 8
  python run_segmentation_curation.py /path/to/data --min-cell-size 30 --max-small-cell 40 --threads 4
  python run_segmentation_curation.py /path/to/data --suffix _curated --visualizations --threads 6
        """,
    )

    parser.add_argument(
        "input_dir", help="Root directory containing TIF files to process"
    )

    parser.add_argument(
        "--min-cell-size",
        type=int,
        default=50,
        help="Minimum size threshold for creating new cells from background regions (default: 50)",
    )

    parser.add_argument(
        "--max-small-cell",
        type=int,
        default=50,
        help="Maximum size for highlighting small cells in visualization (default: 50)",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="_cur",
        help="Suffix to add to output filenames (default: _cur)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="_seg",
        help="Pattern that input filenames should contain (default: _seg)",
    )

    parser.add_argument(
        "--visualizations",
        action="store_true",
        help="Save visualization plots (default: False)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information (default: False)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of worker threads (default: auto-detect based on CPU cores)",
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)

    # Determine number of threads
    if args.threads is None:
        import os

        max_workers = min(
            32, (os.cpu_count() or 1) + 4
        )  # Default ThreadPoolExecutor logic
    else:
        max_workers = args.threads

    # Find TIF files
    print(f"Searching for TIF files containing '{args.pattern}' in {input_dir}...")
    tif_files = find_tif_files(input_dir, args.pattern)

    if not tif_files:
        print(f"No TIF files found containing '{args.pattern}' in {input_dir}")
        sys.exit(0)

    print(f"Found {len(tif_files)} TIF files to process")

    # Prepare curator parameters
    curator_params = {
        "min_cell_size": args.min_cell_size,
        "max_small_cell_size": args.max_small_cell,
        "verbose": args.verbose,
    }

    # Process files
    print("\nStarting multithreaded segmentation curation...")
    print(f"Min cell size: {args.min_cell_size}")
    print(f"Max small cell size: {args.max_small_cell}")
    print(f"Output suffix: {args.suffix}")
    print(f"Save visualizations: {args.visualizations}")
    print(f"Worker threads: {max_workers}")
    print("-" * 50)

    start_total_time = time.time()

    # Process files using multiple threads
    results, stats = process_files_multithreaded(
        tif_files=tif_files,
        curator_params=curator_params,
        suffix=args.suffix,
        save_visualizations=args.visualizations,
        verbose=args.verbose,
        max_workers=max_workers,
    )

    total_wall_time = time.time() - start_total_time

    # Print summary
    print("\n" + "=" * 50)
    print("MULTITHREADED SEGMENTATION CURATION SUMMARY")
    print("=" * 50)
    print(f"Total files found: {stats['total_files']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Skipped (already exist): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Worker threads used: {max_workers}")
    print(f"Total processing time: {stats['total_time']:.2f} seconds")
    print(f"Wall clock time: {total_wall_time:.2f} seconds")
    print(f"Speedup factor: {stats['total_time'] / total_wall_time:.2f}x")

    if stats["successful"] > 0:
        print(
            f"Average time per file: {stats['total_time'] / stats['successful']:.2f} seconds"
        )

    if stats["failed_files"]:
        print("\nFailed files:")
        for failed_file in stats["failed_files"]:
            print(f"  - {failed_file}")

    print(f"\nCurated segmentation masks saved with suffix '{args.suffix}'")

    if args.visualizations:
        print("Visualization plots saved in corresponding subdirectories")

    # Performance tip
    if stats["total_files"] > max_workers and total_wall_time > 60:
        optimal_threads = min(stats["total_files"], max_workers * 2)
        print(
            f"\nTip: For large datasets, try --threads {optimal_threads} for potentially better performance"
        )


if __name__ == "__main__":
    main()
