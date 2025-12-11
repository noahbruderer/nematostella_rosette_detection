import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import tifffile
from morphosnaker import segmentation, utils
from tqdm import tqdm


def process_single_image(
    args: Tuple[str, str, "segmentation.Segmentation", "utils.ImageProcessor"],
) -> Tuple[str, bool]:
    """Process a single image with the given segmentation module."""
    image_path, output_path, seg_module, image_processor = args

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load and process the image
        image = tifffile.imread(image_path)
        print(f"Processing {os.path.basename(image_path)} - Shape: {image.shape}")

        masks = seg_module.predict(image)
        image_processor.save(masks, output_path)

        return (image_path, True)
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return (image_path, False)


def segment_images_parallel(
    model_path: str, experiment_dir: str, num_threads: int = 12
) -> None:
    """
    Segment all images found in the experiment directory while preserving structure.

    Args:
        model_path: Path to the segmentation model
        experiment_dir: Main experiment directory to search for images
        num_threads: Number of parallel threads
    """

    # Initialize modules
    print("Initializing image processor and segmentation module...")
    image_processor = utils.ImageProcessor()
    seg_module = segmentation.Segmentation(method="cellpose")

    # Load the model
    print(f"Loading custom Cellpose model from {model_path}...")
    seg_module.load_model(model_path)

    # Create output directory with _seg suffix
    experiment_name = os.path.basename(os.path.normpath(experiment_dir))
    parent_dir = os.path.dirname(os.path.normpath(experiment_dir))
    output_base_dir = os.path.join(parent_dir, f"{experiment_name}_seg")

    print(f"Input directory: {experiment_dir}")
    print(f"Output directory: {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)

    # Collect all .tif files while preserving structure
    image_files = []
    output_paths = []

    print("Scanning for .tif/.tiff files in all subdirectories...")
    for root, dirs, files in os.walk(experiment_dir):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                image_path = os.path.join(root, file)
                image_files.append(image_path)

                # Calculate relative path from experiment_dir
                rel_path = os.path.relpath(root, experiment_dir)

                # Create corresponding output path with _seg suffix for each directory level
                if rel_path == ".":
                    # Files in root directory
                    output_dir = output_base_dir
                else:
                    # Recreate subdirectory structure with _seg suffix to each dir
                    path_parts = Path(rel_path).parts
                    seg_parts = [f"{part}_seg" for part in path_parts]
                    output_dir = os.path.join(output_base_dir, *seg_parts)

                # Add _seg suffix to filename (before extension)
                name, ext = os.path.splitext(file)
                output_filename = f"{name}_seg{ext}"
                output_path = os.path.join(output_dir, output_filename)
                output_paths.append(output_path)

    total_images = len(image_files)

    if total_images == 0:
        print("No .tif/.tiff images found in the input directory!")
        return

    print(f"Found {total_images} .tif/.tiff images to process")

    # Show directory structure preview
    unique_dirs = set(os.path.dirname(p) for p in output_paths)
    print(f"Will create {len(unique_dirs)} output directories")

    # Show first few directories as preview
    preview_dirs = sorted(list(unique_dirs))[:5]
    for d in preview_dirs:
        print(f"  → {d}")
    if len(unique_dirs) > 5:
        print(f"  ... and {len(unique_dirs) - 5} more directories")

    # Prepare arguments for parallel processing
    process_args = [
        (img_path, out_path, seg_module, image_processor)
        for img_path, out_path in zip(image_files, output_paths)
    ]

    # Process images in parallel
    print(f"\nProcessing images using {num_threads} threads...")
    successful = 0
    failed = 0
    failed_images = []

    with tqdm(total=total_images, desc="Segmenting images", unit="img") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(process_single_image, args): args[0]
                for args in process_args
            }

            # Process completed tasks
            for future in as_completed(future_to_image):
                image_path, success = future.result()

                if success:
                    successful += 1
                else:
                    failed += 1
                    failed_images.append(image_path)

                # Update progress bar
                pbar.set_postfix(
                    {
                        "✓": successful,
                        "✗": failed,
                        "rate": f"{(successful / (successful + failed)) * 100:.1f}%",
                    }
                )
                pbar.update(1)

    # Print final summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"✓ Successfully processed: {successful}/{total_images} images")

    if failed > 0:
        print(f"✗ Failed to process: {failed} images")
        print("\nFailed images:")
        for img in failed_images[:10]:  # Show first 10 failed
            print(f"  - {img}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")

    print(f"\nOverall success rate: {(successful / total_images) * 100:.1f}%")
    print(f"Output directory: {output_base_dir}")


# Example usage
if __name__ == "__main__":
    import time

    # ============================================================================
    # USER CONFIGURATION - CHANGE THESE PATHS FOR YOUR SYSTEM
    # ============================================================================

    # Path to your model
    MODEL_PATH = "/Users/noahbruderer/local_work_files/nematostella_rosette_project/Timepoint_experiment_replicate_all/models/full_databse"

    # Base directory where your data is located (change this to your path)
    BASE_DIR = "/Users/noahbruderer/Desktop"

    # Name of the experiment folder that contains all your images
    EXPERIMENT_NAME = "250707_R3_timeseries_inhibitors_CF"

    # Number of parallel threads to use
    NUM_THREADS = 12

    # ============================================================================
    # CONSTRUCTED PATHS (automatically built from above configuration)
    # ============================================================================

    # Full path to experiment directory (will search for ALL images inside)
    EXPERIMENT_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)

    print("=" * 60)
    print("SEGMENTATION CONFIGURATION")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Experiment directory: {EXPERIMENT_DIR}")
    print(f"Will search for ALL .tif files in: {EXPERIMENT_DIR}")
    print(f"Output will be saved to: {EXPERIMENT_DIR}_seg")
    print(f"Using {NUM_THREADS} threads")
    print("=" * 60)

    # Run the parallel processing
    start_time = time.time()

    segment_images_parallel(
        model_path=MODEL_PATH,
        experiment_dir=EXPERIMENT_DIR,
        num_threads=NUM_THREADS,
    )

    end_time = time.time()

    # Print execution time
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per thread: {total_time / NUM_THREADS:.2f} seconds")
