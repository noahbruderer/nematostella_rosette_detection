import os
import shutil
import sys
import time
from pathlib import Path

# Fix the import - add the parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try relative import first (if this is a package)
    from .image_segmentations_controle_time_series import segment_images_parallel

    print("Successfully imported from relative path")
except ImportError:
    try:
        # Try direct import
        from image_segmentations_controle_time_series import segment_images_parallel

        print("Successfully imported from direct path")
    except ImportError as e:
        print(f"ERROR: Could not import segment_images_parallel: {e}")
        print(
            "Make sure image_segmentations_controle_time_series.py is in the scripts directory"
        )
        sys.exit(1)


def create_test_subset(source_dir, test_dir, max_files=5, max_depth=2):
    """
    Create a small test subset of your data for testing.

    Args:
        source_dir: Original data directory
        test_dir: Where to create test subset
        max_files: Maximum files per directory
        max_depth: Maximum directory depth to copy
    """
    print(f"Creating test subset from {source_dir}")
    print(f"Test directory: {test_dir}")

    if os.path.exists(test_dir):
        response = input("Test directory exists. Remove it? (y/n): ")
        if response.lower() == "y":
            shutil.rmtree(test_dir)
        else:
            print("Using existing test directory")
            return test_dir

    files_copied = 0
    dirs_created = 0

    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, source_dir)
        depth = len(Path(rel_path).parts)

        # Skip if too deep
        if depth > max_depth:
            dirs[:] = []  # Don't recurse deeper
            continue

        # Create corresponding directory in test set
        if rel_path == ".":
            dest_dir = test_dir
        else:
            dest_dir = os.path.join(test_dir, rel_path)

        os.makedirs(dest_dir, exist_ok=True)
        dirs_created += 1

        # Copy only first few .tif files
        tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        for i, file in enumerate(tif_files[:max_files]):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_dir, file)
            shutil.copy2(src_file, dst_file)
            files_copied += 1
            print(f"  Copied: {rel_path}/{file}")

    print("\nTest subset created:")
    print(f"  - {dirs_created} directories")
    print(f"  - {files_copied} files")
    return test_dir


def run_test_segmentation(test_mode="subset"):
    """
    Run a test segmentation on a small subset of data.

    Args:
        test_mode: One of "subset", "single", "dry_run", "full"
    """

    # ============================================================================
    # TEST CONFIGURATION
    # ============================================================================

    # Your existing paths
    MODEL_PATH = "/Users/noahbruderer/local_work_files/nematostella_rosette_project/Timepoint_experiment_replicate_all/models/full_database"
    BASE_DIR = "/Users/noahbruderer/Desktop"
    EXPERIMENT_NAME = "250707_R3_timeseries_inhibitors_CF"
    IMAGE_SUBDIR = "250707_timeseries_live_R3"

    # Test configuration
    TEST_MAX_FILES = 3  # Number of files per directory for test
    TEST_MAX_DEPTH = 2  # Maximum directory depth for test
    TEST_NUM_THREADS = 4  # Fewer threads for testing

    # ============================================================================

    print("\n" + "=" * 60)
    print(f"SEGMENTATION TEST RUN - MODE: {test_mode}")
    print("=" * 60)

    if test_mode == "subset":
        # Create a small test subset
        source_dir = os.path.join(BASE_DIR, EXPERIMENT_NAME, IMAGE_SUBDIR)
        test_base_dir = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_TEST")
        test_image_dir = os.path.join(test_base_dir, IMAGE_SUBDIR)

        # Check if source directory exists
        if not os.path.exists(source_dir):
            print(f"ERROR: Source directory does not exist: {source_dir}")
            return

        # Create test subset
        create_test_subset(
            source_dir=source_dir,
            test_dir=test_image_dir,
            max_files=TEST_MAX_FILES,
            max_depth=TEST_MAX_DEPTH,
        )

        print("\n" + "=" * 60)
        print("Running segmentation on test subset...")
        print("=" * 60)

        # Run segmentation on test subset
        segment_images_parallel(
            model_path=MODEL_PATH,
            image_dir=test_image_dir,
            base_data_dir=test_base_dir,
            num_threads=TEST_NUM_THREADS,
        )

        print("\n" + "=" * 60)
        print("TEST COMPLETE!")
        print(f"Check results in: {os.path.join(test_base_dir, f'{IMAGE_SUBDIR}_seg')}")
        print("If results look good, run the full segmentation.")
        print("=" * 60)

    elif test_mode == "single":
        # Test on just one file
        print("Testing on single file...")

        # Find first .tif file
        source_dir = os.path.join(BASE_DIR, EXPERIMENT_NAME, IMAGE_SUBDIR)

        if not os.path.exists(source_dir):
            print(f"ERROR: Source directory does not exist: {source_dir}")
            return

        first_tif = None
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith((".tif", ".tiff")):
                    first_tif = os.path.join(root, file)
                    break
            if first_tif:
                break

        if first_tif:
            print(f"Testing on: {first_tif}")
            # Create temporary directory with just one file
            test_dir = os.path.join(BASE_DIR, "SINGLE_FILE_TEST")
            os.makedirs(test_dir, exist_ok=True)
            shutil.copy2(first_tif, os.path.join(test_dir, os.path.basename(first_tif)))

            segment_images_parallel(
                model_path=MODEL_PATH,
                image_dir=test_dir,
                base_data_dir=BASE_DIR,
                num_threads=1,
            )
        else:
            print("No .tif files found!")

    elif test_mode == "dry_run":
        # Dry run - just show what would be processed
        print("DRY RUN - No actual processing")
        source_dir = os.path.join(BASE_DIR, EXPERIMENT_NAME, IMAGE_SUBDIR)

        if not os.path.exists(source_dir):
            print(f"ERROR: Source directory does not exist: {source_dir}")
            print("Please check that the path is correct")
            return

        file_count = 0
        dir_structure = {}

        for root, dirs, files in os.walk(source_dir):
            tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
            if tif_files:
                rel_path = os.path.relpath(root, source_dir)
                dir_structure[rel_path] = len(tif_files)
                file_count += len(tif_files)

        print(f"\nFound {file_count} total .tif files")
        print("\nDirectory structure:")
        for path, count in sorted(dir_structure.items())[:10]:
            print(f"  {path}: {count} files")
        if len(dir_structure) > 10:
            print(f"  ... and {len(dir_structure) - 10} more directories")

        print(f"\nEstimated output size: ~{file_count} segmentation masks")
        print(
            f"Output location would be: {os.path.join(BASE_DIR, EXPERIMENT_NAME, f'{IMAGE_SUBDIR}_seg')}"
        )

    elif test_mode == "full":
        # Run the full segmentation
        print("Running FULL segmentation...")

        IMAGE_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, IMAGE_SUBDIR)
        BASE_DATA_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)

        if not os.path.exists(IMAGE_DIR):
            print(f"ERROR: Image directory does not exist: {IMAGE_DIR}")
            return

        segment_images_parallel(
            model_path=MODEL_PATH,
            image_dir=IMAGE_DIR,
            base_data_dir=BASE_DATA_DIR,
            num_threads=12,
        )

    else:
        print(f"Unknown test mode: {test_mode}")
        print("Valid modes: subset, single, dry_run, full")


if __name__ == "__main__":
    print("Starting test segmentation script...")
    print(f"Python version: {sys.version}")
    print(f"Script location: {os.path.abspath(__file__)}")

    # 1. First do a dry run to see what will be processed
    print("\n>>> STEP 1: DRY RUN")
    run_test_segmentation(test_mode="dry_run")

    # Ask user if they want to continue
    response = input("\nContinue with subset test? (y/n): ")
    if response.lower() == "y":
        # 2. Then test on a subset
        print("\n>>> STEP 2: SUBSET TEST")
        run_test_segmentation(test_mode="subset")
    else:
        print("Test cancelled by user")

    # 3. Optionally run full processing
    # print("\n>>> STEP 3: FULL RUN")
    # run_test_segmentation(test_mode="full")
