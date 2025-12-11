#!/usr/bin/env python3

"""
Checks the merge between raw metadata and curated rosette metadata. (v2)

This script performs a 'dry run' of the merge that would happen in
the main Snakemake pipeline.

It loads:
1. The 'sample_metadata.tsv' (all raw .tif files)
2. The 'curated_rosette_metadata.tsv' (all curated segmentation files)

It then performs a left merge based on the new 'unique_sample_id' system.
It links the "base ID" (e.g., R1-T0-A1-I0) from the curated file to all
matching raw files (e.g., R1-T0-A1-I0-PRI, R1-T0-A1-I0-MAX, etc.).

It reports how many raw files have a matching curated file and,
crucially, lists any raw files that are MISSING a match.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(raw_meta_file, curated_meta_file, output_log):
    """
    Main execution function.
    """
    raw_path = Path(raw_meta_file)
    curated_path = Path(curated_meta_file)
    log_path = Path(output_log)

    # --- 1. Load Files ---
    try:
        raw_df = pd.read_csv(raw_path, sep="\t")
        print(f"✅ Loaded {len(raw_df)} raw samples from: {raw_path}")
    except FileNotFoundError:
        print(f"Error: Raw metadata file not found at: {raw_path}", file=sys.stderr)
        sys.exit(1)

    try:
        curated_df = pd.read_csv(curated_path, sep="\t")
        print(f"✅ Loaded {len(curated_df)} curated samples from: {curated_path}")
    except FileNotFoundError:
        print(
            f"Error: Curated metadata file not found at: {curated_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 2. Define Merge Keys (Updated) ---
    # We now merge on the 'unique_sample_id'.
    # We must first create a "base_id" in the raw_df to match.

    # Ensure the new ID columns exist
    if "unique_sample_id" not in raw_df.columns:
        print(
            "Error: Merge key 'unique_sample_id' not found in raw metadata.",
            file=sys.stderr,
        )
        sys.exit(1)
    if "unique_sample_id" not in curated_df.columns:
        print(
            "Error: Merge key 'unique_sample_id' not found in curated metadata.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 3. Pre-flight Check: Duplicates ---
    # Check for duplicates in the *curated* file on the new ID.
    duplicates = curated_df[
        curated_df.duplicated(subset=["unique_sample_id"], keep=False)
    ]
    if not duplicates.empty:
        print(
            f"\n🚨 ERROR: Found {len(duplicates)} duplicate entries in curated file on 'unique_sample_id'!"
        )
        print("This will cause errors. Please fix the curated metadata generator.")
        print(duplicates.sort_values(by=["unique_sample_id"]))
        sys.exit(1)
    else:
        print("✅ No duplicates found in curated file. Merge key is clean.")

    # --- 4. Perform Merge (Updated Logic) ---
    #

    # Step 4a: Create the 'base_sample_id' in the raw dataframe.
    # This splits "R1-T0-A1-I0-PRI" into "R1-T0-A1-I0"
    print("Creating 'base_sample_id' in raw metadata for merging...")
    try:
        raw_df["base_sample_id"] = (
            raw_df["unique_sample_id"].str.rsplit("-", n=1).str[0]
        )
    except Exception as e:
        print(f"Error creating 'base_sample_id': {e}")
        print(
            "Please ensure 'unique_sample_id' in raw_metadata is in the expected format."
        )
        sys.exit(1)

    # Step 4b: Perform the left merge
    print("Performing left merge...")
    merged_df = pd.merge(
        raw_df,
        curated_df,
        left_on="base_sample_id",  # Key from raw_df (e.g., R1-T0-A1-I0)
        right_on="unique_sample_id",  # Key from curated_df (e.g., R1-T0-A1-I0)
        how="left",  # Keep all rows from the "left" (raw_df)
        suffixes=("_raw", "_curated"),  # Add suffixes to all overlapping columns
    )

    # --- 5. Report Findings ---
    # We check for 'isna()' on 'curated_rosette_path', which is the
    # unique column from curated_df.
    missing_df = merged_df[merged_df["curated_rosette_path"].isna()]

    total_raw = len(raw_df)
    total_missing = len(missing_df)
    total_found = total_raw - total_missing

    print("\n--- 🔗 Merge Check ---")
    print(f"Total Raw Samples:       {total_raw}")
    print(f"Curated Files Found:   {total_found}")
    print(f"Curated Files MISSING: {total_missing}")

    if total_missing > 0:
        print(f"\n🚨 WARNING: {total_missing} raw samples are missing a curated file.")
        print("The following raw images have no match (showing first 50):")

        # Create a nice identifier for the user (Updated)
        cols_to_show = [
            "unique_sample_id_raw",  # The full ID from sample_metadata.tsv
            "raw_image_path",  # The original file path
            "base_sample_id",  # The "base ID" we tried to match
        ]

        # Ensure columns exist before trying to print them
        cols_to_print = [col for col in cols_to_show if col in missing_df.columns]

        with pd.option_context("display.max_rows", 50):
            print(missing_df[cols_to_print].to_string(index=False))

        # Save to log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(log_path, sep="\t", index=False)
        print(f"\nFull list of {total_missing} missing files saved to: {log_path}")
    else:
        print("\n🎉 SUCCESS! All raw samples have a matching curated file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check merge between raw and curated metadata TSV files."
    )
    parser.add_argument(
        "--raw-metadata",
        default="config/sample_metadata.tsv",
        help="Path to the 'sample_metadata.tsv' file (RAW files).",
    )
    parser.add_argument(
        "--curated-metadata",
        default="config/curated_rosette_metadata.tsv",
        help="Path to the 'curated_rosette_metadata.tsv' file (CURATED files).",
    )
    parser.add_argument(
        "--output-log",
        default="logs/missing_curated_files.tsv",
        help="Path to save the TSV of missing files.",
    )

    args = parser.parse_args()
    main(args.raw_metadata, args.curated_metadata, args.output_log)
