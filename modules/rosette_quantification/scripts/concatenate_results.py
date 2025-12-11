#!/usr/bin/env python3
"""
Concatenate Pre-Processed Cell Properties Results
=================================================
Concatenates CSV files that already contain full metadata. This is a simplified
version that assumes metadata (replicate, timepoint, etc.) is already present
in the input files.

Author: Noah Bruderer
Date: 2025
"""

import argparse
import os

import pandas as pd


def concatenate_preprocessed_csvs(input_files):
    """
    Simply reads a list of CSV files and concatenates them into a single DataFrame.
    """
    if not input_files:
        print("Warning: No input files provided for concatenation.")
        return pd.DataFrame()

    all_dfs = []
    print(f"Reading {len(input_files)} CSV files...")
    for csv_file in input_files:
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if not all_dfs:
        print("Error: No valid dataframes could be loaded.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def create_animal_aggregated_data(combined_df):
    """
    Creates animal-level aggregated data for statistical analysis.
    This function groups by the metadata columns that are now present in the combined dataframe.
    """
    if combined_df.empty:
        return pd.DataFrame()

    print("Creating animal-level aggregated data...")

    # Define the columns to group by for aggregation
    group_cols = ["replicate", "timepoint", "animal"]

    # Check if all grouping columns exist
    if not all(col in combined_df.columns for col in group_cols):
        print(f"Error: Missing one or more grouping columns in the data: {group_cols}")
        return pd.DataFrame()

    # Define the aggregations to perform
    agg_funcs = {
        "total_cells": ("is_in_rosette", "count"),
        "rosette_cells": ("is_in_rosette", "sum"),
        "total_rosettes": ("rosette_id", "nunique"),
    }

    # Dynamically add image count if the column exists
    if "image" in combined_df.columns:
        agg_funcs["image_count"] = ("image", pd.Series.nunique)

    # Automatically find all other numerical columns to average
    numerical_cols = combined_df.select_dtypes(include=["float64", "int64"]).columns
    for col in numerical_cols:
        # Exclude columns that are already used for grouping or are IDs
        if col not in group_cols + [
            "label",
            "is_in_rosette",
            "image",
            "rosette_id",
            "rosette_nbr",
        ]:
            if f"avg_{col}" not in agg_funcs:
                agg_funcs[f"avg_{col}"] = (col, "mean")

    # Perform the aggregation
    animal_aggregated = combined_df.groupby(group_cols).agg(**agg_funcs).reset_index()

    # Calculate the final rosette percentage
    if (
        "total_cells" in animal_aggregated.columns
        and "rosette_cells" in animal_aggregated.columns
    ):
        animal_aggregated["rosette_percentage"] = (
            animal_aggregated["rosette_cells"] / animal_aggregated["total_cells"] * 100
        )

    return animal_aggregated


def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(
        description="Concatenate cell properties CSVs that already contain metadata."
    )
    parser.add_argument(
        "--input-file-list",
        required=True,
        help="A text file with one input CSV path per line.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the combined (cell-level) CSV file.",
    )
    parser.add_argument(
        "--output-aggregated",
        required=True,
        help="Output path for the animal-aggregated CSV file.",
    )

    args = parser.parse_args()

    # Read the list of file paths from the input text file
    try:
        with open(args.input_file_list, "r") as f:
            input_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file list not found at {args.input_file_list}")
        return 1

    # 1. Concatenate all cell-level data
    combined_df = concatenate_preprocessed_csvs(input_files)

    if combined_df.empty:
        print("No data was concatenated. Exiting.")
        return 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined_df.to_csv(args.output, index=False)
    print(
        f"✅ Saved combined cell-level data ({len(combined_df)} rows) to {args.output}"
    )

    # 2. Create and save the animal-aggregated data
    aggregated_df = create_animal_aggregated_data(combined_df)

    if not aggregated_df.empty:
        os.makedirs(os.path.dirname(args.output_aggregated), exist_ok=True)
        aggregated_df.to_csv(args.output_aggregated, index=False)
        print(
            f"✅ Saved aggregated animal-level data ({len(aggregated_df)} rows) to {args.output_aggregated}"
        )

    return 0


if __name__ == "__main__":
    exit(main())
