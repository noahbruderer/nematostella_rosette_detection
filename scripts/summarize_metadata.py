#!/usr/bin/env python3

"""
Summarizes the content of the master metadata TSV file. (v4)

This script reads the 'sample_metadata.tsv' and generates:
1. A summary table, grouped by experimental conditions.
2. A saved TSV file of this summary.
3. A set of overview plots visualizing the data distribution.

v4 changes:
- Reads the final 'sample_metadata.tsv' with 'unique_sample_id'.
- Aggregates by all experimental variables.
- Counts both total images (from 'unique_sample_id') and unique animals.
- Generates plots based on this new summary.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(metadata_file, output_dir):
    """
    Main execution function.
    """
    metadata_path = Path(metadata_file)
    output_path = Path(output_dir)

    # --- 1. Load Data ---
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Loading metadata from {metadata_path}...")
    try:
        df = pd.read_csv(metadata_path, sep="\t")
    except Exception as e:
        print(f"Error reading TSV file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Total raw image files found: {len(df)}")

    # --- 2. Create Summary Table ---

    # Define the "conditions" to group by
    group_keys = [
        "experiment_type",
        "replicate",
        "timepoint",
        "treatment",
        "is_control",
    ]

    print("Generating summary table by condition...")

    # Aggregate by the group keys
    summary_df = (
        df.groupby(group_keys)
        .agg(
            # Count images by counting the unique IDs
            total_images=("unique_sample_id", "size"),
            # Count unique animals using the animal_id column
            unique_animals=("animal_id", "nunique"),
        )
        .reset_index()
    )

    # Sort the summary
    summary_df = summary_df.sort_values(by=group_keys)

    # --- 3. Print and Save Summary ---
    print("\n" + "=" * 80)
    print("METADATA SUMMARY BY CONDITION")
    print("=" * 80)
    with pd.option_context("display.max_rows", None, "display.width", 1000):
        print(summary_df.to_string())
    print("=" * 80)

    # Save the summary to a file
    summary_file = output_path / "condition_summary.tsv"
    summary_df.to_csv(summary_file, sep="\t", index=False)
    print(f"\n✅ Summary table saved to: {summary_file}")

    # --- 4. Generate Visualizations ---
    print("Generating overview plots...")
    sns.set_theme(style="whitegrid")

    # Prep for plotting
    # Create a numeric sort key for timepoints
    summary_df["timepoint_num"] = pd.to_numeric(
        summary_df["timepoint"].str.replace("T", "").str.replace("D", ""),
        errors="coerce",
    )
    summary_df["timepoint_num"] = summary_df["timepoint_num"].fillna(-1)

    # --- Plot 1: Total Images per Experiment ---
    plot_df = (
        summary_df.groupby(["experiment_type", "replicate"])[
            ["total_images", "unique_animals"]
        ]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="experiment_type", y="total_images", hue="replicate")
    plt.title("Total Image Files by Experiment and Replicate")
    plt.ylabel("Total Image Files (all types)")
    plot1_path = output_path / "1_overview_total_images.png"
    plt.savefig(plot1_path, dpi=100, bbox_inches="tight")
    print(f"Saved: {plot1_path}")
    plt.close()

    # --- Plot 2: Unique Animals per Experiment ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="experiment_type", y="unique_animals", hue="replicate")
    plt.title("Unique Animals by Experiment and Replicate")
    plt.ylabel("Number of Unique Animals")
    plot2_path = output_path / "2_overview_unique_animals.png"
    plt.savefig(plot2_path, dpi=100, bbox_inches="tight")
    print(f"Saved: {plot2_path}")
    plt.close()

    # --- Plot 3: Timeseries & Constant Fed Detail ---
    df_ts = summary_df[
        summary_df["experiment_type"].isin(["timeseries", "constant_fed"])
    ].sort_values("timepoint_num")

    if not df_ts.empty:
        g = sns.catplot(
            data=df_ts,
            x="timepoint",
            y="total_images",
            col="replicate",
            row="experiment_type",
            kind="bar",
            sharex=False,
            height=4,
            aspect=1.5,
        )
        g.fig.suptitle("Image Counts: Timeseries & Constant Fed", y=1.03)
        g.set_axis_labels("Timepoint / Day", "Total Image Files")
        g.set_xticklabels(rotation=45)
        plot3_path = output_path / "3_detail_timeseries_cf_counts.png"
        g.savefig(plot3_path, dpi=100, bbox_inches="tight")
        print(f"Saved: {plot3_path}")
        plt.close()

    # --- Plot 4: Inhibitor Detail ---
    df_inhib = summary_df[summary_df["experiment_type"] == "inhibitor"].sort_values(
        "timepoint_num"
    )

    if not df_inhib.empty:
        g = sns.catplot(
            data=df_inhib,
            x="treatment",
            y="total_images",
            col="timepoint",
            row="replicate",
            kind="bar",
            sharex=False,
            height=4,
            aspect=1.2,
        )
        g.fig.suptitle("Image Counts: Inhibitor Experiments", y=1.03)
        g.set_axis_labels("Treatment", "Total Image Files")
        g.set_xticklabels(rotation=90)
        plot4_path = output_path / "4_detail_inhibitor_counts.png"
        g.savefig(plot4_path, dpi=100, bbox_inches="tight")
        print(f"Saved: {plot4_path}")
        plt.close()

    print("\n✅ Summary generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize the master metadata TSV file."
    )
    parser.add_argument(
        "-m",
        "--metadata-file",
        default="config/sample_metadata.tsv",
        help="Path to the 'sample_metadata.tsv' file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="results/metadata_summary",
        help="Directory to save the output table and plots.",
    )

    args = parser.parse_args()
    main(args.metadata_file, args.output_dir)
