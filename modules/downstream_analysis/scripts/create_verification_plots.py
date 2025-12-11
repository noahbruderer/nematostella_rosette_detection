#!/usr/bin/env python3
"""
Create Verification Plots
=========================

Creates verification plots and charts for concatenated cell properties data.

Author: Noah Bruderer
Date: 2025
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _sort_timepoints(df):
    """
    Sorts the dataframe by timepoint in natural order (e.g., T0, T2, T10).
    If the 'timepoint' column exists, it converts it to a sorted categorical type.
    """
    if "timepoint" in df.columns:
        # Create a copy to avoid SettingWithCopyWarning
        timepoints = df["timepoint"].unique()
        # Sort using a key that extracts the number from the string
        try:
            sorted_timepoints = sorted(
                timepoints, key=lambda x: int(re.search(r"\d+", str(x)).group())
            )
            tp_cat = pd.CategoricalDtype(categories=sorted_timepoints, ordered=True)
            df["timepoint"] = df["timepoint"].astype(tp_cat)
            print("Timepoints sorted in natural order.")
        except (AttributeError, TypeError):
            print(
                "Could not parse numbers from timepoints for natural sorting. Using default sorting."
            )
    return df


def create_image_count_verification(df, output_dir, experiment_type="regular"):
    """
    Create verification plots showing image counts per animal.

    Args:
        df: Combined dataframe
        output_dir: Output directory for plots
        experiment_type: Type of experiment
    """
    verification_dir = os.path.join(output_dir, f"{experiment_type}_input_verification")
    os.makedirs(verification_dir, exist_ok=True)

    print(f"Creating image count verification for {experiment_type} experiment...")

    if "image" not in df.columns:
        print("No 'image' column found - skipping image verification")
        return

    group_cols = ["replicate", "timepoint", "animal"]
    if experiment_type == "inhibitor":
        group_cols.extend(["inhibitor", "is_control"])

    # Count images per animal
    image_counts = df.groupby(group_cols)["image"].nunique().reset_index()
    image_counts.rename(columns={"image": "image_count"}, inplace=True)

    # Create summary statistics
    stats_df = (
        image_counts.groupby(
            ["replicate", "timepoint"]
            + (["inhibitor", "is_control"] if experiment_type == "inhibitor" else [])
        )["image_count"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )

    stats_path = os.path.join(verification_dir, "image_count_statistics.csv")
    stats_df.to_csv(stats_path, index=False)

    # Create histogram of image counts
    plt.figure(figsize=(10, 6))

    if experiment_type == "inhibitor" and "is_control" in df.columns:
        sns.histplot(
            data=image_counts,
            x="image_count",
            hue="is_control",
            palette={True: "royalblue", False: "darkorange"},
            multiple="dodge",
        )
        plt.legend(title="Condition", labels=["Control", "Treatment"])
    else:
        sns.histplot(data=image_counts, x="image_count", color="teal")

    plt.title("Distribution of Image Counts per Animal", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Images per Animal", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    histogram_path = os.path.join(verification_dir, "image_count_histogram.png")
    plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create summary bar chart by timepoint (will be correctly ordered)
    plt.figure(figsize=(12, 8))

    # Group by the now-ordered timepoint column
    timepoint_summary = (
        image_counts.groupby("timepoint", observed=True)["image_count"]
        .mean()
        .reset_index()
    )

    sns.barplot(
        data=timepoint_summary, x="timepoint", y="image_count", palette="viridis"
    )

    for p in plt.gca().patches:
        plt.annotate(
            f"{p.get_height():.1f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontweight="bold",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.title(
        "Average Images per Animal by Timepoint (Sorted)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Timepoint", fontsize=12)
    plt.ylabel("Average Number of Images", fontsize=12)
    plt.ylim(0, plt.ylim()[1] * 1.1)  # Add space for labels
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    summary_path = os.path.join(verification_dir, "timepoint_summary_sorted.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created image verification plots in {verification_dir}")


def create_rosette_statistics_plots(df, output_dir, experiment_type="regular"):
    """
    Create rosette statistics plots.

    Args:
        df: Combined dataframe
        output_dir: Output directory for plots
        experiment_type: Type of experiment
    """
    rosette_dir = os.path.join(output_dir, f"{experiment_type}_rosette_statistics")
    os.makedirs(rosette_dir, exist_ok=True)

    print("Creating rosette statistics plots...")

    if "is_in_rosette" not in df.columns:
        print("No 'is_in_rosette' column found - skipping rosette statistics")
        return

    # Overall rosette statistics
    total_cells = len(df)
    rosette_cells = df["is_in_rosette"].sum()
    rosette_percentage = (rosette_cells / total_cells) * 100 if total_cells > 0 else 0

    summary_text = f"""
ROSETTE STATISTICS SUMMARY
========================

Total Cells: {total_cells:,}
Cells in Rosettes: {rosette_cells:,}
Rosette Percentage: {rosette_percentage:.2f}%
"""

    with open(os.path.join(rosette_dir, "rosette_summary.txt"), "w") as f:
        f.write(summary_text)

    # Rosette percentage by timepoint (will be correctly ordered)
    if "timepoint" in df.columns:
        timepoint_stats = (
            df.groupby("timepoint", observed=True)
            .agg(
                total_cells=("is_in_rosette", "count"),
                rosette_cells=("is_in_rosette", "sum"),
            )
            .reset_index()
        )
        timepoint_stats["rosette_percentage"] = (
            timepoint_stats["rosette_cells"] / timepoint_stats["total_cells"] * 100
        )

        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=timepoint_stats,
            x="timepoint",
            y="rosette_percentage",
            palette="plasma",
        )

        for p in plt.gca().patches:
            plt.annotate(
                f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontweight="bold",
                xytext=(0, 5),
                textcoords="offset points",
            )

        plt.title(
            "Rosette Percentage by Timepoint (Sorted)", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Timepoint", fontsize=12)
        plt.ylabel("Rosette Percentage (%)", fontsize=12)
        plt.ylim(0, plt.ylim()[1] * 1.1)  # Add space for labels
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        timepoint_path = os.path.join(rosette_dir, "rosette_by_timepoint_sorted.png")
        plt.savefig(timepoint_path, dpi=300, bbox_inches="tight")
        plt.close()

    # **NEW**: Heatmap of rosettes per 1000 cells by replicate and timepoint
    if "replicate" in df.columns and "timepoint" in df.columns:
        print("Creating rosette heatmap per 1000 cells...")
        rosette_heatmap_data = (
            df.groupby(["replicate", "timepoint"], observed=True)
            .agg(
                total_cells=("is_in_rosette", "count"),
                rosette_cells=("is_in_rosette", "sum"),
            )
            .reset_index()
        )

        rosette_heatmap_data["rosettes_per_1000_cells"] = (
            rosette_heatmap_data["rosette_cells"]
            / rosette_heatmap_data["total_cells"]
            * 1000
        )

        pivot_rosettes = rosette_heatmap_data.pivot(
            index="replicate",
            columns="timepoint",
            values="rosettes_per_1000_cells",
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_rosettes,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            linewidths=0.5,
            linecolor="black",
            cbar_kws={"label": "Rosette Cells per 1000 Cells"},
        )
        plt.title(
            "Rosette Cells per 1000 Cells by Replicate and Timepoint",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Timepoint", fontsize=12)
        plt.ylabel("Replicate", fontsize=12)
        plt.yticks(rotation=0)
        plt.tight_layout()

        heatmap_path = os.path.join(rosette_dir, "rosette_heatmap_per_1000_cells.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()

    # For inhibitor experiments, create control vs treatment comparison
    if experiment_type == "inhibitor" and "is_control" in df.columns:
        # (Rest of the function remains the same...)
        treatment_stats = (
            df.groupby("is_control")
            .agg(
                total_cells=("is_in_rosette", "count"),
                rosette_cells=("is_in_rosette", "sum"),
            )
            .reset_index()
        )
        treatment_stats["rosette_percentage"] = (
            treatment_stats["rosette_cells"] / treatment_stats["total_cells"] * 100
        )

        plt.figure(figsize=(8, 6))
        colors = {True: "royalblue", False: "darkorange"}
        bars = plt.bar(
            [0, 1],
            treatment_stats["rosette_percentage"],
            color=[colors[True], colors[False]],
            alpha=0.8,
        )

        for bar, percentage in zip(bars, treatment_stats["rosette_percentage"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{percentage:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                xytext=(0, 5),
                textcoords="offset points",
            )

        plt.xticks([0, 1], ["Control", "Treatment"])
        plt.title(
            "Rosette Percentage: Control vs Treatment", fontsize=14, fontweight="bold"
        )
        plt.ylabel("Rosette Percentage (%)", fontsize=12)
        plt.ylim(0, plt.ylim()[1] * 1.1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        treatment_path = os.path.join(rosette_dir, "control_vs_treatment.png")
        plt.savefig(treatment_path, dpi=300, bbox_inches="tight")
        plt.close()

        treatment_stats["condition"] = treatment_stats["is_control"].map(
            {True: "Control", False: "Treatment"}
        )
        treatment_stats.to_csv(
            os.path.join(rosette_dir, "treatment_statistics.csv"), index=False
        )

    print(f"Created rosette statistics plots in {rosette_dir}")


def create_cell_count_plots(df, output_dir, experiment_type="regular"):
    """
    Create cell count analysis plots.

    Args:
        df: Combined dataframe
        output_dir: Output directory for plots
        experiment_type: Type of experiment
    """
    cell_count_dir = os.path.join(output_dir, f"{experiment_type}_cell_counts")
    os.makedirs(cell_count_dir, exist_ok=True)

    print("Creating cell count plots...")

    # Cell counts by animal
    if "animal" in df.columns and "timepoint" in df.columns:
        animal_counts = (
            df.groupby(["replicate", "timepoint", "animal"], observed=True)
            .size()
            .reset_index(name="cell_count")
        )

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=animal_counts, x="timepoint", y="cell_count", palette="Set2")
        plt.title(
            "Cell Count Distribution by Timepoint (Sorted)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Timepoint", fontsize=12)
        plt.ylabel("Cells per Animal", fontsize=12)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        boxplot_path = os.path.join(cell_count_dir, "cell_count_boxplot_sorted.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        plt.close()

        animal_counts.to_csv(
            os.path.join(cell_count_dir, "animal_cell_counts.csv"), index=False
        )

    print(f"Created cell count plots in {cell_count_dir}")


def create_sample_overview(df, output_dir, experiment_type="regular"):
    """
    Create sample overview plots and statistics.

    Args:
        df: Combined dataframe
        output_dir: Output directory for plots
        experiment_type: Type of experiment
    """
    overview_dir = os.path.join(output_dir, f"{experiment_type}_overview")
    os.makedirs(overview_dir, exist_ok=True)

    print("Creating sample overview...")

    if "replicate" not in df.columns or "timepoint" not in df.columns:
        print(
            "Skipping overview plots due to missing 'replicate' or 'timepoint' columns."
        )
        return

    # Sample counts by replicate and timepoint
    sample_counts = (
        df.groupby(["replicate", "timepoint"], observed=True)
        .agg(
            unique_samples=("sample", "nunique"),
            total_cells=("sample", "count"),
            unique_animals=("animal", "nunique")
            if "animal" in df.columns
            else ("sample", lambda x: 0),
        )
        .reset_index()
    )

    # Create heatmap of sample counts (with better scaling)
    if not sample_counts.empty:
        pivot_samples = sample_counts.pivot(
            index="replicate", columns="timepoint", values="unique_samples"
        )

        # **IMPROVED**: Wider figure for better column spacing
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            pivot_samples,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.5,
            linecolor="black",
            cbar_kws={"label": "Number of Unique Samples"},
        )
        plt.title(
            "Number of Samples by Replicate and Timepoint",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Timepoint", fontsize=12)
        plt.ylabel("Replicate", fontsize=12)
        plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
        plt.tight_layout()

        heatmap_path = os.path.join(overview_dir, "sample_count_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()

    sample_counts.to_csv(os.path.join(overview_dir, "sample_overview.csv"), index=False)

    # (Rest of the function remains the same...)
    summary_stats = {
        "total_rows": len(df),
        "unique_samples": df["sample"].nunique() if "sample" in df.columns else 0,
        "unique_timepoints": df["timepoint"].nunique()
        if "timepoint" in df.columns
        else 0,
        "unique_replicates": df["replicate"].nunique()
        if "replicate" in df.columns
        else 0,
        "unique_animals": df["animal"].nunique() if "animal" in df.columns else 0,
    }

    if experiment_type == "inhibitor":
        summary_stats.update(
            {
                "unique_inhibitors": df["inhibitor"].nunique()
                if "inhibitor" in df.columns
                else 0,
                "control_samples": len(df[df["is_control"] == True])
                if "is_control" in df.columns
                else 0,
                "treatment_samples": len(df[df["is_control"] == False])
                if "is_control" in df.columns
                else 0,
            }
        )

    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(os.path.join(overview_dir, "experiment_summary.csv"), index=False)

    print(f"Created overview in {overview_dir}")


def main():
    """Main function for Snakemake integration."""
    parser = argparse.ArgumentParser(
        description="Create verification plots for concatenated cell properties data"
    )
    parser.add_argument("--input", required=True, help="Input combined CSV file")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--experiment-type",
        choices=["regular", "inhibitor", "timeseries"],
        default="regular",
        help="Type of experiment",
    )
    parser.add_argument("--flag-file", help="Flag file to create when complete")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print(f"Loading data from {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    if df.empty:
        print("No data to process!")
        # Create flag file even if empty to satisfy Snakemake
        if args.flag_file:
            with open(args.flag_file, "w") as f:
                f.write("Completed with no data to process.\n")
        return 0
    print("Enforcing consistent string types for identifier columns...")
    for col in ["replicate", "timepoint", "animal", "image", "sample"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    # **NEW**: Apply natural sorting for timepoints globally
    df = _sort_timepoints(df)

    try:
        create_image_count_verification(df, args.output_dir, args.experiment_type)
        create_rosette_statistics_plots(df, args.output_dir, args.experiment_type)
        create_cell_count_plots(df, args.output_dir, args.experiment_type)
        create_sample_overview(df, args.output_dir, args.experiment_type)

        print(f"All verification plots created successfully in {args.output_dir}")

    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback

        traceback.print_exc()
        return 1

    if args.flag_file:
        with open(args.flag_file, "w") as f:
            f.write(
                f"Verification plots completed successfully for {args.experiment_type} experiment\n"
            )
            f.write(f"Generated at: {pd.Timestamp.now(tz='UTC')}\n")
            f.write(f"Input data: {len(df)} rows\n")

    return 0


if __name__ == "__main__":
    exit(main())
