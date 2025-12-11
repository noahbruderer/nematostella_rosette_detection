#!/usr/bin/env python3
"""
Create Verification Plots for Inhibitor Experiments (Corrected)
=============================================================
Generates QC plots, now with corrected logic for identifying and counting unique images.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)


def plot_sample_counts(df: pd.DataFrame, output_dir: Path, inhibitor: str):
    """Generates a heatmap of unique image counts per condition."""
    print("Generating corrected sample count heatmap...")

    # Define unique images based on the full set of metadata columns
    # This correctly identifies a unique image capture
    count_df = (
        df.groupby(["replicate", "timepoint", "is_control", "animal", "image"])
        .size()
        .reset_index(name="n_cells")
    )

    # Now, count how many unique images exist for each experimental condition
    summary_counts = (
        count_df.groupby(["replicate", "timepoint", "is_control"])
        .size()
        .reset_index(name="unique_image_count")
    )

    pivot = summary_counts.pivot_table(
        index=["replicate", "is_control"],
        columns="timepoint",
        values="unique_image_count",
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": "Number of Unique Images"},
    )
    plt.title(
        f"Unique Image Counts for {inhibitor} Experiment",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "image_count_heatmap_corrected.png", dpi=300)
    plt.close()
    print("...heatmap saved as image_count_heatmap_corrected.png")


# --- The other functions (plot_rosette_statistics, etc.) remain the same ---
def plot_rosette_statistics(df: pd.DataFrame, output_dir: Path, inhibitor: str):
    """Plots rosette percentage, comparing control vs. treatment."""
    print("Plotting rosette statistics...")
    animal_agg = (
        df.groupby(["replicate", "timepoint", "animal", "is_control"])
        .agg(
            total_cells=("label", "count"),
            rosette_cells=("is_in_rosette", lambda x: x.sum()),
        )
        .reset_index()
    )
    animal_agg["rosette_percentage"] = (
        animal_agg["rosette_cells"] / animal_agg["total_cells"]
    ) * 100

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=animal_agg,
        x="timepoint",
        y="rosette_percentage",
        hue="is_control",
        palette={True: "skyblue", False: "salmon"},
    )
    plt.title(
        f"Rosette Percentage per Animal: {inhibitor} vs. Control",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Cells in Rosettes (%)")
    plt.xlabel("Timepoint")
    plt.legend(title="Is Control", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "rosette_percentage_boxplot.png", dpi=300)
    plt.close()


def plot_cell_area_distribution(df: pd.DataFrame, output_dir: Path, inhibitor: str):
    """Plots cell area distributions for control vs. treatment."""
    print("Plotting cell area distributions...")
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=df,
        x="timepoint",
        y="area",
        hue="is_control",
        split=True,
        inner="quart",
        palette={True: "skyblue", False: "salmon"},
    )
    plt.title(
        f"Cell Area Distribution: {inhibitor} vs. Control",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Cell Area (pixels²)")
    plt.xlabel("Timepoint")
    plt.yscale("log")
    plt.legend(title="Is Control", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "cell_area_violinplot.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create verification plots for inhibitor experiment data."
    )
    parser.add_argument("--input", required=True, help="Input combined CSV file.")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for plots."
    )
    parser.add_argument(
        "--inhibitor-name",
        required=True,
        help="Name of the inhibitor to filter and label plots.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading data from {args.input}...")
    try:
        full_df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- START FIX ---

    # 1. Clean 'inhibitor' column of whitespace
    if pd.api.types.is_string_dtype(full_df["inhibitor"]):
        full_df["inhibitor_clean"] = full_df["inhibitor"].str.strip()
    else:
        full_df["inhibitor_clean"] = full_df["inhibitor"].astype(str).str.strip()

    # 2. Convert 'is_control' to a reliable boolean
    if pd.api.types.is_string_dtype(full_df["is_control"]):
        full_df["is_control_bool"] = (
            full_df["is_control"].str.strip().str.lower() == "true"
        )
    else:
        full_df["is_control_bool"] = full_df["is_control"].astype(bool)

    print(f"Standardizing control rows for inhibitor: {args.inhibitor_name}...")

    # 3. Find ALL control rows that match this inhibitor
    control_mask = full_df["inhibitor_clean"].str.contains(
        args.inhibitor_name, case=False, na=False
    ) & (full_df["is_control_bool"] == True)

    # 4. Standardize the inhibitor name for these rows
    full_df.loc[control_mask, "inhibitor_clean"] = args.inhibitor_name

    # 5. Now, filter for the standardized inhibitor name
    print(f"Filtering for standardized inhibitor name: {args.inhibitor_name}")
    df_inhib = full_df[full_df["inhibitor_clean"] == args.inhibitor_name].copy()

    # 6. Ensure 'is_control' is boolean, as the plot functions expect it
    df_inhib["is_control"] = df_inhib["is_control_bool"]

    # --- END FIX ---

    if df_inhib.empty:
        print(f"Warning: No data found for inhibitor '{args.inhibitor_name}'. Exiting.")
        return

    try:
        df_inhib["timepoint_num"] = (
            df_inhib["timepoint"].str.replace("T", "").astype(int)
        )
        df_inhib = df_inhib.sort_values("timepoint_num").drop(columns=["timepoint_num"])
    except Exception:
        print("Could not sort timepoints numerically, using alphanumeric sorting.")

    # Now the plotting functions will receive the complete data
    plot_sample_counts(df_inhib, output_dir, args.inhibitor_name)
    plot_rosette_statistics(df_inhib, output_dir, args.inhibitor_name)
    plot_cell_area_distribution(df_inhib, output_dir, args.inhibitor_name)

    print(f"\n✅ Verification plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
