#!/usr/bin/env python3
"""
Summarize Variance Components with LMM and Exploratory Plots
=============================================================
This script provides a statistically robust analysis of the sources of variance
in the time series data. It uses a Linear Mixed-Effects Model (LMM) as the
primary method and supplements it with clear exploratory data visualizations.

Author: Gemini, for the Rosette Pipeline
Date: 2025
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM

sns.set_theme(style="whitegrid", context="talk")


class VarianceAnalyzer:
    """A class to analyze and visualize sources of variance in cell data."""

    def __init__(self, raw_data_path: str, output_dir: str):
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.df = None
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(
            f"Variance Analyzer initialized. Results will be saved to: {self.output_dir}"
        )

    def load_and_prepare_data(self):
        """Loads and prepares the raw cell data for analysis."""
        print(f"--- 1. Loading and preparing data from {self.raw_data_path} ---")
        try:
            self.df = pd.read_csv(self.raw_data_path)
        except FileNotFoundError:
            print(f"❌ ERROR: Raw data file not found at {self.raw_data_path}.")
            return False

        # Use log-transformed area as the response variable for better normality
        self.df["log_area"] = np.log1p(self.df["area"])

        # Create a numerical timepoint for the model's fixed effect
        self.df["timepoint_num"] = (
            self.df["timepoint"].str.extract(r"(\d+)").astype(int)
        )

        # Define categorical variables for random effects
        self.df["animal"] = self.df["animal"].astype("category")
        self.df["unique_image_id"] = (
            self.df["replicate"].astype(str)
            + "_"
            + self.df["animal"].astype(str)
            + "_"
            + self.df["image"].astype(str)
        ).astype("category")

        print(f"✅ Data loaded and prepared with {len(self.df)} cell measurements.")
        return True

    def run_lmm_analysis(self):
        """
        Fits a Linear Mixed-Effects Model to partition variance and saves the results.
        This is the primary statistical analysis.
        """
        if self.df is None:
            return

        print("\n--- 2. Fitting Linear Mixed-Effects Model (LMM) ---")

        # Determine the correct model structure based on the data hierarchy
        images_per_animal = self.df.groupby("animal")["unique_image_id"].nunique()
        has_nested_images = (images_per_animal > 1).any()

        if has_nested_images:
            # Full nested model: variance from animal, and from image within animal
            model_formula = "log_area ~ timepoint_num"
            groups = self.df['animal']
            print("INFO: Using nested random effects model for Animal -> Image.")
            model = MixedLM.from_formula(model_formula, data=self.df, groups=groups)
        else:
            # Simpler model: variance from animal only
            model_formula = "log_area ~ timepoint_num"
            groups = self.df['animal']
            print("INFO: Using simple random effects model for Animal.")
            model = MixedLM.from_formula(model_formula, data=self.df, groups=groups)

        # Fit the model
        result = model.fit()

        # Save the detailed model summary
        summary_path = self.output_dir / "lmm_model_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Model Formula: {model_formula}\n\n")
            f.write(str(result.summary()))
        print(f"✅ Full LMM summary saved to: {summary_path}")

        # Extract and interpret variance components
        variances = {"Cell-to-Cell (Residual)": result.scale}
        
        # Check if variance components were estimated
        if len(result.vcomp) == 0:
            print("WARNING: Variance components were not estimated (model convergence issue)")
            print("Using only residual variance for analysis")
            # Create dummy variance components for plotting
            if has_nested_images:
                variances["Between Animals"] = 0.0
                variances["Between Images (within Animal)"] = 0.0
            else:
                variances["Between Animals"] = 0.0
        else:
            if has_nested_images:
                variances["Between Animals"] = result.vcomp[0] if len(result.vcomp) > 0 else 0.0
                variances["Between Images (within Animal)"] = result.vcomp[1] if len(result.vcomp) > 1 else 0.0
            else:
                variances["Between Animals"] = result.vcomp[0] if len(result.vcomp) > 0 else 0.0

        vc_df = pd.DataFrame.from_dict(variances, orient="index", columns=["Variance"])
        vc_df["Proportion (%)"] = (vc_df["Variance"] / vc_df["Variance"].sum()) * 100
        print("\nStatistical Partition of Variance (from LMM):")
        print(vc_df.to_string(float_format="%.4f"))
        vc_df.to_csv(self.output_dir / "lmm_variance_components.csv")
        # Plot the variance proportions
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(x=vc_df.index, y=vc_df["Proportion (%)"], palette="plasma")
        ax.set_title(
            "Statistical Hierarchy of Variance (from LMM)", fontsize=18, weight="bold"
        )
        ax.set_ylabel("Percentage of Total Variance (%)")
        ax.set_xlabel("Source of Variance")
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )

        plot_path = self.output_dir / "lmm_variance_components_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ LMM variance plot saved to: {plot_path}")

    def create_exploratory_plot(self):
        """
        Creates an exploratory plot to directly visualize the data hierarchy.
        This replaces the old, misleading heatmap.
        """
        if self.df is None:
            return

        print("\n--- 3. Generating Exploratory Data Visualization ---")
        plt.figure(figsize=(16, 9))

        # Use a boxplot to show the distribution at each timepoint
        sns.boxplot(
            data=self.df,
            x="timepoint_num",
            y="log_area",
            color="lightgray",
            showfliers=False,  # Hide outlier points as the stripplot will show the distribution
        )
        # Overlay a stripplot to show the mean of each animal
        sns.stripplot(
            data=self.df.groupby(["timepoint_num", "animal"])["log_area"]
            .mean()
            .reset_index(),
            x="timepoint_num",
            y="log_area",
            color="black",
            size=8,
            jitter=0.2,
            edgecolor="white",
            linewidth=1,
            label="Animal Mean",
        )

        plt.title("Exploratory View of Cell Size Variance", fontsize=18, weight="bold")
        plt.xlabel("Timepoint")
        plt.ylabel("Log-Transformed Cell Area")

        # Use original timepoint labels for the x-axis
        sorted_timepoints = self.df.sort_values("timepoint_num")["timepoint"].unique()
        plt.xticks(ticks=range(len(sorted_timepoints)), labels=sorted_timepoints)
        plt.legend()

        plot_path = self.output_dir / "exploratory_variance_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Exploratory plot saved to: {plot_path}")

    def run_analysis(self):
        """Runs the full analysis pipeline."""
        if self.load_and_prepare_data():
            self.run_lmm_analysis()
            self.create_exploratory_plot()


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and visualize variance components from cell analysis using LMM."
    )
    # The script now only needs one input: the raw, combined data.
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the combined, raw cell properties CSV file (e.g., combined_timeseries_cell_properties.csv).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the output plots and summaries.",
    )
    args = parser.parse_args()

    analyzer = VarianceAnalyzer(
        raw_data_path=args.input_csv, output_dir=args.output_dir
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
