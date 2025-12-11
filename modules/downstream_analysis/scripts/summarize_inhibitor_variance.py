#!/usr/bin/env python3
"""
Summarize and Compare Control vs. Treatment Effects for an Inhibitor
===================================================================
This script takes the detailed comparison summary from the advanced inhibitor
analysis, calculates the overall average effects, and produces a final
summary CSV and a corresponding heatmap visualization.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


class InhibitorEffectSummarizer:
    """A class to summarize and visualize the effects of an inhibitor."""

    def __init__(self, input_csv: str, output_dir: str, inhibitor_name: str):
        self.input_path = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.inhibitor_name = inhibitor_name
        self.df = None
        self.summary_df = None
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Summarizer initialized for inhibitor '{self.inhibitor_name}'.")

    def load_data(self) -> bool:
        """Loads the summary data from the input CSV."""
        print(f"--- 1. Loading data from {self.input_path} ---")
        try:
            self.df = pd.read_csv(self.input_path)
            if self.df.empty:
                print(
                    "⚠️ Warning: The input CSV is empty. No summary will be generated."
                )
                return False
            print("✅ Data loaded successfully.")
            return True
        except FileNotFoundError:
            print(f"❌ ERROR: Input file not found at {self.input_path}")
            return False

    def calculate_summary_metrics(self):
        """Calculates the final summary metrics for the heatmap."""
        print("--- 2. Calculating summary metrics ---")

        # Calculate the percent change in the mean size for each cell population
        self.df["pct_change_mean_pop1"] = (
            (self.df["treatment_mean_pop1"] - self.df["control_mean_pop1"])
            / self.df["control_mean_pop1"]
        ) * 100
        self.df["pct_change_mean_pop2"] = (
            (self.df["treatment_mean_pop2"] - self.df["control_mean_pop2"])
            / self.df["control_mean_pop2"]
        ) * 100

        # Calculate the change in the proportion (weight) of the small cell population
        # This is an absolute change in percentage points
        self.df["proportion_change_pop1"] = (
            self.df["treatment_weight_pop1"] - self.df["control_weight_pop1"]
        ) * 100

        # Average these effects across all replicate/timepoint comparisons
        summary_data = {
            "Avg. % Change in Small Cell Size": self.df["pct_change_mean_pop1"].mean(),
            "Avg. % Change in Large Cell Size": self.df["pct_change_mean_pop2"].mean(),
            "Avg. Change in Small Cell Proportion (pp)": self.df[
                "proportion_change_pop1"
            ].mean(),
        }

        self.summary_df = pd.DataFrame([summary_data])
        print("✅ Summary metrics calculated.")
        print(self.summary_df.to_string(index=False, float_format="%.2f"))

    def save_summary_csv(self):
        """Saves the final summary data to a CSV file for reproducibility."""
        if self.summary_df is None:
            return

        print("--- 3. Saving summary data to CSV ---")
        output_path = self.output_dir / f"{self.inhibitor_name}_effect_summary.csv"
        self.summary_df.to_csv(output_path, index=False, float_format="%.4f")
        print(f"✅ Summary data saved to: {output_path}")

    def create_summary_heatmap(self):
        """Generates and saves a heatmap of the summary effects."""
        if self.summary_df is None:
            return

        print("--- 4. Generating summary heatmap ---")
        plt.figure(figsize=(12, 5))
        sns.heatmap(
            self.summary_df,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",  # Diverging colormap is perfect for showing positive/negative effects
            center=0,
            linewidths=0.5,
            cbar_kws={"label": "Average Change vs. Control"},
        )
        plt.title(
            f"Summary of '{self.inhibitor_name}' Effect vs. Control",
            fontsize=16,
            weight="bold",
        )
        plt.yticks(rotation=0)  # Keep the single row label horizontal

        output_path = (
            self.output_dir / f"{self.inhibitor_name}_effect_summary_heatmap.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Heatmap saved to: {output_path}")

    def run_summary(self):
        """Executes the full summary pipeline."""
        if self.load_data():
            self.calculate_summary_metrics()
            self.save_summary_csv()
            self.create_summary_heatmap()


def main():
    parser = argparse.ArgumentParser(
        description="Summarize inhibitor experiment results and create a summary heatmap."
    )
    parser.add_argument(
        "--input-csv", required=True, help="Path to 'control_vs_treatment_summary.csv'."
    )
    parser.add_argument(
        "--inhibitor-name", required=True, help="Name of the inhibitor being analyzed."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save the outputs."
    )
    args = parser.parse_args()

    summarizer = InhibitorEffectSummarizer(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        inhibitor_name=args.inhibitor_name,
    )
    summarizer.run_summary()


if __name__ == "__main__":
    main()
