#!/usr/bin/env python3
"""
Hybrid Advanced Inhibitor Analysis
==================================
Combines per-image GMM fitting with population-level control-vs-treatment comparisons.
- Fits GMM to each unique image and saves parameters.
- Filters outliers on a per-image basis.
- Aggregates cell populations to compare Control vs. Treatment groups.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.mixture import GaussianMixture

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)


class HybridInhibitorAnalyzer:
    """A class to perform comprehensive analysis of inhibitor experiments."""

    def __init__(self, data_path: str, output_dir: str, inhibitor_name: str):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.inhibitor_name = inhibitor_name
        self.df = None
        self.gmm_params = {}  # Stores per-image GMM results
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Analyzer initialized for inhibitor '{self.inhibitor_name}'.")
        print(f"Results will be saved to '{self.output_dir}'.")

    def load_data(self):
        """Loads data, filters for the specific inhibitor, and defines unique images."""
        print(f"Loading data from {self.data_path}...")
        try:
            full_df = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"❌ ERROR: Could not read file: {e}")
            raise

        # --- NEW LOGIC (v6) ---

        # 1. CRITICAL: Clean the 'inhibitor' column first!
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

        print(f"Standardizing control rows for inhibitor: {self.inhibitor_name}...")

        # 3. Find ALL control rows that match this inhibitor
        control_mask = full_df["inhibitor_clean"].str.contains(
            self.inhibitor_name, case=False, na=False
        ) & (full_df["is_control_bool"] == True)

        # 4. Standardize the inhibitor name for these rows *before* filtering.
        full_df.loc[control_mask, "inhibitor_clean"] = self.inhibitor_name

        # 5. Now, the main filter is simple.
        print(f"Filtering for standardized inhibitor name: {self.inhibitor_name}...")
        self.df = full_df[full_df["inhibitor_clean"] == self.inhibitor_name].copy()

        # Clean up temporary columns
        self.df = self.df.drop(columns=["inhibitor_clean", "is_control_bool"])
        # --- END NEW LOGIC ---

        if self.df.empty:
            raise ValueError(f"No data found for inhibitor '{self.inhibitor_name}'.")

        # The rest of the script expects strings for "is_control"
        self.df["is_control"] = self.df["is_control"].astype(str)

        # Create a robust unique identifier for each image
        for col in ["replicate", "timepoint", "animal", "image", "inhibitor"]:
            self.df[col] = self.df[col].astype(str)

        self.df["unique_image_id"] = self.df[
            ["replicate", "timepoint", "animal", "image", "inhibitor", "is_control"]
        ].agg("_".join, axis=1)

        print(f"Found {self.df['unique_image_id'].nunique()} unique images to analyze.")

        # Final check to ensure we have both treatments and controls
        if not (self.df["is_control"] == "True").any():
            print(
                f"⚠️  WARNING: No CONTROL data was found for inhibitor '{self.inhibitor_name}'."
            )
        if not (self.df["is_control"] == "False").any():
            print(
                f"⚠️  WARNING: No TREATMENT data was found for inhibitor '{self.inhibitor_name}'."
            )

        return self.df

    def filter_outlier_cells(self, percentile=99.99):
        """Filter out the largest cells (outliers) from each image individually."""
        if self.df is None:
            self.load_data()

        print(
            f"Filtering out cells larger than the {percentile}th percentile for each image..."
        )
        original_count = len(self.df)

        # Group by the unique image ID and apply the percentile filter to each group
        self.df = (
            self.df.groupby("unique_image_id")
            .apply(lambda g: g[g.area <= g.area.quantile(percentile / 100.0)])
            .reset_index(drop=True)
        )

        removed_count = original_count - len(self.df)
        print(
            f"Removed {removed_count} cells ({removed_count / original_count:.2%}) as outliers."
        )
        return self.df

    def fit_gmm_per_image(self, n_components=2, min_cells=10):
        """Fit a GMM to each unique image after log-transformation."""
        if self.df is None:
            self.load_data()

        print(f"Fitting {n_components}-component GMM to each unique image...")
        for image_id, group in self.df.groupby("unique_image_id"):
            if len(group) < min_cells:
                continue

            # Log-transform area for GMM fitting
            log_areas = np.log1p(group["area"].values).reshape(-1, 1)
            gmm = GaussianMixture(n_components=n_components, random_state=42).fit(
                log_areas
            )

            idx = np.argsort(gmm.means_.flatten())
            log_means, log_vars, weights = (
                gmm.means_.flatten()[idx],
                gmm.covariances_.flatten()[idx],
                gmm.weights_[idx],
            )

            # Store everything needed for later
            self.gmm_params[image_id] = {
                "params": group.iloc[0],  # Full metadata row
                "log_means": log_means,
                "log_variances": log_vars,
                "weights": weights,
                "cell_data": group["area"].values,
                "n_samples": len(group),
            }
        print(f"Successfully fitted GMMs for {len(self.gmm_params)} images.")

    def save_gmm_params_to_csv(self):
        """Saves the GMM parameters from each image fit to a CSV file."""
        if not self.gmm_params:
            print("No GMM parameters to save. Run fit_gmm_per_image() first.")
            return

        rows = []
        for image_id, data in self.gmm_params.items():
            meta = data["params"]
            row = {
                "unique_image_id": image_id,
                "replicate": meta["replicate"],
                "timepoint": meta["timepoint"],
                "animal": meta["animal"],
                "image": meta["image"],
                "inhibitor": meta["inhibitor"],
                "is_control": meta["is_control"],
                "n_samples": data["n_samples"],
            }
            # Add GMM params for each component
            for i in range(len(data["log_means"])):
                row[f"log_mean_{i}"] = data["log_means"][i]
                row[f"log_variance_{i}"] = data["log_variances"][i]
                row[f"weight_{i}"] = data["weights"][i]
                # Also save mean on original scale
                row[f"mean_{i}"] = np.expm1(data["log_means"][i])
            rows.append(row)

        df = pd.DataFrame(rows)
        output_path = self.output_dir / "gmm_parameters_per_image.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved per-image GMM parameters for {len(df)} images to {output_path}")

    # In scripts/advanced_inhibitor_analysis.py

    def compare_control_vs_treatment(self):
        """Aggregates per-image data to compare control vs. treatment at the population level."""
        if not self.gmm_params:
            self.fit_gmm_per_image()

        comparison_results = []

        # Group images by replicate and timepoint
        grouped_images = {}
        for image_id, data in self.gmm_params.items():
            key = (data["params"]["replicate"], data["params"]["timepoint"])
            if key not in grouped_images:
                grouped_images[key] = []
            grouped_images[key].append(data)

        for (replicate, timepoint), all_images_in_group in grouped_images.items():
            print(
                f"\n--- Comparing Control vs Treatment for Replicate: {replicate}, Timepoint: {timepoint} ---"
            )

            # --- START: MODIFIED SECTION ---

            # First, collect the data arrays without concatenating immediately
            control_data_list = [
                d["cell_data"]
                for d in all_images_in_group
                if d["params"]["is_control"] == "True"
            ]
            treatment_data_list = [
                d["cell_data"]
                for d in all_images_in_group
                if d["params"]["is_control"] == "False"
            ]

            # Now, check if either list is empty. If so, we can't compare.
            if not control_data_list or not treatment_data_list:
                print(
                    f"Skipping: Missing control (found {len(control_data_list)}) or "
                    f"treatment (found {len(treatment_data_list)}) data for this group."
                )
                continue

            # If both lists have data, proceed with concatenation
            control_cells = np.concatenate(control_data_list)
            treatment_cells = np.concatenate(treatment_data_list)

            if len(control_cells) < 20 or len(treatment_cells) < 20:
                print(
                    f"Skipping: Insufficient cell counts (Control: {len(control_cells)}, Treatment: {len(treatment_cells)})."
                )
                continue

            # --- END: MODIFIED SECTION ---

            # This logic is from the previous script to analyze the aggregated populations
            control_pop_params = self._fit_gmm_to_population(control_cells)
            treatment_pop_params = self._fit_gmm_to_population(treatment_cells)
            stats = self._perform_statistical_tests(
                control_cells, treatment_cells, control_pop_params, treatment_pop_params
            )

            result = {
                "replicate": replicate,
                "timepoint": timepoint,
                **{f"control_{k}": v for k, v in control_pop_params.items()},
                **{f"treatment_{k}": v for k, v in treatment_pop_params.items()},
                **stats,
            }
            comparison_results.append(result)
            self._visualize_population_comparison(result)

        results_df = pd.DataFrame(comparison_results)
        results_df.to_csv(
            self.output_dir / "control_vs_treatment_summary.csv", index=False
        )
        print("\n✅ Population-level analysis complete. Summary CSV saved.")
        return results_df

    # --- Helper methods for population analysis (from previous script) ---
    def _fit_gmm_to_population(self, cell_areas, n_components=2):
        log_areas = np.log1p(cell_areas).reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, random_state=42).fit(log_areas)
        idx = np.argsort(gmm.means_.flatten())
        log_means, log_vars, weights = (
            gmm.means_.flatten()[idx],
            gmm.covariances_.flatten()[idx],
            gmm.weights_[idx],
        )
        labels = gmm.predict(log_areas)
        sorted_labels = np.zeros_like(labels)
        for i, original_idx in enumerate(idx):
            sorted_labels[labels == original_idx] = i
        return {
            "n_cells": len(cell_areas),
            "mean_pop1": np.expm1(log_means[0]),
            "mean_pop2": np.expm1(log_means[1]),
            "log_mean_pop1": log_means[0],
            "log_mean_pop2": log_means[1],
            "log_var_pop1": log_vars[0],
            "log_var_pop2": log_vars[1],
            "weight_pop1": weights[0],
            "weight_pop2": weights[1],
            "count_pop1": np.sum(sorted_labels == 0),
            "count_pop2": np.sum(sorted_labels == 1),
        }

    def _perform_statistical_tests(
        self, control_cells, treatment_cells, control_params, treatment_params
    ):
        ks_stat, ks_pval = ks_2samp(control_cells, treatment_cells)
        contingency = [
            [control_params["count_pop1"], control_params["count_pop2"]],
            [treatment_params["count_pop1"], treatment_params["count_pop2"]],
        ]
        chi2_stat, chi2_pval, _, _ = chi2_contingency(contingency)
        diffs = {
            "mean_diff_pop1": treatment_params["mean_pop1"]
            - control_params["mean_pop1"],
            "mean_diff_pop2": treatment_params["mean_pop2"]
            - control_params["mean_pop2"],
            "weight_diff_pop1": treatment_params["weight_pop1"]
            - control_params["weight_pop1"],
        }
        return {"ks_pvalue": ks_pval, "chi2_pvalue": chi2_pval, **diffs}

    def _visualize_population_comparison(self, result: dict):
        # This visualization function is the same as the one that produced your image
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        replicate, timepoint = result["replicate"], result["timepoint"]
        df_group = self.df[
            (self.df["replicate"] == replicate) & (self.df["timepoint"] == timepoint)
        ]

        sns.histplot(
            data=df_group[df_group["is_control"] == "True"],
            x="area",
            bins=50,
            stat="density",
            ax=axes[0],
            color="skyblue",
            label="Data",
        )
        self._plot_gmm_fit(axes[0], "control", result)
        axes[0].set_title(f"Control (n={result['control_n_cells']})")
        axes[0].legend()

        sns.histplot(
            data=df_group[df_group["is_control"] == "False"],
            x="area",
            bins=50,
            stat="density",
            ax=axes[1],
            color="salmon",
            label="Data",
        )
        self._plot_gmm_fit(axes[1], "treatment", result)
        axes[1].set_title(f"Treatment (n={result['treatment_n_cells']})")
        axes[1].legend()

        fig.suptitle(
            f"{self.inhibitor_name} vs Control | Replicate: {replicate}, Timepoint: {timepoint}",
            fontsize=16,
            fontweight="bold",
        )
        stats_text = f"KS Test (distribution): p={result['ks_pvalue']:.4f}\nChi-Square Test (proportions): p={result['chi2_pvalue']:.4f}"
        fig.text(
            0.5,
            0.01,
            stats_text,
            ha="center",
            fontsize=12,
            bbox=dict(facecolor="gold", alpha=0.2),
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(
            self.output_dir / f"comparison_{replicate}_{timepoint}.png", dpi=300
        )
        plt.close()

    def _plot_gmm_fit(self, ax, prefix, result):
        min_x, max_x = ax.get_xlim()
        x = np.linspace(min_x, max_x, 1000)
        for i in range(1, 3):
            log_mean, log_var, weight = (
                result[f"{prefix}_log_mean_pop{i}"],
                result[f"{prefix}_log_var_pop{i}"],
                result[f"{prefix}_weight_pop{i}"],
            )
            pdf = scipy.stats.lognorm.pdf(x, s=np.sqrt(log_var), scale=np.exp(log_mean))
            ax.plot(x, weight * pdf, linestyle="--", label=f"Pop {i} (w={weight:.2f})")

        total_pdf = sum(
            [
                result[f"{prefix}_weight_pop{i}"]
                * scipy.stats.lognorm.pdf(
                    x,
                    s=np.sqrt(result[f"{prefix}_log_var_pop{i}"]),
                    scale=np.exp(result[f"{prefix}_log_mean_pop{i}"]),
                )
                for i in range(1, 3)
            ]
        )
        ax.plot(x, total_pdf, "r-", label="Combined GMM")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid advanced analysis of inhibitor experiments."
    )
    parser.add_argument("--input", required=True, help="Input combined CSV file.")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for analysis."
    )
    parser.add_argument(
        "--inhibitor-name", required=True, help="Name of the inhibitor to analyze."
    )
    args = parser.parse_args()

    analyzer = HybridInhibitorAnalyzer(
        data_path=args.input,
        output_dir=args.output_dir,
        inhibitor_name=args.inhibitor_name,
    )

    # Run the full, integrated pipeline
    analyzer.load_data()
    analyzer.filter_outlier_cells()
    analyzer.fit_gmm_per_image()
    analyzer.save_gmm_params_to_csv()
    analyzer.compare_control_vs_treatment()


if __name__ == "__main__":
    main()
