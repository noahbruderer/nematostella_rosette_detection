#!/usr/bin/env python3
"""
Advanced Cell Analysis Script
============================

Performs advanced statistical analysis of cell populations using Gaussian Mixture Models.
Includes cross-sectional time series analysis, animal comparisons, and distribution fitting.

Author: Noah Bruderer
Date: 2025
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.stats import ks_2samp, lognorm, norm
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.2)


class CellImageAnalyzer:
    """
    A comprehensive class to analyze cell distributions between images from the same animal
    and across timepoints using Gaussian Mixture Models.
    """

    def __init__(self, data_path, output_dir="output"):
        """Initialize the analyzer with data path and output directory."""
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize data attributes
        self.df = None
        self.gmm_params = {}
        self.gmm_params_df = None  # To store the dataframe of parameters

        print(f"Analyzer initialized. Data will be loaded from {data_path}")
        print(f"Results will be saved to {self.output_dir}")

    def load_data(self):
        """Load and preprocess the cell data."""
        print("Loading data...")

        # Load the dataset
        self.df = pd.read_csv(self.data_path)

        # Make sure all relevant columns are strings for consistent handling
        for col in ["replicate", "timepoint", "animal", "image"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)

        # Create a unique identifier for each image
        self.df["unique_image_id"] = (
            self.df["replicate"]
            + "_"
            + self.df["timepoint"]
            + "_"
            + self.df["animal"]
            + "_"
            + self.df["image"]
        )

        # Basic info about the dataset
        print(f"Loaded {len(self.df)} cell measurements")
        print(f"Dataset contains {self.df['animal'].nunique()} animals")
        print(f"Dataset contains {self.df['timepoint'].nunique()} timepoints")
        print(f"Dataset contains {self.df['unique_image_id'].nunique()} unique images")

        return self.df

    def fit_gmm_per_image(self, n_components=2, min_cells=10, use_log_transform=True):
        """
        Fit a Gaussian Mixture Model to each image in the dataset,
        with optional log transformation for better fitting.

        Parameters:
        -----------
        n_components : int, default=2
            Number of components to use in the GMM
        min_cells : int, default=10
            Minimum number of cells required to fit a GMM
        use_log_transform : bool, default=True
            Whether to apply log transformation to the data before fitting
        """
        if self.df is None:
            self.load_data()

        print(
            f"Fitting {n_components}-component GMM to each image{' with log transformation' if use_log_transform else ''}..."
        )

        # Dictionary to store GMM parameters for each image
        self.gmm_params = {}

        # Process each unique image separately
        for unique_image_id, group in self.df.groupby("unique_image_id"):
            # Get the first row to extract metadata
            first_row = group.iloc[0]
            replicate_id = first_row["replicate"]
            timepoint_id = first_row["timepoint"]
            animal_id = first_row["animal"]
            image_id = first_row["image"]

            # Only fit if we have enough data points
            if len(group) < min_cells:
                print(
                    f"Skipping image {image_id} (animal {animal_id}, timepoint {timepoint_id}) due to insufficient data ({len(group)} cells)"
                )
                continue

            # Get cell areas
            areas = group["area"].values.reshape(-1, 1)

            # Apply log transformation if requested
            if use_log_transform:
                # Use log1p which is log(1+x) to handle zeros and small values
                fit_data = np.log1p(areas)
            else:
                fit_data = areas

            # Fit the GMM
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(fit_data)

            # Extract parameters
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            weights = gmm.weights_

            # Sort components by mean value for consistent labeling
            idx = np.argsort(means)
            means = means[idx]
            variances = variances[idx]
            weights = weights[idx]
            predictions = gmm.predict(fit_data)
            # Map predictions to the sorted indices to ensure 0 is always the smaller pop
            label_map = {
                original_idx: sorted_idx for sorted_idx, original_idx in enumerate(idx)
            }
            sorted_predictions = np.array([label_map[p] for p in predictions])

            # 2. Calculate rosette metrics for the entire image
            rosette_cells_df = group[group["is_in_rosette"]]
            total_rosette_cells = len(rosette_cells_df)
            total_rosettes = (
                rosette_cells_df["rosette_id"].nunique()
                if "rosette_id" in rosette_cells_df.columns
                else 0
            )

            # 3. Calculate population counts
            small_pop_count = np.sum(sorted_predictions == 0)
            large_pop_count = np.sum(sorted_predictions == 1)

            # 4. Calculate the requested ratios (handle division by zero)
            rosettes_per_small_pop = (
                total_rosettes / small_pop_count if small_pop_count > 0 else 0
            )
            rosettes_per_large_pop = (
                total_rosettes / large_pop_count if large_pop_count > 0 else 0
            )
            rosettes_per_all_cells = (
                total_rosettes / len(group) if len(group) > 0 else 0
            )

            # *** NEW METRICS ***
            rosettes_per_1000_small_cells = rosettes_per_small_pop * 1000
            rosettes_per_1000_large_cells = rosettes_per_large_pop * 1000

            # Transform parameters back to original scale if needed
            if use_log_transform:
                # Convert from log space to original space
                orig_means = np.expm1(means)  # exp(means) - 1
                # For log-normal distributions:
                # Var[X] ≈ [exp(σ²) - 1] × [exp(2μ + σ²)]
                orig_variances = np.array(
                    [
                        (np.exp(var) - 1) * np.exp(2 * means[i] + var)
                        for i, var in enumerate(variances)
                    ]
                )

                # Store both log-space and original-space parameters
                key = unique_image_id
                self.gmm_params[key] = {
                    "means": orig_means,  # Store transformed means
                    "variances": orig_variances,  # Store transformed variances
                    "weights": weights,
                    "log_means": means,  # Also store log-space parameters
                    "log_variances": variances,
                    "n_samples": len(group),
                    "replicate": replicate_id,
                    "timepoint": timepoint_id,
                    "animal": animal_id,
                    "image": image_id,
                    "cell_data": areas.flatten(),  # Store actual cell data for comparison
                    "log_transformed": use_log_transform,  # Record whether log transform was used
                    "total_rosettes": total_rosettes,
                    "total_rosette_cells": total_rosette_cells,
                    "small_pop_count": small_pop_count,
                    "large_pop_count": large_pop_count,
                    "rosettes_per_small_pop": rosettes_per_small_pop,
                    "rosettes_per_large_pop": rosettes_per_large_pop,
                    "rosettes_per_all_cells": rosettes_per_all_cells,
                    # *** ADDED NEW METRICS TO DICTIONARY ***
                    "rosettes_per_1000_small_cells": rosettes_per_1000_small_cells,
                    "rosettes_per_1000_large_cells": rosettes_per_1000_large_cells,
                }
            else:
                # If no transformation, original and fitting params are the same
                key = unique_image_id
                self.gmm_params[key] = {
                    "means": means,
                    "variances": variances,
                    "weights": weights,
                    "log_means": None,  # No log-space parameters
                    "log_variances": None,
                    "n_samples": len(group),
                    "replicate": replicate_id,
                    "timepoint": timepoint_id,
                    "animal": animal_id,
                    "image": image_id,
                    "cell_data": areas.flatten(),  # Store actual cell data for comparison
                    "log_transformed": use_log_transform,  # Record whether log transform was used
                    "total_rosettes": total_rosettes,
                    "total_rosette_cells": total_rosette_cells,
                    "small_pop_count": small_pop_count,
                    "large_pop_count": large_pop_count,
                    "rosettes_per_small_pop": rosettes_per_small_pop,
                    "rosettes_per_large_pop": rosettes_per_large_pop,
                    "rosettes_per_all_cells": rosettes_per_all_cells,
                    # *** ADDED NEW METRICS TO DICTIONARY ***
                    "rosettes_per_1000_small_cells": rosettes_per_1000_small_cells,
                    "rosettes_per_1000_large_cells": rosettes_per_1000_large_cells,
                }

        print(f"Fitted GMMs for {len(self.gmm_params)} unique images")
        return self.gmm_params

    def save_gmm_params_to_csv(self, output_file_path):
        """
        Save all GMM parameters to a CSV file.

        Parameters:
        -----------
        output_file_path : str
            Path where the CSV file should be saved
        """
        # Convert the dictionary to a list of dictionaries for DataFrame creation
        rows = []

        for unique_image_id, params in self.gmm_params.items():
            # Create a row with basic metadata
            row = {
                "unique_image_id": unique_image_id,
                "replicate": params["replicate"],
                "timepoint": params["timepoint"],
                "animal": params["animal"],
                "image": params["image"],
                "n_samples": params["n_samples"],
                "log_transformed": params["log_transformed"],
                "total_rosettes": params.get("total_rosettes", 0),
                "total_rosette_cells": params.get("total_rosette_cells", 0),
                "small_pop_count": params.get("small_pop_count", 0),
                "large_pop_count": params.get("large_pop_count", 0),
                "rosettes_per_small_pop": params.get("rosettes_per_small_pop", 0),
                "rosettes_per_large_pop": params.get("rosettes_per_large_pop", 0),
                "rosettes_per_all_cells": params.get("rosettes_per_all_cells", 0),
                # *** ADDED NEW METRICS TO CSV ROW ***
                "rosettes_per_1000_small_cells": params.get(
                    "rosettes_per_1000_small_cells", 0
                ),
                "rosettes_per_1000_large_cells": params.get(
                    "rosettes_per_1000_large_cells", 0
                ),
            }

            # Add GMM parameters for each component
            n_components = len(params["means"])
            for i in range(n_components):
                row[f"mean_{i}"] = params["means"][i]
                row[f"variance_{i}"] = params["variances"][i]
                row[f"weight_{i}"] = params["weights"][i]

                # Add log-space parameters if available
                if params["log_means"] is not None:
                    row[f"log_mean_{i}"] = params["log_means"][i]
                    row[f"log_variance_{i}"] = params["log_variances"][i]

            rows.append(row)

        # Create DataFrame and save to CSV
        self.gmm_params_df = pd.DataFrame(rows)
        self.gmm_params_df.to_csv(output_file_path, index=False)

        print(f"Saved GMM parameters for {len(rows)} images to {output_file_path}")
        return self.gmm_params_df

    def plot_rosette_metrics_over_time(self):
        """Plots the new rosette metrics as a function of time."""
        if self.gmm_params_df is None:
            print("GMM parameters not available. Run save_gmm_params_to_csv first.")
            return

        print("Generating plot of rosette metrics over time...")
        df = self.gmm_params_df.copy()

        # Sort timepoints numerically for correct plotting order
        # Sort timepoints numerically (handles 'T' or 'D')
        try:
            df["timepoint_num"] = df["timepoint"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("timepoint_num")
        except Exception as e:
            print(
                f"Warning: Could not sort timepoints numerically ({e}). Using alphabetical order."
            )

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot for small cells
        sns.lineplot(
            data=df,
            x="timepoint_num",
            y="rosettes_per_1000_small_cells",
            ax=axes[0],
            marker="o",
            errorbar="ci",
        )
        axes[0].set_title("Rosette Formation Rate in Small Cell Population Over Time")
        axes[0].set_ylabel("Rosettes per 1000 Small Cells")

        # Plot for large cells
        sns.lineplot(
            data=df,
            x="timepoint_num",
            y="rosettes_per_1000_large_cells",
            ax=axes[1],
            marker="o",
            color="coral",
            errorbar="ci",
        )
        axes[1].set_title("Rosette Formation Rate in Large Cell Population Over Time")
        axes[1].set_ylabel("Rosettes per 1000 Large Cells")
        axes[1].set_xlabel("Timepoint")

        # Use original timepoint labels for x-axis
        axes[1].set_xticks(df["timepoint_num"].unique())
        axes[1].set_xticklabels(df.sort_values("timepoint_num")["timepoint"].unique())

        plt.tight_layout()

        # Save figure
        summary_dir = self.output_dir / "timepoint_populations" / "summary"
        summary_dir.mkdir(exist_ok=True, parents=True)
        filename = "rosette_metrics_over_time.png"
        plt.savefig(summary_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved time series plot to {summary_dir / filename}")

    def compare_same_animal_images(self):
        """Compare distributions between images from the same animal."""
        if not self.gmm_params:
            self.fit_gmm_per_image()

        print("Comparing images from the same animal...")

        # Create output directory
        comp_dir = self.output_dir / "image_comparisons"
        comp_dir.mkdir(exist_ok=True, parents=True)

        # Dictionary of animals and their images, grouped by replicate, timepoint, and animal
        animal_images = {}
        for img_id, params in self.gmm_params.items():
            # Include replicate in the key to ensure independence
            animal_key = (params["replicate"], params["timepoint"], params["animal"])
            if animal_key not in animal_images:
                animal_images[animal_key] = []
            animal_images[animal_key].append(img_id)

        # List for results
        comparison_results = []

        # Only analyze animals with multiple images
        for animal_key, image_ids in animal_images.items():
            if any(len(ids) > 1 for ids in animal_images.values()):
                for animal_key, image_ids in animal_images.items():
                    if len(image_ids) <= 1:
                        continue

            # Unpack the key
            replicate, timepoint, animal = animal_key
            print(
                f"Analyzing replicate {replicate}, timepoint {timepoint}, animal {animal} ({len(image_ids)} images)"
            )

            # Compare all pairs of images for this animal
            for i in range(len(image_ids)):
                for j in range(i + 1, len(image_ids)):
                    img1_id = image_ids[i]
                    img2_id = image_ids[j]

                    # Get parameters and data
                    img1_params = self.gmm_params[img1_id]
                    img2_params = self.gmm_params[img2_id]
                    img1_data = img1_params["cell_data"]
                    img2_data = img2_params["cell_data"]

                    # 1. KS test for overall distribution
                    ks_stat, ks_pval = ks_2samp(img1_data, img2_data)

                    # 2. Compare GMM components
                    # Calculate absolute and percentage differences
                    mean1_diff = abs(img1_params["means"][0] - img2_params["means"][0])
                    mean2_diff = abs(img1_params["means"][1] - img2_params["means"][1])

                    mean1_pct = (
                        100
                        * mean1_diff
                        / ((img1_params["means"][0] + img2_params["means"][0]) / 2)
                    )
                    mean2_pct = (
                        100
                        * mean2_diff
                        / ((img1_params["means"][1] + img2_params["means"][1]) / 2)
                    )

                    # Store results
                    comparison_results.append(
                        {
                            "replicate": replicate,
                            "timepoint": timepoint,
                            "animal": animal,
                            "image1": img1_params["image"],
                            "image2": img2_params["image"],
                            "cells_img1": img1_params["n_samples"],
                            "cells_img2": img2_params["n_samples"],
                            "ks_statistic": ks_stat,
                            "ks_pvalue": ks_pval,
                            "distributions_significantly_different": ks_pval < 0.05,
                            "mean_diff_pop1": mean1_diff,
                            "mean_diff_pop2": mean2_diff,
                            "mean_pct_diff_pop1": mean1_pct,
                            "mean_pct_diff_pop2": mean2_pct,
                        }
                    )

                    # Generate visualizations for this comparison
                    self._visualize_image_comparison(img1_id, img2_id, comp_dir)

        # Convert to DataFrame
        results_df = pd.DataFrame(comparison_results)

        # Save results
        results_df.to_csv(comp_dir / "same_animal_image_comparison.csv", index=False)

        # Summarize results
        print("\nComparison Summary:")
        print(
            f"Analyzed {len(animal_images)} animal-replicate-timepoint combinations with multiple images"
        )
        print(f"Total comparisons: {len(comparison_results)}")

        if len(comparison_results) > 0:
            sig_diff = results_df["distributions_significantly_different"].mean() * 100
            print(
                f"Percentage of significantly different distributions: {sig_diff:.1f}%"
            )

            mean_pop1_diff = results_df["mean_pct_diff_pop1"].mean()
            mean_pop2_diff = results_df["mean_pct_diff_pop2"].mean()
            print(
                f"Average percent difference in population 1 mean: {mean_pop1_diff:.2f}%"
            )
            print(
                f"Average percent difference in population 2 mean: {mean_pop2_diff:.2f}%"
            )
        else:
            print(
                "No comparisons were made. Check if you have multiple images per animal."
            )

        return results_df

    def _visualize_image_comparison(self, img1_id, img2_id, output_dir):
        """Create visualization comparing two images from the same animal."""
        img1_params = self.gmm_params[img1_id]
        img2_params = self.gmm_params[img2_id]

        # Extract metadata
        replicate = img1_params["replicate"]
        timepoint = img1_params["timepoint"]
        animal = img1_params["animal"]
        img1_name = img1_params["image"]
        img2_name = img2_params["image"]

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot data and GMM for image 1
        axes[0].hist(
            img1_params["cell_data"],
            bins=30,
            density=True,
            alpha=0.6,
            label=f"Image {img1_name}",
        )
        x = np.linspace(
            min(img1_params["cell_data"]), max(img1_params["cell_data"]), 1000
        )

        # Plot components and combined GMM
        y_combined = np.zeros_like(x)

        # Check if log transform was used
        if img1_params["log_transformed"] and img1_params["log_means"] is not None:
            # Use log-normal distribution on original scale
            for i in range(len(img1_params["means"])):
                # Use log-space parameters for the PDF
                log_mean = img1_params["log_means"][i]
                log_var = img1_params["log_variances"][i]
                weight = img1_params["weights"][i]

                # Use log-normal PDF instead of normal PDF
                # scipy.stats.lognorm parameters are different from normal
                # shape=sigma, scale=exp(mu)
                sigma = np.sqrt(log_var)  # standard deviation in log space
                scale = np.exp(log_mean)  # exp(mu)

                y = weight * scipy.stats.lognorm.pdf(x, s=sigma, scale=scale)
                axes[0].plot(
                    x,
                    y,
                    "--",
                    label=f"Component {i + 1} (μ={img1_params['means'][i]:.1f})",
                )
                y_combined += y
        else:
            # Use normal distribution for untransformed data
            for i in range(len(img1_params["means"])):
                mean = img1_params["means"][i]
                var = img1_params["variances"][i]
                weight = img1_params["weights"][i]
                y = weight * norm.pdf(x, mean, np.sqrt(var))
                axes[0].plot(x, y, "--", label=f"Component {i + 1} (μ={mean:.1f})")
                y_combined += y

        axes[0].plot(x, y_combined, "r-", label="Combined GMM")
        axes[0].set_title(f"Image {img1_name}")
        axes[0].legend()

        # Plot data and GMM for image 2
        axes[1].hist(
            img2_params["cell_data"],
            bins=30,
            density=True,
            alpha=0.6,
            label=f"Image {img2_name}",
        )
        x = np.linspace(
            min(img2_params["cell_data"]), max(img2_params["cell_data"]), 1000
        )

        # Plot components and combined GMM
        y_combined = np.zeros_like(x)

        # Check if log transform was used
        if img2_params["log_transformed"] and img2_params["log_means"] is not None:
            # Use log-normal distribution on original scale
            for i in range(len(img2_params["means"])):
                # Use log-space parameters for the PDF
                log_mean = img2_params["log_means"][i]
                log_var = img2_params["log_variances"][i]
                weight = img2_params["weights"][i]

                # Use log-normal PDF instead of normal PDF
                sigma = np.sqrt(log_var)  # standard deviation in log space
                scale = np.exp(log_mean)  # exp(mu)

                y = weight * scipy.stats.lognorm.pdf(x, s=sigma, scale=scale)
                axes[1].plot(
                    x,
                    y,
                    "--",
                    label=f"Component {i + 1} (μ={img2_params['means'][i]:.1f})",
                )
                y_combined += y
        else:
            # Use normal distribution for untransformed data
            for i in range(len(img2_params["means"])):
                mean = img2_params["means"][i]
                var = img2_params["variances"][i]
                weight = img2_params["weights"][i]
                y = weight * norm.pdf(x, mean, np.sqrt(var))
                axes[1].plot(x, y, "--", label=f"Component {i + 1} (μ={mean:.1f})")
                y_combined += y

        axes[1].plot(x, y_combined, "r-", label="Combined GMM")
        axes[1].set_title(f"Image {img2_name}")
        axes[1].legend()

        plt.suptitle(
            f"Replicate {replicate}, Timepoint {timepoint}, Animal {animal}: Image Comparison",
            fontsize=16,
        )
        plt.tight_layout()

        # Save figure
        filename = f"rep{replicate}_tp{timepoint}_animal{animal}_img{img1_name}_vs_img{img2_name}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def compare_animals_same_timepoint(self):
        """
        Compare cell distributions between different animals within the same replicate and timepoint.
        This accounts for the hierarchical nature of the data by first aggregating image data per animal.
        """
        if not self.gmm_params:
            self.fit_gmm_per_image()

        print("Comparing animals within the same replicate and timepoint...")

        # Create output directory
        comp_dir = self.output_dir / "animal_comparisons"
        comp_dir.mkdir(exist_ok=True, parents=True)

        # Group images by replicate and timepoint
        replicate_timepoint_groups = {}
        for img_id, params in self.gmm_params.items():
            key = (params["replicate"], params["timepoint"])
            if key not in replicate_timepoint_groups:
                replicate_timepoint_groups[key] = {}

            # Group by animal
            animal = params["animal"]
            if animal not in replicate_timepoint_groups[key]:
                replicate_timepoint_groups[key][animal] = []

            replicate_timepoint_groups[key][animal].append(img_id)

        # List for results
        comparison_results = []

        # Process each replicate-timepoint group
        for (replicate, timepoint), animals in replicate_timepoint_groups.items():
            # Only proceed if we have multiple animals
            if len(animals) <= 1:
                print(
                    f"Skipping replicate {replicate}, timepoint {timepoint}: only one animal"
                )
                continue

            print(
                f"Analyzing replicate {replicate}, timepoint {timepoint} with {len(animals)} animals"
            )

            # For each animal, aggregate cell data from all images
            animal_aggregated_data = {}
            for animal, img_ids in animals.items():
                # Combine cell data from all images for this animal
                combined_cells = np.concatenate(
                    [self.gmm_params[img_id]["cell_data"] for img_id in img_ids]
                )
                animal_aggregated_data[animal] = {
                    "cell_data": combined_cells,
                    "n_samples": len(combined_cells),
                    "n_images": len(img_ids),
                    "image_ids": img_ids,
                }

                print(
                    f"  Animal {animal}: {len(combined_cells)} cells from {len(img_ids)} images"
                )

            # Compare all pairs of animals
            animal_list = list(animal_aggregated_data.keys())
            for i in range(len(animal_list)):
                for j in range(i + 1, len(animal_list)):
                    animal1 = animal_list[i]
                    animal2 = animal_list[j]

                    # Get aggregated data
                    animal1_data = animal_aggregated_data[animal1]
                    animal2_data = animal_aggregated_data[animal2]

                    # 1. KS test for overall distribution
                    ks_stat, ks_pval = ks_2samp(
                        animal1_data["cell_data"], animal2_data["cell_data"]
                    )

                    # 2. Fit GMM to each animal's aggregated data
                    # This more accurately captures the animal's cell population distribution
                    gmm1 = GaussianMixture(n_components=2, random_state=42)
                    gmm2 = GaussianMixture(n_components=2, random_state=42)

                    areas1 = animal1_data["cell_data"].reshape(-1, 1)
                    areas2 = animal2_data["cell_data"].reshape(-1, 1)

                    # Apply log transformation
                    fit_data1 = np.log1p(areas1)
                    fit_data2 = np.log1p(areas2)

                    gmm1.fit(fit_data1)
                    gmm2.fit(fit_data2)

                    # Sort components by mean for consistent comparison
                    idx1 = np.argsort(gmm1.means_.flatten())
                    idx2 = np.argsort(gmm2.means_.flatten())

                    log_means1 = gmm1.means_.flatten()[idx1]
                    log_means2 = gmm2.means_.flatten()[idx2]

                    weights1 = gmm1.weights_[idx1]
                    weights2 = gmm2.weights_[idx2]

                    log_variances1 = gmm1.covariances_.flatten()[idx1]
                    log_variances2 = gmm2.covariances_.flatten()[idx2]

                    # Transform back to original space
                    means1 = np.expm1(log_means1)
                    means2 = np.expm1(log_means2)

                    variances1 = np.array(
                        [
                            (np.exp(var) - 1) * np.exp(2 * log_means1[i] + var)
                            for i, var in enumerate(log_variances1)
                        ]
                    )
                    variances2 = np.array(
                        [
                            (np.exp(var) - 1) * np.exp(2 * log_means2[i] + var)
                            for i, var in enumerate(log_variances2)
                        ]
                    )

                    # Calculate differences
                    mean1_diff = abs(means1[0] - means2[0])
                    mean2_diff = abs(means1[1] - means2[1])

                    mean1_pct = 100 * mean1_diff / ((means1[0] + means2[0]) / 2)
                    mean2_pct = 100 * mean2_diff / ((means1[1] + means2[1]) / 2)

                    # Weight differences
                    weight1_diff = abs(weights1[0] - weights2[0])
                    weight2_diff = abs(weights1[1] - weights2[1])

                    # Variance differences
                    var1_diff = abs(variances1[0] - variances2[0])
                    var2_diff = abs(variances1[1] - variances2[1])

                    var1_pct = 100 * var1_diff / ((variances1[0] + variances2[0]) / 2)
                    var2_pct = 100 * var2_diff / ((variances1[1] + variances2[1]) / 2)

                    # Store results
                    comparison_results.append(
                        {
                            "replicate": replicate,
                            "timepoint": timepoint,
                            "animal1": animal1,
                            "animal2": animal2,
                            "cells_animal1": animal1_data["n_samples"],
                            "cells_animal2": animal2_data["n_samples"],
                            "images_animal1": animal1_data["n_images"],
                            "images_animal2": animal2_data["n_images"],
                            "ks_statistic": ks_stat,
                            "ks_pvalue": ks_pval,
                            "distributions_significantly_different": ks_pval < 0.05,
                            "mean_diff_pop1": mean1_diff,
                            "mean_diff_pop2": mean2_diff,
                            "mean_pct_diff_pop1": mean1_pct,
                            "mean_pct_diff_pop2": mean2_pct,
                            "weight_diff_pop1": weight1_diff,
                            "weight_diff_pop2": weight2_diff,
                            "var_diff_pop1": var1_diff,
                            "var_diff_pop2": var2_diff,
                            "var_pct_diff_pop1": var1_pct,
                            "var_pct_diff_pop2": var2_pct,
                        }
                    )

                    # Visualize the comparison
                    self._visualize_animal_comparison(
                        replicate,
                        timepoint,
                        animal1,
                        animal2,
                        animal1_data,
                        animal2_data,
                        means1,
                        means2,
                        variances1,
                        variances2,
                        weights1,
                        weights2,
                        comp_dir,
                        log_means1,
                        log_means2,
                        log_variances1,
                        log_variances2,
                    )

        # Convert to DataFrame
        results_df = pd.DataFrame(comparison_results)

        # Save results
        results_df.to_csv(
            comp_dir / "animal_comparison_same_timepoint.csv", index=False
        )

        # Summarize results
        print("\nAnimal Comparison Summary:")
        print(
            f"Analyzed {len(replicate_timepoint_groups)} replicate-timepoint combinations"
        )
        print(f"Total animal pair comparisons: {len(comparison_results)}")

        if len(comparison_results) > 0:
            sig_diff = results_df["distributions_significantly_different"].mean() * 100
            print(
                f"Percentage of significantly different distributions: {sig_diff:.1f}%"
            )

            mean_pop1_diff = results_df["mean_pct_diff_pop1"].mean()
            mean_pop2_diff = results_df["mean_pct_diff_pop2"].mean()
            print(
                f"Average percent difference in population 1 mean: {mean_pop1_diff:.2f}%"
            )
            print(
                f"Average percent difference in population 2 mean: {mean_pop2_diff:.2f}%"
            )

            var_pop1_diff = results_df["var_pct_diff_pop1"].mean()
            var_pop2_diff = results_df["var_pct_diff_pop2"].mean()
            print(
                f"Average percent difference in population 1 variance: {var_pop1_diff:.2f}%"
            )
            print(
                f"Average percent difference in population 2 variance: {var_pop2_diff:.2f}%"
            )
        else:
            print(
                "No comparisons were made. Check if you have multiple animals per replicate-timepoint."
            )

        return results_df

    def _visualize_animal_comparison(
        self,
        replicate,
        timepoint,
        animal1,
        animal2,
        animal1_data,
        animal2_data,
        means1,
        means2,
        variances1,
        variances2,
        weights1,
        weights2,
        output_dir,
        log_means1,
        log_means2,
        log_variances1,
        log_variances2,
    ):
        """
        Create visualization comparing two animals from the same replicate and timepoint,
        properly handling log-transformed data.
        """
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot data and GMM for animal 1
        axes[0].hist(
            animal1_data["cell_data"],
            bins=30,
            density=True,
            alpha=0.6,
            label=f"Animal {animal1} (n={animal1_data['n_samples']} cells, {animal1_data['n_images']} images)",
        )

        x = np.linspace(
            min(animal1_data["cell_data"]), max(animal1_data["cell_data"]), 1000
        )

        # Plot components and combined GMM using log-normal distribution
        y_combined = np.zeros_like(x)
        for i in range(len(means1)):
            # Use log-normal distribution for the transformed data
            log_mean = log_means1[i]
            log_var = log_variances1[i]
            weight = weights1[i]

            # For log-normal distribution in scipy, we need:
            # Shape (s) = standard deviation in log space
            # Scale = exp(mean) in log space
            sigma = np.sqrt(log_var)  # standard deviation in log space
            scale = np.exp(log_mean)  # exp(mu)

            # x values must be > 0 for lognorm
            x_valid = x.copy()
            x_valid[x_valid <= 0] = 1e-10  # Replace any non-positive values

            y = weight * scipy.stats.lognorm.pdf(x_valid, s=sigma, scale=scale)

            # Label with the original scale mean and variance
            axes[0].plot(
                x,
                y,
                "--",
                label=f"Component {i + 1} (μ={means1[i]:.1f}, σ²={variances1[i]:.1f}, w={weight:.2f})",
            )
            y_combined += y

        axes[0].plot(x, y_combined, "r-", label="Combined GMM")
        axes[0].set_title(f"Animal {animal1}")
        axes[0].legend()

        # Plot data and GMM for animal 2
        axes[1].hist(
            animal2_data["cell_data"],
            bins=30,
            density=True,
            alpha=0.6,
            label=f"Animal {animal2} (n={animal2_data['n_samples']} cells, {animal2_data['n_images']} images)",
        )

        x = np.linspace(
            min(animal2_data["cell_data"]), max(animal2_data["cell_data"]), 1000
        )

        # Plot components and combined GMM using log-normal distribution
        y_combined = np.zeros_like(x)
        for i in range(len(means2)):
            # Use log-normal distribution for the transformed data
            log_mean = log_means2[i]
            log_var = log_variances2[i]
            weight = weights2[i]

            # For log-normal distribution in scipy, we need:
            # Shape (s) = standard deviation in log space
            # Scale = exp(mean) in log space
            sigma = np.sqrt(log_var)  # standard deviation in log space
            scale = np.exp(log_mean)  # exp(mu)

            # x values must be > 0 for lognorm
            x_valid = x.copy()
            x_valid[x_valid <= 0] = 1e-10  # Replace any non-positive values

            y = weight * scipy.stats.lognorm.pdf(x_valid, s=sigma, scale=scale)

            # Label with the original scale mean and variance
            axes[1].plot(
                x,
                y,
                "--",
                label=f"Component {i + 1} (μ={means2[i]:.1f}, σ²={variances2[i]:.1f}, w={weight:.2f})",
            )
            y_combined += y

        axes[1].plot(x, y_combined, "r-", label="Combined GMM")
        axes[1].set_title(f"Animal {animal2}")
        axes[1].legend()

        # Add KS test result
        ks_stat, ks_pval = ks_2samp(
            animal1_data["cell_data"], animal2_data["cell_data"]
        )
        significance = (
            "significantly different"
            if ks_pval < 0.05
            else "not significantly different"
        )
        plt.figtext(
            0.5,
            0.01,
            f"KS test: p={ks_pval:.4f} ({significance})",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5},
        )

        plt.suptitle(
            f"Replicate {replicate}, Timepoint {timepoint}: Animal Comparison (Log-Transformed)",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save figure
        filename = (
            f"rep{replicate}_tp{timepoint}_animal{animal1}_vs_animal{animal2}.png"
        )
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def compare_timepoint_populations(self):
        """
        Compare cell populations between different timepoints at the population level.
        This method assumes different animals at each timepoint (cross-sectional study design).
        """
        if self.df is None:
            self.load_data()

        if not self.gmm_params:
            self.fit_gmm_per_image()

        print(
            "Comparing cell populations across timepoints (cross-sectional analysis)..."
        )

        # Create output directory
        comp_dir = self.output_dir / "timepoint_populations"
        comp_dir.mkdir(exist_ok=True, parents=True)

        # Step 1: Aggregate data by replicate and timepoint
        replicate_timepoint_data = {}

        # Group by replicate and timepoint, aggregating all animals
        for img_id, params in self.gmm_params.items():
            replicate = params["replicate"]
            timepoint = params["timepoint"]

            key = (replicate, timepoint)
            if key not in replicate_timepoint_data:
                replicate_timepoint_data[key] = {
                    "cell_data": [],
                    "n_cells": 0,
                    "n_images": 0,
                    "n_animals": set(),
                    "image_ids": [],
                }

            # Append cell data
            replicate_timepoint_data[key]["cell_data"].extend(
                params["cell_data"].tolist()
            )
            replicate_timepoint_data[key]["n_cells"] += len(params["cell_data"])
            replicate_timepoint_data[key]["n_images"] += 1
            replicate_timepoint_data[key]["n_animals"].add(params["animal"])
            replicate_timepoint_data[key]["image_ids"].append(img_id)

        # Convert to arrays for analysis
        for key in replicate_timepoint_data:
            replicate_timepoint_data[key]["cell_data"] = np.array(
                replicate_timepoint_data[key]["cell_data"]
            )
            replicate_timepoint_data[key]["n_animals"] = len(
                replicate_timepoint_data[key]["n_animals"]
            )

        # Step 2: For each replicate, compare across timepoints
        timepoint_comparison_results = []

        # Process each replicate
        for replicate in sorted(
            set([key[0] for key in replicate_timepoint_data.keys()])
        ):
            # Get all timepoints for this replicate
            replicate_timepoints = [
                key[1] for key in replicate_timepoint_data.keys() if key[0] == replicate
            ]

            # Only proceed if we have multiple timepoints
            if len(replicate_timepoints) <= 1:
                print(f"Skipping replicate {replicate}: only one timepoint available")
                continue

            # Sort timepoints (they might be strings, so convert to numeric if needed)
            try:
                sorted_timepoints = sorted(
                    replicate_timepoints, key=lambda x: float(x.replace("T", ""))
                )
            except:
                sorted_timepoints = sorted(replicate_timepoints)

            print(
                f"Analyzing replicate {replicate} across {len(sorted_timepoints)} timepoints"
            )

            # For each pair of consecutive timepoints
            for i in range(len(sorted_timepoints) - 1):
                tp1 = sorted_timepoints[i]
                tp2 = sorted_timepoints[i + 1]

                # Get aggregated data
                tp1_data = replicate_timepoint_data[(replicate, tp1)]
                tp2_data = replicate_timepoint_data[(replicate, tp2)]

                # Convert to numpy arrays for analysis
                tp1_cells = tp1_data["cell_data"]
                tp2_cells = tp2_data["cell_data"]

                # 1. Fit GMM to each timepoint's aggregated data
                gmm1 = GaussianMixture(n_components=2, random_state=42)
                gmm2 = GaussianMixture(n_components=2, random_state=42)

                # Apply log transformation
                fit_data1 = np.log1p(tp1_cells.reshape(-1, 1))
                fit_data2 = np.log1p(tp2_cells.reshape(-1, 1))

                gmm1.fit(fit_data1)
                gmm2.fit(fit_data2)

                # Sort components by mean
                idx1 = np.argsort(gmm1.means_.flatten())
                idx2 = np.argsort(gmm2.means_.flatten())

                log_means1 = gmm1.means_.flatten()[idx1]
                log_means2 = gmm2.means_.flatten()[idx2]

                weights1 = gmm1.weights_[idx1]
                weights2 = gmm2.weights_[idx2]

                log_variances1 = gmm1.covariances_.flatten()[idx1]
                log_variances2 = gmm2.covariances_.flatten()[idx2]

                # Transform back to original space
                means1 = np.expm1(log_means1)
                means2 = np.expm1(log_means2)

                variances1 = np.array(
                    [
                        (np.exp(var) - 1) * np.exp(2 * log_means1[i] + var)
                        for i, var in enumerate(log_variances1)
                    ]
                )
                variances2 = np.array(
                    [
                        (np.exp(var) - 1) * np.exp(2 * log_means2[i] + var)
                        for i, var in enumerate(log_variances2)
                    ]
                )

                # 2. Predict which component each cell belongs to
                tp1_component_probs = gmm1.predict_proba(fit_data1)
                tp1_components = np.argmax(tp1_component_probs, axis=1)

                tp2_component_probs = gmm2.predict_proba(fit_data2)
                tp2_components = np.argmax(tp2_component_probs, axis=1)

                # Count cells in each population
                tp1_pop1_count = np.sum(tp1_components == 0)
                tp1_pop2_count = np.sum(tp1_components == 1)
                tp2_pop1_count = np.sum(tp2_components == 0)
                tp2_pop2_count = np.sum(tp2_components == 1)

                # 3. Calculate proportion of cells in each population
                tp1_prop_pop1 = tp1_pop1_count / len(tp1_cells)
                tp2_prop_pop1 = tp2_pop1_count / len(tp2_cells)

                # 4. Calculate relative changes
                abs_change_pop1 = tp2_pop1_count - tp1_pop1_count
                abs_change_pop2 = tp2_pop2_count - tp1_pop2_count

                pct_change_pop1 = (
                    ((tp2_pop1_count / tp1_pop1_count) - 1) * 100
                    if tp1_pop1_count > 0
                    else np.nan
                )
                pct_change_pop2 = (
                    ((tp2_pop2_count / tp1_pop2_count) - 1) * 100
                    if tp1_pop2_count > 0
                    else np.nan
                )

                prop_change = (tp2_prop_pop1 - tp1_prop_pop1) * 100

                # 5. Statistical tests
                contingency = np.array(
                    [[tp1_pop1_count, tp1_pop2_count], [tp2_pop1_count, tp2_pop2_count]]
                )

                chi2, chi2_pval, _, _ = scipy.stats.chi2_contingency(contingency)
                ks_stat, ks_pval = scipy.stats.ks_2samp(tp1_cells, tp2_cells)

                # Calculate mean and variance changes
                mean_diff_pop1 = means2[0] - means1[0]
                mean_diff_pop2 = means2[1] - means1[1]

                mean_pct_diff_pop1 = (means2[0] / means1[0] - 1) * 100
                mean_pct_diff_pop2 = (means2[1] / means1[1] - 1) * 100

                var_diff_pop1 = variances2[0] - variances1[0]
                var_diff_pop2 = variances2[1] - variances1[1]

                var_pct_diff_pop1 = (variances2[0] / variances1[0] - 1) * 100
                var_pct_diff_pop2 = (variances2[1] / variances1[1] - 1) * 100

                # 6. Store results
                timepoint_comparison_results.append(
                    {
                        "replicate": replicate,
                        "timepoint1": tp1,
                        "timepoint2": tp2,
                        "n_cells_tp1": len(tp1_cells),
                        "n_cells_tp2": len(tp2_cells),
                        "n_animals_tp1": tp1_data["n_animals"],
                        "n_animals_tp2": tp2_data["n_animals"],
                        "n_images_tp1": tp1_data["n_images"],
                        "n_images_tp2": tp2_data["n_images"],
                        "small_cell_count_tp1": tp1_pop1_count,
                        "small_cell_count_tp2": tp2_pop1_count,
                        "large_cell_count_tp1": tp1_pop2_count,
                        "large_cell_count_tp2": tp2_pop2_count,
                        "small_cell_prop_tp1": tp1_prop_pop1 * 100,
                        "small_cell_prop_tp2": tp2_prop_pop1 * 100,
                        "small_cell_mean_tp1": means1[0],
                        "small_cell_mean_tp2": means2[0],
                        "large_cell_mean_tp1": means1[1],
                        "large_cell_mean_tp2": means2[1],
                        "small_cell_var_tp1": variances1[0],
                        "small_cell_var_tp2": variances2[0],
                        "large_cell_var_tp1": variances1[1],
                        "large_cell_var_tp2": variances2[1],
                        "small_cell_weight_tp1": weights1[0],
                        "small_cell_weight_tp2": weights2[0],
                        "absolute_change_small": abs_change_pop1,
                        "absolute_change_large": abs_change_pop2,
                        "percent_change_small": pct_change_pop1,
                        "percent_change_large": pct_change_pop2,
                        "proportion_change_small": prop_change,
                        "mean_diff_pop1": mean_diff_pop1,
                        "mean_diff_pop2": mean_diff_pop2,
                        "mean_pct_diff_pop1": mean_pct_diff_pop1,
                        "mean_pct_diff_pop2": mean_pct_diff_pop2,
                        "var_diff_pop1": var_diff_pop1,
                        "var_diff_pop2": var_diff_pop2,
                        "var_pct_diff_pop1": var_pct_diff_pop1,
                        "var_pct_diff_pop2": var_pct_diff_pop2,
                        "chi2_statistic": chi2,
                        "chi2_pvalue": chi2_pval,
                        "populations_prop_significantly_different": chi2_pval < 0.05,
                        "ks_statistic": ks_stat,
                        "ks_pvalue": ks_pval,
                        "distributions_significantly_different": ks_pval < 0.05,
                    }
                )

                # 7. Visualize the comparison
                self._visualize_population_timepoint_comparison(
                    replicate,
                    tp1,
                    tp2,
                    tp1_cells,
                    tp2_cells,
                    means1,
                    means2,
                    variances1,
                    variances2,
                    weights1,
                    weights2,
                    tp1_pop1_count,
                    tp1_pop2_count,
                    tp2_pop1_count,
                    tp2_pop2_count,
                    tp1_data["n_animals"],
                    tp2_data["n_animals"],
                    chi2_pval,
                    ks_pval,
                    comp_dir,
                )

        # Convert to DataFrame and save
        results_df = pd.DataFrame(timepoint_comparison_results)
        results_df.to_csv(
            comp_dir / "timepoint_population_comparison_results.csv", index=False
        )

        # Create summary visualizations
        if len(timepoint_comparison_results) > 0:
            self._create_population_timepoint_summary(results_df, comp_dir)

        # Summarize results
        print("\nTimepoint Population Comparison Summary:")
        print(
            f"Total timepoint-to-timepoint comparisons: {len(timepoint_comparison_results)}"
        )

        if len(timepoint_comparison_results) > 0:
            sig_diff_chi2 = (
                results_df["populations_prop_significantly_different"].mean() * 100
            )
            print(
                f"Percentage of significantly different population proportions (Chi-square): {sig_diff_chi2:.1f}%"
            )

            sig_diff_ks = (
                results_df["distributions_significantly_different"].mean() * 100
            )
            print(
                f"Percentage of significantly different distributions (KS test): {sig_diff_ks:.1f}%"
            )

        return results_df

    def _visualize_population_timepoint_comparison(
        self,
        replicate,
        tp1,
        tp2,
        tp1_cells,
        tp2_cells,
        means1,
        means2,
        variances1,
        variances2,
        weights1,
        weights2,
        tp1_pop1_count,
        tp1_pop2_count,
        tp2_pop1_count,
        tp2_pop2_count,
        n_animals_tp1,
        n_animals_tp2,
        chi2_pval,
        ks_pval,
        comp_dir,
    ):
        """Create visualization comparing cell population distributions between two timepoints."""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

        # Plot for first timepoint
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(
            tp1_cells,
            bins=30,
            density=True,
            alpha=0.6,
            label=f"Timepoint {tp1} (n={len(tp1_cells)} cells, {n_animals_tp1} animals)",
        )

        x = np.linspace(
            min(min(tp1_cells), min(tp2_cells)),
            max(max(tp1_cells), max(tp2_cells)),
            1000,
        )

        # Plot GMM components using log-normal distribution
        y_combined = np.zeros_like(x)
        log_means1 = np.log1p(means1)
        log_variances1 = np.array(
            [np.log(1 + variances1[i] / (means1[i] ** 2)) for i in range(len(means1))]
        )

        for i in range(len(means1)):
            weight = weights1[i]
            log_mean = log_means1[i]
            log_var = log_variances1[i]
            sigma = np.sqrt(log_var)
            scale = np.exp(log_mean)

            x_valid = x.copy()
            x_valid[x_valid <= 0] = 1e-10

            y = weight * scipy.stats.lognorm.pdf(x_valid, s=sigma, scale=scale)
            pop_label = "Small Cell Pop" if i == 0 else "Large Cell Pop"
            count = tp1_pop1_count if i == 0 else tp1_pop2_count
            prop = (
                (tp1_pop1_count / (tp1_pop1_count + tp1_pop2_count) * 100)
                if i == 0
                else (tp1_pop2_count / (tp1_pop1_count + tp1_pop2_count) * 100)
            )
            ax1.plot(
                x,
                y,
                "--",
                label=f"{pop_label} (μ={means1[i]:.1f}, n={count}, {prop:.1f}%)",
            )
            y_combined += y

        ax1.plot(x, y_combined, "r-", label="Combined GMM")
        ax1.set_title(f"Timepoint {tp1}")
        ax1.legend()

        # Plot for second timepoint
        ax2 = fig.add_subplot(gs[1, :])
        ax2.hist(
            tp2_cells,
            bins=30,
            density=True,
            alpha=0.6,
            label=f"Timepoint {tp2} (n={len(tp2_cells)} cells, {n_animals_tp2} animals)",
        )

        # Plot GMM components
        y_combined = np.zeros_like(x)
        log_means2 = np.log1p(means2)
        log_variances2 = np.array(
            [np.log(1 + variances2[i] / (means2[i] ** 2)) for i in range(len(means2))]
        )

        for i in range(len(means2)):
            weight = weights2[i]
            log_mean = log_means2[i]
            log_var = log_variances2[i]
            sigma = np.sqrt(log_var)
            scale = np.exp(log_mean)

            x_valid = x.copy()
            x_valid[x_valid <= 0] = 1e-10

            y = weight * scipy.stats.lognorm.pdf(x_valid, s=sigma, scale=scale)
            pop_label = "Small Cell Pop" if i == 0 else "Large Cell Pop"
            count = tp2_pop1_count if i == 0 else tp2_pop2_count
            prop = (
                (tp2_pop1_count / (tp2_pop1_count + tp2_pop2_count) * 100)
                if i == 0
                else (tp2_pop2_count / (tp2_pop1_count + tp2_pop2_count) * 100)
            )
            ax2.plot(
                x,
                y,
                "--",
                label=f"{pop_label} (μ={means2[i]:.1f}, n={count}, {prop:.1f}%)",
            )
            y_combined += y

        ax2.plot(x, y_combined, "r-", label="Combined GMM")
        ax2.set_title(f"Timepoint {tp2}")
        ax2.legend()

        # Plot proportions comparison
        ax3 = fig.add_subplot(gs[2, 0])
        total_tp1 = tp1_pop1_count + tp1_pop2_count
        total_tp2 = tp2_pop1_count + tp2_pop2_count

        tp1_small_prop = (tp1_pop1_count / total_tp1) * 100
        tp1_large_prop = (tp1_pop2_count / total_tp1) * 100
        tp2_small_prop = (tp2_pop1_count / total_tp2) * 100
        tp2_large_prop = (tp2_pop2_count / total_tp2) * 100

        props = np.array(
            [[tp1_small_prop, tp1_large_prop], [tp2_small_prop, tp2_large_prop]]
        )

        bar_width = 0.35
        x = np.array([0, 1])

        ax3.bar(x - bar_width / 2, props[0], bar_width, label=f"Timepoint {tp1}")
        ax3.bar(x + bar_width / 2, props[1], bar_width, label=f"Timepoint {tp2}")

        ax3.set_xticks(x)
        ax3.set_xticklabels(["Small Cells", "Large Cells"])
        ax3.set_ylabel("Percentage of Cells (%)")
        ax3.set_title("Cell Population Proportions")
        ax3.legend()

        # Plot proportion change
        ax4 = fig.add_subplot(gs[2, 1])
        small_prop_change = tp2_small_prop - tp1_small_prop
        large_prop_change = tp2_large_prop - tp1_large_prop

        ax4.bar(
            [0, 1],
            [small_prop_change, large_prop_change],
            color=["lightblue", "lightgreen"],
        )

        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(["Small Cells", "Large Cells"])
        ax4.set_ylabel("Change in Proportion (percentage points)")
        ax4.set_title("Change in Cell Population Proportions")
        ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add statistical test results
        chi2_result = (
            "significantly different"
            if chi2_pval < 0.05
            else "not significantly different"
        )
        ks_result = (
            "significantly different"
            if ks_pval < 0.05
            else "not significantly different"
        )

        plt.figtext(
            0.5,
            0.01,
            f"Chi-square test: p={chi2_pval:.4f} ({chi2_result})\n"
            f"KS test: p={ks_pval:.4f} ({ks_result})",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5},
        )

        plt.suptitle(
            f"Replicate {replicate}: Cell Population Changes from {tp1} to {tp2}",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save figure
        filename = f"rep{replicate}_tp{tp1}_to_tp{tp2}.png"
        plt.savefig(comp_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _create_population_timepoint_summary(self, results_df, output_dir):
        """Create summary visualizations for timepoint population comparisons."""
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(exist_ok=True, parents=True)

        # Create transition summary
        results_df["transition"] = (
            results_df["timepoint1"] + "->" + results_df["timepoint2"]
        )

        transition_summary = (
            results_df.groupby("transition")
            .agg(
                {
                    "percent_change_small": "mean",
                    "percent_change_large": "mean",
                    "proportion_change_small": "mean",
                    "mean_pct_diff_pop1": "mean",
                    "mean_pct_diff_pop2": "mean",
                    "populations_prop_significantly_different": "mean",
                    "distributions_significantly_different": "mean",
                    "replicate": "nunique",
                }
            )
            .reset_index()
        )

        # Convert boolean columns to percentages
        transition_summary["populations_prop_significantly_different"] *= 100
        transition_summary["distributions_significantly_different"] *= 100

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Population count change
        transitions = transition_summary["transition"]
        small_changes = transition_summary["percent_change_small"]
        large_changes = transition_summary["percent_change_large"]

        x = np.arange(len(transitions))
        width = 0.35

        axes[0, 0].bar(
            x - width / 2, small_changes, width, label="Small Cells", color="skyblue"
        )
        axes[0, 0].bar(
            x + width / 2, large_changes, width, label="Large Cells", color="lightgreen"
        )
        axes[0, 0].set_xlabel("Timepoint Transition")
        axes[0, 0].set_ylabel("Average Percent Change (%)")
        axes[0, 0].set_title("Cell Population Count Change")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(transitions)
        axes[0, 0].legend()
        axes[0, 0].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Cell size change
        mean_changes_small = transition_summary["mean_pct_diff_pop1"]
        mean_changes_large = transition_summary["mean_pct_diff_pop2"]

        axes[0, 1].bar(
            x - width / 2,
            mean_changes_small,
            width,
            label="Small Cells",
            color="skyblue",
        )
        axes[0, 1].bar(
            x + width / 2,
            mean_changes_large,
            width,
            label="Large Cells",
            color="lightgreen",
        )
        axes[0, 1].set_xlabel("Timepoint Transition")
        axes[0, 1].set_ylabel("Average Percent Change (%)")
        axes[0, 1].set_title("Cell Size Change")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(transitions)
        axes[0, 1].legend()
        axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Significant differences
        sig_pop = transition_summary["populations_prop_significantly_different"]
        sig_dist = transition_summary["distributions_significantly_different"]

        axes[1, 0].bar(
            x - width / 2,
            sig_pop,
            width,
            label="Population Proportions",
            color="coral",
        )
        axes[1, 0].bar(
            x + width / 2, sig_dist, width, label="Overall Distribution", color="purple"
        )
        axes[1, 0].set_xlabel("Timepoint Transition")
        axes[1, 0].set_ylabel("% Significant Differences")
        axes[1, 0].set_title("Percentage of Significant Differences")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(transitions)
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].legend()

        # Proportion change
        prop_changes = transition_summary["proportion_change_small"]
        axes[1, 1].bar(x, prop_changes, color="orange")
        axes[1, 1].set_xlabel("Timepoint Transition")
        axes[1, 1].set_ylabel("Average Change (percentage points)")
        axes[1, 1].set_title("Small Cell Proportion Change")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(transitions)
        axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        plt.suptitle("Timepoint Population Analysis Summary", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            summary_dir / "timepoint_analysis_summary.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save summary table
        transition_summary.to_csv(summary_dir / "transition_summary.csv", index=False)

    def filter_outlier_cells(self, percentile=99.99):
        """Filter out the largest cells (outliers) from each image."""
        if self.df is None:
            self.load_data()

        print(f"Filtering out cells larger than the {percentile}th percentile...")

        original_count = len(self.df)
        filtered_dfs = []

        for unique_id, group in self.df.groupby("unique_image_id"):
            threshold = np.percentile(group["area"], percentile)
            filtered_group = group[group["area"] <= threshold]
            filtered_dfs.append(filtered_group)

        self.df = pd.concat(filtered_dfs)

        removed_count = original_count - len(self.df)
        removed_percent = (removed_count / original_count) * 100
        print(f"Removed {removed_count} cells ({removed_percent:.3f}% of data)")

        return self.df

    def run_analysis(
        self,
        compare_images=True,
        compare_animals=True,
        compare_timepoints=True,
        filter_outliers=True,
        outlier_percentile=99.995,
    ):
        """Run the complete analysis pipeline."""
        print("Starting cell population analysis...")

        # Step 1: Load data
        self.load_data()

        # Step 2: Filter outliers (optional)
        if filter_outliers:
            self.filter_outlier_cells(percentile=outlier_percentile)

        # Step 3: Fit GMMs
        self.fit_gmm_per_image()

        # Step 4: Save GMM parameters
        gmm_params_path = self.output_dir / "gmm_parameters.csv"
        self.save_gmm_params_to_csv(gmm_params_path)

        # Step 5: Run comparisons
        if compare_images:
            self.compare_same_animal_images()

        if compare_animals:
            self.compare_animals_same_timepoint()

        if compare_timepoints:
            self.compare_timepoint_populations()

        # *** CALL THE NEW PLOTTING FUNCTION ***
        self.plot_rosette_metrics_over_time()

        print("\nAnalysis completed!")
        print(f"All results and visualizations saved to: {self.output_dir}")


def plot_gmm_for_image(
    analyzer, unique_image_id, save_path=None, percentile_cutoff=99.99
):
    """Plot the GMM fit for a specific image with proper handling of log-transformed data."""
    if unique_image_id not in analyzer.gmm_params:
        print(f"No GMM fit available for {unique_image_id}")
        return

    params = analyzer.gmm_params[unique_image_id]
    cell_data_original = params["cell_data"]
    log_transformed = params["log_transformed"]

    # Filter outliers
    cutoff_value = np.percentile(cell_data_original, percentile_cutoff)
    cell_data = cell_data_original[cell_data_original <= cutoff_value]

    n_excluded = len(cell_data_original) - len(cell_data)
    if n_excluded > 0:
        print(
            f"Excluded {n_excluded} cells ({n_excluded / len(cell_data_original) * 100:.2f}%) "
            f"above the {percentile_cutoff} percentile cutoff ({cutoff_value:.2f})"
        )

    # Create figure
    plt.figure(figsize=(10, 6))
    plt.hist(
        cell_data, bins=30, density=True, alpha=0.6, label="Cell area distribution"
    )

    # Plot GMM components
    x = np.linspace(min(cell_data), max(cell_data), 1000)
    mix = np.zeros_like(x)

    for i in range(len(params["means"])):
        weight = params["weights"][i]
        if log_transformed:
            log_mean = params["log_means"][i]
            log_var = params["log_variances"][i]
            log_std = np.sqrt(log_var)

            # Use lognormal distribution
            component = weight * lognorm.pdf(x + 1, s=log_std, scale=np.exp(log_mean))
            plt.plot(x, component, label=f"Component {i + 1} (weight={weight:.2f})")
            mix += component
        else:
            mean = params["means"][i]
            var = params["variances"][i]
            std = np.sqrt(var)
            component = weight * norm.pdf(x, mean, std)
            plt.plot(x, component, label=f"Component {i + 1} (weight={weight:.2f})")
            mix += component

    plt.plot(x, mix, "k-", linewidth=2, label="GMM mixture")

    title = (
        f"Image: {params['image']} (Animal: {params['animal']}, "
        f"Timepoint: {params['timepoint']}, n={len(cell_data)} cells, "
        f"excluded: {n_excluded} cells)"
    )
    plt.title(title)
    plt.xlabel("Cell Area")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced Cell Population Analysis with GMM fitting"
    )
    parser.add_argument("--input", required=True, help="Input CSV file with cell data")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument(
        "--compare-images", action="store_true", help="Compare images from same animal"
    )
    parser.add_argument(
        "--compare-animals",
        action="store_true",
        help="Compare animals at same timepoint",
    )
    parser.add_argument(
        "--compare-timepoints",
        action="store_true",
        help="Compare timepoints (cross-sectional)",
    )
    parser.add_argument(
        "--filter-outliers", action="store_true", help="Filter outlier cells"
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=99.995,
        help="Outlier percentile threshold",
    )

    args = parser.parse_args()

    # Create analyzer and run analysis
    analyzer = CellImageAnalyzer(args.input, args.output)
    analyzer.run_analysis(
        compare_images=args.compare_images,
        compare_animals=args.compare_animals,
        compare_timepoints=args.compare_timepoints,
        filter_outliers=args.filter_outliers,
        outlier_percentile=args.outlier_percentile,
    )


if __name__ == "__main__":
    main()
