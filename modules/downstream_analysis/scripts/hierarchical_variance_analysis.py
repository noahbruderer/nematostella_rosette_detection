#!/usr/bin/env python3
"""
Hierarchical Variance Analysis for Time Series Data
==================================================

Performs a three-level hierarchical variance decomposition:
1. Between timepoints vs within timepoints
2. Between animals vs within animals (for each timepoint)  
3. Between images vs within images (for each animal)

This approach is designed for cross-sectional designs where:
- Animals are unique to each timepoint (sacrificed)
- Multiple images per animal
- Multiple animals per timepoint

Author: Noah Bruderer
Date: 2025
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings("ignore")

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.2)


class HierarchicalVarianceAnalyzer:
    """Performs hierarchical variance decomposition for cross-sectional time series."""
    
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.df = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for analysis."""
        print(f"Loading data from: {self.input_path}")
        
        self.df = pd.read_csv(self.input_path)
        
        # Create derived variables
        self.df['log_area'] = np.log(self.df['area'])
        self.df['timepoint_num'] = self.df['timepoint'].str.replace('T', '').astype(int)
        
        # Ensure all columns are strings for concatenation
        for col in ['replicate', 'timepoint', 'animal', 'image']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
        
        self.df['unique_image_id'] = (
            self.df['replicate'] + "_" + 
            self.df['timepoint'] + "_" + 
            self.df['animal'] + "_" + 
            self.df['image']
        )
        
        print(f"Loaded {len(self.df):,} cells across {self.df['timepoint'].nunique()} timepoints")
        print(f"Animals per timepoint: {self.df.groupby('timepoint')['animal'].nunique().to_dict()}")
        print(f"Images per animal: {self.df.groupby(['timepoint', 'animal'])['unique_image_id'].nunique().mean():.1f} (average)")
        
    def run_hierarchical_analysis(self):
        """Run three-level hierarchical variance decomposition."""
        print("\n" + "="*60)
        print("HIERARCHICAL VARIANCE DECOMPOSITION")
        print("="*60)
        
        results = {}
        
        # Level 1: Between vs Within Timepoint Variance
        print("\n🔍 LEVEL 1: Between vs Within Timepoint Variance")
        print("-" * 50)
        
        # Aggregate to animal level first
        animal_data = self.df.groupby(['timepoint_num', 'animal']).agg({
            'log_area': 'mean',
            'timepoint': 'first'
        }).reset_index()
        
        # Model 1: Overall timepoint effect
        try:
            model1 = MixedLM.from_formula(
                "log_area ~ timepoint_num", 
                data=animal_data, 
                groups=animal_data['timepoint_num']
            )
            result1 = model1.fit()
            
            between_timepoint_var = result1.cov_re.iloc[0, 0] if hasattr(result1, 'cov_re') else 0
            within_timepoint_var = result1.scale
            
            print(f"   Between timepoint variance: {between_timepoint_var:.6f}")
            print(f"   Within timepoint variance: {within_timepoint_var:.6f}")
            
            results['level1'] = {
                'between_timepoint': between_timepoint_var,
                'within_timepoint': within_timepoint_var
            }
        except Exception as e:
            print(f"   Model 1 failed: {e}")
            results['level1'] = {'between_timepoint': 0, 'within_timepoint': 0}
        
        # Level 2: Between vs Within Animal Variance (by timepoint)
        print("\n🔍 LEVEL 2: Between vs Within Animal Variance")
        print("-" * 50)
        
        between_animal_vars = []
        within_animal_vars = []
        
        for tp in sorted(self.df['timepoint_num'].unique()):
            tp_data = self.df[self.df['timepoint_num'] == tp].copy()
            n_animals = tp_data['animal'].nunique()
            n_cells = len(tp_data)
            
            print(f"   T{int(tp)}: {n_animals} animals, {n_cells:,} cells")
            
            if n_animals > 1 and n_cells > 20:
                try:
                    model2 = MixedLM.from_formula(
                        "log_area ~ 1", 
                        data=tp_data, 
                        groups=tp_data['animal']
                    )
                    result2 = model2.fit()
                    
                    between_animal_var = result2.cov_re.iloc[0, 0] if hasattr(result2, 'cov_re') else 0
                    within_animal_var = result2.scale
                    
                    between_animal_vars.append(between_animal_var)
                    within_animal_vars.append(within_animal_var)
                    
                    print(f"      Between animals: {between_animal_var:.6f}")
                    print(f"      Within animals:  {within_animal_var:.6f}")
                    
                except Exception as e:
                    print(f"      Model failed: {e}")
            else:
                print(f"      Insufficient data for analysis")
        
        avg_between_animal = np.mean(between_animal_vars) if between_animal_vars else 0
        avg_within_animal = np.mean(within_animal_vars) if within_animal_vars else 0
        
        print(f"\n   Average between animal variance: {avg_between_animal:.6f}")
        print(f"   Average within animal variance:  {avg_within_animal:.6f}")
        
        results['level2'] = {
            'between_animal': avg_between_animal,
            'within_animal': avg_within_animal,
            'by_timepoint': list(zip(between_animal_vars, within_animal_vars))
        }
        
        # Level 3: Between vs Within Image Variance (by animal)
        print("\n🔍 LEVEL 3: Between vs Within Image Variance")
        print("-" * 50)
        
        image_to_image_vars = []
        cell_to_cell_vars = []
        
        animals_with_multiple_images = 0
        
        for animal in self.df['animal'].unique():
            animal_data = self.df[self.df['animal'] == animal]
            images = animal_data['unique_image_id'].unique()
            
            if len(images) > 1:
                animals_with_multiple_images += 1
                
                # Calculate image-to-image variance (means across images)
                image_means = animal_data.groupby('unique_image_id')['log_area'].mean()
                image_var = image_means.var()
                image_to_image_vars.append(image_var)
                
                # Calculate average cell-to-cell variance within images
                cell_vars = []
                for img in images:
                    img_data = animal_data[animal_data['unique_image_id'] == img]
                    if len(img_data) > 1:
                        cell_vars.append(img_data['log_area'].var())
                
                if cell_vars:
                    avg_cell_var = np.mean(cell_vars)
                    cell_to_cell_vars.append(avg_cell_var)
        
        avg_image_to_image = np.mean(image_to_image_vars) if image_to_image_vars else 0
        avg_cell_to_cell = np.mean(cell_to_cell_vars) if cell_to_cell_vars else 0
        
        print(f"   Animals with multiple images: {animals_with_multiple_images}")
        print(f"   Average image-to-image variance: {avg_image_to_image:.6f}")
        print(f"   Average cell-to-cell variance:   {avg_cell_to_cell:.6f}")
        
        results['level3'] = {
            'image_to_image': avg_image_to_image,
            'cell_to_cell': avg_cell_to_cell,
            'n_animals_multiple_images': animals_with_multiple_images
        }
        
        return results
    
    def create_variance_summary(self, results):
        """Create comprehensive variance summary and visualization."""
        print("\n" + "="*60)
        print("VARIANCE COMPONENT SUMMARY")
        print("="*60)
        
        # Extract variance components
        between_timepoint = results['level1']['between_timepoint']
        between_animal = results['level2']['between_animal'] 
        image_to_image = results['level3']['image_to_image']
        cell_to_cell = results['level3']['cell_to_cell']
        
        # Calculate total variance
        total_var = between_timepoint + between_animal + image_to_image + cell_to_cell
        
        # Create summary DataFrame
        variance_data = {
            'Level': [
                '1. Between Timepoints',
                '2. Between Animals (within timepoint)', 
                '3. Between Images (within animal)',
                '4. Between Cells (within image)',
                'TOTAL'
            ],
            'Variance': [
                between_timepoint,
                between_animal,
                image_to_image, 
                cell_to_cell,
                total_var
            ],
            'Proportion': [
                between_timepoint / total_var if total_var > 0 else 0,
                between_animal / total_var if total_var > 0 else 0,
                image_to_image / total_var if total_var > 0 else 0,
                cell_to_cell / total_var if total_var > 0 else 0,
                1.0
            ],
            'Percentage': [
                100 * between_timepoint / total_var if total_var > 0 else 0,
                100 * between_animal / total_var if total_var > 0 else 0,
                100 * image_to_image / total_var if total_var > 0 else 0,
                100 * cell_to_cell / total_var if total_var > 0 else 0,
                100.0
            ]
        }
        
        summary_df = pd.DataFrame(variance_data)
        
        # Print summary
        print(summary_df.to_string(index=False, float_format='%.6f'))
        
        # Identify the largest variance source
        max_idx = summary_df.iloc[:-1]['Proportion'].idxmax()  # Exclude total row
        largest_source = summary_df.iloc[max_idx]['Level']
        largest_percent = summary_df.iloc[max_idx]['Percentage']
        
        print(f"\n🎯 LARGEST VARIANCE SOURCE: {largest_source}")
        print(f"   Accounts for {largest_percent:.1f}% of total variance")
        
        if 'Between Images' in largest_source:
            print("   → This suggests high technical variability between images from the same animal")
        elif 'Between Cells' in largest_source:
            print("   → This suggests high biological variability between individual cells")
        elif 'Between Animals' in largest_source:
            print("   → This suggests high biological variability between animals")
        elif 'Between Timepoints' in largest_source:
            print("   → This suggests significant temporal changes (good for your hypothesis!)")
        
        # Save results
        summary_df.to_csv(self.output_dir / "lmm_variance_components.csv", index=False)
        
        # Create visualization
        self.create_variance_plot(summary_df[:-1])  # Exclude total row for plot
        
        return summary_df
    
    def create_variance_plot(self, summary_df):
        """Create variance component visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax1.bar(range(len(summary_df)), summary_df['Percentage'], color=colors)
        ax1.set_xlabel('Variance Source')
        ax1.set_ylabel('Percentage of Total Variance (%)')
        ax1.set_title('Hierarchical Variance Decomposition')
        ax1.set_xticks(range(len(summary_df)))
        ax1.set_xticklabels([label.split('. ')[1] for label in summary_df['Level']], rotation=45)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, summary_df['Percentage']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(summary_df['Percentage'], labels=[label.split('. ')[1] for label in summary_df['Level']], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Variance Component Proportions')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "lmm_variance_components_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Variance plot saved to: {self.output_dir / 'lmm_variance_components_plot.png'}")
    
    def save_detailed_summary(self, results, summary_df):
        """Save detailed text summary."""
        summary_path = self.output_dir / "lmm_model_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("HIERARCHICAL VARIANCE DECOMPOSITION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ANALYSIS OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write("This analysis decomposes variance into hierarchical levels:\n")
            f.write("1. Between timepoints (temporal changes)\n")
            f.write("2. Between animals within timepoints (biological variation)\n") 
            f.write("3. Between images within animals (technical variation)\n")
            f.write("4. Between cells within images (cellular heterogeneity)\n\n")
            
            f.write("LEVEL 1 - TIMEPOINT EFFECTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Between timepoint variance: {results['level1']['between_timepoint']:.6f}\n")
            f.write(f"Within timepoint variance:  {results['level1']['within_timepoint']:.6f}\n\n")
            
            f.write("LEVEL 2 - ANIMAL EFFECTS (by timepoint):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average between animal variance: {results['level2']['between_animal']:.6f}\n")
            f.write(f"Average within animal variance:  {results['level2']['within_animal']:.6f}\n\n")
            
            f.write("LEVEL 3 - IMAGE EFFECTS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Image-to-image variance: {results['level3']['image_to_image']:.6f}\n")
            f.write(f"Cell-to-cell variance:   {results['level3']['cell_to_cell']:.6f}\n")
            f.write(f"Animals with multiple images: {results['level3']['n_animals_multiple_images']}\n\n")
            
            f.write("VARIANCE COMPONENT SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(summary_df.to_string(index=False, float_format='%.6f'))
            f.write("\n\n")
            
            # Add interpretation
            max_idx = summary_df.iloc[:-1]['Proportion'].idxmax()
            largest_source = summary_df.iloc[max_idx]['Level']
            largest_percent = summary_df.iloc[max_idx]['Percentage']
            
            f.write("INTERPRETATION:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Largest variance source: {largest_source} ({largest_percent:.1f}%)\n\n")
            
            if largest_percent > 50:
                f.write("⚠️  This source dominates total variance - investigate further!\n")
            elif largest_percent > 30:
                f.write("ℹ️  This is the primary source but others are also substantial\n")
            else:
                f.write("ℹ️  Variance is distributed across multiple sources\n")
        
        print(f"✅ Detailed summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical variance analysis for time series data")
    parser.add_argument("--input-csv", required=True, help="Input CSV file path")
    parser.add_argument("--output-dir", required=True, help="Output directory path")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = HierarchicalVarianceAnalyzer(args.input_csv, args.output_dir)
    analyzer.load_and_prepare_data()
    results = analyzer.run_hierarchical_analysis()
    summary_df = analyzer.create_variance_summary(results)
    analyzer.save_detailed_summary(results, summary_df)
    
    print(f"\n✅ Hierarchical variance analysis completed!")
    print(f"   Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()