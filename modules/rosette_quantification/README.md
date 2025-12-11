# Rosette Quantification Module

A comprehensive pipeline for quantifying rosettes in biological images and analyzing cell population dynamics across different experimental conditions.

## 📁 Directory Structure

```
rosette_quantification/
├── README.md                 # This file
├── config.yaml              # Configuration parameters
├── rules.smk                # Snakemake workflow rules
├── pyproject.toml           # Poetry dependencies
├── poetry.lock              # Locked dependencies
└── scripts/                 # Analysis scripts
    ├── helper_functions.py  # Core analysis utilities
    ├── quantify_rosettes.py # Basic rosette quantification
    ├── process_inhibitor_sample.py  # Inhibitor experiment processing
    ├── concatenate_results.py      # Data aggregation and metadata extraction
    ├── create_verification_plots.py # Data validation visualizations
    └── advanced_cell_analysis.py   # Advanced statistical analysis with GMM
```

## 🔬 Analysis Pipeline Overview

The module processes three types of experiments:
1. **Regular quantification**: Standard rosette counting from curated masks
2. **Inhibitor experiments**: Drug treatment effects on rosette formation
3. **Time series experiments**: Cross-sectional analysis of rosette dynamics over time

## 📜 Script Descriptions

### Core Analysis Scripts

#### `helper_functions.py`
**Purpose**: Core utilities for rosette detection and analysis

**Key Functions**:
- `identify_rosette_cells()`: Maps rosette regions to cell segmentation masks
- `label_cells_from_boundaries()`: Creates labeled instances from boundary masks
- `measure_cell_properties_with_rosettes_and_neighbors()`: Extracts comprehensive cell features
- `neigh_graph_pipeline()`: Builds cell neighborhood networks
- `highlight_cells_with_rosettes_with_boundaries()`: Creates interactive visualizations

**Features**:
- NetworkX graph analysis with GML export compatibility
- Cell property measurements (area, perimeter, rosette membership)
- Interactive HTML visualizations with Plotly

---

#### `quantify_rosettes.py`
**Purpose**: Basic rosette quantification from curated segmentation and rosette masks

**Usage**:
```bash
poetry run python scripts/quantify_rosettes.py \
    --segmentation path/to/masks_curated.tif \
    --rosettes path/to/rosettes_curated.tif \
    --output results/sample_name \
    --sample sample_name \
    --min-rosette-size 10 \
    --max-rosette-size 1000 \
    --normalization-factor 1000
```

**Outputs**:
- `quantification_SAMPLE.csv`: Summary metrics (total cells, rosettes, rosettes per 1000 cells)
- `cell_properties_SAMPLE.csv`: Individual cell measurements
- `rosette_highlight_vis_SAMPLE.html`: Interactive visualization
- `timing_SAMPLE.csv`: Performance benchmarks

---

#### `process_inhibitor_sample.py`
**Purpose**: Specialized processing for inhibitor experiment samples with metadata extraction

**Usage**:
```bash
poetry run python scripts/process_inhibitor_sample.py \
    --segmentation path/to/segmentation.tif \
    --rosettes path/to/rosettes_curated.tif \
    --output results/sample \
    --sample sample_name
```

**Features**:
- Automatic metadata parsing from sample names (replicate, timepoint, inhibitor, control status)
- Enhanced error handling for batch processing
- Comprehensive summary statistics with treatment information

**Outputs**:
- `SAMPLE_cell_properties_with_metadata.csv`: Cell data with experimental metadata
- `SAMPLE_summary.csv`: Sample-level statistics
- `SAMPLE_timing.csv`: Processing benchmarks

---

### Data Processing Scripts

#### `concatenate_results.py`
**Purpose**: Aggregates individual sample results into unified datasets with metadata extraction

**Usage**:
```bash
poetry run python scripts/concatenate_results.py \
    --input-files results/*/cell_properties_*.csv \
    --output combined_cell_properties.csv \
    --output-aggregated animal_aggregated_data.csv \
    --experiment-type timeseries \
    --base-dir results/
```

**Key Features**:
- **Smart metadata extraction**: Parses sample names and directory paths to extract:
  - Replicate ID (R1, R2, R3, ...)
  - Timepoint (T0, T1, T5, T20, ...)
  - Animal ID
  - Image ID
  - Inhibitor information (for inhibitor experiments)
  - Control/treatment status

- **Multi-level aggregation**:
  - Cell-level: Individual cell measurements with metadata
  - Animal-level: Aggregated statistics per animal for statistical analysis

- **Experiment type handling**:
  - `regular`: Standard quantification experiments
  - `inhibitor`: Drug treatment experiments with control/treatment parsing
  - `timeseries`: Cross-sectional time series experiments

**Outputs**:
- Combined CSV with all cell measurements and extracted metadata
- Animal-aggregated CSV for statistical analysis
- Comprehensive summary statistics

---

#### `create_verification_plots.py`
**Purpose**: Generates comprehensive data validation and quality control visualizations

**Usage**:
```bash
poetry run python scripts/create_verification_plots.py \
    --input combined_cell_properties.csv \
    --output-dir verification/ \
    --experiment-type timeseries \
    --flag-file verification_complete.flag
```

**Verification Categories**:

1. **Input Verification**:
   - Image count distributions per animal
   - Sample size validation
   - Data completeness checks

2. **Rosette Statistics**:
   - Overall rosette percentages
   - Distribution by timepoint/condition
   - Control vs treatment comparisons (inhibitor experiments)

3. **Cell Count Analysis**:
   - Cell count distributions by animal/timepoint
   - Statistical summaries and outlier detection

4. **Sample Overview**:
   - Sample count heatmaps by replicate/timepoint
   - Experiment design validation
   - Data structure summaries

**Output Structure**:
```
verification/
├── {experiment_type}_input_verification/
│   ├── image_count_histogram.png
│   ├── image_count_statistics.csv
│   └── timepoint_summary.png
├── {experiment_type}_rosette_statistics/
│   ├── rosette_summary.txt
│   ├── rosette_by_timepoint.png
│   └── control_vs_treatment.png  (inhibitor only)
├── {experiment_type}_cell_counts/
│   ├── cell_count_boxplot.png
│   └── animal_cell_counts.csv
└── {experiment_type}_overview/
    ├── sample_count_heatmap.png
    ├── sample_overview.csv
    └── experiment_summary.csv
```

---

### Advanced Analysis Script

#### `advanced_cell_analysis.py`
**Purpose**: Comprehensive statistical analysis using Gaussian Mixture Models for population-level comparisons

**Usage**:
```bash
poetry run python scripts/advanced_cell_analysis.py \
    --input combined_cell_properties.csv \
    --output advanced_analysis/ \
    --compare-images \
    --compare-animals \
    --compare-timepoints \
    --filter-outliers \
    --outlier-percentile 99.995
```

**Key Features**:

1. **Gaussian Mixture Model Fitting**:
   - Fits 2-component GMM to cell area distributions per image
   - Optional log transformation for biological data
   - Handles small and large cell populations separately

2. **Multi-Level Comparisons**:
   - **Image-level**: Compare distributions between images from the same animal
   - **Animal-level**: Compare distributions between animals at same timepoint  
   - **Population-level**: Cross-sectional timepoint analysis

3. **Statistical Tests**:
   - Kolmogorov-Smirnov tests for distribution differences
   - Chi-square tests for population proportion changes
   - Parameter comparison (means, variances, mixture weights)

4. **Advanced Visualizations**:
   - GMM fit overlays on histograms
   - Population comparison plots
   - Time series trend analysis
   - Statistical significance heatmaps

**Python API**:
```python
from scripts.advanced_cell_analysis import CellImageAnalyzer

analyzer = CellImageAnalyzer(
    data_path="combined_cell_properties.csv",
    output_dir="advanced_analysis/"
)

# Load and preprocess data
analyzer.load_data()
analyzer.filter_outlier_cells(percentile=99.995)

# Fit GMMs to each image
analyzer.fit_gmm_per_image(n_components=2, use_log_transform=True)

# Save GMM parameters
analyzer.save_gmm_params_to_csv("gmm_parameters.csv")

# Run comparative analyses
analyzer.compare_same_animal_images()
analyzer.compare_animals_same_timepoint() 
analyzer.compare_timepoint_populations()

# Or run everything at once
analyzer.run_analysis(
    compare_images=True,
    compare_animals=True,
    compare_timepoints=True,
    filter_outliers=True
)
```

**Output Structure**:
```
advanced_analysis/
├── gmm_parameters.csv
├── image_comparisons/
│   ├── same_animal_image_comparison.csv
│   └── rep*_tp*_animal*_comparison.png
├── animal_comparisons/
│   ├── animal_comparison_same_timepoint.csv
│   └── rep*_tp*_animal*_vs_animal*.png
└── timepoint_populations/
    ├── timepoint_population_comparison_results.csv
    ├── rep*_tp*_to_tp*.png
    └── summary/
        ├── timepoint_analysis_summary.png
        └── transition_summary.csv
```

## 🔧 Configuration

### Main Configuration (`config.yaml`)

Key parameters that can be adjusted:

```yaml
# Quantification parameters
quantification:
  min_rosette_size: 10      # Minimum rosette size (pixels)
  max_rosette_size: 1000    # Maximum rosette size (pixels)
  min_cell_size: 50         # Minimum cell size (pixels)
  normalization_factor: 1000 # Rosettes per X cells

# Experiment configurations
inhibitor_experiments:
  enabled: false            # Enable inhibitor processing
  input_dir: "/path/to/inhibitor/data"
  
timeseries_experiments:
  enabled: true             # Enable time series processing
  input_dir: "/path/to/timeseries/data"
  expected_file_patterns:
    segmentation: "_seg_cur.tif"
    rosettes_curated: "_seg_cur copy.tif"
```

## 🚀 Running the Pipeline

### Complete Snakemake Pipeline
```bash
# Run all rosette quantification
snakemake --cores 4 results/rosette_quantification.flag

# Run time series experiments only
snakemake --cores 4 results/rosette_quantification_timeseries.flag

# Run with concatenation and verification
snakemake --cores 4 results/rosette_quantification/verification_plots.flag
```

### Individual Script Execution
```bash
# Basic quantification
python scripts/quantify_rosettes.py --segmentation ... --rosettes ... --output ... --sample ...

# Concatenate results
python scripts/concatenate_results.py --input-files results/*/cell_*.csv --output combined.csv --experiment-type timeseries

# Create verification plots
python scripts/create_verification_plots.py --input combined.csv --output-dir verification/ --experiment-type timeseries

# Advanced analysis
python scripts/advanced_cell_analysis.py --input combined.csv --output advanced/ --compare-timepoints
```

## 📊 Expected Data Format

### Input Requirements
- **Segmentation masks**: `.tif` files with labeled cell instances
- **Rosette masks**: `.tif` files with rosette region annotations
- **Directory structure**: Organized by experiment type with consistent naming

### Sample Naming Conventions
The pipeline automatically extracts metadata from sample names:

- **Regular**: `R{replicate}_T{timepoint}_{animal}_{image}`
- **Inhibitor**: `R{replicate}_T{timepoint}_{inhibitor}_{animal}_{image}` or `R{replicate}_T{timepoint}_{inhibitor}_dmso_{animal}_{image}`
- **Time series**: Directory structure: `T{timepoint}_seg/{sample_dir}/files.tif`

### Output Data Columns
Cell-level data includes:
- `area`, `perimeter`: Cell morphology
- `is_in_rosette`: Boolean rosette membership
- `rosette_id`: Unique rosette identifier
- `replicate`, `timepoint`, `animal`, `image`: Experimental metadata
- `sample`: Original sample identifier

## 🔍 Quality Control

### Data Validation Checklist
1. **Sample completeness**: All expected timepoints/conditions present
2. **Image counts**: Consistent number of images per animal
3. **Cell counts**: Reasonable cell numbers per image
4. **Rosette percentages**: Within expected biological ranges
5. **Missing data**: No unexpected missing values

### Common Issues and Solutions
- **Low cell counts**: Check segmentation quality, adjust `min_cell_size`
- **High rosette percentages**: Verify rosette annotations, check `min_rosette_size`
- **Missing metadata**: Ensure consistent sample naming conventions
- **Processing failures**: Check log files in `logs/rosette_quantification/`

## 📈 Interpreting Results

### Basic Quantification
- **Total cells**: Number of segmented cells per sample
- **Total rosettes**: Number of detected rosette structures  
- **Rosettes per 1000 cells**: Normalized rosette frequency for comparison

### Advanced Analysis
- **GMM Components**: Small vs large cell populations with distinct size distributions
- **KS p-values**: Distribution similarity (p < 0.05 = significantly different)
- **Population changes**: Percentage changes in cell populations over time
- **Statistical significance**: Proportion and distribution changes across conditions

## 🐛 Troubleshooting

### Common Errors
1. **File not found**: Check file paths and naming conventions
2. **Memory errors**: Reduce batch size or increase memory allocation
3. **GMM fitting failures**: Insufficient cells per image (< 10 cells)
4. **Missing dependencies**: Run `poetry install --no-root`

### Performance Optimization
- Use `--cores` flag to parallelize processing
- Filter outlier cells to improve GMM fitting
- Process subsets for testing before full pipeline runs

## 📚 References

- **NetworkX**: Graph analysis for cell neighborhoods
- **scikit-learn**: Gaussian Mixture Model implementation  
- **SciPy**: Statistical testing (KS test, Chi-square)
- **Plotly**: Interactive visualizations
- **Snakemake**: Workflow management and parallelization