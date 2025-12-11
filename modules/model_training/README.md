# Model Training Module

Train geometric rosette detection models using Attention U-Net architecture with configurable experiment parameters.

## Overview

This module trains deep learning models for rosette detection using processed training data from the `training_data_preparation` module. It supports multiple experiments with different hyperparameters, data augmentation settings, and training configurations.

## Architecture

- **Model**: Attention U-Net with skip connections and attention gates
- **Input**: 2-channel (geometric-only) or 3-channel (with intensity) image patches
- **Output**: Binary segmentation masks for rosette detection
- **Framework**: PyTorch with albumentations for data augmentation

## Configuration

### Main Config File: `config.yaml`

The module uses a configuration-driven approach where all parameters are defined in YAML:

```yaml
# Model architecture settings
model:
  architecture: "attention_unet"
  use_intensity: false          # true = 3 channels, false = 2 channels
  patch_size: 512
  num_classes: 1               # Binary rosette detection

# Default training parameters (can be overridden per experiment)
default_training:
  num_epochs: 100
  batch_size: 4
  learning_rate: 0.001
  val_split: 0.2              # 20% validation split
  seed: 42
  
  # Data augmentation settings
  augmentation:
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation: 0.5
    elastic_transform: 0.25
    affine_transform: 0.5
    coarse_dropout: 0.2

# Experiment matrix - each experiment can override defaults
experiments:
  - name: "baseline_test"
    description: "Quick test run - 1 epoch baseline"
    num_epochs: 1
    
  - name: "baseline_full"
    description: "Full baseline training"
    num_epochs: 100
    
  - name: "higher_lr"
    description: "Higher learning rate experiment"
    num_epochs: 50
    learning_rate: 0.01
    
  - name: "larger_batch"
    description: "Larger batch size experiment"
    batch_size: 8
    
  - name: "no_augmentation"
    description: "Training without data augmentation"
    augmentation:
      horizontal_flip: 0.0
      vertical_flip: 0.0
      rotation: 0.0
      elastic_transform: 0.0
      affine_transform: 0.0
      coarse_dropout: 0.0
```

## Usage

### Run All Experiments
```bash
# From the main pipeline directory
python -m snakemake model_training_all -c1
```

### Run Single Experiment
```bash
# Run just the baseline test (1 epoch)
python -m snakemake results/model_training/baseline_test/best_model.pth -c1
```

### Run Specific Scripts Manually

#### Train a Model
```bash
cd modules/model_training
poetry run python scripts/train_model.py \
    --split-manifest "../../results/model_training/split_manifest_baseline_test.json" \
    --data-dir "../../results/training_data_preparation" \
    --output-dir "../../results/model_training/baseline_test" \
    --config "config.yaml" \
    --experiment-name "baseline_test" \
    --no-timestamp
```

#### Create Train/Val Split
```bash
cd modules/model_training  
poetry run python scripts/create_split.py \
    --samples T1_1 T1_2 T2_1 T2_2 \
    --config-dir "../../results/model_training" \
    --config-file "config.yaml" \
    --experiment-name "baseline_test" \
    --output-name "split_manifest_baseline_test.json"
```

## Input Requirements

The module requires processed training data from `training_data_preparation`:

```
results/training_data_preparation/
├── sample_1/
│   ├── processed_data_sample_1.h5      # Required: Image patches and labels
│   ├── cell_properties_sample_1.csv    # Optional: Cell metadata
│   └── neighborhood_graph_sample_1.pkl # Optional: Spatial relationships
├── sample_2/
│   └── ...
```

## Output Structure

```
results/model_training/
├── split_manifest_baseline_test.json          # Train/val split metadata
├── split_manifest_baseline_full.json
├── ...
├── baseline_test/                              # Experiment results
│   ├── best_model.pth                         # Trained model weights
│   ├── config.json                            # Experiment configuration
│   ├── training_plots.png                     # Loss/metric curves
│   └── training.log                           # Training logs
├── baseline_full/
│   └── ...
├── experiment_summary.csv                     # Comparison across experiments
└── model_training.flag                        # Module completion marker
```

## Design Patterns

This module follows the **Script-Side Configuration Loading** pattern:

- **Snakemake** passes only workflow structure parameters (file paths, experiment names)
- **Python scripts** load ALL execution parameters from config files
- **Benefits**: Clean separation of concerns, easy to add new parameters, single source of truth

### What Snakemake Passes:
- File paths (`--split-manifest`, `--data-dir`, `--output-dir`)
- Config loading (`--config`, `--experiment-name`)  
- Workflow behavior (`--no-timestamp`)

### What Scripts Load from Config:
- Training hyperparameters (`num_epochs`, `batch_size`, `learning_rate`)
- Model parameters (`patch_size`, `use_intensity`, `architecture`)
- Data augmentation settings (per-experiment overrides)

## Adding New Experiments

1. **Add to config.yaml**:
```yaml
experiments:
  - name: "my_experiment"
    description: "Custom experiment description"
    num_epochs: 75
    learning_rate: 0.005
    batch_size: 6
    augmentation:
      horizontal_flip: 0.8  # Override specific augmentation
```

2. **Run the pipeline** - new experiment will be automatically discovered and executed.

## Customizing Parameters

### Override Individual Parameters (for debugging):
```bash
poetry run python scripts/train_model.py \
    --config "config.yaml" \
    --experiment-name "baseline_test" \
    --override num_epochs=1 \
    --override batch_size=2 \
    # ... other args
```

### Add New Hyperparameters:
1. Add to `config.yaml` defaults and experiment sections
2. Update `scripts/train_model.py` parameter loading section
3. No changes needed in Snakemake rules

## Hardware Requirements

- **GPU/MPS recommended** for reasonable training times
- **Memory**: 4GB+ RAM per training job
- **Storage**: ~500MB per experiment (model + logs + plots)
- **Time**: 1-12 hours depending on epochs and data size

## Troubleshooting

### Common Issues:

**1. Out of Memory**
```bash
# Reduce batch size in config
batch_size: 2  # instead of 4
```

**2. Training Too Slow**
```bash
# Use test experiment first
python -m snakemake results/model_training/baseline_test/best_model.pth -c1
```

**3. Data Loading Errors**
```bash
# Check training data exists
ls results/training_data_preparation/*/processed_data_*.h5
```

**4. Config Parameter Not Loading**
- Check YAML syntax and indentation
- Verify experiment name matches exactly
- Check parameter spelling in both config and script

### Logs and Debugging:

- **Training logs**: `logs/model_training/{experiment_name}.log`
- **Snakemake logs**: `.snakemake/log/`
- **Script output**: Captured in rule logs with full parameter details

## Dependencies

Managed via Poetry in `pyproject.toml`:
- `torch` - Deep learning framework
- `albumentations` - Data augmentation
- `h5py` - HDF5 file reading
- `scikit-learn` - Train/val splitting
- `matplotlib` - Training plots
- `pyyaml` - Configuration loading

Install with:
```bash
cd modules/model_training
poetry install --no-root
```