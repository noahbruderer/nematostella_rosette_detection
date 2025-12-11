# Inference Rosettes Module

## Overview

The **inference_rosettes** module performs rosette detection inference using trained Attention UNet models. This module loads model parameters directly from saved training configurations to ensure perfect consistency between training and inference.

## Features

- **Model Parameter Loading**: Automatically loads model architecture and parameters from training config.json files
- **Sliding Window Inference**: Handles large images through patch-based inference with overlap handling
- **Multi-format Output**: Saves predictions as H5 files and visualizations as PNG images
- **Flexible Batch Processing**: Configurable batch size for memory optimization
- **Comprehensive Visualization**: Creates 4-panel visualizations showing raw image, cell contours, prediction heatmap, and overlay

## Directory Structure

```
modules/inference_rosettes/
├── config.yaml          # Module configuration
├── rules.smk            # Snakemake rules
├── pyproject.toml       # Poetry dependencies
├── scripts/
│   └── inference.py     # Main inference script
└── README.md           # This file
```

## Configuration

The module configuration (`config.yaml`) specifies:

- **Model Selection**: Which trained experiment to use for inference
- **Input/Output Paths**: Data directories and output locations  
- **Inference Parameters**: Batch size, confidence thresholds
- **Resource Requirements**: Memory, threads, time limits

### Key Configuration Options

```yaml
model_selection:
  experiment_name: "baseline_test"  # Which trained model to use

inference:
  batch_size: 8                    # Inference batch size
  confidence_threshold: 0.5        # Binary prediction threshold
  save_predictions: true           # Save raw predictions (H5)
  save_visualizations: true        # Save visualization images (PNG)
```

## Usage

### Single Sample Inference

Run inference on a specific sample:

```bash
snakemake results/inference_rosettes/SAMPLE_NAME/predictions_SAMPLE_NAME.h5 -p
```

### Generate Visualizations

Create visualization for a specific sample:

```bash
snakemake results/inference_rosettes/SAMPLE_NAME/visualization_SAMPLE_NAME.png -p
```

### Batch Inference

Run inference on all available samples:

```bash
snakemake results/inference_rosettes.flag -p
```

## Input Requirements

The module requires:

1. **Processed Training Data**: H5 files from the `training_data_preparation` module
2. **Trained Model**: Model weights and config from the `model_training` module

### Expected Input Structure

```
results/training_data_preparation/SAMPLE_NAME/processed_data_SAMPLE_NAME.h5
results/model_training/EXPERIMENT_NAME/best_model.pth
results/model_training/EXPERIMENT_NAME/config.json
```

## Outputs

### Predictions (H5 Files)

- **Location**: `results/inference_rosettes/SAMPLE_NAME/predictions_SAMPLE_NAME.h5`
- **Content**: Raw prediction probability maps (0.0 to 1.0)
- **Format**: HDF5 with dataset 'predictions' and metadata attributes

### Visualizations (PNG Images)

- **Location**: `results/inference_rosettes/SAMPLE_NAME/visualization_SAMPLE_NAME.png`
- **Content**: 4-panel visualization showing:
  1. **Raw Image**: Original microscopy image
  2. **Cell Contours**: Cell boundaries overlaid on raw image
  3. **Prediction Heatmap**: Raw probability predictions with colorbar
  4. **Prediction Overlay**: Binary predictions overlaid on raw image

### Summary Metrics

- **Location**: `results/inference_rosettes/inference_metrics.csv`
- **Content**: Summary statistics for all processed samples

## Algorithm Details

### Model Loading

1. Loads model configuration from `config.json` to determine architecture parameters
2. Creates Attention UNet model with correct number of input channels
3. Loads pre-trained weights from `best_model.pth`

### Data Preprocessing

Applies the same preprocessing as training:
- Creates cell boundary features using `find_boundaries()`
- Generates binary cell masks from instance segmentation
- Stacks geometric features (2-channel) or adds intensity (3-channel)

### Sliding Window Inference

For large images:
1. Divides image into overlapping patches (default 512x512 with 64px overlap)
2. Processes patches in configurable batches
3. Aggregates overlapping predictions using weighted averaging

### Post-processing

- Applies confidence thresholding for binary predictions
- Creates multi-panel visualizations
- Saves results in structured output format

## Dependencies

Key dependencies managed through Poetry:

- **PyTorch**: Model inference and tensor operations
- **scikit-image**: Image processing and boundary detection
- **h5py**: HDF5 file handling
- **matplotlib**: Visualization generation
- **numpy**: Numerical computations

## Model Consistency

The module ensures perfect consistency with training by:

- **Loading exact model parameters** from training config.json
- **Using identical preprocessing** pipeline as training
- **Importing model architecture** directly from training module
- **Matching input channel configuration** (geometric-only vs with-intensity)

## Performance Considerations

- **Memory Usage**: Adjust `batch_size` based on available GPU/CPU memory
- **Processing Speed**: Larger patches reduce overhead but increase memory usage
- **Overlap**: More overlap improves prediction quality but increases computation

## Troubleshooting

### Common Issues

1. **Missing Model Files**: Ensure training module has completed successfully
2. **Memory Errors**: Reduce batch_size in configuration
3. **Path Issues**: Verify input data exists in expected locations
4. **Architecture Mismatch**: Check that model_selection.experiment_name points to existing model

### Debug Mode

For debugging, run with verbose output:

```bash
snakemake results/inference_rosettes/SAMPLE_NAME/predictions_SAMPLE_NAME.h5 -p --verbose
```

## Integration

This module integrates with the rosette detection pipeline:

- **Depends on**: `model_training` (for trained models)
- **Uses data from**: `training_data_preparation` (for processed H5 files)
- **Enables**: Downstream analysis and quantification modules