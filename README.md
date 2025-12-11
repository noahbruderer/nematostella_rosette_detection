# Rosette Analysis Pipeline

A comprehensive Snakemake workflow for automated rosette detection in biological images using deep learning and image analysis.

## Quick Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd nematostella_rosette_detection

# Install dependencies
poetry install --no-root
poetry shell
```

### 2. Configure Data Paths

Set the following environment variables before running the pipeline:

```bash
# Path to your raw data directory
export ROSETTE_DATA_ROOT="/path/to/your/data"

# Working directory (can be current directory)
export ROSETTE_WORK_DIR="$(pwd)"

# Add to your ~/.bashrc or ~/.zshrc for persistence
echo 'export ROSETTE_DATA_ROOT="/path/to/your/data"' >> ~/.bashrc
echo 'export ROSETTE_WORK_DIR="$(pwd)"' >> ~/.bashrc
```

### 3. Data Structure Expected

Your `ROSETTE_DATA_ROOT` should have this structure:

```
your_data/
├── timeseries/
│   └── 1_raw/
├── inhibitors/
│   └── 1_raw/
├── constant_fed/
└── (curated rosette annotations)
```

### 4. Download Model (Champion Baseline - 61.3% F1)

Since the trained model (120MB) is too large for GitHub, download it from Google Drive:

**Download from shared folder:**
1. Visit: https://drive.google.com/drive/folders/1Z9teE1vwKcxnfmPDPrBWmuL01XE7Oa1N
2. Download `best_model.pth` 
3. Place it in: `results/model_training/champion_baseline/best_model.pth`

**Quick setup:**
```bash
# Create the model directory
mkdir -p results/model_training/champion_baseline

# Then manually download and place the model file from the link above
# Or use gdown if available:
# pip install gdown
# gdown --folder https://drive.google.com/drive/folders/1Z9teE1vwKcxnfmPDPrBWmuL01XE7Oa1N
```

**Alternative: Use your own model**
- Train a new model using the pipeline
- Replace `champion_baseline` in config with your model name

## Pipeline Overview

This pipeline provides an end-to-end solution for:

- **Image Segmentation**: Cell segmentation using CellPose
- **Post-processing**: Segmentation mask curation and cleanup
- **Model Training**: Training Attention UNet models with multiple experiments
- **Inference**: Automated rosette detection using trained models (F1: 61.3%)
- **Quantification**: Rosette analysis across different experimental conditions
- **Analysis**: Statistical analysis and visualization

### Pipeline Flow
```
Raw Images → Image Segmentation → Post-processing → Inference → Quantification → Analysis
                                                         ↗
Training Labels → Training Data Preparation → Model Training ↗
```

## Running the Pipeline

### Full Pipeline
```bash
python -m snakemake all -c8
```

### Individual Modules
```bash
# Image segmentation only
python -m snakemake image_segmentation_all -c4

# Model training only
python -m snakemake model_training_all -c2

# Inference only
python -m snakemake inference_rosettes_all -c1
```

### Configuration Dashboard
```bash
# View current configuration and sample discovery
python -m snakemake show_config_glob
```

## Model Performance

The pipeline uses the **Champion Baseline** model with:
- **F1 Score**: 61.33%
- **IoU**: 50.91% 
- **Architecture**: Attention UNet
- **Input**: 2-channel (cell boundaries + cell masks)

## Configuration

### Main Configuration: `config/pipeline.yaml`

Key settings you can modify:
```yaml
global:
  num_workers: 8          # CPU cores to use
  default_memory_gb: 8    # Memory per job
  
modules:
  # Enable/disable modules
  image_segmentation:
    enabled: true
  inference_rosettes:
    enabled: true
    model_selection:
      experiment_name: "champion_baseline"  # Which model to use
```

### Module-Specific Configs
- `modules/image_segmentation/config.yaml` - CellPose parameters
- `modules/model_training/config.yaml` - Training experiments
- `modules/inference_rosettes/config.yaml` - Inference settings

## Output Structure

```
results/
├── image_segmentation/           # Cell segmentation masks
├── segmentation_post_processing/ # Curated masks
├── model_training/              # Trained models
│   └── champion_baseline/       # Best model (F1: 61.3%)
├── inference_rosettes/          # Rosette predictions
├── rosette_quantification/      # Quantified results
├── downstream_analysis/         # Statistical analysis
└── final_report.html           # Summary report
```

## Troubleshooting

### Common Issues

**Environment Variables Not Set**
```bash
# Check if variables are set
echo $ROSETTE_DATA_ROOT
echo $ROSETTE_WORK_DIR

# Set them if missing
export ROSETTE_DATA_ROOT="/path/to/your/data"
export ROSETTE_WORK_DIR="$(pwd)"
```

**Missing Model File**
```bash
# Check if model exists
ls -la results/model_training/champion_baseline/best_model.pth

# Download or train if missing (see setup instructions above)
```

**Memory Issues**
```yaml
# Edit config/pipeline.yaml
global:
  default_memory_gb: 16    # Increase memory
  num_workers: 4           # Reduce parallel jobs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
