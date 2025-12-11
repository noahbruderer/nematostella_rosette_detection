import os
import json
import pandas as pd
from snakemake.io import glob_wildcards

# Get the input directory from the config for building paths
MODULE_NAME = "inference_rosettes"

# Input directories
SEGMENTATION_MASKS_DIR = config["global"]["output_base"] + "/results/segmentation_post_processing"
MODELS_DIR = config["modules"][MODULE_NAME]["config"]["inputs"]["models_dir"]

# =============================================================================
# METADATA-DRIVEN SAMPLE DISCOVERY
# =============================================================================
# This ensures we attempt to run inference on ALL samples defined in your metadata,
# which triggers the upstream post-processing rules to generate the missing files.
try:
    METADATA_PATH = config["sample_metadata_file"]
    metadata_df = pd.read_csv(METADATA_PATH, sep="\t")
    # Use the unique_sample_id as the wildcard
    INFERENCE_SAMPLES = sorted(metadata_df["unique_sample_id"].unique().tolist())
    print(f"Inference rosettes: Scheduled {len(INFERENCE_SAMPLES)} samples from metadata for processing.")
except Exception as e:
    print(f"Inference rosettes warning: Metadata load failed ({e}). Falling back to disk scan (reactive mode).")
    # Fallback: only process files that already exist on disk
    INFERENCE_SAMPLES, = glob_wildcards(os.path.join(SEGMENTATION_MASKS_DIR, "{sample}/{sample}_masks_curated.tif"))
    INFERENCE_SAMPLES = sorted(INFERENCE_SAMPLES)

# Get model selection from config
MODEL_EXPERIMENT = config["modules"][MODULE_NAME]["config"]["model_selection"]["experiment_name"]

# Helper function to load model config
def get_model_config_path():
    """Get path to the trained model's config file."""
    # Use the model specified in config (champion_baseline)
    model_name = config["modules"][MODULE_NAME]["config"]["model_selection"]["experiment_name"]
    return config["global"]["output_base"] + f"/results/model_training/{model_name}/config.json"

def get_model_weights_path():
    """Get path to the trained model's weights file."""
    # Use the model specified in config (champion_baseline)
    model_name = config["modules"][MODULE_NAME]["config"]["model_selection"]["experiment_name"]
    return config["global"]["output_base"] + f"/results/model_training/{model_name}/best_model.pth"

# Module-specific rule all
rule inference_rosettes_all:
    input:
        # Generate predictions for all samples found in metadata
        expand(f"results/{MODULE_NAME}/{{sample}}/{{sample}}_pred.tif", sample=INFERENCE_SAMPLES),
        # Summary metrics
        f"results/{MODULE_NAME}/inference_metrics.csv",
        # Human in the loop verification files
        f"results/{MODULE_NAME}/napari_verification_files.flag"
    output:
        f"results/{MODULE_NAME}.flag"
    params:
        num_samples=len(INFERENCE_SAMPLES),
        model_experiment=MODEL_EXPERIMENT
    shell:
        """
        echo "Inference rosettes module completed successfully at $(date)" > {output}
        echo "Processed {params.num_samples} samples using model '{params.model_experiment}'" >> {output}
        echo "Generated files:" >> {output}
        for f in {input}; do echo "  $f" >> {output}; done
        """

rule inference_rosettes_predict:
    input:
        # CRITICAL FIX: Point to the output of the post-processing module.
        # This forces Snakemake to run the curation rule if this file doesn't exist.
        curated_mask="results/segmentation_post_processing/{sample}/{sample}_masks_curated.tif",
        
        # Trained model files
        model_weights=lambda wildcards: get_model_weights_path(),
        model_config=lambda wildcards: get_model_config_path()
    output:
        # Output predictions from geometric_inference.py
        predictions=f"results/{MODULE_NAME}/{{sample}}/{{sample}}_pred.tif"
    log:
        f"logs/{MODULE_NAME}/{{sample}}.log"
    params:
        config_file=os.path.abspath(config["modules"][MODULE_NAME]["config_file"]),
        output_dir=lambda wildcards: os.path.abspath(f"results/inference_rosettes/{wildcards.sample}"),
        sample_name="{sample}",
        model_experiment=MODEL_EXPERIMENT,
        # Inference parameters from config
        batch_size=config["modules"][MODULE_NAME]["config"]["inference"]["batch_size"],
        confidence_threshold=config["modules"][MODULE_NAME]["config"]["inference"]["confidence_threshold"],
        patch_size=config["modules"][MODULE_NAME]["config"]["inference"]["patch_size"],
        overlap=config["modules"][MODULE_NAME]["config"]["inference"]["overlap"]
    threads: config["global"]["num_workers"]
    resources:
        mem_mb=lambda wildcards, attempt: config["global"]["default_memory_gb"] * 1024 * attempt,
        runtime=config["modules"][MODULE_NAME]["config"]["resources"]["time_limit_hours"] * 60
    shell:
        """
        echo "Starting inference for sample: {wildcards.sample}"
        echo "Using model from experiment: {params.model_experiment}"
        
        # Ensure log directory exists
        mkdir -p $(dirname {workflow.basedir}/{log})
        
        # Ensure output directory exists
        mkdir -p {params.output_dir}
        
        # Create symlink with expected naming convention for geometric_inference.py
        TEMP_DIR=$(mktemp -d)
        # Note: input.curated_mask is now explicitly defined in the input section above
        ln -s "$(realpath {input.curated_mask})" "$TEMP_DIR/{wildcards.sample}_seg_cur.tif"
        
        # Change to inference module directory and run geometric inference
        cd {workflow.basedir}/modules/inference_rosettes && \\
        poetry install --no-root && \\
        poetry run python scripts/geometric_inference.py \\
            "{input.model_weights}" \\
            "$TEMP_DIR" \\
            --output_dir "{params.output_dir}" \\
            2>&1 | tee {workflow.basedir}/{log}
        
        # Clean up temporary directory
        rm -rf "$TEMP_DIR"
        """

rule create_inference_summary:
    input:
        # Collect all prediction files
        predictions=expand(f"results/{MODULE_NAME}/{{sample}}/{{sample}}_pred.tif", sample=INFERENCE_SAMPLES)
    output:
        summary=f"results/{MODULE_NAME}/inference_metrics.csv"
    params:
        model_experiment=MODEL_EXPERIMENT
    run:
        import pandas as pd
        import numpy as np
        
        # Create summary metrics from all predictions
        summary_data = []
        for pred_file in input.predictions:
            sample_name = pred_file.split('/')[-2]  # Extract sample name from path
            
            # For now, create placeholder metrics
            summary_data.append({
                'sample': sample_name,
                'model_experiment': params.model_experiment,
                'prediction_file': pred_file,
                'inference_completed': True
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output.summary, index=False)
        
        print(f"Inference summary saved to {output.summary}")
        print(f"Completed inference for {len(summary_data)} samples using model '{params.model_experiment}'")

rule create_napari_verification_files:
    input:
        # Wait for all predictions to complete
        predictions=expand(f"results/{MODULE_NAME}/{{sample}}/{{sample}}_pred.tif", sample=INFERENCE_SAMPLES),
        # Original curated masks (using the correct path structure now)
        segmentation_masks=expand(f"results/segmentation_post_processing/{{sample}}/{{sample}}_masks_curated.tif", sample=INFERENCE_SAMPLES)
    output:
        flag=f"results/{MODULE_NAME}/napari_verification_files.flag"
    params:
        output_dir=f"results/{MODULE_NAME}/napari_verification",
        num_samples=len(INFERENCE_SAMPLES)
    log:
        f"logs/{MODULE_NAME}/napari_verification.log"
    shell:
        """
        echo "Creating Napari verification files for {params.num_samples} samples..."
        
        # Ensure output directory exists
        mkdir -p {params.output_dir}
        # FIX: Ensure the directory for the output flag file also exists
        mkdir -p $(dirname {output.flag})
        
        # Change to inference module directory and run human in the loop script
        cd {workflow.basedir}/modules/inference_rosettes && \\
        poetry install --no-root && \\
        poetry run python -c "
import sys
sys.path.append('scripts')
from building_human_in_the_loop_files import create_napari_structure_flexible
from pathlib import Path

# Create organized structure from inference results
source_dir = Path('{workflow.basedir}/results/{MODULE_NAME}')
output_dir = Path('{workflow.basedir}/{params.output_dir}')

print(f'Organizing files from {{source_dir}} to {{output_dir}}')
create_napari_structure_flexible(str(source_dir), str(output_dir))
print('✅ Napari verification files created successfully!')
" 2>&1 | tee {workflow.basedir}/{log}
        
        echo "Napari verification files created at $(date)" > {output.flag}
        echo "Organized {params.num_samples} samples for verification" >> {output.flag}
        echo "Files available in: {params.output_dir}" >> {output.flag}
        """