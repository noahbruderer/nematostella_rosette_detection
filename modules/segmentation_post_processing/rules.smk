import os
from snakemake.io import glob_wildcards

# Get the input directory from the config for building paths
MODULE_NAME = "segmentation_post_processing"

# Input directory for segmentation masks from image_segmentation module
SEGMENTATION_MASKS_DIR = config["modules"][MODULE_NAME]["config"]["inputs"]["segmentation_masks_dir"]
MASK_FILE_PATTERN = config["modules"][MODULE_NAME]["config"]["inputs"]["mask_file_pattern"]

# Discover available segmentation masks for post-processing
def discover_segmentation_masks():
    """Discover all segmentation mask files from the image_segmentation module."""
    mask_files = []
    sample_paths = {}  # Maps sample name to full mask file path
    
    # Search for mask files in the segmentation output directory
    pattern = os.path.join(SEGMENTATION_MASKS_DIR, "**", MASK_FILE_PATTERN)
    try:
        import glob
        files = glob.glob(pattern, recursive=True)
        mask_files.extend(files)
    except:
        # Fallback if recursive search fails
        pattern = os.path.join(SEGMENTATION_MASKS_DIR, "*", MASK_FILE_PATTERN)
        files = glob.glob(pattern)
        mask_files.extend(files)
    
    # Extract sample names and store full paths
    samples = []
    for file_path in mask_files:
        if os.path.exists(file_path):  # Verify file exists
            # Extract sample name from directory structure
            # Expected structure: results/image_segmentation/SAMPLE_NAME/SAMPLE_NAME_masks.tif
            parent_dir = os.path.dirname(file_path)
            sample_name = os.path.basename(parent_dir)
            if sample_name:
                samples.append(sample_name)
                sample_paths[sample_name] = os.path.abspath(file_path)
    
    return sorted(list(set(samples))), sample_paths  # Remove duplicates and sort

# Discover samples
SEGMENTATION_SAMPLES, SAMPLE_MASK_PATHS = discover_segmentation_masks()
print(f"Segmentation post-processing: Discovered {len(SEGMENTATION_SAMPLES)} segmentation masks: {SEGMENTATION_SAMPLES[:5]}{'...' if len(SEGMENTATION_SAMPLES) > 5 else ''}")

# Helper function to find the actual mask file for a sample
def get_mask_file_for_sample(sample_name):
    """Find the actual mask file path for a given sample name."""
    if sample_name in SAMPLE_MASK_PATHS:
        return SAMPLE_MASK_PATHS[sample_name]
    else:
        raise FileNotFoundError(f"No mask file found for sample {sample_name}")

# Module-specific rule all
rule segmentation_post_processing_all:
    input:
        # Generate curated masks for all samples
        expand(f"results/{MODULE_NAME}/{{sample}}/{{sample}}_masks_curated.tif", sample=SEGMENTATION_SAMPLES),
        expand(f"results/{MODULE_NAME}/{{sample}}/{{sample}}_curation_stats.csv", sample=SEGMENTATION_SAMPLES),
        # HTML visualizations created by the curator
        expand(f"results/{MODULE_NAME}/{{sample}}/raw_mask_vis_{{sample}}.html", sample=SEGMENTATION_SAMPLES),
        expand(f"results/{MODULE_NAME}/{{sample}}/processed_mask_vis_{{sample}}.html", sample=SEGMENTATION_SAMPLES),
        expand(f"results/{MODULE_NAME}/{{sample}}/cell_area_distribution_comparison_{{sample}}.html", sample=SEGMENTATION_SAMPLES),
        # QC images
        expand(f"results/{MODULE_NAME}/{{sample}}/{{sample}}_curation_qc.png", sample=SEGMENTATION_SAMPLES),
        # Overview visualization
        # f"results/{MODULE_NAME}/curation_overview.png"
    output:
        f"results/{MODULE_NAME}.flag"
    params:
        num_samples=len(SEGMENTATION_SAMPLES)
    shell:
        """
        echo "Segmentation post-processing module completed successfully at $(date)" > {output}
        echo "Processed {params.num_samples} segmentation samples" >> {output}
        echo "Generated files:" >> {output}
        for f in {input}; do echo "  $f" >> {output}; done
        """

rule segmentation_post_processing_curate:
    input:
        # Dynamic input - find the actual mask file for this sample
        mask_file=lambda wildcards: get_mask_file_for_sample(wildcards.sample)
    output:
        # Output curation results
        curated_masks=f"results/{MODULE_NAME}/{{sample}}/{{sample}}_masks_curated.tif",
        statistics=f"results/{MODULE_NAME}/{{sample}}/{{sample}}_curation_stats.csv",
        raw_vis=f"results/{MODULE_NAME}/{{sample}}/raw_mask_vis_{{sample}}.html",
        processed_vis=f"results/{MODULE_NAME}/{{sample}}/processed_mask_vis_{{sample}}.html",
        comparison_vis=f"results/{MODULE_NAME}/{{sample}}/cell_area_distribution_comparison_{{sample}}.html",
        qc_image=f"results/{MODULE_NAME}/{{sample}}/{{sample}}_curation_qc.png"
    log:
        f"logs/{MODULE_NAME}/{{sample}}.log"
    params:
        config_file=os.path.abspath(config["modules"][MODULE_NAME]["config_file"]),
        output_dir=lambda wildcards: os.path.abspath(f"results/segmentation_post_processing/{wildcards.sample}"),
        module_name=MODULE_NAME,
        sample_name="{sample}",
        # Curation parameters from config
        min_cell_size=config["modules"][MODULE_NAME]["config"]["curation"]["min_cell_size"],
        max_small_cell_size=config["modules"][MODULE_NAME]["config"]["curation"]["max_small_cell_size"],
        verbose="true" if config["modules"][MODULE_NAME]["config"]["curation"]["verbose"] else "false",
        create_visualizations="true" if config["modules"][MODULE_NAME]["config"]["curation"]["create_visualizations"] else "false",
        save_statistics="true" if config["modules"][MODULE_NAME]["config"]["curation"]["save_statistics"] else "false",
        save_qc_images="true" if config["modules"][MODULE_NAME]["config"]["curation"]["save_qc_images"] else "false"
    threads: config["global"]["num_workers"]
    resources:
        mem_mb=lambda wildcards, attempt: config["global"]["default_memory_gb"] * 1024 * attempt,
        runtime=config["modules"][MODULE_NAME]["config"]["resources"]["time_limit_hours"] * 60
    shell:
        """
        echo "Starting segmentation curation for sample: {wildcards.sample}"
        echo "Input mask: {input.mask_file}"
        echo "Min cell size: {params.min_cell_size}"
        
        # Ensure log directory exists
        mkdir -p $(dirname {workflow.basedir}/{log})
        
        # Ensure output directory exists
        mkdir -p {params.output_dir}
        
        # Change to post-processing module directory and run curation
        cd {workflow.basedir}/modules/{params.module_name} && \\
        poetry run python scripts/curate_segmentation.py \\
            --input-mask "{input.mask_file}" \\
            --output-dir "{params.output_dir}" \\
            --sample-name "{params.sample_name}" \\
            --min-cell-size {params.min_cell_size} \\
            --max-small-cell-size {params.max_small_cell_size} \\
            --verbose {params.verbose} \\
            --create-visualizations {params.create_visualizations} \\
            --save-statistics {params.save_statistics} \\
            --save-qc-images {params.save_qc_images} \\
            2>&1 | tee {workflow.basedir}/{log}
        """

# rule create_curation_overview:
#     input:
#         # Collect all QC images
#         qc_images=expand(f"results/{MODULE_NAME}/{{sample}}/{{sample}}_curation_qc.png", sample=SEGMENTATION_SAMPLES)
#     output:
#         overview=f"results/{MODULE_NAME}/curation_overview.png"
#     params:
#         samples_per_overview=config["modules"][MODULE_NAME]["config"]["quality_control"]["samples_per_overview"]
#     run:
#         import matplotlib
#         matplotlib.use('Agg')  # Set non-interactive backend
#         import matplotlib.pyplot as plt
#         import matplotlib.image as mpimg
#         import numpy as np
#         from pathlib import Path
        
#         # Load QC images
#         images = []
#         sample_names = []
#         for qc_file in input.qc_images[:params.samples_per_overview]:
#             try:
#                 img = mpimg.imread(qc_file)
#                 images.append(img)
#                 # Extract sample name from path
#                 sample_name = Path(qc_file).parent.name
#                 sample_names.append(sample_name)
#             except Exception as e:
#                 print(f"Warning: Could not load {qc_file}: {e}")
        
#         if len(images) == 0:
#             print("No QC images found, creating placeholder overview")
#             fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#             ax.text(0.5, 0.5, 'No curation results available', 
#                    ha='center', va='center', transform=ax.transAxes, fontsize=16)
#             ax.axis('off')
#         else:
#             # Create overview grid
#             n_images = len(images)
#             cols = min(3, n_images)
#             rows = (n_images + cols - 1) // cols
            
#             fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
#             if rows == 1 and cols == 1:
#                 axes = [axes]
#             elif rows == 1:
#                 axes = [axes]
#             else:
#                 axes = axes.flatten()
            
#             for i, (img, sample) in enumerate(zip(images, sample_names)):
#                 if i < len(axes):
#                     axes[i].imshow(img)
#                     axes[i].set_title(f'{sample}', fontsize=10)
#                     axes[i].axis('off')
            
#             # Hide unused subplots
#             for i in range(len(images), len(axes)):
#                 axes[i].axis('off')
        
#         plt.tight_layout()
#         plt.savefig(output.overview, dpi=150, bbox_inches='tight')
#         plt.close()
        
#         print(f"Curation overview saved to {output.overview}")
#         print(f"Included {len(images)} samples in overview")