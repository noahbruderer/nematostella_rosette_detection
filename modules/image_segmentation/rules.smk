import os
import pandas as pd
from snakemake.io import expand

# Get the module name from the config (or set it manually)
MODULE_NAME = "image_segmentation"

# =============================================================================
# METADATA-DRIVEN SAMPLE DISCOVERY (Using unique_sample_id)
# =============================================================================
# This approach reads the master metadata file to get all samples.
# This is much faster and more robust than re-scanning the file system.

try:
    # 1. Load the master metadata file (defined in your main pipeline.yaml)
    METADATA_PATH = config["sample_metadata_file"]
    metadata_df = pd.read_csv(METADATA_PATH, sep="\t")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Missing metadata file: {METADATA_PATH}. "
        "Please run the metadata generation module first."
    )
except KeyError:
    raise KeyError(
        "Could not find 'sample_metadata_file' in your main config. "
        "Please add it to your pipeline.yaml."
    )

# 2. Get the list of all unique sample IDs (this is our wildcard {sample})
#    *** THIS IS THE KEY CHANGE ***
IMAGE_SAMPLES = metadata_df["unique_sample_id"].unique().tolist()
print(
    f"Image segmentation: Discovered {len(IMAGE_SAMPLES)} unique samples from {METADATA_PATH}"
)

# 3. Create a lookup dictionary: {unique_sample_id} -> /full/path/to/raw_image.tif
#    We need the raw data root paths from the main config.
BASE_PATHS = {
    "timeseries": config["timeseries_raw_root"],
    "inhibitor": config["inhibitor_raw_root"],
    "constant_fed": config["constant_fed_raw_root"],
}

# Filter metadata to only include rows with a valid experiment_type
valid_rows = metadata_df[metadata_df["experiment_type"].isin(BASE_PATHS.keys())]

if len(valid_rows) < len(metadata_df):
    print(
        f"Warning: {len(metadata_df) - len(valid_rows)} rows in metadata "
        "have an 'experiment_type' not found in BASE_PATHS. These will be skipped."
    )

# Create the {sample} -> {full_path} map
SAMPLE_PATHS = {}
for _, row in valid_rows.iterrows():
    try:
        base_path = BASE_PATHS[row["experiment_type"]]
        full_path = os.path.abspath(os.path.join(base_path, row["raw_image_path"]))
        
        # *** THIS IS THE SECOND KEY CHANGE ***
        # Use the NEW unique_sample_id as the dictionary key
        SAMPLE_PATHS[row["unique_sample_id"]] = full_path
        
    except Exception as e:
        print(f"Error building path for {row['unique_sample_id']}: {e}")

if not IMAGE_SAMPLES:
    print("Warning: No image samples were found in the metadata file.")

# =============================================================================
# SNAKEMAKE RULES
# =============================================================================

# Module-specific rule all
rule image_segmentation_all:
    input:
        # Use the clean IMAGE_SAMPLES list (now unique_sample_ids)
        expand(
            f"results/{MODULE_NAME}/{{sample}}/{{sample}}_masks.tif",
            sample=IMAGE_SAMPLES,
        ),
        expand(
            f"results/{MODULE_NAME}/{{sample}}/{{sample}}_outlines.png",
            sample=IMAGE_SAMPLES,
        ),
        expand(
            f"results/{MODULE_NAME}/{{sample}}/{{sample}}_stats.csv",
            sample=IMAGE_SAMPLES,
        ),
    output:
        f"results/{MODULE_NAME}.flag",
    params:
        num_samples=len(IMAGE_SAMPLES),
    shell:
        """
        echo "Image segmentation module completed successfully at $(date)" > {output}
        echo "Processed {params.num_samples} image samples" >> {output}
        echo "Generated files:" >> {output}
        for f in {input}; do echo "  $f" >> {output}; done
        """

rule image_segmentation_segment:
    input:
        # DYNAMIC INPUT: Use the fast dictionary lookup
        # The {sample} wildcard will be "R1-T0-A1-I0-PRI", etc.
        raw_image=lambda wildcards: SAMPLE_PATHS[wildcards.sample],
    output:
        # The {sample} wildcard is now the unique_sample_id
        masks=f"results/{MODULE_NAME}/{{sample}}/{{sample}}_masks.tif",
        outlines=f"results/{MODULE_NAME}/{{sample}}/{{sample}}_outlines.png",
        statistics=f"results/{MODULE_NAME}/{{sample}}/{{sample}}_stats.csv",
        qc_image=f"results/{MODULE_NAME}/{{sample}}/{{sample}}_qc.png",
    log:
        f"logs/{MODULE_NAME}/{{sample}}.log",
    params:
        # Get module-specific config file path from main config
        config_file=os.path.abspath(config["modules"][MODULE_NAME]["config_file"]),
        output_dir=lambda wildcards: os.path.abspath(
            f"results/image_segmentation/{wildcards.sample}"
        ),
        module_name=MODULE_NAME,
        sample_name="{sample}",
        # Cellpose parameters from module config
        model_type=config["modules"][MODULE_NAME]["config"]["cellpose"]["model_type"],
        custom_model_path=config["modules"][MODULE_NAME]["config"]["cellpose"].get(
            "custom_model_path", ""
        ),
        diameter=config["modules"][MODULE_NAME]["config"]["cellpose"]["diameter"],
        flow_threshold=config["modules"][MODULE_NAME]["config"]["cellpose"][
            "flow_threshold"
        ],
        cellprob_threshold=config["modules"][MODULE_NAME]["config"]["cellpose"][
            "cellprob_threshold"
        ],
        use_gpu="true"
        if config["modules"][MODULE_NAME]["config"]["cellpose"]["use_gpu"]
        else "false",
        channels=config["modules"][MODULE_NAME]["config"]["cellpose"]["channels"],
        min_size=config["modules"][MODULE_NAME]["config"]["cellpose"]["min_size"],
        save_outlines="true"
        if config["modules"][MODULE_NAME]["config"]["cellpose"]["save_outlines"]
        else "false",
    threads: config["global"]["default_threads"]
    resources:
        # Set memory per job
        mem_mb=lambda wildcards, attempt: 4 * 1024 * attempt,
        runtime=config["modules"][MODULE_NAME]["config"]["resources"][
            "time_limit_hours"
        ]
        * 60,
    shell:
        """
        echo "Starting segmentation for sample: {wildcards.sample}"
        echo "Input image: {input.raw_image}"
        echo "Model type: {params.model_type}"

        # Ensure log directory exists
        mkdir -p $(dirname {workflow.basedir}/{log})

        # Ensure output directory exists
        mkdir -p {params.output_dir}

        # Change to segmentation module directory and run segmentation
        # This assumes your main Snakefile is in the root, one level above 'modules/'
        cd {workflow.basedir}/modules/{params.module_name} && \
        poetry run python scripts/cellpose3.py \
            --input-image "{input.raw_image}" \
            --output-dir "{params.output_dir}" \
            --sample-name "{params.sample_name}" \
            --model-type "{params.model_type}" \
            --custom-model-path "{params.custom_model_path}" \
            --diameter {params.diameter} \
            --flow-threshold {params.flow_threshold} \
            --cellprob-threshold {params.cellprob_threshold} \
            --channels {params.channels[0]} {params.channels[1]} \
            --min-size {params.min_size} \
            --use-gpu {params.use_gpu} \
            --save-outlines {params.save_outlines} \
            --save-masks true \
            --save-statistics true \
            --save-qc true \
            2>&1 | tee {workflow.basedir}/{log}
        """