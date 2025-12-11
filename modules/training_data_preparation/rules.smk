import os
from snakemake.io import glob_wildcards

# Get the input directory from the config for building paths
MODULE_NAME = "training_data_preparation"

INPUT_DIR = config["modules"][MODULE_NAME]["config"]["inputs"]["input_dir"]
# Discover samples specific to this module
TRAINING_SAMPLES, = glob_wildcards(os.path.join(INPUT_DIR, "{sample}_seg copy.svg"))
print(f"Training data preparation: Discovered {len(TRAINING_SAMPLES)} samples: {TRAINING_SAMPLES[:5]}{'...' if len(TRAINING_SAMPLES) > 5 else ''}")

# Module-specific rule all
rule training_data_preparation_all:
    input:
        # All outputs this module should create
        expand(f"results/{MODULE_NAME}/{{sample}}/processed_data_{{sample}}.h5", sample=TRAINING_SAMPLES),
        expand(f"results/{MODULE_NAME}/{{sample}}/cell_properties_{{sample}}.csv", sample=TRAINING_SAMPLES),
        expand(f"results/{MODULE_NAME}/{{sample}}/neighborhood_graph_{{sample}}.pkl", sample=TRAINING_SAMPLES),
        expand(f"results/{MODULE_NAME}/{{sample}}/summary_{{sample}}.html", sample=TRAINING_SAMPLES)
    output:
        f"results/{MODULE_NAME}.flag"
    params:
        num_samples=len(TRAINING_SAMPLES)
    shell:
        """
        echo "Training data preparation module completed successfully at $(date)" > {output}
        echo "Processed {params.num_samples} samples" >> {output}
        echo "Generated files:" >> {output}
        for f in {input}; do echo "  $f" >> {output}; done
        """

rule training_data_preparation_process:
    input:
        # Use the file structure from your screenshot to define the exact inputs for a sample
        mask=os.path.join(INPUT_DIR, "{sample}_seg copy.svg", "{sample}_seg.tif"),
        rosette=os.path.join(INPUT_DIR, "{sample}_seg copy.svg", "{sample}_seg copy.tif"), # Adjust if name differs
        raw=os.path.join(INPUT_DIR, "{sample}_seg copy.svg", "{sample}.tif")
    output:
        # Define all specific output files with consistent naming: {type}_{sample}
        processed_data=f"results/{MODULE_NAME}/{{sample}}/processed_data_{{sample}}.h5",
        cell_properties=f"results/{MODULE_NAME}/{{sample}}/cell_properties_{{sample}}.csv",
        neighborhood_graph=f"results/{MODULE_NAME}/{{sample}}/neighborhood_graph_{{sample}}.pkl",
        visualization=f"results/{MODULE_NAME}/{{sample}}/summary_{{sample}}.html"
    log:
        f"logs/{MODULE_NAME}/{{sample}}.log"
    params:
        sample_name="{sample}",
        config_file=os.path.abspath(config["modules"][MODULE_NAME]["config_file"]),
        poetry_project=MODULE_NAME, # From your config.yaml
        output_dir=os.path.abspath(f"results/{MODULE_NAME}/{{sample}}"),
        log_file=lambda wildcards: os.path.abspath(f"logs/{MODULE_NAME}/{wildcards.sample}.log")
    threads: 1 # The script itself is single-threaded
    shell:
        """
        cd {workflow.basedir}/modules/training_data_preparation && \
        poetry run python scripts/data_preparation.py \
            --sample-name "{params.sample_name}" \
            --mask-file "{input.mask}" \
            --rosette-file "{input.rosette}" \
            --raw-image-file "{input.raw}" \
            --output-dir "{params.output_dir}" \
            --config "{params.config_file}" \
            --log-file "{params.log_file}"
        """