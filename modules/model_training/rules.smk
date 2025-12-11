import os
import json
from snakemake.io import glob_wildcards

# Get the input directory from the config for building paths
MODULE_NAME = "model_training"

# Input directory where training data preparation outputs are located
TRAINING_DATA_DIR = config["modules"][MODULE_NAME]["config"]["inputs"]["training_data_dir"]

# Discover available training samples for initial discovery (sorted for reproducibility)
def discover_training_samples():
    """Discover all training samples with processed data."""
    try:
        TRAINING_SAMPLES, = glob_wildcards(os.path.join(TRAINING_DATA_DIR, "{sample}/processed_data_{sample}.h5"))
        return sorted(TRAINING_SAMPLES)
    except:
        return []

TRAINING_SAMPLES = discover_training_samples()
print(f"Model training: Discovered {len(TRAINING_SAMPLES)} training samples: {TRAINING_SAMPLES[:5]}{'...' if len(TRAINING_SAMPLES) > 5 else ''}")

# Get experiment names for expand() - this is safe because it only needs the list
def get_experiment_names():
    """Get list of experiment names. Called during rule execution when config is loaded."""
    try:
        return [exp['name'] for exp in config["modules"][MODULE_NAME]["config"]["experiments"]]
    except (KeyError, TypeError):
        # Fallback during initial parsing - return empty list, real list will be available during execution
        return []

EXP_NAMES = get_experiment_names()
if EXP_NAMES:
    print(f"Model training: Found {len(EXP_NAMES)} experiments: {EXP_NAMES}")

# Helper functions for getting experiment parameters (now with safe config access)
def get_experiment_param(config, module_name, exp_name, param_name):
    """Get parameter for an experiment, falling back to defaults."""
    # Safe config access since this is called during rule execution
    defaults = config["modules"][module_name]["config"]["default_training"]
    experiment_params = config["modules"][module_name]["config"]["experiments"]
    
    # Find the specific experiment
    experiment = next((exp for exp in experiment_params if exp['name'] == exp_name), {})
    
    # Handle nested parameters like augmentation
    if '.' in param_name:
        main_param, sub_param = param_name.split('.', 1)
        if main_param in experiment:
            return experiment[main_param].get(sub_param, defaults.get(main_param, {}).get(sub_param))
        return defaults.get(main_param, {}).get(sub_param)
    
    return experiment.get(param_name, defaults.get(param_name))

# Helper function to get experiment description
def get_experiment_description(exp_name):
    """Get description for an experiment."""
    try:
        experiments = config["modules"][MODULE_NAME]["config"]["experiments"]
        experiment = next((exp for exp in experiments if exp['name'] == exp_name), {})
        return experiment.get('description', 'No description')
    except:
        return 'No description'

# Module-specific rule all - runs ALL experiments
rule model_training_all:
    input:
        # Generate outputs for ALL experiments
        expand(f"results/model_training/{{exp_name}}/best_model.pth", exp_name=EXP_NAMES),
        expand(f"results/model_training/{{exp_name}}/config.json", exp_name=EXP_NAMES),
        expand(f"results/model_training/{{exp_name}}/training_plots.png", exp_name=EXP_NAMES),
        expand(f"results/model_training/{{exp_name}}/training.log", exp_name=EXP_NAMES),
        # Simple expand for the manifests
        expand(f"results/{MODULE_NAME}/split_manifest_{{exp_name}}.json", exp_name=EXP_NAMES),
        # Final experiment comparison report
        f"results/{MODULE_NAME}/experiment_summary.csv"
    output:
        f"results/{MODULE_NAME}.flag"
    params:
        num_experiments=len(EXP_NAMES),
        experiment_names=" ".join(EXP_NAMES)
    shell:
        """
        echo "Model training module completed successfully at $(date)" > {output}
        echo "Completed {params.num_experiments} experiments: {params.experiment_names}" >> {output}
        echo "Generated files:" >> {output}
        for f in {input}; do echo "  $f" >> {output}; done
        """

rule create_train_val_split:
    input:
        # Require all training data to be prepared first
        training_data=expand(f"{TRAINING_DATA_DIR}/{{sample}}/processed_data_{{sample}}.h5", sample=TRAINING_SAMPLES)
    output:
        # Generate unique manifest per experiment
        split_manifest=f"results/{MODULE_NAME}/split_manifest_{{exp_name}}.json"
    log:
        f"logs/{MODULE_NAME}/create_split_{{exp_name}}.log"
    params:
        samples=" ".join(TRAINING_SAMPLES),
        output_dir=os.path.abspath("results/model_training"),
        config_file=os.path.abspath(config["modules"][MODULE_NAME]["config_file"]),
        module_name=MODULE_NAME,
        exp_name="{exp_name}",
        # Default parameters - experiment-specific values will be loaded by the script
        default_val_split=config["modules"][MODULE_NAME]["config"]["default_training"]["val_split"],
        default_seed=config["modules"][MODULE_NAME]["config"]["default_training"]["seed"]
    shell:
        """
        echo "Creating train/validation split for experiment: {wildcards.exp_name}"
        
        # Ensure log directory exists
        mkdir -p $(dirname {workflow.basedir}/{log})
        
        # Ensure output directory exists
        mkdir -p {params.output_dir}
        
        # Change to model_training directory and run split creation
        cd {workflow.basedir}/modules/{params.module_name} && \\
        poetry install --no-root && \\
        poetry run python scripts/create_split.py \\
            --samples {params.samples} \\
            --config-dir {params.output_dir} \\
            --config-file {params.config_file} \\
            --experiment-name {params.exp_name} \\
            --output-name "split_manifest_{wildcards.exp_name}.json" \\
            2>&1 | tee {workflow.basedir}/{log}
        """

rule model_training_train:
    input:
        # Simple, direct dependency on the experiment's manifest
        split_manifest=f"results/{MODULE_NAME}/split_manifest_{{exp_name}}.json"
    output:
        # Define experiment-specific output files using wildcard - each experiment gets its own subdirectory
        best_model=f"results/model_training/{{exp_name}}/best_model.pth",
        config_json=f"results/model_training/{{exp_name}}/config.json",
        training_plots=f"results/model_training/{{exp_name}}/training_plots.png",
        training_log=f"results/model_training/{{exp_name}}/training.log"
    log:
        f"logs/{MODULE_NAME}/{{exp_name}}.log"
    params:
        # Workflow structure parameters only
        config_file=os.path.abspath(config["modules"][MODULE_NAME]["config_file"]),
        training_data_dir=os.path.abspath(TRAINING_DATA_DIR),
        output_dir=lambda wildcards: os.path.abspath(f"results/model_training/{wildcards.exp_name}"),
        split_manifest_abs=lambda wildcards: os.path.abspath(f"results/model_training/split_manifest_{wildcards.exp_name}.json"),
        module_name=MODULE_NAME,
        exp_name="{exp_name}"
    threads: config["global"]["default_threads"]
    resources:
        mem_mb=lambda wildcards, attempt: config["global"]["default_memory_gb"] * 1024 * attempt,
        runtime=720  # 12 hours in minutes
    shell:
        """
        echo "Starting experiment: {wildcards.exp_name}"
        echo "Training data directory: {params.training_data_dir}"
        echo "Output directory: {params.output_dir}"
        
        # Ensure log directory exists
        mkdir -p $(dirname {workflow.basedir}/{log})
        
        # Ensure output directory exists for the experiment
        mkdir -p {params.output_dir}
        
        # Change to model_training directory and run training  
        cd {workflow.basedir}/modules/{params.module_name} && \\
        poetry install --no-root && \\
        poetry run python scripts/train_model.py \\
            --split-manifest "{params.split_manifest_abs}" \\
            --data-dir "{params.training_data_dir}" \\
            --output-dir "{params.output_dir}" \\
            --config "{params.config_file}" \\
            --experiment-name "{params.exp_name}" \\
            --no-timestamp \\
            2>&1 | tee {workflow.basedir}/{log}
        
        # Copy log to experiment results directory for reproducibility
        cp {workflow.basedir}/{log} {params.output_dir}/training.log
        """

rule create_experiment_summary:
    input:
        # Collect all experiment config files
        configs=expand(f"results/{MODULE_NAME}/{{exp_name}}/config.json", exp_name=EXP_NAMES)
    output:
        summary=f"results/{MODULE_NAME}/experiment_summary.csv"
    log:
        f"logs/{MODULE_NAME}/experiment_summary.log"
    params:
        config_files=lambda wildcards, input: " ".join([f'"{f}"' for f in input.configs]),
        pipeline_config=os.path.abspath(config["modules"][MODULE_NAME]["config_file"]),
        module_name=MODULE_NAME,
        num_experiments=len(EXP_NAMES)
    shell:
        """
        echo "Creating experiment summary for {params.num_experiments} experiments"
        
        # Ensure log directory exists
        mkdir -p $(dirname {workflow.basedir}/{log})
        
        # Ensure output directory exists
        mkdir -p $(dirname {output.summary})
        
        # Change to model_training directory and run summary creation
        cd {workflow.basedir}/modules/{params.module_name} && \\
        poetry install --no-root && \\
        poetry run python scripts/create_experiment_summary.py \\
            --config-files {params.config_files} \\
            --output {output.summary} \\
            --pipeline-config {params.pipeline_config} \\
            --module-name {params.module_name} \\
            2>&1 | tee {workflow.basedir}/{log}
        """