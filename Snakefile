"""
Rosette Analysis Pipeline - Main Snakefile
==========================================

This Snakefile orchestrates the complete rosette analysis pipeline using
a modular architecture that preserves existing Poetry environments while
adding Snakemake workflow orchestration.

Author: Rosette Pipeline Team
Version: 1.0 - Refined Snakemake-native implementation
"""

import yaml
import os
from pathlib import Path
from datetime import datetime
from snakemake.io import glob_wildcards, expand

# =============================================================================
# Configuration Loading (Snakemake-native approach)
# =============================================================================


# Load main pipeline configuration
configfile: "config/pipeline.yaml"


# Dynamically load and resolve module configs using Snakemake's native capabilities
for module_name, settings in config["modules"].items():
    if settings.get("enabled", False):
        config_file_path = settings["config_file"]

        # Check if config file exists
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Module config file not found: {config_file_path}")

        with open(config_file_path, "r") as f:
            module_config_str = f.read()

        # Use Snakemake's built-in string formatting to resolve global variables
        try:
            template_vars = {"global": config["global"]}
            resolved_config_str = module_config_str.format(**template_vars)
            config["modules"][module_name]["config"] = yaml.safe_load(
                resolved_config_str
            )
        except KeyError as e:
            raise ValueError(
                f"Template variable not found in module {module_name}: {e}"
            )

# =============================================================================
# Configuration Validation (integrated approach)
# =============================================================================


def validate_config(config):
    """Validate the loaded configuration."""
    # Validate global configuration
    required_global_keys = ["base_data_dir", "output_base", "num_workers", "log_level"]
    for key in required_global_keys:
        if key not in config.get("global", {}):
            raise ValueError(f"Missing required global config key: {key}")

    # Validate module configurations
    for module_name, settings in config["modules"].items():
        if settings.get("enabled", False):
            if "config_file" not in settings:
                raise ValueError(f"Module {module_name} missing config_file")
            if "config" not in settings:
                raise ValueError(f"Module {module_name} config failed to load")

            module_config = settings["config"]
            if "environment" not in module_config:
                raise ValueError(
                    f"Module {module_name} missing environment configuration"
                )
            if "poetry_project" not in module_config["environment"]:
                raise ValueError(f"Module {module_name} missing poetry_project")


# Call validation after config loading
validate_config(config)

# =============================================================================
# Module Flag Collection
# =============================================================================


def get_enabled_module_flags():
    """Get flag files from enabled modules."""
    flags = []

    for module_name, settings in config["modules"].items():
        if settings.get("enabled", False):
            flags.append(f"results/{module_name}.flag")

    return flags


# =============================================================================
# Dynamic Sample Discovery (Module-specific)
# =============================================================================
# Sample discovery is now handled within each module's rules file
# This allows each module to discover samples from its specific input requirements

# =============================================================================
# Include Module Rules
# =============================================================================
# Set up the module rules here

if config["modules"]["image_segmentation"]["enabled"]:

    include: "modules/image_segmentation/rules.smk"


if config["modules"]["segmentation_post_processing"]["enabled"]:

    include: "modules/segmentation_post_processing/rules.smk"


if config["modules"]["training_data_preparation"]["enabled"]:

    include: "modules/training_data_preparation/rules.smk"


if config["modules"]["model_training"]["enabled"]:

    include: "modules/model_training/rules.smk"


if config["modules"]["inference_rosettes"]["enabled"]:

    include: "modules/inference_rosettes/rules.smk"


if config["modules"]["rosette_quantification"]["enabled"]:
    # Conditionally include the TIME SERIES workflow
    if config["modules"]["rosette_quantification"].get(
        "run_timeseries_analysis", False
    ):

        include: "modules/rosette_quantification/rules/timeseries.smk"

    # Conditionally include the INHIBITOR workflow
    if config["modules"]["rosette_quantification"].get(
        "run_inhibitor_analysis", False
    ):

        include: "modules/rosette_quantification/rules/inhibitor.smk"

    if config["modules"]["rosette_quantification"].get(
            "run_constant_fed_analysis", False
        ):
            include: "modules/rosette_quantification/rules/constant_fed.smk"

if config["modules"]["downstream_analysis"]["enabled"]:
    # Conditionally include the TIME SERIES workflow
    if config["modules"]["downstream_analysis"].get("run_timeseries_analysis", False):

        include: "modules/downstream_analysis/rules/timeseries.smk"

    # Conditionally include the INHIBITOR workflow
    if config["modules"]["downstream_analysis"].get(
        "run_inhibitor_analysis", False
    ):

        include: "modules/downstream_analysis/rules/inhibitor.smk"

    if config["modules"]["downstream_analysis"].get("run_constant_fed_analysis", False):
            include: "modules/downstream_analysis/rules/constant_fed.smk"
if config["modules"].get("reporting", {}).get("enabled", False):

    include: "modules/reporting/rules.smk"


# # --- Corrected Downstream Analysis Includes ---
# if config["modules"]["downstream_analysis"]["enabled"]:
#     # Conditionally include the TIME SERIES analysis
#     if config["modules"]["downstream_analysis"].get("run_timeseries_analysis", False):

#         include: "modules/downstream_analysis/rules/timeseries.smk"


# We will add inhibitor analysis rules here later
# if config["modules"]["downstream_analysis"].get("run_inhibitor_analysis", False):
#     include: "modules/downstream_analysis/rules/inhibitor.smk"

# =============================================================================
# Main Pipeline Rules
# =============================================================================


# Main target rule
rule all:
    input:
        "results/pipeline_complete.flag",


# Pipeline completion rule
rule pipeline_complete:
    input:
        # Just depend on module completion flags - much cleaner!
        get_enabled_module_flags(),
    output:
        "results/pipeline_complete.flag",
    run:
        enabled_modules = [
            m for m, s in config["modules"].items() if s.get("enabled", False)
        ]

        with open(output[0], "w") as f:
            f.write(f"Pipeline execution completed successfully at {datetime.now()}\n")
            f.write(f"Enabled modules: {', '.join(enabled_modules)}\n")
            f.write(f"Completed modules: {len(input)}\n")
            f.write("--- Module Completion Status ---\n")
            for flag_file in sorted(input):
                f.write(f"✅ {flag_file}\n")


# =============================================================================
# Utility Rules
# =============================================================================


# Configuration validation rule
rule validate_config:
    output:
        "results/config_validation.txt",
    run:
        validation_results = []
        validation_results.append(
            f"Configuration validation completed at {datetime.now()}"
        )
        validation_results.append(
            f"Global config keys: {list(config['global'].keys())}"
        )
        validation_results.append(
            f"Enabled modules: {[m for m, s in config['modules'].items() if s.get('enabled', False)]}"
        )

        for module_name, settings in config["modules"].items():
            if settings.get("enabled", False):
                module_config = settings["config"]
                validation_results.append(f"\\nModule: {module_name}")
                validation_results.append(
                    f"  Input dir: {module_config.get('inputs', {}).get('input_dir', 'N/A')}"
                )
                validation_results.append(
                    f"  Output dir: {module_config.get('outputs', {}).get('output_dir', 'N/A')}"
                )
                validation_results.append(
                    f"  Poetry project: {module_config.get('environment', {}).get('poetry_project', 'N/A')}"
                )

        with open(output[0], "w") as f:
            f.write("\\n".join(validation_results))
rule show_config_glob:
    run:
        print("\n" + "=" * 80)
        print("🧪 ROSETTE ANALYSIS PIPELINE - STATUS DASHBOARD")
        print("=" * 80)

        # --- 1. Global Configuration ---
        print("\n📂 CONFIGURATION PATHS")
        print(f"  • Raw Metadata TSV:     {config.get('sample_metadata_file', 'Not Set')}")
        print(f"  • Curated Metadata TSV: {config.get('curated_metadata_file', 'Not Set')}")
        print(f"  • Output Directory:     {config['global'].get('output_base')}")
        print(f"  • Curated Data Root:    {config['global'].get('human_curated_rosette_data_root')}")

        # --- 2. Module Status & Data Discovery ---
        print("\n📊 DATA DISCOVERY")
        print(f"  {'MODULE':<25} | {'STATUS':<10} | {'SAMPLES':<10}")
        print("  " + "-" * 50)

        def report_status(label, var_name, config_key):
            """Helper to print status of a module based on config and loaded variables."""
            # check config enabled status
            is_enabled = config["modules"].get(config_key, {}).get("enabled", False)
            
            status_str = "✅ ON" if is_enabled else "⚪ OFF"
            
            count_str = "---"
            if is_enabled:
                # Check if the list variable exists in the global namespace (loaded by include:)
                if var_name in globals():
                    count = len(globals()[var_name])
                    count_str = f"{count}"
                else:
                    count_str = "Err"

            print(f"  {label:<25} | {status_str:<10} | {count_str:<10}")

        # Report on the main pipeline steps
        report_status("Image Segmentation", "IMAGE_SAMPLES", "image_segmentation")
        report_status("Post-Processing", "SEGMENTATION_SAMPLES", "segmentation_post_processing")
        report_status("Inference", "INFERENCE_SAMPLES", "inference_rosettes")
        
        print("\n  [Quantification Workflows]")
        # These share the 'rosette_quantification' config key but have separate lists
        
        # Time Series
        ts_count = len(globals().get("TIMESERIES_SAMPLES", [])) if "TIMESERIES_SAMPLES" in globals() else "-"
        ts_enabled = config["modules"]["rosette_quantification"].get("run_timeseries_analysis", False)
        print(f"  {'Time Series':<25} | {'✅ ON' if ts_enabled else '⚪ OFF':<10} | {ts_count:<10}")

        # Inhibitors
        inh_count = len(globals().get("INHIBITOR_SAMPLES", [])) if "INHIBITOR_SAMPLES" in globals() else "-"
        inh_enabled = config["modules"]["rosette_quantification"].get("run_inhibitor_analysis", False)
        print(f"  {'Inhibitors':<25} | {'✅ ON' if inh_enabled else '⚪ OFF':<10} | {inh_count:<10}")

        # Constant Fed
        cf_count = len(globals().get("CF_SAMPLES", [])) if "CF_SAMPLES" in globals() else "-"
        cf_enabled = config["modules"]["rosette_quantification"].get("run_constant_fed_analysis", False)
        print(f"  {'Constant Fed':<25} | {'✅ ON' if cf_enabled else '⚪ OFF':<10} | {cf_count:<10}")

        print("\n" + "=" * 80 + "\n")

# Clean intermediate files
rule clean:
    shell:
        """
        rm -rf results/intermediate/
        rm -f results/pipeline_complete.flag
        echo "Cleaned intermediate files"
        """


# Deep clean (removes all results)
rule clean_all:
    shell:
        """
        rm -rf results/
        rm -rf logs/
        echo "Cleaned all results and logs"
        """


# =============================================================================
# Development and Debugging Rules
# =============================================================================


# Show resolved configuration
rule show_config:
    run:
        print("=== PIPELINE CONFIGURATION ===")
        print(f"Global config: {config['global']}")
        print(
            f"Enabled modules: {[m for m, s in config['modules'].items() if s.get('enabled', False)]}"
        )
        print(f"Available samples: {SAMPLES}")

        for module_name, settings in config["modules"].items():
            if settings.get("enabled", False):
                print(f"\\n=== MODULE: {module_name} ===")
                module_config = settings["config"]
                print(f"Input dir: {module_config.get('inputs', {}).get('input_dir')}")
                print(
                    f"Output dir: {module_config.get('outputs', {}).get('output_dir')}"
                )
                print(
                    f"Poetry project: {module_config.get('environment', {}).get('poetry_project')}"
                )
