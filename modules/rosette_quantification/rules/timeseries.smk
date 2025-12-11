# Snakemake Rules for Time Series Quantification (Namespaced)
# ===========================================================
import pandas as pd
import os
from snakemake.io import touch, expand

MODULE_NAME = "rosette_quantification"

# =============================================================================
# 1. LOAD METADATA
# =============================================================================
try:
    CURATED_METADATA_PATH = config.get("curated_metadata_file")
    curated_df = pd.read_csv(CURATED_METADATA_PATH, sep="\t")
    
    # Clean strings
    df_obj = curated_df.select_dtypes(['object'])
    curated_df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    
    # Filter for Timeseries
    curated_df["experiment_type_lower"] = curated_df["experiment_type"].str.lower()
    ts_df = curated_df[curated_df["experiment_type_lower"] == "timeseries"]
    
    # 1. List of samples
    TIMESERIES_SAMPLES = ts_df["unique_sample_id"].unique().tolist()
    
    # 2. Direct Maps (Unique Variable Names to avoid collision)
    TS_ROSETTE_PATH_MAP = ts_df.set_index("unique_sample_id")["curated_rosette_path"].to_dict()
    TS_BASE_SEG_PATH_MAP = ts_df.set_index("unique_sample_id")["base_segmentation_path"].to_dict()
    TS_METADATA_MAP = ts_df.set_index("unique_sample_id").to_dict("index")
    
    print(f"Rosette Quantification (TS): Loaded {len(TIMESERIES_SAMPLES)} samples.")

except Exception as e:
    print(f"Error loading metadata (TS): {e}")
    TIMESERIES_SAMPLES = []
    TS_ROSETTE_PATH_MAP = {}
    TS_BASE_SEG_PATH_MAP = {}
    TS_METADATA_MAP = {}

# =============================================================================
# 2. INPUT FUNCTIONS
# =============================================================================

def get_ts_rosette_input(wildcards):
    """Gets the path to the CURATED rosette annotation mask."""
    target = wildcards.sample
    
    # Use the TS-specific map
    if target in TS_ROSETTE_PATH_MAP:
        path = TS_ROSETTE_PATH_MAP[target]
        if path.startswith("/"):
            return path
        return os.path.join(config["global"]["human_curated_rosette_data_root"], path)
    
    raise ValueError(f"Sample '{target}' not found in Timeseries Metadata.")

def get_ts_base_seg_input(wildcards):
    """Gets the path to the ORIGINAL base segmentation mask."""
    import pandas as pd  # <-- ADD THIS IMPORT
    
    target = wildcards.sample
    if target in TS_BASE_SEG_PATH_MAP:
        path = TS_BASE_SEG_PATH_MAP[target]
        
        # --- THIS IS THE FIX ---
        # Check for both the string "NA" AND the float NaN value
        if pd.isna(path) or path == "NA":
        # --- END FIX ---
             raise ValueError(f"Sample '{target}' has no 'base_segmentation_path' in TSV.")
        
        if path.startswith("/"):
            return path
        # Use the global root to build the absolute path
        return os.path.join(config["global"]["human_curated_rosette_data_root"], path)
    
    raise ValueError(f"Sample '{target}' not found in Timeseries Metadata.")
def get_ts_meta(wildcards, key):
    return str(TS_METADATA_MAP.get(wildcards.sample, {}).get(key, "Unknown"))

# =============================================================================
# 3. RULES
# =============================================================================

rule timeseries_quantification_all:
    input:
        f"results/{MODULE_NAME}/combined_timeseries_cell_properties.csv",
        f"results/{MODULE_NAME}/animal_aggregated_timeseries_cell_properties.csv"
    output:
        touch(f"results/rosette_quantification_timeseries.flag")

rule quantify_timeseries_sample:
    input:
        # --- THIS IS THE KEY CHANGE ---
        # It no longer looks at 'segmentation_post_processing'.
        # It now reads the 'base_segmentation_path' from your TSV.
        segmentation=get_ts_base_seg_input,
        rosettes=get_ts_rosette_input
    output:
        cell_properties=f"results/{MODULE_NAME}/timeseries/{{sample}}/cell_properties_{{sample}}.csv",
        summary=f"results/{MODULE_NAME}/timeseries/{{sample}}/quantification_{{sample}}.csv"
    params:
        output_dir=lambda wc: f"results/{MODULE_NAME}/timeseries/{wc.sample}",
        sample_name="{sample}",
        replicate=lambda wc: get_ts_meta(wc, "replicate"),
        timepoint=lambda wc: get_ts_meta(wc, "timepoint"),
        animal=lambda wc: get_ts_meta(wc, "animal_id"),
        image=lambda wc: get_ts_meta(wc, "image_id"),
        inhibitor="None",
        is_control="False",
        min_rosette_size=config["modules"][MODULE_NAME]["config"]["quantification"]["min_rosette_size"],
        max_rosette_size=config["modules"][MODULE_NAME]["config"]["quantification"]["max_rosette_size"],
        normalization_factor=config["modules"][MODULE_NAME]["config"]["quantification"]["normalization_factor"]
    log:
        f"logs/{MODULE_NAME}/timeseries_{{sample}}.log"
    shell:
        """
        poetry run python modules/rosette_quantification/scripts/quantify_rosettes.py \
            --segmentation "{input.segmentation}" \
            --rosettes "{input.rosettes}" \
            --output "{params.output_dir}" \
            --sample "{params.sample_name}" \
            --replicate "{params.replicate}" \
            --timepoint "{params.timepoint}" \
            --animal "{params.animal}" \
            --image "{params.image}" \
            --inhibitor "{params.inhibitor}" \
            --is-control "{params.is_control}" \
            --min-rosette-size {params.min_rosette_size} \
            --max-rosette-size {params.max_rosette_size} \
            --normalization-factor {params.normalization_factor} > {log} 2>&1
        """

rule concatenate_timeseries_results:
    input:
        csv_files=expand(f"results/{MODULE_NAME}/timeseries/{{sample}}/cell_properties_{{sample}}.csv", sample=TIMESERIES_SAMPLES)
    output:
        combined=f"results/{MODULE_NAME}/combined_timeseries_cell_properties.csv",
        aggregated=f"results/{MODULE_NAME}/animal_aggregated_timeseries_cell_properties.csv"
    log:
        f"logs/{MODULE_NAME}/timeseries_concatenate.log"
    run:
        from tempfile import NamedTemporaryFile
        import subprocess
        
        if not input.csv_files:
            from pathlib import Path
            Path(output.combined).touch()
            Path(output.aggregated).touch()
        else:
            with NamedTemporaryFile(mode="w", delete=False) as tmp:
                for f in input.csv_files:
                    if os.path.exists(f):
                        tmp.write(os.path.abspath(f) + "\n")
                input_list_file = tmp.name

            command = [
                "poetry", "run", "python",
                "modules/rosette_quantification/scripts/concatenate_results.py",
                "--input-file-list", input_list_file,
                "--output", os.path.abspath(str(output.combined)),
                "--output-aggregated", os.path.abspath(str(output.aggregated))
            ]
            
            with open(log[0], "w") as logfile:
                subprocess.run(command, check=True, stdout=logfile, stderr=subprocess.STDOUT)
            os.remove(input_list_file)