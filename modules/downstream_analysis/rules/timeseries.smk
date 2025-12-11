# Snakemake Rules for Time Series Downstream Analysis (Refactored)
# =============================================================

from snakemake.io import touch

ANALYSIS_CONFIG = config["modules"]["downstream_analysis"]["config"]


# --- 1. Main Target Rule for the ENTIRE Time Series Analysis ---
# Renamed for clarity and consistency.
rule timeseries_donwstream_analysis_all:
    input:
        # Variance analysis
        lmm_plot=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/variance_summary/lmm_variance_components_plot.png",
        lmm_summary=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/variance_summary/lmm_model_summary.txt",
        # GMM population analysis
        tp_comp=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/advanced_analysis/timepoint_populations/timepoint_population_comparison_results.csv",
        # QC verification plots
        verification_flag=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/verification_plots.flag",
    output:
        touch("results/downstream_analysis_timeseries.flag"),


# --- 2. Create Verification Plots (QC) ---
rule create_timeseries_verification_plots:
    input:
        combined=ANALYSIS_CONFIG["inputs"]["timeseries_combined_data"],
    output:
        # Use a flag here since this is a terminal analysis (no other rules depend on its plots).
        flag=touch(
            f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/verification_plots.flag"
        ),
    params:
        # Standardized directory name for consistency.
        output_dir=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/verification_plots",
    log:
        f"logs/downstream_analysis_timeseries/verification_plots.log",
    shell:
        """
        poetry run python modules/downstream_analysis/scripts/create_verification_plots.py \\
            --input {input.combined} \\
            --output-dir {params.output_dir} \\
            --experiment-type timeseries \\
            --flag-file {output.flag} > {log} 2>&1
        """


# In rules/timeseries.smk


rule advanced_analysis_timeseries:
    input:
        combined=ANALYSIS_CONFIG["inputs"]["timeseries_combined_data"],
    output:
        # We now only promise the one file that is guaranteed to be created.
        tp_comp=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/advanced_analysis/timepoint_populations/timepoint_population_comparison_results.csv",
    params:
        output_dir=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/advanced_analysis",
    log:
        f"logs/downstream_analysis_timeseries/advanced_analysis.log",
    shell:
        # We now only ask the script to perform the analysis that is possible.
        """
        poetry run python modules/downstream_analysis/scripts/advanced_cell_analysis.py \\
            --input {input.combined} \\
            --output {params.output_dir} \\
            --compare-timepoints \\
            --filter-outliers --outlier-percentile 99.995 > {log} 2>&1
        """


# In rules/timeseries.smk


rule summarize_timeseries_variance:
    input:
        raw_data=ANALYSIS_CONFIG["inputs"]["timeseries_combined_data"],
    output:
        lmm_plot=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/variance_summary/lmm_variance_components_plot.png",
        lmm_summary=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/variance_summary/lmm_model_summary.txt",
        lmm_csv=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/variance_summary/lmm_variance_components.csv",
    params:
        output_dir=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/variance_summary",
    log:
        f"logs/downstream_analysis_timeseries/summarize_variance.log",
    shell:
        """
        # --- CHANGE THIS LINE ---
        poetry run python modules/downstream_analysis/scripts/analyze_timeseries_variance.py \\
            --input-csv {input.raw_data} \\
            --output-dir {params.output_dir} > {log} 2>&1
        """
