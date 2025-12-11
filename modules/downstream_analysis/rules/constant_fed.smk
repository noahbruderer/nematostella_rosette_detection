# Snakemake Rules for Constant Fed Downstream Analysis
# ====================================================

from snakemake.io import touch

ANALYSIS_CONFIG = config["modules"]["downstream_analysis"]["config"]

# --- 1. Main Target Rule ---
rule constant_fed_downstream_analysis_all:
    input:
        # Reuse the variance analysis logic
        lmm_plot=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/variance_summary/lmm_variance_components_plot.png",
        lmm_summary=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/variance_summary/lmm_model_summary.txt",
        # Reuse the population analysis logic
        tp_comp=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/advanced_analysis/timepoint_populations/timepoint_population_comparison_results.csv",
        # QC plots
        verification_flag=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/verification_plots.flag",
    output:
        touch("results/downstream_analysis_constant_fed.flag"),

# --- 2. Create Verification Plots ---
rule create_cf_verification_plots:
    input:
        combined=ANALYSIS_CONFIG["inputs"]["constant_fed_combined_data"],
    output:
        flag=touch(f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/verification_plots.flag"),
    params:
        output_dir=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/verification_plots",
    log:
        f"logs/downstream_analysis_constant_fed/verification_plots.log",
    shell:
        """
        # Reusing the timeseries script because CF is essentially a timeseries (Days instead of Hours)
        poetry run python modules/downstream_analysis/scripts/create_verification_plots.py \
            --input {input.combined} \
            --output-dir {params.output_dir} \
            --experiment-type timeseries \
            --flag-file {output.flag} > {log} 2>&1
        """

# --- 3. Advanced Analysis (Population Shifts) ---
rule advanced_analysis_cf:
    input:
        combined=ANALYSIS_CONFIG["inputs"]["constant_fed_combined_data"],
    output:
        tp_comp=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/advanced_analysis/timepoint_populations/timepoint_population_comparison_results.csv",
    params:
        output_dir=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/advanced_analysis",
    log:
        f"logs/downstream_analysis_constant_fed/advanced_analysis.log",
    shell:
        """
        poetry run python modules/downstream_analysis/scripts/advanced_cell_analysis.py \
            --input {input.combined} \
            --output {params.output_dir} \
            --compare-timepoints \
            --filter-outliers --outlier-percentile 99.995 > {log} 2>&1
        """

# --- 4. Summarize Variance ---
rule summarize_cf_variance:
    input:
        raw_data=ANALYSIS_CONFIG["inputs"]["constant_fed_combined_data"],
    output:
        lmm_plot=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/variance_summary/lmm_variance_components_plot.png",
        lmm_summary=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/variance_summary/lmm_model_summary.txt",
    params:
        output_dir=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/variance_summary",
    log:
        f"logs/downstream_analysis_constant_fed/summarize_variance.log",
    shell:
        """
        poetry run python modules/downstream_analysis/scripts/analyze_timeseries_variance.py \
            --input-csv {input.raw_data} \
            --output-dir {params.output_dir} > {log} 2>&1
        """