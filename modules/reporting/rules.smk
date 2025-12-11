# modules/reporting/rules.smk

import os

ANALYSIS_CONFIG = config["modules"]["downstream_analysis"]["config"]
INHIBITOR_NAMES = list(ANALYSIS_CONFIG["inputs"]["inhibitor_inputs"].keys())


rule generate_report:
    input:
        # --- List EVERY file the report needs ---
        # Timeseries Files (FIXED PATHS)
        ts_lmm_plot=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/variance_summary/lmm_variance_components_plot.png",
        ts_lmm_csv=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/variance_summary/lmm_variance_components.csv",
        
        # Timeseries Rosette Plots (NEW)
        ts_rosette_time_plot=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/advanced_analysis/timepoint_populations/summary/rosette_metrics_over_time.png",
        ts_rosette_boxplot=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/verification_plots/timeseries_rosette_statistics/rosette_by_timepoint_sorted.png",
        ts_rosette_heatmap=f"{ANALYSIS_CONFIG['outputs']['timeseries_output_dir']}/verification_plots/timeseries_rosette_statistics/rosette_heatmap_per_1000_cells.png",

        # Constant Fed Files (NEW)
        cf_lmm_plot=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/variance_summary/lmm_variance_components_plot.png",
        cf_lmm_csv=f"{ANALYSIS_CONFIG['outputs']['constant_fed_output_dir']}/variance_summary/lmm_variance_components.csv",
        
        # Inhibitor Files (using expand for all inhibitors)
        inhib_heatmaps=expand(
            f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/advanced_analysis/variance_summary/{{inhibitor}}_effect_summary_heatmap.png",
            inhibitor=INHIBITOR_NAMES,
        ),
        inhib_summaries=expand(
            f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/advanced_analysis/variance_summary/{{inhibitor}}_effect_summary.csv",
            inhibitor=INHIBITOR_NAMES,
        ),
        # Inhibitor Rosette Plots
        inhib_rosette_plots=expand(
            f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/verification_plots/rosette_percentage_boxplot.png",
            inhibitor=INHIBITOR_NAMES,
        ),
    output:
        html="results/final_report.html",
        pdf="results/final_report.pdf",
    params:
        # We still need the inhibitor names to label the plots correctly
        inhibitor_names=" ".join(INHIBITOR_NAMES),
    log:
        "logs/generate_report.log",
    shell:
        """
        poetry run python modules/reporting/scripts/generate_report.py \\
            --ts-lmm-plot {input.ts_lmm_plot} \\
            --ts-lmm-csv {input.ts_lmm_csv} \\
            --ts-rosette-time-plot {input.ts_rosette_time_plot} \\
            --ts-rosette-boxplot {input.ts_rosette_boxplot} \\
            --ts-rosette-heatmap {input.ts_rosette_heatmap} \\
            --cf-lmm-plot {input.cf_lmm_plot} \\
            --cf-lmm-csv {input.cf_lmm_csv} \\
            --inhibitor-heatmaps {input.inhib_heatmaps} \\
            --inhibitor-summaries {input.inhib_summaries} \\
            --inhibitor-rosette-plots {input.inhib_rosette_plots} \\
            --inhibitor-names {params.inhibitor_names} \\
            --template modules/reporting/scripts/report_template.html \\
            --output-html {output.html} \\
            --output-pdf {output.pdf} > {log} 2>&1
        """


# The reporting_all rule remains the same
rule reporting_all:
    input:
        "results/final_report.html",
        "results/final_report.pdf",
    output:
        touch("results/reporting.flag"),