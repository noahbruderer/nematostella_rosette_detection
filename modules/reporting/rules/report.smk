import os

ANALYSIS_CONFIG = config["modules"]["downstream_analysis"]["config"]
INHIBITOR_NAMES = list(ANALYSIS_CONFIG["inputs"]["inhibitor_inputs"].keys())


rule generate_report:
    input:
        ts_flag="results/downstream_analysis_timeseries.flag",
        inhib_flag="results/downstream_analysis_inhibitors.flag",
    output:
        html="results/final_report.html",
        pdf="results/final_report.pdf",
    params:
        timeseries_dir=ANALYSIS_CONFIG["outputs"]["timeseries_output_dir"],
        inhibitor_dir=ANALYSIS_CONFIG["outputs"]["inhibitor_output_dir"],
        inhibitors=" ".join(INHIBITOR_NAMES),
    log:
        "logs/generate_report.log",
    shell:
        """
        poetry run python modules/reporting/scripts/generate_report.py \\
            --timeseries-dir {params.timeseries_dir} \\
            --inhibitor-dir {params.inhibitor_dir} \\
            --inhibitors {params.inhibitors} \\
            --template modules/reporting/scripts/report_template.html \\
            --output-html {output.html} \\
            --output-pdf {output.pdf} > {log} 2>&1
        """
