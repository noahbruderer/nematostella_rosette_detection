# Snakemake Rules for Inhibitor Downstream Analysis (Refactored)
# =============================================================

import os
from snakemake.io import touch, expand

ANALYSIS_CONFIG = config["modules"]["downstream_analysis"]["config"]
INHIBITOR_NAMES = list(ANALYSIS_CONFIG["inputs"]["inhibitor_inputs"].keys())


# --- 1. Main Target Rule for the ENTIRE Inhibitor Analysis ---
# This is now the single, clear goal for this workflow.
rule inhibitor_downstream_analysis_all:
    input:
        # This rule requires the final summary heatmap for every inhibitor.
        # This is the key that ensures the whole pipeline runs to completion.
        expand(
            os.path.join(
                ANALYSIS_CONFIG["outputs"]["inhibitor_output_dir"],
                "{inhibitor}",
                "advanced_analysis",
                "variance_summary",
                "{inhibitor}_effect_summary_heatmap.png",
            ),
            inhibitor=INHIBITOR_NAMES,
        ),
        # It also depends on the verification plots being done.
        expand(
            os.path.join(
                ANALYSIS_CONFIG["outputs"]["inhibitor_output_dir"],
                "{inhibitor}",
                "verification_plots.flag",
            ),
            inhibitor=INHIBITOR_NAMES,
        ),
    output:
        # The final output is a single flag indicating this entire sub-workflow is complete.
        touch("results/downstream_analysis_inhibitors.flag"),


# --- 2. Rules for a single inhibitor ---


rule create_inhibitor_verification_plots:
    """Creates QC plots for a single inhibitor."""
    input:
        # --- THIS IS THE FIX ---
        # Change this from a hard-coded string to the lambda function
        combined=lambda wildcards: ANALYSIS_CONFIG["inputs"]["inhibitor_inputs"][wildcards.inhibitor],
    output:
        flag=touch(
            f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/verification_plots.flag"
        ),
    params:
        output_dir=f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/verification_plots",
    log:
        f"logs/downstream_analysis_inhibitors/{{inhibitor}}_verification.log",
    shell:
        """
        poetry run python modules/downstream_analysis/scripts/create_inhibitor_verification_plots.py \
            --input {input.combined} \
            --output-dir {params.output_dir} \
            --inhibitor-name "{wildcards.inhibitor}" > {log} 2>&1
        """


rule advanced_analysis_inhibitor:
    """Runs advanced GMM analysis and saves results to a dedicated folder."""
    input:
        combined=lambda wildcards: ANALYSIS_CONFIG["inputs"]["inhibitor_inputs"][wildcards.inhibitor],
    output:
        # The key output this rule is responsible for is the summary CSV.
        summary_csv=f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/advanced_analysis/control_vs_treatment_summary.csv",
    params:
        # The script is now told to save everything inside this specific inhibitor's advanced_analysis folder.
        output_dir=f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/advanced_analysis",
    log:
        f"logs/downstream_analysis_inhibitors/{{inhibitor}}_advanced_analysis.log",
    shell:
        """
        poetry run python modules/downstream_analysis/scripts/advanced_inhibitor_analysis.py \
            --input {input.combined} \
            --output-dir {params.output_dir} \
            --inhibitor-name "{wildcards.inhibitor}" > {log} 2>&1
        """


# The rule can also be renamed for clarity
rule summarize_inhibitor_effects:
    """Creates the final summary CSV and heatmap from the advanced analysis results."""
    input:
        summary_csv=f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/advanced_analysis/control_vs_treatment_summary.csv",
    output:
        # Declare both the heatmap and the new summary CSV as outputs
        heatmap=f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/advanced_analysis/variance_summary/{{inhibitor}}_effect_summary_heatmap.png",
        summary_data=f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/advanced_analysis/variance_summary/{{inhibitor}}_effect_summary.csv",
    params:
        inhibitor_name="{inhibitor}",
        output_dir=f"{ANALYSIS_CONFIG['outputs']['inhibitor_output_dir']}/{{inhibitor}}/advanced_analysis/variance_summary",
    log:
        f"logs/downstream_analysis_inhibitors/{{inhibitor}}_summarize_effects.log",
    shell:
        """
        poetry run python modules/downstream_analysis/scripts/summarize_inhibitor_variance.py \\
            --input-csv {input.summary_csv} \\
            --inhibitor-name {params.inhibitor_name} \\
            --output-dir {params.output_dir} > {log} 2>&1
        """
