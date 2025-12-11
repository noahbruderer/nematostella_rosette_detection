#!/usr/bin/env python3
"""
Generate a summary report from the pipeline's analysis outputs.
"""

import argparse
import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML


def image_to_base64(path: str) -> str:
    """Converts an image file to a Base64 encoded string."""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"⚠️ Warning: Image not found at {path}")
        return ""


def csv_to_html(path: str) -> str:
    """Reads a CSV and converts it to an HTML table."""
    try:
        df = pd.read_csv(path)
        return df.to_html(
            index=False, float_format="%.3f", classes="table table-striped"
        )
    except FileNotFoundError:
        print(f"⚠️ Warning: CSV not found at {path}")
        return "<p>Summary table could not be found.</p>"


def main():
    parser = argparse.ArgumentParser(description="Generate analysis report.")
    # Arguments for Timeseries
    parser.add_argument("--ts-lmm-plot", required=True)
    parser.add_argument("--ts-lmm-csv", required=True)
    parser.add_argument("--ts-rosette-time-plot", required=True)  # NEW
    parser.add_argument("--ts-rosette-boxplot", required=True)  # NEW
    parser.add_argument("--ts-rosette-heatmap", required=True)  # NEW

    # Arguments for Constant-Fed (NEW)
    parser.add_argument("--cf-lmm-plot", required=True)
    parser.add_argument("--cf-lmm-csv", required=True)

    # Arguments for Inhibitors
    parser.add_argument("--inhibitor-heatmaps", required=True, nargs="+")
    parser.add_argument("--inhibitor-summaries", required=True, nargs="+")
    parser.add_argument("--inhibitor-rosette-plots", required=True, nargs="+")
    parser.add_argument("--inhibitor-names", required=True, nargs="+")

    # Input/Output
    parser.add_argument("--template", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    print("--- 1. Assembling data for report ---")
    context = {
        "title": "Rosette Quantification Analysis Report",
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Timeseries Data
        "timeseries_lmm_plot": image_to_base64(args.ts_lmm_plot),
        "timeseries_lmm_csv_table": csv_to_html(args.ts_lmm_csv),
        "timeseries_rosette_time_plot": image_to_base64(
            args.ts_rosette_time_plot
        ),  # NEW
        "timeseries_rosette_boxplot": image_to_base64(args.ts_rosette_boxplot),  # NEW
        "timeseries_rosette_heatmap": image_to_base64(args.ts_rosette_heatmap),  # NEW
        # Constant-Fed Data (NEW)
        "cf_lmm_plot": image_to_base64(args.cf_lmm_plot),
        "cf_lmm_csv_table": csv_to_html(args.cf_lmm_csv),
        # Inhibitor Data
        "inhibitors": [],
    }

    print("--- 2. Gathering inhibitor results ---")
    # Zip the names and file paths together to process each inhibitor
    for name, heatmap_path, summary_path, rosette_path in zip(
        args.inhibitor_names,
        args.inhibitor_heatmaps,
        args.inhibitor_summaries,
        args.inhibitor_rosette_plots,
    ):
        print(f"     - Processing {name}...")
        inhibitor_data = {
            "name": name,
            "heatmap": image_to_base64(heatmap_path),
            "summary_table": csv_to_html(summary_path),
            "rosette_plot": image_to_base64(rosette_path),
        }
        context["inhibitors"].append(inhibitor_data)

    print("--- 3. Rendering HTML from template ---")
    env = Environment(loader=FileSystemLoader(Path(args.template).parent))
    template = env.get_template(Path(args.template).name)
    html_output = template.render(context)

    with open(args.output_html, "w") as f:
        f.write(html_output)
    print(f"✅ HTML report saved to: {args.output_html}")

    print("--- 4. Converting HTML to PDF ---")
    HTML(string=html_output, base_url=__file__).write_pdf(args.output_pdf)
    print(f"✅ PDF report saved to: {args.output_pdf}")


if __name__ == "__main__":
    main()
