#!/usr/bin/env python3
"""
Generates a master metadata TSV file for ALL CURATED ROSETTE MASKS. (v10)

Features:
- Scans R1/R2, R3, and Inhibitor directories recursively.
- Generates standardized unique_sample_ids.
- Calculates paths relative to a provided --global-root.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# =============================================================================
# 1. Helper Functions
# =============================================================================


def create_base_unique_id(
    replicate, timepoint, animal_id, image_id, treatment="none", is_control=False
):
    """Standardized ID generation: R#-T#-TRT-A#-I#"""
    rep = str(replicate).upper().replace("R", "")
    tp = str(timepoint).upper().replace("T", "")

    sane_treatment = None
    if treatment and treatment.lower() not in ["none", "na", ""]:
        sane_treatment = re.sub(r"[\s\W]+", "_", str(treatment).upper())
        if is_control:
            sane_treatment += "_CTRL"

    parts = [f"R{rep}", f"T{tp}"]
    if sane_treatment:
        parts.append(sane_treatment)
    parts.extend([f"A{animal_id}", f"I{image_id}"])

    return "-".join(parts)


def get_relative_path_safe(file_path, root_path):
    """
    Calculates path relative to root.
    Returns absolute path if relative calculation fails (safety fallback).
    """
    try:
        return str(file_path.relative_to(root_path))
    except ValueError:
        try:
            # Try resolving symlinks
            return str(file_path.resolve().relative_to(root_path.resolve()))
        except ValueError:
            print(
                f"⚠️  WARNING: File is outside global root. Using absolute path.\n    File: {file_path}\n    Root: {root_path}"
            )
            return str(file_path)


# =============================================================================
# 2. Parsers for Specific Folder Structures
# =============================================================================


def parse_r1r2_manual(sample_dir: Path, global_root: Path) -> dict | None:
    """
    Parses standard Timeseries folders (e.g., T21_7).
    """
    folder_name = sample_dir.name
    # Regex: T21_7 or T0_9_3
    match = re.match(r"^T(\d+)_(\d+)(?:_(\d+))?.*$", folder_name, re.IGNORECASE)
    if not match:
        return None

    timepoint, animal_id, image_id_suffix = match.groups()
    image_id = image_id_suffix if image_id_suffix else "0"
    replicate = "R1"  # inferred default

    # Priority Search for Manual Files
    candidates = list(sample_dir.glob("*segmentation*copy.tif"))
    if not candidates:
        candidates = list(sample_dir.glob("*segmentation.tif"))
    if not candidates:
        candidates = list(
            sample_dir.glob("*.tif")
        )  # Fallback for manual folder where file=folder name

    if not candidates:
        return None

    # Pick the best candidate (usually the first match)
    curated_file = candidates[0]

    # Create ID
    unique_id = create_base_unique_id(replicate, f"T{timepoint}", animal_id, image_id)

    return {
        "unique_sample_id": unique_id,
        "experiment_type": "timeseries",
        "replicate": f"R{replicate.replace('R', '')}",
        "timepoint": f"T{timepoint}",
        "animal_id": int(animal_id),
        "image_id": int(image_id),
        "treatment": "none",
        "is_control": False,
        "curated_rosette_path": get_relative_path_safe(curated_file, global_root),
    }


def parse_r3_manual(sample_dir: Path, global_root: Path) -> dict | None:
    """
    Parses R3/Napari folders (e.g., R3_T21_18_0001).
    """
    folder_name = sample_dir.name
    match = re.match(r"^R(\d+)_T(\d+)_(\d+)(?:_(\d+))?.*$", folder_name, re.IGNORECASE)
    if not match:
        return None

    replicate, timepoint, animal_id, image_id_suffix = match.groups()
    image_id = str(int(image_id_suffix)) if image_id_suffix else "0"

    # Search patterns
    candidates = list(sample_dir.glob("*masks_curated*copy.tif"))
    if not candidates:
        candidates = list(sample_dir.glob("*masks_curated.tif"))
    if not candidates:
        candidates = list(sample_dir.glob("*seg_cur.tif"))

    if not candidates:
        return None

    curated_file = candidates[0]
    unique_id = create_base_unique_id(replicate, f"T{timepoint}", animal_id, image_id)

    return {
        "unique_sample_id": unique_id,
        "experiment_type": "timeseries",
        "replicate": f"R{replicate}",
        "timepoint": f"T{timepoint}",
        "animal_id": int(animal_id),
        "image_id": int(image_id),
        "treatment": "none",
        "is_control": False,
        "curated_rosette_path": get_relative_path_safe(curated_file, global_root),
    }


def parse_inhibitor_manual(sample_dir: Path, global_root: Path) -> dict | None:
    """
    Parses Inhibitor folders (e.g., R1_T20_AZD8055_1).
    """
    folder_name = sample_dir.name

    # Regex: R1_T20_AZD8055_1
    match = re.match(
        r"^(R\d+)_T(\d+)_(.*)_(\d+)(?:_(\d+))?$", folder_name, re.IGNORECASE
    )

    if not match:
        return None

    replicate, timepoint, treatment_str, animal_id, image_id_suffix = match.groups()
    image_id = str(int(image_id_suffix)) if image_id_suffix else "0"

    # Treatment cleanup
    is_control = False
    if "_dmso" in treatment_str.lower() or "dmso_" in treatment_str.lower():
        is_control = True
        treatment_str = re.sub(r"_?dmso_?", "", treatment_str, flags=re.IGNORECASE)

    if "staroporine" in treatment_str.lower():
        treatment_str = "staurosporine"

    # File Search
    candidates = list(sample_dir.glob("*segmentation*copy.tif"))
    if not candidates:
        candidates = list(sample_dir.glob("*segmentation.tif"))
    if not candidates:
        candidates = list(sample_dir.glob("*rosettes_curated.tif"))

    if not candidates:
        return None

    curated_file = candidates[0]

    unique_id = create_base_unique_id(
        replicate,
        f"T{timepoint}",
        animal_id,
        image_id,
        treatment=treatment_str,
        is_control=is_control,
    )

    return {
        "unique_sample_id": unique_id,
        "experiment_type": "inhibitor",
        "replicate": f"R{replicate.replace('R', '')}",
        "timepoint": f"T{timepoint}",
        "animal_id": int(animal_id),
        "image_id": int(image_id),
        "treatment": treatment_str.upper(),
        "is_control": is_control,
        "curated_rosette_path": get_relative_path_safe(curated_file, global_root),
    }


# =============================================================================
# 3. Main Execution
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-root", required=True)
    # Paths for all experiments
    parser.add_argument("--ts-r1r2-root", required=True)
    parser.add_argument("--ts-r3-root", required=True)
    parser.add_argument("--ts-manual-root", required=True)
    parser.add_argument("--inhib-r1r2-root", required=True)
    parser.add_argument("--inhib-r3-root", required=True)
    parser.add_argument("--inhib-qvd-root", required=True)
    parser.add_argument("--output-file", required=True)

    args = parser.parse_args()
    global_root = Path(args.global_root)

    all_data = []

    # Generic scanner function
    def scan_and_parse(path_str, parser_func, label):
        p = Path(path_str)
        print(f"\n🔍 Scanning {label} at: {p}")
        if not p.exists():
            print("  ❌ Path does not exist!")
            return

        count = 0
        # Recursive walk to find all sample folders
        for folder in p.rglob("*"):
            if folder.is_dir():
                data = parser_func(folder, global_root)
                if data:
                    all_data.append(data)
                    count += 1
        print(f"  ✅ Found {count} samples.")

    # --- Execute Scans ---
    scan_and_parse(args.ts_r1r2_root, parse_r1r2_manual, "Time Series R1/R2")
    scan_and_parse(args.ts_r3_root, parse_r3_manual, "Time Series R3")
    scan_and_parse(args.ts_manual_root, parse_r1r2_manual, "Time Series Manual")

    scan_and_parse(args.inhib_r1r2_root, parse_inhibitor_manual, "Inhibitor R1/R2")
    scan_and_parse(args.inhib_r3_root, parse_inhibitor_manual, "Inhibitor R3")
    scan_and_parse(args.inhib_qvd_root, parse_inhibitor_manual, "Inhibitor QVD")

    # --- Save Results ---
    if all_data:
        df = pd.DataFrame(all_data)

        # Clean strings (just in case)
        df_obj = df.select_dtypes(["object"])
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["unique_sample_id"])
        after = len(df)

        df.to_csv(args.output_file, sep="\t", index=False)

        print("\n" + "=" * 60)
        print("🎉 METADATA GENERATION COMPLETE")
        print(f"   Total Valid Samples: {after}")
        if before != after:
            print(f"   (Removed {before - after} duplicates)")
        print(f"   Saved to: {args.output_file}")
        print("=" * 60)

        # Print sample of the path to verify it's relative
        print("Example Path (should be relative):")
        print(f"   {df['curated_rosette_path'].iloc[0]}")

    else:
        print("\n❌ ERROR: No samples found in any directory.")


if __name__ == "__main__":
    main()
# python3 scripts/generate_metadata_rosette_predictions.py \
#   --ts-manual-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/4_manually_annotated_rosettes_training_dataset/241120_timeseries_Rosette_labels_1st_half_manually_annotated" \
#   --ts-r1r2-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/7_final_segmentation_rosettes_R1_R2/250206_Rosette_timeseries_image_stacking_vf/human_in_the_loop" \
#   --ts-r3-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/8_segmentation_&_rosettes_R3/250707_timeseries_live_R3_seg" \
#   --inhib-r1r2-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250212_Rosette_inhibitor_R1_R2/rosette_inhibitor_human_in_the_loop_data" \
#   --inhib-r3-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250714_inhibitors_R3_seg" \
#   --inhib-qvd-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250723_qvd_inhibitor_seg" \
#   --output-file "config/curated_rosette_metadata.tsv"
