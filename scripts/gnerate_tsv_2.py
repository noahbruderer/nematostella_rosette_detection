#!/usr/bin/env python3

"""
Generates a master metadata TSV file for ALL CURATED ROSETTE MASKS. (v10)

This script walks through the curated data directories for all experiments:
1. Timeseries R1/R2
2. Timeseries R3
3. Timeseries Manual
4. Inhibitor R1/R2
5. Inhibitor R3
6. Inhibitor QVD
7. Constant Fed (NEW)

v10 changes:
- Added 'constant_fed' dataset processing.
- Generates 'unique_sample_id' for all rows (matching the raw metadata format,
  but without image type suffixes like -PRI or -SINGLE).
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# --- Regex Patterns ---

# TIMESERIES_FOLDER_PATTERN: e.g. "R2_T0_10_0001"
TIMESERIES_FOLDER_PATTERN = re.compile(
    r"^(?:(R\d)_)?(T\d+)_(\d+)(?:_(\d+))?", re.IGNORECASE
)

# INHIBITOR_FOLDER_PATTERN: e.g. "R1_T20_AZD8055_1"
INHIBITOR_FOLDER_PATTERN = re.compile(
    r"^(R\d)(?:_(T\d+))?_(.*?)_(\d+)(?:_(\d+))?$", re.IGNORECASE
)

# CONSTANT_FED_PATTERN: e.g. "CF_R1_D10_1_1"
CONSTANT_FED_PATTERN = re.compile(r"^CF_(R\d)_(D\d+)_(\d+)(?:_(\d+))?", re.IGNORECASE)


# =============================================================================
# UNIQUE ID GENERATION
# =============================================================================
def create_unique_id(
    replicate,
    timepoint,
    animal_id,
    image_id,
    treatment="none",
    is_control=False,
):
    """
    Creates a standardized unique sample ID.
    Matches the logic of the raw metadata script, but EXCLUDES the image_type suffix.
    Output format: R1-T20-A1-I1 (or R1-T20-TREAT-A1-I1)
    """
    # 1. Sanitize base components
    rep = str(replicate).upper().replace("R", "")
    tp_str = str(timepoint).upper()
    anim = str(animal_id)
    img = str(image_id)

    # 2. Handle timepoint prefix (D vs T)
    if tp_str.startswith("D"):
        tp = f"D{tp_str.replace('D', '')}"  # D1 -> D1
    else:
        tp = f"T{tp_str.replace('T', '')}"  # T20 -> T20, NA -> TNA

    rep_str = f"R{rep}"  # 1 -> R1

    # 3. Sanitize treatment and add control flag
    sane_treatment = re.sub(r"[\s\W]+", "_", str(treatment).upper())
    if sane_treatment in ["NONE", "NA", "", "UNKNOWN"]:
        sane_treatment = None

    if is_control and sane_treatment:
        sane_treatment = f"{sane_treatment}_CTRL"

    # 4. Build base ID string
    base_id_parts = [rep_str, tp]
    if sane_treatment:
        base_id_parts.append(sane_treatment)
    base_id_parts.extend([f"A{anim}", f"I{img}"])

    # Join parts (e.g., R1-T20-A1-I1)
    return "-".join(base_id_parts)


# =============================================================================
# PARSING LOGIC
# =============================================================================


def parse_image_id(id_match: str | None) -> str:
    """Implements the 'implicit 0' logic for the image_id."""
    if id_match:
        return str(int(id_match))
    else:
        return "0"


def parse_timeseries_folder_name(
    sample_dir: Path, r1r2_root_list: list[Path]
) -> dict | None:
    """Parses metadata from a timeseries folder name."""
    folder_name = sample_dir.name
    clean_folder_name = re.sub(r"(_seg)?(\s*copy)?(\.svg)?$", "", folder_name)

    match = TIMESERIES_FOLDER_PATTERN.search(clean_folder_name)
    if not match:
        # Handle the one-off "T0_9_3_z4" case
        if folder_name == "T0_9_3_z4":
            unique_id = create_unique_id("R1", "T0", "9", "3")
            return {
                "unique_sample_id": unique_id,
                "experiment_type": "timeseries",
                "replicate": "R1",
                "timepoint": "T0",
                "animal_id": "9",
                "image_id": "3",
                "treatment": "none",
                "is_control": False,
            }
        return None

    replicate_in_name, timepoint, animal_id, image_id_suffix = match.groups()

    replicate = replicate_in_name
    if not replicate:
        # Infer R1 vs R3 based on root path
        is_r1r2 = False
        for root in r1r2_root_list:
            if root in sample_dir.parents:
                is_r1r2 = True
                break
        replicate = "R1" if is_r1r2 else "R3"

    image_id = parse_image_id(image_id_suffix)

    unique_id = create_unique_id(replicate, timepoint, animal_id, image_id)

    return {
        "unique_sample_id": unique_id,
        "experiment_type": "timeseries",
        "replicate": replicate.upper(),
        "timepoint": timepoint.upper(),
        "animal_id": str(int(animal_id)),
        "image_id": image_id,
        "treatment": "none",
        "is_control": False,
    }


def parse_inhibitor_folder_name(sample_dir: Path) -> dict | None:
    """Parses metadata from an INHIBITOR folder name."""
    folder_name = sample_dir.name
    match = INHIBITOR_FOLDER_PATTERN.match(folder_name)
    if not match:
        return None

    replicate, timepoint_in_filename, treatment_str, animal_id, image_id_suffix = (
        match.groups()
    )

    # Determine Timepoint
    timepoint = timepoint_in_filename
    if not timepoint:
        for parent_dir in [sample_dir.parent, sample_dir.parent.parent]:
            tp_match = re.search(r"(T\d+)", parent_dir.name, re.IGNORECASE)
            if tp_match:
                timepoint = tp_match.group(1)
                break

    safe_timepoint = timepoint.upper() if timepoint else "NA"

    # Determine Treatment / Control
    is_control = False
    if "_dmso" in treatment_str.lower():
        is_control = True
        treatment_str = treatment_str.lower().replace("_dmso", "").replace("dmso_", "")

    if "staroporine" in treatment_str:
        treatment_str = treatment_str.replace("staroporine", "staurosporine")

    if "zvad" in treatment_str:
        treatment_str = treatment_str.replace("_zvad", "_ZVAD").replace(
            "zvad_", "ZVAD_"
        )

    safe_treatment = treatment_str.upper() if treatment_str else "UNKNOWN"
    image_id = parse_image_id(image_id_suffix)

    unique_id = create_unique_id(
        replicate,
        safe_timepoint,
        animal_id,
        image_id,
        treatment=safe_treatment,
        is_control=is_control,
    )

    return {
        "unique_sample_id": unique_id,
        "experiment_type": "inhibitor",
        "replicate": replicate.upper(),
        "timepoint": safe_timepoint,
        "animal_id": str(int(animal_id)),
        "image_id": image_id,
        "treatment": safe_treatment,
        "is_control": is_control,
    }


def parse_constant_fed_name(sample_dir: Path) -> dict | None:
    """
    Parses metadata from a CONSTANT FED folder/file name.
    e.g., "CF_R1_D10_1_1"
    """
    # Use stem to handle both folders and files (removes .tif extension if present)
    name_stem = sample_dir.stem

    match = CONSTANT_FED_PATTERN.search(name_stem)
    if not match:
        return None

    replicate, day, animal_id, image_id_suffix = match.groups()
    image_id = parse_image_id(image_id_suffix)

    unique_id = create_unique_id(
        replicate, day, animal_id, image_id, treatment="none", is_control=False
    )

    return {
        "unique_sample_id": unique_id,
        "experiment_type": "constant_fed",
        "replicate": replicate.upper(),
        "timepoint": day.upper(),
        "animal_id": str(int(animal_id)),
        "image_id": image_id,
        "treatment": "none",
        "is_control": False,
    }


# =============================================================================
# FILE FINDING LOGIC
# =============================================================================


def find_curated_file_r1r2(sample_dir: Path) -> Path | None:
    copy_file = sample_dir / "segmentation copy.tif"
    if copy_file.exists():
        return copy_file
    main_file = sample_dir / "segmentation.tif"
    if main_file.exists():
        return main_file
    return None


def find_curated_file_r3_style(sample_dir: Path) -> Path | None:
    name = sample_dir.name
    # Prioritize copy
    copy_file = sample_dir / f"02_segmentation_{name}_seg_cur copy.tif"
    if copy_file.exists():
        return copy_file

    main_file = sample_dir / f"02_segmentation_{name}_seg_cur.tif"
    if main_file.exists():
        return main_file

    # Fallback
    copy_file_ts = sample_dir / f"02_segmentation_{name}_masks_curated copy.tif"
    if copy_file_ts.exists():
        return copy_file_ts
    main_file_ts = sample_dir / f"02_segmentation_{name}_masks_curated.tif"
    if main_file_ts.exists():
        return main_file_ts
    return None


def find_curated_file_manual(sample_dir: Path) -> Path | None:
    expected_filename = f"{sample_dir.stem}.tif"
    file_path = sample_dir / expected_filename
    if file_path.exists():
        return file_path
    return None


# =============================================================================
# MAIN
# =============================================================================


def main(
    ts_r1r2_root,
    ts_r3_root,
    ts_manual_root,
    inhib_r1r2_root,
    inhib_r3_root,
    inhib_qvd_root,
    constant_fed_root,
    output_file,
):
    ts_r1r2_path = Path(ts_r1r2_root)
    ts_r3_path = Path(ts_r3_root)
    ts_manual_path = Path(ts_manual_root)
    inhib_r1r2_path = Path(inhib_r1r2_root)
    inhib_r3_path = Path(inhib_r3_root)
    inhib_qvd_path = Path(inhib_qvd_root)
    constant_fed_path = Path(constant_fed_root)
    output_path = Path(output_file)

    r1r2_known_roots = [ts_r1r2_path, ts_manual_path]

    # Check directories
    paths_to_check = [
        (ts_r1r2_path, "Timeseries R1/R2"),
        (ts_r3_path, "Timeseries R3"),
        (ts_manual_path, "Timeseries Manual"),
        (inhib_r1r2_path, "Inhibitor R1/R2"),
        (inhib_r3_path, "Inhibitor R3"),
        (inhib_qvd_path, "Inhibitor QVD"),
        (constant_fed_path, "Constant Fed"),
    ]

    for p, name in paths_to_check:
        if not p.is_dir():
            print(f"Error: {name} curated root not found: {p}", file=sys.stderr)
            sys.exit(1)

    all_curated_data = []

    # --- Part 1: Timeseries R1/R2 ---
    print(f"Scanning Timeseries R1/R2 data in: {ts_r1r2_path}...")
    for sample_dir in ts_r1r2_path.iterdir():
        if not sample_dir.is_dir():
            continue

        data_dict = parse_timeseries_folder_name(sample_dir, r1r2_known_roots)
        if data_dict and (curated_file := find_curated_file_r1r2(sample_dir)):
            data_dict["curated_rosette_path"] = str(curated_file)
            all_curated_data.append(data_dict)

    # --- Part 2: Timeseries R3 ---
    print(f"Scanning Timeseries R3 data in: {ts_r3_path}...")
    for timepoint_dir in ts_r3_path.iterdir():
        if not timepoint_dir.is_dir():
            continue
        for sample_dir in timepoint_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            data_dict = parse_timeseries_folder_name(sample_dir, r1r2_known_roots)
            if data_dict and (curated_file := find_curated_file_r3_style(sample_dir)):
                data_dict["curated_rosette_path"] = str(curated_file)
                all_curated_data.append(data_dict)

    # --- Part 3: Inhibitor R1/R2 ---
    print(f"Scanning Inhibitor R1/R2 data in: {inhib_r1r2_path}...")
    for treatment_dir in inhib_r1r2_path.iterdir():
        if not treatment_dir.is_dir():
            continue
        for sample_dir in treatment_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            data_dict = parse_inhibitor_folder_name(sample_dir)
            if data_dict and (curated_file := find_curated_file_r1r2(sample_dir)):
                data_dict["curated_rosette_path"] = str(curated_file)
                all_curated_data.append(data_dict)

    # --- Part 4: Inhibitor R3 ---
    print(f"Scanning Inhibitor R3 data in: {inhib_r3_path}...")
    for treatment_dir in inhib_r3_path.iterdir():
        if not treatment_dir.is_dir():
            continue
        for sample_dir in treatment_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            data_dict = parse_inhibitor_folder_name(sample_dir)
            if data_dict and (curated_file := find_curated_file_r3_style(sample_dir)):
                data_dict["curated_rosette_path"] = str(curated_file)
                all_curated_data.append(data_dict)

    # --- Part 5: Inhibitor QVD ---
    print(f"Scanning Inhibitor QVD data in: {inhib_qvd_path}...")
    for treatment_dir in inhib_qvd_path.iterdir():
        if not treatment_dir.is_dir():
            continue
        for sample_dir in treatment_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            data_dict = parse_inhibitor_folder_name(sample_dir)
            if data_dict and (curated_file := find_curated_file_r3_style(sample_dir)):
                data_dict["curated_rosette_path"] = str(curated_file)
                all_curated_data.append(data_dict)

    # --- Part 6: Timeseries MANUAL ---
    print(f"Scanning Timeseries Manual data in: {ts_manual_path}...")
    for sample_dir in ts_manual_path.iterdir():
        if not sample_dir.is_dir():
            continue

        data_dict = parse_timeseries_folder_name(sample_dir, r1r2_known_roots)
        if not data_dict:
            continue

        if ".svg" in sample_dir.name:
            curated_file = find_curated_file_manual(sample_dir)
        else:
            curated_file = find_curated_file_r3_style(sample_dir)

        if curated_file:
            data_dict["curated_rosette_path"] = str(curated_file)
            all_curated_data.append(data_dict)

    # --- Part 7: CONSTANT FED (New) ---
    print(f"Scanning Constant Fed data in: {constant_fed_path}...")
    # Iterate both directories and files to be safe
    for item in constant_fed_path.iterdir():
        data_dict = parse_constant_fed_name(item)
        if not data_dict:
            continue

        curated_file = None

        # If item is a directory (common for R3/newer pipelines), look inside
        if item.is_dir():
            curated_file = find_curated_file_r3_style(item)
            # Fallback: check if it is r1/r2 style
            if not curated_file:
                curated_file = find_curated_file_r1r2(item)

        # If item is a file (e.g. CF_..._seg_cur.tif)
        elif item.is_file() and item.suffix == ".tif":
            # We assume the file itself is the curated mask if it's in this folder
            curated_file = item

        if curated_file:
            data_dict["curated_rosette_path"] = str(curated_file)
            all_curated_data.append(data_dict)

    if not all_curated_data:
        print("Error: No valid curated files were found.", file=sys.stderr)
        sys.exit(1)

    # --- Create and Save DataFrame ---
    print(f"\nFound {len(all_curated_data)} valid curated samples.")

    df = pd.DataFrame(all_curated_data)

    all_columns = [
        "unique_sample_id",
        "experiment_type",
        "replicate",
        "timepoint",
        "treatment",
        "is_control",
        "animal_id",
        "image_id",
        "curated_rosette_path",
    ]

    for col in all_columns:
        if col not in df.columns:
            df[col] = "NA"

    df = df[all_columns]

    # Deduplicate based on unique_sample_id
    pre_dedupe_len = len(df)
    df = df.drop_duplicates(subset=["unique_sample_id"], keep="first")
    post_dedupe_len = len(df)

    if pre_dedupe_len != post_dedupe_len:
        print(f"\nWarning: Removed {pre_dedupe_len - post_dedupe_len} duplicate IDs.")

    # Sort
    df = df.sort_values(
        by=["experiment_type", "replicate", "timepoint", "animal_id", "image_id"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)

    print(f"\n✅ Success! Curated metadata file saved to: {output_path}")
    print("\n--- Example Rows ---")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metadata TSV for ALL curated rosette segmentation masks."
    )
    # Timeseries
    parser.add_argument("--ts-r1r2-root", required=True)
    parser.add_argument("--ts-r3-root", required=True)
    parser.add_argument("--ts-manual-root", required=True)
    # Inhibitor
    parser.add_argument("--inhib-r1r2-root", required=True)
    parser.add_argument("--inhib-r3-root", required=True)
    parser.add_argument("--inhib-qvd-root", required=True)
    # Constant Fed (New)
    parser.add_argument("--constant-fed-root", required=True)
    # Output
    parser.add_argument("--output-file", default="config/curated_rosette_metadata.tsv")

    args = parser.parse_args()

    main(
        args.ts_r1r2_root,
        args.ts_r3_root,
        args.ts_manual_root,
        args.inhib_r1r2_root,
        args.inhib_r3_root,
        args.inhib_qvd_root,
        args.constant_fed_root,
        args.output_file,
    )

# Example usage:
# python3 scripts/gnerate_tsv_2.py \
#   --ts-manual-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/4_manually_annotated_rosettes_training_dataset/241120_timeseries_Rosette_labels_1st_half_manually_annotated" \
#   --ts-r1r2-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/7_final_segmentation_rosettes_R1_R2/250206_Rosette_timeseries_image_stacking_vf/human_in_the_loop" \
#   --ts-r3-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/8_segmentation_&_rosettes_R3/250707_timeseries_live_R3_seg" \
#   --inhib-r1r2-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250212_Rosette_inhibitor_R1_R2/rosette_inhibitor_human_in_the_loop_data" \
#   --inhib-r3-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250714_inhibitors_R3_seg" \
#   --inhib-qvd-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250723_qvd_inhibitor_seg" \
#   --constant-fed-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/constant_fed/2_250720_constant_fed_seg" \
#   --output-file "config/curated_rosette_metadata_other_test.tsv"
# python3 scripts/gnerate_tsv_2.py \
#   --ts-manual-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/4_manually_annotated_rosettes_training_dataset/241120_timeseries_Rosette_labels_1st_half_manually_annotated" \
#   --ts-r1r2-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/7_final_segmentation_rosettes_R1_R2/250206_Rosette_timeseries_image_stacking_vf/human_in_the_loop" \
#   --ts-r3-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/8_segmentation_&_rosettes_R3/250707_timeseries_live_R3_seg" \
#   --inhib-r1r2-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250212_Rosette_inhibitor_R1_R2/rosette_inhibitor_human_in_the_loop_data" \
#   --inhib-r3-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250714_inhibitors_R3_seg" \
#   --inhib-qvd-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250723_qvd_inhibitor_seg" \
#   --output-file "config/curated_rosette_metadata_other_test.tsv"
