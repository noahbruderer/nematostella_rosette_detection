#!/usr/bin/env python3

"""
Generates a master metadata TSV file for ALL CURATED ROSETTE MASKS. (v11)

This script walks through the curated data directories for all experiments
and generates a 'unique_sample_id' for all rows.

v11 changes:
- Adds a new column: 'base_segmentation_path'.
- This new column points to the *original* cell segmentation mask (e.g., ...seg.tif)
  that the rosette annotation (...seg copy.tif) was drawn on top of.
- This allows for a future "label transfer" pipeline if the erosion
  method proves insufficient.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# --- Regex Patterns ---
TIMESERIES_FOLDER_PATTERN = re.compile(
    r"^(?:(R\d)_)?(T\d+)_(\d+)(?:_(\d+))?", re.IGNORECASE
)
INHIBITOR_FOLDER_PATTERN = re.compile(
    r"^(R\d)(?:_(T\d+))?_(.*?)_(\d+)(?:_(\d+))?$", re.IGNORECASE
)
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
    rep = str(replicate).upper().replace("R", "")
    tp_str = str(timepoint).upper()
    anim = str(animal_id)
    img = str(image_id)

    if tp_str.startswith("D"):
        tp = f"D{tp_str.replace('D', '')}"
    else:
        tp = f"T{tp_str.replace('T', '')}"

    rep_str = f"R{rep}"

    sane_treatment = re.sub(r"[\s\W]+", "_", str(treatment).upper())
    if sane_treatment in ["NONE", "NA", "", "UNKNOWN"]:
        sane_treatment = None

    if is_control and sane_treatment:
        sane_treatment = f"{sane_treatment}_CTRL"

    base_id_parts = [rep_str, tp]
    if sane_treatment:
        base_id_parts.append(sane_treatment)
    base_id_parts.extend([f"A{anim}", f"I{img}"])

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

    timepoint = timepoint_in_filename
    if not timepoint:
        for parent_dir in [sample_dir.parent, sample_dir.parent.parent]:
            tp_match = re.search(r"(T\d+)", parent_dir.name, re.IGNORECASE)
            if tp_match:
                timepoint = tp_match.group(1)
                break

    safe_timepoint = timepoint.upper() if timepoint else "NA"

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
    """Parses metadata from a CONSTANT FED folder/file name."""
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
# FILE FINDING LOGIC (MODIFIED TO RETURN BOTH PATHS)
# =============================================================================


def find_curated_file_r1r2(sample_dir: Path) -> tuple[Path | None, Path | None]:
    """Finds (curated_rosette_mask, base_cell_mask)"""
    curated_rosette_path = sample_dir / "segmentation copy.tif"
    base_segmentation_path = sample_dir / "segmentation.tif"

    if curated_rosette_path.exists() and base_segmentation_path.exists():
        # Standard case: ...copy.tif is rosettes, ...seg.tif is base
        return curated_rosette_path, base_segmentation_path

    # Fallback: if no 'copy.tif' exists, assume '...seg.tif' is BOTH.
    if base_segmentation_path.exists() and not curated_rosette_path.exists():
        return base_segmentation_path, base_segmentation_path

    # Fallback: only copy exists (unlikely, but possible)
    if curated_rosette_path.exists() and not base_segmentation_path.exists():
        return curated_rosette_path, None  # No base seg

    return None, None


def find_curated_file_r3_style(sample_dir: Path) -> tuple[Path | None, Path | None]:
    """Finds (curated_rosette_mask, base_cell_mask)"""
    name = sample_dir.name

    # Define R3-style paths
    curated_rosette_path = sample_dir / f"02_segmentation_{name}_seg_cur copy.tif"
    base_segmentation_path = sample_dir / f"02_segmentation_{name}_seg_cur.tif"

    if curated_rosette_path.exists() and base_segmentation_path.exists():
        return curated_rosette_path, base_segmentation_path

    # Fallback: Check R3 timeseries style (masks_curated)
    curated_rosette_path_ts = (
        sample_dir / f"02_segmentation_{name}_masks_curated copy.tif"
    )
    base_segmentation_path_ts = sample_dir / f"02_segmentation_{name}_masks_curated.tif"

    if curated_rosette_path_ts.exists() and base_segmentation_path_ts.exists():
        return curated_rosette_path_ts, base_segmentation_path_ts

    # Fallback: If no 'copy' file, assume the base file is also the curated file.
    if base_segmentation_path.exists() and not curated_rosette_path.exists():
        return base_segmentation_path, base_segmentation_path
    if base_segmentation_path_ts.exists() and not curated_rosette_path_ts.exists():
        return base_segmentation_path_ts, base_segmentation_path_ts

    # --- ADDED NEW FALLBACK ---
    # Final fallback for R3 naming (e.g. 04_...rosettes.tif and 02_...cells.tif)
    curated_final = sample_dir / f"04_curated_{name}_rosettes_curated.tif"
    base_final = sample_dir / f"02_segmentation_{name}_cells.tif"
    if curated_final.exists() and base_final.exists():
        return curated_final, base_final
    # --- END NEW FALLBACK ---

    return None, None


def find_curated_file_manual(sample_dir: Path) -> tuple[Path | None, Path | None]:
    """Finds (curated_rosette_mask, base_cell_mask)"""

    # 1. Find the curated file (e.g., .../T0_2_1_seg copy.tif)
    curated_file = sample_dir / f"{sample_dir.stem}.tif"
    if not curated_file.exists():
        return None, None  # If no curated file, fail

    # 2. Find the base file (e.g., .../T0_2_1_seg.tif)
    # This replaces " copy.tif" with ".tif", which works for spaced filenames
    base_file_name = curated_file.name.replace(" copy.tif", ".tif")
    base_seg_file = sample_dir / base_file_name

    if base_seg_file.exists() and base_seg_file != curated_file:
        # Found both!
        return curated_file, base_seg_file
    else:
        # Fallback: if no base file is found, return None for the base path
        return curated_file, None


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
    global_root,  # Added global root
):
    ts_r1r2_path = Path(ts_r1r2_root)
    ts_r3_path = Path(ts_r3_root)
    ts_manual_path = Path(ts_manual_root)
    inhib_r1r2_path = Path(inhib_r1r2_root)
    inhib_r3_path = Path(inhib_r3_root)
    inhib_qvd_path = Path(inhib_qvd_root)
    constant_fed_path = Path(constant_fed_root)
    output_path = Path(output_file)
    global_root_path = Path(global_root)  # Added global root

    r1r2_known_roots = [ts_r1r2_path, ts_manual_path]

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

    # Helper function to process paths and add to list
    def process_and_add(data_dict, curated_file, base_seg_file):
        if curated_file:
            data_dict["curated_rosette_path"] = get_relative_path_safe(
                curated_file, global_root_path
            )
            data_dict["base_segmentation_path"] = (
                get_relative_path_safe(base_seg_file, global_root_path)
                if base_seg_file
                else "NA"
            )
            all_curated_data.append(data_dict)

    def get_relative_path_safe(file_path, root_path):
        """Calculates path relative to root, with fallback to absolute."""
        try:
            # Try to make path relative
            return str(file_path.relative_to(root_path))
        except ValueError:
            # If it fails (e.g., on different drive or outside root), use absolute path
            return str(file_path.resolve())

    # --- Part 1: Timeseries R1/R2 ---
    print(f"Scanning Timeseries R1/R2 data in: {ts_r1r2_path}...")
    for sample_dir in ts_r1r2_path.iterdir():
        if not sample_dir.is_dir():
            continue
        data_dict = parse_timeseries_folder_name(sample_dir, r1r2_known_roots)
        if data_dict:
            curated_file, base_seg_file = find_curated_file_r1r2(sample_dir)
            process_and_add(data_dict, curated_file, base_seg_file)

    # --- Part 2: Timeseries R3 ---
    print(f"Scanning Timeseries R3 data in: {ts_r3_path}...")
    for timepoint_dir in ts_r3_path.iterdir():
        if not timepoint_dir.is_dir():
            continue
        for sample_dir in timepoint_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            data_dict = parse_timeseries_folder_name(sample_dir, r1r2_known_roots)
            if data_dict:
                curated_file, base_seg_file = find_curated_file_r3_style(sample_dir)
                process_and_add(data_dict, curated_file, base_seg_file)

    # --- Part 3: Inhibitor R1/R2 ---
    print(f"Scanning Inhibitor R1/R2 data in: {inhib_r1r2_path}...")
    for treatment_dir in inhib_r1r2_path.iterdir():
        if not treatment_dir.is_dir():
            continue
        for sample_dir in treatment_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            data_dict = parse_inhibitor_folder_name(sample_dir)
            if data_dict:
                curated_file, base_seg_file = find_curated_file_r1r2(sample_dir)
                process_and_add(data_dict, curated_file, base_seg_file)

    # --- Part 4: Inhibitor R3 ---
    print(f"Scanning Inhibitor R3 data in: {inhib_r3_path}...")
    for treatment_dir in inhib_r3_path.iterdir():
        if not treatment_dir.is_dir():
            continue
        for sample_dir in treatment_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            data_dict = parse_inhibitor_folder_name(sample_dir)
            if data_dict:
                curated_file, base_seg_file = find_curated_file_r3_style(sample_dir)
                process_and_add(data_dict, curated_file, base_seg_file)

    # --- Part 5: Inhibitor QVD ---
    print(f"Scanning Inhibitor QVD data in: {inhib_qvd_path}...")
    for treatment_dir in inhib_qvd_path.iterdir():
        if not treatment_dir.is_dir():
            continue
        for sample_dir in treatment_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            data_dict = parse_inhibitor_folder_name(sample_dir)
            if data_dict:
                curated_file, base_seg_file = find_curated_file_r3_style(sample_dir)
                process_and_add(data_dict, curated_file, base_seg_file)

    # --- Part 6: Timeseries MANUAL ---
    print(f"Scanning Timeseries Manual data in: {ts_manual_path}...")
    for sample_dir in ts_manual_path.iterdir():
        if not sample_dir.is_dir():
            continue
        data_dict = parse_timeseries_folder_name(sample_dir, r1r2_known_roots)
        if not data_dict:
            continue

        curated_file, base_seg_file = None, None
        if ".svg" in sample_dir.name:
            curated_file, base_seg_file = find_curated_file_manual(sample_dir)
        else:
            curated_file, base_seg_file = find_curated_file_r3_style(sample_dir)
            if not curated_file:  # Fallback to r1r2
                curated_file, base_seg_file = find_curated_file_r1r2(sample_dir)

        process_and_add(data_dict, curated_file, base_seg_file)

    # --- Part 7: CONSTANT FED ---
    print(f"Scanning Constant Fed data in: {constant_fed_path}...")
    for item in constant_fed_path.iterdir():
        data_dict = parse_constant_fed_name(item)
        if not data_dict:
            continue

        curated_file, base_seg_file = None, None
        if item.is_dir():
            curated_file, base_seg_file = find_curated_file_r3_style(item)
            if not curated_file:
                curated_file, base_seg_file = find_curated_file_r1r2(item)
        elif item.is_file() and item.suffix == ".tif":
            # This logic is tricky. Assume file is curated, but can't find base.
            curated_file = item
            base_seg_file = None

        process_and_add(data_dict, curated_file, base_seg_file)

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
        "base_segmentation_path",  # <-- NEW COLUMN ADDED
    ]

    for col in all_columns:
        if col not in df.columns:
            df[col] = "NA"

    df = df[all_columns]

    pre_dedupe_len = len(df)
    df = df.drop_duplicates(subset=["unique_sample_id"], keep="first")
    post_dedupe_len = len(df)

    if pre_dedupe_len != post_dedupe_len:
        print(f"\nWarning: Removed {pre_dedupe_len - post_dedupe_len} duplicate IDs.")

    df = df.sort_values(
        by=["experiment_type", "replicate", "timepoint", "animal_id", "image_id"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)

    print(f"\n✅ Success! Curated metadata file saved to: {output_path}")
    print("\n--- Example Rows (Note the new 'base_segmentation_path' column) ---")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metadata TSV for ALL curated rosette segmentation masks."
    )
    # --- ADDED GLOBAL ROOT ---
    parser.add_argument(
        "--global-root",
        required=True,
        help="The absolute root path for all data, to make file paths relative.",
    )
    # Timeseries
    parser.add_argument("--ts-r1r2-root", required=True)
    parser.add_argument("--ts-r3-root", required=True)
    parser.add_argument("--ts-manual-root", required=True)
    # Inhibitor
    parser.add_argument("--inhib-r1r2-root", required=True)
    parser.add_argument("--inhib-r3-root", required=True)
    parser.add_argument("--inhib-qvd-root", required=True)
    # Constant Fed
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
        args.global_root,  # <-- PASSING NEW ARG
    )


# python3 scripts/curated_rosette_metadata_init_segmentation.py \
#     --global-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project" \
#     --ts-manual-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/4_manually_annotated_rosettes_training_dataset/241120_timeseries_Rosette_labels_1st_half_manually_annotated" \
#     --ts-r1r2-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/7_final_segmentation_rosettes_R1_R2/250206_Rosette_timeseries_image_stacking_vf/human_in_the_loop" \
#     --ts-r3-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/8_segmentation_&_rosettes_R3/250707_timeseries_live_R3_seg" \
#     --inhib-r1r2-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250212_Rosette_inhibitor_R1_R2/rosette_inhibitor_human_in_the_loop_data" \
#     --inhib-r3-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250714_inhibitors_R3_seg" \
#     --inhib-qvd-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/inhibitors/2_segmentation_&_rosettes/250723_qvd_inhibitor_seg" \
#     --constant-fed-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/constant_fed/2_250720_constant_fed_seg" \
#     --output-file "config/curated_rosette_metadata.tsv"
