#!/usr/bin/env python3

"""
Generates a master metadata TSV file for the segmentation pipeline. (v8)

This script walks through the 'timeseries_raw_root', 'inhibitor_raw_root',
and 'constant_fed_raw_root' directories, parses all metadata from the
complex file and directory names, and creates a single 'sample_metadata.tsv' file.

This script is intended to be run once to bootstrap the metadata file.

v8 changes:
- (Fix) `create_unique_id` no longer creates "RR1" or "TT0".
- (Fix) All `parse_..._name` functions now pass the file *stem* to
  `parse_image_type`, allowing it to correctly identify z-stacks.
- (New) Added a duplicate-skipping set to the `main` function to
  robustly handle any remaining duplicates by "keeping the first one."
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# --- Regex Patterns ---
INHIBITOR_PATTERN = re.compile(
    r"^(R\d)(?:_(T\d+))?_(.*?)_(\d+)(?:_(\d+))?\.tif$", re.IGNORECASE
)
TIMESERIES_PATTERN = re.compile(r"(?:(R\d)_)?(T\d+)_(\d+)(?:_(\d+))?", re.IGNORECASE)
CONSTANT_FED_PATTERN = re.compile(
    r"^CF_(R\d)_(D\d+)_(\d+)(?:_(\d+))?\.tif$", re.IGNORECASE
)


# =============================================================================
# create_unique_id FUNCTION (FIXED)
# =============================================================================
def create_unique_id(
    exp_type,
    replicate,
    timepoint,
    animal_id,
    image_id,
    image_type,
    treatment="none",
    is_control=False,
):
    """
    Creates a clean, standardized, and unique sample ID. (v3)
    """
    # 1. Sanitize base components
    #    (Strips R, T, D to prevent RR1, TT0, etc.)
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
    if sane_treatment in ["NONE", "NA", ""]:
        sane_treatment = None

    if is_control and sane_treatment:
        sane_treatment = f"{sane_treatment}_CTRL"  # e.g., U0126_CTRL

    # 4. Abbreviate image_type
    img_type_map = {
        "primary": "PRI",
        "max_projection": "MAX",
        "z_projection": "ZPROJ",
        "single_stack": "SINGLE",
    }
    img_type_abbrev = img_type_map.get(image_type, image_type.upper())

    # 5. Build base ID string
    base_id_parts = [rep_str, tp]
    if sane_treatment:
        base_id_parts.append(sane_treatment)
    base_id_parts.extend([f"A{anim}", f"I{img}"])

    base_id = "-".join(base_id_parts)

    # 6. Return final unique ID
    return f"{base_id}-{img_type_abbrev}"


# =============================================================================


def parse_image_type(filename_stem: str) -> str:
    """
    Parses the 'image_type' based on keywords in the filename ST.
    """
    stem_lower = filename_stem.lower()

    if "max_" in stem_lower:
        return "max_projection"
    if "projection_z" in stem_lower:
        return "z_projection"
    if "singlestack_z" in stem_lower or "siglestack_z" in stem_lower:
        return "single_stack"
    if re.search(r"_z\d+$", stem_lower):  # Catch T0_9_3_z3
        return "single_stack"
    return "primary"


def parse_image_id(id_match: str | None) -> str:
    """
    Implements the 'implicit 0' logic for the image_id.
    """
    if id_match:
        return str(int(id_match))
    else:
        return "0"


def parse_inhibitor_name(filename: str, file_path: Path) -> dict | None:
    """
    Parses metadata from an INHIBITOR filename.
    """
    # Get stem *before* matching
    file_stem = file_path.stem

    match = INHIBITOR_PATTERN.match(filename)
    if not match:
        return None

    replicate, timepoint_in_filename, treatment_str, animal_id, image_id_suffix = (
        match.groups()
    )

    timepoint = timepoint_in_filename
    if not timepoint:
        for parent_dir in [
            file_path.parent,
            file_path.parent.parent,
            file_path.parent.parent.parent,
        ]:
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
    # --- FIXED: Pass file_stem, not filename ---
    image_type = parse_image_type(file_stem)

    unique_id = create_unique_id(
        "inhibitor",
        replicate,
        safe_timepoint,
        animal_id,
        image_id,
        image_type,
        treatment=safe_treatment,
        is_control=is_control,
    )

    return {
        "unique_sample_id": unique_id,
        "raw_image_id": file_stem,
        "experiment_type": "inhibitor",
        "replicate": replicate.upper(),
        "timepoint": safe_timepoint,
        "animal_id": str(int(animal_id)),
        "image_id": image_id,
        "image_type": image_type,
        "treatment": safe_treatment,
        "is_control": is_control,
        "raw_image_path": file_path,
    }


def parse_timeseries_name(filename: str, file_path: Path) -> dict | None:
    """
    Parses metadata from a TIMESERIES filename and its directory.
    """
    # Get stem *before* matching
    file_stem = file_path.stem

    # --- FIXED: Search the stem, not the full filename ---
    match = TIMESERIES_PATTERN.search(file_stem)
    if not match:
        return None

    replicate_in_filename, timepoint, animal_id, image_id_suffix = match.groups()

    replicate = None
    if replicate_in_filename:
        replicate = replicate_in_filename
    else:
        for parent_dir in [
            file_path.parent,
            file_path.parent.parent,
            file_path.parent.parent.parent,
        ]:
            replicate_match = re.search(r"(R\d)", parent_dir.name, re.IGNORECASE)
            if replicate_match:
                replicate = replicate_match.group(1)
                break

    if not replicate:
        return None

    image_id = parse_image_id(image_id_suffix)
    # --- FIXED: Pass file_stem, not filename ---
    image_type = parse_image_type(file_stem)

    unique_id = create_unique_id(
        "timeseries",
        replicate,
        timepoint,
        animal_id,
        image_id,
        image_type,
    )

    return {
        "unique_sample_id": unique_id,
        "raw_image_id": file_stem,
        "experiment_type": "timeseries",
        "replicate": replicate.upper(),
        "timepoint": timepoint.upper(),
        "animal_id": str(int(animal_id)),
        "image_id": image_id,
        "image_type": image_type,
        "treatment": "none",
        "is_control": False,
        "raw_image_path": file_path,
    }


def parse_constant_fed_name(filename: str, file_path: Path) -> dict | None:
    """
    Parses metadata from a CONSTANT_FED filename.
    """
    # Get stem *before* matching
    file_stem = file_path.stem

    match = CONSTANT_FED_PATTERN.match(filename)
    if not match:
        return None

    replicate, day, animal_id, image_id_suffix = match.groups()
    image_id = parse_image_id(image_id_suffix)
    # --- FIXED: Pass file_stem, not filename ---
    image_type = parse_image_type(file_stem)

    unique_id = create_unique_id(
        "constant_fed",
        replicate,
        day,
        animal_id,
        image_id,
        image_type,
    )

    return {
        "unique_sample_id": unique_id,
        "raw_image_id": file_stem,
        "experiment_type": "constant_fed",
        "replicate": replicate.upper(),
        "timepoint": day.upper(),
        "animal_id": str(int(animal_id)),
        "image_id": image_id,
        "image_type": image_type,
        "treatment": "none",
        "is_control": False,
        "raw_image_path": file_path,
    }


def main(timeseries_root, inhibitor_root, constant_fed_root, output_file):
    """
    Main execution function.
    """
    timeseries_path = Path(timeseries_root)
    inhibitor_path = Path(inhibitor_root)
    constant_fed_path = Path(constant_fed_root)
    output_path = Path(output_file)

    # Check paths
    if not timeseries_path.is_dir():
        print(f"Error: Timeseries root not found: {timeseries_path}", file=sys.stderr)
        sys.exit(1)
    if not inhibitor_path.is_dir():
        print(f"Error: Inhibitor root not found: {inhibitor_path}", file=sys.stderr)
        sys.exit(1)
    if not constant_fed_path.is_dir():
        print(
            f"Error: Constant Fed root not found: {constant_fed_path}", file=sys.stderr
        )
        sys.exit(1)

    all_samples_data = []
    # --- NEW: Set to track duplicates ---
    seen_unique_ids = set()

    # --- Part 1: Process Timeseries Data ---
    print(f"Scanning Timeseries data in: {timeseries_path}...")
    for file_path in sorted(
        timeseries_path.rglob("*.tif")
    ):  # Sort for consistent "first"
        sample_data = parse_timeseries_name(file_path.name, file_path)
        if sample_data:
            # --- DUPLICATE CHECK ---
            unique_id = sample_data["unique_sample_id"]
            if unique_id in seen_unique_ids:
                print(f"  [Skipping duplicate ID]: {file_path.name} (ID: {unique_id})")
                continue
            seen_unique_ids.add(unique_id)
            # --- END CHECK ---
            sample_data["raw_image_path"] = file_path.relative_to(timeseries_path)
            all_samples_data.append(sample_data)
        else:
            print(
                f"  [Skipping Timeseries file]: {file_path.name} (No valid pattern found)"
            )

    # --- Part 2: Process Inhibitor Data ---
    print(f"Scanning Inhibitor data in: {inhibitor_path}...")
    for file_path in sorted(inhibitor_path.rglob("*.tif")):
        sample_data = parse_inhibitor_name(file_path.name, file_path)
        if sample_data:
            # --- DUPLICATE CHECK ---
            unique_id = sample_data["unique_sample_id"]
            if unique_id in seen_unique_ids:
                print(f"  [Skipping duplicate ID]: {file_path.name} (ID: {unique_id})")
                continue
            seen_unique_ids.add(unique_id)
            # --- END CHECK ---
            sample_data["raw_image_path"] = file_path.relative_to(inhibitor_path)
            all_samples_data.append(sample_data)
        else:
            print(
                f"  [Skipping Inhibitor file]: {file_path.name} (No valid pattern found)"
            )

    # --- Part 3: Process Constant Fed Data ---
    print(f"Scanning Constant Fed data in: {constant_fed_path}...")
    for file_path in sorted(constant_fed_path.rglob("*.tif")):
        sample_data = parse_constant_fed_name(file_path.name, file_path)
        if sample_data:
            # --- DUPLICATE CHECK ---
            unique_id = sample_data["unique_sample_id"]
            if unique_id in seen_unique_ids:
                print(f"  [Skipping duplicate ID]: {file_path.name} (ID: {unique_id})")
                continue
            seen_unique_ids.add(unique_id)
            # --- END CHECK ---
            sample_data["raw_image_path"] = file_path.relative_to(constant_fed_path)
            all_samples_data.append(sample_data)
        else:
            print(
                f"  [Skipping Constant Fed file]: {file_path.name} (No valid pattern found)"
            )

    if not all_samples_data:
        print(
            "Error: No valid .tif files were found. Check paths and naming.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Part 4: Create and Save DataFrame ---
    print(f"\nFound {len(all_samples_data)} valid samples.")

    all_columns = [
        "unique_sample_id",
        "raw_image_id",
        "experiment_type",
        "replicate",
        "timepoint",
        "animal_id",
        "image_id",
        "image_type",
        "treatment",
        "is_control",
        "raw_image_path",
        "curated_image_id",
        "curated_image_path",
        "segmentation_path",
        "notes",
    ]

    df = pd.DataFrame(all_samples_data)

    for col in all_columns:
        if col not in df.columns:
            df[col] = "NA"

    df = df[all_columns]

    # --- FINAL DUPLICATE CHECK (This should now pass) ---
    duplicates = df[df.duplicated(subset=["unique_sample_id"], keep=False)]
    if not duplicates.empty:
        print("\n\n" + "=" * 80)
        print(f"🚨 ERROR: Found {len(duplicates)} duplicate unique_sample_ids!")
        print("This should not happen with the 'seen_unique_ids' set. Check logic.")
        print(duplicates.sort_values("unique_sample_id"))
        print("=" * 80 + "\n")
        sys.exit(1)
    else:
        print("✅ All new unique_sample_ids are unique.")
    # --- END DUPLICATE CHECK ---

    df = df.sort_values(
        by=["experiment_type", "replicate", "timepoint", "animal_id", "image_id"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, sep="\t", index=False)

    print(f"\n✅ Success! Metadata file saved to: {output_path}")
    print("\n--- Example Rows (Timeseries) ---")
    print(df[df["experiment_type"] == "timeseries"].head())
    print("\n--- Example Rows (Inhibitor) ---")
    print(df[df["experiment_type"] == "inhibitor"].head())
    print("\n--- Example Rows (Constant Fed) ---")
    print(df[df["experiment_type"] == "constant_fed"].head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metadata TSV for segmentation pipeline."
    )
    parser.add_argument(
        "--timeseries-root",
        required=True,
        help="Root directory for 'timeseries/1_raw' data.",
    )
    parser.add_argument(
        "--inhibitor-root",
        required=True,
        help="Root directory for 'inhibitors/1_raw' data.",
    )
    parser.add_argument(
        "--constant-fed-root",
        required=True,
        help="Root directory for 'constant_fed' raw data.",
    )
    parser.add_argument(
        "--output-file",
        default="config/sample_metadata.tsv",
        help="Path to save the output TSV file.",
    )

    args = parser.parse_args()

    main(
        args.timeseries_root,
        args.inhibitor_root,
        args.constant_fed_root,
        args.output_file,
    )
# python scripts/generate_metadata.py \
# --timeseries-root "/Volumes/FELLES/SENTRE/SARS/Sars S12/Ines/segmentation project/timeseries/1_raw" \
# --inhibitor-root "/Volumes/FELLES/SENTRE/SARS/S12/Ines/segmentation project/inhibitors/1_raw" \
# --constant-fed-root "/Volumes/FELLES/SENTRE/SARS/S12/Ines/segmentation project/constant_fed" \
# --output-file "config/sample_metadata.tsv"
