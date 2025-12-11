#!/usr/bin/env python3
"""
Reconcile Curated Metadata IDs
==============================

Matches samples in the 'curated' metadata to samples in the 'raw' metadata
and updates the curated unique_sample_id to match the raw format.

This prevents having to re-run segmentation when IDs differ slightly
(e.g., 'R1-T0-A1-I1' vs 'R1-T0-A1-I1-PRI').
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def reconcile_ids(raw_path, curated_path, output_path):
    print("--- Metadata Reconciliation ---")
    print(f"Raw (Source of Truth): {raw_path}")
    print(f"Curated (To Update):   {curated_path}")

    # 1. Load Dataframes
    try:
        raw_df = pd.read_csv(raw_path, sep="\t")
        cur_df = pd.read_csv(curated_path, sep="\t")
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        sys.exit(1)

    # 2. Create Mapping Dictionary from Raw Data
    # We assume Raw ID is "BASE-SUFFIX" (e.g., R1-T0-A1-I1-PRI)
    # We want to map "BASE" -> "BASE-SUFFIX"

    # Priority for suffixes if duplicates exist (e.g., if you have both PRI and MAX)
    # Lower index = higher priority
    SUFFIX_PRIORITY = ["PRI", "MAX", "ZPROJ", "SINGLE"]

    mapping = {}

    # Sort raw_df so high priority suffixes appear first
    # We creates a temp column for sorting priority
    def get_priority(uid):
        parts = uid.rsplit("-", 1)
        if len(parts) < 2:
            return 99
        suffix = parts[1]
        if suffix in SUFFIX_PRIORITY:
            return SUFFIX_PRIORITY.index(suffix)
        return 99

    raw_df["priority"] = raw_df["unique_sample_id"].apply(get_priority)
    raw_df_sorted = raw_df.sort_values("priority")

    print(f"Building map from {len(raw_df)} raw samples...")

    for _, row in raw_df_sorted.iterrows():
        raw_id = row["unique_sample_id"]

        # Logic: Split off the suffix to get the base
        # "R1-T0-A1-I1-PRI" -> "R1-T0-A1-I1"
        if "-" in raw_id:
            base_id = raw_id.rsplit("-", 1)[0]

            # Only map if not already mapped (preserves priority sorting)
            if base_id not in mapping:
                mapping[base_id] = raw_id

            # Also map the raw_id to itself just in case
            if raw_id not in mapping:
                mapping[raw_id] = raw_id

    # 3. Update Curated Dataframe
    print(f"Updating {len(cur_df)} curated samples...")

    updated_count = 0
    missing_count = 0

    new_ids = []

    for idx, row in cur_df.iterrows():
        cur_id = row["unique_sample_id"]

        # Try direct match first
        if cur_id in mapping:
            new_ids.append(mapping[cur_id])
            if mapping[cur_id] != cur_id:
                updated_count += 1
        else:
            # If not found, check if the curated ID *is* the base ID
            # (The loop above mapped base_id -> raw_id)
            if cur_id in mapping:
                new_ids.append(mapping[cur_id])
                updated_count += 1
            else:
                print(f"⚠️  No match found for curated ID: {cur_id}")
                new_ids.append(cur_id)  # Keep original
                missing_count += 1

    cur_df["unique_sample_id"] = new_ids

    # 4. Save Result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cur_df.to_csv(output_path, sep="\t", index=False)

    print("-" * 40)
    print("✅ Process Complete.")
    print(f"   Updated IDs: {updated_count}")
    print(f"   Unmatched:   {missing_count}")
    print(f"   Saved to:    {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Curated TSV IDs to Raw TSV IDs")
    parser.add_argument("--raw", required=True, help="Path to sample_metadata.tsv")
    parser.add_argument(
        "--curated", required=True, help="Path to curated_rosette_metadata.tsv"
    )
    parser.add_argument("--output", required=True, help="Path for the fixed output TSV")

    args = parser.parse_args()

    reconcile_ids(args.raw, args.curated, args.output)
