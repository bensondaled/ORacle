#!/usr/bin/env python3
"""
Compute normalization statistics and create scaled dataset.

Ultra memory-efficient approach:
  Pass 1: For each column, stream through all files reading only that column
  Pass 2: For each file, load, scale, save, free

Scales ALL numeric columns EXCEPT binary flags (columns with only 0/1 values).

Usage:
    python compute_normalization_stats.py --data-dir /path/to/institutions --output-dir /path/to/scaled
"""

import argparse
import json
import gc
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import pyarrow.feather as pf


def find_institution_files(data_dir: str) -> List[Path]:
    """Find all institution feather files."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("institution_*.feather"))
    if not files:
        files = sorted(data_path.glob("*.feather"))
    return files


# Columns that should NEVER be scaled (IDs, categorical, binary flags)
DO_NOT_SCALE = {
    # IDs / keys
    "case_id",
    "mpog_case_id",
    # Categorical
    "institution",
    "sex",
    # Binary flag columns
    "phys_bp_sys_non_invasive_flag",
    "phys_bp_dias_non_invasive_flag",
    "phys_bp_mean_non_invasive_flag",
    "phys_spo2_%_flag",
    "phys_spo2_pulse_rate_flag",
    "phys_end_tidal_co2_(mmhg)_flag",
    "phys_sevoflurane_exp_%_flag",
    "phys_isoflurane_exp_%_flag",
    "phys_desflurane_exp_%_flag",
    "phys_nitrous_exp_%_flag",
    "meds_propofol_flag",
    "meds_fentanyl_flag",
    "meds_ketamine_flag",
    "meds_dexmedetomidine_flag",
    "meds_remifentanil_flag",
    "meds_phenylephrine_flag",
    "meds_norepinephrine_flag",
    "meds_epinephrine_flag",
    "meds_vasopressin_flag",
    "meds_ephedrine_flag",
    "meds_hydromorphone_flag",
    "meds_glycopyrrolate_flag",
    "meds_etomidate_flag",
    "meds_esmolol_flag",
    "meds_labetalol_flag",
}


def detect_columns_to_scale(files: List[Path]) -> List[str]:
    """
    Detect which columns to scale by reading the first file.

    Returns all numeric columns EXCEPT those in DO_NOT_SCALE.
    """
    print("Detecting columns to scale from first file...")

    table = pf.read_table(files[0])
    df = table.to_pandas()
    del table

    columns_to_scale = []
    excluded_columns = []
    skipped_columns = []

    for col in df.columns:
        # Skip columns in the exclusion list
        if col in DO_NOT_SCALE:
            excluded_columns.append(col)
            continue

        # Skip non-numeric (handle pandas extension dtypes)
        try:
            if not np.issubdtype(df[col].dtype, np.number):
                skipped_columns.append(col)
                continue
        except TypeError:
            # Pandas extension dtype (StringDtype, etc)
            skipped_columns.append(col)
            continue

        columns_to_scale.append(col)

    del df
    gc.collect()

    print(f"  Numeric columns to scale: {len(columns_to_scale)}")
    print(f"  Excluded (IDs/categorical/flags): {len(excluded_columns)}")
    if excluded_columns:
        print(f"    -> {excluded_columns}")
    print(f"  Non-numeric (skipped): {len(skipped_columns)}")

    return columns_to_scale


def pass1_compute_minmax(files: List[Path], columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Pass 1: Compute min/max one column at a time.

    For each column, iterate through all files reading ONLY that column.
    Maximum memory = 1 column from 1 file.
    """
    print("\n" + "=" * 60)
    print("PASS 1: Computing min/max (one column at a time)")
    print("=" * 60)

    stats = {}

    for i, col in enumerate(columns):
        print(f"[{i+1}/{len(columns)}] {col}...", end=" ", flush=True)
        col_min = np.inf
        col_max = -np.inf

        for file_path in files:
            try:
                # Read ONLY this one column
                table = pf.read_table(file_path, columns=[col])
                arr = table.column(col).to_numpy()

                # Filter out nulls/nans
                if np.issubdtype(arr.dtype, np.floating):
                    valid = arr[~np.isnan(arr)]
                else:
                    valid = arr[arr != None]

                if len(valid) > 0:
                    col_min = min(col_min, float(np.min(valid)))
                    col_max = max(col_max, float(np.max(valid)))

                del table, arr, valid
                gc.collect()

            except Exception:
                continue

        # Handle edge cases
        if np.isinf(col_min) or np.isinf(col_max):
            col_min, col_max = 0.0, 1.0
            print("NO DATA")
        elif col_min == col_max:
            col_min -= 1
            col_max += 1
            print(f"constant, expanded to [{col_min:.2f}, {col_max:.2f}]")
        else:
            print(f"[{col_min:.2f}, {col_max:.2f}]")

        stats[col] = {'min': col_min, 'max': col_max}

    return stats


def pass2_scale_and_save(
    files: List[Path],
    stats: Dict[str, Dict[str, float]],
    output_dir: Path
):
    """
    Pass 2: Scale each file and save.
    """
    print("\n" + "=" * 60)
    print("PASS 2: Scaling and saving")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(stats.keys())

    for i, file_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {file_path.name}...", end=" ", flush=True)

        try:
            table = pf.read_table(file_path)
            df = table.to_pandas()
            del table

            rows = len(df)

            # Scale each column
            for col in columns:
                if col not in df.columns:
                    continue

                min_val = stats[col]['min']
                max_val = stats[col]['max']
                range_val = max_val - min_val

                if range_val > 0:
                    df[col] = (df[col] - min_val) / range_val

            # Save
            output_path = output_dir / file_path.name
            df.to_feather(output_path)

            print(f"{rows:,} rows")

            del df
            gc.collect()

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    print(f"\nScaled files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--stats-file', type=str, default=None)
    parser.add_argument('--skip-scaling', action='store_true')

    args = parser.parse_args()

    files = find_institution_files(args.data_dir)
    if not files:
        print(f"ERROR: No feather files found in {args.data_dir}")
        return 1

    print(f"Found {len(files)} files in {args.data_dir}")

    # Auto-detect columns to scale
    columns = detect_columns_to_scale(files)

    # Pass 1
    stats = pass1_compute_minmax(files, columns)

    # Save stats
    if args.stats_file:
        with open(args.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved stats to: {args.stats_file}")

    # Pass 2
    if not args.skip_scaling:
        pass2_scale_and_save(files, stats, Path(args.output_dir))

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
