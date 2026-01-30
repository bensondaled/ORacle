#!/usr/bin/env python3
"""
Compute normalization statistics and create scaled dataset.

Two-pass memory-efficient approach:
  Pass 1: Stream through files one at a time to find global min/max
  Pass 2: Apply scaling to each file and save scaled versions

Usage:
    python compute_normalization_stats.py --data-dir /path/to/institutions --output-dir /path/to/scaled
"""

import argparse
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# Columns to normalize (vitals + targets)
NORMALIZE_COLS = [
    # Blood pressure
    'phys_bp_sys_non_invasive',
    'phys_bp_dias_non_invasive',
    'phys_bp_mean_non_invasive',
    # Oxygenation
    'phys_spo2_%',
    'phys_spo2_pulse_rate',
    # Respiratory
    'phys_end_tidal_co2_(mmhg)',
    # Anesthetic gases
    'phys_sevoflurane_exp_%',
    'phys_isoflurane_exp_%',
    'phys_desflurane_exp_%',
    'phys_nitrous_exp_%',
]


def find_institution_files(data_dir: str) -> List[Path]:
    """Find all institution feather files in directory."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("institution_*.feather"))
    if not files:
        files = sorted(data_path.glob("*.feather"))
    return files


def pass1_compute_minmax(files: List[Path], columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Pass 1: Stream through all files to compute global min/max.

    Loads one file at a time to minimize memory usage.
    """
    print("=" * 60)
    print("PASS 1: Computing global min/max statistics")
    print("=" * 60)

    running_min = {col: np.inf for col in columns}
    running_max = {col: -np.inf for col in columns}

    for i, file_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] Reading {file_path.name}...", end=" ", flush=True)

        try:
            df = pd.read_feather(file_path)
            rows = len(df)

            for col in columns:
                if col not in df.columns:
                    continue
                values = df[col].dropna().values
                if len(values) == 0:
                    continue
                running_min[col] = min(running_min[col], float(values.min()))
                running_max[col] = max(running_max[col], float(values.max()))

            print(f"{rows:,} rows")

            # Free memory explicitly
            del df
            gc.collect()

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Build stats dict
    stats = {}
    print("\n" + "-" * 60)
    print(f"{'Column':<40} {'Min':>10} {'Max':>10}")
    print("-" * 60)

    for col in columns:
        min_val = running_min[col]
        max_val = running_max[col]

        # Handle edge cases
        if np.isinf(min_val) or np.isinf(max_val):
            print(f"{col:<40} {'NO DATA':>10} {'NO DATA':>10}")
            min_val, max_val = 0.0, 1.0
        elif min_val == max_val:
            print(f"{col:<40} {min_val:>10.2f} {max_val:>10.2f} (constant, expanding)")
            min_val -= 1
            max_val += 1
        else:
            print(f"{col:<40} {min_val:>10.2f} {max_val:>10.2f}")

        stats[col] = {'min': min_val, 'max': max_val}

    print("-" * 60)
    return stats


def pass2_scale_and_save(
    files: List[Path],
    stats: Dict[str, Dict[str, float]],
    columns: List[str],
    output_dir: Path
):
    """
    Pass 2: Apply min-max scaling to each file and save.

    Loads one file at a time, scales, saves, frees memory.
    """
    print("\n" + "=" * 60)
    print("PASS 2: Scaling and saving files")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, file_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing {file_path.name}...", end=" ", flush=True)

        try:
            df = pd.read_feather(file_path)
            rows = len(df)

            # Apply min-max scaling to each column
            for col in columns:
                if col not in df.columns:
                    continue
                if col not in stats:
                    continue

                min_val = stats[col]['min']
                max_val = stats[col]['max']
                range_val = max_val - min_val

                if range_val > 0:
                    df[col] = (df[col] - min_val) / range_val
                else:
                    df[col] = 0.0

            # Save scaled file
            output_path = output_dir / file_path.name
            df.to_feather(output_path)

            print(f"{rows:,} rows -> {output_path.name}")

            # Free memory
            del df
            gc.collect()

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    print("-" * 60)
    print(f"Scaled files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute normalization stats and create scaled dataset'
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Directory containing institution feather files'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory to save scaled feather files'
    )
    parser.add_argument(
        '--stats-file', type=str, default=None,
        help='Path to save/load normalization stats JSON (optional)'
    )
    parser.add_argument(
        '--skip-scaling', action='store_true',
        help='Only compute stats, skip pass 2 scaling'
    )

    args = parser.parse_args()

    # Find files
    files = find_institution_files(args.data_dir)
    if not files:
        print(f"ERROR: No feather files found in {args.data_dir}")
        return 1

    print(f"Found {len(files)} institution files in {args.data_dir}")
    print()

    # Pass 1: Compute min/max
    stats = pass1_compute_minmax(files, NORMALIZE_COLS)

    # Save stats if requested
    if args.stats_file:
        stats_path = Path(args.stats_file)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved stats to: {stats_path}")

    # Pass 2: Scale and save
    if not args.skip_scaling:
        output_dir = Path(args.output_dir)
        pass2_scale_and_save(files, stats, NORMALIZE_COLS, output_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
