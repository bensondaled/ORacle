#!/usr/bin/env python3
"""
Compute normalization statistics (min/max) across all institution files.

Efficient single-pass streaming approach:
- Loads one feather file at a time
- Tracks running min/max per feature
- Saves to JSON for use during training

Usage:
    python compute_normalization_stats.py --data-dir /path/to/institutions --output stats.json
    python compute_normalization_stats.py --data-dir /path/to/institutions --train-institutions 1001,1002,1003
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


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

# Clinical fallback ranges (used if data is missing or has issues)
CLINICAL_FALLBACKS = {
    'phys_bp_sys_non_invasive': {'min': 40, 'max': 200},
    'phys_bp_dias_non_invasive': {'min': 20, 'max': 120},
    'phys_bp_mean_non_invasive': {'min': 30, 'max': 150},
    'phys_spo2_%': {'min': 70, 'max': 100},
    'phys_spo2_pulse_rate': {'min': 20, 'max': 200},
    'phys_end_tidal_co2_(mmhg)': {'min': 10, 'max': 80},
    'phys_sevoflurane_exp_%': {'min': 0, 'max': 8},
    'phys_isoflurane_exp_%': {'min': 0, 'max': 4},
    'phys_desflurane_exp_%': {'min': 0, 'max': 18},
    'phys_nitrous_exp_%': {'min': 0, 'max': 80},
}


def find_institution_files(data_dir: str, pattern: str = "institution_*.feather") -> List[Path]:
    """Find all institution feather files in directory."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob(pattern))
    if not files:
        # Try alternative patterns
        files = sorted(data_path.glob("*.feather"))
    return files


def compute_stats_streaming(
    data_dir: str,
    institutions: Optional[List[int]] = None,
    columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute min/max statistics by streaming through files.

    Memory efficient: loads one file at a time, tracks running stats.

    Args:
        data_dir: Directory containing institution feather files
        institutions: Optional list of institution numbers to include (None = all)
        columns: Columns to compute stats for (None = NORMALIZE_COLS)

    Returns:
        Dict mapping column name to {'min': float, 'max': float, 'mean': float, 'std': float}
    """
    columns = columns or NORMALIZE_COLS
    files = find_institution_files(data_dir)

    if not files:
        raise ValueError(f"No feather files found in {data_dir}")

    print(f"Found {len(files)} institution files")

    # Filter to specific institutions if requested
    if institutions:
        inst_set = set(institutions)
        filtered_files = []
        for f in files:
            # Extract institution number from filename
            try:
                inst_num = int(f.stem.split('_')[-1])
                if inst_num in inst_set:
                    filtered_files.append(f)
            except ValueError:
                continue
        files = filtered_files
        print(f"Filtered to {len(files)} files for institutions: {institutions}")

    # Initialize running statistics
    running_min = {col: np.inf for col in columns}
    running_max = {col: -np.inf for col in columns}
    running_sum = {col: 0.0 for col in columns}
    running_sum_sq = {col: 0.0 for col in columns}
    running_count = {col: 0 for col in columns}

    total_rows = 0

    # Stream through files
    for file_path in tqdm(files, desc="Computing stats"):
        try:
            df = pd.read_feather(file_path)
            total_rows += len(df)

            for col in columns:
                if col not in df.columns:
                    continue

                # Get non-null values
                values = df[col].dropna().values
                if len(values) == 0:
                    continue

                # Update running stats
                running_min[col] = min(running_min[col], values.min())
                running_max[col] = max(running_max[col], values.max())
                running_sum[col] += values.sum()
                running_sum_sq[col] += (values ** 2).sum()
                running_count[col] += len(values)

            # Free memory
            del df

        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
            continue

    print(f"\nProcessed {total_rows:,} total rows across {len(files)} files")

    # Compute final statistics
    stats = {}
    for col in columns:
        count = running_count[col]

        if count == 0:
            # Use clinical fallback
            print(f"Warning: No data for {col}, using clinical fallback")
            stats[col] = CLINICAL_FALLBACKS.get(col, {'min': 0, 'max': 1})
            stats[col]['mean'] = (stats[col]['min'] + stats[col]['max']) / 2
            stats[col]['std'] = (stats[col]['max'] - stats[col]['min']) / 4
            stats[col]['count'] = 0
        else:
            mean = running_sum[col] / count
            variance = (running_sum_sq[col] / count) - (mean ** 2)
            std = np.sqrt(max(0, variance))  # Ensure non-negative

            stats[col] = {
                'min': float(running_min[col]),
                'max': float(running_max[col]),
                'mean': float(mean),
                'std': float(std),
                'count': int(count)
            }

            # Sanity check: if min == max, expand range slightly
            if stats[col]['min'] == stats[col]['max']:
                print(f"Warning: {col} has constant value {stats[col]['min']}, expanding range")
                stats[col]['min'] -= 1
                stats[col]['max'] += 1

    return stats


def print_stats_summary(stats: Dict[str, Dict[str, float]]):
    """Print a nice summary of the statistics."""
    print("\n" + "=" * 70)
    print("NORMALIZATION STATISTICS SUMMARY")
    print("=" * 70)
    print(f"{'Column':<40} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 70)

    for col, s in stats.items():
        print(f"{col:<40} {s['min']:>10.2f} {s['max']:>10.2f} {s['mean']:>10.2f} {s['std']:>10.2f}")

    print("=" * 70)


def save_stats(stats: Dict[str, Dict[str, float]], output_path: str):
    """Save statistics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved normalization stats to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute normalization statistics across institution files')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing institution feather files')
    parser.add_argument('--output', type=str, default='normalization_stats.json',
                        help='Output JSON file path (default: normalization_stats.json)')
    parser.add_argument('--train-institutions', type=str, default=None,
                        help='Comma-separated list of institution numbers to use (default: all)')
    parser.add_argument('--columns', type=str, default=None,
                        help='Comma-separated list of columns to normalize (default: vital columns)')

    args = parser.parse_args()

    # Parse institution list
    institutions = None
    if args.train_institutions:
        institutions = [int(x.strip()) for x in args.train_institutions.split(',')]

    # Parse columns
    columns = None
    if args.columns:
        columns = [x.strip() for x in args.columns.split(',')]

    # Compute stats
    print(f"Computing normalization statistics from: {args.data_dir}")
    stats = compute_stats_streaming(args.data_dir, institutions, columns)

    # Print summary
    print_stats_summary(stats)

    # Save
    save_stats(stats, args.output)

    print("\nDone! Use these stats in your config.yaml or load in your dataset.")


if __name__ == '__main__':
    main()
