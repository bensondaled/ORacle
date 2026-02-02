#!/usr/bin/env python3
"""
Data loading utilities for institution-based scaling experiments.

Handles:
- Institution metadata (region, size)
- Balanced institution selection by region
- Train/val splitting by case
- Debug mode (1% sampling)
"""

import pandas as pd
import pyarrow.feather as pf
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
import numpy as np

# =============================================================================
# Institution Metadata
# =============================================================================

# Fixed test set: 1 medium-to-large institution per region
TEST_INSTITUTIONS = {1004, 1016, 1018, 1027}

# Institution metadata: {institution_id: (region, n_cases)}
INSTITUTION_METADATA = {
    1001: ("South", 517704),
    1002: ("West", 129574),
    1003: ("South", 182684),
    1004: ("South", 303621),      # TEST
    1005: ("Midwest", 46926),
    1006: ("South", 252639),
    1007: ("Midwest", 27387),
    1008: ("Midwest", 419432),
    1009: ("Midwest", 211315),
    1010: ("Midwest", 232587),
    1011: ("Midwest", 148370),
    1012: ("Midwest", 77045),
    1013: ("Midwest", 152858),
    1014: ("Northeast", 245798),
    1015: ("Midwest", 133491),
    1016: ("Midwest", 331379),    # TEST
    1017: ("Northeast", 69570),
    1018: ("Northeast", 317516),  # TEST
    1019: ("South", 78408),
    1020: ("Midwest", 230152),
    1021: ("Northeast", 308562),
    1022: ("Midwest", 131931),
    1023: ("Midwest", 92358),
    1024: ("South", 294783),
    1025: ("West", 1149),
    1026: ("Midwest", 124713),
    1027: ("West", 259585),       # TEST
    1028: ("Northeast", 501323),
    1029: ("Midwest", 78689),
    1030: ("West", 7042),
    1031: ("Northeast", 17427),
    1032: ("Northeast", 102743),
    1033: ("West", 218008),
    1034: ("Midwest", 71893),
    1035: ("South", 154858),
    1036: ("Midwest", 207194),
    1037: ("Northeast", 418300),
    1038: ("Northeast", 75704),
    1039: ("South", 207439),
    1040: ("Midwest", 21687),
    1041: ("Midwest", 385489),
    1042: ("Midwest", 71258),
    1043: ("South", 155174),
    1044: ("West", 292489),
    1045: ("Midwest", 134536),
    1046: ("Midwest", 88426),
    1047: ("Northeast", 246704),
    1048: ("Midwest", 91289),
    1049: ("Midwest", 50843),
    1050: ("Midwest", 67255),
    1051: ("South", 103087),
    1052: ("Northeast", 331455),
    1053: ("Midwest", 144494),
    1054: ("Midwest", 446521),
    1055: ("Midwest", 325640),
    1056: ("West", 258014),
    1057: ("West", 270591),
    1058: ("South", 405332),
    1059: ("Midwest", 145394),
    1060: ("Northeast", 150082),
    1061: ("Northeast", 379853),
    1062: ("Northeast", 105702),
    1063: ("South", 136558),
    1064: ("Northeast", 160363),
    1065: ("Midwest", 54353),
    1066: ("Midwest", 166572),
    1067: ("South", 119052),
    1068: ("South", 180954),
    1069: ("Midwest", 63081),
    1070: ("Midwest", 64561),
    1071: ("Midwest", 49903),
}

# Available institutions for training (excluding test set)
AVAILABLE_INSTITUTIONS = set(INSTITUTION_METADATA.keys()) - TEST_INSTITUTIONS

# Region counts (excluding test institutions)
REGIONS = ["South", "Midwest", "Northeast", "West"]


def get_institutions_by_region(exclude_test: bool = True) -> Dict[str, List[Tuple[int, int]]]:
    """
    Group institutions by region, sorted by size (descending).

    Returns:
        Dict mapping region -> list of (institution_id, n_cases) sorted by size
    """
    by_region = {r: [] for r in REGIONS}

    for inst_id, (region, n_cases) in INSTITUTION_METADATA.items():
        if exclude_test and inst_id in TEST_INSTITUTIONS:
            continue
        by_region[region].append((inst_id, n_cases))

    # Sort each region by size descending
    for region in by_region:
        by_region[region].sort(key=lambda x: x[1], reverse=True)

    return by_region


def select_institutions(n: int, seed: int = 42) -> List[int]:
    """
    Select n institutions balanced across regions, prioritizing larger ones.

    Strategy:
    - Compute target count per region proportional to available institutions
    - Within each region, select largest institutions first
    - Handle remainder by giving to regions with most remaining institutions

    Args:
        n: Number of institutions to select (max 67)
        seed: Random seed for any tie-breaking

    Returns:
        List of institution IDs
    """
    np.random.seed(seed)

    n = min(n, len(AVAILABLE_INSTITUTIONS))
    by_region = get_institutions_by_region(exclude_test=True)

    # Count available per region
    region_counts = {r: len(insts) for r, insts in by_region.items()}
    total_available = sum(region_counts.values())

    # Compute proportional targets
    targets = {}
    allocated = 0
    for region in REGIONS:
        prop = region_counts[region] / total_available
        targets[region] = int(n * prop)
        allocated += targets[region]

    # Distribute remainder to regions with most institutions
    remainder = n - allocated
    sorted_regions = sorted(REGIONS, key=lambda r: region_counts[r], reverse=True)
    for i in range(remainder):
        targets[sorted_regions[i % len(sorted_regions)]] += 1

    # Select from each region (largest first)
    selected = []
    for region in REGIONS:
        target = min(targets[region], len(by_region[region]))
        selected.extend([inst_id for inst_id, _ in by_region[region][:target]])

    return sorted(selected)


def get_institution_file(data_dir: Path, inst_id: int) -> Optional[Path]:
    """Get the feather file path for an institution."""
    # Try different naming patterns
    patterns = [
        f"institution_{inst_id}.feather",
        f"inst_{inst_id}.feather",
        f"{inst_id}.feather",
    ]

    for pattern in patterns:
        path = data_dir / pattern
        if path.exists():
            return path

    return None


def get_institution_file_paths(
    data_dir: str,
    institution_ids: List[int],
) -> List[str]:
    """
    Get file paths for specified institutions (for streaming dataset).

    Args:
        data_dir: Directory containing institution feather files
        institution_ids: List of institution IDs

    Returns:
        List of file paths that exist
    """
    data_path = Path(data_dir)
    file_paths = []

    for inst_id in institution_ids:
        file_path = get_institution_file(data_path, inst_id)
        if file_path is not None:
            file_paths.append(str(file_path))
        else:
            print(f"  Warning: No file found for institution {inst_id}")

    print(f"Found {len(file_paths)}/{len(institution_ids)} institution files")
    return file_paths


def load_institutions_data(
    data_dir: str,
    institution_ids: List[int],
    debug_frac: Optional[float] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load data from specified institutions.

    Args:
        data_dir: Directory containing institution feather files
        institution_ids: List of institution IDs to load
        debug_frac: If set, sample this fraction of cases (e.g., 0.01 for 1%)
        seed: Random seed for debug sampling

    Returns:
        Combined DataFrame with all institutions' data
    """
    data_path = Path(data_dir)
    dfs = []

    print(f"Loading {len(institution_ids)} institutions...")

    for inst_id in institution_ids:
        file_path = get_institution_file(data_path, inst_id)

        if file_path is None:
            print(f"  Warning: No file found for institution {inst_id}")
            continue

        df = pd.read_feather(file_path)

        # Debug mode: sample 1% of cases
        if debug_frac is not None and debug_frac < 1.0:
            case_ids = df['mpog_case_id'].unique()
            np.random.seed(seed + inst_id)  # Reproducible per-institution
            n_sample = max(1, int(len(case_ids) * debug_frac))
            sampled_ids = np.random.choice(case_ids, size=n_sample, replace=False)
            df = df[df['mpog_case_id'].isin(sampled_ids)]

        n_cases = df['mpog_case_id'].nunique()
        print(f"  {inst_id}: {len(df):,} rows, {n_cases:,} cases")
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No data loaded from {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(combined):,} rows, {combined['mpog_case_id'].nunique():,} cases")

    return combined


def create_train_val_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val by case ID.

    Args:
        df: Input DataFrame
        val_frac: Fraction of cases for validation
        seed: Random seed

    Returns:
        (train_df, val_df)
    """
    np.random.seed(seed)

    case_ids = df['mpog_case_id'].unique()
    n_val = int(len(case_ids) * val_frac)

    val_ids = set(np.random.choice(case_ids, size=n_val, replace=False))

    train_df = df[~df['mpog_case_id'].isin(val_ids)].reset_index(drop=True)
    val_df = df[df['mpog_case_id'].isin(val_ids)].reset_index(drop=True)

    print(f"Train: {len(train_df):,} rows, {train_df['mpog_case_id'].nunique():,} cases")
    print(f"Val:   {len(val_df):,} rows, {val_df['mpog_case_id'].nunique():,} cases")

    return train_df, val_df


def load_test_data(
    data_dir: str,
    debug_frac: Optional[float] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load the fixed test set (4 held-out institutions).

    Args:
        data_dir: Directory containing institution feather files
        debug_frac: If set, sample this fraction of cases
        seed: Random seed

    Returns:
        Test DataFrame
    """
    print(f"\nLoading test set ({len(TEST_INSTITUTIONS)} institutions)...")
    return load_institutions_data(
        data_dir=data_dir,
        institution_ids=list(TEST_INSTITUTIONS),
        debug_frac=debug_frac,
        seed=seed
    )


def get_scale_summary(n: int, seed: int = 42) -> Dict:
    """
    Get summary of what institutions would be selected at a given scale.

    Returns:
        Dict with region breakdown and total cases
    """
    selected = select_institutions(n, seed)
    by_region = get_institutions_by_region(exclude_test=True)

    summary = {
        "n_requested": n,
        "n_selected": len(selected),
        "institutions": selected,
        "regions": {},
        "total_cases": 0,
    }

    for region in REGIONS:
        region_insts = [i for i in selected if INSTITUTION_METADATA[i][0] == region]
        region_cases = sum(INSTITUTION_METADATA[i][1] for i in region_insts)
        summary["regions"][region] = {
            "count": len(region_insts),
            "institutions": region_insts,
            "cases": region_cases,
        }
        summary["total_cases"] += region_cases

    return summary


if __name__ == "__main__":
    # Demo: show institution selection at each scale
    print("=" * 60)
    print("Institution Selection Preview")
    print("=" * 60)
    print(f"Test institutions (held out): {sorted(TEST_INSTITUTIONS)}")
    print(f"Available for training: {len(AVAILABLE_INSTITUTIONS)}")

    for n in [5, 10, 20, 40, 60]:
        summary = get_scale_summary(n)
        label = f"Scale {n}" if n < 60 else f"Scale {n} (all available)"
        print(f"\n{label}:")
        print(f"  Selected: {summary['n_selected']} institutions, {summary['total_cases']:,} cases")
        for region, info in summary["regions"].items():
            print(f"    {region}: {info['count']} institutions, {info['cases']:,} cases")
