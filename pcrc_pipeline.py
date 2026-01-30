#!/usr/bin/env python3
"""
PCRC 247 Dataset Pipeline - COMPLETE (Extraction + Preprocessing)
===================================================================
Extracts VITALS and MEDICATIONS data from PCRC 247 Aghaeepour DuckDB files
and creates FULLY PREPROCESSED minute-by-minute dataset matching or_foundational format.

ALL-IN-ONE PIPELINE:
- Extracts vitals, gases, medications from DuckDB
- Creates minute-by-minute grid
- Applies FULL preprocessing (matching or_foundational exactly):
  * _flag columns for all numeric columns
  * Sensible ranges (outliers -> NaN)
  * Proxy filling (invasive BP -> non-invasive BP)
  * Imputation (ffill -> bfill -> global median)
  * Fill gases/meds with 0
  * Manual fixes (glycopyrrolate, weight=0, sex=0->1)
  * Clip limits
  * _bolus columns for medications
  * Truncate to 20hrs max
  * Convert dtypes to float32

EFFICIENT PROCESSING:
- Processes data by INSTITUTION for memory efficiency
- Checkpoint/resume capability - saves per-institution files
- Final merge step combines all institutions

Output columns:
- phys_bp_sys_non_invasive, phys_bp_dias_non_invasive, phys_bp_mean_non_invasive
- phys_spo2_%, phys_spo2_pulse_rate, phys_end_tidal_co2_(mmhg)
- phys_sevoflurane_exp_%, phys_isoflurane_exp_%, phys_desflurane_exp_%, phys_nitrous_exp_%
- meds_{drug} - dose in mg/min
- meds_{drug}_bolus - bolus doses only
- {col}_flag - True if measured, False if absent/imputed
- age, sex, weight, case_id, mpog_case_id, institution, time_since_start

Usage:
    python pcrc_pipeline.py --base-path /path/to/PCRC_247_Aghaeepour --output-dir ./output
    python pcrc_pipeline.py --base-path /path/to/PCRC_247_Aghaeepour --institution 123
    python pcrc_pipeline.py --merge-only --output-dir ./output
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import gc
import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm

# =============================================================================
# MEMORY & DTYPE OPTIMIZATION CONFIGURATION
# =============================================================================

# Optimal dtypes for each column type - minimizes memory usage
# float16 has too little precision for clinical values, so use float32
DTYPE_CONFIG = {
    # Vitals - use float32 (saves 50% vs float64)
    'bp_sys': 'float32',
    'bp_dias': 'float32',
    'bp_mean': 'float32',
    'spo2': 'float32',
    'pulse_rate': 'float32',
    'etco2': 'float32',
    'sevoflurane': 'float32',
    'isoflurane': 'float32',
    'desflurane': 'float32',
    'nitrous': 'float32',

    # Demographics - use smallest type that fits
    'age': 'float32',  # 0-120 years, need float for NaN
    'sex': 'int8',     # 0/1/2
    'weight': 'float32',  # 40-230 kg
    'case_id': 'int32',  # millions of cases possible
    'institution': 'int16',  # < 32K institutions
    'time_since_start': 'int16',  # max 720 minutes (12h)
    'time_to_airway': 'float32',

    # Medications - float32 for rates
    'meds_default': 'float32',

    # Doses in extraction
    'dose': 'float32',
    'converted_dose': 'float32',
}

def optimize_dtypes(df: pd.DataFrame, stage: str = 'generic') -> pd.DataFrame:
    """
    Optimize DataFrame dtypes for memory efficiency.

    Args:
        df: DataFrame to optimize
        stage: 'extraction', 'grid', 'final' for context-specific optimization

    Returns:
        Optimized DataFrame (modified in place for efficiency)
    """
    for col in df.columns:
        if col in DTYPE_CONFIG:
            try:
                df[col] = df[col].astype(DTYPE_CONFIG[col])
            except (ValueError, TypeError):
                pass  # Keep original dtype if conversion fails
        elif col.startswith('meds_'):
            try:
                df[col] = df[col].astype('float32')
            except (ValueError, TypeError):
                pass
        elif col.startswith('phys_'):
            try:
                df[col] = df[col].astype('float32')
            except (ValueError, TypeError):
                pass
        # Convert object columns with few unique values to category
        elif df[col].dtype == 'object':
            n_unique = df[col].nunique()
            if n_unique < 100:  # Low cardinality - use category
                df[col] = df[col].astype('category')

    return df

def clear_memory():
    """Force garbage collection to free memory."""
    gc.collect()

# =============================================================================
# CONFIGURATION
# =============================================================================

# DuckDB file paths relative to base directory
DUCKDB_PATHS = {
    'caseinfo': 'case_info/pcrc_caseinfo.duckdb',
    'medications': 'medications/pcrc_medications.duckdb',
    'bloodpressure': 'bloodpressure_observations/bloodpressure_observations.duckdb',
    'spo2': 'spo2_observations/spo2_observations.duckdb',
    'pulserate': 'pulserate_observations/pulserate_observations.duckdb',
    'etco2_mmhg': 'etco2_mmhg_3235/etco2_mmhg_3235.duckdb',
    'sevoflurane': 'sevoflurane_exp_3270/sevoflurane_exp_3270.duckdb',
    'isoflurane': 'isoflurane_exp_3260/isoflurane_exp_3260.duckdb',
    'desflurane': 'desflurane_exp_3280/desflurane_exp_3280.duckdb',
    'nitrous': 'nitrous_exp_3255/nitrous_exp_3255.duckdb',
}

# =============================================================================
# MEDICATION CONFIGURATION
# =============================================================================

# Canonical medication names to extract
# These are the key anesthetic/hemodynamic medications
CANONICAL_MEDS = [
    'propofol', 'fentanyl', 'ketamine', 'dexmedetomidine', 'remifentanil',
    'phenylephrine', 'norepinephrine', 'epinephrine', 'vasopressin', 'ephedrine',
    'hydromorphone', 'glycopyrrolate', 'etomidate', 'esmolol', 'labetalol',
]

# Path to medication mapping CSV (raw_med -> Canonical_Med)
MED_MAPPING_CSV = os.path.join(os.path.dirname(__file__), 'mpog_tables', 'med_info.csv')

# Routes that indicate SYSTEMIC administration (IV, IM)
# Other routes (LOCAL INFILTRATION, EPIDURAL, INTRATHECAL, etc.) have different pharmacokinetics
SYSTEMIC_ROUTES = {'IV', 'IM', 'INTRAVENOUS', 'INTRAMUSCULAR'}

# Routes that are LOCAL/REGIONAL - track separately or exclude
# These are low systemic absorption and different clinical meaning
LOCAL_ROUTES = {
    'LOCAL INFILTRATION', 'EPIDURAL', 'INTRATHECAL', 'SPINAL',
    'OPHTHALMIC', 'IRRIGATION', 'INTRADERMAL', 'SUBCUTANEOUS',
    'TOPICAL', 'NERVE BLOCK', 'REGIONAL'
}

# Unit conversions to mg/min (standard unit for output)
# Format: [multiplier, weight_based (True if needs patient weight)]
UNIT_CONVERSIONS = {
    # Bolus units (result: mg)
    'mcg': [1e-3, False],
    'mg': [1, False],
    'g': [1e3, False],
    'units': [1, False],  # For vasopressin
    'meq': [1, False],

    # Infusion rates (result: mg/min)
    # Weight-based rates
    'mcg/kg/hr': [1e-3/60, True],
    'mg/kg/hr': [1/60, True],
    'mcg/kg/min': [1e-3, True],
    'mg/kg/min': [1, True],

    # Non-weight-based rates (common for vasopressors)
    'mcg/min': [1e-3, False],
    'mcg/hr': [1e-3/60, False],
    'mg/min': [1, False],
    'mg/hr': [1/60, False],
    'units/hr': [1/60, False],  # Vasopressin: units/hr
    'units/min': [1, False],

    # Volume-based (for epidural infusions - need concentration)
    'ml/hr': [1/60, False],  # Will need concentration adjustment
    'ml/min': [1, False],
}

# Compound medication concentrations - parsed from med_info.csv raw names
# Format: {lowercase_raw_med: {drug: concentration_mcg_per_ml}}
# These are used for ML/HR infusions to calculate actual drug dose
COMPOUND_MED_CONCENTRATIONS = {
    # Fentanyl compounds (concentration in mcg/ml from the name)
    'bupivacaine w/ fentanyl 0.0625%/ 10 mcg/ml': {'fentanyl': 10.0},
    'bupivacaine w/ fentanyl 0.0625% / 2 mcg/ml': {'fentanyl': 2.0},
    'bupivacaine w/ fentanyl 0.0625% / 3mcg/ml': {'fentanyl': 3.0},
    'bupivacaine w/ fentanyl 0.0625% / 5 mcg/ml': {'fentanyl': 5.0},
    'bupivacaine w/ fentanyl 0.05% / 2 mcg/ml': {'fentanyl': 2.0},
    'bupivacaine w/ fentanyl 0.05% / 3 mcg/ml': {'fentanyl': 3.0},
    'bupivacaine w/ fentanyl 0.05% / 5 mcg/ml': {'fentanyl': 5.0},
    'bupivacaine w/ fentanyl 0.08% / 2 mcg/ml': {'fentanyl': 2.0},
    'bupivacaine w/ fentanyl 0.1% / 2 mcg/ml ': {'fentanyl': 2.0},
    'bupivacaine w/ fentanyl 0.125% / 2 mcg/ml': {'fentanyl': 2.0},
    'bupivacaine w/ fentanyl 0.125% / 3 mcg/ml': {'fentanyl': 3.0},
    'bupivacaine w/ fentanyl 0.125% / 4 mcg/ml': {'fentanyl': 4.0},
    'bupivacaine w/ fentanyl 0.125% / 5 mcg/ml': {'fentanyl': 5.0},
    'bupivacaine w/ fentanyl 0.167% / 16.67 mcg/ml': {'fentanyl': 16.67},
    'bupivacaine w/ fentanyl 0.18% / 14.2 mcg/ml': {'fentanyl': 14.2},
    'bupivacaine w/ fentanyl 0.25% / 10 mcg/ml': {'fentanyl': 10.0},
    'bupivacaine w/ fentanyl 0.5% / 3 mcg/ml': {'fentanyl': 3.0},
    'bupivacaine w/ fentanyl 0.5% / 5mcg/ml': {'fentanyl': 5.0},
    'ropivacaine w/ fentanyl 0.025% / 3 mcg/ml': {'fentanyl': 3.0},
    'ropivacaine w/ fentanyl  0.125% / 2 mcg/ml': {'fentanyl': 2.0},
    'ropivacaine w/ fentanyl 0.1% / 2 mcg/ml': {'fentanyl': 2.0},
    'ropivacaine w/ fentanyl 0.2% / 2 mcg/ml': {'fentanyl': 2.0},
    'ropivacaine w/ fentanyl 0.2% / 3 mcg/ml': {'fentanyl': 3.0},
    'lidocaine w/ fentanyl w/ epinephrine w/ bicarbonate 2% / 5 mcg/ml / 1:200000': {'fentanyl': 5.0},
    'bupivacaine w/ fentanyl w/ epinephrine  37.5mg / 750mcg / 0.125mg': {'fentanyl': 5.0},  # 750mcg/150ml typical

    # Hydromorphone compounds (concentration in mcg/ml from the name)
    'bupivacaine w/ hydromorphone 0.0625% /10 mcg/ml': {'hydromorphone': 10.0},
    'bupivacaine w/ hydromorphone 0.0625% / 5mcg/ml': {'hydromorphone': 5.0},
    'bupivacaine w/ hydromorphone 0.0725% / 5mcg/ml': {'hydromorphone': 5.0},
    'bupivacaine w/ hydromorphone 0.05% /3 mcg/ml': {'hydromorphone': 3.0},
    'bupivacaine w/ hydromorphone 0.05% /10 mcg/ml': {'hydromorphone': 10.0},
    'bupivacaine w/ hydromorphone 0.1% / 20mcg/ml': {'hydromorphone': 20.0},
    'bupivacaine w/ hydromorphone 0.1% / unspecified': {'hydromorphone': 10.0},  # Assume 10 mcg/ml
    'bupivacaine w/ hydromorphone 0.125% / 5 mcg/ml': {'hydromorphone': 5.0},
    'bupivacaine w/ hydromorphone 0.125% / 10 mcg/ml': {'hydromorphone': 10.0},
    'bupivacaine w/ hydromorphone 0.125% / 20 mcg/ml': {'hydromorphone': 20.0},
    'bupivacaine 0.125% w/ hydromorphone 25 mcg/ml': {'hydromorphone': 25.0},
    'ropivacaine w/ hydromorphone 0.1% / 5 mcg/ml': {'hydromorphone': 5.0},
    'ropivacaine w/ hydromorphone 0.1% / 10 mcg/ml': {'hydromorphone': 10.0},
    'ropivacaine / hydromorphone / ketorolac': {'hydromorphone': 10.0},  # Assume standard 10 mcg/ml

    # Propofol/ketamine combos (propofol 10 mg/ml, ketamine varies)
    'propofol w/ ketamine 10mg/ml + unspecified ketamine ': {'propofol': 10000.0, 'ketamine': 1000.0},  # mcg/ml
    'propofol w/ ketamine 10 mg/ml + 1 mg/ml': {'propofol': 10000.0, 'ketamine': 1000.0},

    # Propofol/remifentanil (propofol 10 mg/ml, remifentanil 20 mcg/ml)
    'propofol w/ remifentanil 10 mg/ml + 20 mcg/ml': {'propofol': 10000.0, 'remifentanil': 20.0},

    # Propofol/alfentanil
    'propofol w/ alfentanil 10 mg/ml + 50 mcg/ml': {'propofol': 10000.0},  # Alfentanil not in our list
}

# Infusions that should be treated as boluses (cumulative volume)
INFUSIONS_THAT_ARE_BOLUSES = [
    'normosol', 'ns infusion', 'lr infusion', 'lr iv infusion',
]

# Filtering criteria (matching or_foundational)
MIN_AGE = 15
MIN_WEIGHT = 40
MAX_WEIGHT = 230
MAX_CASE_DURATION_HOURS = 12
MAX_CASE_MINUTES_FINAL = 20 * 60  # Truncate to 20 hours max

# =============================================================================
# PREPROCESSING CONFIGURATION (from or_foundational)
# =============================================================================

# Columns to impute with ffill -> bfill -> global median
IMPUTE_COLS = [
    'phys_bp_sys_non_invasive', 'phys_bp_dias_non_invasive', 'phys_bp_mean_non_invasive',
    'phys_bp_sys_arterial_line_(invasive,_peripheral)',
    'phys_bp_dias_arterial_line_(invasive,_peripheral)',
    'phys_bp_mean_arterial_line_(invasive,_peripheral)',
    'phys_spo2_%', 'phys_spo2_pulse_rate', 'phys_end_tidal_co2_(mmhg)',
]

# Excluded from imputation - fill with 0 instead (gases)
IMPUTE_EXCLUDE = [
    'phys_sevoflurane_exp_%', 'phys_isoflurane_exp_%',
    'phys_desflurane_exp_%', 'phys_nitrous_exp_%',
]

# Proxy filling: invasive BP can fill non-invasive BP
PROXIES = [
    ('phys_bp_dias_non_invasive', 'phys_bp_dias_arterial_line_(invasive,_peripheral)'),
    ('phys_bp_sys_non_invasive', 'phys_bp_sys_arterial_line_(invasive,_peripheral)'),
    ('phys_bp_mean_non_invasive', 'phys_bp_mean_arterial_line_(invasive,_peripheral)'),
]

# Sensible ranges - values outside become NaN
SENSIBLE_RANGES = {
    'phys_bp_dias_non_invasive': (20, 200),
    'phys_bp_dias_arterial_line_(invasive,_peripheral)': (20, 200),
    'phys_bp_sys_non_invasive': (40, 300),
    'phys_bp_sys_arterial_line_(invasive,_peripheral)': (40, 300),
    'phys_bp_mean_non_invasive': (35, 150),
    'phys_bp_mean_arterial_line_(invasive,_peripheral)': (35, 150),
    'phys_spo2_%': (60.0, 100.0),
    'phys_spo2_pulse_rate': (25.0, 300.0),
}

# Clip limits from or_foundational/params/params.py
CLIP_LIMITS = {
    # Medications (upper limit only, lower is 0)
    'meds_dexmedetomidine': 0.100, 'meds_ephedrine': 20.0, 'meds_epinephrine': 1.0,
    'meds_esmolol': 100.0, 'meds_etomidate': 40.0, 'meds_fentanyl': 0.500,
    'meds_glycopyrrolate': 0.4, 'meds_hydromorphone': 4.0, 'meds_ketamine': 250.0,
    'meds_labetalol': 100.0, 'meds_norepinephrine': 0.050, 'meds_phenylephrine': 0.500,
    'meds_propofol': 400.0, 'meds_remifentanil': 0.500, 'meds_vasopressin': 2.0,
    # Vitals (min, max)
    'phys_bp_dias_arterial_line_(invasive,_peripheral)': (0, 150),
    'phys_bp_dias_non_invasive': (0, 150),
    'phys_bp_mean_arterial_line_(invasive,_peripheral)': (0, 200),
    'phys_bp_mean_non_invasive': (0, 200),
    'phys_bp_sys_arterial_line_(invasive,_peripheral)': (0, 250),
    'phys_bp_sys_non_invasive': (0, 250),
    'phys_end_tidal_co2_(mmhg)': (0, 80),
    'phys_sevoflurane_exp_%': (0, 8.0),
    'phys_isoflurane_exp_%': (0, 4.0),
    'phys_desflurane_exp_%': (0, 18.0),
    'phys_nitrous_exp_%': (0, 80.0),
    'phys_spo2_%': (30.0, 100.0),
    'phys_spo2_pulse_rate': (25.0, 300.0),
}

# Bolus detection thresholds (from or_foundational/params/params.py)
BOLUS_MINS = {
    'meds_propofol': 20, 'meds_phenylephrine': 0.050, 'meds_norepinephrine': 0.008,
    'meds_epinephrine': 0.008, 'meds_vasopressin': 0.5, 'meds_ketamine': 5,
    'meds_remifentanil': 0.025, 'meds_fentanyl': 0.0125, 'meds_dexmedetomidine': 0.004,
    'meds_ephedrine': 1, 'meds_esmolol': 5, 'meds_glycopyrrolate': 0.05,
    'meds_labetalol': 2.5, 'meds_hydromorphone': 0.1, 'meds_etomidate': 4,
}
BOLUS_MAXS = {
    'meds_propofol': 400, 'meds_phenylephrine': 0.600, 'meds_norepinephrine': 0.032,
    'meds_epinephrine': 1.0, 'meds_vasopressin': 5.0, 'meds_ketamine': 100,
    'meds_remifentanil': 0.500, 'meds_fentanyl': 1.0, 'meds_dexmedetomidine': 0.120,
    'meds_ephedrine': 50, 'meds_esmolol': 250, 'meds_glycopyrrolate': 0.800,
    'meds_labetalol': 50, 'meds_hydromorphone': 3.0, 'meds_etomidate': 60,
}
BOLUS_TYPICALS = {
    'meds_phenylephrine': [0.050, 0.100, 0.200],
    'meds_norepinephrine': [0.008, 0.004],
    'meds_epinephrine': [0.008, 0.004],
    'meds_vasopressin': [0.5, 1.0],
    'meds_fentanyl': [0.025, 0.050, 0.100, 0.150, 0.200, 0.250],
    'meds_dexmedetomidine': [0.004],
    'meds_ephedrine': [5, 10],
    'meds_glycopyrrolate': [0.2, 0.4],
}

# Default medians for imputation fallback
DEFAULT_MEDIANS = {
    'phys_bp_sys_non_invasive': 120.0, 'phys_bp_dias_non_invasive': 70.0,
    'phys_bp_mean_non_invasive': 85.0, 'phys_spo2_%': 98.0,
    'phys_spo2_pulse_rate': 75.0, 'phys_end_tidal_co2_(mmhg)': 35.0,
    'weight': 75.0,
}

# =============================================================================
# MEDICATION MAPPING (loaded from CSV)
# =============================================================================

def load_med_mapping() -> dict:
    """
    Load the raw_med -> Canonical_Med mapping from med_info.csv.
    Returns a dict: {lowercase_raw_med: canonical_med}

    This handles ALL 103 different raw medication name variants.
    """
    mapping = {}

    if os.path.exists(MED_MAPPING_CSV):
        try:
            df = pd.read_csv(MED_MAPPING_CSV)
            for _, row in df.iterrows():
                raw = str(row['raw_med']).lower().strip()
                canonical = str(row['Canonical_Med']).lower().strip()
                if raw and canonical and canonical in CANONICAL_MEDS:
                    mapping[raw] = canonical
            print(f"  Loaded {len(mapping)} medication mappings from {MED_MAPPING_CSV}")
        except Exception as e:
            print(f"  Warning: Could not load med mapping: {e}")

    return mapping

# Global medication mapping (loaded once)
_MED_MAPPING = None

def get_med_mapping() -> dict:
    """Get the medication mapping, loading it if necessary."""
    global _MED_MAPPING
    if _MED_MAPPING is None:
        _MED_MAPPING = load_med_mapping()
    return _MED_MAPPING


def map_med_to_canonical(med_name: str) -> str:
    """
    Map a raw medication name to its canonical form.
    Uses the med_info.csv mapping first, then falls back to simple matching.

    Returns canonical med name or None if not a tracked medication.
    """
    if pd.isna(med_name):
        return None

    med_lower = str(med_name).lower().strip()
    mapping = get_med_mapping()

    # First try exact match from CSV mapping
    if med_lower in mapping:
        return mapping[med_lower]

    # Fallback: simple contains match for canonical meds
    # This catches cases not in the CSV
    for canonical in CANONICAL_MEDS:
        if canonical in med_lower:
            return canonical

    return None


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def get_db_path(base_path: str, key: str) -> str:
    """Get full path to a DuckDB file."""
    return os.path.join(base_path, DUCKDB_PATHS[key])


def query_duckdb(db_path: str, query: str) -> pd.DataFrame:
    """Execute query on DuckDB file and return DataFrame."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        result = con.execute(query).fetchdf()
    finally:
        con.close()
    return result


def get_table_name(db_path: str) -> str:
    """Get the main table name from a DuckDB file."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = con.execute("SHOW TABLES").fetchdf()
        if len(tables) == 0:
            raise ValueError(f"No tables found in {db_path}")
        return tables.iloc[0, 0]
    finally:
        con.close()


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_caseinfo(base_path: str, institution: Optional[int] = None) -> pd.DataFrame:
    """Extract case information with filtering, optionally for a specific institution."""
    db_path = get_db_path(base_path, 'caseinfo')
    table_name = get_table_name(db_path)

    institution_filter = f"AND InstitutionNumber = {institution}" if institution else ""

    query = f"""
    SELECT
        MPOGCaseID,
        InstitutionNumber,
        Age_Years as age,
        Sex as sex,
        Weight as weight,
        Height as height,
        AnesthesiaStart,
        AnesthesiaEnd,
        IntubationTime
    FROM {table_name}
    WHERE Age_Years >= {MIN_AGE}
      AND Weight >= {MIN_WEIGHT}
      AND Weight <= {MAX_WEIGHT}
      {institution_filter}
    """

    df = query_duckdb(db_path, query)

    # Calculate case duration and filter
    df['AnesthesiaStart'] = pd.to_datetime(df['AnesthesiaStart'], errors='coerce')
    df['AnesthesiaEnd'] = pd.to_datetime(df['AnesthesiaEnd'], errors='coerce')
    df['IntubationTime'] = pd.to_datetime(df['IntubationTime'], errors='coerce')

    df['case_duration_hours'] = (df['AnesthesiaEnd'] - df['AnesthesiaStart']).dt.total_seconds() / 3600
    df = df[df['case_duration_hours'] <= MAX_CASE_DURATION_HOURS].copy()
    df = df[df['case_duration_hours'] > 0].copy()  # Must have valid duration

    # Encode sex: M=1, F=2, unknown=0
    df['sex'] = df['sex'].map({'Male': 1, 'M': 1, 'Female': 2, 'F': 2}).fillna(0).astype('float32')

    print(f"Extracted {len(df)} valid cases from caseinfo" +
          (f" (institution {institution})" if institution else ""))
    return df


def get_all_institutions(base_path: str) -> List[int]:
    """Get list of all institution numbers."""
    db_path = get_db_path(base_path, 'caseinfo')
    table_name = get_table_name(db_path)

    query = f"""
    SELECT DISTINCT InstitutionNumber
    FROM {table_name}
    WHERE InstitutionNumber IS NOT NULL
    ORDER BY InstitutionNumber
    """

    df = query_duckdb(db_path, query)
    institutions = df['InstitutionNumber'].dropna().astype(int).tolist()
    print(f"Found {len(institutions)} institutions")
    return institutions


def get_institution_case_counts(base_path: str) -> pd.DataFrame:
    """Get case counts per institution (after filtering)."""
    db_path = get_db_path(base_path, 'caseinfo')
    table_name = get_table_name(db_path)

    query = f"""
    SELECT
        InstitutionNumber,
        COUNT(*) as total_cases,
        SUM(CASE WHEN Age_Years >= {MIN_AGE}
                  AND Weight >= {MIN_WEIGHT}
                  AND Weight <= {MAX_WEIGHT} THEN 1 ELSE 0 END) as valid_cases
    FROM {table_name}
    WHERE InstitutionNumber IS NOT NULL
    GROUP BY InstitutionNumber
    ORDER BY valid_cases DESC
    """

    return query_duckdb(db_path, query)


def extract_vitals_bp(base_path: str, case_ids: list) -> pd.DataFrame:
    """Extract blood pressure observations."""
    db_path = get_db_path(base_path, 'bloodpressure')
    table_name = get_table_name(db_path)

    chunk_size = 10000
    all_data = []

    for i in range(0, len(case_ids), chunk_size):
        chunk_ids = case_ids[i:i+chunk_size]
        ids_str = "','".join(chunk_ids)

        query = f"""
        SELECT
            MPOGCaseID,
            BPObsTime as obs_time,
            BPObsSystolic as bp_sys,
            BPObsDiastolic as bp_dias,
            COALESCE(BPObsMeanRecorded, BPObsMeanComputed) as bp_mean
        FROM {table_name}
        WHERE MPOGCaseID IN ('{ids_str}')
        """

        chunk_df = query_duckdb(db_path, query)
        all_data.append(chunk_df)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df['obs_time'] = pd.to_datetime(df['obs_time'], errors='coerce')

    # Optimize dtypes immediately to reduce memory
    for col in ['bp_sys', 'bp_dias', 'bp_mean']:
        if col in df.columns:
            df[col] = df[col].astype('float32')

    print(f"Extracted {len(df):,} BP records ({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    return df


def extract_vitals_spo2(base_path: str, case_ids: list) -> pd.DataFrame:
    """Extract SpO2 observations."""
    db_path = get_db_path(base_path, 'spo2')
    table_name = get_table_name(db_path)

    chunk_size = 10000
    all_data = []

    for i in range(0, len(case_ids), chunk_size):
        chunk_ids = case_ids[i:i+chunk_size]
        ids_str = "','".join(chunk_ids)

        query = f"""
        SELECT
            MPOGCaseID,
            SpO2ObsTime as obs_time,
            SpO2Obs as spo2
        FROM {table_name}
        WHERE MPOGCaseID IN ('{ids_str}')
        """

        chunk_df = query_duckdb(db_path, query)
        all_data.append(chunk_df)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df['obs_time'] = pd.to_datetime(df['obs_time'], errors='coerce')
    if 'spo2' in df.columns:
        df['spo2'] = df['spo2'].astype('float32')

    print(f"Extracted {len(df):,} SpO2 records ({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    return df


def extract_vitals_pulserate(base_path: str, case_ids: list) -> pd.DataFrame:
    """Extract pulse rate observations."""
    db_path = get_db_path(base_path, 'pulserate')
    table_name = get_table_name(db_path)

    chunk_size = 10000
    all_data = []

    for i in range(0, len(case_ids), chunk_size):
        chunk_ids = case_ids[i:i+chunk_size]
        ids_str = "','".join(chunk_ids)

        query = f"""
        SELECT
            MPOGCaseID,
            PulseRateObsPulseRateTime as obs_time,
            PulseRateObsPulseRate as pulse_rate
        FROM {table_name}
        WHERE MPOGCaseID IN ('{ids_str}')
        """

        chunk_df = query_duckdb(db_path, query)
        all_data.append(chunk_df)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df['obs_time'] = pd.to_datetime(df['obs_time'], errors='coerce')
    if 'pulse_rate' in df.columns:
        df['pulse_rate'] = df['pulse_rate'].astype('float32')

    print(f"Extracted {len(df):,} pulse rate records ({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    return df


def extract_vitals_etco2(base_path: str, case_ids: list) -> pd.DataFrame:
    """Extract EtCO2 observations."""
    db_path = get_db_path(base_path, 'etco2_mmhg')
    table_name = get_table_name(db_path)

    chunk_size = 10000
    all_data = []

    for i in range(0, len(case_ids), chunk_size):
        chunk_ids = case_ids[i:i+chunk_size]
        ids_str = "','".join(chunk_ids)

        query = f"""
        SELECT
            MPOGCaseID,
            Value_3235_DT as obs_time,
            TRY_CAST(Value_3235 AS DOUBLE) as etco2
        FROM {table_name}
        WHERE MPOGCaseID IN ('{ids_str}')
        """

        chunk_df = query_duckdb(db_path, query)
        all_data.append(chunk_df)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df['obs_time'] = pd.to_datetime(df['obs_time'], errors='coerce')
    if 'etco2' in df.columns:
        df['etco2'] = df['etco2'].astype('float32')

    print(f"Extracted {len(df):,} EtCO2 records ({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    return df


def extract_anesthetic_gas(base_path: str, case_ids: list, gas_key: str, value_col: str) -> pd.DataFrame:
    """Extract anesthetic gas observations."""
    db_path = get_db_path(base_path, gas_key)

    if not os.path.exists(db_path):
        print(f"Warning: {gas_key} database not found at {db_path}")
        return pd.DataFrame()

    table_name = get_table_name(db_path)

    chunk_size = 10000
    all_data = []

    for i in range(0, len(case_ids), chunk_size):
        chunk_ids = case_ids[i:i+chunk_size]
        ids_str = "','".join(chunk_ids)

        # Get column names dynamically
        con = duckdb.connect(db_path, read_only=True)
        cols = con.execute(f"DESCRIBE {table_name}").fetchdf()['column_name'].tolist()
        con.close()

        # Find the value and datetime columns
        dt_col = [c for c in cols if '_DT' in c][0] if any('_DT' in c for c in cols) else None
        val_col = [c for c in cols if c not in ['MPOGCaseID'] and '_DT' not in c][0] if len(cols) > 2 else None

        if not dt_col or not val_col:
            return pd.DataFrame()

        query = f"""
        SELECT
            MPOGCaseID,
            {dt_col} as obs_time,
            TRY_CAST({val_col} AS DOUBLE) as value
        FROM {table_name}
        WHERE MPOGCaseID IN ('{ids_str}')
        """

        chunk_df = query_duckdb(db_path, query)
        all_data.append(chunk_df)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df['obs_time'] = pd.to_datetime(df['obs_time'], errors='coerce')
    df = df.rename(columns={'value': value_col})
    if value_col in df.columns:
        df[value_col] = df[value_col].astype('float32')

    print(f"Extracted {len(df):,} {gas_key} records ({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    return df


# =============================================================================
# MEDICATION EXTRACTION AND TRANSFORMATION
# =============================================================================

def extract_medications(base_path: str, case_ids: list) -> pd.DataFrame:
    """Extract medication data from DuckDB."""
    db_path = get_db_path(base_path, 'medications')

    if not os.path.exists(db_path):
        print(f"Warning: Medications database not found at {db_path}")
        return pd.DataFrame()

    table_name = get_table_name(db_path)

    # =========================================================================
    # DEBUG: Understand why case IDs might not match
    # =========================================================================
    print(f"\n    === MEDICATION EXTRACTION DEBUG ===")
    print(f"    Querying meds for {len(case_ids)} cases from caseinfo")

    # Sample caseinfo IDs - show type and format
    sample_caseinfo = case_ids[:5]
    print(f"    Sample caseinfo IDs: {sample_caseinfo}")
    print(f"    Caseinfo ID types: {[type(x).__name__ for x in sample_caseinfo]}")
    print(f"    Caseinfo ID lengths: {[len(str(x)) for x in sample_caseinfo]}")

    # Check what's in medications table
    con = duckdb.connect(db_path, read_only=True)

    # Total unique cases in meds table
    total_med_cases = con.execute(f"SELECT COUNT(DISTINCT MPOGCaseID) as cnt FROM {table_name}").fetchone()[0]
    print(f"    Total unique cases in medications table: {total_med_cases:,}")

    # Sample medication table IDs
    sample_med_ids = con.execute(f"SELECT DISTINCT MPOGCaseID FROM {table_name} LIMIT 5").fetchdf()
    sample_med_list = sample_med_ids['MPOGCaseID'].tolist()
    print(f"    Sample medication table IDs: {sample_med_list}")
    print(f"    Medication ID types: {[type(x).__name__ for x in sample_med_list]}")
    print(f"    Medication ID lengths: {[len(str(x)) for x in sample_med_list]}")

    # Check if ANY of our case_ids exist in medications (quick test)
    test_ids = case_ids[:100]
    test_ids_str = "','".join([str(x).strip() for x in test_ids])
    test_query = f"SELECT COUNT(DISTINCT MPOGCaseID) as cnt FROM {table_name} WHERE MPOGCaseID IN ('{test_ids_str}')"
    test_match = con.execute(test_query).fetchone()[0]
    print(f"    Quick test: {test_match}/100 of first 100 caseinfo IDs found in meds table")

    # Check if first caseinfo ID exists at all
    first_id = str(case_ids[0]).strip()
    exact_match = con.execute(f"SELECT COUNT(*) FROM {table_name} WHERE MPOGCaseID = '{first_id}'").fetchone()[0]
    like_match = con.execute(f"SELECT COUNT(*) FROM {table_name} WHERE MPOGCaseID LIKE '%{first_id[-10:]}%'").fetchone()[0]
    print(f"    First ID '{first_id}': exact_match={exact_match}, partial_match={like_match}")

    # Check if meds table has institution info
    try:
        cols = con.execute(f"DESCRIBE {table_name}").fetchdf()['column_name'].tolist()
        print(f"    Medications table columns: {cols}")
        if 'InstitutionNumber' in cols:
            inst_counts = con.execute(f"SELECT InstitutionNumber, COUNT(DISTINCT MPOGCaseID) as cases FROM {table_name} GROUP BY InstitutionNumber ORDER BY cases DESC LIMIT 5").fetchdf()
            print(f"    Meds table institution distribution (top 5):\n{inst_counts.to_string()}")
    except Exception as e:
        print(f"    Could not get column info: {e}")

    con.close()
    print(f"    === END DEBUG ===\n")

    chunk_size = 10000
    all_data = []

    for i in range(0, len(case_ids), chunk_size):
        chunk_ids = case_ids[i:i+chunk_size]
        # Ensure case_ids are strings and stripped
        chunk_ids = [str(cid).strip() for cid in chunk_ids]
        ids_str = "','".join(chunk_ids)

        query = f"""
        SELECT
            MPOGCaseID,
            Medication as med_name,
            Medication_Route as med_route,
            Medication_DoseType as dose_type,
            Medication_Dose as dose,
            Medication_UOM as uom,
            DoseStart_DT as dose_start,
            DoseEnd_DT as dose_end,
            Weight_kg as weight
        FROM {table_name}
        WHERE MPOGCaseID IN ('{ids_str}')
        """

        try:
            chunk_df = query_duckdb(db_path, query)
            all_data.append(chunk_df)
        except Exception as e:
            print(f"  Warning: Error extracting medications chunk: {e}")

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)

    # Parse timestamps
    df['dose_start'] = pd.to_datetime(df['dose_start'], errors='coerce')
    df['dose_end'] = pd.to_datetime(df['dose_end'], errors='coerce')

    # Optimize dtypes for memory efficiency
    if 'dose' in df.columns:
        df['dose'] = pd.to_numeric(df['dose'], errors='coerce').astype('float32')
    if 'weight' in df.columns:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').astype('float32')
    # Use category for low-cardinality string columns
    for col in ['med_route', 'dose_type', 'uom']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    n_cases_with_meds = df['MPOGCaseID'].nunique()
    # Debug: show sample of what we actually got
    if len(df) > 0:
        sample_meds_ids = df['MPOGCaseID'].head(3).tolist()
        print(f"    Meds table sample IDs: {sample_meds_ids}")
    print(f"Extracted {len(df):,} medication records for {n_cases_with_meds:,} cases ({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    return df


def normalize_med_name(med_name: str) -> tuple:
    """
    Normalize medication name to canonical form.

    Returns: (canonical_name, concentration_mcg_per_ml or None)

    For compound medications (e.g., "BUPIVACAINE W/ FENTANYL 0.0625% / 2 MCG/ML"),
    extracts the active drug and its concentration.
    """
    if pd.isna(med_name):
        return (None, None)

    med_lower = med_name.lower().strip()

    # Check compound medications first (need to extract concentration)
    for pattern, drugs in COMPOUND_MED_CONCENTRATIONS.items():
        if pattern in med_lower:
            # Return the first tracked drug and its concentration
            for drug, conc in drugs.items():
                if drug in CANONICAL_MEDS:
                    return (drug, conc)

    # Check if any canonical med is contained in the name
    for canonical in CANONICAL_MEDS:
        if canonical in med_lower:
            return (canonical, None)

    return (None, None)


def is_systemic_route(route: str) -> bool:
    """Check if route indicates systemic drug delivery."""
    if pd.isna(route):
        return True  # Assume systemic if missing
    route_upper = route.upper().strip()
    return route_upper in SYSTEMIC_ROUTES


def convert_dose_to_mg(dose: float, uom: str, weight: float,
                       concentration_mcg_ml: float = None) -> tuple:
    """
    Convert dose to standard units (mg for bolus, mg/min for infusion).

    For volume-based infusions (ML/HR), uses concentration to convert to drug mass.

    Returns: (converted_dose, is_infusion)
    """
    if pd.isna(dose) or pd.isna(uom):
        return (np.nan, False)

    uom_lower = uom.lower().strip()

    # Determine if infusion based on unit (more reliable than dose_type field)
    is_infusion = '/' in uom_lower

    # Handle volume-based units (ML/HR) - need concentration
    if uom_lower in ('ml/hr', 'ml/min'):
        if concentration_mcg_ml is not None and concentration_mcg_ml > 0:
            # Convert ml/hr * mcg/ml → mcg/hr → mg/min
            if uom_lower == 'ml/hr':
                # dose (ml/hr) * concentration (mcg/ml) = mcg/hr
                mcg_per_hr = dose * concentration_mcg_ml
                mg_per_min = mcg_per_hr * 1e-3 / 60
                return (mg_per_min, True)
            else:  # ml/min
                mcg_per_min = dose * concentration_mcg_ml
                mg_per_min = mcg_per_min * 1e-3
                return (mg_per_min, True)
        else:
            # No concentration available - skip
            return (np.nan, is_infusion)

    # Get conversion factor
    if uom_lower not in UNIT_CONVERSIONS:
        # Try to find a close match
        for key in UNIT_CONVERSIONS:
            if key in uom_lower or uom_lower in key:
                uom_lower = key
                break
        else:
            return (np.nan, is_infusion)

    conv_factor, is_weight_based = UNIT_CONVERSIONS[uom_lower]

    # Apply conversion
    converted_dose = dose * conv_factor

    # Apply weight if needed
    if is_weight_based:
        if pd.isna(weight) or weight <= 0:
            weight = 75.0  # Default weight
        converted_dose *= weight

    return (converted_dose, is_infusion)


def process_medications_for_case(meds_df: pd.DataFrame, case_id: str,
                                  case_start: pd.Timestamp, case_end: pd.Timestamp,
                                  patient_weight: float) -> pd.DataFrame:
    """
    Process medications for a single case into minute-by-minute format.

    Medical logic:
    - Only includes SYSTEMIC routes (IV, IM) for true drug effect
    - Handles compound medications (e.g., epidural fentanyl) with concentration extraction
    - Boluses: Total dose (mg) assigned to start minute
    - Infusions: Rate (mg/min) spread across duration from start to end

    Output units: mg/min for each medication column
    """
    # Create minute index
    total_minutes = int((case_end - case_start).total_seconds() / 60) + 1
    minute_index = pd.date_range(start=case_start, periods=total_minutes, freq='min')

    # Initialize result with zeros for all medication columns
    result = pd.DataFrame({'obs_time': minute_index})
    for med in CANONICAL_MEDS:
        result[f'meds_{med}'] = 0.0

    if meds_df.empty:
        return result

    # Filter to this case
    case_meds = meds_df[meds_df['MPOGCaseID'] == case_id].copy()
    if case_meds.empty:
        return result

    # Process each medication record
    for _, row in case_meds.iterrows():
        # Normalize medication name and get concentration if compound med
        canonical_name, concentration_mcg_ml = normalize_med_name(row['med_name'])
        if canonical_name is None:
            continue

        col_name = f'meds_{canonical_name}'
        if col_name not in result.columns:
            continue

        # Check route - only process systemic routes for most meds
        # Exception: compound meds with concentration (epidural opioids) are processed
        route = row['med_route'] if 'med_route' in row.index else None
        if concentration_mcg_ml is None and not is_systemic_route(route):
            # Skip non-systemic routes for simple medications
            # (e.g., skip "EPINEPHRINE" via LOCAL INFILTRATION)
            continue

        # Use weight from med record, fallback to patient weight
        weight = row['weight'] if pd.notna(row['weight']) and row['weight'] > 0 else patient_weight

        # Convert dose (pass concentration for volume-based units)
        converted_dose, is_infusion = convert_dose_to_mg(
            row['dose'], row['uom'], weight, concentration_mcg_ml
        )

        if pd.isna(converted_dose) or converted_dose <= 0:
            continue

        dose_start = row['dose_start']
        dose_end = row['dose_end']

        if pd.isna(dose_start):
            continue

        # Clip to case boundaries
        dose_start = max(dose_start, case_start)

        if is_infusion and pd.notna(dose_end):
            # Infusion: spread rate over duration
            dose_end = min(dose_end, case_end)

            # Find minutes within infusion period
            mask = (result['obs_time'] >= dose_start) & (result['obs_time'] < dose_end)

            # Add infusion rate (already in mg/min from conversion)
            result.loc[mask, col_name] += converted_dose
        else:
            # Bolus: assign to start minute only
            start_minute = dose_start.floor('min')
            mask = result['obs_time'] == start_minute

            # For bolus, dose is total mg - assign to that minute
            result.loc[mask, col_name] += converted_dose

    return result


# =============================================================================
# MINUTE-BY-MINUTE AGGREGATION - FULLY VECTORIZED (FAST)
# =============================================================================

def create_minute_grid_vectorized(caseinfo: pd.DataFrame) -> pd.DataFrame:
    """
    Create minute-by-minute grid for ALL cases at once - TRULY VECTORIZED.

    Uses numpy.repeat to expand case data to minute level without any Python loops.
    This is 10-50x faster than iterrows approach.
    """
    print("  Creating minute grid for all cases...")

    # Filter valid cases
    valid = caseinfo[caseinfo['AnesthesiaStart'].notna() & caseinfo['AnesthesiaEnd'].notna()].copy()

    if len(valid) == 0:
        return pd.DataFrame()

    # Calculate minutes per case
    valid['total_minutes'] = ((valid['AnesthesiaEnd'] - valid['AnesthesiaStart']).dt.total_seconds() / 60).astype(int) + 1
    valid['total_minutes'] = valid['total_minutes'].clip(upper=MAX_CASE_DURATION_HOURS * 60 + 1)
    valid = valid.reset_index(drop=True)

    # TRULY VECTORIZED: Use numpy.repeat to expand all cases at once
    n_cases = len(valid)
    minutes_per_case = valid['total_minutes'].values
    total_rows = minutes_per_case.sum()

    print(f"    Expanding {n_cases} cases to {total_rows:,} minute rows...")

    # Repeat case indices for each minute
    case_indices = np.repeat(np.arange(n_cases), minutes_per_case)

    # Generate time_since_start for each row - TRULY VECTORIZED
    # This creates [0,1,2,...,n1-1, 0,1,2,...,n2-1, ...] for each case
    # Use cumsum trick: count from start, subtract case boundary positions
    cumulative_mins = np.arange(total_rows)
    case_starts = np.repeat(np.cumsum(np.concatenate([[0], minutes_per_case[:-1]])), minutes_per_case)
    time_since_start = cumulative_mins - case_starts

    # Extract case data using repeated indices (vectorized) with optimized dtypes
    grid = pd.DataFrame({
        'MPOGCaseID': valid['MPOGCaseID'].values[case_indices],
        'time_since_start': time_since_start.astype('int16'),  # Max 720 mins fits in int16
        'age': valid['age'].values[case_indices].astype('float32'),
        'sex': valid['sex'].values[case_indices].astype('int8'),
        'weight': valid['weight'].values[case_indices].astype('float32'),
        'case_start': valid['AnesthesiaStart'].values[case_indices],
        'intubation_time': valid['IntubationTime'].values[case_indices],
    })

    # Calculate obs_time vectorized
    grid['obs_time'] = grid['case_start'] + pd.to_timedelta(grid['time_since_start'], unit='min')

    mem_mb = grid.memory_usage(deep=True).sum() / 1e6
    print(f"  Created grid: {len(grid):,} rows for {grid['MPOGCaseID'].nunique():,} cases ({mem_mb:.1f} MB)")

    return grid


def aggregate_vitals_vectorized(grid: pd.DataFrame, vitals_df: pd.DataFrame,
                                value_col: str, output_col: str) -> pd.DataFrame:
    """
    Aggregate vitals to minute level for ALL cases at once - VECTORIZED.
    Uses merge + groupby instead of per-case loops.
    """
    if vitals_df.empty or len(grid) == 0:
        grid[output_col] = np.nan
        return grid

    # Floor observation times to minute
    vitals_df = vitals_df.copy()
    vitals_df['minute'] = vitals_df['obs_time'].dt.floor('min')

    # Aggregate to minute level across ALL cases at once
    agg = vitals_df.groupby(['MPOGCaseID', 'minute'])[value_col].mean().reset_index()
    agg = agg.rename(columns={value_col: output_col, 'minute': 'obs_time'})

    # Merge into grid
    grid = grid.merge(agg, on=['MPOGCaseID', 'obs_time'], how='left')

    return grid


def process_medications_vectorized(grid: pd.DataFrame, meds_df: pd.DataFrame,
                                   caseinfo: pd.DataFrame) -> pd.DataFrame:
    """
    Process medications for ALL cases at once - VECTORIZED where possible.

    Still needs some iteration for infusion spreading, but pre-filters
    and uses vectorized operations where possible.
    """
    # Initialize medication columns with zeros
    for med in CANONICAL_MEDS:
        grid[f'meds_{med}'] = 0.0

    if meds_df.empty:
        return grid

    print("  Processing medications (vectorized with CSV mapping)...")

    # Pre-filter medications to only canonical meds and systemic routes
    meds_df = meds_df.copy()

    # Load the medication mapping from CSV (handles all 103 raw med variants)
    med_mapping = get_med_mapping()

    # Map raw med names to canonical using the CSV mapping
    # This is the PROPER way - using the curated mapping file
    meds_df['med_name_lower'] = meds_df['med_name'].str.lower().str.strip()

    # First: exact match from CSV mapping (vectorized via map)
    meds_df['canonical_med'] = meds_df['med_name_lower'].map(med_mapping)

    # Fallback for any not in CSV: simple contains match
    unmatched_mask = meds_df['canonical_med'].isna()
    if unmatched_mask.any():
        for med in CANONICAL_MEDS:
            mask = unmatched_mask & meds_df['med_name_lower'].str.contains(med, na=False)
            meds_df.loc[mask, 'canonical_med'] = med
            unmatched_mask = meds_df['canonical_med'].isna()

    # Report matching stats
    total_rows = len(meds_df)
    matched_rows = meds_df['canonical_med'].notna().sum()
    print(f"    Matched {matched_rows:,} / {total_rows:,} medication records ({100*matched_rows/total_rows:.1f}%)")

    # Filter to only tracked meds
    meds_df = meds_df[meds_df['canonical_med'].notna()].copy()

    if meds_df.empty:
        return grid

    # Filter systemic routes (vectorized)
    # IMPORTANT: This excludes epidural/intrathecal/local infiltration routes
    systemic_mask = meds_df['med_route'].str.upper().isin(SYSTEMIC_ROUTES) | meds_df['med_route'].isna()
    meds_df = meds_df[systemic_mask].copy()

    print(f"    After systemic route filter: {len(meds_df):,} records")

    if meds_df.empty:
        return grid

    # Get case weights lookup
    weight_lookup = caseinfo.set_index('MPOGCaseID')['weight'].to_dict()

    # Convert doses vectorized
    meds_df['uom_lower'] = meds_df['uom'].str.lower().str.strip()
    meds_df['is_infusion'] = meds_df['uom_lower'].str.contains('/', na=False)

    # Get concentration for compound meds (for ML/HR conversions) - VECTORIZED
    # Build a flat lookup: (med_name_lower, canonical) -> concentration
    # This avoids slow row-by-row .apply()
    concentration_lookup = {}
    for med_name, conc_dict in COMPOUND_MED_CONCENTRATIONS.items():
        for canonical, conc in conc_dict.items():
            concentration_lookup[(med_name, canonical)] = conc

    # Create lookup keys and map (vectorized)
    lookup_keys = list(zip(meds_df['med_name_lower'], meds_df['canonical_med']))
    meds_df['concentration_mcg_ml'] = pd.Series(lookup_keys).map(concentration_lookup).values

    # Apply unit conversions vectorized
    meds_df['converted_dose'] = np.nan

    for unit, (factor, weight_based) in UNIT_CONVERSIONS.items():
        mask = meds_df['uom_lower'] == unit
        if mask.any():
            doses = meds_df.loc[mask, 'dose'] * factor

            # Special handling for ML/HR - need concentration
            if unit in ['ml/hr', 'ml/min']:
                # For volume-based infusions, multiply by concentration to get drug dose
                # concentration is in mcg/ml, so result is mcg/min or mcg/hr
                conc = meds_df.loc[mask, 'concentration_mcg_ml'].fillna(0)
                doses = doses * conc * 1e-3  # Convert mcg to mg

            if weight_based:
                weights = meds_df.loc[mask, 'weight'].fillna(
                    meds_df.loc[mask, 'MPOGCaseID'].map(weight_lookup)
                ).fillna(75.0)
                doses = doses * weights
            meds_df.loc[mask, 'converted_dose'] = doses

    # Filter valid doses
    meds_df = meds_df[meds_df['converted_dose'].notna() & (meds_df['converted_dose'] > 0)].copy()

    if meds_df.empty:
        return grid

    # Process boluses (fast - just assign to minute)
    bolus_df = meds_df[~meds_df['is_infusion']].copy()
    if not bolus_df.empty:
        bolus_df['minute'] = bolus_df['dose_start'].dt.floor('min')

        # Aggregate boluses by case, minute, med
        bolus_agg = bolus_df.groupby(['MPOGCaseID', 'minute', 'canonical_med'])['converted_dose'].sum().reset_index()

        # Pivot to wide format
        for med in CANONICAL_MEDS:
            med_bolus = bolus_agg[bolus_agg['canonical_med'] == med]
            if not med_bolus.empty:
                col = f'meds_{med}'
                # Merge bolus doses
                merge_df = med_bolus[['MPOGCaseID', 'minute', 'converted_dose']].rename(
                    columns={'minute': 'obs_time', 'converted_dose': f'{col}_bolus'}
                )
                grid = grid.merge(merge_df, on=['MPOGCaseID', 'obs_time'], how='left')
                grid[col] = grid[col] + grid[f'{col}_bolus'].fillna(0)
                grid = grid.drop(columns=[f'{col}_bolus'])

    # Process infusions (vectorized expansion + merge)
    infusion_df = meds_df[meds_df['is_infusion'] & meds_df['dose_end'].notna()].copy()
    if not infusion_df.empty:
        # Filter valid infusions
        infusion_df = infusion_df[
            infusion_df['dose_start'].notna() &
            (infusion_df['dose_end'] > infusion_df['dose_start'])
        ].copy()

        if not infusion_df.empty:
            print(f"    Processing {len(infusion_df)} infusion records (vectorized)...")

            # Floor start/end times
            infusion_df['start_min'] = infusion_df['dose_start'].dt.floor('min')
            infusion_df['end_min'] = infusion_df['dose_end'].dt.floor('min')

            # Calculate number of minutes each infusion spans
            infusion_df['n_minutes'] = (
                (infusion_df['end_min'] - infusion_df['start_min']).dt.total_seconds() / 60
            ).astype(int).clip(lower=1)

            # Expand infusions to minute level (vectorized)
            n_infusions = len(infusion_df)
            mins_per_inf = infusion_df['n_minutes'].values

            # Only expand if total is reasonable (< 10M rows)
            total_inf_rows = mins_per_inf.sum()
            if total_inf_rows < 10_000_000:
                # Vectorized expansion - no Python loops
                inf_indices = np.repeat(np.arange(n_infusions), mins_per_inf)
                # Use cumsum trick instead of list comprehension
                cumulative_mins = np.arange(total_inf_rows)
                inf_starts = np.repeat(np.cumsum(np.concatenate([[0], mins_per_inf[:-1]])), mins_per_inf)
                minute_offsets = cumulative_mins - inf_starts

                expanded_inf = pd.DataFrame({
                    'MPOGCaseID': infusion_df['MPOGCaseID'].values[inf_indices],
                    'canonical_med': infusion_df['canonical_med'].values[inf_indices],
                    'converted_dose': infusion_df['converted_dose'].values[inf_indices],
                    'obs_time': infusion_df['start_min'].values[inf_indices] +
                               pd.to_timedelta(minute_offsets, unit='min')
                })

                # Aggregate by case, time, med
                inf_agg = expanded_inf.groupby(
                    ['MPOGCaseID', 'obs_time', 'canonical_med']
                )['converted_dose'].sum().reset_index()

                # Merge each med separately
                for med in CANONICAL_MEDS:
                    med_inf = inf_agg[inf_agg['canonical_med'] == med]
                    if not med_inf.empty:
                        col = f'meds_{med}'
                        merge_df = med_inf[['MPOGCaseID', 'obs_time', 'converted_dose']].rename(
                            columns={'converted_dose': f'{col}_inf'}
                        )
                        grid = grid.merge(merge_df, on=['MPOGCaseID', 'obs_time'], how='left')
                        grid[col] = grid[col] + grid[f'{col}_inf'].fillna(0)
                        grid = grid.drop(columns=[f'{col}_inf'])
            else:
                # Fallback for very large datasets: chunked vectorized processing
                print(f"    Large infusion set ({total_inf_rows:,} rows), using chunked approach...")
                chunk_size = 2_000_000  # Process 2M rows at a time

                # Process in chunks to limit memory usage
                chunk_starts = np.cumsum(np.concatenate([[0], mins_per_inf]))
                n_chunks = (total_inf_rows + chunk_size - 1) // chunk_size

                all_expanded = []
                for chunk_idx in range(n_chunks):
                    start_row = chunk_idx * chunk_size
                    end_row = min((chunk_idx + 1) * chunk_size, total_inf_rows)

                    # Find which infusions are in this chunk
                    first_inf = np.searchsorted(chunk_starts[1:], start_row, side='right')
                    last_inf = np.searchsorted(chunk_starts[:-1], end_row, side='left')

                    if first_inf >= last_inf:
                        continue

                    chunk_mins = mins_per_inf[first_inf:last_inf]
                    chunk_total = chunk_mins.sum()

                    if chunk_total == 0:
                        continue

                    chunk_indices = np.repeat(np.arange(first_inf, last_inf), chunk_mins)
                    cumulative = np.arange(chunk_total)
                    starts = np.repeat(np.cumsum(np.concatenate([[0], chunk_mins[:-1]])), chunk_mins)
                    offsets = cumulative - starts

                    chunk_df = pd.DataFrame({
                        'MPOGCaseID': infusion_df['MPOGCaseID'].values[chunk_indices],
                        'canonical_med': infusion_df['canonical_med'].values[chunk_indices],
                        'converted_dose': infusion_df['converted_dose'].values[chunk_indices],
                        'obs_time': infusion_df['start_min'].values[chunk_indices] +
                                   pd.to_timedelta(offsets, unit='min')
                    })
                    all_expanded.append(chunk_df)

                if all_expanded:
                    expanded_inf = pd.concat(all_expanded, ignore_index=True)
                    inf_agg = expanded_inf.groupby(
                        ['MPOGCaseID', 'obs_time', 'canonical_med']
                    )['converted_dose'].sum().reset_index()

                    for med in CANONICAL_MEDS:
                        med_inf = inf_agg[inf_agg['canonical_med'] == med]
                        if not med_inf.empty:
                            col = f'meds_{med}'
                            merge_df = med_inf[['MPOGCaseID', 'obs_time', 'converted_dose']].rename(
                                columns={'converted_dose': f'{col}_inf'}
                            )
                            grid = grid.merge(merge_df, on=['MPOGCaseID', 'obs_time'], how='left')
                            grid[col] = grid[col] + grid[f'{col}_inf'].fillna(0)
                            grid = grid.drop(columns=[f'{col}_inf'])

    return grid


def create_minute_by_minute_dataset(caseinfo: pd.DataFrame,
                                     bp_df: pd.DataFrame, spo2_df: pd.DataFrame,
                                     pulse_df: pd.DataFrame, etco2_df: pd.DataFrame,
                                     gas_dfs: dict, meds_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create the full minute-by-minute dataset - FULLY VECTORIZED.

    This version processes ALL cases at once using groupby operations
    instead of per-case loops. 10-100x faster than the loop version.
    """
    print("Creating minute-by-minute dataset (vectorized)...")

    # Step 1: Create minute grid for all cases
    grid = create_minute_grid_vectorized(caseinfo)

    if len(grid) == 0:
        return pd.DataFrame()

    # Step 2: Aggregate vitals - all vectorized
    print("  Aggregating vitals...")

    if not bp_df.empty:
        grid = aggregate_vitals_vectorized(grid, bp_df, 'bp_sys', 'bp_sys')
        grid = aggregate_vitals_vectorized(grid, bp_df, 'bp_dias', 'bp_dias')
        grid = aggregate_vitals_vectorized(grid, bp_df, 'bp_mean', 'bp_mean')
    else:
        grid['bp_sys'] = np.nan
        grid['bp_dias'] = np.nan
        grid['bp_mean'] = np.nan

    if not spo2_df.empty:
        grid = aggregate_vitals_vectorized(grid, spo2_df, 'spo2', 'spo2')
    else:
        grid['spo2'] = np.nan

    if not pulse_df.empty:
        grid = aggregate_vitals_vectorized(grid, pulse_df, 'pulse_rate', 'pulse_rate')
    else:
        grid['pulse_rate'] = np.nan

    if not etco2_df.empty:
        grid = aggregate_vitals_vectorized(grid, etco2_df, 'etco2', 'etco2')
    else:
        grid['etco2'] = np.nan

    # Anesthetic gases
    for gas_name, gas_df in gas_dfs.items():
        if gas_df is not None and not gas_df.empty:
            grid = aggregate_vitals_vectorized(grid, gas_df, gas_name, gas_name)
        else:
            grid[gas_name] = np.nan

    # Step 3: Process medications
    if meds_df is not None and not meds_df.empty:
        grid = process_medications_vectorized(grid, meds_df, caseinfo)
    else:
        for med in CANONICAL_MEDS:
            grid[f'meds_{med}'] = 0.0

    # Step 4: Calculate time_to_airway (vectorized)
    print("  Calculating derived columns...")
    grid['time_to_airway'] = np.where(
        grid['intubation_time'].notna(),
        (grid['intubation_time'] - grid['obs_time']).dt.total_seconds() / 60,
        np.nan
    )

    # Step 5: Add case_id (sequential)
    case_id_map = {cid: idx for idx, cid in enumerate(caseinfo['MPOGCaseID'].unique())}
    grid['case_id'] = grid['MPOGCaseID'].map(case_id_map)
    grid['mpog_case_id'] = grid['MPOGCaseID']

    # Step 6: Rename columns
    column_mapping = {
        'bp_sys': 'phys_bp_sys_non_invasive',
        'bp_dias': 'phys_bp_dias_non_invasive',
        'bp_mean': 'phys_bp_mean_non_invasive',
        'spo2': 'phys_spo2_%',
        'pulse_rate': 'phys_spo2_pulse_rate',
        'etco2': 'phys_end_tidal_co2_(mmhg)',
        'sevoflurane': 'phys_sevoflurane_exp_%',
        'isoflurane': 'phys_isoflurane_exp_%',
        'desflurane': 'phys_desflurane_exp_%',
        'nitrous': 'phys_nitrous_exp_%',
    }
    grid = grid.rename(columns=column_mapping)

    # Step 7: Drop internal columns
    grid = grid.drop(columns=['obs_time', 'MPOGCaseID', 'case_start', 'intubation_time'], errors='ignore')

    # Step 8: Convert dtypes
    for col in ['age', 'sex', 'weight', 'case_id']:
        if col in grid.columns:
            grid[col] = grid[col].astype('float32')

    grid['time_since_start'] = grid['time_since_start'].astype('int32')
    grid['minutes_elapsed'] = grid['time_since_start']  # Alias for ORacle-main compatibility

    for med in CANONICAL_MEDS:
        col = f'meds_{med}'
        if col in grid.columns:
            grid[col] = grid[col].astype('float32')

    grid['mpog_case_id'] = grid['mpog_case_id'].astype('str')

    print(f"  Final: {len(grid):,} rows, {grid['mpog_case_id'].nunique():,} cases")

    return grid


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def _extract_bp_wrapper(args):
    """Wrapper for parallel extraction."""
    base_path, case_ids = args
    return extract_vitals_bp(base_path, case_ids)

def _extract_spo2_wrapper(args):
    """Wrapper for parallel extraction."""
    base_path, case_ids = args
    return extract_vitals_spo2(base_path, case_ids)

def _extract_pulse_wrapper(args):
    """Wrapper for parallel extraction."""
    base_path, case_ids = args
    return extract_vitals_pulserate(base_path, case_ids)

def _extract_etco2_wrapper(args):
    """Wrapper for parallel extraction."""
    base_path, case_ids = args
    return extract_vitals_etco2(base_path, case_ids)

def _extract_gas_wrapper(args):
    """Wrapper for parallel gas extraction."""
    base_path, case_ids, gas_key, col_name = args
    try:
        return (col_name, extract_anesthetic_gas(base_path, case_ids, gas_key, col_name))
    except Exception:
        return (col_name, pd.DataFrame())

def _extract_meds_wrapper(args):
    """Wrapper for parallel extraction."""
    base_path, case_ids = args
    return extract_medications(base_path, case_ids)


# =============================================================================
# PREPROCESSING FUNCTIONS (from or_foundational)
# =============================================================================

def preprocess_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline matching or_foundational format.

    Steps:
    1. Create _flag columns (BEFORE any modifications)
    2. Apply sensible ranges (outliers -> NaN)
    3. Apply proxy filling (invasive BP -> non-invasive BP)
    4. Impute vitals (ffill -> bfill -> global median)
    5. Fill gases with 0
    6. Fill medications with 0
    7. Fill remaining NaN with 0
    8. Manual fixes (glycopyrrolate, weight=0, sex=0->1)
    9. Apply clip limits
    10. Create _bolus columns
    11. Truncate to 20hrs max
    12. Convert dtypes (float32)
    """
    from pandas.api.types import is_numeric_dtype

    if len(df) == 0:
        return df

    case_col = 'mpog_case_id' if 'mpog_case_id' in df.columns else 'case_id'
    time_col = 'time_since_start' if 'time_since_start' in df.columns else 'minutes_elapsed'

    if verbose:
        print(f"  Preprocessing {len(df):,} rows, {df[case_col].nunique():,} cases...")

    # Sort by case and time (required for ffill/bfill)
    df = df.sort_values([case_col, time_col]).reset_index(drop=True)

    # STEP 1: Create _flag columns BEFORE any modifications
    for col in df.columns:
        if col.endswith('_flag'):
            continue
        if is_numeric_dtype(df[col]) and (col.startswith('phys_') or col.startswith('meds_')):
            flag_col = f'{col}_flag'
            if flag_col not in df.columns:
                df[flag_col] = df[col].notna()

    # STEP 2: Apply sensible ranges (outliers -> NaN)
    for col, (min_val, max_val) in SENSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        flag_col = f'{col}_flag'
        outlier_mask = (df[col] < min_val) | (df[col] > max_val)
        df.loc[outlier_mask, col] = np.nan
        if flag_col in df.columns:
            df.loc[outlier_mask, flag_col] = False

    # STEP 3: Proxy filling (invasive BP -> non-invasive BP)
    for primary, proxy in PROXIES:
        if primary not in df.columns or proxy not in df.columns:
            continue
        flag_primary = f'{primary}_flag'
        flag_proxy = f'{proxy}_flag'
        if flag_primary in df.columns and flag_proxy in df.columns:
            need_fill = (df[flag_primary] == False)
            proxy_has = (df[flag_proxy] == True)
            can_fill = need_fill & proxy_has
            df.loc[can_fill, primary] = df.loc[can_fill, proxy]

    # Compute global medians for imputation
    global_medians = {}
    for col in IMPUTE_COLS + ['weight']:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                global_medians[col] = vals.median()
    for col, default in DEFAULT_MEDIANS.items():
        if col not in global_medians:
            global_medians[col] = default

    # STEP 4: Impute vitals (ffill -> bfill -> global median)
    for col in IMPUTE_COLS:
        if col not in df.columns or col in IMPUTE_EXCLUDE:
            continue
        # ffill then bfill within each case
        df[col] = df.groupby(case_col)[col].transform(lambda x: x.ffill().bfill())
        # Fill remaining with global median
        df[col] = df[col].fillna(global_medians.get(col, 0.0))

    # STEP 5: Fill gases with 0 (excluded from imputation)
    for col in IMPUTE_EXCLUDE:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # STEP 6: Fill medications with 0
    med_cols = [c for c in df.columns if c.startswith('meds_') and not c.endswith('_flag') and not c.endswith('_bolus')]
    for col in med_cols:
        df[col] = df[col].fillna(0.0)

    # STEP 7: Fill remaining NaN with 0 for numeric columns
    for col in df.columns:
        if col.endswith('_flag'):
            continue
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0.0)

    # STEP 8: Manual fixes
    # Fix glycopyrrolate > 1.0 (likely in mcg, divide by 1000)
    if 'meds_glycopyrrolate' in df.columns:
        mask = df['meds_glycopyrrolate'] > 1.0
        df.loc[mask, 'meds_glycopyrrolate'] = df.loc[mask, 'meds_glycopyrrolate'] / 1000

    # Fix weight = 0
    if 'weight' in df.columns:
        df.loc[df['weight'] == 0, 'weight'] = global_medians.get('weight', 75.0)

    # Normalize sex: 0 -> 1 (unknown -> male)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].replace(0, 1)

    # Replace inf with 0
    df = df.replace([np.inf, -np.inf], 0.0)

    # STEP 9: Apply clip limits
    for col, limit in CLIP_LIMITS.items():
        if col not in df.columns:
            continue
        if isinstance(limit, tuple):
            minn, maxx = limit
        else:
            minn, maxx = 0, limit
        df[col] = df[col].clip(minn, maxx)

    # STEP 10: Create _bolus columns
    for med_col in med_cols:
        if med_col not in df.columns:
            continue
        bolus_min = BOLUS_MINS.get(med_col, 1e-10)
        bolus_max = BOLUS_MAXS.get(med_col, 1e15)
        bolus_typicals = BOLUS_TYPICALS.get(med_col, [])

        dose_data = df[med_col].values
        bolus_type1 = dose_data > bolus_min
        bolus_type2 = np.zeros(len(dose_data), dtype=bool)
        for typical in bolus_typicals:
            bolus_type2 |= np.isclose(dose_data, typical, rtol=0.01)
        bolus_toohigh = bolus_type1 & (dose_data > bolus_max)
        is_bolus = (bolus_type1 | bolus_type2) & (~bolus_toohigh)

        bolus_col = f'{med_col}_bolus'
        df[bolus_col] = np.where(is_bolus, dose_data, 0.0)

    # STEP 11: Truncate to 20hrs max
    if time_col in df.columns:
        df = df[df[time_col] < MAX_CASE_MINUTES_FINAL].copy()

    # STEP 12: Convert dtypes to float32
    for col in df.columns:
        if df[col].dtype in [float, np.float64]:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype in [int, np.int64] and not col.endswith('_flag'):
            df[col] = df[col].astype(np.int32)

    if verbose:
        print(f"  Preprocessing complete: {len(df):,} rows, {df[case_col].nunique():,} cases")

    return df


def process_institution(base_path: str, institution: int, output_dir: str,
                        max_cases: Optional[int] = None,
                        force_reprocess: bool = False,
                        n_threads: int = 4) -> Tuple[int, str, int]:
    """
    Process a single institution and save to file.

    CHECKPOINTING: Each institution is saved as a separate file. If the script
    is interrupted, it will skip already-processed institutions on restart.
    Use force_reprocess=True to reprocess even if checkpoint exists.

    PARALLELISM: Uses ThreadPoolExecutor to extract different data types
    (BP, SpO2, pulse, gases, meds) in parallel within each institution.

    Returns: (institution_number, output_file_path, num_cases)
    """
    from concurrent.futures import ThreadPoolExecutor

    output_file = os.path.join(output_dir, f"institution_{institution}.feather")

    # Skip if already processed (checkpoint) - unless force_reprocess
    if not force_reprocess and os.path.exists(output_file):
        try:
            # Verify file is valid by reading metadata
            existing = pd.read_feather(output_file)
            num_cases = existing['mpog_case_id'].nunique()
            num_rows = len(existing)
            print(f"  Institution {institution}: CHECKPOINT EXISTS ({num_cases} cases, {num_rows:,} rows), skipping")
            return (institution, output_file, num_cases)
        except Exception as e:
            print(f"  Institution {institution}: Checkpoint corrupted ({e}), reprocessing...")

    print(f"\n{'='*60}")
    print(f"Processing Institution {institution} (using {n_threads} threads)")
    print(f"{'='*60}")

    # Extract case info for this institution
    caseinfo = extract_caseinfo(base_path, institution=institution)

    if len(caseinfo) == 0:
        print(f"  No valid cases for institution {institution}, skipping")
        return (institution, None, 0)

    if max_cases:
        caseinfo = caseinfo.head(max_cases)

    case_ids = caseinfo['MPOGCaseID'].tolist()
    print(f"  Extracting data for {len(case_ids)} cases in parallel...")

    # PARALLEL EXTRACTION: Extract all data types simultaneously
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Submit all extraction tasks
        bp_future = executor.submit(_extract_bp_wrapper, (base_path, case_ids))
        spo2_future = executor.submit(_extract_spo2_wrapper, (base_path, case_ids))
        pulse_future = executor.submit(_extract_pulse_wrapper, (base_path, case_ids))
        etco2_future = executor.submit(_extract_etco2_wrapper, (base_path, case_ids))
        meds_future = executor.submit(_extract_meds_wrapper, (base_path, case_ids))

        # Gas extractions
        gas_futures = []
        for gas_key, col_name in [('sevoflurane', 'sevoflurane'),
                                   ('isoflurane', 'isoflurane'),
                                   ('desflurane', 'desflurane'),
                                   ('nitrous', 'nitrous')]:
            gas_futures.append(
                executor.submit(_extract_gas_wrapper, (base_path, case_ids, gas_key, col_name))
            )

        # Collect results
        bp_df = bp_future.result()
        print(f"    BP: {len(bp_df):,} rows")

        spo2_df = spo2_future.result()
        print(f"    SpO2: {len(spo2_df):,} rows")

        pulse_df = pulse_future.result()
        print(f"    Pulse: {len(pulse_df):,} rows")

        etco2_df = etco2_future.result()
        print(f"    EtCO2: {len(etco2_df):,} rows")

        meds_df = meds_future.result()
        print(f"    Meds: {len(meds_df):,} rows")

        gas_dfs = {}
        for future in gas_futures:
            col_name, gas_df = future.result()
            gas_dfs[col_name] = gas_df
            print(f"    {col_name}: {len(gas_df):,} rows")

    # Create minute-by-minute dataset (vitals + medications) - vectorized
    print(f"  Creating minute-by-minute dataset (vectorized)...")
    final_df = create_minute_by_minute_dataset(
        caseinfo, bp_df, spo2_df, pulse_df, etco2_df, gas_dfs, meds_df
    )

    # Free memory from source dataframes (they've been processed into final_df)
    del bp_df, spo2_df, pulse_df, etco2_df, gas_dfs, meds_df, caseinfo
    clear_memory()

    if len(final_df) == 0:
        print(f"  No data produced for institution {institution}")
        return (institution, None, 0)

    # Add institution column with optimized dtype
    final_df['institution'] = np.int16(institution)

    # PREPROCESSING: Apply all or_foundational preprocessing steps
    print(f"  Applying preprocessing (or_foundational format)...")
    final_df = preprocess_dataframe(final_df, verbose=True)

    if len(final_df) == 0:
        print(f"  No data after preprocessing for institution {institution}")
        return (institution, None, 0)

    # Final dtype optimization pass
    final_df = optimize_dtypes(final_df, stage='final')

    # Report memory before saving
    mem_mb = final_df.memory_usage(deep=True).sum() / 1e6
    print(f"  Final DataFrame memory: {mem_mb:.1f} MB")

    # Save to file
    final_df.to_feather(output_file)
    num_cases = final_df['mpog_case_id'].nunique()
    print(f"  Institution {institution}: Saved {num_cases:,} cases, {len(final_df):,} rows")

    # Free memory after saving
    del final_df
    clear_memory()

    return (institution, output_file, num_cases)


def merge_institution_files(output_dir: str, final_output: str) -> pd.DataFrame:
    """Merge all institution files into a single dataset - memory efficient."""
    import glob

    pattern = os.path.join(output_dir, "institution_*.feather")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No institution files found in {output_dir}")

    print(f"\nMerging {len(files)} institution files...")

    # Memory-efficient loading: load one at a time and concatenate incrementally
    all_dfs = []
    total_rows = 0
    for f in tqdm(files, desc="Loading files"):
        try:
            df = pd.read_feather(f)
            total_rows += len(df)
            all_dfs.append(df)

            # Periodically clear memory if list gets large
            if len(all_dfs) % 10 == 0:
                clear_memory()
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")

    if not all_dfs:
        raise ValueError("No valid institution files to merge")

    print(f"  Total rows to merge: {total_rows:,}")

    # Concatenate all
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Free memory from individual dataframes
    del all_dfs
    clear_memory()

    # Reassign sequential case_id across all institutions
    unique_cases = final_df['mpog_case_id'].unique()
    case_id_map = {mpog_id: idx for idx, mpog_id in enumerate(unique_cases)}
    final_df['case_id'] = final_df['mpog_case_id'].map(case_id_map).astype('int32')

    # Final optimization pass
    final_df = optimize_dtypes(final_df, stage='final')

    mem_gb = final_df.memory_usage(deep=True).sum() / 1e9
    print(f"  Merged DataFrame: {len(final_df):,} rows, {final_df['mpog_case_id'].nunique():,} cases ({mem_gb:.2f} GB)")

    # Save merged file
    print(f"\nSaving merged dataset to {final_output}...")
    if final_output.endswith('.feather'):
        final_df.to_feather(final_output)
    elif final_output.endswith('.parquet'):
        # Parquet with compression for smaller file size
        final_df.to_parquet(final_output, index=False, compression='snappy')
    else:
        final_df.to_feather(final_output)

    return final_df


def run_pipeline(base_path: str, output_dir: str, final_output: str,
                 institution: Optional[int] = None,
                 max_cases: Optional[int] = None,
                 workers: int = 1,
                 n_threads: int = 8,
                 merge_only: bool = False,
                 force_reprocess: bool = False):
    """Run the full data processing pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PCRC 247 Dataset Pipeline (Vitals + Medications)")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print(f"Output dir: {output_dir}")
    print(f"Final output: {final_output}")
    print(f"Workers (institutions): {workers}")
    print(f"Threads (within institution): {n_threads}")
    print(f"Force reprocess: {force_reprocess}")
    print()

    if merge_only:
        # Just merge existing files
        final_df = merge_institution_files(output_dir, final_output)
    elif institution is not None:
        # Process single institution
        inst_num, out_file, num_cases = process_institution(
            base_path, institution, output_dir, max_cases, force_reprocess, n_threads
        )
        if out_file:
            print(f"\nSingle institution processed: {out_file}")
        return
    else:
        # Process all institutions
        institutions = get_all_institutions(base_path)

        # Show case counts
        print("\nCase counts per institution:")
        counts = get_institution_case_counts(base_path)
        print(counts.to_string(index=False))
        print()

        if workers > 1:
            # Parallel processing across institutions (less recommended)
            print(f"\nProcessing {len(institutions)} institutions with {workers} workers...")
            print(f"  (Each worker uses {n_threads} threads for extraction)")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_institution, base_path, inst, output_dir, max_cases, force_reprocess, n_threads): inst
                    for inst in institutions
                }

                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Institutions"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        inst = futures[future]
                        print(f"  Error processing institution {inst}: {e}")
        else:
            # Sequential processing - ONE institution at a time, parallel WITHIN
            print(f"\nProcessing {len(institutions)} institutions sequentially...")
            print(f"  (Using {n_threads} threads for parallel extraction within each)")
            results = []
            for inst in institutions:
                try:
                    result = process_institution(base_path, inst, output_dir, max_cases, force_reprocess, n_threads)
                    results.append(result)
                except Exception as e:
                    print(f"  Error processing institution {inst}: {e}")

        # Merge all institution files
        final_df = merge_institution_files(output_dir, final_output)

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total rows: {len(final_df):,}")
    print(f"Total cases: {final_df['mpog_case_id'].nunique():,}")
    print(f"Institutions: {final_df['institution'].nunique() if 'institution' in final_df.columns else 'N/A'}")
    print(f"Columns: {len(final_df.columns)}")
    print(f"\nOutput columns:")
    for col in sorted(final_df.columns):
        print(f"  - {col}")

    return final_df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PCRC 247 Dataset Pipeline (Vitals + Medications) - Institution-Based Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all institutions (auto-skips already completed):
  python pcrc_pipeline.py --base-path /nfs/turbo/pcrc247 --output-dir ./output

  # Process single institution:
  python pcrc_pipeline.py --base-path /nfs/turbo/pcrc247 --institution 123

  # Parallel processing with 4 workers:
  python pcrc_pipeline.py --base-path /nfs/turbo/pcrc247 --workers 4

  # Force reprocess all (ignore checkpoints):
  python pcrc_pipeline.py --base-path /nfs/turbo/pcrc247 --force

  # Just merge already-processed institution files:
  python pcrc_pipeline.py --merge-only --output-dir ./output

  # Test with limited cases per institution:
  python pcrc_pipeline.py --base-path /nfs/turbo/pcrc247 --max-cases 100
        """
    )

    parser.add_argument('--base-path', type=str,
                        help='Path to PCRC 247 Aghaeepour directory')
    parser.add_argument('--output-dir', type=str, default='./pcrc_output',
                        help='Directory for per-institution output files (default: ./pcrc_output)')
    parser.add_argument('--final-output', type=str, default='processed_pcrc.feather',
                        help='Final merged output file (default: processed_pcrc.feather)')
    parser.add_argument('--institution', type=int, default=None,
                        help='Process only this institution number')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for institutions (default: 1)')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads for parallel extraction WITHIN each institution (default: 8)')
    parser.add_argument('--max-cases', type=int, default=None,
                        help='Limit cases per institution (for testing)')
    parser.add_argument('--merge-only', action='store_true',
                        help='Only merge existing institution files, skip processing')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess all institutions (ignore checkpoints)')
    parser.add_argument('--list-institutions', action='store_true',
                        help='List all institutions and case counts, then exit')

    args = parser.parse_args()

    # Validation
    if not args.merge_only and not args.list_institutions and not args.base_path:
        parser.error("--base-path is required unless using --merge-only or --list-institutions")

    if args.list_institutions:
        if not args.base_path:
            parser.error("--base-path is required for --list-institutions")
        print("Institutions and case counts:")
        counts = get_institution_case_counts(args.base_path)
        print(counts.to_string(index=False))
        print(f"\nTotal institutions: {len(counts)}")
        print(f"Total valid cases: {counts['valid_cases'].sum():,}")
        return

    run_pipeline(
        base_path=args.base_path,
        output_dir=args.output_dir,
        final_output=args.final_output,
        institution=args.institution,
        max_cases=args.max_cases,
        workers=args.workers,
        n_threads=args.threads,
        merge_only=args.merge_only,
        force_reprocess=args.force
    )


if __name__ == '__main__':
    main()
