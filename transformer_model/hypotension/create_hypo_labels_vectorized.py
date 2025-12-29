"""
Create hypotension onset labels for CV splits - VECTORIZED VERSION.

Label definition:
- hypo_onset_label = 1: Next 3 minutes BP mean < 65 mmHg AND last 5 BP > 65 mmHg (NEW onset)
- hypo_onset_type = "ongoing": Any of last 5 BP < 65 mmHg (masked during training)
- hypo_onset_label = 0: Otherwise (no hypotension)

Threshold: 65 mmHg on pre-normalized scale
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def load_scalers():
    """Load scalers to convert 65 mmHg to normalized value."""
    with open('/remote/home/marcgh/CV_intraop_model/cache/scalers.json', 'r') as f:
        scalers = json.load(f)

    bp_mean_scaler = scalers['phys_bp_mean_non_invasive']
    bp_min = bp_mean_scaler['min']
    bp_max = bp_mean_scaler['max']

    # Convert 65 mmHg to normalized [0-1] value
    threshold_normalized = (65.0 - bp_min) / (bp_max - bp_min)

    print(f"BP Mean scaler: min={bp_min:.2f}, max={bp_max:.2f}")
    print(f"65 mmHg threshold → normalized value: {threshold_normalized:.6f}")

    return threshold_normalized

def create_hypo_labels_vectorized(df, threshold_normalized, lookback=5, lookahead=3):
    """
    Create hypotension labels for a dataframe using vectorized operations.

    Args:
        df: DataFrame with normalized BP mean values
        threshold_normalized: Normalized threshold (65 mmHg in [0-1] scale)
        lookback: Number of previous timesteps to check (5)
        lookahead: Number of future timesteps to check (3)

    Returns:
        DataFrame with hypo_onset_label and hypo_onset_type columns
    """
    print(f"\nProcessing {len(df):,} rows...")
    print(f"Unique cases: {df['mpog_case_id'].nunique():,}")

    # Initialize columns
    df = df.copy()
    df['hypo_onset_label'] = 0
    df['hypo_onset_type'] = 'none'

    # Sort by case and time
    df = df.sort_values(['mpog_case_id', 'time_since_start']).reset_index(drop=True)

    # Create binary indicator: is BP < threshold?
    df['_is_hypo'] = (df['phys_bp_mean_non_invasive'] < threshold_normalized).astype(int)

    # For each case, compute rolling windows
    print("Computing rolling windows per case...")

    def process_case(group):
        """Process a single case with vectorized operations."""
        n = len(group)

        # Initialize arrays
        is_hypo = group['_is_hypo'].values
        hypo_type = np.full(n, 'none', dtype=object)
        hypo_label = np.zeros(n, dtype=int)

        # Compute lookback: any of last 5 BP < threshold?
        # Using cumsum trick for efficient rolling max
        ongoing = np.zeros(n, dtype=bool)
        for i in range(n):
            start_lookback = max(0, i - lookback)
            if np.any(is_hypo[start_lookback:i]):  # Check i-5 to i-1 (exclusive of i)
                ongoing[i] = True

        # Compute lookahead: any of next 3 BP < threshold?
        future_hypo = np.zeros(n, dtype=bool)
        for i in range(n):
            end_lookahead = min(n, i + 1 + lookahead)
            if np.any(is_hypo[i+1:end_lookahead]):  # Check i+1 to i+3
                future_hypo[i] = True

        # Apply labeling logic
        # 1. Mark ongoing
        hypo_type[ongoing] = 'ongoing'
        hypo_label[ongoing] = 0  # Will be masked

        # 2. Mark true onset (not ongoing AND future hypo)
        true_onset = (~ongoing) & future_hypo
        hypo_type[true_onset] = 'true_onset'
        hypo_label[true_onset] = 1

        # 3. Rest are already 'none' with label 0

        return pd.DataFrame({
            'hypo_onset_type': hypo_type,
            'hypo_onset_label': hypo_label
        }, index=group.index)

    # Process each case in parallel
    results = []
    for case_id, group in tqdm(df.groupby('mpog_case_id', sort=False), desc="Processing cases"):
        results.append(process_case(group))

    # Combine results
    labels_df = pd.concat(results)

    # Assign to original dataframe
    df['hypo_onset_type'] = labels_df['hypo_onset_type']
    df['hypo_onset_label'] = labels_df['hypo_onset_label']

    # Drop temporary column
    df = df.drop(columns=['_is_hypo'])

    # Print statistics
    onset_count = (df['hypo_onset_type'] == 'true_onset').sum()
    ongoing_count = (df['hypo_onset_type'] == 'ongoing').sum()
    none_count = (df['hypo_onset_type'] == 'none').sum()

    print(f"\nLabel distribution:")
    print(f"  true_onset (label=1): {onset_count:,} ({100*onset_count/len(df):.2f}%)")
    print(f"  ongoing (masked):     {ongoing_count:,} ({100*ongoing_count/len(df):.2f}%)")
    print(f"  none (label=0):       {none_count:,} ({100*none_count/len(df):.2f}%)")

    return df

def process_cv_split(cv_run, split_type, threshold_normalized):
    """Process a single CV split (train or test)."""
    base_path = Path('/remote/home/marcgh/CV_intraop_model/cv_splits_new')
    input_file = base_path / f'cv_run_{cv_run}' / f'{split_type}_NORMALIZED.feather'
    output_file = base_path / f'cv_run_{cv_run}' / f'{split_type}_NORMALIZED_with_hypo.feather'

    print(f"\n{'='*80}")
    print(f"Processing: CV Run {cv_run} - {split_type.upper()}")
    print(f"{'='*80}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    # Load data
    df = pd.read_feather(input_file)
    print(f"Loaded: {len(df):,} rows, {df['mpog_case_id'].nunique():,} cases")

    # Check BP mean column exists
    if 'phys_bp_mean_non_invasive' not in df.columns:
        print(f"ERROR: 'phys_bp_mean_non_invasive' not found in columns!")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    # Check BP mean range
    bp_mean = df['phys_bp_mean_non_invasive'].dropna()
    print(f"BP Mean range: [{bp_mean.min():.6f}, {bp_mean.max():.6f}], mean={bp_mean.mean():.6f}")

    # Create labels (vectorized)
    df_labeled = create_hypo_labels_vectorized(df, threshold_normalized)

    # Save
    df_labeled.to_feather(output_file)
    print(f"\n✓ Saved to: {output_file}")

    return {
        'cv_run': cv_run,
        'split': split_type,
        'total_rows': len(df_labeled),
        'total_cases': df_labeled['mpog_case_id'].nunique(),
        'true_onset': (df_labeled['hypo_onset_type'] == 'true_onset').sum(),
        'ongoing': (df_labeled['hypo_onset_type'] == 'ongoing').sum(),
        'none': (df_labeled['hypo_onset_type'] == 'none').sum(),
        'prevalence_onset': 100 * (df_labeled['hypo_onset_type'] == 'true_onset').sum() / len(df_labeled),
    }

def main():
    print("="*80)
    print("CREATING HYPOTENSION LABELS FOR CV SPLITS (VECTORIZED)")
    print("="*80)
    print("\nDefinition:")
    print("  - Label = 1 (true_onset): Next 3 min BP < 65 mmHg AND last 5 BP > 65 mmHg")
    print("  - Type = ongoing (masked): Any of last 5 BP < 65 mmHg")
    print("  - Label = 0 (none): Otherwise")
    print("")

    # Load threshold
    threshold_normalized = load_scalers()

    # Process all CV splits
    results = []
    for cv_run in [1, 2, 3]:
        for split_type in ['train', 'test']:
            result = process_cv_split(cv_run, split_type, threshold_normalized)
            if result:
                results.append(result)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: HYPOTENSION LABEL PREVALENCE")
    print("="*80)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # Save summary
    summary_file = '/remote/home/marcgh/CV_intraop_model/hypo_label_summary.csv'
    results_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the prevalence rates above")
    print("2. If satisfied, update configs to use *_with_hypo.feather files")
    print("3. Run training: python /remote/home/marcgh/CV_intraop_model/src/run_hypo_classifier.py")

if __name__ == '__main__':
    main()
