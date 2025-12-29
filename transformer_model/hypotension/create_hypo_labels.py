"""
Create hypotension onset labels for CV splits.

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

def create_hypo_labels(df, threshold_normalized, lookback=5, lookahead=3):
    """
    Create hypotension labels for a dataframe.

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

    # Add columns
    df = df.copy()
    df['hypo_onset_label'] = 0
    df['hypo_onset_type'] = 'none'

    # Sort by case and time
    df = df.sort_values(['mpog_case_id', 'time_since_start']).reset_index(drop=True)

    # Group by case
    grouped = df.groupby('mpog_case_id')

    onset_count = 0
    ongoing_count = 0
    none_count = 0

    for case_id, group in tqdm(grouped, desc="Processing cases"):
        indices = group.index.tolist()
        bp_values = group['phys_bp_mean_non_invasive'].values

        for i, idx in enumerate(indices):
            # Get last 5 BP values (or fewer if near start)
            start_lookback = max(0, i - lookback)
            last_5_bp = bp_values[start_lookback:i+1]

            # Get next 3 BP values (or fewer if near end)
            end_lookahead = min(len(bp_values), i + 1 + lookahead)
            next_3_bp = bp_values[i+1:end_lookahead]

            # Check if any of last 5 BP < threshold (ongoing hypotension)
            if len(last_5_bp) > 0 and np.any(last_5_bp < threshold_normalized):
                df.at[idx, 'hypo_onset_type'] = 'ongoing'
                df.at[idx, 'hypo_onset_label'] = 0  # Masked, so label doesn't matter
                ongoing_count += 1
                continue

            # Check for NEW onset: last 5 > threshold AND any of next 3 < threshold
            last_5_ok = len(last_5_bp) == 0 or np.all(last_5_bp >= threshold_normalized)
            next_3_hypo = len(next_3_bp) > 0 and np.any(next_3_bp < threshold_normalized)

            if last_5_ok and next_3_hypo:
                df.at[idx, 'hypo_onset_type'] = 'true_onset'
                df.at[idx, 'hypo_onset_label'] = 1
                onset_count += 1
            else:
                df.at[idx, 'hypo_onset_type'] = 'none'
                df.at[idx, 'hypo_onset_label'] = 0
                none_count += 1

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

    # Create labels
    df_labeled = create_hypo_labels(df, threshold_normalized)

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
    print("CREATING HYPOTENSION LABELS FOR CV SPLITS")
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
