import torch
import numpy as np
import logging
from intraop_dataset import IntraOpDataset

logger = logging.getLogger(__name__)

class MultiTargetDataset(IntraOpDataset):
    """Dataset for multi-target classification with 4 binary targets."""

    def __init__(self, df, config, vocabs, split="train", balance_hypo_finetune=False, hypo_balance_ratio=1.0):
        # Initialize parent class with all expected arguments
        super().__init__(df, config, vocabs, split, balance_hypo_finetune, hypo_balance_ratio)

        # Get target columns from config
        self.classification_target_cols = config.get("target_classification_cols", [
            "y_onset_next3_after5normo",
            "y_drop30pct_vs_baseline_next3",
            "y_event_before_hypo_next3",
            "y_rise20pct_within3_from_current"
        ])

        # Keep the original target_cols for the parent class
        # (it needs these for autoregressive prediction structure)

        # Get mask column
        self.mask_col = config.get("mask_col", "m_recent_hypo_5min")

        # Log target distributions
        for target_col in self.classification_target_cols:
            if target_col in self.df.columns:
                values = self.df[target_col].values
                unique, counts = np.unique(values[~np.isnan(values)], return_counts=True)
                # Convert scaled values back to meaningful labels
                binary_dict = {}
                for val, count in zip(unique, counts):
                    if val < 0.5:
                        key = 0
                    else:
                        key = 1
                    binary_dict[key] = binary_dict.get(key, 0) + count
                logger.info(f"🎯 {target_col} distribution: {binary_dict}")
            else:
                logger.warning(f"⚠️ Target column {target_col} not found in dataframe!")

        # Log mask distribution
        if self.mask_col in self.df.columns:
            mask_values = self.df[self.mask_col].values
            unique, counts = np.unique(mask_values[~np.isnan(mask_values)], return_counts=True)
            logger.info(f"😷 Mask {self.mask_col} distribution: {dict(zip(unique.astype(int), counts))}")

    def __getitem__(self, idx):
        # Get base sample from parent class
        sample = super().__getitem__(idx)

        # Get the sample info to find corresponding row in original dataframe
        # samples are stored as tuples (case_id, end_idx)
        case_id, end_idx = self.samples[idx]

        # Get the row data for this sample (last timestep)
        case_data = self.df[self.df['mpog_case_id'] == case_id]
        if len(case_data) > end_idx:
            last_row = case_data.iloc[end_idx]
        else:
            # Fallback to last available row
            last_row = case_data.iloc[-1] if len(case_data) > 0 else None

        # Extract multi-target labels from the last timestep
        multi_targets = []
        for target_col in self.classification_target_cols:
            if last_row is not None and target_col in last_row.index:
                target_value = float(last_row.get(target_col, 0))
                # Convert scaled values to binary (0.99999... -> 1, 0.0 -> 0)
                target_value = 1.0 if target_value > 0.5 else 0.0
            else:
                target_value = 0.0
            multi_targets.append(target_value)

        # Extract mask value
        if last_row is not None and self.mask_col in last_row.index:
            mask_value = float(last_row.get(self.mask_col, 1))
        else:
            mask_value = 1.0  # Default to valid if no mask

        # Add multi-target labels to sample
        sample['multi_target_labels'] = torch.tensor(multi_targets, dtype=torch.float32)
        sample['multi_target_mask'] = torch.tensor(mask_value, dtype=torch.float32)

        # Add individual target labels for easy access
        for i, target_col in enumerate(self.classification_target_cols):
            sample[target_col] = torch.tensor(multi_targets[i], dtype=torch.float32)

        # Add mask for easy access
        sample[self.mask_col] = torch.tensor(mask_value, dtype=torch.float32)

        # Also keep the original hypo_onset_label for compatibility
        # Use the first target as a proxy
        if len(multi_targets) > 0:
            sample['hypo_onset_label'] = torch.tensor(multi_targets[0], dtype=torch.float32)

        return sample