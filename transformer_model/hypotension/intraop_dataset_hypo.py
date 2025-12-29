import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import random

logger = logging.getLogger(__name__)

class IntraOpDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        vocabs: Dict[str, Any],
        split: str = "train",
        balance_hypo_finetune: bool = False,
        hypo_balance_ratio: float = 1.0
    ):
        """
        Dataset that keeps ALL data (including ongoing) with configurable early prediction.
        Masking happens in loss computation.
        """
        logger.info(f"Initializing IntraOpDataset with df rows={len(df)}, cases={df['mpog_case_id'].nunique()}")
        start_time = pd.Timestamp.now()
        
        self.df = df.copy()
        self.config = config
        self.vital_cols = config['vital_cols']
        self.med_cols = config['med_cols']
        self.bolus_cols = config.get('bolus_cols', [])
        self.gas_cols = config.get('gas_cols', [])  # NEW: Gas columns for anesthetic gases
        self.max_len = config['max_len']
        self.future_steps = config['future_steps']
        self.static_cat_cols = config.get('static_categoricals', [])
        self.static_num_cols = config.get('static_numericals', [])
        self.target_col = config.get('target_col', 'phys_bp_mean_non_invasive')
        self.vocab_maps = vocabs or {}
        self.split = split
        self.balance_hypo_finetune = balance_hypo_finetune
        self.hypo_balance_ratio = hypo_balance_ratio
        
        # NEW: Configurable minimum past steps for early prediction
        self.min_past_steps = config.get('min_past_steps', 1)  # Default to 1 for early prediction
        logger.info(f"🎯 Using min_past_steps = {self.min_past_steps} (can predict from t={self.min_past_steps})")
        
        # Track data composition for assessment
        self.data_stats = {
            'total_samples': len(self.df),
            'total_cases': df['mpog_case_id'].nunique(),
            'split': split,
            'min_past_steps': self.min_past_steps
        }
        
        if 'hypo_onset_type' in self.df.columns:
            type_dist = self.df['hypo_onset_type'].value_counts().to_dict()
            self.data_stats['onset_type_distribution'] = type_dist
            logger.info(f"✅ Data composition: {type_dist}")
        
        # Process static categorical columns
        for col in self.static_cat_cols:
            if col in self.df.columns and col in self.vocab_maps:
                self.df[col] = self.df[col].map(self.vocab_maps[col]).fillna(0).astype(int)

        # Filter cases by minimum length (updated for early prediction)
        # min_case_len = self.future_steps + self.min_past_steps  # Much smaller minimum now
        min_case_len = self.min_past_steps  # Much smaller minimum now



        case_lengths = self.df.groupby('mpog_case_id').size()
        valid_case_ids = case_lengths[case_lengths >= min_case_len].index.tolist()
        
        logger.info(f"📊 Cases: {len(case_lengths)} total, {len(valid_case_ids)} valid (≥{min_case_len} rows)")
        self.df = self.df[self.df['mpog_case_id'].isin(valid_case_ids)].copy()
        
        # Remove only clearly invalid labels
        if 'hypo_onset_label' in self.df.columns:
            valid_labels = self.df['hypo_onset_label'].isin([0, 1])
            invalid_count = (~valid_labels).sum()
            if invalid_count > 0:
                logger.info(f"Removing {invalid_count} invalid labels")
                self.df = self.df[valid_labels].copy()

        self.grouped = self.df.groupby('mpog_case_id')
        self.data_stats['final_cases'] = len(self.grouped)

        # Generate samples (keeping ALL types, with early prediction)
        logger.info("🎯 Generating samples...")
        all_samples = self._generate_samples()

        if balance_hypo_finetune and split == "train":
            self.samples = self._balance_hypo_samples(all_samples, ratio=hypo_balance_ratio)
            logger.info(f"⚖️ Balanced hypo-onset samples: {len(self.samples)} total")
        else:
            self.samples = all_samples

        self.data_stats['final_samples'] = len(self.samples)
        
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"✅ Dataset initialized in {elapsed:.2f}s with {len(self.samples)} samples")
        self._log_data_summary()

    def _generate_samples(self) -> List[Tuple[str, int]]:
        """Generate samples with early prediction capability - KEEP ALL ONSET TYPES"""
        samples = []
        stride = self.config.get('stride', 5)
        
        case_ids = list(self.grouped.groups.keys())
        logger.info(f"Processing {len(case_ids)} cases with stride={stride}, min_past_steps={self.min_past_steps}...")
        
        total_early_predictions = 0  # Track very early predictions (t≤5)
        
        for i, cid in enumerate(case_ids):
            if i % 1000 == 0:
                logger.info(f"Progress: {i}/{len(case_ids)} cases ({i/len(case_ids)*100:.1f}%)")
                
            group = self.grouped.get_group(cid)
            num_rows = len(group)
            # max_prediction_point = num_rows - self.future_steps add if you want to limit till end - future_steps
            max_prediction_point = num_rows 
            if max_prediction_point <= self.min_past_steps:
                continue

            # CHANGED: Start from min_past_steps instead of 10
            for end in range(self.min_past_steps, max_prediction_point, stride):
                current_idx = end - 1
                if current_idx < len(group):
                    existing_label = group.iloc[current_idx]['hypo_onset_label']
                    # KEEP ALL SAMPLES - no filtering by type
                    if existing_label in [0, 1]:
                        samples.append((cid, end))
                        
                        # Track early predictions
                        if end <= 5:
                            total_early_predictions += 1
        
        logger.info(f"Generated {len(samples)} samples (including all onset types)")
        logger.info(f"Early predictions (t≤5): {total_early_predictions}")
        logger.info(f"Earliest prediction timepoint: t={self.min_past_steps}")
        
        return samples

    def _balance_hypo_samples(self, samples, ratio=1.0):
        """Balance samples - consider masking in balancing logic"""
        pos, neg, ongoing = [], [], []
        
        for cid, end in samples:
            group = self.grouped.get_group(cid)
            current_idx = end - 1
            
            if current_idx < len(group):
                label = group.iloc[current_idx]['hypo_onset_label']
                onset_type = group.iloc[current_idx]['hypo_onset_type']
                
                if onset_type == 'ongoing':
                    ongoing.append((cid, end))
                elif label == 1:
                    pos.append((cid, end))
                elif label == 0:
                    neg.append((cid, end))

        logger.info(f"Sample composition: {len(pos)} positive, {len(neg)} negative, {len(ongoing)} ongoing")

        # Balance only true_onset vs none (exclude ongoing from balancing)
        if len(pos) > 0:
            n_neg = min(len(neg), int(len(pos) * ratio))
            neg_sampled = random.sample(neg, n_neg) if n_neg > 0 else []
            balanced_samples = pos + neg_sampled + ongoing  # Keep ongoing for masking analysis
        else:
            balanced_samples = neg + ongoing
            
        logger.info(f"Balanced: {len(pos)} positive, {len(neg_sampled) if len(pos) > 0 else len(neg)} negative, {len(ongoing)} ongoing (to be masked)")
        return balanced_samples

    def _log_data_summary(self):
        """Log data composition summary"""
        logger.info("\n" + "="*60)
        logger.info(f"📊 DATA SUMMARY - {self.split.upper()}")
        logger.info("="*60)
        logger.info(f"Total samples: {self.data_stats['final_samples']}")
        logger.info(f"Cases: {self.data_stats['final_cases']}")
        logger.info(f"Min past steps: {self.data_stats['min_past_steps']} (earliest prediction: t={self.min_past_steps})")
        if 'onset_type_distribution' in self.data_stats:
            for onset_type, count in self.data_stats['onset_type_distribution'].items():
                logger.info(f"  {onset_type}: {count}")
        logger.info("Note: All data kept - masking happens in loss computation")
        logger.info("="*60)

    def get_data_report(self) -> Dict[str, Any]:
        """Get data composition report for assessment"""
        return self.data_stats.copy()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cid, end = self.samples[idx]
        group = self.grouped.get_group(cid)
        
        # Calculate window boundaries
        start = max(0, end - self.max_len)
        window = group.iloc[start:end]
        seq_len = len(window)
        
        # Get the current timestep info for prediction
        current_idx = end - 1
        current_row = group.iloc[current_idx]
        
        # Initialize arrays with zeros
        vitals = np.zeros((self.max_len, len(self.vital_cols)), dtype=np.float32)
        meds = np.zeros((self.max_len, len(self.med_cols)), dtype=np.float32)
        bolus = np.zeros((self.max_len, len(self.bolus_cols)), dtype=np.float32) if self.bolus_cols else np.zeros((self.max_len, 0), dtype=np.float32)
        gases = np.zeros((self.max_len, len(self.gas_cols)), dtype=np.float32) if self.gas_cols else np.zeros((self.max_len, 0), dtype=np.float32)  # NEW
        attention_mask = torch.zeros(self.max_len, dtype=torch.bool)

        # Fill arrays with actual data (right-aligned)
        if seq_len > 0:
            vitals[-seq_len:] = window[self.vital_cols].values.astype(np.float32)
            meds[-seq_len:] = window[self.med_cols].values.astype(np.float32)
            if self.bolus_cols:
                bolus[-seq_len:] = window[self.bolus_cols].values.astype(np.float32)
            if self.gas_cols:  # NEW: Fill gases array
                gases[-seq_len:] = window[self.gas_cols].values.astype(np.float32)
            attention_mask[-seq_len:] = True

        # Get bolus values from current timestep
        bolus_values = np.zeros(len(self.bolus_cols), dtype=np.float32)
        if self.bolus_cols:
            for i, col in enumerate(self.bolus_cols):
                bolus_values[i] = float(current_row.get(col, 0.0))

        # Get future BP values
        future_start = end
        future_end = min(end + self.future_steps, len(group))
        if future_end > future_start:
            future_bp = group[self.target_col].iloc[future_start:future_end].values.astype(np.float32)
            if len(future_bp) < self.future_steps:
                padded_future = np.zeros(self.future_steps, dtype=np.float32)
                padded_future[:len(future_bp)] = future_bp
                future_bp = padded_future
        else:
            future_bp = np.zeros(self.future_steps, dtype=np.float32)
        
        target = torch.tensor(future_bp, dtype=torch.float32).unsqueeze(-1)

        # Static features
        static_cat = {}
        for col in self.static_cat_cols:
            if col in current_row.index:
                vocab_map = self.vocab_maps.get(col, {})
                value = vocab_map.get(current_row[col], 0)
                static_cat[col] = torch.tensor(value, dtype=torch.long)
            else:
                static_cat[col] = torch.tensor(0, dtype=torch.long)
                
        static_num = None
        if self.static_num_cols:
            num_values = []
            for col in self.static_num_cols:
                if col in current_row.index:
                    num_values.append(float(current_row[col]))
                else:
                    num_values.append(0.0)
            static_num = torch.tensor(num_values, dtype=torch.float32)

        # Handle BP features with early prediction considerations
        if self.target_col in self.vital_cols:
            bp_index = self.vital_cols.index(self.target_col)
            if seq_len > 0:
                actual_bp_values = vitals[-seq_len:, bp_index]
                # For early predictions, we might have fewer than 5 BP values
                if seq_len >= 5:
                    last5_bp = actual_bp_values[-5:].tolist()
                else:
                    # Pad with zeros at the beginning for early predictions
                    last5_bp = [0.0] * (5 - seq_len) + actual_bp_values.tolist()
            else:
                last5_bp = [0.0] * 5
        else:
            last5_bp = [0.0] * 5

        # Get labels (keeping ALL types)
        hypo_onset_label = float(current_row['hypo_onset_label'])
        hypo_onset_type = current_row['hypo_onset_type']

        return {
            'vitals': torch.tensor(vitals, dtype=torch.float32),
            'meds': torch.tensor(meds, dtype=torch.float32),
            'bolus': torch.tensor(bolus, dtype=torch.float32),
            'gases': torch.tensor(gases, dtype=torch.float32),  # NEW: Anesthetic gases
            'last_input_bolus': torch.tensor(bolus_values, dtype=torch.float32),
            'target': target,
            'attention_mask': attention_mask,
            'static_cat': static_cat,
            'static_num': static_num,
            'original_index': torch.tensor(current_row.name, dtype=torch.long),
            'minutes_elapsed': torch.tensor(float(current_row.get('time_since_start', 0)), dtype=torch.float32),
            'mpog_case_id': cid,
            'last5_bp_values': torch.tensor(last5_bp, dtype=torch.float32),
            'last_known_bp': torch.tensor(last5_bp[-1], dtype=torch.float32),
            'hypo_onset_label': torch.tensor(hypo_onset_label, dtype=torch.float32),
            'hypo_onset_type': hypo_onset_type,  # Keep as string
            # Assessment fields
            'actual_sequence_length': torch.tensor(seq_len, dtype=torch.long),
            'prediction_timestep': torch.tensor(end, dtype=torch.long),
            'current_bp': torch.tensor(float(current_row[self.target_col]), dtype=torch.float32),
            'min_future_bp': torch.tensor(future_bp.min(), dtype=torch.float32),
            'dataset_split': self.split,
            # NEW: Early prediction tracking
            'prediction_time_point': torch.tensor(end, dtype=torch.long),
            'is_early_prediction': torch.tensor(end <= 5, dtype=torch.bool),
            'available_history_length': torch.tensor(seq_len, dtype=torch.long),
            'has_limited_history': torch.tensor(seq_len < 5, dtype=torch.bool),
        }

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        keys = batch[0].keys()
        
        for key in keys:
            if key == "static_cat":
                result[key] = {}
                for cat_key in batch[0][key].keys():
                    result[key][cat_key] = torch.stack([item[key][cat_key] for item in batch])
            elif key == "static_num":
                static_nums = [item[key] for item in batch if item[key] is not None]
                if static_nums:
                    result[key] = torch.stack(static_nums)
                else:
                    result[key] = None
            elif key in ["hypo_onset_type", "mpog_case_id", "dataset_split"]:
                result[key] = [item[key] for item in batch]
            else:
                result[key] = torch.stack([item[key] for item in batch])
        return result
