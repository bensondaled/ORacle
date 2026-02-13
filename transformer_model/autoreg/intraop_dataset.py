import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import random
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class Normalizer:
    """
    Handles normalization and denormalization of features.

    Supports:
    - MinMax: (x - min) / (max - min) -> [0, 1]
    - ZScore: (x - mean) / std -> ~N(0, 1)
    """

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('normalization', {}).get('enabled', False)
        self.method = config.get('normalization', {}).get('method', 'minmax')
        self.stats = {}

        if not self.enabled:
            return

        # Try to load stats from file
        stats_file = config.get('normalization', {}).get('stats_file')
        if stats_file and Path(stats_file).exists():
            logger.info(f"Loading normalization stats from: {stats_file}")
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            # Use clinical fallback ranges
            logger.info("Using clinical fallback ranges for normalization")
            clinical_ranges = config.get('normalization', {}).get('clinical_ranges', {})
            for col, ranges in clinical_ranges.items():
                self.stats[col] = {
                    'min': ranges.get('min', 0),
                    'max': ranges.get('max', 1),
                    'mean': (ranges.get('min', 0) + ranges.get('max', 1)) / 2,
                    'std': (ranges.get('max', 1) - ranges.get('min', 0)) / 4
                }

        logger.info(f"Normalization enabled: method={self.method}, columns={len(self.stats)}")

    def normalize(self, values: np.ndarray, col_name: str) -> np.ndarray:
        """Normalize values for a given column."""
        if not self.enabled or col_name not in self.stats:
            return values

        s = self.stats[col_name]

        if self.method == 'minmax':
            denom = s['max'] - s['min']
            if denom == 0:
                denom = 1
            return (values - s['min']) / denom
        elif self.method == 'zscore':
            if s['std'] == 0:
                return values - s['mean']
            return (values - s['mean']) / s['std']
        else:
            return values

    def denormalize(self, values: np.ndarray, col_name: str) -> np.ndarray:
        """Denormalize values back to original scale."""
        if not self.enabled or col_name not in self.stats:
            return values

        s = self.stats[col_name]

        if self.method == 'minmax':
            return values * (s['max'] - s['min']) + s['min']
        elif self.method == 'zscore':
            return values * s['std'] + s['mean']
        else:
            return values

    def normalize_df_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize specific columns in a DataFrame (in-place)."""
        if not self.enabled:
            return df

        for col in columns:
            if col in df.columns and col in self.stats:
                df[col] = self.normalize(df[col].values, col)

        return df

    def get_stats(self, col_name: str) -> Optional[Dict[str, float]]:
        """Get normalization stats for a column."""
        return self.stats.get(col_name)

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
        logger.info(f"Initializing IntraOpDataset with df rows={len(df)}, cases={df['mpog_case_id'].nunique()}")
        self.df = df.copy()
        self.config = config
        self.vital_cols = config['vital_cols']
        self.med_cols = config['med_cols']
        self.gas_cols = config.get('gas_cols', [])  # FIXED: Add gas columns
        self.bolus_cols = config.get('bolus_cols', [])
        self.max_len = config['max_len']
        self.future_steps = config['future_steps']
        self.static_cat_cols = config.get('static_categoricals', [])
        self.static_num_cols = config.get('static_numericals', [])
        self.patient_id_col = config.get('patient_id_col', 'mpog_case_id') # Add this line
        self.target_cols = config.get('target_cols', ['phys_bp_mean_non_invasive'])

        # Flag columns for imputation masking
        self.vital_flag_cols = config.get('vital_flag_cols', [])
        self.target_flag_cols = config.get('target_flag_cols', [])
        self.use_imputation_masking = config.get('use_imputation_masking', False)

        # Initialize normalizer
        self.normalizer = Normalizer(config)
        self.vocab_maps = vocabs or {}
        self.split = split
        self.balance_hypo_finetune = balance_hypo_finetune
        self.hypo_balance_ratio = hypo_balance_ratio

        self.hypo_onset_labels = np.array(self.df.get("hypo_onset_label", [0] * len(self.df)), dtype=np.float32)
        self.hypo_onset_types = np.array(self.df.get("hypo_onset_type", ["none"] * len(self.df)), dtype=str)

        unique_labels, counts = np.unique(self.hypo_onset_labels, return_counts=True)
        logger.info(f"🦧 hypo_onset_label counts: {dict(zip(unique_labels.astype(int), counts))}")
        unique_types, type_counts = np.unique(self.hypo_onset_types, return_counts=True)
        logger.info(f"🦧 hypo_onset_type counts: {dict(zip(unique_types, type_counts))}")

        # Validate required columns
        required_cols = set(self.vital_cols + self.med_cols + self.gas_cols + self.target_cols + self.bolus_cols)
        if 'mpog_case_id' not in self.df.columns:
            raise ValueError("Missing 'mpog_case_id' column")
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        # Check for flag columns if imputation masking is enabled
        if self.use_imputation_masking:
            available_vital_flags = [c for c in self.vital_flag_cols if c in self.df.columns]
            available_target_flags = [c for c in self.target_flag_cols if c in self.df.columns]
            if available_vital_flags:
                logger.info(f"Found {len(available_vital_flags)} vital flag columns for imputation masking")
                self.vital_flag_cols = available_vital_flags
            else:
                logger.warning("No vital flag columns found, imputation masking disabled for inputs")
                self.vital_flag_cols = []
            if available_target_flags:
                logger.info(f"Found {len(available_target_flags)} target flag columns for imputation masking")
                self.target_flag_cols = available_target_flags
            else:
                logger.warning("No target flag columns found, imputation masking disabled for targets")
                self.target_flag_cols = []

        # Process static categorical columns
        for col in self.static_cat_cols:
            if col in self.df.columns and col in self.vocab_maps:
                self.df[col] = self.df[col].map(self.vocab_maps[col]).fillna(0).astype(int)
            else:
                logger.warning(f"Skipping static categorical column {col}: missing in df or vocabs")

        # Filter cases with sufficient length for early prediction
        # OPTIMIZED: Use groupby.size() instead of iterating with get_group()
        min_len = self.future_steps + 1
        logger.info(f"🔍 Filtering cases (min_len={min_len})...")

        # Fast: get all group sizes at once
        group_sizes = self.df.groupby('mpog_case_id').size()
        n_total_cases = len(group_sizes)

        # Fast: filter by size
        valid_case_ids = group_sizes[group_sizes >= min_len].index
        n_valid_cases = len(valid_case_ids)
        logger.info(f"🔍 Valid cases: {n_valid_cases:,}/{n_total_cases:,} ({n_valid_cases/n_total_cases*100:.1f}%)")

        # Filter DataFrame
        self.df = self.df[self.df['mpog_case_id'].isin(valid_case_ids)]
        logger.info(f"🔍 Filtered DataFrame: {len(self.df):,} rows")

        # OPTIMIZED: Pre-compute case boundaries for fast sample generation
        # Sort by case_id for efficient indexing
        self.df = self.df.sort_values(['mpog_case_id']).reset_index(drop=True)

        # Compute start/end indices for each case
        case_ids_array = self.df['mpog_case_id'].values
        case_boundaries = {}
        current_case = None
        start_idx = 0

        for i, cid in enumerate(case_ids_array):
            if cid != current_case:
                if current_case is not None:
                    case_boundaries[current_case] = (start_idx, i)
                current_case = cid
                start_idx = i
        if current_case is not None:
            case_boundaries[current_case] = (start_idx, len(case_ids_array))

        self.case_boundaries = case_boundaries
        logger.info(f"🔍 Indexed {len(case_boundaries):,} cases")

        all_samples = self._generate_samples()

        if balance_hypo_finetune and split == "train":
            self.samples = self._balance_hypo_samples(all_samples, ratio=hypo_balance_ratio)
            logger.info(f"✅ Balanced hypo-onset samples: {len(self.samples)} total")
        else:
            self.samples = all_samples

        unique_cases_in_samples = len(set(cid for cid, _ in self.samples))
        logger.info(f"📦 Generated {len(self.samples)} samples across {unique_cases_in_samples} cases")

    def _generate_samples(self) -> List[Tuple[str, int]]:
        """Generate (case_id, end_idx) samples. OPTIMIZED using pre-computed boundaries."""
        samples = []
        stride = self.config.get('stride', 1)
        required_past_steps = self.config.get('min_history_steps', 5)

        case_ids = list(self.case_boundaries.keys())
        n_cases = len(case_ids)
        logger.info(f"🔍 Generating samples for {n_cases:,} cases (stride={stride})...")

        # OPTIMIZED: Use pre-computed boundaries, no get_group() calls
        for i, cid in enumerate(case_ids):
            start_idx, end_idx = self.case_boundaries[cid]
            num_rows = end_idx - start_idx
            max_end = num_rows - self.future_steps + 1

            if max_end <= required_past_steps:
                continue

            # Generate sample indices
            for end in range(required_past_steps, max_end, stride):
                samples.append((cid, end))

            # Progress logging every 50k cases
            if (i + 1) % 50000 == 0:
                logger.info(f"🔍 Processed {i+1:,}/{n_cases:,} cases, {len(samples):,} samples so far")

        logger.info(f"🔍 Generated {len(samples):,} total samples")
        return samples

    def _get_case_data(self, cid: str) -> pd.DataFrame:
        """Get data for a case using pre-computed boundaries. OPTIMIZED."""
        case_start, case_end = self.case_boundaries[cid]
        return self.df.iloc[case_start:case_end]

    def _balance_hypo_samples(self, samples, ratio=1.0):
        pos, neg = [], []

        for cid, end in samples:
            group = self._get_case_data(cid)
            start = max(0, end - self.max_len)
            window = group.iloc[start:end]

            # Get label from the last timestep in window
            if len(window) > 0:
                label = float(window.iloc[-1].get("hypo_onset_label", 0))
            else:
                label = 0.0

            if label == 1:
                pos.append((cid, end))
            else:
                neg.append((cid, end))

        # Balance the samples
        if len(pos) > 0:
            n_neg = min(len(neg), int(len(pos) * ratio))
            neg_sampled = random.sample(neg, n_neg) if n_neg > 0 else []
            balanced_samples = pos + neg_sampled
        else:
            # If no positive samples, take all negative samples
            balanced_samples = neg

        logger.info(f"Balanced samples: {len(pos)} positive, {len(neg_sampled) if len(pos) > 0 else len(neg)} negative")
        return balanced_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cid, end = self.samples[idx]
        group = self._get_case_data(cid)  # OPTIMIZED: Use pre-computed boundaries

        # Calculate window boundaries
        start = max(0, end - self.max_len)
        window = group.iloc[start:end]
        future = group.iloc[end:end + self.future_steps]
        seq_len = len(window)
        
        # PERFORMANCE: Removed expensive memory monitoring (every 1000 samples)
        # This was causing slowdown during data loading
        
        # DTYPE OPTIMIZED: Use float16 for input data to save memory and bandwidth
        vitals = np.zeros((self.max_len, len(self.vital_cols)), dtype=np.float16)
        meds = np.zeros((self.max_len, len(self.med_cols)), dtype=np.float16)
        gases = np.zeros((self.max_len, len(self.gas_cols)), dtype=np.float16) if self.gas_cols else np.zeros((self.max_len, 0), dtype=np.float16)
        bolus = np.zeros((self.max_len, len(self.bolus_cols)), dtype=np.float16) if self.bolus_cols else np.zeros((self.max_len, 0), dtype=np.float16)
        attention_mask = torch.zeros(self.max_len, dtype=torch.bool)

        # Initialize vital flags (True = measured, False = imputed/masked)
        # Default to True (all measured) if no flag columns available
        vital_flags = np.ones((self.max_len, len(self.vital_flag_cols)), dtype=np.bool_) if self.vital_flag_cols else None

        # Fill arrays with actual data (right-aligned)
        if seq_len > 0:
            # DTYPE OPTIMIZED: Use float16 for input data
            vital_values = window[self.vital_cols].values.astype(np.float32)

            # Apply normalization to vitals
            if self.normalizer.enabled:
                for i, col in enumerate(self.vital_cols):
                    vital_values[:, i] = self.normalizer.normalize(vital_values[:, i], col)

            vitals[-seq_len:] = vital_values.astype(np.float16)
            meds[-seq_len:] = window[self.med_cols].values.astype(np.float16)

            if self.gas_cols:
                gases[-seq_len:] = window[self.gas_cols].values.astype(np.float16)

            if self.bolus_cols:
                bolus[-seq_len:] = window[self.bolus_cols].values.astype(np.float16)

            attention_mask[-seq_len:] = True

            # Load vital flags and apply imputation masking
            if self.use_imputation_masking and self.vital_flag_cols:
                # Load flag values (True = measured, False = imputed)
                flag_values = window[self.vital_flag_cols].values.astype(np.bool_)
                vital_flags[-seq_len:] = flag_values

                # Mask imputed values in vitals (set to 0 where flag is False)
                # Only mask the columns that have corresponding flags
                for i, flag_col in enumerate(self.vital_flag_cols):
                    # Find the corresponding vital column
                    vital_col = flag_col.replace('_flag', '')
                    if vital_col in self.vital_cols:
                        vital_idx = self.vital_cols.index(vital_col)
                        # Set imputed values to 0
                        imputed_mask = ~flag_values[:, i]
                        vitals[-seq_len:, vital_idx][imputed_mask] = 0.0

        # Get bolus values from the current timestep (last row in window)
        bolus_values = np.zeros(len(self.bolus_cols), dtype=np.float16)
        if self.bolus_cols and seq_len > 0:
            last_row = window.iloc[-1]
            for i, col in enumerate(self.bolus_cols):
                bolus_values[i] = float(last_row.get(col, 0.0))

        # DTYPE OPTIMIZED: Use float16 for targets (will be upcast to float32 during loss computation)
        if len(future) != self.future_steps:
            # Pad or truncate as needed
            future_values = np.zeros((self.future_steps, len(self.target_cols)), dtype=np.float32)
            actual_len = min(len(future), self.future_steps)
            if actual_len > 0:
                future_values[:actual_len] = future[self.target_cols].iloc[:actual_len].values.astype(np.float32)
            target_array = future_values
        else:
            target_array = future[self.target_cols].values.astype(np.float32)

        # Apply normalization to targets
        if self.normalizer.enabled:
            for i, col in enumerate(self.target_cols):
                target_array[:, i] = self.normalizer.normalize(target_array[:, i], col)

        target = torch.tensor(np.nan_to_num(target_array, nan=0.0), dtype=torch.float32)

        # Initialize target flags (True = measured, False = imputed/masked)
        # Default to True (all measured) if no flag columns available
        target_flags = np.ones((self.future_steps, len(self.target_flag_cols)), dtype=np.bool_) if self.target_flag_cols else None

        # Load target flags for future timesteps
        if self.use_imputation_masking and self.target_flag_cols:
            if len(future) != self.future_steps:
                # Pad flags with False (treat as imputed) for padded timesteps
                target_flags = np.zeros((self.future_steps, len(self.target_flag_cols)), dtype=np.bool_)
                actual_len = min(len(future), self.future_steps)
                if actual_len > 0:
                    target_flags[:actual_len] = future[self.target_flag_cols].iloc[:actual_len].values.astype(np.bool_)
            else:
                target_flags = future[self.target_flag_cols].values.astype(np.bool_)

        # Static features from current timestep
        current_row_idx = end - 1
        current_row = group.iloc[current_row_idx]
        
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

        # Handle BP-related features
        if self.target_cols[0] in self.vital_cols:
            bp_index = self.vital_cols.index(self.target_cols[0])
            
            # Get actual BP values from non-padded part
            if seq_len > 0:
                actual_bp_values = vitals[-seq_len:, bp_index]
                # Get last 5 values, pad with zeros if needed
                if seq_len >= 5:
                    last5_bp = actual_bp_values[-5:].tolist()
                else:
                    last5_bp = [0.0] * (5 - seq_len) + actual_bp_values.tolist()
            else:
                last5_bp = [0.0] * 5
        else:
            last5_bp = [0.0] * 5

        # Labels from current timestep
        if seq_len > 0:
            hypo_onset_label = float(window.iloc[-1].get("hypo_onset_label", 0))
            hypo_onset_type = str(window.iloc[-1].get("hypo_onset_type", "none"))
        else:
            hypo_onset_label = 0.0
            hypo_onset_type = "none"

        # Handle UUID strings by converting to hash
        case_id_value = current_row[self.patient_id_col]
        if isinstance(case_id_value, str):
            patient_id = torch.tensor(hash(case_id_value) % (2**31), dtype=torch.long)
        else:
            patient_id = torch.tensor(int(case_id_value), dtype=torch.long)

        # DTYPE OPTIMIZED: Create float16 tensors for memory efficiency (autocast will handle precision)
        result = {
            'vitals': torch.from_numpy(vitals),  # float16
            'meds': torch.from_numpy(meds),     # float16
            'gases': torch.from_numpy(gases),   # float16
            'bolus': torch.from_numpy(bolus),   # float16
            'last_input_bolus': torch.from_numpy(bolus_values),  # float16
            'target': target,
            'attention_mask': attention_mask,
            'static_cat': static_cat,
            'static_num': static_num,
            'patient_id': patient_id,
            'original_index': torch.tensor(current_row.name, dtype=torch.long),
            'minutes_elapsed': torch.tensor(float(current_row.get('time_since_start', 0)), dtype=torch.float32),
            'mpog_case_id': cid,
            'last5_bp_values': torch.from_numpy(np.array(last5_bp, dtype=np.float16)),
            'last_known_bp': torch.tensor(last5_bp[-1], dtype=torch.float32),
            'hypo_onset_label': torch.tensor(hypo_onset_label, dtype=torch.float32),
            'hypo_onset_type': hypo_onset_type,
            'actual_sequence_length': torch.tensor(seq_len, dtype=torch.long),
            'prediction_timestep': torch.tensor(end, dtype=torch.long),
        }

        # Add flag tensors for imputation masking (True = measured, False = imputed)
        if vital_flags is not None:
            result['vital_flags'] = torch.from_numpy(vital_flags)  # [max_len, num_flag_cols]
        if target_flags is not None:
            result['target_flags'] = torch.from_numpy(target_flags)  # [future_steps, num_flag_cols]

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def denormalize_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Denormalize model predictions back to original scale.

        Args:
            preds: Predictions [B, T, N] or [B, T] where N is number of target columns

        Returns:
            Denormalized predictions in original units (e.g., mmHg for BP)
        """
        if not self.normalizer.enabled:
            return preds

        preds_np = preds.cpu().numpy()
        result = np.zeros_like(preds_np)

        if preds_np.ndim == 2:
            # Single target [B, T]
            col = self.target_cols[0]
            result = self.normalizer.denormalize(preds_np, col)
        else:
            # Multi-target [B, T, N]
            for i, col in enumerate(self.target_cols):
                if i < preds_np.shape[-1]:
                    result[..., i] = self.normalizer.denormalize(preds_np[..., i], col)

        return torch.tensor(result, dtype=preds.dtype, device=preds.device)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        keys = batch[0].keys()

        for key in keys:
            if key == "static_cat":
                # Handle static categorical features
                result[key] = {}
                for cat_key in batch[0][key].keys():
                    result[key][cat_key] = torch.stack([item[key][cat_key] for item in batch])

            elif key == "static_num":
                # Handle static numerical features
                static_nums = [item[key] for item in batch if item[key] is not None]
                if static_nums:
                    result[key] = torch.stack(static_nums)
                else:
                    result[key] = None

            elif key in ["hypo_onset_type", "mpog_case_id", "patient_id"]:
                # Keep as lists for string values
                result[key] = [item[key] for item in batch]

            elif key in ["vital_flags", "target_flags"]:
                # Handle optional flag tensors - only include if present in all items
                if all(key in item for item in batch):
                    result[key] = torch.stack([item[key] for item in batch])
                else:
                    result[key] = None

            else:
                # Stack tensor values
                result[key] = torch.stack([item[key] for item in batch])

        return result


class StreamingIntraOpDataset(torch.utils.data.IterableDataset):
    """
    Memory-efficient streaming dataset that loads one institution file at a time.

    Instead of loading all data into RAM, this loads institution files sequentially,
    generates samples from each, then moves to the next file.

    Usage:
        dataset = StreamingIntraOpDataset(
            file_paths=['/path/to/inst_1001.feather', '/path/to/inst_1002.feather', ...],
            config=config,
            vocabs=vocabs,
        )
        loader = DataLoader(dataset, batch_size=512, num_workers=0)  # num_workers=0 required
    """

    def __init__(
        self,
        file_paths: List[str],
        config: Dict[str, Any],
        vocabs: Dict[str, Any],
        split: str = "train",
        shuffle_files: bool = True,
        shuffle_samples: bool = True,
        debug_frac: Optional[float] = None,
        seed: int = 42,
    ):
        self.file_paths = list(file_paths)
        self.config = config
        self.vocabs = vocabs
        self.split = split
        self.shuffle_files = shuffle_files
        self.shuffle_samples = shuffle_samples
        self.debug_frac = debug_frac
        self.seed = seed

        # Extract config values
        self.vital_cols = config['vital_cols']
        self.med_cols = config['med_cols']
        self.gas_cols = config.get('gas_cols', [])
        self.bolus_cols = config.get('bolus_cols', [])
        self.max_len = config['max_len']
        self.future_steps = config['future_steps']
        self.static_cat_cols = config.get('static_categoricals', [])
        self.static_num_cols = config.get('static_numericals', [])
        self.target_cols = config.get('target_cols', ['phys_bp_mean_non_invasive'])
        self.vital_flag_cols = config.get('vital_flag_cols', [])
        self.target_flag_cols = config.get('target_flag_cols', [])
        self.use_imputation_masking = config.get('use_imputation_masking', False)

        self.normalizer = Normalizer(config)
        self.vocab_maps = vocabs or {}

        logger.info(f"StreamingIntraOpDataset: {len(file_paths)} files, split={split}")

    def __iter__(self):
        """Iterate through all files, yielding samples from each."""
        worker_info = torch.utils.data.get_worker_info()

        # Get file list for this worker (if using multiple workers)
        if worker_info is None:
            file_list = self.file_paths
            worker_seed = self.seed
        else:
            # Split files among workers
            per_worker = len(self.file_paths) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.file_paths)
            file_list = self.file_paths[start:end]
            worker_seed = self.seed + worker_id

        # Shuffle file order
        if self.shuffle_files:
            rng = random.Random(worker_seed)
            file_list = file_list.copy()
            rng.shuffle(file_list)

        # Process each file
        for file_path in file_list:
            try:
                yield from self._process_file(file_path, worker_seed)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue

    def _process_file(self, file_path: str, seed: int):
        """Load a file and yield all samples from it. OPTIMIZED."""
        # Load file
        df = pd.read_feather(file_path)
        logger.info(f"  Loading {Path(file_path).name}: {len(df):,} rows")

        # Debug mode: sample cases
        if self.debug_frac is not None and self.debug_frac < 1.0:
            case_ids = df['mpog_case_id'].unique()
            rng = np.random.RandomState(seed)
            n_sample = max(1, int(len(case_ids) * self.debug_frac))
            sampled_ids = rng.choice(case_ids, size=n_sample, replace=False)
            df = df[df['mpog_case_id'].isin(sampled_ids)]

        if len(df) == 0:
            return

        # Process static categorical columns
        for col in self.static_cat_cols:
            if col in df.columns and col in self.vocab_maps:
                df[col] = df[col].map(self.vocab_maps[col]).fillna(0).astype(int)

        # OPTIMIZED: Filter valid cases using groupby.size() instead of get_group()
        min_len = self.future_steps + 1
        group_sizes = df.groupby('mpog_case_id').size()
        valid_case_ids = group_sizes[group_sizes >= min_len].index

        if len(valid_case_ids) == 0:
            return

        df = df[df['mpog_case_id'].isin(valid_case_ids)]

        # OPTIMIZED: Sort and compute case boundaries
        df = df.sort_values(['mpog_case_id']).reset_index(drop=True)
        case_ids_array = df['mpog_case_id'].values
        case_boundaries = {}
        current_case = None
        start_idx = 0

        for i, cid in enumerate(case_ids_array):
            if cid != current_case:
                if current_case is not None:
                    case_boundaries[current_case] = (start_idx, i)
                current_case = cid
                start_idx = i
        if current_case is not None:
            case_boundaries[current_case] = (start_idx, len(case_ids_array))

        # Generate samples using boundaries
        samples = []
        stride = self.config.get('stride', 1)
        required_past_steps = self.config.get('min_history_steps', 5)

        for cid, (case_start, case_end) in case_boundaries.items():
            num_rows = case_end - case_start
            max_end = num_rows - self.future_steps + 1

            if max_end <= required_past_steps:
                continue

            for end in range(required_past_steps, max_end, stride):
                # Store (case_id, sample_end_idx, case_start_in_df, case_end_in_df)
                samples.append((cid, end, case_start, case_end))

        logger.info(f"    Generated {len(samples):,} samples from {len(case_boundaries)} cases")

        # Shuffle samples within file
        if self.shuffle_samples:
            rng = random.Random(seed)
            rng.shuffle(samples)

        # Yield each sample
        for cid, end, case_start, case_end in samples:
            try:
                # Get case data using boundaries
                case_df = df.iloc[case_start:case_end]
                yield self._get_sample(cid, end, case_df)
            except Exception as e:
                continue  # Skip bad samples silently

    def _get_sample(self, cid: str, end: int, group: pd.DataFrame) -> Dict[str, Any]:
        """Generate a single sample - mirrors IntraOpDataset.__getitem__."""
        start = max(0, end - self.max_len)
        window = group.iloc[start:end]
        future = group.iloc[end:end + self.future_steps]
        seq_len = len(window)

        # Initialize arrays
        vitals = np.zeros((self.max_len, len(self.vital_cols)), dtype=np.float16)
        meds = np.zeros((self.max_len, len(self.med_cols)), dtype=np.float16)
        gases = np.zeros((self.max_len, len(self.gas_cols)), dtype=np.float16) if self.gas_cols else np.zeros((self.max_len, 0), dtype=np.float16)
        bolus = np.zeros((self.max_len, len(self.bolus_cols)), dtype=np.float16) if self.bolus_cols else np.zeros((self.max_len, 0), dtype=np.float16)
        attention_mask = np.zeros(self.max_len, dtype=np.bool_)

        # Fill arrays (right-aligned)
        if seq_len > 0:
            vital_values = window[self.vital_cols].values
            if self.normalizer.enabled:
                for i, col in enumerate(self.vital_cols):
                    vital_values[:, i] = self.normalizer.normalize(vital_values[:, i], col)
            vitals[-seq_len:] = vital_values.astype(np.float16)
            meds[-seq_len:] = window[self.med_cols].values.astype(np.float16)

            if self.gas_cols:
                gases[-seq_len:] = window[self.gas_cols].values.astype(np.float16)
            if self.bolus_cols:
                bolus[-seq_len:] = window[self.bolus_cols].values.astype(np.float16)

            attention_mask[-seq_len:] = True

        # Target
        target = np.zeros((self.future_steps, len(self.target_cols)), dtype=np.float32)
        if len(future) > 0:
            target_values = future[self.target_cols].values[:self.future_steps]
            if self.normalizer.enabled:
                for i, col in enumerate(self.target_cols):
                    target_values[:, i] = self.normalizer.normalize(target_values[:, i], col)
            target[:len(target_values)] = target_values

        # Last input bolus
        bolus_values = np.zeros(len(self.bolus_cols), dtype=np.float16)
        if self.bolus_cols and seq_len > 0:
            bolus_values = window[self.bolus_cols].iloc[-1].values.astype(np.float16)

        # Current row values
        current_row = window.iloc[-1] if seq_len > 0 else group.iloc[0]

        # Last 5 BP values
        last5_bp = [60.0] * 5
        if seq_len > 0:
            bp_col = self.target_cols[0]
            bp_vals = window[bp_col].values[-5:]
            for i, v in enumerate(bp_vals):
                last5_bp[-(len(bp_vals) - i)] = float(v) if not np.isnan(v) else 60.0

        # Hypo labels
        hypo_onset_label = float(current_row.get("hypo_onset_label", 0))
        hypo_onset_type = str(current_row.get("hypo_onset_type", "none"))

        # Static features
        static_cat = {col: torch.tensor(int(current_row.get(col, 0)), dtype=torch.long)
                     for col in self.static_cat_cols if col in current_row.index}
        static_num_vals = [float(current_row.get(col, 0)) for col in self.static_num_cols]
        static_num = torch.tensor(static_num_vals, dtype=torch.float32) if static_num_vals else torch.zeros(1)

        return {
            'vitals': torch.from_numpy(vitals),
            'meds': torch.from_numpy(meds),
            'gases': torch.from_numpy(gases),
            'bolus': torch.from_numpy(bolus),
            'last_input_bolus': torch.from_numpy(bolus_values),
            'attention_mask': torch.from_numpy(attention_mask),
            'target': torch.from_numpy(target),
            'static_cat': static_cat,
            'static_num': static_num,
            'patient_id': torch.tensor(hash(cid) % (2**31), dtype=torch.long),
            'mpog_case_id': cid,
            'last5_bp_values': torch.from_numpy(np.array(last5_bp, dtype=np.float16)),
            'last_known_bp': torch.tensor(last5_bp[-1], dtype=torch.float32),
            'hypo_onset_label': torch.tensor(hypo_onset_label, dtype=torch.float32),
            'hypo_onset_type': hypo_onset_type,
            'actual_sequence_length': torch.tensor(seq_len, dtype=torch.long),
            'prediction_timestep': torch.tensor(end, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Same collate function as IntraOpDataset."""
        return IntraOpDataset.collate_fn(batch)


class FastInferenceDataset(Dataset):
    """
    Ultra-fast inference-only dataset.

    Optimizations vs IntraOpDataset:
    - Pre-converts entire DataFrame to numpy arrays (one-time cost)
    - Uses direct numpy indexing instead of pandas iloc
    - Pre-normalizes all data upfront
    - Minimal tensor creation in __getitem__
    - No per-sample DataFrame operations

    ~5-10x faster than IntraOpDataset for inference.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        vocabs: Dict[str, Any],
    ):
        logger.info(f"FastInferenceDataset: {len(df):,} rows, {df['mpog_case_id'].nunique()} cases")

        self.config = config
        self.vital_cols = config['vital_cols']
        self.med_cols = config['med_cols']
        self.gas_cols = config.get('gas_cols', [])
        self.bolus_cols = config.get('bolus_cols', [])
        self.max_len = config['max_len']
        self.future_steps = config['future_steps']
        self.target_cols = config.get('target_cols', ['phys_bp_mean_non_invasive'])
        self.static_cat_cols = config.get('static_categoricals', [])
        self.static_num_cols = config.get('static_numericals', [])
        self.vocab_maps = vocabs or {}

        # Pre-convert static categoricals
        df = df.copy()
        for col in self.static_cat_cols:
            if col in df.columns and col in self.vocab_maps:
                df[col] = df[col].map(self.vocab_maps[col]).fillna(0).astype(np.int32)

        # Filter valid cases
        min_len = self.future_steps + 1
        group_sizes = df.groupby('mpog_case_id').size()
        valid_case_ids = group_sizes[group_sizes >= min_len].index
        df = df[df['mpog_case_id'].isin(valid_case_ids)]
        df = df.sort_values(['mpog_case_id']).reset_index(drop=True)

        logger.info(f"  After filtering: {len(df):,} rows, {len(valid_case_ids)} cases")

        # ===== PRE-CONVERT TO NUMPY (one-time cost) =====
        # Initialize normalizer
        normalizer = Normalizer(config)

        # Vitals - normalize and convert
        vitals_np = df[self.vital_cols].values.astype(np.float32)
        if normalizer.enabled:
            for i, col in enumerate(self.vital_cols):
                vitals_np[:, i] = normalizer.normalize(vitals_np[:, i], col)
        self.vitals = vitals_np.astype(np.float16)

        # Meds
        self.meds = df[self.med_cols].values.astype(np.float16)

        # Gases
        self.gases = df[self.gas_cols].values.astype(np.float16) if self.gas_cols else np.zeros((len(df), 0), dtype=np.float16)

        # Bolus
        self.bolus = df[self.bolus_cols].values.astype(np.float16) if self.bolus_cols else np.zeros((len(df), 0), dtype=np.float16)

        # Targets - normalize
        targets_np = df[self.target_cols].values.astype(np.float32)
        if normalizer.enabled:
            for i, col in enumerate(self.target_cols):
                targets_np[:, i] = normalizer.normalize(targets_np[:, i], col)
        self.targets = targets_np

        # Static categoricals
        self.static_cats = {}
        for col in self.static_cat_cols:
            if col in df.columns:
                self.static_cats[col] = df[col].values.astype(np.int64)

        # Static numericals
        if self.static_num_cols:
            self.static_nums = df[self.static_num_cols].values.astype(np.float32)
        else:
            self.static_nums = None

        # Case IDs for hashing
        self.case_ids = df['mpog_case_id'].values

        # ===== COMPUTE CASE BOUNDARIES =====
        case_boundaries = {}
        current_case = None
        start_idx = 0
        case_ids_array = self.case_ids

        for i, cid in enumerate(case_ids_array):
            if cid != current_case:
                if current_case is not None:
                    case_boundaries[current_case] = (start_idx, i)
                current_case = cid
                start_idx = i
        if current_case is not None:
            case_boundaries[current_case] = (start_idx, len(case_ids_array))

        self.case_boundaries = case_boundaries

        # ===== GENERATE SAMPLES =====
        samples = []
        stride = config.get('stride', 1)
        required_past_steps = config.get('min_history_steps', 5)

        for cid, (case_start, case_end) in case_boundaries.items():
            num_rows = case_end - case_start
            max_end = num_rows - self.future_steps + 1

            if max_end <= required_past_steps:
                continue

            for end in range(required_past_steps, max_end, stride):
                # Store absolute indices
                abs_end = case_start + end
                samples.append((case_start, abs_end))

        self.samples = samples
        logger.info(f"  Generated {len(samples):,} samples")

        # Pre-allocate output arrays (reused across calls)
        self._vitals_buf = np.zeros((self.max_len, len(self.vital_cols)), dtype=np.float16)
        self._meds_buf = np.zeros((self.max_len, len(self.med_cols)), dtype=np.float16)
        self._gases_buf = np.zeros((self.max_len, len(self.gas_cols)), dtype=np.float16)
        self._bolus_buf = np.zeros((self.max_len, len(self.bolus_cols)), dtype=np.float16)
        self._mask_buf = np.zeros(self.max_len, dtype=np.bool_)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_start, abs_end = self.samples[idx]

        # Calculate window
        window_start = max(case_start, abs_end - self.max_len)
        seq_len = abs_end - window_start
        future_end = min(abs_end + self.future_steps, len(self.vitals))
        actual_future = future_end - abs_end

        # Fast array filling using pre-allocated buffers
        # Reset buffers
        vitals = np.zeros((self.max_len, len(self.vital_cols)), dtype=np.float16)
        meds = np.zeros((self.max_len, len(self.med_cols)), dtype=np.float16)
        gases = np.zeros((self.max_len, self.gases.shape[1]), dtype=np.float16)
        bolus = np.zeros((self.max_len, self.bolus.shape[1]), dtype=np.float16)
        mask = np.zeros(self.max_len, dtype=np.bool_)

        # Fill with data (right-aligned)
        if seq_len > 0:
            vitals[-seq_len:] = self.vitals[window_start:abs_end]
            meds[-seq_len:] = self.meds[window_start:abs_end]
            if self.gases.shape[1] > 0:
                gases[-seq_len:] = self.gases[window_start:abs_end]
            if self.bolus.shape[1] > 0:
                bolus[-seq_len:] = self.bolus[window_start:abs_end]
            mask[-seq_len:] = True

        # Target (future values)
        target = np.zeros((self.future_steps, len(self.target_cols)), dtype=np.float32)
        if actual_future > 0:
            target[:actual_future] = self.targets[abs_end:future_end]

        # Static features from last timestep
        last_idx = abs_end - 1
        static_cat = {col: torch.tensor(self.static_cats[col][last_idx], dtype=torch.long)
                     for col in self.static_cat_cols if col in self.static_cats}

        static_num = None
        if self.static_nums is not None:
            static_num = torch.from_numpy(self.static_nums[last_idx].copy())

        # Case ID hash
        cid = self.case_ids[last_idx]
        patient_id = hash(cid) % (2**31)

        return {
            'vitals': torch.from_numpy(vitals),
            'meds': torch.from_numpy(meds),
            'gases': torch.from_numpy(gases),
            'bolus': torch.from_numpy(bolus),
            'attention_mask': torch.from_numpy(mask),
            'target': torch.from_numpy(target),
            'static_cat': static_cat,
            'static_num': static_num,
            'patient_id': torch.tensor(patient_id, dtype=torch.long),
            'actual_sequence_length': torch.tensor(seq_len, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimized collate for fast inference."""
        result = {}

        # Stack all tensor fields
        for key in ['vitals', 'meds', 'gases', 'bolus', 'attention_mask', 'target',
                    'patient_id', 'actual_sequence_length']:
            result[key] = torch.stack([item[key] for item in batch])

        # Static categoricals
        result['static_cat'] = {}
        if batch[0]['static_cat']:
            for cat_key in batch[0]['static_cat'].keys():
                result['static_cat'][cat_key] = torch.stack([item['static_cat'][cat_key] for item in batch])

        # Static numericals
        static_nums = [item['static_num'] for item in batch if item['static_num'] is not None]
        result['static_num'] = torch.stack(static_nums) if static_nums else None

        return result