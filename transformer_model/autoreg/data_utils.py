from pathlib import Path
import pandas as pd
from utils import get_cached_batches # or update paths if moved
from categorical_utils import build_vocab
from preprocess_and_cache import preprocess_and_cache
import logging
logger = logging.getLogger(__name__)
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader,WeightedRandomSampler
import time
from typing import Dict
from run_scripts import make_loader
logger = logging.getLogger(__name__)


def subsample_cases(df, config, split_name: str):
    if df.empty:
        raise ValueError(f"No data in split {split_name}.")
    future_steps = config.get('future_steps', 15)
    
    # Fix: Use proper groupby filtering instead of problematic .filter() method
    grouped = df.groupby("mpog_case_id")
    valid_case_ids = [cid for cid in grouped.groups.keys() if len(grouped.get_group(cid)) >= future_steps + 1]
    df = df[df['mpog_case_id'].isin(valid_case_ids)]
    
    if df.empty:
        raise ValueError(f"No valid cases found in {split_name} split after filtering.")
    frac = config.get("data_percentage", 1.0)
    if frac < 1.0:
        ids = df['mpog_case_id'].drop_duplicates()
        sampled = ids.sample(frac=frac, random_state=42)
        df = df[df['mpog_case_id'].isin(sampled)]
    return df.reset_index(drop=True)


def load_or_sample_splits(config):
    pct = int(config.get('data_percentage', 1.0) * 100)
    cache_dir = Path(config['cache_path'])
    split_files = {s: cache_dir / f"cached_{s}_{pct}pct.feather" for s in ['train','val','test']}
    if all(f.exists() for f in split_files.values()):
        print(f"✅ Found cached tiny splits for {pct}%")
        splits = {s: pd.read_feather(path) for s, path in split_files.items()}
    else:
        print(f"⏳ Creating tiny splits for {pct}%...")
        full_df = preprocess_and_cache(config, case_id_subset=None)
        splits = {s: subsample_cases(full_df[full_df.split==s], config, s) for s in ['train','val','test']}
        for s, df in splits.items():
            df.to_feather(split_files[s])
        print(f"✅ Saved tiny splits to cache for {pct}%")
    return splits

# ✅ Make sure this import is present
def load_and_split_data(config, dataset_class):
    import time
    start = time.time()

    # Step 1: Load or sample raw case-level splits
    splits = load_or_sample_splits(config)
    print(f"✅ Loaded tiny splits. ({time.time() - start:.2f}s)")

    # Step 2: Build vocab
    vocabs = build_vocab(splits['train'], config.get('static_categoricals', []))

    # Step 3: Build datasets, with sample-level balancing only for training
    # Skip expensive dataset creation if we're using cached batches
    use_cache = config.get('use_batch_caching', False)
    datasets = {}
    
    if use_cache:
        # Check if all cached batch files exist before skipping dataset creation
        from pathlib import Path
        from utils import generate_cache_key
        cache_dir = Path(config['cache_path']) / 'batches'
        all_caches_exist = True
        
        for split in ['train', 'val', 'test']:
            key = generate_cache_key(config, split)
            cache_file = cache_dir / f"{split}_batches_{key}.pt"
            if not cache_file.exists():
                all_caches_exist = False
                break
                
        if all_caches_exist:
            logger.info("🚀 All cached batches found - skipping expensive dataset creation")
            # Create minimal dummy datasets just for interface compatibility
            for split in ['train', 'val', 'test']:
                datasets[split] = None  # Will be handled in get_cached_batches
        else:
            logger.info("⚠️ Some cached batches missing - creating full datasets")
            all_caches_exist = False
    
    if not use_cache or not all_caches_exist:
        for split in ['train', 'val', 'test']:
            balance_this_split = (
                split == 'train' and
                config.get("balance_hypo_finetune", False)
            )
            datasets[split] = dataset_class(
                df=splits[split],
                config=config,
                vocabs=vocabs,
                split=split,
                balance_hypo_finetune=balance_this_split,
                hypo_balance_ratio=config.get("hypo_balance_ratio", 1.0)
            )

    # 🔍 Insert this logging block here:
    if config.get("finetune_hypo_only", False):
        if config.get("balance_hypo_finetune", False):
            logger.info("🧪 Using balanced hypo-onset batch caching")
        else:
            logger.info("🧪 Using unbalanced hypo-onset batches")

    print(f"✅ Built datasets. ({time.time() - start:.2f}s)")

    # Step 4: Build DataLoaders (with or without caching)
    loaders = {}
    use_cache = config.get('use_batch_caching', False)
    for split in ['train', 'val', 'test']:
        if use_cache:
            loaders[split] = get_cached_batches(
                dataset=datasets[split],
                split_name=split,
                config=config,
                logger=logger
            )
        else:
            loaders[split] = make_loader(
                dataset=datasets[split],
                shuffle=(split == 'train'),
                config=config
            )

    # Step 5: Log basic stats
    for split, df in splits.items():
        n_rows = len(df)
        n_cases = df['mpog_case_id'].nunique()
        print(f"Loaded {split}: {n_rows:,} rows / {n_cases:,} cases")

    return splits, datasets, loaders


