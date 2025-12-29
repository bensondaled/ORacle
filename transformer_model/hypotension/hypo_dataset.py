import torch
from torch.utils.data import Dataset
import pandas as pd
from intraop_dataset import IntraOpDataset  # Adjust import as needed


def build_vocab(df, cat_cols):
    """Builds vocabulary for categorical columns."""
    vocabs = {}
    for col in cat_cols:
        unique_vals = df[col].dropna().unique()
        vocabs[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    return vocabs

def map_to_vocab(df, vocabs):
    """Maps categorical columns to vocab indices."""
    df = df.copy()
    for col in vocabs:
        mapped = df[col].map(vocabs[col])
        df[col] = mapped.fillna(0).infer_objects(copy=False).astype(int)
    return df
import torch
from torch.utils.data import Dataset
import pandas as pd
from categorical_utils import build_vocab, map_to_vocab
from intraop_dataset import IntraOpDataset  # Adjust import as needed

import torch
from torch.utils.data import Dataset
import pandas as pd
from categorical_utils import build_vocab, map_to_vocab
from intraop_dataset import IntraOpDataset  # Adjust import as needed

import torch
from torch.utils.data import Dataset
import pandas as pd
from categorical_utils import build_vocab, map_to_vocab
from intraop_dataset import IntraOpDataset  # Adjust import as needed

class HypoDataset(Dataset):
    """
    A dataset for hypotension onset classification.
    Assumes samples are pre-filtered to hypo_onset_type in ('true_onset', 'none').
    """
    def __init__(self, samples):
        print(f"🛠️ Initializing HypoDataset with {len(samples)} samples...")
        start = pd.Timestamp.now()
        self.samples = samples  # Assume pre-filtered
        print(f"✅ HypoDataset initialized in {(pd.Timestamp.now() - start).total_seconds():.2f}s with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        output = {
            "vitals": s["vitals"].detach().clone() if torch.is_tensor(s["vitals"]) else torch.tensor(s["vitals"], dtype=torch.float),
            "meds": s["meds"].detach().clone() if torch.is_tensor(s["meds"]) else torch.tensor(s["meds"], dtype=torch.float),
            "bolus": s["bolus"].detach().clone() if torch.is_tensor(s["bolus"]) else torch.tensor(s["bolus"], dtype=torch.float),
            "attention_mask": s["attention_mask"].detach().clone() if torch.is_tensor(s["attention_mask"]) else torch.tensor(s["attention_mask"], dtype=torch.bool),
            "hypo_onset_label": s["hypo_onset_label"].detach().clone() if torch.is_tensor(s["hypo_onset_label"]) else torch.tensor(s["hypo_onset_label"], dtype=torch.long),
            "hypo_onset_type": torch.tensor(1 if s["hypo_onset_type"] == "true_onset" else 0, dtype=torch.bool),  # Convert to tensor
        }
        
        if "static_cat" in s and s["static_cat"] is not None:
            output["static_cat"] = {
                k: v.detach().clone() if torch.is_tensor(v) else torch.tensor(v, dtype=torch.long)
                for k, v in s["static_cat"].items()
            }
        else:
            output["static_cat"] = None

        if "static_num" in s and s["static_num"] is not None:
            output["static_num"] = s["static_num"].detach().clone() if torch.is_tensor(s["static_num"]) else torch.tensor(s["static_num"], dtype=torch.float)
        else:
            output["static_num"] = None

        return output

def load_datasets_for_hypo(config, dataset_class=IntraOpDataset, filter_types=None):
    """Loads train/val/test datasets for hypo classifier with vocab mapping and optional filtering."""
    print("📂 Loading feather files...")
    start = pd.Timestamp.now()
    try:
        train_df = pd.read_feather(config["train_path"])
        val_df = pd.read_feather(config["val_path"])
        test_df = pd.read_feather(config["test_path"])
        print(f"✅ Feather files loaded in {(pd.Timestamp.now() - start).total_seconds():.2f}s: train_rows={len(train_df)}, val_rows={len(val_df)}, test_rows={len(test_df)}")
    except Exception as e:
        print(f"❌ Error loading feather files: {e}")
        raise

    if filter_types:
        print(f"🧹 Filtering DataFrames for hypo_onset_type in {filter_types}...")
        start = pd.Timestamp.now()
        try:
            train_df = train_df[train_df["hypo_onset_type"].isin(filter_types)]
            val_df = val_df[val_df["hypo_onset_type"].isin(filter_types)]
            test_df = test_df[test_df["hypo_onset_type"].isin(filter_types)]
            print(f"✅ DataFrames filtered in {(pd.Timestamp.now() - start).total_seconds():.2f}s: train_rows={len(train_df)}, val_rows={len(val_df)}, test_rows={len(test_df)}")
        except Exception as e:
            print(f"❌ Error filtering DataFrames: {e}")
            raise

    print("🗂️ Building vocab...")
    start = pd.Timestamp.now()
    try:
        cat_cols = config.get("static_categoricals", [])
        vocabs = build_vocab(train_df, cat_cols)
        print(f"✅ Vocab built in {(pd.Timestamp.now() - start).total_seconds():.2f}s: { {k: len(v) for k, v in vocabs.items()} }")
    except Exception as e:
        print(f"❌ Error building vocab: {e}")
        raise

    print("🗺️ Mapping vocab to splits...")
    start = pd.Timestamp.now()
    try:
        train_df = map_to_vocab(train_df, vocabs)
        val_df = map_to_vocab(val_df, vocabs)
        test_df = map_to_vocab(test_df, vocabs)
        print(f"✅ Vocab mapped in {(pd.Timestamp.now() - start).total_seconds():.2f}s")
    except Exception as e:
        print(f"❌ Error mapping vocab: {e}")
        raise

    print("🛠️ Building IntraOp datasets...")
    start = pd.Timestamp.now()
    try:
        train_dataset = dataset_class(train_df, config, vocabs, split="train")
        val_dataset = dataset_class(val_df, config, vocabs, split="val")
        test_dataset = dataset_class(test_df, config, vocabs, split="test")
        print(f"✅ IntraOp datasets built in {(pd.Timestamp.now() - start).total_seconds():.2f}s: train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, test_samples={len(test_dataset)}")
    except Exception as e:
        print(f"❌ Error building IntraOp datasets: {e}")
        raise

    return train_dataset, val_dataset, test_dataset