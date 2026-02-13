#!/usr/bin/env python3
"""Create vocabs.json for an existing model directory."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "transformer_model" / "autoreg"))
from scaling_data_loader import get_institution_file

# Config
MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
DATA_DIR = Path("/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/output_all/pcrc247_20260121_scaled")

# Load experiment info
info_path = MODEL_DIR / "experiment_info.json"
with open(info_path) as f:
    exp_info = json.load(f)

inst_id = exp_info["training_institution"]
config = exp_info["config"]

print(f"Training institution: {inst_id}")
print(f"Loading training data...")

# Load training data
train_file = get_institution_file(DATA_DIR, inst_id)
df = pd.read_feather(train_file)
print(f"  Rows: {len(df):,}")

# Build vocabs
vocabs = {}
for col in config.get("static_categoricals", []):
    if col in df.columns:
        unique_vals = df[col].dropna().unique()
        vocabs[col] = {str(v): i + 1 for i, v in enumerate(sorted(unique_vals))}
        print(f"  {col}: {len(vocabs[col])} values")

# Save
out_path = MODEL_DIR / "vocabs.json"
with open(out_path, "w") as f:
    json.dump(vocabs, f, indent=2)

print(f"\nSaved: {out_path}")
