#!/usr/bin/env python3
"""
ORacle Demo Script

Demonstrates running the ORacle model for intraoperative vital sign prediction
and hypotension onset detection using the demo dataset.

Usage:
    cd ORacle
    python demo/run_demo.py
    
    # Or with a custom checkpoint:
    python demo/run_demo.py --checkpoint path/to/model.pt
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "transformer_model" / "autoreg"))

import argparse
import yaml
import pandas as pd
import torch

# Import model components
from transformer_model.autoreg.model import IntraOpPredictor
from transformer_model.autoreg.intraop_dataset import IntraOpDataset


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_batch(dataset, indices, device):
    """Prepare a batch of samples for model input."""
    batch_items = [dataset[i] for i in indices]
    batch = IntraOpDataset.collate_fn(batch_items)
    
    # Move tensors to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            batch[key] = {k: v.to(device) for k, v in batch[key].items()}
    
    return batch


def run_inference(model, batch, config, device):
    """Run model inference on a batch."""
    with torch.no_grad():
        # Ensure float32 for model input
        vitals = batch['vitals'].float()
        meds = batch['meds'].float()
        gases = batch['gases'].float()
        bolus = batch['bolus'].float()
        attention_mask = batch['attention_mask']
        static_cat = batch['static_cat']
        static_num = batch['static_num'].float() if batch['static_num'] is not None else None
        
        # Forward pass
        preds, hypo_fused_logits, hypo_bp_logits = model(
            vitals=vitals,
            meds=meds,
            gases=gases,
            bolus=bolus,
            attention_mask=attention_mask,
            static_cat=static_cat,
            static_num=static_num,
            future_steps=config['future_steps']
        )
        
    return preds, hypo_fused_logits, hypo_bp_logits


def main():
    parser = argparse.ArgumentParser(description='ORacle Demo')
    parser.add_argument('--checkpoint', type=str, default='demo/demo_checkpoint.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='demo/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default='demo/demo_data.feather',
                        help='Path to demo data')
    args = parser.parse_args()

    print("=" * 70)
    print("ORacle Demo - Intraoperative Vital Sign Prediction")
    print("=" * 70)
    print()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load config
    config_path = PROJECT_ROOT / args.config
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # Load demo data
    data_path = PROJECT_ROOT / args.data
    print(f"Loading demo data from {data_path}...")
    df = pd.read_feather(data_path)
    
    n_cases = df['mpog_case_id'].nunique()
    n_timepoints = len(df)
    print(f"Loaded {n_cases} surgical cases with {n_timepoints:,} total timepoints")
    print()

    # Create dataset
    print("Creating dataset...")
    dataset = IntraOpDataset(
        df=df,
        config=config,
        vocabs={},
        split='test',
        balance_hypo_finetune=False
    )
    print(f"Created {len(dataset)} samples")
    print()

    # Initialize model
    print("Initializing model...")
    model = IntraOpPredictor(config)
    
    # Load checkpoint
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if 'model_state' in state:
            model.load_state_dict(state['model_state'])
        else:
            model.load_state_dict(state)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
