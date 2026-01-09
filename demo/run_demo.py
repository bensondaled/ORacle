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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint (optional)')
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
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}...")
            state = torch.load(checkpoint_path, map_location=device)
            if 'model_state' in state:
                model.load_state_dict(state['model_state'])
            else:
                model.load_state_dict(state)
            print("Checkpoint loaded successfully!")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Running with randomly initialized weights (demo mode)")
    else:
        print("No checkpoint provided - running with randomly initialized weights")
        print("(For actual predictions, provide a trained checkpoint)")
    
    model.to(device)
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # Run inference on samples
    print("-" * 70)
    print("Running Inference")
    print("-" * 70)
    print()

    # Sample a few cases for demo
    case_ids = df['mpog_case_id'].unique()
    results = []

    for case_id in case_ids[:5]:  # Demo on first 5 cases
        # Find samples for this case
        case_samples = [i for i, (cid, _) in enumerate(dataset.samples) if cid == case_id]
        
        if not case_samples:
            continue
            
        # Use the last sample (most recent state)
        sample_idx = case_samples[-1]
        batch = prepare_batch(dataset, [sample_idx], device)
        
        # Run inference
        preds, hypo_fused_logits, hypo_bp_logits = run_inference(model, batch, config, device)
        
        # Get predictions
        map_predictions = preds[0, :, 0].cpu().numpy()  # [T] - MAP predictions
        
        # Get hypotension risk
        if hypo_fused_logits is not None:
            hypo_prob = torch.sigmoid(hypo_fused_logits[0, 0]).item() * 100
        else:
            hypo_prob = 0.0
            
        # Get input vitals
        last_vitals = batch['vitals'][0, -1, :].cpu().numpy()
        vital_cols = config['vital_cols']
        map_idx = vital_cols.index('phys_bp_mean_non_invasive')
        hr_idx = vital_cols.index('phys_spo2_pulse_rate')
        spo2_idx = vital_cols.index('phys_spo2_%')
        
        current_map = last_vitals[map_idx]
        current_hr = last_vitals[hr_idx]
        current_spo2 = last_vitals[spo2_idx]
        
        # Ground truth
        gt_hypo = batch['hypo_onset_label'][0].item()
        
        print(f"Case {case_id}:")
        print(f"  Current vitals: MAP={current_map:.0f} mmHg, HR={current_hr:.0f} bpm, SpO2={current_spo2:.0f}%")
        print(f"  Predicted MAP (next {config['future_steps']} min): ", end="")
        print(f"[{', '.join([f'{x:.1f}' for x in map_predictions[:5]])}, ...]")
        print(f"  Hypotension risk: {hypo_prob:.1f}%", end="")
        if hypo_prob > 50:
            print(" *** HIGH RISK ***", end="")
        print()
        if gt_hypo > 0:
            print(f"  [Ground truth: Hypotension onset at this timepoint]")
        print()
        
        results.append({
            'case_id': case_id,
            'current_map': current_map,
            'current_hr': current_hr,
            'predicted_map_1min': map_predictions[0],
            'predicted_map_5min': map_predictions[4] if len(map_predictions) > 4 else map_predictions[-1],
            'predicted_map_15min': map_predictions[-1],
            'hypo_risk_pct': hypo_prob,
            'gt_hypo_onset': gt_hypo
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = PROJECT_ROOT / 'demo' / 'demo_results.csv'
    results_df.to_csv(results_path, index=False)
    
    print("-" * 70)
    print("Summary")
    print("-" * 70)
    print(f"Processed {len(results)} cases")
    print(f"Results saved to: {results_path}")
    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    if not args.checkpoint:
        print()
        print("Note: Predictions were made with randomly initialized weights.")
        print("For meaningful predictions, train a model or provide a checkpoint:")
        print("  python demo/run_demo.py --checkpoint path/to/trained_model.pt")

    return 0


if __name__ == '__main__':
    sys.exit(main())
