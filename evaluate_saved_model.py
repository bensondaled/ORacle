#!/usr/bin/env python3
"""
Evaluate Saved Model on Test Institutions
==========================================

Load a saved model and evaluate on all other institutions.

Usage:
    python evaluate_saved_model.py --model-dir ./single_inst_outputs/inst_1056_seed42_20260204_005053
    python evaluate_saved_model.py --model-dir ./single_inst_outputs/inst_1056_* --debug
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add transformer_model to path
sys.path.insert(0, str(Path(__file__).parent / "transformer_model" / "autoreg"))

from scaling_data_loader import (
    get_institution_file,
    INSTITUTION_METADATA,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved model on test institutions")

    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Directory containing best_model.pt and experiment_info.json"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/output_all/pcrc247_20260121_scaled",
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: model-dir)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode (1% data per institution)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir

    print("=" * 60)
    print("EVALUATE SAVED MODEL ON TEST INSTITUTIONS")
    print("=" * 60)
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")

    # Check files exist
    model_path = model_dir / "best_model.pt"
    info_path = model_dir / "experiment_info.json"

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    if not info_path.exists():
        print(f"ERROR: Experiment info not found: {info_path}")
        sys.exit(1)

    # Load experiment info
    with open(info_path) as f:
        exp_info = json.load(f)

    training_institution = exp_info["training_institution"]
    config = exp_info["config"]

    print(f"Training institution: {training_institution}")
    print(f"Debug mode: {args.debug}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device)

    from model import IntraOpPredictor

    model = IntraOpPredictor(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"Validation loss: {checkpoint.get('val_loss', '?'):.4f}")

    # Get test institution files (all except training)
    all_institutions = set(INSTITUTION_METADATA.keys()) - {training_institution}
    test_file_paths = []
    for inst_id in sorted(all_institutions):
        fp = get_institution_file(Path(args.data_dir), inst_id)
        if fp:
            test_file_paths.append(str(fp))

    print(f"\nTest institutions: {len(test_file_paths)}")

    # Build vocabs from training data
    print("\nBuilding vocabs from training institution...")
    import pandas as pd
    train_file = get_institution_file(Path(args.data_dir), training_institution)
    train_df = pd.read_parquet(train_file)

    if args.debug:
        train_df = train_df.sample(frac=0.01, random_state=42)

    vocabs = {}
    for col in config.get("static_categoricals", []):
        if col in train_df.columns:
            unique_vals = train_df[col].dropna().unique()
            vocabs[col] = {v: i + 1 for i, v in enumerate(sorted(unique_vals))}

    del train_df

    # Import evaluation function from single_institution_experiment
    from single_institution_experiment import evaluate_per_institution

    # Evaluate
    print("\n" + "=" * 60)
    print("PER-INSTITUTION TEST EVALUATION")
    print("=" * 60)

    debug_frac = 0.01 if args.debug else None

    test_results = evaluate_per_institution(
        model=model,
        file_paths=test_file_paths,
        config=config,
        vocabs=vocabs,
        device=device,
        batch_size=args.batch_size,
        debug_frac=debug_frac,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"test_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Aggregate metrics
    all_mse = []
    all_mae = []
    all_n = []

    for inst_id, metrics in test_results.items():
        overall = metrics.get("overall", metrics)  # Handle both formats
        mse = overall.get("mse", 0)
        mae = overall.get("mae", 0)
        n = overall.get("n_samples", 0)

        all_mse.append(mse)
        all_mae.append(mae)
        all_n.append(n)

        print(f"  {inst_id}: MSE={mse:.4f}, MAE={mae:.4f}, N={n:,}")

    print("-" * 60)
    print(f"  Mean MSE: {np.mean(all_mse):.4f}")
    print(f"  Mean MAE: {np.mean(all_mae):.4f}")
    print(f"  Total samples: {sum(all_n):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
