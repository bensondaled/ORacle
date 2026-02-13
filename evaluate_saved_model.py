#!/usr/bin/env python3
"""
Evaluate Saved Model on Test Institutions
==========================================

Load a saved model and evaluate on all other institutions.

Usage:
    # Just compute metrics
    python evaluate_saved_model.py --model-dir ./single_inst_outputs/inst_1056_seed42_20260204_005053

    # Save predictions per row to new parquet files
    python evaluate_saved_model.py --model-dir ./single_inst_outputs/inst_1056_* --save-predictions

    # Debug mode
    python evaluate_saved_model.py --model-dir ./single_inst_outputs/inst_1056_* --debug
"""

import argparse
import json
import sys
import time
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
        "--batch-size", type=int, default=65536,
        help="Batch size for evaluation (default 65536, decrease if OOM)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500000,
        help="Rows per chunk for writing (default 500k, larger = faster but more memory)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="DataLoader workers (default 8, increase if CPU bottlenecked)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode (1% data per institution)"
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Save row-level predictions to parquet files"
    )
    parser.add_argument(
        "--predictions-dir", type=str, default=None,
        help="Directory to save prediction files (default: output-dir/predictions)"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="oracle-eval",
        help="WandB project name"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable WandB logging"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Use half precision (FP16) for faster inference"
    )

    return parser.parse_args()


def predict_with_row_output(
    model: torch.nn.Module,
    file_paths: list,
    config: dict,
    vocabs: dict,
    device: torch.device,
    output_dir: Path,
    batch_size: int = 65536,
    debug_frac: float = None,
    chunk_size: int = 500000,  # Write in chunks (larger = faster, more memory)
    num_workers: int = 8,
    wandb_run=None,
    use_half: bool = False,
) -> dict:
    """
    Generate predictions for each row and save to parquet.

    Memory efficient: streams predictions in chunks, writes incrementally.
    Processes one institution at a time.

    Returns dict of {inst_id: output_file_path}
    """
    import gc
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from torch.utils.data import DataLoader, Subset
    from intraop_dataset import FastInferenceDataset

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_cols = config.get('target_cols', ['phys_bp_mean_non_invasive'])
    n_targets = len(target_cols)
    n_timepoints = config.get('future_steps', 15)

    # Build prediction column names
    pred_cols = []
    for tgt_name in target_cols[:n_targets]:
        for t in range(n_timepoints):
            pred_cols.append(f"pred_{tgt_name}_t{t+1}")

    results = {}

    for file_path in file_paths:
        # Extract institution ID
        fname = Path(file_path).stem
        inst_id = int(fname.split("_")[-1]) if "_" in fname else int(fname)

        print(f"\n  Processing institution {inst_id}...")
        inst_start = time.time()

        # Load dataframe
        df = pd.read_feather(file_path)

        if debug_frac:
            df = df.sample(frac=debug_frac, random_state=42).reset_index(drop=True)

        n_rows = len(df)
        print(f"    Rows: {n_rows:,}")

        # Create fast inference dataset (pre-converts all data to numpy)
        dataset = FastInferenceDataset(
            df=df,
            config=config,
            vocabs=vocabs,
        )

        n_valid = len(dataset)
        print(f"    Valid samples: {n_valid:,}")

        # Output file
        output_file = output_dir / f"predictions_{inst_id}.parquet"
        pq_writer = None
        rows_written = 0

        try:
            # Process in chunks
            for chunk_start in range(0, n_valid, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_valid)
                chunk_indices = list(range(chunk_start, chunk_end))

                # Create subset dataset for this chunk
                subset = Subset(dataset, chunk_indices)
                loader = DataLoader(
                    subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=4,
                    drop_last=False,
                )

                # Collect predictions for this chunk
                chunk_preds = []

                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_half):
                    for batch in loader:
                        # Move all tensors to GPU non-blocking
                        for k, v in batch.items():
                            if torch.is_tensor(v):
                                batch[k] = v.to(device, non_blocking=True)
                            elif k == "static_cat" and isinstance(v, dict):
                                batch[k] = {sk: sv.to(device, non_blocking=True) for sk, sv in v.items()}

                        preds, _, _ = model(
                            vitals=batch["vitals"],
                            meds=batch["meds"],
                            gases=batch["gases"],
                            bolus=batch["bolus"],
                            attention_mask=batch["attention_mask"].bool(),
                            static_cat=batch.get("static_cat"),
                            static_num=batch.get("static_num"),
                            future_steps=n_timepoints,
                        )

                        if preds.dim() == 2:
                            preds = preds.unsqueeze(-1)

                        chunk_preds.append(preds.float().cpu().numpy())

                # Stack chunk predictions
                chunk_preds = np.concatenate(chunk_preds, axis=0)

                # Get corresponding rows from df
                chunk_df = df.iloc[chunk_start:chunk_end].copy()

                # Add prediction columns
                for tgt_idx, tgt_name in enumerate(target_cols[:n_targets]):
                    for t in range(n_timepoints):
                        col_name = f"pred_{tgt_name}_t{t+1}"
                        chunk_df[col_name] = chunk_preds[:, t, tgt_idx]

                # Write chunk to parquet
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                if pq_writer is None:
                    pq_writer = pq.ParquetWriter(output_file, table.schema)
                pq_writer.write_table(table)

                rows_written += len(chunk_df)
                print(f"    Written {rows_written:,}/{n_valid:,} rows", end='\r')

                # Cleanup chunk
                del chunk_preds, chunk_df, table
                gc.collect()

        finally:
            if pq_writer is not None:
                pq_writer.close()

        elapsed = time.time() - inst_start
        throughput = rows_written / elapsed if elapsed > 0 else 0
        print(f"    Saved: {output_file} ({rows_written:,} rows in {elapsed/60:.1f}min, {throughput:.0f} rows/s)")
        results[inst_id] = str(output_file)

        # Log to WandB
        if wandb_run is not None:
            import wandb
            wandb.log({
                f"inst_{inst_id}/rows": rows_written,
                f"inst_{inst_id}/time_min": elapsed / 60,
                f"inst_{inst_id}/throughput": throughput,
                "institutions_completed": len(results),
                "total_rows_written": sum(int(Path(p).stem.split("_")[-1]) if "_" in Path(p).stem else 0 for p in results.values()) + rows_written,
            })

        # Cleanup institution
        del df, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


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

    # Half precision for faster inference
    if args.half:
        model = model.half()
        print("Using half precision (FP16)")

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"Validation loss: {checkpoint.get('val_loss', '?'):.4f}")

    # GPU memory info
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {allocated:.1f}GB used / {total_mem:.1f}GB total")

    # Initialize WandB
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"eval_inst{training_institution}",
                config={
                    "training_institution": training_institution,
                    "model_epoch": checkpoint.get("epoch"),
                    "val_loss": checkpoint.get("val_loss"),
                    "save_predictions": args.save_predictions,
                    "debug": args.debug,
                    **config,
                },
            )
            print(f"WandB run: {wandb_run.url}")
        except Exception as e:
            print(f"WandB init failed: {e}")
            wandb_run = None

    # Get test institution files (all except training)
    all_institutions = set(INSTITUTION_METADATA.keys()) - {training_institution}
    test_file_paths = []
    for inst_id in sorted(all_institutions):
        fp = get_institution_file(Path(args.data_dir), inst_id)
        if fp:
            test_file_paths.append(str(fp))

    print(f"\nTest institutions: {len(test_file_paths)}")

    # Load vocabs (saved during training)
    vocabs_path = model_dir / "vocabs.json"
    if vocabs_path.exists():
        print(f"\nLoading vocabs from {vocabs_path}")
        with open(vocabs_path) as f:
            vocabs = json.load(f)
    else:
        # Fallback: rebuild from training data
        print("\nBuilding vocabs from training institution...")
        import pandas as pd
        train_file = get_institution_file(Path(args.data_dir), training_institution)
        print(f"  Training file: {train_file}")

        if train_file is None or not Path(train_file).exists():
            print(f"  ERROR: Training file not found for institution {training_institution}")
            print(f"  Data dir: {args.data_dir}")
            sys.exit(1)

        train_df = pd.read_feather(train_file)

        if args.debug:
            train_df = train_df.sample(frac=0.01, random_state=42)

        vocabs = {}
        for col in config.get("static_categoricals", []):
            if col in train_df.columns:
                unique_vals = train_df[col].dropna().unique()
                vocabs[col] = {v: i + 1 for i, v in enumerate(sorted(unique_vals))}

        del train_df

    debug_frac = 0.01 if args.debug else None

    # Mode: save predictions or just compute metrics
    if args.save_predictions:
        print("\n" + "=" * 60)
        print("SAVING ROW-LEVEL PREDICTIONS")
        print("=" * 60)

        pred_dir = Path(args.predictions_dir) if args.predictions_dir else output_dir / "predictions"

        prediction_files = predict_with_row_output(
            model=model,
            file_paths=test_file_paths,
            config=config,
            vocabs=vocabs,
            device=device,
            output_dir=pred_dir,
            batch_size=args.batch_size,
            debug_frac=debug_frac,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            wandb_run=wandb_run,
            use_half=args.half,
        )

        # Save index of prediction files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_file = output_dir / f"prediction_files_{timestamp}.json"
        with open(index_file, "w") as f:
            json.dump(prediction_files, f, indent=2)

        print(f"\nPrediction files index: {index_file}")
        print(f"Predictions saved to: {pred_dir}")

        # Finish WandB
        if wandb_run is not None:
            import wandb
            wandb.log({"total_institutions": len(prediction_files)})
            wandb.finish()
        return

    # Otherwise: compute metrics
    from single_institution_experiment import evaluate_per_institution

    print("\n" + "=" * 60)
    print("PER-INSTITUTION TEST EVALUATION")
    print("=" * 60)

    test_results = evaluate_per_institution(
        model=model,
        file_paths=test_file_paths,
        config=config,
        vocabs=vocabs,
        device=device,
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
