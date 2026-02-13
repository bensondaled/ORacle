#!/usr/bin/env python3
"""
Single Institution Experiment
=============================

Train on ONE institution, validate on same institution, test on ALL others.
Reports per-institution test metrics to measure generalization.

Usage:
    python single_institution_experiment.py --institution 1056
    python single_institution_experiment.py --institution 1056 --debug
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

# Add transformer_model to path
sys.path.insert(0, str(Path(__file__).parent / "transformer_model" / "autoreg"))

from scaling_data_loader import (
    get_institution_file_paths,
    get_institution_file,
    INSTITUTION_METADATA,
    TEST_INSTITUTIONS,
    REGIONS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Single institution experiment")

    parser.add_argument(
        "--institution", type=int, required=True,
        help="Institution ID to train on (e.g., 1056)"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/output_all/pcrc247_20260121_scaled",
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./single_inst_outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config YAML path"
    )

    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)

    # Debug
    parser.add_argument("--debug", action="store_true", help="Debug mode (1% data, 3 epochs)")

    # WandB
    parser.add_argument("--wandb-project", type=str, default="oracle-single-inst")
    parser.add_argument("--no-wandb", action="store_true")

    return parser.parse_args()


def load_base_config(config_path=None):
    """Load base config from YAML."""
    import yaml

    if config_path is None:
        config_path = Path(__file__).parent / "transformer_model" / "autoreg" / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def build_vocab(df: pd.DataFrame, categorical_cols: list) -> dict:
    """Build vocabulary for categorical columns."""
    vocabs = {}
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            vocabs[col] = {v: i + 1 for i, v in enumerate(sorted(unique_vals))}
    return vocabs


def evaluate_per_institution(
    model: nn.Module,
    file_paths: list,
    config: dict,
    vocabs: dict,
    device: torch.device,
    debug_frac: float = None,
) -> dict:
    """
    Evaluate model on each institution separately.

    Returns dict: {institution_id: {
        overall: {mse, mae, rmse, n_samples},
        per_target: {target_name: {mse, mae, rmse}},
        per_timepoint: {t: {mse, mae, rmse}},
        per_target_timepoint: {target_name: {t: {mse, mae, rmse}}}
    }}
    """
    import pandas as pd

    model.eval()
    results = {}

    target_cols = config.get('target_cols', ['phys_bp_mean_non_invasive'])
    n_targets = len(target_cols)
    n_timepoints = config.get('future_steps', 15)

    for file_path in file_paths:
        # Extract institution ID from filename
        fname = Path(file_path).stem
        inst_id = int(fname.split("_")[-1]) if "_" in fname else int(fname)

        print(f"\n  Evaluating institution {inst_id}...")

        # Load institution data (one at a time - fits in memory)
        print(f"    Loading data...", flush=True)
        df = pd.read_feather(file_path)

        if debug_frac is not None:
            case_ids = df['mpog_case_id'].unique()
            n_sample = max(1, int(len(case_ids) * debug_frac))
            sampled_ids = np.random.choice(case_ids, size=n_sample, replace=False)
            df = df[df['mpog_case_id'].isin(sampled_ids)]

        print(f"    Rows: {len(df):,}", flush=True)

        # Use fast IntraOpDataset (not streaming)
        from intraop_dataset import IntraOpDataset
        dataset = IntraOpDataset(
            df=df,
            config=config,
            vocabs=vocabs,
            split="test",
        )
        print(f"    Samples: {len(dataset):,}, starting inference...", flush=True)

        loader = DataLoader(
            dataset,
            batch_size=config["batch_size_bp"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Accumulators for detailed metrics
        # Shape: [n_timepoints, n_targets]
        sum_sq_error = np.zeros((n_timepoints, n_targets))
        sum_abs_error = np.zeros((n_timepoints, n_targets))
        count = np.zeros((n_timepoints, n_targets))

        n_samples = 0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                n_batches += 1
                if n_batches % 100 == 0:
                    print(f"    Batch {n_batches}, samples so far: {n_samples:,}", flush=True)

                # Move to device
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)
                    elif k == "static_cat" and isinstance(v, dict):
                        batch[k] = {sk: sv.to(device) for sk, sv in v.items()}

                # Forward pass
                preds, _, _ = model(
                    vitals=batch["vitals"].float(),
                    meds=batch["meds"].float(),
                    gases=batch["gases"].float(),
                    bolus=batch["bolus"].float(),
                    attention_mask=batch["attention_mask"].bool(),
                    static_cat=batch.get("static_cat"),
                    static_num=batch.get("static_num"),
                    future_steps=n_timepoints,
                )

                # preds shape: [batch, n_timepoints, n_targets] or [batch, n_timepoints]
                # targets shape: [batch, n_timepoints, n_targets]
                targets = batch["target"].float()

                # Handle shape
                if preds.dim() == 2:
                    preds = preds.unsqueeze(-1)  # [batch, time, 1]
                if targets.dim() == 2:
                    targets = targets.unsqueeze(-1)

                # Move to CPU for accumulation
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Compute per timepoint, per target
                for t in range(min(n_timepoints, preds_np.shape[1])):
                    for tgt in range(min(n_targets, preds_np.shape[2])):
                        pred_t = preds_np[:, t, tgt]
                        targ_t = targets_np[:, t, tgt]

                        # Mask non-zero targets
                        mask = targ_t != 0
                        if mask.any():
                            sq_err = (pred_t[mask] - targ_t[mask]) ** 2
                            abs_err = np.abs(pred_t[mask] - targ_t[mask])

                            sum_sq_error[t, tgt] += sq_err.sum()
                            sum_abs_error[t, tgt] += abs_err.sum()
                            count[t, tgt] += mask.sum()

                n_samples += targets.size(0)

        # Compute averages
        if n_samples > 0:
            # Avoid division by zero
            count_safe = np.maximum(count, 1)

            mse_matrix = sum_sq_error / count_safe
            mae_matrix = sum_abs_error / count_safe
            rmse_matrix = np.sqrt(mse_matrix)

            # Overall metrics (weighted average across all)
            total_sq_err = sum_sq_error.sum()
            total_abs_err = sum_abs_error.sum()
            total_count = count.sum()

            overall_mse = total_sq_err / total_count if total_count > 0 else 0
            overall_mae = total_abs_err / total_count if total_count > 0 else 0
            overall_rmse = np.sqrt(overall_mse)

            # Per-target metrics (average across timepoints)
            per_target = {}
            for tgt_idx, tgt_name in enumerate(target_cols[:n_targets]):
                tgt_count = count[:, tgt_idx].sum()
                if tgt_count > 0:
                    per_target[tgt_name] = {
                        "mse": float(sum_sq_error[:, tgt_idx].sum() / tgt_count),
                        "mae": float(sum_abs_error[:, tgt_idx].sum() / tgt_count),
                        "rmse": float(np.sqrt(sum_sq_error[:, tgt_idx].sum() / tgt_count)),
                    }

            # Per-timepoint metrics (average across targets)
            per_timepoint = {}
            for t in range(n_timepoints):
                t_count = count[t, :].sum()
                if t_count > 0:
                    per_timepoint[t + 1] = {  # 1-indexed for readability
                        "mse": float(sum_sq_error[t, :].sum() / t_count),
                        "mae": float(sum_abs_error[t, :].sum() / t_count),
                        "rmse": float(np.sqrt(sum_sq_error[t, :].sum() / t_count)),
                    }

            # Per-target-timepoint (full matrix)
            per_target_timepoint = {}
            for tgt_idx, tgt_name in enumerate(target_cols[:n_targets]):
                per_target_timepoint[tgt_name] = {}
                for t in range(n_timepoints):
                    if count[t, tgt_idx] > 0:
                        per_target_timepoint[tgt_name][t + 1] = {
                            "mse": float(mse_matrix[t, tgt_idx]),
                            "mae": float(mae_matrix[t, tgt_idx]),
                            "rmse": float(rmse_matrix[t, tgt_idx]),
                        }

            results[inst_id] = {
                "overall": {
                    "mse": float(overall_mse),
                    "mae": float(overall_mae),
                    "rmse": float(overall_rmse),
                    "n_samples": int(n_samples),
                },
                "per_target": per_target,
                "per_timepoint": per_timepoint,
                "per_target_timepoint": per_target_timepoint,
            }

            print(f"    Overall - MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}, N: {n_samples:,}")

            # Print per-target summary
            print(f"    Per-target MSE: ", end="")
            for tgt_name, m in per_target.items():
                short_name = tgt_name.split("_")[-1][:8]
                print(f"{short_name}={m['mse']:.3f} ", end="")
            print()

        else:
            results[inst_id] = {
                "overall": {"mse": 0, "mae": 0, "rmse": 0, "n_samples": 0},
                "per_target": {},
                "per_timepoint": {},
                "per_target_timepoint": {},
            }
            print(f"    No samples!")

        # Cleanup
        del dataset, loader
        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inst_{args.institution}_seed{args.seed}_{timestamp}"
    if args.debug:
        run_name += "_debug"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_base_config(args.config)
    config["batch_size_bp"] = args.batch_size
    config["lr_bp"] = args.lr
    config["epochs_bp"] = 3 if args.debug else args.epochs
    config["use_mixed_precision"] = True
    config["gradient_accumulation_steps"] = args.grad_accum
    config["use_compile"] = False
    if "normalization" not in config:
        config["normalization"] = {}
    config["normalization"]["enabled"] = False

    debug_frac = 0.01 if args.debug else None

    print("=" * 60)
    print(f"SINGLE INSTITUTION EXPERIMENT")
    print("=" * 60)
    print(f"Training institution: {args.institution}")
    print(f"Train/Val split: {100*(1-args.val_frac):.0f}% / {100*args.val_frac:.0f}%")
    print(f"Test: ALL other institutions")
    print(f"Output: {output_dir}")

    # Check institution exists
    if args.institution not in INSTITUTION_METADATA:
        print(f"ERROR: Institution {args.institution} not in metadata!")
        print(f"Available: {sorted(INSTITUTION_METADATA.keys())}")
        sys.exit(1)

    inst_info = INSTITUTION_METADATA[args.institution]
    print(f"\nInstitution {args.institution}:")
    print(f"  Region: {inst_info[0]}")
    print(f"  Cases: {inst_info[1]:,}")

    # Get training institution file
    train_file = get_institution_file(Path(args.data_dir), args.institution)
    if train_file is None:
        print(f"ERROR: No file found for institution {args.institution}")
        sys.exit(1)

    print(f"\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load training institution data
    print(f"Loading {train_file.name}...")
    df = pd.read_feather(train_file)
    print(f"  Rows: {len(df):,}, Cases: {df['mpog_case_id'].nunique():,}")

    # Debug sampling
    if debug_frac:
        case_ids = df['mpog_case_id'].unique()
        np.random.seed(args.seed)
        n_sample = max(100, int(len(case_ids) * debug_frac))
        sampled_ids = np.random.choice(case_ids, size=n_sample, replace=False)
        df = df[df['mpog_case_id'].isin(sampled_ids)]
        print(f"  Debug mode: sampled {n_sample} cases, {len(df):,} rows")

    # Split train/val by case
    case_ids = df['mpog_case_id'].unique()
    np.random.seed(args.seed)
    np.random.shuffle(case_ids)
    n_val = int(len(case_ids) * args.val_frac)
    val_case_ids = set(case_ids[:n_val])
    train_case_ids = set(case_ids[n_val:])

    train_df = df[df['mpog_case_id'].isin(train_case_ids)].copy()
    val_df = df[df['mpog_case_id'].isin(val_case_ids)].copy()

    print(f"\nTrain: {len(train_case_ids):,} cases, {len(train_df):,} rows")
    print(f"Val: {len(val_case_ids):,} cases, {len(val_df):,} rows")

    del df
    gc.collect()

    # Build vocab
    vocabs = build_vocab(train_df, config.get("static_categoricals", []))

    # Save vocabs for later evaluation
    with open(output_dir / "vocabs.json", "w") as f:
        json.dump(vocabs, f, indent=2)

    # Get test institution files (all except training institution)
    all_institutions = set(INSTITUTION_METADATA.keys()) - {args.institution}
    test_file_paths = []
    for inst_id in sorted(all_institutions):
        fp = get_institution_file(Path(args.data_dir), inst_id)
        if fp:
            test_file_paths.append(str(fp))

    print(f"\nTest institutions: {len(test_file_paths)} (all except {args.institution})")

    # Save experiment info
    exp_info = {
        "training_institution": args.institution,
        "training_region": inst_info[0],
        "train_cases": len(train_case_ids),
        "val_cases": len(val_case_ids),
        "test_institutions": len(test_file_paths),
        "debug_mode": args.debug,
        "config": {k: v for k, v in config.items() if not callable(v)},
    }
    with open(output_dir / "experiment_info.json", "w") as f:
        json.dump(exp_info, f, indent=2, default=str)

    # Import model and training
    from intraop_dataset import IntraOpDataset
    from model import IntraOpPredictor
    from train_autoreg import train_autoreg_epoch

    # Create datasets
    print("\n" + "=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)

    train_dataset = IntraOpDataset(
        df=train_df,
        config=config,
        vocabs=vocabs,
        split="train",
    )

    val_dataset = IntraOpDataset(
        df=val_df,
        config=config,
        vocabs=vocabs,
        split="val",
    )

    del train_df, val_df
    gc.collect()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size_bp"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=IntraOpDataset.collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size_bp"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=IntraOpDataset.collate_fn,
    )

    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    # Create model
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    model = IntraOpPredictor(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_bp"],
        weight_decay=config.get("wd_bp", 1e-4),
    )

    scheduler = None
    if config.get("use_lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs_bp"],
            eta_min=1e-6,
        )

    # Mixed precision
    use_amp = config.get("use_mixed_precision", False) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    print(f"Mixed precision: {use_amp}")

    # WandB
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "institution": args.institution,
                    "region": inst_info[0],
                    **exp_info,
                },
            )
        except Exception as e:
            print(f"WandB init failed: {e}")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0

    # Simple evaluation function
    future_steps = config.get("future_steps", 15)

    def evaluate(model, loader, device):
        model.eval()
        total_loss = 0
        total_mse = 0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)
                    elif k == "static_cat" and isinstance(v, dict):
                        batch[k] = {sk: sv.to(device) for sk, sv in v.items()}

                preds, _, _ = model(
                    vitals=batch["vitals"].float(),
                    meds=batch["meds"].float(),
                    gases=batch["gases"].float(),
                    bolus=batch["bolus"].float(),
                    attention_mask=batch["attention_mask"].bool(),
                    static_cat=batch.get("static_cat"),
                    static_num=batch.get("static_num"),
                    future_steps=future_steps,
                )

                targets = batch["target"].float()
                mask = targets != 0
                if mask.any():
                    mse = ((preds - targets) ** 2)[mask].mean()
                    total_mse += mse.item()
                    total_loss += mse.item()
                    n_batches += 1

        model.train()
        if n_batches > 0:
            return {"loss": total_loss / n_batches, "mse": total_mse / n_batches}
        return {"loss": 0, "mse": 0}

    for epoch in range(1, config["epochs_bp"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs_bp']}")
        print("-" * 40)

        # Train
        train_metrics, global_step = train_autoreg_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            epoch=epoch,
            global_step=global_step,
            save_path=output_dir,
            scaler=scaler,
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
        print(f"Val Loss:   {val_metrics.get('loss', 0):.4f}")
        print(f"Val MSE:    {val_metrics.get('mse', 0):.4f}")

        if wandb_run:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics.get("loss", 0),
                "val/loss": val_metrics.get("loss", 0),
                "val/mse": val_metrics.get("mse", 0),
            }, step=global_step)

        # Save best
        if val_metrics.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, output_dir / "best_model.pt")
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    print(f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

    # Load best model
    print("\n" + "=" * 60)
    print("PER-INSTITUTION TEST EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on each test institution
    test_results = evaluate_per_institution(
        model=model,
        file_paths=test_file_paths,
        config=config,
        vocabs=vocabs,
        device=device,
        debug_frac=debug_frac,
    )

    # Save results
    with open(output_dir / "test_results_per_institution.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    # Aggregate by region
    region_results = {r: [] for r in REGIONS}
    for inst_id, metrics in test_results.items():
        if inst_id in INSTITUTION_METADATA:
            region = INSTITUTION_METADATA[inst_id][0]
            region_results[region].append(metrics)

    print(f"\n{'Institution':<12} {'Region':<12} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'N Samples':>12}")
    print("-" * 70)

    all_mse = []
    all_mae = []
    total_samples = 0

    for inst_id in sorted(test_results.keys()):
        m = test_results[inst_id]
        region = INSTITUTION_METADATA.get(inst_id, ("Unknown",))[0]
        print(f"{inst_id:<12} {region:<12} {m['mse']:>10.4f} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['n_samples']:>12,}")
        if m['n_samples'] > 0:
            all_mse.append(m['mse'])
            all_mae.append(m['mae'])
            total_samples += m['n_samples']

    print("-" * 70)
    if all_mse:
        print(f"{'MEAN':<12} {'':<12} {np.mean(all_mse):>10.4f} {np.mean(all_mae):>10.4f} {np.sqrt(np.mean(all_mse)):>10.4f} {total_samples:>12,}")
        print(f"{'STD':<12} {'':<12} {np.std(all_mse):>10.4f} {np.std(all_mae):>10.4f}")

    # Region summary
    print(f"\n{'Region':<12} {'N Inst':>8} {'Mean MSE':>10} {'Mean MAE':>10} {'Mean RMSE':>10}")
    print("-" * 55)
    for region in REGIONS:
        metrics_list = region_results[region]
        if metrics_list:
            mses = [m['mse'] for m in metrics_list if m['n_samples'] > 0]
            maes = [m['mae'] for m in metrics_list if m['n_samples'] > 0]
            if mses:
                print(f"{region:<12} {len(mses):>8} {np.mean(mses):>10.4f} {np.mean(maes):>10.4f} {np.sqrt(np.mean(mses)):>10.4f}")

    # Log to WandB
    if wandb_run:
        import wandb
        wandb.log({
            "test/mean_mse": np.mean(all_mse) if all_mse else 0,
            "test/mean_mae": np.mean(all_mae) if all_mae else 0,
            "test/std_mse": np.std(all_mse) if all_mse else 0,
            "test/n_institutions": len(test_results),
        })

        # Log per-institution as table
        table_data = [[inst_id, INSTITUTION_METADATA.get(inst_id, ("Unknown",))[0],
                       m['mse'], m['mae'], m['rmse'], m['n_samples']]
                      for inst_id, m in test_results.items()]
        wandb.log({"test/per_institution": wandb.Table(
            columns=["Institution", "Region", "MSE", "MAE", "RMSE", "N_Samples"],
            data=table_data
        )})

        wandb.finish()

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
