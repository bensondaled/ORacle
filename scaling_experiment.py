#!/usr/bin/env python3
"""
Scaling Experiment: Train on progressively more institutions.

Usage:
    python scaling_experiment.py --num-institutions 5 --data-dir /path/to/scaled
    python scaling_experiment.py --num-institutions 70 --debug  # 1% data for testing

Scales: 5, 10, 20, 40, 60, 70 institutions
Test set: 4 held-out institutions (1004, 1016, 1018, 1027)
"""

import argparse
import gc
import os
import sys
import yaml
import json
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
    select_institutions,
    load_institutions_data,
    load_test_data,
    create_train_val_split,
    TEST_INSTITUTIONS,
    INSTITUTION_METADATA,
    get_scale_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scaling experiment")

    # Core arguments
    parser.add_argument(
        "--num-institutions", type=int, required=True,
        help="Number of training institutions (5, 10, 20, 40, 60). 60 = all available (67)"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory with scaled institution feather files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./scaling_outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Base config YAML (defaults to transformer_model/autoreg/config.yaml)"
    )

    # Experiment settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (reduced for V100 16GB)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction")

    # Debug mode
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: use 1%% of data, 3 epochs"
    )

    # WandB
    parser.add_argument("--wandb-project", type=str, default="oracle-scaling-study")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")

    return parser.parse_args()


def load_base_config(config_path: str = None) -> dict:
    """Load base configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "transformer_model" / "autoreg" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def setup_wandb(args, config: dict, selected_institutions: list):
    """Initialize WandB run."""
    if args.no_wandb:
        return None

    try:
        import wandb

        run_name = f"scale_{args.num_institutions}_seed{args.seed}"
        if args.debug:
            run_name += "_debug"

        run = wandb.init(
            project=args.wandb_project + ("_debug" if args.debug else ""),
            entity=args.wandb_entity,
            name=run_name,
            config={
                **config,
                "num_institutions": args.num_institutions,
                "selected_institutions": selected_institutions,
                "test_institutions": list(TEST_INSTITUTIONS),
                "debug_mode": args.debug,
                "seed": args.seed,
            },
            tags=[f"scale_{args.num_institutions}", "scaling_study"],
        )
        return run
    except Exception as e:
        print(f"Warning: WandB init failed: {e}")
        return None


def build_vocab(df: pd.DataFrame, categorical_cols: list) -> dict:
    """Build vocabulary for categorical columns."""
    vocabs = {}
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            vocabs[col] = {v: i for i, v in enumerate(sorted(unique_vals))}
    return vocabs


@torch.no_grad()
def evaluate(model, loader, config, device):
    """Simple evaluation function - computes loss and MSE."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    for batch in loader:
        # Move to device
        vitals = batch["vitals"].to(device, dtype=torch.float32)
        meds = batch["meds"].to(device, dtype=torch.float32)
        gases = batch["gases"].to(device, dtype=torch.float32)
        bolus = batch["bolus"].to(device, dtype=torch.float32)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.bool)
        targets = batch["target"].to(device, dtype=torch.float32)

        static_cat = batch.get("static_cat")
        if static_cat is not None:
            static_cat = {k: v.to(device) for k, v in static_cat.items()}
        static_num = batch.get("static_num")
        if static_num is not None:
            static_num = static_num.to(device, dtype=torch.float32)

        # Forward pass
        preds, _, _ = model(
            vitals=vitals,
            meds=meds,
            gases=gases,
            bolus=bolus,
            attention_mask=attention_mask,
            static_cat=static_cat,
            static_num=static_num,
            future_steps=config["future_steps"],
        )

        # Compute MSE
        mse = torch.nn.functional.mse_loss(preds, targets)
        total_mse += mse.item()
        total_loss += mse.item()  # Using MSE as loss for simplicity
        n_batches += 1

    model.train()

    if n_batches == 0:
        return {"loss": 0.0, "mse": 0.0}

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
    }


def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"scale_{args.num_institutions}_{timestamp}"
    if args.debug:
        run_name += "_debug"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    config = load_base_config(args.config)

    # Override config with CLI args
    config["batch_size_bp"] = args.batch_size
    config["lr_bp"] = args.lr
    config["epochs_bp"] = 3 if args.debug else args.epochs
    config["save_path"] = str(output_dir)
    config["cache_path"] = str(output_dir / "cache")

    # IMPORTANT: Data is already scaled - disable normalization to avoid double-scaling
    if "normalization" not in config:
        config["normalization"] = {}
    config["normalization"]["enabled"] = False

    # Disable torch.compile - cluster missing Python dev headers
    config["use_compile"] = False

    # Memory optimizations for V100 (16GB)
    config["use_mixed_precision"] = True  # FP16 training
    config["gradient_accumulation_steps"] = args.grad_accum  # Simulate larger batches

    # Debug mode settings
    debug_frac = 0.01 if args.debug else None

    # Select institutions
    print("=" * 60)
    print(f"SCALING EXPERIMENT: {args.num_institutions} institutions")
    print("=" * 60)

    selected = select_institutions(args.num_institutions, args.seed)
    summary = get_scale_summary(args.num_institutions, args.seed)

    print(f"\nSelected {len(selected)} institutions:")
    for region, info in summary["regions"].items():
        print(f"  {region}: {info['count']} institutions ({info['cases']:,} cases)")
    print(f"  Total: {summary['total_cases']:,} cases")

    if args.debug:
        print(f"\n[DEBUG MODE] Using 1% of cases, {config['epochs_bp']} epochs")

    # Initialize WandB
    wandb_run = setup_wandb(args, config, selected)

    # Load training data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train_val_df = load_institutions_data(
        data_dir=args.data_dir,
        institution_ids=selected,
        debug_frac=debug_frac,
        seed=args.seed,
    )

    # Split train/val
    train_df, val_df = create_train_val_split(
        train_val_df, val_frac=args.val_frac, seed=args.seed
    )
    del train_val_df
    gc.collect()

    # Load test data
    test_df = load_test_data(
        data_dir=args.data_dir,
        debug_frac=debug_frac,
        seed=args.seed,
    )

    # Save data info
    data_info = {
        "num_institutions": args.num_institutions,
        "selected_institutions": selected,
        "test_institutions": list(TEST_INSTITUTIONS),
        "train_cases": int(train_df['mpog_case_id'].nunique()),
        "train_rows": len(train_df),
        "val_cases": int(val_df['mpog_case_id'].nunique()),
        "val_rows": len(val_df),
        "test_cases": int(test_df['mpog_case_id'].nunique()),
        "test_rows": len(test_df),
        "debug_mode": args.debug,
    }

    with open(output_dir / "data_info.json", "w") as f:
        json.dump(data_info, f, indent=2)

    # Build vocab
    vocabs = build_vocab(train_df, config.get("static_categoricals", []))

    # Import dataset and model classes
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

    test_dataset = IntraOpDataset(
        df=test_df,
        config=config,
        vocabs=vocabs,
        split="test",
    )

    # Free DataFrames
    del train_df, val_df, test_df
    gc.collect()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size_bp"],
        shuffle=True,
        num_workers=min(config.get("num_workers", 8), 8),
        pin_memory=True,
        collate_fn=IntraOpDataset.collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size_bp"],
        shuffle=False,
        num_workers=min(config.get("num_workers", 8), 8),
        pin_memory=True,
        collate_fn=IntraOpDataset.collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size_bp"],
        shuffle=False,
        num_workers=min(config.get("num_workers", 8), 8),
        pin_memory=True,
        collate_fn=IntraOpDataset.collate_fn,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Create model
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    model = IntraOpPredictor(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if config.get("use_compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("Model compiled with torch.compile()")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_bp"],
        weight_decay=config.get("wd_bp", 1e-4),
    )

    # Scheduler
    scheduler = None
    if config.get("use_lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs_bp"],
            eta_min=config.get("scheduler_eta_min", 1e-6),
        )

    # Mixed precision scaler
    use_amp = config.get("use_mixed_precision", False) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    print(f"Mixed precision (AMP): {use_amp}")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0

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
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            config=config,
            device=device,
        )

        # Log metrics
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
                "lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

        # Save best model
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
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

        # Note: scheduler.step() is called inside train_autoreg_epoch per batch

    print(f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

    # Load best model for test evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        config=config,
        device=device,
    )

    print(f"Test Loss: {test_metrics.get('loss', 0):.4f}")
    print(f"Test MSE:  {test_metrics.get('mse', 0):.4f}")

    if wandb_run:
        import wandb
        wandb.log({
            "test/loss": test_metrics.get("loss", 0),
            "test/mse": test_metrics.get("mse", 0),
            "best_epoch": best_epoch,
        })
        wandb.finish()

    # Save final results
    results = {
        **data_info,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_metrics.get("loss", 0)),
        "test_mse": float(test_metrics.get("mse", 0)),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
