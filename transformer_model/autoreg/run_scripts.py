import os, sys, time, logging
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.amp import GradScaler
import wandb
from typing import Dict, Any,  Tuple
from torch.utils.data import  DataLoader,WeightedRandomSampler
from train_autoreg import train_autoreg_epoch
from eval_autoreg import eval_autoreg
from evaluation_autoreg import  evaluate_best_model
from utils import get_device,  load_latest_checkpoint, build_scheduler, collate_fn
import yaml
from evaluation_autoreg import evaluate_best_model
# Initialize logger
logger = logging.getLogger(__name__)
import csv


def freeze_for_hypo_finetune(model: torch.nn.Module, allowed: Tuple[str] = ("encoder", "hypo_fused_head")):
    logger.info(f"🔒 Freezing all layers except: {allowed}")
    for name, param in model.named_parameters():
        if any(k in name for k in allowed):
            param.requires_grad = True
            logger.info(f"✅ Unfrozen: {name}")
        else:
            param.requires_grad = False
            logger.info(f"🧊 Frozen: {name}")


def run_training_loop(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    config: Dict[str, Any],
    logger: logging.Logger,
    run_name: str,
    test_df,
    loss_module=None,
    loss_weight_module=None
) -> Tuple[Any, int, Path]:
    """
    Main training loop: trains model, applies early stopping, saves best checkpoint,
    and runs final evaluation.
    """
    save_dir = Path(config['save_path']) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    scaler = GradScaler(enabled=config.get("use_mixed_precision", False))
    # 🧪 Print which head is enabled
    print(f"🧠 Head active → BP: {config.get('use_hypo_onset_bp', False)} | Fused: {config.get('use_hypo_onset_fused', False)}")

    # 🔓 Ensure proper gradients are tracked for finetuning
    if config.get("finetune_hypo_only", False):
        logger.info("✅ Using earlier freezing config — no override here.")

    # ✅ Always count trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    if trainable_count == 0:
        raise RuntimeError("🚨 No parameters were unfrozen. Check naming or config.")


    # === Build optimizer with separate LR & WD for each module ===
    # Separate hypotension head parameters for higher learning rate
    hypo_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'hypo' in name.lower():
                hypo_params.append(param)
                logger.info(f"📈 Hypo param (high LR): {name}")
            else:
                other_params.append(param)
    
    # Create parameter groups with differential learning rates
    hypo_lr_mult = config.get('hypo_head_learning_rate_mult', 10.0)
    base_lr = config['lr_bp']
    
    param_groups = [
        {
            "params": other_params,
            "lr": base_lr,
            "weight_decay": config['wd_bp']
        }
    ]
    
    # Add hypotension parameters with higher LR if they exist
    if hypo_params:
        param_groups.append({
            "params": hypo_params,
            "lr": base_lr * hypo_lr_mult,  # Higher LR for hypotension heads
            "weight_decay": config.get('hypo_head_weight_decay', 0.001)  # Less regularization
        })
        logger.info(f"🎯 Hypotension heads using {hypo_lr_mult}x learning rate: {base_lr * hypo_lr_mult:.2e}")

    if loss_module is not None:
        param_groups.append({
            "params": loss_module.parameters(),
            "lr": config.get("lr_loss_module", config['lr_bp']),
            "weight_decay": 0.0
        })
    if loss_weight_module is not None:
        param_groups.append({
            "params": loss_weight_module.parameters(),
            "lr": config.get("lr_loss_weights", config['lr_bp']),
            "weight_decay": 0.0
        })
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = build_scheduler(optimizer, config)

    # === Optionally resume ===
    if config.get('resume', True):
        model, optimizer, scheduler, start_epoch = load_latest_checkpoint(
            config, model, optimizer, scheduler
        )
    else:
        start_epoch = 0
    global_step = 0  # ✅ ADD THIS LINE HERE

    best_loss = float('inf')
    best_state = None
    best_epoch = -1
    patience = config.get('early_stopping_patience', 10)
    counter = 0
    log_every_epoch = config.get('log_every_epochs', 1)
    num_epochs = config['epochs_bp']

    for epoch in range(start_epoch, num_epochs):
        config['epoch_index'] = epoch

        # ─── TRAIN ─────────────────────────────────────────────
        logger.info(f"Epoch {epoch} started.")
        model.train()
        train_metrics, global_step = train_autoreg_epoch(
            model=model,
            dataloader=loaders['train'],
            optimizer=optimizer,
            scheduler=None,
            config=config,
            device=device,
            epoch=epoch,
            scaler=scaler,
            loss_module=loss_module,
            loss_weight_module=loss_weight_module,
            global_step=global_step,
        )

        # ─── VALIDATION ────────────────────────────────────────
        model.eval()
        val_metrics = eval_autoreg(
            model=model,
            dataloader=loaders['val'],
            save_path=save_dir / "eval_csvs",
            config=config,
            device=device,
            loss_module=loss_module,
            loss_weight_module=loss_weight_module,
            global_step=global_step, # Pass global_step
        )
        val_loss = val_metrics.get('loss')
        if val_loss is None or torch.isnan(torch.tensor(val_loss)):
            logger.warning(f"Bad val_loss at epoch {epoch}: {val_loss}")
            break

        # ─── LOG EPOCH METRICS ─────────────────────────────────
        if epoch % log_every_epoch == 0 or epoch == num_epochs - 1:
            log_data = {
                'global_step': global_step, # Use the actual global_step
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'val/loss': val_metrics['loss'],
            }
            if loss_weight_module is not None:
                lw = loss_weight_module._get_weights()
                for k, v in lw.items():
                    log_data[f"learned_loss_weight/normalized/{k}"] = v.item()
            elif loss_module is not None and hasattr(loss_module, "_get_weights"):
                lw = loss_module._get_weights()
                for k, v in lw.items():
                    log_data[f"learned_loss_weight/{k}"] = v.item()
                if hasattr(loss_module, "logit_quantile"):
                    log_data["learned_quantile"] = torch.sigmoid(loss_module.logit_quantile).item()
            wandb.log(log_data) # Log only once
        csv_path = Path(config['save_path']) / config['run_name'] / "metrics_log.csv"
        header_written = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not header_written:
                writer.writeheader()
            writer.writerow(log_data)

        # ✅ NEW: Save checkpoint for this epoch (always)
        epoch_ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
        }
        epoch_ckpt_path = save_dir / f"epoch_{epoch:03d}.pt"
        torch.save(epoch_ckpt, epoch_ckpt_path)
        logger.info(f"💾 Saved checkpoint: {epoch_ckpt_path.name}")

        # ─── CHECK FOR IMPROVEMENT ─────────────────────────────
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch
            counter = 0
            ckpt = {
                'model_state': best_state,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
            }
            ckpt_path = save_dir / f"best_{run_name}_ep{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            logger.info(f"✅ Saved best model at epoch {epoch}, val_loss={best_loss:.6f}")
        else:
            counter += 1
            logger.info(f"No improvement: {counter}/{patience}")
            if counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break


        # ─── STEP SCHEDULER ────────────────────────────────────
        if scheduler and config.get("use_lr_scheduler", True):
            scheduler.step()

        # ─── OPTIONAL EARLY TEST EVAL ───────────────────────────
        # Optional early test eval
        if epoch == 1 and config.get("eval_at_epoch1", True):
            logger.info("🔍 Running quick test-set eval at epoch 1")
            test_metrics_1 = eval_autoreg(
                model=model,
                save_path=save_dir/"eval_csvs", 
                dataloader=loaders['test'],
                config=config,
                device=device,
                loss_module=loss_module,
                loss_weight_module=loss_weight_module,
            )
            log1 = {f"epoch1/{k}": v for k, v in test_metrics_1.items() if k != 'df_summary'}
            wandb.log(log1)
            logger.info("🔍 Quick eval at epoch 1 complete")

    # Final best model evaluation
    best_ckpt_path = save_dir / f"best_{run_name}_ep{best_epoch}.pt"
    if best_ckpt_path.exists():
        output_df = evaluate_best_model(
            model=model,
            checkpoint_path=best_ckpt_path,
            loaders=loaders,
            config=config,
            run_name=run_name,
            save_dir=save_dir,
            test_df=test_df,
            loss_module=loss_module,
            loss_weight_module=loss_weight_module,
            global_step=0,  # Use separate step counter for best model eval
            best_epoch=best_epoch,
        )
    else:
        raise RuntimeError(f"No best checkpoint found at {best_ckpt_path}")

    return best_state, best_epoch, best_ckpt_path, output_df
# Collate and loader helpers remain unchanged

def make_loader(dataset, shuffle, config):
    if config.get("use_balanced_sampler", False):
        labels = np.array(dataset.hypo_onset_labels)
        balance_info = compute_class_balance(labels)
        logger.info(f"📊 Dataset label distribution before sampling: {balance_info}")

        counts = np.bincount(labels)
        weights = 1.0 / (counts + 1e-6)  # avoid division by zero
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        loader_kwargs = {
            'batch_size': config["batch_size_bp"],
            'sampler': sampler,
            'collate_fn': dataset.collate_fn,
            'num_workers': config.get('num_workers', 0),
            'pin_memory': config.get('pin_memory', False)
        }
        if config.get('num_workers', 0) > 0 and config.get('prefetch_factor'):
            loader_kwargs['prefetch_factor'] = config['prefetch_factor']
        return DataLoader(dataset, **loader_kwargs)

    # Default behavior
    loader_kwargs = {
        'batch_size': config["batch_size_bp"],
        'shuffle': shuffle,
        'collate_fn': dataset.collate_fn,
        'num_workers': config.get('num_workers', 0),
        'pin_memory': config.get('pin_memory', False)
    }
    if config.get('num_workers', 0) > 0 and config.get('prefetch_factor'):
        loader_kwargs['prefetch_factor'] = config['prefetch_factor']
    return DataLoader(dataset, **loader_kwargs)
def compute_class_balance(labels):
    labels = np.array(labels)
    total = len(labels)
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    pos_ratio = positives / (total + 1e-6)
    return {
        "total": int(total),
        "positives": int(positives),
        "negatives": int(negatives),
        "positive_ratio": float(pos_ratio)
    }
