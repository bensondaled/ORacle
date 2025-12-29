import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
import logging
from pathlib import Path
from typing import Dict, Optional
import psutil
import json # Added import
from logging_utils import log_sigmoid_histogram_bp
from preprocessing_utils import inverse_scale_minmax
from losses import compute_loss
from losses.loss_schedule import get_scheduled_loss_weights
from utils import compute_bolus_mask, compute_classification_metrics, to_numpy
from bolus_analysis import compute_bolus_response_metrics

logger = logging.getLogger(__name__)

def eval_autoreg(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: Dict,
    device: torch.device,
    save_path: Path,
    loss_weight_module: Optional[torch.nn.Module] = None,
    loss_module: Optional[torch.nn.Module] = None,
    global_step: int = 0 # Add global_step as an argument
) -> Dict:
    """
    Evaluate autoregressive BP model (and optional hypotension heads).
    Saves a summary CSV to save_path and returns a dict containing metrics and the DataFrame.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Load scalers
    scalers_path = Path(config["cache_path"]) / "scalers.json"
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers file not found at {scalers_path}. Run preprocessing first.")
    with open(scalers_path, 'r') as f:
        scalers_data = json.load(f)

    # Reset one-time warnings
    logger._warned_no_fused = False
    logger._warned_no_bp = False

    # Determine logging frequency
    try:
        log_every = config.get("log_every_batches", max(1, len(dataloader) // 10))
    except TypeError:
        log_every = 1

    # Accumulators
    all_preds = []
    all_targets = []
    all_masks = []
    all_idxs = []
    all_case_ids = []
    all_elapsed = []
    all_last5 = []
    all_last_input = []
    all_hypo_fused = []
    all_hypo_bp = []
    all_hypo_lbls = []
    all_hypo_types = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # global_step = config.get("epoch_index", 0) * 10000 + batch_idx # Removed: not needed for eval batch logging

            # Move only necessary tensors to device for better performance
            # Move tensors to device with proper dtype conversion
            float_keys = ["vitals", "meds", "gases", "bolus", "static_num"]
            other_keys = ["attention_mask", "target"]
            
            for k in float_keys:
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device=device, dtype=torch.float32, non_blocking=True)
            
            for k in other_keys:
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)
            
            # Handle static_cat separately
            if "static_cat" in batch:
                batch["static_cat"] = {k: v.to(device, non_blocking=True) for k, v in batch["static_cat"].items()}
            # Debug logging removed for performance

            # Forward pass
            preds, fused_logits, bp_logits = model(
                vitals=batch["vitals"],
                meds=batch["meds"],
                gases=batch["gases"],  # FIXED: Added missing gases parameter
                bolus=batch["bolus"],
                attention_mask=batch["attention_mask"].to(device, dtype=torch.bool),
                static_cat={k: v.to(device) for k, v in batch["static_cat"].items()},
                static_num=batch["static_num"].to(device) if batch["static_num"] is not None else None,
                future_steps=config["future_steps"],
            )

            # Compute masks
            bolus_mask = compute_bolus_mask(batch, config["future_steps"], device)
            trigger_mask = (batch["last_input_bolus"] > 0).any(dim=-1)

            # Prepare onset labels
            onset_lbl = batch.get("hypo_onset_label")
            if onset_lbl is not None:
                onset_lbl = onset_lbl.to(device) if torch.is_tensor(onset_lbl) else torch.tensor(onset_lbl, device=device)
            else:
                logger.warning("⚠️ Missing onset_labels in batch")

            # Check for missing logits
            if logger.isEnabledFor(logging.WARNING):
                if config.get("use_hypo_onset_fused") and fused_logits is None and not logger._warned_no_fused:
                    logger.warning("⚠️ No hypotension logits (fused) found!")
                    logger._warned_no_fused = True
                if config.get("use_hypo_onset_bp") and bp_logits is None and not logger._warned_no_bp:
                    logger.warning("⚠️ No hypotension logits (bp) found!")
                    logger._warned_no_bp = True

            # Compute loss
            task_mode = config.get("task_mode", "autoreg")
            
            if task_mode == "hypo":
                # Hypotension-only mode: compute only classification loss
                loss = torch.tensor(0.0, device=device)
                loss_metrics = {}
                
                # Direct BCE loss for hypotension classification
                if fused_logits is not None and onset_lbl is not None:
                    # Apply masking (exclude ongoing samples)
                    onset_types = batch.get("hypo_onset_type")
                    if onset_types is not None:
                        mask = torch.tensor([t in ["true_onset", "none"] for t in onset_types], 
                                          dtype=torch.bool, device=device)
                        if mask.any():
                            filtered_logits = fused_logits.squeeze(-1)[mask]
                            filtered_labels = onset_lbl[mask].float()
                            
                            pos_weight = torch.tensor(config.get("hypo_pos_weight", 20.0), device=device)
                            loss = F.binary_cross_entropy_with_logits(
                                filtered_logits, filtered_labels, pos_weight=pos_weight
                            )
                            loss_metrics["hypo_onset_fused"] = loss.item()
            else:
                # Regular mode with full loss computation
                if loss_module is not None:
                    logits_for_loss = fused_logits if config.get("use_hypo_onset_fused") else bp_logits
                    loss, loss_metrics = loss_module(
                        preds=preds,
                        targets=batch["target"],
                        last_input=batch["last_known_bp"].to(device),
                        bolus_trigger_mask=trigger_mask,
                        onset_labels=onset_lbl,
                        onset_logits=logits_for_loss,
                        last_input_bolus=batch["last_input_bolus"],
                    )
                else:
                    if loss_weight_module is not None:
                        lw = loss_weight_module()
                    else:
                        lw = get_scheduled_loss_weights(config, epoch=config.get("epoch_index", 0))
                    # wandb.log({f"loss_weights/{k}": v for k, v in lw.items()}, step=global_step) # Removed: not needed for eval batch logging

                    loss, loss_metrics = compute_loss(
                        preds=preds,
                        targets=batch["target"],
                        loss_weights=lw,
                        bolus_mask=bolus_mask,
                        bolus_trigger_mask=trigger_mask,
                        config=config,
                        hypo_fused_logits=fused_logits,
                        hypo_bp_logits=bp_logits,
                        onset_labels=onset_lbl,
                        onset_types=batch.get("hypo_onset_type"),
                    )

            total_loss += loss.item()
            num_batches += 1

            # Convert metrics
            loss_metrics = {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_metrics.items()}

            # Classification metrics
            cls_metrics = {}
            if onset_lbl is not None and fused_logits is not None:
                cls_metrics.update(compute_classification_metrics(fused_logits, onset_lbl, "hypo_fused"))
                if bp_logits is not None:
                    cls_metrics.update(compute_classification_metrics(bp_logits, onset_lbl, "hypo_bp"))

            # Accumulate
            all_preds.append(to_numpy(preds))
            all_targets.append(to_numpy(batch["target"]))
            all_masks.append(to_numpy(~torch.isnan(batch["target"]).any(dim=-1)))
            all_idxs.extend(batch["original_index"].detach().cpu().tolist())
            all_case_ids.extend(batch["mpog_case_id"])
            all_elapsed.extend(batch["minutes_elapsed"].detach().cpu().tolist())
            all_last5.append(
                to_numpy(batch.get("last5_bp_values", None))
                if batch.get("last5_bp_values") is not None else np.zeros((len(batch["mpog_case_id"]), 5))
            )
            all_last_input.append(to_numpy(batch["last_input_bolus"]))
            if fused_logits is not None:
                all_hypo_fused.extend(to_numpy(fused_logits))
            if bp_logits is not None:
                all_hypo_bp.extend(to_numpy(bp_logits))
            if onset_lbl is not None:
                all_hypo_lbls.extend(to_numpy(onset_lbl))
            all_hypo_types.extend(batch.get("hypo_onset_type", ["none"] * len(batch["mpog_case_id"])))

            # Clear GPU cache periodically to prevent memory buildup
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
            
            # Clear batch tensors from GPU memory
            del batch
            if 'preds' in locals():
                del preds
            if 'fused_logits' in locals():
                del fused_logits
            if 'bp_logits' in locals():
                del bp_logits

    # Log distributions
    if all_hypo_fused:
        arr = np.array(all_hypo_fused).flatten()
        try:
            wandb.log({
                "eval/hypo_fused_logits_mean": arr.mean(),
                "eval/hypo_fused_logits_std":  arr.std(),
                "eval/hypo_fused_logits_min":  arr.min(),
                "eval/hypo_fused_logits_max":  arr.max(),
            }, step=global_step) # Log at epoch level
        except (AttributeError, wandb.Error):
            pass  # wandb not initialized
    if all_hypo_bp:
        arr = np.array(all_hypo_bp).flatten()
        try:
            wandb.log({
                "eval/hypo_bp_logits_mean": arr.mean(),
                "eval/hypo_bp_logits_std":  arr.std(),
                "eval/hypo_bp_logits_min":  arr.min(),
                "eval/hypo_bp_logits_max":  arr.max(),
            }, step=global_step) # Log at epoch level
        except (AttributeError, wandb.Error):
            pass  # wandb not initialized

    # Ground-truth prevalence
    if all_hypo_lbls:
        lab_arr = np.array(all_hypo_lbls).flatten()
        logger.info(f"🩺 Ground truth hypotension label prevalence: {lab_arr.mean():.2%} positives")

    if num_batches == 0:
        raise RuntimeError("No batches processed during evaluation.")

    # Average loss
    avg_loss = total_loss / num_batches
    try:
        wandb.log({"eval_avg_loss": avg_loss}, step=global_step)
    except (AttributeError, wandb.Error):
        pass  # wandb not initialized
    logger.info(f"[EVAL] End of Eval: allocated={torch.cuda.memory_allocated()/1e6:.1f}MB")

    # Concatenate all accumulated predictions and targets
    preds_np = np.concatenate(all_preds, axis=0)  # [N, T, num_targets]
    targets_np = np.concatenate(all_targets, axis=0) # [N, T, num_targets]
    masks_np = np.concatenate(all_masks, axis=0) # [N, T]

    logger.debug(f"Concatenated all_last_input shape: {np.concatenate(all_last_input, axis=0).shape}, dtype: {np.concatenate(all_last_input, axis=0).dtype}")
    logger.debug(f"Concatenated all_last5 shape: {np.concatenate(all_last5, axis=0).shape}, dtype: {np.concatenate(all_last5, axis=0).dtype}")

    expected_len = preds_np.shape[0] # Define expected_len here

    # Apply inverse scaling
    for i, col_name in enumerate(config["target_cols"]):
        scaler_info = scalers_data.get(col_name)
        if scaler_info:
            min_val, max_val = scaler_info["min"], scaler_info["max"]
            preds_np[:, :, i] = inverse_scale_minmax(preds_np[:, :, i], min_val, max_val)
            targets_np[:, :, i] = inverse_scale_minmax(targets_np[:, :, i], min_val, max_val)
        else:
            logger.warning(f"Scaler data not found for {col_name}. Skipping inverse scaling for this target.")

    df_summary = pd.DataFrame({
        "original_index": all_idxs,
        "mpog_case_id": all_case_ids,
        "minutes_elapsed": all_elapsed,
        "last5_bp_values": [list(row) for row in np.concatenate(all_last5, axis=0)],
        "last_input_bolus": [list(row) for row in np.concatenate(all_last_input, axis=0)],
        "hypo_fused_logits": all_hypo_fused if all_hypo_fused else [None] * expected_len,
        "hypo_bp_logits": all_hypo_bp if all_hypo_bp else [None] * expected_len,
        "hypo_onset_label": all_hypo_lbls if all_hypo_lbls else [None] * expected_len,
        "hypo_onset_type": all_hypo_types,
        "mask": [list(row) for row in masks_np] # Assign the common mask here
    })

    # Add multi-target predictions and actuals
    for i, col_name in enumerate(config["target_cols"]):
        df_summary[f"prediction_{col_name}"] = list(preds_np[:, :, i])
        df_summary[f"actual_{col_name}"] = list(targets_np[:, :, i])

    # Save summary as feather (much faster than CSV)
    save_path.mkdir(parents=True, exist_ok=True)
    feather_path = save_path / f"eval_summary_step_{global_step}.feather"
    df_summary.to_feather(feather_path)
    logger.info(f"Saved evaluation summary to {feather_path}")

    # Compute bolus response metrics
    logger.info("🧪 Computing bolus response metrics...")
    # For bolus analysis, use mean BP column (typically the first target)
    bp_col_idx = 0  # phys_bp_mean_non_invasive is usually first
    
    # Get bolus data as proper numpy array instead of lists
    bolus_data_raw = np.concatenate(all_last_input, axis=0)  # [N, num_bolus_cols]
    
    bolus_metrics = compute_bolus_response_metrics(
        preds=preds_np[:, :, bp_col_idx],  # Only BP predictions
        targets=targets_np[:, :, bp_col_idx],  # Only BP targets 
        last_input=np.concatenate(all_last5, axis=0)[:, -1],  # Last known BP value (last of 5)
        last_input_bolus=bolus_data_raw,  # Raw numpy array
        num_bolus_cols=len(config.get('bolus_cols', []))
    )
    
    # Log bolus metrics to wandb
    if bolus_metrics:
        try:
            wandb.log(bolus_metrics, step=global_step)
            logger.info(f"✅ Logged {len(bolus_metrics)} bolus metrics")
        except (AttributeError, wandb.Error):
            logger.info(f"✅ Computed {len(bolus_metrics)} bolus metrics (wandb not initialized)")
    else:
        logger.info("⚠️ No bolus events detected in evaluation data")

    # Return metrics dictionary
    return {
        "loss": avg_loss,
        "num_batches": num_batches,
        "df_summary": df_summary
    }
