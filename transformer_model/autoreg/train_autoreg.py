import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from losses import compute_loss
from losses.custom_losses import compute_bolus_conditional_loss
from losses.loss_schedule import get_scheduled_loss_weights
from utils import compute_bolus_mask, move_batch_to_device
from bolus_utils import build_group_bolus_masks

logger = logging.getLogger(__name__)

def safe_log_metrics(metrics: Dict[str, float], step: int):
    valid = {}
    for k, v in metrics.items():
        # Convert tensors to scalars immediately
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                valid[k] = float(v.detach().cpu().item())
            else:
                continue  # Skip non-scalar tensors
        elif isinstance(v, (float, int)) and not (torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v))):
            valid[k] = float(v)
    
    if valid:
        # Only log if wandb is initialized
        try:
            if wandb.run is not None:
                wandb.log(valid, step=step)
        except:
            pass  # Skip logging if wandb not initialized

    # Clear the valid dict
    del valid


def train_autoreg_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    config: Dict,
    device: torch.device,
    epoch: int,
    global_step: int = 0,
    save_path: Path = Path("."),
    scaler: Optional[GradScaler] = None,
    loss_module: Optional[nn.Module] = None,
    loss_weight_module: Optional[nn.Module] = None
) -> Tuple[Dict[str, float], int]:
    model.train()
    if loss_weight_module:
        loss_weight_module.train()

    task_mode = config.get("task_mode", "autoreg")
    use_amp = config.get("use_mixed_precision", False) and torch.cuda.is_available()
    grad_clip_max_norm = config.get("grad_clip_max_norm", 0.0)

    try:
        total_steps = len(dataloader)
    except TypeError:
        total_steps = config.get("approx_batches_per_epoch", 1000)

    log_every = config.get("log_every_batches", 100)  # OPTIMIZED: Reduce logging frequency

    epoch_loss = 0.0
    metrics_acc: Dict[str, float] = {}
    log_rows = []
    consecutive_nan_losses = 0
    max_consecutive_nans = 50  # Stop training if too many consecutive NaN losses

    logger.info(f"Epoch {epoch} start. Task mode: {task_mode}")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", total=total_steps)):
        global_step += 1 # Increment global_step at the very beginning of the loop

        optimizer.zero_grad(set_to_none=True)
        # OPTIMIZED: Faster device transfer with proper dtype conversion
        float_keys = ["vitals", "meds", "gases", "bolus", "static_num"]
        for k, v in batch.items():
            if torch.is_tensor(v):
                if k in float_keys:
                    batch[k] = v.to(device=device, dtype=torch.float32, non_blocking=True)
                else:
                    batch[k] = v.to(device, non_blocking=True)
            elif k == "static_cat" and isinstance(v, dict):
                batch[k] = {sk: sv.to(device, non_blocking=True) for sk, sv in v.items()}

        with autocast(enabled=use_amp, device_type="cuda"):
            preds, fused_logits, bp_logits = model(
                vitals=batch["vitals"],
                meds=batch["meds"],
                gases=batch["gases"],  # FIXED: Added missing gases parameter
                bolus=batch["bolus"],
                attention_mask=batch["attention_mask"].to(device, dtype=torch.bool),
                static_cat=batch.get("static_cat"),
                static_num=batch.get("static_num"),
                future_steps=config["future_steps"],
            )

            # Determine loss weights
            if loss_module is None:
                lw = loss_weight_module() if loss_weight_module else get_scheduled_loss_weights(config, epoch)
                # Only log loss weights at the first logging step or when it's time to log
                if global_step == 1 or (batch_idx % log_every == 0):
                    safe_log_metrics({f"loss_weights/{k}": v for k, v in lw.items()}, step=global_step)
            else:
                lw = None

            loss = torch.tensor(0.0, device=device)
            batch_metrics: Dict[str, float] = {}

            # Autoregressive loss
            if task_mode in ("autoreg", "joint"):
                # Compute trigger mask for bolus conditional loss
                trigger_mask = (batch["last_input_bolus"] > 0).any(dim=-1)
                
                # NOTE: These masks are computed but not currently used in the loss function.
                # They may be needed for future bolus analysis or more complex loss weighting.
                # Keeping minimal computation for potential future use:
                # bolus_mask = compute_bolus_mask(batch, config["future_steps"], device)  # Not used in current loss
                # group_mask = build_group_bolus_masks(batch["bolus"], config["bolus_cols"])  # Not used in current loss

                if loss_module:
                    l, m = loss_module(
                        preds=preds,
                        targets=batch["target"],
                        last_input=batch["last_known_bp"],
                        bolus_trigger_mask=trigger_mask,
                        last_input_bolus=batch["last_input_bolus"],
                        onset_labels=batch.get("hypo_onset_label"),
                        onset_logits=(fused_logits if config.get("use_hypo_onset_fused") else bp_logits),
                        onset_types=batch.get("hypo_onset_type"),
                        last5_bp_values=batch.get("last5_bp_values"),
                    )
                else:
                    # SIMPLIFIED: Use last_input_bolus directly for drug weighting
                    trigger_mask = (batch["last_input_bolus"] > 0).any(dim=-1)

                    # Get target flags for imputation masking (if available)
                    target_flags = batch.get("target_flags")
                    if target_flags is not None:
                        target_flags = target_flags.to(device=device, dtype=torch.bool, non_blocking=True)

                    l, m = compute_bolus_conditional_loss(
                        preds=preds,
                        targets=batch["target"],
                        loss_weights=lw,
                        config=config,
                        bolus_trigger_mask=trigger_mask,
                        last_input_bolus=batch["last_input_bolus"],  # Use this directly
                        target_flags=target_flags  # Pass imputation mask
                    )

                    # Log imputation masking statistics (only occasionally to avoid overhead)
                    if target_flags is not None and batch_idx % log_every == 0:
                        total_values = target_flags.numel()
                        measured_values = target_flags.sum().item()
                        imputed_ratio = 1.0 - (measured_values / total_values) if total_values > 0 else 0.0
                        m['imputation/target_imputed_ratio'] = imputed_ratio
                        m['imputation/target_measured_count'] = measured_values

                    # Clean up
                    del trigger_mask
                    if target_flags is not None:
                        del target_flags
                    
                loss = loss + l
                # FIXED: Filter out non-numeric metrics (like drug names)
                batch_metrics.update({k: float(v) for k, v in m.items() 
                                     if isinstance(v, (int, float, torch.Tensor)) and k != 'dominant_drug'})
                # Clean up loss component
                del l, m

            # Hypotension classification loss
            if task_mode in ("hypo", "joint"):
                if task_mode == "hypo":
                    # Hypotension-only mode: compute only classification loss
                    hl = torch.tensor(0.0, device=device)
                    hm = {}
                    
                    # Direct BCE loss for hypotension classification
                    if fused_logits is not None and batch.get("hypo_onset_label") is not None:
                        # Apply masking (exclude ongoing samples)
                        onset_types = batch.get("hypo_onset_type")
                        if onset_types is not None:
                            mask = torch.tensor([t in ["true_onset", "none"] for t in onset_types], 
                                              dtype=torch.bool, device=device)
                            if mask.any():
                                filtered_logits = fused_logits.squeeze(-1)[mask]
                                filtered_labels = batch["hypo_onset_label"][mask].float()
                                
                                # Stability checks
                                if torch.any(torch.isnan(filtered_logits)) or torch.any(torch.isinf(filtered_logits)):
                                    logger.warning(f"[Hypo Only] NaN/Inf in logits detected, skipping batch")
                                    hl = torch.tensor(0.0, device=device)
                                elif torch.any(torch.isnan(filtered_labels)):
                                    logger.warning(f"[Hypo Only] NaN in labels detected, skipping batch")
                                    hl = torch.tensor(0.0, device=device)
                                else:
                                    # Clamp logits to prevent overflow
                                    filtered_logits = torch.clamp(filtered_logits, min=-10.0, max=10.0)
                                    
                                    pos_weight = torch.tensor(config.get("hypo_pos_weight", 20.0), device=device)
                                    hl = F.binary_cross_entropy_with_logits(
                                        filtered_logits, filtered_labels, pos_weight=pos_weight
                                    )
                                    
                                    # Additional NaN check after loss computation
                                    if torch.isnan(hl) or torch.isinf(hl):
                                        logger.warning(f"[Hypo Only] NaN/Inf loss after BCE, setting to 0")
                                        hl = torch.tensor(0.0, device=device)
                                    
                                    hm["hypo_onset_fused"] = hl.item()
                                    
                                    # Debug logging
                                    if config.get("debug_hypo_gradients", False) and batch_idx % 100 == 0:
                                        logger.debug(f"[Hypo Only] Samples: {filtered_logits.numel()}, "
                                                   f"Positive: {filtered_labels.sum().item():.0f}, "
                                                   f"Logits range: [{filtered_logits.min().item():.3f}, {filtered_logits.max().item():.3f}], "
                                                   f"Loss: {hl.item():.4f}")
                            else:
                                logger.warning(f"[Hypo Only] No valid samples after masking")
                                hl = torch.tensor(0.0, device=device)
                        else:
                            logger.warning(f"[Hypo Only] No onset_types in batch")
                            hl = torch.tensor(0.0, device=device)
                else:
                    # Joint mode: use full loss computation
                    if loss_module:
                        hl, hm = loss_module(
                            preds=preds,
                            targets=batch["target"],
                            last_input=batch["last_known_bp"],
                            bolus_trigger_mask=None,
                            onset_labels=batch.get("hypo_onset_label"),
                            onset_logits=(fused_logits if config.get("use_hypo_onset_fused") else bp_logits),
                            onset_types=batch.get("hypo_onset_type"),
                        )
                    else:
                        hl, hm = compute_loss(
                            preds=preds,
                            targets=batch["target"],
                            loss_weights=lw,
                            config=config,
                            hypo_fused_logits=fused_logits,
                            hypo_bp_logits=bp_logits,
                            onset_labels=batch.get("hypo_onset_label"),
                            onset_types=batch.get("hypo_onset_type"),
                        )
                loss = loss + hl
                # FIXED: Filter out non-numeric metrics for hypo loss too
                batch_metrics.update({k: float(v) for k, v in hm.items() 
                                     if isinstance(v, (int, float, torch.Tensor)) and k != 'dominant_drug'})
                # Clean up hypo loss components
                del hl, hm

        # Enhanced NaN detection and handling
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
            if torch.isnan(loss) or torch.isinf(loss):
                consecutive_nan_losses += 1
                logger.warning(f"NaN/Inf loss at step {global_step}, skipping backward (consecutive: {consecutive_nan_losses})")
                
                # Early stopping if too many consecutive NaN losses
                if consecutive_nan_losses >= max_consecutive_nans:
                    logger.error(f"Training stopped: {consecutive_nan_losses} consecutive NaN/Inf losses")
                    raise RuntimeError(f"Training unstable: {consecutive_nan_losses} consecutive NaN losses")
                    
            elif task_mode == "hypo" and loss.item() == 0.0:
                # In hypotension-only mode, 0 loss means no valid samples - this is normal
                consecutive_nan_losses = 0  # Reset counter for valid zero losses
            else:
                consecutive_nan_losses += 1
                logger.warning(f"Zero loss at step {global_step}, skipping backward (consecutive: {consecutive_nan_losses})")
            
            # CRITICAL: Clean up before continuing
            del loss, preds, fused_logits, bp_logits, batch_metrics, batch
            torch.cuda.empty_cache()
            continue
        else:
            # Reset consecutive NaN counter on successful loss computation
            consecutive_nan_losses = 0

        # Backprop (gradients already zeroed at loop start)
        if use_amp and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Enhanced gradient monitoring with hypotension-specific tracking
        if config.get("log_gradient_norm", False) and batch_idx % (log_every * 5) == 0:
            total_grad_norm = 0.0
            hypo_grad_norm = 0.0
            other_grad_norm = 0.0
            param_count = 0
            hypo_param_count = 0
            max_grad_norm = 0.0
            
            for name, param in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    param_norm = param.grad.norm().item()
                    
                    if 'hypo' in name.lower():
                        hypo_grad_norm += param_norm ** 2
                        hypo_param_count += 1
                        # Log individual hypotension gradients for debugging
                        if config.get("debug_hypo_gradients", False):
                            logger.debug(f"[Hypo Grad] {name}: {param_norm:.6f}")
                    else:
                        other_grad_norm += param_norm ** 2
                    
                    total_grad_norm += param_norm ** 2
                    max_grad_norm = max(max_grad_norm, param_norm)
                    param_count += 1
            
            if param_count > 0:
                total_grad_norm = (total_grad_norm ** 0.5)
                hypo_grad_norm = (hypo_grad_norm ** 0.5) if hypo_param_count > 0 else 0
                other_grad_norm = (other_grad_norm ** 0.5)
                
                grad_metrics = {
                    "grad_norm/total": total_grad_norm,
                    "grad_norm/avg_per_param": total_grad_norm / param_count,
                    "grad_norm/max": max_grad_norm,
                    "grad_norm/param_count": param_count
                }
                
                if hypo_param_count > 0:
                    grad_metrics["grad_norm/hypo_total"] = hypo_grad_norm
                    grad_metrics["grad_norm/hypo_avg"] = hypo_grad_norm / hypo_param_count
                    grad_metrics["grad_norm/other_total"] = other_grad_norm
                    
                safe_log_metrics(grad_metrics, step=global_step)

        # Enhanced gradient clipping
        # Note: When using mixed precision (scaler), NaN/Inf detection is handled automatically by scaler.step()
        # Manual checking would interfere with the scaler's state management
        if grad_clip_max_norm > 0:
            # Only manually check for NaN if NOT using mixed precision
            if not (use_amp and scaler):
                nan_grads = False
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        logger.warning(f"NaN/Inf gradients detected in {name}")
                        nan_grads = True
                        break

                if nan_grads:
                    logger.warning(f"Skipping optimizer step due to NaN gradients at step {global_step}")
                    optimizer.zero_grad(set_to_none=True)
                    del loss, preds, fused_logits, bp_logits, batch_metrics, batch
                    torch.cuda.empty_cache()
                    continue

            # Perform gradient clipping
            if use_amp and scaler:
                # Unscale before clipping when using mixed precision
                scaler.unscale_(optimizer)

            total_norm = nn.utils.clip_grad_norm_(
                list(model.parameters()) +
                (list(loss_weight_module.parameters()) if loss_weight_module else []),
                grad_clip_max_norm
            )

            # Log extreme gradient norms
            if total_norm > grad_clip_max_norm * 2:
                logger.warning(f"Large gradient norm {total_norm:.2f} clipped to {grad_clip_max_norm}")

        # Optimizer step
        if use_amp and scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if scheduler and config.get("use_lr_scheduler", True):
            scheduler.step()

        # CRITICAL: Detach loss value to avoid keeping computation graph
        loss_item = loss.detach().item()
        
        # Accumulate
        epoch_loss += loss_item
        for k, v in batch_metrics.items():
            if isinstance(v, (int, float)):  # Only accumulate numeric values
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
            else:  # For strings like 'dominant_drug', keep the latest value
                metrics_acc[k] = v

        # OPTIMIZED: Only log if we're going to use it
        if batch_idx % log_every == 0 or batch_idx == total_steps - 1:
            log_rows.append({"step": global_step, "loss": loss_item, **batch_metrics})

        # CRITICAL: Complete memory cleanup - delete ALL batch variables
        del loss, preds, fused_logits, bp_logits, batch_metrics, batch
        
        # Clean up any drug-specific tensors from bolus_conditional_loss
        if 'bolus_data' in locals():
            del bolus_data
        if 'trigger_mask' in locals():
            del trigger_mask
        
        # OPTIMIZED: Less frequent autocast cache clearing
        if use_amp and batch_idx % 20 == 0:
            torch.clear_autocast_cache()
        
        # OPTIMIZED: Less frequent cache clearing for better performance
        if batch_idx % 50 == 0:  # Every 50 batches instead of 5
            torch.cuda.empty_cache()
        
        # OPTIMIZED: Less frequent garbage collection
        if batch_idx % 200 == 0:  # Every 200 batches instead of 25
            import gc
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            
        # FIXED: Clear log_rows periodically to prevent memory buildup
        if len(log_rows) > 1000:
            # Save intermediate results and clear
            temp_df = pd.DataFrame(log_rows)
            temp_path = save_path / f"train_epoch{epoch}_batch{batch_idx}.csv"
            save_path.mkdir(parents=True, exist_ok=True)
            temp_df.to_csv(temp_path, index=False)
            log_rows.clear()
            del temp_df

        # Logging
        if batch_idx % log_every == 0:
            # Only average numeric metrics
            avg_metrics = {}
            for k, v in metrics_acc.items():
                if isinstance(v, (int, float)):
                    avg_metrics[f"train/{k}"] = v / (batch_idx + 1)
            safe_log_metrics(avg_metrics, step=global_step)
            
            # OPTIMIZED: Simplified drug logging (only every 500 batches)
            if 'dominant_drug' in metrics_acc and batch_idx % 500 == 0:
                logger.info(f"🎯 Dominant drug: {metrics_acc.get('dominant_drug', 'unknown')}")
            
            # Log individual loss components to wandb including hypotension losses
            loss_components = {}
            for key in ['mse', 'dilate', 'shape', 'temporal', 'mse_effective_weight', 'dilate_effective_weight', 
                       'hypo_loss', 'hypo_fused_loss', 'hypo_bp_loss']:
                if key in metrics_acc:
                    loss_components[f"train/{key}"] = metrics_acc[key] / (batch_idx + 1)
            
            if loss_components:
                safe_log_metrics(loss_components, step=global_step)

    if global_step == 0:
        raise RuntimeError("No training batches processed")

    # Final metrics - only average numeric values
    averaged = {}
    for k, v in metrics_acc.items():
        if isinstance(v, (int, float)):
            averaged[k] = v / (global_step % total_steps or total_steps)
    averaged["loss"] = epoch_loss / (global_step % total_steps or total_steps)
    safe_log_metrics({**{"epoch": epoch}, **averaged}, step=global_step)

    # Save CSV log
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / f"train_epoch{epoch}.csv"
    if log_rows:  # Only save if there are rows left
        pd.DataFrame(log_rows).to_csv(csv_path, index=False)
        logger.info(f"Saved training log to {csv_path}")
    
    logger.info(f"Epoch {epoch+1} done: avg_loss={averaged['loss']:.6f}")

    return averaged, global_step