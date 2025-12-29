import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from hypo_onset_predictor import HypoOnsetPredictor
from utils import move_batch_to_device
from eval_hypo_classifier import eval_hypo_classifier


def train_hypo_classifier(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    config,
    device,
    epoch,
    mask_ongoing=True,  # NEW: Control ongoing masking
    save_predictions=False,
    output_dir="predictions"
):
    """
    Trains the HypoOnsetPredictor model for one epoch with loss-level masking of ongoing cases.
    
    Args:
        model (HypoOnsetPredictor): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        config (dict): Configuration dictionary with hyperparameters.
        device (torch.device): Device to run the model on (CPU/GPU).
        epoch (int): Current epoch number.
        mask_ongoing (bool): Whether to mask ongoing cases in loss computation.
        save_predictions (bool): Whether to save predictions to a CSV file.
        output_dir (str): Directory to save prediction tables.

    Returns:
        dict: Dictionary containing average training loss, validation metrics, and masking stats.
    """
    # Initialize metrics
    metrics = {"epoch": epoch, "avg_train_loss": 0.0}
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # NEW: Track masking statistics
    masking_stats = {
        'total_samples': 0,
        'ongoing_masked': 0,
        'true_onset_samples': 0,
        'none_samples': 0,
        'samples_used_for_loss': 0,
        'mask_rate': 0.0
    }
    
    # Prediction collection
    all_preds, all_labels, all_case_ids, all_timesteps = [], [], [], []
    all_types, all_masks = [], []  # NEW: Track types and masks

    # Training loop
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
        try:
            optimizer.zero_grad(set_to_none=True)
            move_batch_to_device(batch, device)

            # Forward pass on ALL samples (including ongoing)
            logits = model(
                vitals=batch["vitals"],
                meds=batch["meds"],
                bolus=batch["bolus"],
                attention_mask=batch["attention_mask"].to(device, dtype=torch.bool),
                gases=batch.get("gases"),  # NEW: Pass gases to model
                static_cat=batch.get("static_cat"),
                static_num=batch.get("static_num"),
            )

            labels = batch["hypo_onset_label"].float()
            types = batch["hypo_onset_type"]
            
            # Update total samples count
            batch_size = len(labels)
            masking_stats['total_samples'] += batch_size
            
            # NEW: CREATE MASK - Include only true_onset and none, exclude ongoing
            if mask_ongoing:
                mask = torch.tensor([
                    t in ["true_onset", "none"] for t in types
                ], dtype=torch.bool, device=labels.device)
                
                # Count what we're masking/using
                ongoing_count = sum(1 for t in types if t == "ongoing")
                true_onset_count = sum(1 for t in types if t == "true_onset")
                none_count = sum(1 for t in types if t == "none")
                
                masking_stats['ongoing_masked'] += ongoing_count
                masking_stats['true_onset_samples'] += true_onset_count
                masking_stats['none_samples'] += none_count
                masking_stats['samples_used_for_loss'] += mask.sum().item()
                
                if batch_idx % 50 == 0:  # Log every 50 batches
                    print(f"[Batch {batch_idx}] 🎭 Masking: {ongoing_count} ongoing, using {mask.sum().item()}/{batch_size} samples", flush=True)
                    
            else:
                # No masking - use all samples
                mask = torch.ones_like(labels, dtype=torch.bool)
                masking_stats['samples_used_for_loss'] += batch_size

            # Apply mask to get samples used for loss computation
            masked_logits = logits[mask]
            masked_labels = labels[mask]
            
            if masked_labels.numel() == 0:
                print(f"[Batch {batch_idx}] ⚠️ No samples after masking - skipping", flush=True)
                continue

            # Debug: Label distribution (only for used samples)
            if batch_idx % 100 == 0:  # Less frequent logging
                label_vals, counts = torch.unique(masked_labels, return_counts=True)
                label_dist = {int(k.item()): int(v.item()) for k, v in zip(label_vals, counts)}
                
                # Type distribution for this batch
                used_types = [types[i] for i, m in enumerate(mask) if m]
                type_dist = {t: used_types.count(t) for t in set(used_types)}
                
                print(f"[Batch {batch_idx}] ✅ Used samples - Label counts: {label_dist}, Type counts: {type_dist}", flush=True)

            # Compute BCE loss on masked samples only
            probs_masked = torch.sigmoid(masked_logits)
            pos_weight_val = config.get("hypo_pos_weight", 10.0)
            pos_weight = torch.tensor(pos_weight_val, device=device)
            loss = F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, pos_weight=pos_weight
            )
            
            if batch_idx % 100 == 0:
                print(f"[Batch {batch_idx}] 📊 Loss: {loss.item():.4f}, Used: {masked_labels.numel()}/{batch_size}", flush=True)

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Batch {batch_idx}] ❌ Invalid loss: {loss.item()}, skipping", flush=True)
                continue

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Collect predictions (for ALL samples, including masked ones)
            if save_predictions:
                all_probs = torch.sigmoid(logits)  # Predictions for ALL samples
                all_preds.append(all_probs.cpu())
                all_labels.append(labels.cpu())
                all_case_ids.extend(batch["mpog_case_id"])
                all_timesteps.append(batch["prediction_timestep"].cpu())
                all_types.extend(types)  # NEW: Track onset types
                all_masks.append(mask.cpu())  # NEW: Track which were masked

            # Enhanced logging to WandB
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "samples_used": masked_labels.numel(),
                    "samples_total": batch_size,
                    "samples_masked": (~mask).sum().item(),
                    "mask_rate": (~mask).float().mean().item(),
                    "logits_mean": logits.mean().item(),
                    "logits_std": logits.std().item(),
                    "probs_min_used": probs_masked.min().item(),
                    "probs_max_used": probs_masked.max().item(),
                    "label_pos_ratio": masked_labels.mean().item(),
                    "batch_idx": batch_idx,
                },
                step=epoch * 1000 + batch_idx,
            )

            total_loss += loss.item()
            num_batches += 1

        except Exception as e:
            print(f"[Batch {batch_idx}] ❌ Error: {e}", flush=True)
            continue

    # Compute metrics and masking summary
    avg_loss = total_loss / max(num_batches, 1)
    metrics["avg_train_loss"] = avg_loss
    
    # Calculate final masking rate
    if masking_stats['total_samples'] > 0:
        masking_stats['mask_rate'] = masking_stats['ongoing_masked'] / masking_stats['total_samples']
    
    # Log masking summary
    print(f"\n📊 EPOCH {epoch} MASKING SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples processed: {masking_stats['total_samples']}")
    print(f"Ongoing samples masked: {masking_stats['ongoing_masked']}")
    print(f"True onset samples used: {masking_stats['true_onset_samples']}")
    print(f"None samples used: {masking_stats['none_samples']}")
    print(f"Total samples used for loss: {masking_stats['samples_used_for_loss']}")
    print(f"Masking rate: {masking_stats['mask_rate']*100:.1f}%")
    print(f"Average loss: {avg_loss:.6f}")
    print(f"{'='*50}")
    
    # Log to WandB
    wandb.log({
        "epoch": epoch, 
        "avg_train_loss": avg_loss,
        "epoch_masking_stats": masking_stats
    })
    
    # Add masking stats to metrics
    metrics.update({"masking_stats": masking_stats})

    # Save training predictions with masking info
    if save_predictions and all_preds:
        try:
            preds = torch.cat(all_preds).numpy()
            labels = torch.cat(all_labels).numpy()
            timesteps = torch.cat(all_timesteps).numpy()
            masks = torch.cat(all_masks).numpy()

            # Create enhanced DataFrame with masking information
            prediction_table = pd.DataFrame({
                "mpog_case_id": all_case_ids,
                "prediction_timestep": timesteps,
                "predicted_prob": preds,
                "true_label": labels,
                "hypo_onset_type": all_types,  # NEW: Include onset type
                "was_masked": ~masks,  # NEW: Track if sample was masked
                "used_for_training": masks,  # NEW: Track if used for loss
                "epoch": epoch,
                "dataset_split": "train"
            })

            # Add last5_bp_values if available
            try:
                all_last5_bp = []
                for batch in train_dataloader:
                    if "last5_bp_values" in batch:
                        all_last5_bp.append(batch["last5_bp_values"].cpu())
                if all_last5_bp:
                    last5_bp = torch.cat(all_last5_bp).numpy()
                    for i in range(5):
                        prediction_table[f"last_bp_t-{4-i}"] = last5_bp[:, i]
            except Exception as e:
                print(f"⚠️ Could not add last5_bp_values: {e}")

            # Save to CSV
            import os
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"train_predictions_epoch_{epoch}.csv")
            prediction_table.to_csv(output_path, index=False)
            print(f"✅ Training prediction table saved to {output_path}", flush=True)

            # Log summary of saved predictions
            print(f"📋 Prediction table summary:")
            print(f"  Total predictions: {len(prediction_table)}")
            print(f"  Masked samples: {prediction_table['was_masked'].sum()}")
            print(f"  Used for training: {prediction_table['used_for_training'].sum()}")
            type_counts = prediction_table['hypo_onset_type'].value_counts()
            for onset_type, count in type_counts.items():
                masked_count = prediction_table[
                    (prediction_table['hypo_onset_type'] == onset_type) & 
                    (prediction_table['was_masked'])
                ].shape[0]
                print(f"  {onset_type}: {count} total, {masked_count} masked")

            # Log table to WandB
            wandb.log({"train_prediction_table": wandb.Table(dataframe=prediction_table)})

        except Exception as e:
            print(f"❌ Error saving training prediction table: {e}", flush=True)

    # Validation with masking awareness
    if val_dataloader:
        print(f"\n📊 Running validation for epoch {epoch}...", flush=True)
        try:
            # Use updated eval function that handles masking
            val_metrics = eval_hypo_classifier_with_masking(
                model, val_dataloader, device, mask_ongoing=mask_ongoing
            )
            
            metrics.update({
                "val_auc": val_metrics["auc"],
                "val_f1": val_metrics["f1"],
                "val_masking_stats": val_metrics.get("masking_stats", {})
            })
            
            wandb.log({
                "val_auc": val_metrics["auc"],
                "val_f1": val_metrics["f1"],
                "val_masking_stats": val_metrics.get("masking_stats", {})
            }, step=epoch * 1000 + num_batches)
            
            print(f"✅ Validation metrics: AUC={val_metrics['auc']:.4f}, F1={val_metrics['f1']:.4f}", flush=True)
            if "masking_stats" in val_metrics:
                val_mask_stats = val_metrics["masking_stats"]
                print(f"   Validation masking: {val_mask_stats['ongoing_masked']} ongoing masked, {val_mask_stats['samples_used_for_loss']} used", flush=True)

            # Collect validation predictions with masking info
            if save_predictions:
                val_predictions = collect_validation_predictions_with_masking(
                    model, val_dataloader, device, mask_ongoing, epoch
                )
                
                # Save validation predictions
                val_output_path = os.path.join(output_dir, f"val_predictions_epoch_{epoch}.csv")
                val_predictions.to_csv(val_output_path, index=False)
                print(f"✅ Validation prediction table saved to {val_output_path}", flush=True)

                # Log to WandB
                wandb.log({"val_prediction_table": wandb.Table(dataframe=val_predictions)})

                # Log confusion matrix (only for samples used in evaluation)
                try:
                    eval_samples = val_predictions[val_predictions['used_for_evaluation']]
                    if len(eval_samples) > 0:
                        preds_bin = (eval_samples['predicted_prob'] > 0.5).astype(int)
                        true_labels = eval_samples['true_label'].astype(int)
                        tn, fp, fn, tp = confusion_matrix(true_labels, preds_bin).ravel()
                        wandb.log({
                            "val_TP": tp, "val_TN": tn, "val_FP": fp, "val_FN": fn,
                        }, step=epoch * 1000 + num_batches)
                        print(f"🔍 Validation Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}", flush=True)
                except Exception as e:
                    print(f"❌ Error logging confusion matrix: {e}", flush=True)

        except Exception as e:
            print(f"❌ Error during validation: {e}", flush=True)

    return metrics


def eval_hypo_classifier_with_masking(model, dataloader, device, mask_ongoing=True):
    """
    Evaluation function that handles masking of ongoing cases.
    Returns metrics computed only on non-masked samples, but predictions for all.
    """
    model.eval()
    all_preds, all_labels, all_types = [], [], []
    masking_stats = {
        'total_samples': 0,
        'ongoing_masked': 0,
        'samples_used_for_loss': 0
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            move_batch_to_device(batch, device)

            logits = model(
                vitals=batch["vitals"],
                meds=batch["meds"],
                bolus=batch["bolus"],
                attention_mask=batch["attention_mask"].to(device, dtype=torch.bool),
                gases=batch.get("gases"),  # NEW: Pass gases to model
                static_cat=batch.get("static_cat"),
                static_num=batch.get("static_num"),
            )

            labels = batch["hypo_onset_label"].float()
            types = batch["hypo_onset_type"]

            masking_stats['total_samples'] += len(labels)
            
            # Apply same masking logic as training
            if mask_ongoing:
                mask = torch.tensor([
                    t in ["true_onset", "none"] for t in types
                ], dtype=torch.bool, device=labels.device)
                masking_stats['ongoing_masked'] += (~mask).sum().item()
                masking_stats['samples_used_for_loss'] += mask.sum().item()
            else:
                mask = torch.ones_like(labels, dtype=torch.bool)
                masking_stats['samples_used_for_loss'] += len(labels)

            # Collect predictions for ALL samples
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_types.extend(types)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # Compute metrics only on samples that would be used for training
    if mask_ongoing:
        eval_mask = np.array([t in ["true_onset", "none"] for t in all_types])
        eval_preds = preds[eval_mask]
        eval_labels = labels[eval_mask]
    else:
        eval_preds = preds
        eval_labels = labels
    
    # Compute metrics
    if len(np.unique(eval_labels)) > 1:  # Need both classes
        auc = roc_auc_score(eval_labels, eval_preds)
        f1 = f1_score(eval_labels, eval_preds > 0.5)
    else:
        auc, f1 = 0.0, 0.0
        print("⚠️ Only one class present in evaluation set")

    return {
        "auc": auc, 
        "f1": f1, 
        "masking_stats": masking_stats,
        "all_predictions": preds,
        "all_labels": labels,
        "all_types": all_types
    }


def collect_validation_predictions_with_masking(model, dataloader, device, mask_ongoing, epoch):
    """Collect validation predictions with masking information"""
    model.eval()
    all_preds, all_labels, all_case_ids, all_timesteps, all_types = [], [], [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            move_batch_to_device(batch, device)
            
            logits = model(
                vitals=batch["vitals"],
                meds=batch["meds"],
                bolus=batch["bolus"],
                attention_mask=batch["attention_mask"].to(device, dtype=torch.bool),
                gases=batch.get("gases"),  # NEW: Pass gases to model
                static_cat=batch.get("static_cat"),
                static_num=batch.get("static_num"),
            )

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_labels.append(batch["hypo_onset_label"].cpu())
            all_case_ids.extend(batch["mpog_case_id"])
            all_timesteps.append(batch["prediction_timestep"].cpu())
            all_types.extend(batch["hypo_onset_type"])

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    timesteps = torch.cat(all_timesteps).numpy()
    
    # Create mask for evaluation
    if mask_ongoing:
        eval_mask = np.array([t in ["true_onset", "none"] for t in all_types])
    else:
        eval_mask = np.ones(len(all_types), dtype=bool)

    # Create DataFrame
    val_prediction_table = pd.DataFrame({
        "mpog_case_id": all_case_ids,
        "prediction_timestep": timesteps,
        "predicted_prob": preds,
        "true_label": labels,
        "hypo_onset_type": all_types,
        "was_masked": ~eval_mask,
        "used_for_evaluation": eval_mask,
        "epoch": epoch,
        "dataset_split": "validation"
    })

    return val_prediction_table