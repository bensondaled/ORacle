import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import wandb
import logging
from typing import Dict, Any, Optional
from utils import to_numpy, get_device
from logging_utils import configure_wandb_custom_charts, log_rmse_table_as_artifact
from bolus_analysis import analyze_bolus_effects, log_combined_bolus_response_wandb, log_separate_bolus_response_wandb,compute_bolus_response_metrics
from hypo_metrics import compute_seq_rule_cm, log_confusion_matrix, log_learned_classifier_metrics,compute_rule_vs_learned_cm
from summary_metrics import postprocess_and_log_autoreg_metrics
from eval_autoreg import eval_autoreg
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

# === Constants ===
BP_SCALE = 1

def evaluate_best_model(
    model,
    checkpoint_path: Path,
    loaders: Dict[str, Any],
    config: dict,
    run_name: str,
    save_dir: Path,
    test_df: pd.DataFrame,
    loss_module=None,
    loss_weight_module=None,
    best_epoch: int = -1,
    global_step: int = 0
) -> pd.DataFrame:
    """
    Evaluate the best model checkpoint on the test set and save results.
    """
    logger.info(f"📈 Evaluating best model from: {checkpoint_path.name}")
    device = get_device()

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_outputs = eval_autoreg(
        model=model,
        dataloader=loaders['test'],
        save_path=save_dir/"eval_csvs",
        config=config,
        device=device,
        loss_module=loss_module,
        loss_weight_module=loss_weight_module,
        global_step=global_step,
    )

    df_sum = test_outputs['df_summary']
    logger.debug(f"Received df_summary with columns: {df_sum.columns}")
    
    # Get prediction and actual columns for multi-target case
    pred_cols = [col for col in df_sum.columns if col.startswith('prediction_')]
    actual_cols = [col for col in df_sum.columns if col.startswith('actual_')]
    
    if pred_cols:
        # Multi-target case: combine all prediction columns
        pred_arrays = [np.array(df_sum[col].tolist()) for col in pred_cols]
        actual_arrays = [np.array(df_sum[col].tolist()) for col in actual_cols]
        preds = np.stack(pred_arrays, axis=-1)  # [N, T, num_targets]
        actuals = np.stack(actual_arrays, axis=-1)  # [N, T, num_targets]
        logger.debug(f"Multi-target prediction shape: {preds.shape}")
        logger.debug(f"Multi-target actual shape: {actuals.shape}")
    else:
        # Legacy single target case
        preds = df_sum['prediction']
        actuals = df_sum['actual']
        logger.debug(f"Single target prediction shape: {np.array(preds).shape}")
        logger.debug(f"Single target actual shape: {np.array(actuals).shape}")
    
    logger.debug(f"Sample prediction: {preds[0][:5] if len(preds) > 0 else 'None'}")
    logger.debug(f"Sample actual: {actuals[0][:5] if len(actuals) > 0 else 'None'}")
    hypo_onset_type_arr = df_sum.get("hypo_onset_type")  # ← NEW LINE

    # Run the analysis (but we'll use df_sum which already has multi-target columns)
    evaluate_and_analyze_autoreg(
        preds=preds,
        targets=actuals,
        masks=df_sum['mask'],
        original_indices=df_sum['original_index'],
        case_ids=df_sum['mpog_case_id'],
        elapsed_time=df_sum['minutes_elapsed'],
        last5_bp_values=df_sum['last5_bp_values'],
        config=config,
        save_dir=save_dir / "best_model_eval",
        run_name=run_name,
        split='test',
        last_input_bolus=df_sum.get("last_input_bolus"),
        hypo_fused_logits=df_sum.get("hypo_fused_logits"),
        hypo_bp_logits=df_sum.get("hypo_bp_logits"),
        hypo_onset_labels=df_sum.get("hypo_onset_label"),
        hypo_onset_type_arr=hypo_onset_type_arr,  # ← NEW ARG
    )
    
    # Use df_sum which already has the multi-target columns
    output_df = df_sum

    full_dir = save_dir / "best_model_eval"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / f"best_model_test_full_{run_name}.feather"
    logger.info(f"Saving full test trajectories (actual + prediction) to: {full_path}")
    output_df.to_feather(full_path)

    postprocess_and_log_autoreg_metrics(output_df, test_df, run_name, save_dir=full_dir, config=config)

    log_data = {"global_step": best_epoch}
    for key, value in test_outputs.items():
        if key != 'df_summary':
            log_data[f"best_model_eval/{key}"] = value
    wandb.log(log_data)

    return output_df

def evaluate_and_analyze_autoreg(
    preds,
    targets,
    masks,
    original_indices,
    case_ids,
    elapsed_time,
    last5_bp_values,
    config: dict,
    save_dir: Path,
    run_name: str = "autoreg_eval",
    split: str = "test",
    last_input_bolus=None,
    hypo_fused_logits=None,
    hypo_bp_logits=None,
    hypo_onset_labels=None,
    hypo_onset_type_arr=None, 
) -> pd.DataFrame:
    """
    Analyze autoregressive predictions and compute metrics.
    """

    # Convert to stacked arrays
    try:
        # Check if data is already properly shaped (multi-target case)
        if isinstance(preds, np.ndarray) and len(preds.shape) >= 3:
            # Already properly shaped from our multi-target processing
            preds_stacked = preds
            targets_stacked = targets
        else:
            # Legacy case: try to squeeze last dimension if it's size 1
            preds_array = np.array(preds)
            targets_array = np.array(targets)
            
            if preds_array.shape[-1] == 1:
                preds_stacked = preds_array.squeeze(-1)
                targets_stacked = targets_array.squeeze(-1)
            else:
                # Multi-target case: keep all dimensions
                preds_stacked = preds_array
                targets_stacked = targets_array
        
        masks = np.stack([np.array(m).astype(bool) for m in masks])
    except Exception as e:
        logger.exception(f"❌ Failed stacking preds/targets/masks: {e}")
        raise

    logger.debug(f"Stacked shapes: preds={preds_stacked.shape}, targets={targets_stacked.shape}, masks={masks.shape}")
    
    # Update variable names for consistency
    preds = preds_stacked
    targets = targets_stacked
    
    # Get dimensions (handle both single and multi-target cases)
    if len(preds.shape) == 2:
        N, T = preds.shape
        num_targets = 1
    else:
        N, T, num_targets = preds.shape

    # Apply masks (handle broadcasting for multi-target case)
    if len(preds.shape) == 3:
        # Multi-target: expand masks to match [N, T, num_targets]
        masks_expanded = np.expand_dims(masks, axis=-1)
        preds_masked = np.where(masks_expanded, preds, np.nan)
        targets_masked = np.where(masks_expanded, targets, np.nan)
    else:
        # Single target case
        preds_masked = np.where(masks, preds, np.nan)
        targets_masked = np.where(masks, targets, np.nan)

    # === First-timestep metrics ===
    future_steps = config.get("future_steps", 1)
    if masks.shape[1] > 0:
        m0 = masks[:, 0].astype(bool)
        if m0.any():
            if len(preds.shape) == 3:
                # Multi-target: compute metrics for each target separately
                target_cols = config.get("target_cols", [f"target_{i}" for i in range(num_targets)])
                for i, target_name in enumerate(target_cols):
                    if i < num_targets:
                        targets_m0_i = targets_masked[m0, 0, i]
                        preds_m0_i = preds_masked[m0, 0, i]
                        try:
                            mae0_i = mean_absolute_error(targets_m0_i, preds_m0_i)
                            rmse0_i = np.sqrt(mean_squared_error(targets_m0_i, preds_m0_i))
                            wandb.log({f"first_step/{target_name}/mae": mae0_i, f"first_step/{target_name}/rmse": rmse0_i})
                        except Exception as e:
                            logger.warning(f"⚠️ Failed first-step metrics for {target_name}: {e}")
                            wandb.log({f"first_step/{target_name}/mae": np.nan, f"first_step/{target_name}/rmse": np.nan})
                
                # Also compute average across all targets
                try:
                    targets_m0_avg = np.mean(targets_masked[m0, 0], axis=-1)
                    preds_m0_avg = np.mean(preds_masked[m0, 0], axis=-1)
                    mae0_avg = mean_absolute_error(targets_m0_avg, preds_m0_avg)
                    rmse0_avg = np.sqrt(mean_squared_error(targets_m0_avg, preds_m0_avg))
                    wandb.log({"first_step/mae": mae0_avg, "first_step/rmse": rmse0_avg})
                except Exception as e:
                    logger.warning(f"⚠️ Failed first-step average metrics: {e}")
                    wandb.log({"first_step/mae": np.nan, "first_step/rmse": np.nan})
            else:
                # Single target case
                targets_m0 = targets_masked[m0, 0].flatten()
                preds_m0 = preds_masked[m0, 0].flatten()
                try:
                    mae0 = mean_absolute_error(targets_m0, preds_m0)
                    rmse0 = np.sqrt(mean_squared_error(targets_m0, preds_m0))
                    wandb.log({"first_step/mae": mae0, "first_step/rmse": rmse0})
                except Exception as e:
                    logger.warning(f"⚠️ Failed first-step metrics: {e}")
                    wandb.log({"first_step/mae": np.nan, "first_step/rmse": np.nan})
        else:
            wandb.log({"first_step/mae": np.nan, "first_step/rmse": np.nan})
    else:
        wandb.log({"first_step/mae": np.nan, "first_step/rmse": np.nan})

    # === Per-timestep error ===
    per_t_rmse = np.sqrt(np.nanmean((preds_masked - targets_masked)**2, axis=0))
    per_t_mae = np.nanmean(np.abs(preds_masked - targets_masked), axis=0)
    configure_wandb_custom_charts(split, per_t_rmse, per_t_mae)
    log_rmse_table_as_artifact(per_t_rmse, split, config)

    # === Hypotension labels + metrics ===
    # === Hypotension labels + metrics ===
    labels_np = to_numpy(hypo_onset_labels) if hypo_onset_labels is not None else None
    if labels_np is not None:
        labels_np = labels_np.squeeze(-1) if labels_np.ndim > 1 else labels_np
        if labels_np.shape[0] != N:
            labels_np = None

    def find_best_threshold(logits: np.ndarray, labels: np.ndarray) -> float:
        thresholds = np.linspace(0.05, 0.95, 19)
        best_f1 = -1
        best_thresh = 0.5
        for t in thresholds:
            preds = (logits >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        return best_thresh

    # === Learnable classifier: fused logits
    if hypo_fused_logits is not None and labels_np is not None:
        try:
            logits_np = np.asarray(hypo_fused_logits, dtype=np.float32).flatten()
            if np.isnan(logits_np).any():
                raise ValueError("NaNs found in fused logits.")
            best_thresh = find_best_threshold(logits_np, labels_np)
            wandb.log({"hypo_fused/best_f1_threshold": best_thresh})
            logger.info(f"🔍 Best threshold for hypo_fused = {best_thresh:.2f}")
            preds = (logits_np >= best_thresh).astype(int)
            log_learned_classifier_metrics(logits=preds, labels=labels_np, wandb_prefix="hypo_fused_opt")
        except Exception as e:
            logger.warning(f"⚠️ Skipping fused classifier evaluation due to error: {e}")

    # === Learnable classifier: BP logits
    if hypo_bp_logits is not None and labels_np is not None:
        try:
            logits_np = np.asarray(hypo_bp_logits, dtype=np.float32).flatten()
            if np.isnan(logits_np).any():
                raise ValueError("NaNs found in BP logits.")
            best_thresh = find_best_threshold(logits_np, labels_np)
            wandb.log({"hypo_bp/best_f1_threshold": best_thresh})
            logger.info(f"🔍 Best threshold for hypo_bp = {best_thresh:.2f}")
            preds = (logits_np >= best_thresh).astype(int)
            log_learned_classifier_metrics(logits=preds, labels=labels_np, wandb_prefix="hypo_bp_opt")
        except Exception as e:
            logger.warning(f"⚠️ Skipping BP classifier evaluation due to error: {e}")


    # Create summary DataFrame
    df_summary = pd.DataFrame({
        "mpog_case_id": case_ids,
        "minutes_elapsed": elapsed_time,
        "original_index": original_indices,
        "prediction": [p.tolist() for p in preds_masked],
        "actual": [t.tolist() for t in targets_masked],
        "mask": [m.tolist() for m in masks],
        "last5_bp_values": [to_numpy(x).tolist() for x in last5_bp_values],
        "last_known_bp": [float(x[-1]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan
                        for x in to_numpy(last5_bp_values)],
        "last_input_bolus": [to_numpy(x).tolist() if x is not None else None
                            for x in last_input_bolus] if last_input_bolus is not None else [None] * N,
        "hypo_onset_label": labels_np.tolist() if labels_np is not None else [None] * N,
        "hypo_onset_type": hypo_onset_type_arr if hypo_onset_type_arr is not None else [None] * N,  # ← ✅ add this line
    })
    
    # Add multi-target columns if we have them
    if len(preds.shape) == 3 and config.get("target_cols"):
        target_cols = config.get("target_cols", [])
        logger.debug(f"Adding multi-target columns for {len(target_cols)} targets: {target_cols}")
        logger.debug(f"Preds shape: {preds.shape}, targets shape: {targets.shape}")
        for i, col_name in enumerate(target_cols):
            if i < preds.shape[2]:
                df_summary[f"prediction_{col_name}"] = [p[:, i].tolist() for p in preds_masked]
                df_summary[f"actual_{col_name}"] = [t[:, i].tolist() for t in targets_masked]
                logger.debug(f"Added columns: prediction_{col_name}, actual_{col_name}")
    else:
        logger.warning(f"⚠️ Not adding multi-target columns. preds.shape={preds.shape}, target_cols={config.get('target_cols') if config else 'No config'}")
    # Save summary
    save_path = save_dir / f"{split}_pred_summary_{run_name}.feather"
    save_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_feather(save_path)
    logger.info(f"Saved summary to {save_path}")

    # Bolus analysis - focus on mean BP only  
    logger.info(f"🩹 Checking bolus analysis conditions: bolus_cols={config.get('bolus_cols') if config else None}, future_steps={config.get('future_steps') if config else None}, finetune_hypo_only={config.get('finetune_hypo_only', False) if config else None}")
    
    if config.get("bolus_cols") and config.get("future_steps") and not config.get("finetune_hypo_only", False):
        logger.info("🩹 Starting bolus analysis...")
        try:
            # Create a DataFrame with the right structure for bolus analysis
            # We need generic "prediction" and "actual" columns containing mean BP data
            
            # For multi-target, extract mean BP data for bolus analysis
            if len(preds.shape) == 3:
                logger.info("🩹 Extracting mean BP for bolus analysis")
                # Find mean BP index from target columns
                target_cols = config.get("target_cols", [])
                mean_bp_idx = None
                for i, col in enumerate(target_cols):
                    if "mean" in col.lower() or "map" in col.lower():
                        mean_bp_idx = i
                        logger.info(f"🩹 Found mean BP at index {i}: {col}")
                        break
                
                if mean_bp_idx is not None:
                    # Create DataFrame for bolus analysis with mean BP data
                    df_bolus = pd.DataFrame({
                        "mpog_case_id": case_ids,
                        "minutes_elapsed": elapsed_time,
                        "original_index": original_indices,
                        "prediction": [p[:, mean_bp_idx].tolist() for p in preds_masked],
                        "actual": [t[:, mean_bp_idx].tolist() for t in targets_masked],
                        "mask": [m.tolist() for m in masks],
                        "last5_bp_values": [to_numpy(x).tolist() for x in last5_bp_values],
                        "last_known_bp": [float(x[-1]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan
                                        for x in to_numpy(last5_bp_values)],
                        "last_input_bolus": [to_numpy(x).tolist() if x is not None else None
                                            for x in last_input_bolus] if last_input_bolus is not None else [None] * N,
                    })
                    
                    logger.info(f"🩹 Running bolus analysis with DataFrame shape: {df_bolus.shape}")
                    _, _, all_bolus = analyze_bolus_effects(df_bolus, config["bolus_cols"], config["future_steps"])
                    plot_mode = config.get("bolus_plot_mode", "combined")
                    if all_bolus.get("columns"):
                        logger.info(f"🩹 Logging bolus plots in {plot_mode} mode")
                        if plot_mode in ("combined", "both"):
                            log_combined_bolus_response_wandb(all_bolus, split)
                        if plot_mode in ("separate", "both"):
                            log_separate_bolus_response_wandb(all_bolus, split)
                    else:
                        logger.warning("⚠️ No bolus data found for plotting")
                else:
                    logger.warning("⚠️ No mean BP column found in target_cols for bolus analysis")
            else:
                logger.info("🩹 Single target case - using original DataFrame")
                # Single target case - create proper DataFrame structure
                df_bolus = pd.DataFrame({
                    "mpog_case_id": case_ids,
                    "minutes_elapsed": elapsed_time,
                    "original_index": original_indices,
                    "prediction": [p.tolist() for p in preds_masked],
                    "actual": [t.tolist() for t in targets_masked],
                    "mask": [m.tolist() for m in masks],
                    "last5_bp_values": [to_numpy(x).tolist() for x in last5_bp_values],
                    "last_known_bp": [float(x[-1]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan
                                    for x in to_numpy(last5_bp_values)],
                    "last_input_bolus": [to_numpy(x).tolist() if x is not None else None
                                        for x in last_input_bolus] if last_input_bolus is not None else [None] * N,
                })
                
                _, _, all_bolus = analyze_bolus_effects(df_bolus, config["bolus_cols"], config["future_steps"])
                plot_mode = config.get("bolus_plot_mode", "combined")
                if all_bolus.get("columns"):
                    if plot_mode in ("combined", "both"):
                        log_combined_bolus_response_wandb(all_bolus, split)
                    if plot_mode in ("separate", "both"):
                        log_separate_bolus_response_wandb(all_bolus, split)
        except Exception as e:
            logger.exception(f"Error during bolus response plotting: {e}")
    else:
        logger.info("🩹 Bolus analysis conditions not met - skipping")

    # Confusion matrix
    if not config.get("finetune_hypo_only", False):
        logger.info("🧪 hypo_onset_label counts:\n%s", df_summary["hypo_onset_label"].value_counts(dropna=False))
        if "hypo_onset_type" in df_summary:
            logger.info("🧪 hypo_onset_type counts:\n%s", df_summary["hypo_onset_type"].value_counts(dropna=False))

        cm_result = compute_seq_rule_cm(df_summary)
        wandb.log({
            "hypo_seq_rule/accuracy": cm_result["accuracy"],
            "hypo_seq_rule/precision": cm_result["precision"],
            "hypo_seq_rule/recall": cm_result["recall"],
            "hypo_seq_rule/f1": cm_result["f1"],
            "hypo_seq_rule/events_detected": cm_result["events_detected"],
            "hypo_seq_rule/total_candidates": cm_result["total_candidates"],
        })

        # ✅ NEW: Rule-based onset vs learned classifier logits
        if "hypo_onset_type" in df_summary and "hypo_bp_logits" in df_summary:
            cm_learned_vs_rule = compute_rule_vs_learned_cm(df_summary, logits_col="hypo_bp_logits")
            log_confusion_matrix(
                cm_learned_vs_rule["confusion_matrix"],
                wandb_key="hypo_seq_rule/rule_vs_learned_classifier_cm",
                title="True Onset vs Learned Classifier"
            )
            wandb.log({
                f"hypo_seq_rule/learned_comparison/{k}": v
                for k, v in cm_learned_vs_rule.items() if k != "confusion_matrix"
            })

        # Original CM between rule and BP<65 logic
        log_confusion_matrix(
            cm_result["confusion_matrix"],
            wandb_key="hypo_seq_rule/vs_learned_binary_cm",
            title="Rule-onset(>0) vs Learned Binary CM"
        )
        # === Bolus ΔBP metrics ===
        try:
            last_input = df_summary["last_known_bp"]
            
            # For multi-target, extract mean BP only for bolus metrics
            if len(preds.shape) == 3:
                # Find mean BP index
                target_cols = config.get("target_cols", [])
                mean_bp_idx = None
                for i, col in enumerate(target_cols):
                    if "mean" in col.lower() or "map" in col.lower():
                        mean_bp_idx = i
                        break
                
                if mean_bp_idx is not None:
                    preds_bp = preds_masked[:, :, mean_bp_idx]
                    targets_bp = targets_masked[:, :, mean_bp_idx]
                    logger.debug(f"Calling bolus metrics with shapes: preds_bp={preds_bp.shape}, targets_bp={targets_bp.shape}, last_input={np.array(last_input).shape}")
                    num_bolus_cols = len(config.get("bolus_cols", []))
                    # Use raw last_input_bolus array instead of DataFrame column (which converts to lists)
                    bolus_data_raw = np.array([to_numpy(x) for x in last_input_bolus if x is not None]) if last_input_bolus is not None else None
                    bolus_metrics = compute_bolus_response_metrics(preds_bp, targets_bp, last_input, bolus_data_raw, num_bolus_cols)
                    if bolus_metrics:  # Only log if metrics were computed
                        wandb.log(bolus_metrics)
                else:
                    logger.warning("⚠️ No mean BP column found for bolus response metrics")
        except Exception as e:
            logger.warning(f"⚠️ Failed to compute or log bolus response metrics: {e}")


    return df_summary