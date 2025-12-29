import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

SUPPORTED_LOSS_KEYS_DEFAULT = [
    'mse', 'change', 'bolus', 'flatness', 'quantile', 'smoothness',
    'contrastive_bolus', 'dilate', 'shape', 'temporal',
    'hypo_onset_fused', 'hypo_onset_bp', 'corr', 'weighted_mse'
]
def apply_hypotension_onset_rule(preds: torch.Tensor, threshold: float = 0.43) -> torch.Tensor:
    """
    Apply the 3-normal + 2-hypo onset rule to predicted BP.
    Returns binary onset predictions of shape [B]
    """
    # Handle both 2D [B, T] and 3D [B, T, N] tensors
    if preds.dim() == 3:
        # Extract mean arterial pressure (MAP) - typically index 2 for phys_bp_mean_non_invasive
        # If MAP is not available, use the first BP-related column
        map_idx = 2 if preds.shape[-1] > 2 else 0
        preds = preds[:, :, map_idx]  # Extract MAP: [B, T, N] -> [B, T]
    
    B, T = preds.shape
    norm = preds >= threshold
    hypo = preds < threshold

    onset = torch.zeros(B, dtype=torch.bool, device=preds.device)
    for t in range(3, T - 1):
        prev3 = norm[:, t-3:t].all(dim=1)
        next2 = hypo[:, t:t+2].all(dim=1)
        onset |= (prev3 & next2)

    return onset

try:
    import yaml
    with open('configs/token_dilate.yaml', 'r') as f:
        config = yaml.safe_load(f)
    SUPPORTED_LOSS_KEYS = list(config['loss_schedule'].keys()) + ['shape', 'temporal']
except Exception:
    SUPPORTED_LOSS_KEYS = SUPPORTED_LOSS_KEYS_DEFAULT

def mse_loss(preds, targets, sample_weights=None):
    loss = (preds - targets) ** 2
    if sample_weights is not None:
        loss = loss * sample_weights.view(-1, 1, 1)
    return loss.mean()

def weighted_mse_loss(preds, targets, weight_factor=10.0):
    """
    Applies a higher weight to larger errors and periods of greater change.
    """
    mse = (preds - targets) ** 2
    
    # Weight based on the magnitude of the ground truth change
    target_diff = torch.diff(targets, dim=1, prepend=targets[:, :1])
    change_weights = torch.abs(target_diff)
    change_weights = 1.0 + weight_factor * change_weights
    
    weighted_loss = mse * change_weights
    return weighted_loss.mean()

def change_loss(preds, targets):
    pred_diff = torch.diff(preds, dim=1)
    actual_diff = torch.diff(targets, dim=1)
    return F.mse_loss(pred_diff, actual_diff)

def bolus_loss(preds: torch.Tensor, targets: torch.Tensor, bolus_trigger_mask: torch.Tensor, steps_after: int = 3) -> torch.Tensor:
    if preds.shape[0] != bolus_trigger_mask.shape[0]:
        raise ValueError(f"Mismatch: preds {preds.shape} vs bolus_trigger_mask {bolus_trigger_mask.shape}")
    selected_preds = preds[bolus_trigger_mask, :steps_after]
    selected_targets = targets[bolus_trigger_mask, :steps_after]
    if selected_preds.numel() == 0:
        return torch.tensor(0.0, device=preds.device)
    return F.mse_loss(selected_preds, selected_targets)

def flatness_penalty(preds: torch.Tensor, bolus_trigger_mask: torch.Tensor) -> torch.Tensor:
    if preds.shape[0] != bolus_trigger_mask.shape[0]:
        raise ValueError(f"Mismatch: preds {preds.shape} vs bolus_trigger_mask {bolus_trigger_mask.shape}")
    selected_preds = preds[bolus_trigger_mask]
    if selected_preds.numel() == 0 or selected_preds.shape[1] < 3:
        return torch.tensor(0.0, device=preds.device)
    diffs = torch.diff(selected_preds, n=2, dim=1)
    return torch.mean(diffs.pow(2))

def quantile_loss(preds, targets, config=None, quantile=0.9):
    quantile = config.get('quantile_level', quantile) if config else quantile
    diff = targets - preds
    return torch.mean(torch.max((quantile - 1) * diff, quantile * diff))

def smoothness_penalty(preds):
    if preds.shape[1] < 3:
        return torch.tensor(0.0, device=preds.device)
    accel = torch.diff(preds, n=2, dim=1)
    return torch.mean(accel**2)

def contrastive_bolus_loss(preds: torch.Tensor, bolus_trigger_mask: torch.Tensor) -> torch.Tensor:
    if preds.shape[0] != bolus_trigger_mask.shape[0]:
        raise ValueError(f"Mismatch: preds {preds.shape} vs bolus_trigger_mask {bolus_trigger_mask.shape}")
    bolus_preds = preds[bolus_trigger_mask]
    non_bolus_preds = preds[~bolus_trigger_mask]
    if bolus_preds.numel() == 0 or non_bolus_preds.numel() == 0:
        return torch.tensor(0.0, device=preds.device)
    bolus_mean = bolus_preds.mean(dim=1)
    non_bolus_mean = non_bolus_preds.mean(dim=1)
    return F.mse_loss(bolus_mean.mean(), non_bolus_mean.mean())

# COMMENTED OUT: These functions were causing O(B×N×T²) computational explosion
# def soft_dtw_loss(pred, target, gamma=0.01):
#     """EXPENSIVE - Causes training to hang due to nested loops"""
#     pass
# 
# def compute_dtw_path_loss(pred, target, gamma=0.01):
#     """EXPENSIVE - Causes training to hang due to complex DTW computation"""
#     pass


def dilate_loss(preds, targets, alpha=0.5, gamma=0.01):
    """
    FAST DILATE approximation - captures shape and temporal alignment efficiently.
    Args:
        preds: [B, T, N] predictions
        targets: [B, T, N] targets
        alpha: weight for shape vs temporal loss
        gamma: smoothing parameter (unused in fast version)
    """
    # SHAPE LOSS: Use simple L2 distance to capture sequence similarity
    # This approximates the DTW shape matching without expensive DP
    shape_loss = F.mse_loss(preds, targets)
    
    # TEMPORAL LOSS: Penalize temporal misalignment using difference of differences
    # This captures the temporal structure without expensive DTW path computation
    pred_diffs = torch.diff(preds, dim=1)  # [B, T-1, N]
    target_diffs = torch.diff(targets, dim=1)  # [B, T-1, N]
    temporal_loss = F.mse_loss(pred_diffs, target_diffs)
    
    dilate_total = alpha * shape_loss + (1 - alpha) * temporal_loss
    return dilate_total, shape_loss, temporal_loss

def corr_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Pearson-r loss (1 - r) computed over the *time* axis for each sample,
    then averaged over the batch.  Assumes preds/targets shape (B, T).
    """
    # centre along time axis
    preds_c   = preds - preds.mean(dim=1, keepdim=True)
    targets_c = targets - targets.mean(dim=1, keepdim=True)

    cov = (preds_c * targets_c).mean(dim=1)                    # (B,)
    std_pred = preds_c.square().mean(dim=1).sqrt().clamp_min(eps)
    std_targ = targets_c.square().mean(dim=1).sqrt().clamp_min(eps)

    r = cov / (std_pred * std_targ)                            # (B,)
    return (1.0 - r).mean()                                    # scalar


def compute_bolus_conditional_loss(preds, targets, loss_weights, config=None,
                                  bolus_trigger_mask: Optional[torch.Tensor] = None,
                                  last_input_bolus: Optional[torch.Tensor] = None,
                                  return_individual_losses: bool = False):
    """
    SIMPLIFIED drug-specific bolus-conditional weighting using last_input_bolus directly.
    
    Args:
        preds: [B, T, N] or [B, T] predictions
        targets: [B, T, N] or [B, T] targets
        loss_weights: Dict of base loss weights
        config: Configuration dict
        bolus_trigger_mask: [B] boolean mask indicating bolus at last input
        last_input_bolus: [B, num_drugs] tensor with bolus amounts for each drug
        return_individual_losses: Whether to return individual loss tensors
    """
    
    if preds.dim() == 3 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    if targets.dim() == 3 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)

    if torch.isnan(preds).any() or torch.isnan(targets).any():
        logger.warning("NaN detected in preds or targets")
        empty_metrics = {'mse': 0.0, 'dilate': 0.0, 'shape': 0.0, 'temporal': 0.0}
        if return_individual_losses:
            return torch.tensor(0.0, device=preds.device), empty_metrics, {}
        return torch.tensor(0.0, device=preds.device), empty_metrics

    # SIMPLIFIED: Get drug weights and compute multipliers from last_input_bolus
    bolus_column_weights = config.get('bolus_column_weights', {})
    bolus_cols = config.get('bolus_cols', [])
    base_multipliers = config.get('bolus_loss_multipliers', {'mse': 2.0, 'dilate': 3.0})
    
    # Compute per-sample drug-specific multipliers
    sample_multipliers = None
    has_bolus = bolus_trigger_mask is not None and bolus_trigger_mask.any()
    
    if has_bolus and last_input_bolus is not None and bolus_column_weights:
        # Create multipliers for each sample based on their drug administration
        sample_multipliers = {'mse': torch.ones(len(bolus_trigger_mask), device=preds.device),
                             'dilate': torch.ones(len(bolus_trigger_mask), device=preds.device)}
        
        # For each drug column, apply its weight if the sample received that drug
        for i, drug_name in enumerate(bolus_cols):
            if i < last_input_bolus.shape[1] and drug_name in bolus_column_weights:
                drug_weight = bolus_column_weights[drug_name]
                drug_amounts = last_input_bolus[:, i]  # [B] tensor
                drug_mask = drug_amounts > 0
                
                # Apply drug-specific multiplier where this drug was given
                for loss_type in ['mse', 'dilate']:
                    base_mult = base_multipliers.get(loss_type, 1.0)
                    drug_multiplier = 1.0 + (base_mult - 1.0) * drug_weight
                    sample_multipliers[loss_type][drug_mask] *= drug_multiplier
    
    individual_losses = {}
    metrics = {}
    total_loss = torch.tensor(0.0, device=preds.device)
    
    # === MSE LOSS ===
    if loss_weights.get("mse", 0) > 0:
        base_mse_weight = loss_weights["mse"]
        
        if has_bolus and sample_multipliers is not None:
            # SIMPLIFIED: Apply averaged drug-specific weight to batch MSE
            sample_weights = sample_multipliers['mse']  # [B] tensor
            avg_weight_multiplier = sample_weights.mean().item()
            
            # Compute MSE for entire batch (more memory efficient)
            mse = mse_loss(preds, targets)
            mse_contribution = base_mse_weight * avg_weight_multiplier * mse
            
            total_loss += mse_contribution
            metrics['mse'] = mse.item()
            metrics['mse_effective_weight'] = base_mse_weight * avg_weight_multiplier
            
            # Track drug-specific statistics
            bolus_indices = bolus_trigger_mask
            if bolus_indices.any():
                avg_bolus_weight = sample_weights[bolus_indices].mean().item()
                metrics['mse_avg_bolus_weight'] = avg_bolus_weight
            
            if return_individual_losses:
                individual_losses['mse'] = mse_contribution
                
        elif has_bolus:
            # Fallback to simple bolus/non-bolus weighting
            mse = mse_loss(preds, targets)
            bolus_multiplier = base_multipliers.get('mse', 1.0)
            
            bolus_indices = bolus_trigger_mask
            non_bolus_indices = ~bolus_trigger_mask
            
            if bolus_indices.any():
                bolus_mse = mse_loss(preds[bolus_indices], targets[bolus_indices])
                bolus_contribution = base_mse_weight * bolus_multiplier * bolus_mse
                total_loss += bolus_contribution
                
                if return_individual_losses:
                    individual_losses['mse_bolus'] = bolus_contribution
            
            if non_bolus_indices.any():
                non_bolus_mse = mse_loss(preds[non_bolus_indices], targets[non_bolus_indices])
                non_bolus_contribution = base_mse_weight * non_bolus_mse
                total_loss += non_bolus_contribution
                
                if return_individual_losses:
                    individual_losses['mse_non_bolus'] = non_bolus_contribution
            
            bolus_ratio = bolus_indices.sum().float() / len(bolus_indices)
            effective_weight = base_mse_weight * (1 + (bolus_multiplier - 1) * bolus_ratio)
            metrics['mse'] = mse.item()
            metrics['mse_effective_weight'] = effective_weight.item()
        else:
            # No bolus cases, use standard weighting
            mse = mse_loss(preds, targets)
            mse_contribution = base_mse_weight * mse
            total_loss += mse_contribution
            metrics['mse'] = mse.item()
            
            if return_individual_losses:
                individual_losses['mse'] = mse_contribution
    
    # === DILATE LOSS ===
    if loss_weights.get("dilate", 0) > 0:
        # Ensure proper shape for DILATE
        if preds.dim() == 2:  # [B, T] -> [B, T, 1]
            dilate_preds = preds.unsqueeze(-1)
            dilate_targets = targets.unsqueeze(-1)
        else:  # Already [B, T, N]
            dilate_preds = preds
            dilate_targets = targets
            
        base_dilate_weight = loss_weights["dilate"]
        
        if has_bolus and sample_multipliers is not None:
            # MEMORY-EFFICIENT DRUG-SPECIFIC DILATE: Process in chunks
            sample_weights = sample_multipliers['dilate']  # [B] tensor
            batch_size = dilate_preds.shape[0]
            chunk_size = min(32, batch_size)  # Process max 32 samples at once
            
            total_dilate_loss = 0.0
            total_shape_loss = 0.0
            total_temporal_loss = 0.0
            total_weighted_contrib = 0.0
            
            # Process samples in chunks to control memory usage
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_preds = dilate_preds[i:end_idx]
                chunk_targets = dilate_targets[i:end_idx]
                chunk_weights = sample_weights[i:end_idx]
                
                # Compute DILATE for this chunk
                chunk_dilate, chunk_shape, chunk_temporal = dilate_loss(chunk_preds, chunk_targets)
                
                # Weight by drug-specific multipliers
                weighted_chunk_contrib = base_dilate_weight * chunk_weights.mean() * chunk_dilate
                
                # Accumulate
                total_weighted_contrib += weighted_chunk_contrib
                total_dilate_loss += chunk_dilate.item() * len(chunk_weights)
                total_shape_loss += chunk_shape.item() * len(chunk_weights)
                total_temporal_loss += chunk_temporal.item() * len(chunk_weights)
                
                # Clean up chunk tensors immediately
                del chunk_preds, chunk_targets, chunk_weights, chunk_dilate, chunk_shape, chunk_temporal, weighted_chunk_contrib
            
            # Average the accumulated losses
            avg_dilate = total_dilate_loss / batch_size
            avg_shape = total_shape_loss / batch_size  
            avg_temporal = total_temporal_loss / batch_size
            
            total_loss += total_weighted_contrib
            metrics['dilate'] = avg_dilate
            metrics['dilate_effective_weight'] = (base_dilate_weight * sample_weights).mean().item()
            metrics['shape'] = avg_shape
            metrics['temporal'] = avg_temporal
            
            # Track drug-specific statistics
            bolus_indices = bolus_trigger_mask
            if bolus_indices.any():
                avg_bolus_weight = sample_weights[bolus_indices].mean().item()
                metrics['dilate_avg_bolus_weight'] = avg_bolus_weight
            
            if return_individual_losses:
                individual_losses['dilate'] = total_weighted_contrib
                
        elif has_bolus:
            # Fallback to simple bolus/non-bolus weighting
            dilate_total, shape_loss, temporal_loss = dilate_loss(dilate_preds, dilate_targets)
            bolus_multiplier = base_multipliers.get('dilate', 1.0)
            bolus_indices = bolus_trigger_mask
            non_bolus_indices = ~bolus_trigger_mask
            
            # Compute separate DILATE losses
            if bolus_indices.any():
                bolus_dilate, bolus_shape, bolus_temporal = dilate_loss(
                    dilate_preds[bolus_indices], dilate_targets[bolus_indices]
                )
                bolus_contribution = base_dilate_weight * bolus_multiplier * bolus_dilate
                total_loss += bolus_contribution
                
                if return_individual_losses:
                    individual_losses['dilate_bolus'] = bolus_contribution
            
            if non_bolus_indices.any():
                non_bolus_dilate, non_bolus_shape, non_bolus_temporal = dilate_loss(
                    dilate_preds[non_bolus_indices], dilate_targets[non_bolus_indices]
                )
                non_bolus_contribution = base_dilate_weight * non_bolus_dilate
                total_loss += non_bolus_contribution
                
                if return_individual_losses:
                    individual_losses['dilate_non_bolus'] = non_bolus_contribution
            
            # Logging metrics
            bolus_ratio = bolus_indices.sum().float() / len(bolus_indices)
            effective_weight = base_dilate_weight * (1 + (bolus_multiplier - 1) * bolus_ratio)
            metrics['dilate'] = dilate_total.item()
            metrics['dilate_effective_weight'] = effective_weight.item()
            metrics['shape'] = shape_loss.item()
            metrics['temporal'] = temporal_loss.item()
        else:
            # No bolus cases
            dilate_total, shape_loss, temporal_loss = dilate_loss(dilate_preds, dilate_targets)
            dilate_contribution = base_dilate_weight * dilate_total
            total_loss += dilate_contribution
            metrics['dilate'] = dilate_total.item()
            metrics['shape'] = shape_loss.item()
            metrics['temporal'] = temporal_loss.item()
            
            if return_individual_losses:
                individual_losses['dilate'] = dilate_contribution
    
    # Add bolus statistics to metrics
    if bolus_trigger_mask is not None:
        bolus_count = bolus_trigger_mask.sum().item()
        total_count = len(bolus_trigger_mask)
        metrics['bolus_ratio'] = bolus_count / total_count
        metrics['bolus_count'] = bolus_count
        
        # SIMPLIFIED: Add drug statistics using last_input_bolus
        if last_input_bolus is not None and bolus_column_weights:
            drug_stats = {}
            max_weight_drug = None
            max_weight = 0
            
            for i, drug_name in enumerate(bolus_cols):
                if i < last_input_bolus.shape[1] and drug_name in bolus_column_weights:
                    drug_amounts = last_input_bolus[:, i]
                    drug_mask = drug_amounts > 0
                    drug_count = drug_mask.sum().item()
                    
                    if drug_count > 0:
                        drug_weight = bolus_column_weights[drug_name]
                        drug_stats[f'{drug_name}_count'] = drug_count
                        drug_stats[f'{drug_name}_ratio'] = drug_count / total_count
                        
                        # Track most impactful drug
                        if drug_weight > max_weight:
                            max_weight = drug_weight
                            max_weight_drug = drug_name
            
            metrics.update(drug_stats)
            
            if max_weight_drug:
                metrics['dominant_drug'] = max_weight_drug
                metrics['dominant_drug_weight'] = max_weight
    
    if return_individual_losses:
        return total_loss, metrics, individual_losses
    return total_loss, metrics


def compute_loss(preds, targets, loss_weights, bolus_mask=None, config=None,
                 hypo_fused_logits=None, hypo_bp_logits=None, onset_labels=None,
                 onset_types: Optional[List[str]] = None,
                 group_mask: Dict[str, torch.Tensor] = None,
                 bolus_trigger_mask: Optional[torch.Tensor] = None,
                 return_individual_losses: bool = False):

    if preds.dim() == 3 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    if targets.dim() == 3 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)

    if torch.isnan(preds).any() or torch.isnan(targets).any():
        logger.warning("NaN detected in preds or targets")
        return torch.tensor(0.0, device=preds.device), {k: 0.0 for k in SUPPORTED_LOSS_KEYS}

    metrics = {k: 0.0 for k in SUPPORTED_LOSS_KEYS}
    total_loss = torch.tensor(0.0, device=preds.device)
    sample_weights = None
    
    if config.get("use_bolus_weights", False) and group_mask is not None:
        sample_weights = torch.zeros(preds.size(0), device=preds.device)
        for drug, mask in group_mask.items():
            if drug in config.get("bolus_column_weights", {}):
                w = config["bolus_column_weights"][drug]
                sample_weights[mask] += w
        sample_weights = sample_weights + 1.0  # avoid zero
        mse = F.mse_loss(preds, targets, reduction='none')
        mse = (mse.mean(dim=1) * sample_weights.view(-1, 1)).mean()


    mse = mse_loss(preds, targets, sample_weights)
    total_loss += loss_weights.get("mse", 1.0) * mse
    metrics['mse'] = mse.item()

    if loss_weights.get("change", 0) > 0:
        chg_loss = change_loss(preds, targets)
        total_loss += loss_weights["change"] * chg_loss
        metrics['change'] = chg_loss.item()

    if loss_weights.get("flatness", 0) > 0 and bolus_trigger_mask is not None:
        flat_loss = flatness_penalty(preds, bolus_trigger_mask)
        total_loss += loss_weights["flatness"] * flat_loss
        metrics["flatness"] = flat_loss.item()

    if loss_weights.get("bolus", 0) > 0 and bolus_trigger_mask is not None:
        bolus_loss_val = bolus_loss(preds, targets, bolus_trigger_mask)
        total_loss += loss_weights["bolus"] * bolus_loss_val
        metrics["bolus"] = bolus_loss_val.item()

    if loss_weights.get("quantile", 0) > 0:
        quant_loss = quantile_loss(preds, targets, config)
        total_loss += loss_weights["quantile"] * quant_loss
        metrics['quantile'] = quant_loss.item()

    if loss_weights.get("smoothness", 0) > 0:
        smooth_loss = smoothness_penalty(preds)
        total_loss += loss_weights["smoothness"] * smooth_loss
        metrics['smoothness'] = smooth_loss.item()

    if loss_weights.get("contrastive_bolus", 0) > 0 and bolus_trigger_mask is not None:
        contrast_loss = contrastive_bolus_loss(preds, bolus_trigger_mask)
        total_loss += loss_weights["contrastive_bolus"] * contrast_loss
        metrics["contrastive_bolus"] = contrast_loss.item()

    if loss_weights.get("dilate", 0) > 0:
        dilate_total, shape_loss, temporal_loss = dilate_loss(preds, targets)
        total_loss += loss_weights["dilate"] * dilate_total
        metrics['dilate'] = dilate_total.item()
        metrics['shape'] = shape_loss.item()
        metrics['temporal'] = temporal_loss.item()

    # 2) Inside compute_loss() just after mse
    if loss_weights.get("corr", 0) > 0:
        c_loss = corr_loss(preds, targets)
        total_loss += loss_weights["corr"] * c_loss
        metrics["corr"] = c_loss.item()

    if loss_weights.get("weighted_mse", 0) > 0:
        w_mse_loss = weighted_mse_loss(preds, targets)
        total_loss += loss_weights["weighted_mse"] * w_mse_loss
        metrics["weighted_mse"] = w_mse_loss.item()


    def filter_onset(logits, labels):
        if onset_types is None:
            return logits, labels
        # FIX: Include both "true_onset" AND "none", exclude "ongoing"
        mask = torch.tensor([t in ["true_onset", "none"] for t in onset_types], device=preds.device, dtype=torch.bool)
        return logits[mask], labels[mask]
    
    if loss_weights.get("hypo_onset_fused", 0) > 0 and hypo_fused_logits is not None and onset_labels is not None:
        fused_logits, fused_labels = filter_onset(hypo_fused_logits.squeeze(-1), onset_labels)
        
        # Debug logging to verify masking
        if onset_types is not None:
            total_samples = len(onset_types)
            n_true_onset = sum(1 for t in onset_types if t == "true_onset")
            n_none = sum(1 for t in onset_types if t == "none")
            n_ongoing = sum(1 for t in onset_types if t == "ongoing")
            logger.debug(f"[Hypo Loss] Total: {total_samples}, true_onset: {n_true_onset}, none: {n_none}, ongoing: {n_ongoing}")
            logger.debug(f"[Hypo Loss] After filter: {fused_logits.numel()} samples, {fused_labels.sum().item():.0f} positive")

        if fused_logits.numel() > 0:
            # Clamp logits to prevent NaNs from extreme values
            fused_logits = torch.clamp(fused_logits, min=-15.0, max=15.0)

            if config.get("use_focal_loss", False):
                alpha = config.get("hypo_focal_alpha", 0.25)
                gamma = config.get("hypo_focal_gamma", 2.0)
                prob = torch.sigmoid(fused_logits)
                ce_loss = F.binary_cross_entropy_with_logits(fused_logits, fused_labels.float(), reduction='none')
                p_t = prob * fused_labels + (1 - prob) * (1 - fused_labels)
                focal = alpha * ce_loss * ((1 - p_t) ** gamma)
                fused_bce = focal.mean()
            else:
                pos_weight = torch.tensor(config.get("hypo_pos_weight", 1.0), device=preds.device)
                fused_bce = F.binary_cross_entropy_with_logits(fused_logits, fused_labels.float(), pos_weight=pos_weight)

            total_loss += loss_weights["hypo_onset_fused"] * fused_bce
            metrics["hypo_onset_fused"] = fused_bce.item()


    if loss_weights.get("hypo_onset_bp", 0) > 0 and hypo_bp_logits is not None and onset_labels is not None:
        bp_logits, bp_labels = filter_onset(hypo_bp_logits.squeeze(-1), onset_labels)
        if bp_logits.numel() > 0:
            # Clamp logits to prevent NaNs from extreme values
            bp_logits = torch.clamp(bp_logits, min=-15.0, max=15.0)
            if config.get("use_focal_loss", False):
                alpha = config.get("hypo_focal_alpha", 0.25)
                gamma = config.get("hypo_focal_gamma", 2.0)
                prob = torch.sigmoid(bp_logits)
                ce_loss = F.binary_cross_entropy_with_logits(bp_logits, bp_labels.float(), reduction='none')
                p_t = prob * bp_labels + (1 - prob) * (1 - bp_labels)
                focal = alpha * ce_loss * ((1 - p_t) ** gamma)
                bp_bce = focal.mean()
            else:
                bp_bce = F.binary_cross_entropy_with_logits(bp_logits, bp_labels.float())
            total_loss += loss_weights["hypo_onset_bp"] * bp_bce
            metrics["hypo_onset_bp"] = bp_bce.item()

    if loss_weights.get("hypo", 0) > 0 and onset_labels is not None:
        predicted_onsets = apply_hypotension_onset_rule(preds, config.get("hypo_threshold", 0.43)).float()
        true_onsets = onset_labels.float()

        if config.get("use_focal_loss", False):
            alpha = config.get("hypo_focal_alpha", 0.25)
            gamma = config.get("hypo_focal_gamma", 2.0)
            ce = F.binary_cross_entropy(predicted_onsets, true_onsets, reduction='none')
            pt = predicted_onsets * true_onsets + (1 - predicted_onsets) * (1 - true_onsets)
            loss = alpha * ce * ((1 - pt) ** gamma)
        else:
            pos_weight = torch.tensor(config.get("hypo_pos_weight", 1.0), device=preds.device)
            loss = F.binary_cross_entropy(predicted_onsets, true_onsets, weight=pos_weight if true_onsets.sum() > 0 else None)

        total_loss += loss_weights["hypo"] * loss.mean()
        metrics["hypo"] = loss.mean().item()

    # Return individual loss tensors for gradient monitoring if requested
    if return_individual_losses:
        individual_losses = {}
        for key in SUPPORTED_LOSS_KEYS:
            if key in metrics and loss_weights.get(key, 0) > 0:
                # Create individual loss tensors (scaled by weight)
                if key == 'mse':
                    individual_losses[key] = loss_weights[key] * mse
                elif key == 'dilate' and loss_weights.get("dilate", 0) > 0:
                    dilate_total, _, _ = dilate_loss(preds, targets)
                    individual_losses[key] = loss_weights[key] * dilate_total
                # Add other losses as needed...
        return total_loss, metrics, individual_losses
    
    return total_loss, metrics
