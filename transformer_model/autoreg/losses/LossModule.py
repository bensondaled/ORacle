import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

def quantile_loss(preds, targets, quantile=0.9):
    diff = targets - preds
    return torch.mean(torch.max((quantile - 1) * diff, quantile * diff))

class LossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_first = nn.Parameter(torch.tensor(1.0))
        self.w_bolus = nn.Parameter(torch.tensor(1.0))
        self.w_hypo = nn.Parameter(torch.tensor(1.0))
        self.w_trend = nn.Parameter(torch.tensor(1.0))

    def first_pred_loss(self, preds, targets, last_input):
        pred_first = preds[:, 0].squeeze(-1)
        target_first = targets[:, 0].squeeze(-1)
        last_input = last_input.squeeze(-1)
        mse_to_target = F.mse_loss(pred_first, target_first)
        mse_to_last = F.mse_loss(pred_first, last_input)
        penalty = torch.relu(mse_to_target - mse_to_last)
        return mse_to_target + 0.1 * penalty

    def bolus_dynamics_loss(self, preds, targets, bolus_trigger_mask, steps_after=5):
        if bolus_trigger_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)
        selected_preds = preds[bolus_trigger_mask, :steps_after]
        selected_targets = targets[bolus_trigger_mask, :steps_after]
        weights = torch.linspace(1.0, 0.5, steps_after, device=preds.device)
        mse_loss = F.mse_loss(selected_preds * weights, selected_targets * weights)
        pred_diff = torch.diff(selected_preds, dim=1)
        target_diff = torch.diff(selected_targets, dim=1)
        dynamics_loss = F.mse_loss(pred_diff, target_diff)
        return 0.7 * mse_loss + 0.3 * dynamics_loss

    def hypo_aware_loss(self, preds, targets, onset_labels, last5_bp_values=None, onset_logits=None):
        if onset_labels is None or onset_logits is None:
            return torch.tensor(0.0, device=preds.device)
        q_loss = quantile_loss(preds, targets, quantile=0.9)
        bce = F.binary_cross_entropy_with_logits(onset_logits.squeeze(-1), onset_labels.float())
        return 0.7 * q_loss + 0.3 * bce

    def trend_loss(self, preds, targets, alpha=0.8, directional_weight=0.3):
        """
        Enhanced trend loss combining shape MSE, temporal difference MSE, and directional penalty.
        The directional penalty encourages correct prediction of BP change direction.
        """
        # Shape loss (overall fit)
        shape_loss = F.mse_loss(preds, targets)
        
        # Temporal difference loss (dynamics)
        pred_diff = torch.diff(preds, dim=1)
        target_diff = torch.diff(targets, dim=1)
        temporal_loss = F.mse_loss(pred_diff, target_diff)
        
        # Directional penalty (penalize incorrect direction of change)
        pred_sign = torch.sign(pred_diff)
        target_sign = torch.sign(target_diff)
        directional_loss = torch.mean((pred_sign != target_sign).float())
        
        return alpha * shape_loss + (1 - alpha) * (0.7 * temporal_loss + directional_weight * directional_loss)

    def forward(self, preds, targets, last_input, bolus_trigger_mask, onset_labels=None, last5_bp_values=None, onset_logits=None):
        if torch.isnan(preds).any() or torch.isnan(targets).any():
            logger.warning("NaN detected in predictions or targets.")
            return torch.tensor(0.0, device=preds.device), {}

        weights = torch.softmax(torch.stack([self.w_first, self.w_bolus, self.w_hypo, self.w_trend]), dim=0)

        l_first = self.first_pred_loss(preds, targets, last_input)
        l_bolus = self.bolus_dynamics_loss(preds, targets, bolus_trigger_mask)
        l_hypo = self.hypo_aware_loss(preds, targets, onset_labels, last5_bp_values, onset_logits)
        l_trend = self.trend_loss(preds, targets)

        total_loss = (weights[0] * l_first +
                      weights[1] * l_bolus +
                      weights[2] * l_hypo +
                      weights[3] * l_trend)

        metrics = {
            'first_pred': l_first.item(),
            'bolus_dynamics': l_bolus.item(),
            'hypo_aware': l_hypo.item(),
            'trend': l_trend.item(),
            'w_first': weights[0].item(),
            'w_bolus': weights[1].item(),
            'w_hypo': weights[2].item(),
            'w_trend': weights[3].item()
        }
        return total_loss, metrics
