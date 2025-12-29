import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class MultiComponentLoss(nn.Module):
    def __init__(
        self,
        loss_keys: List[str] = ['first', 'change', 'bolus', 'hypo', 'trend'],
        init_weight: float = 1.0,
        min_weight: float = 1e-3,
        quantile_init: float = 0.5,
        steps_after: int = 5,
        alpha_trend: float = 0.8,
        hypo_pos_weight: float = 10.0,
        first_penalty_weight: float = 0.5,
        first_continuity_weight: float = 0.2,
        trend_diff_weights: Tuple[float, float] = (0.7, 0.3),
        hypo_loss_weights: Tuple[float, float] = (0.6, 0.4),
        hypo_focal_alpha: float = 0.25,
        hypo_focal_gamma: float = 2.0,
        use_focal_loss: bool = False,
        bolus_cols: Optional[List[str]] = None,
        bolus_column_weights: Optional[Dict[str, float]] = None,
        use_bolus_weights: bool = False,
        use_quantile_in_hypo: bool = True,
        softmax_temperature: float = 0.5
    ):
        super().__init__()
        self.loss_keys = loss_keys
        self.min_weight = min_weight
        self.steps_after = steps_after
        self.alpha_trend = alpha_trend
        self.hypo_pos_weight = hypo_pos_weight
        self.first_penalty_weight = first_penalty_weight
        self.first_continuity_weight = first_continuity_weight
        self.trend_diff_weights = trend_diff_weights
        self.hypo_loss_weights = hypo_loss_weights
        self.hypo_focal_alpha = hypo_focal_alpha
        self.hypo_focal_gamma = hypo_focal_gamma
        self.use_focal_loss = use_focal_loss
        self.use_quantile_in_hypo = use_quantile_in_hypo
        self.softmax_temperature = softmax_temperature

        self.register_buffer("quantile_reg_weight", torch.tensor(0.01))

        self.log_weights = nn.ParameterDict({
            key: nn.Parameter(torch.log(torch.tensor(init_weight)))
            for key in loss_keys
        })
        self.logit_quantile = nn.Parameter(torch.logit(torch.tensor(quantile_init)))

        self.bolus_cols = bolus_cols or []
        self.use_bolus_weights = use_bolus_weights
        if use_bolus_weights and bolus_column_weights:
            total = float(sum(bolus_column_weights.get(c, 0.0) for c in self.bolus_cols)) or 1.0
            self.bolus_column_weights = {
                col: torch.tensor(bolus_column_weights.get(col, 0.0), dtype=torch.float32) / total
                for col in self.bolus_cols
            }
        else:
            n = len(self.bolus_cols) or 1
            uniform = 1.0 / n
            self.bolus_column_weights = {
                col: torch.tensor(uniform, dtype=torch.float32)
                for col in self.bolus_cols
            }

        self.running_avg = {k: torch.tensor(1.0) for k in self.loss_keys}

    def _normalize_loss(self, k, loss_val):
        self.running_avg[k] = 0.99 * self.running_avg[k] + 0.01 * loss_val.detach()
        return loss_val / (self.running_avg[k] + 1e-6)

    def _get_weights(self) -> Dict[str, torch.Tensor]:
        logits = torch.stack(list(self.log_weights.values()), dim=0)
        soft = torch.softmax(logits / self.softmax_temperature, dim=0)
        clamped = torch.clamp(soft, min=self.min_weight)
        renorm = clamped / clamped.sum()
        return { key: renorm[i] for i, key in enumerate(self.loss_keys) }

    def first_pred_loss(self, preds, targets, last_input, last_input_bolus, last5_bp_values):
        p0, t0, l0 = preds[:, 0], targets[:, 0], last_input
        mse_t = F.mse_loss(p0, t0)
        if last5_bp_values is not None and last5_bp_values.size(1) >= 5:
            x = torch.arange(5, device=preds.device).float()
            y = last5_bp_values
            x_centered = x - x.mean()
            y_centered = y - y.mean(dim=1, keepdim=True)
            input_slope = (x_centered * y_centered).sum(dim=1) / (x_centered ** 2).sum()
        else:
            input_slope = torch.zeros_like(p0)
        if targets.size(1) >= 2:
            slope_target = targets[:, 1] - targets[:, 0]
        else:
            slope_target = targets[:, 0] - last_input
        mismatch = (slope_target - input_slope).abs() > 2.0
        mismatch_ratio = mismatch.float().mean()
        penalty = F.relu(F.mse_loss(p0, l0) - F.mse_loss(t0, l0))
        continuity = F.mse_loss(p0 - l0, t0 - l0)
        return mse_t + self.first_penalty_weight * penalty * mismatch_ratio + self.first_continuity_weight * continuity * mismatch_ratio

    def change_loss(self, preds, targets):
        if preds.size(1) < 2:
            return torch.tensor(0.0, device=preds.device)
        mse = F.mse_loss(preds.diff(dim=1), targets.diff(dim=1))
        pred_c = preds - preds.mean(dim=1, keepdim=True)
        target_c = targets - targets.mean(dim=1, keepdim=True)
        corr = (pred_c * target_c).sum(dim=1) / (pred_c.norm(dim=1) * target_c.norm(dim=1) + 1e-6)
        corr_penalty = 1.0 - corr.mean()
        return 0.7 * mse + 0.3 * corr_penalty

    def bolus_dynamics_loss(self, preds, targets, last_input, last_input_bolus):
        total_loss = torch.tensor(0.0, device=preds.device)
        contributing = 0
        for idx, col in enumerate(self.bolus_cols):
            mask = last_input_bolus[:, idx] > 0
            if not mask.any():
                continue
            contributing += 1
            weight = self.bolus_column_weights[col] if self.use_bolus_weights else torch.tensor(1.0, device=preds.device)
            sp, st = preds[mask, :self.steps_after], targets[mask, :self.steps_after]
            l0 = last_input[mask].unsqueeze(1)
            delta_sp = sp - l0
            delta_st = st - l0
            decay = torch.linspace(2.0, 1.25, sp.size(1), device=preds.device)
            mse = F.mse_loss(delta_sp * decay, delta_st * decay)
            d1 = F.mse_loss(sp.diff(dim=1), st.diff(dim=1)) if sp.size(1) >= 2 else 0.0
            d2 = F.mse_loss(sp.diff(n=2, dim=1), st.diff(n=2, dim=1)) if sp.size(1) >= 3 else 0.0
            total_loss += weight * (0.6 * mse + 0.3 * d1 + 0.1 * d2)
        return total_loss / contributing if contributing > 0 else torch.tensor(0.0, device=preds.device)

    def trend_loss(self, preds, targets):
        shape = F.mse_loss(preds, targets)
        t1 = F.mse_loss(preds.diff(dim=1), targets.diff(dim=1)) if preds.size(1) >= 2 else torch.tensor(0.0, device=preds.device)
        t2 = F.mse_loss(preds.diff(n=2, dim=1), targets.diff(n=2, dim=1)) if preds.size(1) >= 3 else torch.tensor(0.0, device=preds.device)
        return self.alpha_trend * shape + (1 - self.alpha_trend) * (self.trend_diff_weights[0] * t1 + self.trend_diff_weights[1] * t2)

    def _quantile_loss(self, preds, targets):
        q = torch.sigmoid(self.logit_quantile).clamp(0.05, 0.95)
        diff = targets - preds
        return torch.mean(torch.max((q - 1) * diff, q * diff))

    @staticmethod
    def focal_loss(logits, labels, alpha, gamma, pos_weight=None):
        labels = labels.float()
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        alpha_factor = torch.where(labels == 1, alpha, 1 - alpha)
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        mod_factor = (1 - p_t).pow(gamma)
        loss = alpha_factor * mod_factor * bce
        if pos_weight is not None:
            loss = torch.where(labels == 1, loss * pos_weight, loss)
        return loss.mean()

    def hypo_aware_loss(self, preds, targets, onset_labels, onset_logits, onset_types=None):
        if onset_labels is None or onset_logits is None:
            logger.warning("⚠️ Missing onset_labels or logits!")
            return torch.tensor(0.0, device=preds.device)

        ql = self._quantile_loss(preds, targets) if self.use_quantile_in_hypo else torch.tensor(0.0, device=preds.device)
        logits = onset_logits.squeeze(-1)

        if onset_types:
            mask = torch.tensor([t == "true_onset" for t in onset_types], dtype=torch.bool, device=logits.device)
            if isinstance(onset_labels, torch.Tensor):
                onset_labels = onset_labels.to(logits.device)
            else:
                onset_labels = torch.tensor(onset_labels, dtype=torch.float32, device=logits.device)
            logits, onset_labels = logits[mask], onset_labels[mask]

        if logits.numel() == 0:
            return ql

        if torch.allclose(logits, logits[0]):
            logger.warning("⚠️ All hypotension logits are identical — possible model collapse.")

        if self.use_focal_loss:
            cls_loss = self.focal_loss(logits, onset_labels, self.hypo_focal_alpha, self.hypo_focal_gamma, pos_weight=torch.tensor(self.hypo_pos_weight, device=preds.device))
        else:
            cls_loss = F.binary_cross_entropy_with_logits(logits, onset_labels.float(), pos_weight=torch.tensor(self.hypo_pos_weight, device=preds.device))

        return self.hypo_loss_weights[0] * ql + self.hypo_loss_weights[1] * cls_loss

    def forward(self, preds, targets, last_input, bolus_trigger_mask, onset_labels=None, onset_logits=None, onset_types=None, last_input_bolus=None, last5_bp_values=None):
        if not self.loss_keys:
            logger.warning("⚠️ MultiComponentLoss called with empty loss_keys.")
        if preds is None or len(preds) == 0:
            logger.warning("⚠️ preds is empty in MultiComponentLoss.")
        if onset_logits is None and "hypo" in self.loss_keys:
            logger.warning("⚠️ 'hypo' is in loss_keys but onset_logits is None.")

        if preds.dim() == 3: preds = preds.squeeze(-1)
        if targets.dim() == 3: targets = targets.squeeze(-1)
        if last_input.dim() > 1: last_input = last_input.squeeze(-1)


        losses = {}
        if "first" in self.loss_keys:
            losses["first"] = self.first_pred_loss(preds, targets, last_input, last_input_bolus, last5_bp_values)
        if "change" in self.loss_keys:
            losses["change"] = self.change_loss(preds, targets)
        if "bolus" in self.loss_keys and last_input_bolus is not None:
            losses["bolus"] = self.bolus_dynamics_loss(preds, targets, last_input, last_input_bolus)
        if "hypo" in self.loss_keys:
            losses["hypo"] = self.hypo_aware_loss(preds, targets, onset_labels, onset_logits, onset_types)
        if "trend" in self.loss_keys:
            losses["trend"] = self.trend_loss(preds, targets)

        normalized_losses = {k: self._normalize_loss(k, v) for k, v in losses.items()}
        weights = self._get_weights()

        total = sum(normalized_losses[k] * weights[k] for k in normalized_losses)
        if "hypo" in self.loss_keys and self.use_quantile_in_hypo:
            total += self.quantile_reg_weight * (self.logit_quantile ** 2)
            
        metrics = {k: losses[k].item() for k in losses}
        for k, w in weights.items():
            metrics[f"w_{k}"] = w.item()
        for k, param in self.log_weights.items():
            metrics[f"logit_raw/{k}"] = param.item()
        if "hypo" in self.loss_keys and self.use_quantile_in_hypo:
            metrics["quantile_level"] = torch.sigmoid(self.logit_quantile).item()

        return total, metrics
