# /home/marcgh/intraop_model/src/losses/__init__.py
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from .custom_losses import (
    mse_loss,
    flatness_penalty,
    change_loss,
    bolus_loss,
    quantile_loss,
    smoothness_penalty,
    contrastive_bolus_loss,
    dilate_loss,
    compute_loss
)

SUPPORTED_LOSS_KEYS_DEFAULT = [
    'mse', 'change', 'bolus', 'flatness', 'quantile', 'smoothness',
    'contrastive_bolus', 'dilate', 'shape', 'temporal',
    'hypo_onset_fused', 'hypo_onset_bp'
]