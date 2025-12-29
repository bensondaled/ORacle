# /home/marcgh/intraop_model/src/losses/loss_schedule.py

# loss_schedule.py
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def get_scheduled_loss_weights(config: Dict, epoch: int) -> Dict[str, float]:
    """
    Compute loss weights based on the schedule and current epoch.
    Args:
        config: Configuration dictionary with 'loss_schedule'.
        epoch: Current epoch (0-based).
    Returns:
        Dictionary of loss weights.
    """
    loss_schedule = config.get('loss_schedule', {})
    weights = {}

    for loss_name, sched in loss_schedule.items():
        start_weight = sched.get('start_weight', 0.0)
        end_weight = sched.get('end_weight', 0.0)
        num_epochs = sched.get('num_epochs', 10)

        if num_epochs <= 0:
            weight = start_weight
        else:
            # Linear interpolation
            progress = min(epoch / num_epochs, 1.0)
            weight = start_weight + progress * (end_weight - start_weight)
        
        # Map "hypo" to both hypotension loss keys
        if loss_name == "hypo":
            weights["hypo_onset_fused"] = weight
            weights["hypo_onset_bp"] = weight
        else:
            weights[loss_name] = weight

    logger.debug(f"Epoch {epoch}: Computed loss weights {weights}")
    return weights