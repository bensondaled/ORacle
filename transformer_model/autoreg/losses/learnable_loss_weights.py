import torch
import torch.nn as nn
from typing import Dict

class LearnableLossWeights(nn.Module):
    """
    Learnable, positive-constrained loss weights for multi-loss optimization.
    """

    def __init__(self, loss_keys: list[str], init_value: float = 1.0, min_value: float = 1e-3):
        """
        Args:
            loss_keys: List of loss keys to learn weights for.
            init_value: Initial value for all loss weights.
            min_value: Minimum clamp value to avoid zero weights.
        """
        super().__init__()
        self.loss_keys = loss_keys
        self.min_value = min_value

        # Parameters are stored in log-space to ensure positivity when exponentiated
        self.logits = nn.ParameterDict({
            key: nn.Parameter(torch.log(torch.tensor(init_value)))
            for key in loss_keys
        })

    def forward(self) -> Dict[str, torch.Tensor]:
        """Returns a dict of positive loss weights."""
        return {k: torch.clamp(torch.exp(logit), min=self.min_value) for k, logit in self.logits.items()}

    def items(self) -> Dict[str, torch.Tensor]:
        """Same as forward(), for convenience."""
        return self.forward()

    def to_dict(self) -> Dict[str, float]:
        """Returns the current weights as a regular dict (floats, not tensors)."""
        return {k: float(torch.exp(logit).clamp(min=self.min_value).item()) for k, logit in self.logits.items()}
