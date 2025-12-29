
# --- losses/flatness.py ---
import torch

def flatness_penalty(preds):
    diff = torch.diff(preds, dim=1)
    penalty = torch.mean(torch.abs(diff))
    return penalty



