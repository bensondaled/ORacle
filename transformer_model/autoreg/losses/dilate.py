# --- losses/dilate.py ---
# Reference: https://arxiv.org/pdf/2002.03671.pdf
import torch

def dilate_loss(preds, targets, alpha=0.5, gamma=0.01):
    B, T = preds.shape
    preds = preds.view(B, T)
    targets = targets.view(B, T)

    shape_loss = torch.mean((preds - targets) ** 2)
    temporal_loss = 0
    for b in range(B):
        dist_matrix = (torch.arange(T).view(-1, 1) - torch.arange(T).view(1, -1)) ** 2
        dist_matrix = dist_matrix.float().to(preds.device)
        temporal_loss += torch.sum((preds[b] - targets[b]) ** 2 * dist_matrix)
    temporal_loss = gamma * temporal_loss / B

    loss = alpha * shape_loss + (1 - alpha) * temporal_loss
    return loss, shape_loss, temporal_loss
def quantile_loss(preds, targets, quantile=0.9):
    diff = targets - preds
    return torch.mean(torch.max((quantile - 1) * diff, quantile * diff))
def smoothness_penalty(preds):
    accel = torch.diff(preds, n=2, dim=1)
    return torch.mean(accel**2)
def contrastive_bolus_loss(preds, bolus_mask, window=5):
    if bolus_mask.any():
        pre = preds[bolus_mask, -window-1:-1]
        post = preds[bolus_mask, -window:]
        return -torch.mean(post - pre)  # maximize rise/drop
    return torch.tensor(0.0, device=preds.device)
