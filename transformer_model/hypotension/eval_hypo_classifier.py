#/home/marcgh/intraop_model/src/eval_hypo_classifier.py
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import pandas as pd
import os

from utils import move_batch_to_device

def eval_hypo_classifier(model, dataloader, device, save_path=None):
    """
    Evaluate hypotension classifier and optionally save row-level predictions.

    Args:
        model: The hypotension classifier model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        save_path: Optional path to save predictions CSV. If None, predictions are not saved.

    Returns:
        dict: Dictionary with 'auc' and 'f1' scores
    """
    model.eval()
    all_preds, all_labels, all_types = [], [], []
    all_indices, all_case_ids = [], []

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
            valid_mask = torch.ones_like(labels, dtype=torch.bool)

            probs = torch.sigmoid(logits[valid_mask])
            all_preds.append(probs.cpu())
            all_labels.append(labels[valid_mask].cpu())
            all_types.extend(types.cpu().tolist() if torch.is_tensor(types) else types)

            # Store metadata if available
            if "original_index" in batch:
                indices = batch["original_index"]
                if torch.is_tensor(indices):
                    all_indices.extend(indices.cpu().numpy().tolist())
                else:
                    all_indices.extend(indices)
            if "mpog_case_id" in batch:
                case_ids = batch["mpog_case_id"]
                if torch.is_tensor(case_ids):
                    all_case_ids.extend(case_ids.cpu().numpy().tolist())
                else:
                    all_case_ids.extend(case_ids)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds > 0.5)

    # Save predictions if path provided
    if save_path:
        pred_data = {
            'prediction_prob': preds,
            'actual_label': labels,
            'hypo_onset_type': all_types[:len(preds)]  # Match length
        }

        # Add metadata if available
        if all_indices:
            pred_data['original_index'] = all_indices[:len(preds)]
        if all_case_ids:
            pred_data['mpog_case_id'] = all_case_ids[:len(preds)]

        pred_df = pd.DataFrame(pred_data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pred_df.to_csv(save_path, index=False)
        print(f"✅ Saved {len(pred_df):,} predictions to {save_path}")

    return {"auc": auc, "f1": f1}
