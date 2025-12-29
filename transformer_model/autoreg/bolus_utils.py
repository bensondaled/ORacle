import torch
from typing import Dict, List

# Define bolus groupings
BOLUS_GROUPS = {
    "pressors": [
        "meds_ephedrine_bolus",
        "meds_norepinephrine_bolus",
        "meds_phenylephrine_bolus",
        "meds_vasopressin_bolus",
        "meds_epinephrine_bolus"
    ],
    "sedatives": [
        "meds_propofol_bolus",
        "meds_etomidate_bolus",
        "meds_ketamine_bolus"
    ],
    "opioids": [
        "meds_fentanyl_bolus",
        "meds_hydromorphone_bolus",
        "meds_remifentanil_bolus"
    ],
    "anticholinergics": [
        "meds_glycopyrrolate_bolus"
    ],
    "beta_blockers": [
        "meds_esmolol_bolus",
        "meds_labetalol_bolus"
    ],
    "others": [
        "meds_dexmedetomidine_bolus"
    ]
}

def build_group_bolus_masks(bolus_tensor: torch.Tensor, bolus_cols: List[str]) -> Dict[str, torch.Tensor]:
    """
    Returns a dict of boolean masks [B, T] per group indicating bolus events.
    
    Args:
        bolus_tensor: [B, T, num_bolus_types]
        bolus_cols: List of column names corresponding to bolus_tensor last dim
    
    Returns:
        Dict[group_name → BoolTensor of shape [B, T]]
    """
    B, T, _ = bolus_tensor.shape
    group_masks = {}
    col_idx = {name: i for i, name in enumerate(bolus_cols)}

    for group, meds in BOLUS_GROUPS.items():
        mask = torch.zeros((B, T), dtype=torch.bool, device=bolus_tensor.device)
        for med in meds:
            if med in col_idx:
                mask |= bolus_tensor[:, :, col_idx[med]] > 0
        group_masks[group] = mask
    return group_masks
