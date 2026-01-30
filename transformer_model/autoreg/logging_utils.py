# /home/marcgh/intraop_model/src/logging_utils.py
import os
import sys
import logging
from datetime import datetime
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def init_logging(config):
    level = getattr(logging, config.get("logging_level", "info").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    os.environ["BP_LOG_LEVEL"] = str(level)

def init_run(config):
    # Don't append timestamp when resuming
    if config.get('resume', False):
        run_name = config.get('run_name', 'autoreg_run')
    else:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [config.get('run_name', 'autoreg_run')]
        if config.get('use_vae'):
            parts.append('vae')
        if config.get('use_tokens'):
            parts.append('tokens')
        if config.get('decoder_type'):
            parts.append(config['decoder_type'])
        if config.get('loss_mode'):
            parts.append(config['loss_mode'])
        if config.get('data_percentage', 1.0) < 1.0:
            parts.append(f"{int(config['data_percentage']*100)}pct")
        run_name = "_".join(parts) + f"_{now}"
    config['run_name'] = run_name

    # ✅ Initialize W&B with error handling
    try:
        print(f"🔄 Initializing wandb run: {run_name}")
        print(f"🔄 wandb module: {wandb}")
        print(f"🔄 wandb.init available: {hasattr(wandb, 'init')}")

        # Get optional entity from config
        wandb_entity = config.get('wandb_entity', None)

        wandb.init(
            project=config['wandb_project'],
            entity=wandb_entity,
            name=run_name,
            config=config
        )

        # ✅ Define metrics only after wandb.init()
        wandb.define_metric("global_step")
        wandb.define_metric("final_eval_step")
        wandb.log({'initialized': 1})

        print(f"✅ Initialized run: {run_name}")
    except Exception as e:
        print(f"❌ Failed to initialize wandb: {e}")
        print(f"❌ wandb type: {type(wandb)}")
        print(f"❌ wandb dir: {dir(wandb)[:10]}")
        raise
    
    return run_name

# === Chart Helpers ===
def configure_wandb_custom_charts(split: str, per_t_rmse: np.ndarray, per_t_mae: Optional[np.ndarray] = None) -> None:
    if not per_t_rmse.size:
        logger.warning(f"No RMSE data for {split} split. Skipping custom chart.")
        return

    # Handle both single-target and multi-target cases
    if per_t_rmse.ndim == 1:
        # Single target case
        timesteps = np.arange(1, len(per_t_rmse) + 1)
        ys = [per_t_rmse.tolist()]
        keys = ["RMSE"]

        if per_t_mae is not None and per_t_mae.size:
            ys.append(per_t_mae.tolist())
            keys.append("MAE")
    else:
        # Multi-target case: average across targets for plotting
        timesteps = np.arange(1, per_t_rmse.shape[0] + 1)
        avg_rmse = np.mean(per_t_rmse, axis=1)
        ys = [avg_rmse.tolist()]
        keys = ["RMSE_avg"]

        if per_t_mae is not None and per_t_mae.size:
            avg_mae = np.mean(per_t_mae, axis=1) if per_t_mae.ndim > 1 else per_t_mae
            ys.append(avg_mae.tolist())
            keys.append("MAE_avg")

    line_plot = wandb.plot.line_series(
        xs=[timesteps.tolist() for _ in ys],
        ys=ys,
        keys=keys,
        title=f"{split.capitalize()} Performance by Timestep",
        xname="Timestep"
    )
    wandb.log({f"charts/timestep_performance": line_plot})

# === Core Evaluation Pipeline ===
def log_rmse_table_as_artifact(per_t_rmse: np.ndarray, split: str, config: Dict) -> None:
    # Handle both single-target and multi-target cases
    if per_t_rmse.ndim == 1:
        # Single target case
        df_rmse = pd.DataFrame({
            "Timestep": [f"t+{i+1}" for i in range(len(per_t_rmse))],
            "RMSE": per_t_rmse
        })
    else:
        # Multi-target case: average across targets or create separate columns
        target_cols = config.get("target_cols", [f"target_{i}" for i in range(per_t_rmse.shape[1])])
        
        # Create DataFrame with timestep and one column per target
        data = {"Timestep": [f"t+{i+1}" for i in range(per_t_rmse.shape[0])]}
        
        for i, target_name in enumerate(target_cols):
            if i < per_t_rmse.shape[1]:
                data[f"RMSE_{target_name}"] = per_t_rmse[:, i]
        
        # Also add average RMSE across all targets
        data["RMSE_avg"] = np.mean(per_t_rmse, axis=1)
        
        df_rmse = pd.DataFrame(data)

    local_path = Path(config["save_path"]) / f"{split}_rmse_table.csv"
    df_rmse.to_csv(local_path, index=False)

    artifact = wandb.Artifact(f"{split}_rmse_table", type="metrics")
    artifact.add_file(str(local_path))
    wandb.run.log_artifact(artifact)

# === Code Logging ===
def setup_wandb_logging() -> None:
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

def log_sigmoid_histogram_bp(logits, labels, epoch: int = 0):
    logits = np.array(logits).flatten()
    labels = np.array(labels).flatten()
    probs = 1 / (1 + np.exp(-logits))

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    fig, ax = plt.subplots()
    ax.hist(pos_probs, bins=50, alpha=0.6, label='Positive (1)', color='red')
    ax.hist(neg_probs, bins=50, alpha=0.6, label='Negative (0)', color='blue')
    ax.set_xlabel("Sigmoid(BP logits)")
    ax.set_ylabel("Count")
    ax.set_title(f"BP Classifier Output Sigmoid - Epoch {epoch}")
    ax.legend()
    wandb.log({f"bp_logits_histogram/epoch_{epoch}": wandb.Image(fig)})
    plt.close(fig)
    
import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb
from pathlib import Path

def log_sigmoid_histogram_bp(
    logits: list | torch.Tensor,
    labels: list | torch.Tensor,
    epoch: int,
    tag: str = "bp",
    save_path: str = "outputs/logits_histograms"
):
    """
    Logs histogram of sigmoid(logits) for positives vs negatives.
    Also logs to W&B if available.
    """
    logits = torch.tensor(logits).squeeze()
    labels = torch.tensor(labels).squeeze()
    probs = torch.sigmoid(logits)

    pos = probs[labels == 1].cpu().numpy()
    neg = probs[labels == 0].cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(neg, bins=100, alpha=0.6, label='Negative (0)', color='blue')
    plt.hist(pos, bins=100, alpha=0.6, label='Positive (1)', color='red')
    plt.xlabel(f"Sigmoid({tag} logits)")
    plt.ylabel("Count")
    plt.title(f"{tag.upper()} Classifier Output Sigmoid - Epoch {epoch}")
    plt.legend()

    Path(save_path).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_path) / f"{tag}_logits_histogram_epoch_{epoch}.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    # Log to W&B
    wandb.log({f"histogram/{tag}_logits_hist_epoch{epoch}": wandb.Image(str(out_path))})
    print(f"[Logged] {out_path}")
