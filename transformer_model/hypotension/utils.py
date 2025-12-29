import os
import hashlib
import glob
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple,List
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from losses.MultiComponentLoss import MultiComponentLoss
from losses.LossModule import LossModule
from losses.learnable_loss_weights import LearnableLossWeights
from torch.utils.data import DataLoader, WeightedRandomSampler

# ─── Logging Setup ───────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

def setup_logging(level: str = "info"):
    """Setup logging configuration"""
    level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    log_level = level_mapping.get(level.lower(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Device Utility ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return a CUDA device if available, else CPU, with logging."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using device: {device}")
        try:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {name}, Memory: {mem:.2f} GB")
        except Exception:
            logger.debug("Could not query GPU properties.")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

# ─── Plotting Utility ─────────────────────────────────────────────────────────
def save_and_log_plot(fig: plt.Figure, save_dir: str, filename: str, wandb_key: Optional[str] = None) -> None:
    """Save a Matplotlib figure and optionally log to WandB."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    if wandb.run is not None and wandb_key:
        wandb.log({wandb_key: wandb.Image(fig)})
    plt.close(fig)

# ─── Cache Key Utility ───────────────────────────────────────────────────────
def generate_cache_key(config: Dict[str, Any], split_name: str) -> str:
    """
    Generate a unique MD5 key based on relevant config fields,
    split name, and the input file path for the split.
    """

    # --- Parameters already included in the original function ---
    keys: Dict[str, Any] = {
        'data_percentage': config.get('data_percentage', 1.0),
        'stride': config.get('stride', 5),
        'use_tokens': config.get('use_tokens', False),
        'use_vae': config.get('use_vae', False),
        'static_combine_mode': config.get('static_combine_mode', 'concat'),
        # Note: loss_mode is less relevant for a data cache but kept from original
        'loss_mode': config.get('loss_mode', 'scheduled'),
        'max_len': config.get('max_len', 60),
        'future_steps': config.get('future_steps', 15),
        'batch_size_bp': config.get('batch_size_bp', 512),
        'split': split_name
    }
    # --- Additional parameters that define data content/structure ---
    keys['vital_cols'] = config.get('vital_cols', [])
    keys['med_cols'] = config.get('med_cols', [])
    keys['bolus_cols'] = config.get('bolus_cols', [])
    keys['target_col'] = config.get('target_col') # Target column is essential, no default like None or it might hash differently

    keys['static_categoricals'] = config.get('static_categoricals', [])
    keys['static_numericals'] = config.get('static_numericals', [])
    keys['scale_numericals'] = config.get('scale_numericals', False)
    keys["finetune_hypo_only"] = config.get("finetune_hypo_only", False)
    keys["balance_hypo_finetune"] = config.get("balance_hypo_finetune", False)
    keys["hypo_balance_ratio"] = config.get("hypo_balance_ratio", 1.0)

        # --- VAE/Tokenization parameters (conditional) ---
    # Include these if VAE or tokenization is enabled, as they affect representation dimensionality/structure
    use_vae = keys['use_vae']
    use_tokens = keys['use_tokens']

    if use_vae or use_tokens:
        keys['vital_latent_dim'] = config.get('vital_latent_dim')
        keys['med_latent_dim'] = config.get('med_latent_dim')

    if use_tokens:
        keys['vital_clusters'] = config.get('vital_clusters')
        keys['med_clusters'] = config.get('med_clusters')

    # --- Input file path for the specific split ---
    # This is crucial - if the input file changes, the cache is invalid
    path_key_map = {'train': 'train_path', 'val': 'val_path', 'test': 'test_path'}
    input_path_config_key = path_key_map.get(split_name)

    if input_path_config_key:
        # Get the specific path from the config
        keys['input_path'] = config.get(input_path_config_key)
        # It's good practice to ensure the path exists or is meaningful here if possible,
        # but for a cache key, just using the config value is sufficient.
    else:
        # Handle unexpected split names - this should ideally not happen if splits are fixed
        keys['input_path'] = f'unknown_split_{split_name}'

    # Sort keys for consistent hashing regardless of insertion order
    # Convert lists to tuples for hashing
    serializable_keys = {k: (tuple(v) if isinstance(v, list) else v) for k, v in keys.items()}
    text = "_".join(f"{k}={v}" for k, v in sorted(serializable_keys.items()))

    # Generate the MD5 hash
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# ─── Cached Batch Streaming ───────────────────────────────────────────────────

class CachedBatchDataset(IterableDataset):
    """
    MEMORY OPTIMIZED IterableDataset that streams pre-cached batches from disk one at a time.
    Each tensor in the batch is moved to the specified device, nested dicts are
    handled one level deep, and all other items are passed through as-is.
    """
    def __init__(self, cache_file: Path, device: torch.device):
        self.cache_file = cache_file
        self.device = device
        # MEMORY FIX: Load metadata only to determine length, not all batches
        temp_data = torch.load(self.cache_file, map_location="cpu")
        self._length = len(temp_data) if isinstance(temp_data, list) else 1
        # CRITICAL: Clean up immediately to prevent memory accumulation
        del temp_data

    def __iter__(self):
        # MEMORY OPTIMIZED: Load and yield batches one by one, cleaning up immediately
        batches = torch.load(self.cache_file, map_location="cpu")
        
        try:
            for i, batch in enumerate(batches):
                batch_on_device: Dict[str, Any] = {}
                for key, value in batch.items():
                    # 1) Pure tensor → move it
                    if isinstance(value, torch.Tensor):
                        batch_on_device[key] = value.to(self.device)
                    # 2) Dict of tensors/scalars → recurse one level
                    elif isinstance(value, dict):
                        nested: Dict[str, Any] = {}
                        for nk, nv in value.items():
                            if isinstance(nv, torch.Tensor):
                                nested[nk] = nv.to(self.device)
                            else:
                                nested[nk] = nv
                        batch_on_device[key] = nested
                    # 3) Anything else (list, int, etc.) → leave untouched
                    else:
                        batch_on_device[key] = value
                
                yield batch_on_device
                
                # CRITICAL: Clean up the original batch immediately after processing
                del batch
                
                # Periodic garbage collection to prevent accumulation
                if i % 50 == 0:
                    import gc
                    gc.collect()
                    
        finally:
            # CRITICAL: Always clean up the batches list
            del batches

    def __len__(self):
        return self._length

# ─── Batch Caching & Loader Utility ─────────────────────────────────────────
def get_cached_batches(
    dataset: Optional[Dataset],
    split_name: str,
    config: Dict[str, Any],
    logger: logging.Logger
) -> DataLoader:
    """
    Return a DataLoader that streams pre-cached batches if available,
    or builds and caches them if not (or if force_recompute_batches=True).
    """
    cache_dir = Path(config['cache_path']) / 'batches'
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = generate_cache_key(config, split_name)
    cache_file = cache_dir / f"{split_name}_batches_{key}.pt"

    use_cache = config.get('use_batch_caching', True)
    force = config.get('force_recompute_batches', False)
    device = get_device()

    # Delete cache if forced
    if force and cache_file.exists():
        try:
            cache_file.unlink()
            logger.info(f"⚡ force_recompute_batches=True: deleted cache {cache_file.name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete {cache_file.name}: {e}")

    # Stream from cache if available
    if use_cache and cache_file.exists():
        logger.info(f"✅ Streaming cached '{split_name}' batches from {cache_file.name}")
        
        # MEMORY MONITORING: Track memory before/after cache loading
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1024**3
            cached_dataset = CachedBatchDataset(cache_file, device)
            memory_after = torch.cuda.memory_allocated() / 1024**3
            if memory_after - memory_before > 0.1:  # If significant memory increase
                logger.info(f"🔍 Cache loading memory: {memory_before:.2f}GB → {memory_after:.2f}GB (+{memory_after-memory_before:.2f}GB)")
            return DataLoader(cached_dataset, batch_size=None)
        else:
            return DataLoader(CachedBatchDataset(cache_file, device), batch_size=None)

    # Guard against empty dataset
    if dataset is None or len(dataset) == 0:
        logger.error(f"❌ Dataset for split '{split_name}' is empty. Cannot cache batches.")
        raise ValueError(f"Dataset for split '{split_name}' has 0 samples.")

    logger.info(f"⏳ Building and caching '{split_name}' batches...")

    builder = DataLoader(
        dataset,
        batch_size=config.get('batch_size_bp', 512),
        shuffle=(split_name == 'train'),
        num_workers=config.get('num_workers', 8),
        pin_memory=config.get('pin_memory', True),
        persistent_workers=config.get('persistent_workers', True),
        prefetch_factor=config.get('prefetch_factor', 4),
        collate_fn=collate_fn
    )

    batches = []
    for batch in tqdm(builder, desc=f"Caching '{split_name}'"):
        cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batches.append(cpu_batch)

    if use_cache:
        try:
            torch.save(batches, cache_file)
            logger.info(f"💾 Saved {len(batches)} '{split_name}' batches to {cache_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed to save '{split_name}' cache: {e}")

    return DataLoader(CachedBatchDataset(cache_file, device), batch_size=None)


# ─── Bolus Mask Utility ───────────────────────────────────────────────────────
def compute_bolus_mask(
    batch: Dict[str, torch.Tensor],
    future_steps: int,
    device: torch.device
) -> torch.Tensor:
    """
    Produce a [B, future_steps] boolean mask where True indicates any bolus event at last input.
    """
    key = 'last_input_bolus' if 'last_input_bolus' in batch else 'bolus'
    tensor = batch.get(key)
    if tensor is None:
        logger.warning(f"No bolus field ('{key}') in batch. Returning all-False mask.")
        return torch.zeros((next(iter(batch.values())).shape[0], future_steps), dtype=torch.bool, device=device)
    flag = tensor.to(device).gt(0).any(dim=-1, keepdim=True)
    return flag.expand(-1, future_steps)

# ─── Loss Module Factory ─────────────────────────────────────────────────────
# ─── Loss Module Factory ─────────────────────────────────────────────────────
def get_loss_modules(config: dict, device: torch.device):
    mode = config.get("loss_mode", "scheduled")
    if mode == "scheduled":
        return None, None  # loss computed with scheduled weights via compute_loss()

    elif mode in ["learnable_multi", "flexible"]:
        loss_module = MultiComponentLoss(
            loss_keys=config.get("loss_keys", []),
            hypo_pos_weight=config.get("hypo_pos_weight", 10.0),
            use_focal_loss=config.get("use_focal_loss", False),
            use_quantile_in_hypo=config.get("use_quantile_in_hypo", True),
            bolus_cols=config.get("bolus_cols", []),
            bolus_column_weights=config.get("bolus_column_weights", {}),
            use_bolus_weights=config.get("use_bolus_weights", False),
            alpha_trend=config.get("alpha_trend", 0.8),
            hypo_loss_weights=tuple(config.get("hypo_loss_weights", [0.5, 0.5])),
            first_penalty_weight=config.get("first_penalty_weight", 0.5),
            first_continuity_weight=config.get("first_continuity_weight", 0.2),
            trend_diff_weights=tuple(config.get("trend_diff_weights", [0.7, 0.3])),
        ).to(device)
        return loss_module, None

    else:
        raise ValueError(f"❌ Unknown loss_mode: {mode}")


# ─── Checkpoint Utilities ─────────────────────────────────────────────────────
def load_latest_checkpoint(
    config: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[lr_scheduler._LRScheduler] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[lr_scheduler._LRScheduler], int]:
    """Load latest checkpoint and return updated (model, optimizer, scheduler, start_epoch)."""
    save_dir = Path(config['save_path']) / config['run_name']
    ckpts = sorted(save_dir.glob('epoch_*.pt'))
    if not ckpts:
        logger.info("No checkpoint found, starting at epoch 0.")
        return model, optimizer, scheduler, 0
    latest = ckpts[-1]
    logger.info(f"🔁 Resuming from checkpoint: {latest.name}")
    chk = torch.load(latest, map_location=get_device())
    model.load_state_dict(chk['model_state'])
    if optimizer and 'optimizer_state' in chk:
        optimizer.load_state_dict(chk['optimizer_state'])
    if scheduler and 'scheduler_state' in chk:
        scheduler.load_state_dict(chk['scheduler_state'])
    start = chk.get('epoch', 0) + 1
    config['start_epoch'] = start
    if wandb.run:
        wandb.config.update({'resume_from_epoch': start}, allow_val_change=True)
    return model, optimizer, scheduler, start

# ─── Scheduler Factory ───────────────────────────────────────────────────────
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[lr_scheduler._LRScheduler]:
    """Create LR scheduler based on config['lr_scheduler']"""
    name = config.get('lr_scheduler', 'cosine')
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('scheduler_t_max', 10),
            eta_min=config.get('scheduler_eta_min', 1e-6)
        )
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('scheduler_step_size', 10),
            gamma=config.get('scheduler_gamma', 0.1)
        )
    return None

# ─── Batch Device Transfer ───────────────────────────────────────────────────
def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor entries in batch dict to the given device."""
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch

# ─── Load Streamed Eval Batches ──────────────────────────────────────────────
def load_streamed_eval_batches(save_dir: str) -> pd.DataFrame:
    """Aggregate NPZ files under save_dir into a single Pandas DataFrame."""
    files = sorted(glob.glob(str(Path(save_dir) / 'batch_*.npz')))
    if not files:
        raise FileNotFoundError(f"No 'batch_*.npz' files found in {save_dir}")
    dfs = []
    for fp in files:
        data = np.load(fp, allow_pickle=True)
        df = pd.DataFrame({
            'prediction':     data['prediction'].tolist(),
            'actual':         data['actual'].tolist(),
            'mask':           data['mask'].tolist(),
            'original_index': data['original_index'].tolist(),
            'mpog_case_id':   data['mpog_case_id'].tolist(),
            'minutes_elapsed':data['minutes_elapsed'].tolist(),
            'last5_bp_values':data['last5_bp_values'].tolist(),
            'last_input_bolus':data['last_input_bolus'].tolist(),
        })
        dfs.append(df)
    summary = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(summary)} rows from {len(files)} streamed batches in {save_dir}")
    return summary

def compute_classification_metrics(logits, labels, prefix=""):
    if logits is None or labels is None:
        return {}
    probs = torch.sigmoid(logits.squeeze(-1))
    preds = (probs > 0.5).float()
    correct = (preds == labels).float()
    accuracy = correct.mean().item()
    true_pos = (preds * labels).sum().item()
    pred_pos = preds.sum().item()
    actual_pos = labels.sum().item()
    precision = true_pos / max(pred_pos, 1e-6)
    recall = true_pos / max(actual_pos, 1e-6)
    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_positive_count": actual_pos
    }

def apply_finetune_hypo_config_overrides(config: dict) -> dict:
    if not config.get("finetune_hypo_only", False):
        return config

    print("🔧 Enabling finetune_hypo_only mode")

    # ─── Heads: don't override if already set ────────────────
    config.setdefault("use_hypo_onset_bp", False)
    config.setdefault("use_hypo_onset_fused", True)

    # ─── Loss config ─────────────────────────────────────────
    config.setdefault("loss_keys", ["hypo"])
    config.setdefault("task_mode", "hypo")
    config.setdefault("loss_mode", "learnable_multi")

    # ─── Focal or BCE setup ──────────────────────────────────
    use_focal = config.get("use_focal_loss", False)
    config["use_focal_loss"] = use_focal
    if use_focal:
        config.setdefault("hypo_focal_alpha", 0.25)
        config.setdefault("hypo_focal_gamma", 1.0)
    else:
        config.setdefault("hypo_pos_weight", 5.0)

    # ─── Balancing strategy ──────────────────────────────────
    config.setdefault("balance_hypo_finetune", False)
    config.setdefault("loader", {})
    config["loader"].setdefault("use_balanced_sampler", False)
    config["loader"]["shuffle"] = True  # always shuffle

    # ─── Loss weights and schedule (default only) ─────────────
    config.setdefault("loss_weights", {})
    if "hypo_onset_fused" not in config["loss_weights"]:
        config["loss_weights"]["hypo_onset_fused"] = 1.0
    if "hypo_onset_bp" not in config["loss_weights"]:
        config["loss_weights"]["hypo_onset_bp"] = 0.0

    config.setdefault("loss_schedule", {})
    if "hypo_onset_fused" not in config["loss_schedule"]:
        config["loss_schedule"]["hypo_onset_fused"] = {
            "start_weight": 1.0,
            "end_weight": 1.0,
            "num_epochs": config.get("epochs_bp", 5)
        }

    # ─── Epochs ───────────────────────────────────────────────
    config["epochs_bp"] = max(config.get("epochs_bp", 1), 1)

    # ─── Log strategy ─────────────────────────────────────────
    strategy = (
        "Dataset-balanced" if config["balance_hypo_finetune"] else
        "Sampler-balanced" if config["loader"]["use_balanced_sampler"] else
        "Unbalanced"
    )
    head = (
        "Fused" if config.get("use_hypo_onset_fused", False) else
        "BP" if config.get("use_hypo_onset_bp", False) else
        "None"
    )
    print(f"📊 Finetuning: {strategy}, {'Focal' if use_focal else 'BCE'} loss, Head = {head}")

    return config


def to_numpy(x):
    """
    Convert tensors, arrays, or lists into a NumPy ndarray.

    - torch.Tensor   → detached CPU numpy array
    - np.ndarray     → returned unchanged
    - list/tuple of torch.Tensor → stacked numpy array
    - list/tuple of scalars      → numpy array via np.asarray
    - anything else              → np.asarray fallback
    """
    # Tensor → NumPy
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    # Already an ndarray
    if isinstance(x, np.ndarray):
        return x

    # List or tuple
    if isinstance(x, (list, tuple)):
        # All tensors → stack their numpy values
        if all(isinstance(el, torch.Tensor) for el in x):
            # Ensure stacking preserves the last dimension for multi-target predictions
            return np.stack([el.detach().cpu().numpy() for el in x])
        # Scalar list → asarray
        return np.asarray(x)

    # Fallback for pandas Series, scalars, etc.
    return np.asarray(x).reshape(1) if np.isscalar(x) else np.asarray(x)

def apply_finetune_bolus_config_overrides(config: dict) -> dict:
    """
    Modify the config in-place if finetune_bolus_only is enabled.
    Ensures only the bolus-response component is trained and evaluated,
    and turns on per-column bolus weighting if provided.
    """
    if not config.get("finetune_bolus_only", False):
        return config

    print("🔧 Overriding config for finetune_bolus_only mode...")

    # 1) Only train the 'bolus' loss component
    config["loss_keys"] = ["bolus"]
    config["loss_schedule"] = {
        "bolus": {
            "start_weight": 1.0,
            "end_weight":   1.0,
            "num_epochs":   config.get("epochs_bp", 10),
        }
    }
    config["loss_mode"] = "fixed"

    # 2) Disable all other heads and auxiliary losses
    config["use_hypo_onset_bp"]    = False
    config["use_hypo_onset_fused"] = False

    # 4) Bolus balancing via sampling

    # 5) If you provided per-med weights in config["bolus_column_weights"],
    #    enable their use during bolus-response loss.
    if "bolus_column_weights" in config and isinstance(config["bolus_column_weights"], dict):
        config["use_bolus_column_weights"] = True
    else:
        config["use_bolus_column_weights"] = False

    return config


def get_smart_stride_indices(
    df: pd.DataFrame,
    stride: int = 5,
    bolus_threshold: float = 1.0,
    override_colnames: List[str] = ["hypo_onset_label", "last_input_bolus"],
) -> List[int]:
    """
    Smart stride: take every `stride` row, plus any row where:
    - hypo_onset_label == 1
    - OR any value in last_input_bolus >= bolus_threshold
    """
    n = len(df)
    selected = set(range(0, n, stride))

    if "hypo_onset_label" in df.columns:
        selected |= set(df.index[df["hypo_onset_label"] == 1])

    if "last_input_bolus" in df.columns:
        selected |= set(
            df.index[df["last_input_bolus"].apply(lambda x: any(b >= bolus_threshold for b in x))]
        )

    return sorted(selected)

def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if key == 'static_cat':
            if isinstance(batch[0][key], dict) and batch[0][key]:
                subkeys = batch[0][key].keys()
                collated[key] = {
                    subkey: torch.stack([item[key][subkey] for item in batch])
                    for subkey in subkeys
                }
            else:
                collated[key] = {}
        elif key == 'static_num':
            if batch[0][key] is None:
                collated[key] = None
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        elif key == 'last5_bp_values':
            collated[key] = torch.stack([
                item[key] if isinstance(item[key], torch.Tensor)
                else torch.tensor(item[key], dtype=torch.float32)
                for item in batch
            ])
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
                if key == 'bolus':
                    logger.debug(f"Batch bolus: {collated[key].sum().item()}")
            except (TypeError, RuntimeError):
                collated[key] = [item[key] for item in batch]
    return collated

import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def inject_hypotension_heads(model, config):
    # DISABLED: This overwrites properly initialized heads with wrong dimensions
    # if config.get("use_hypo_onset_bp", False):
    #     model.hypo_bp_head = nn.Linear(config["bp_embed_dim"], 1)  # Wrong: should be 144→128→64→1
    # if config.get("use_hypo_onset_fused", False):
    #     model.hypo_fused_head = nn.Linear(config["bp_embed_dim"], 1)  # Wrong: should be 144→128→64→1

    # DISABLED: Broken reinitializer that corrupts nn.Sequential heads
    # if config.get("reset_hypo_head", True):
    #     if hasattr(model, "hypo_bp_head") and isinstance(model.hypo_bp_head, nn.Linear):
    #         logger.info("🔁 Reinitializing hypo_bp_head weights")
    #         nn.init.xavier_uniform_(model.hypo_bp_head.weight)
    #         if model.hypo_bp_head.bias is not None:
    #             nn.init.zeros_(model.hypo_bp_head.bias)
    #     if hasattr(model, "hypo_fused_head") and isinstance(model.hypo_fused_head, nn.Linear):
    #         logger.info("🔁 Reinitializing hypo_fused_head weights")
    #         nn.init.xavier_uniform_(model.hypo_fused_head.weight)
    #         if model.hypo_fused_head.bias is not None:
    #             nn.init.zeros_(model.hypo_fused_head.bias)
    pass