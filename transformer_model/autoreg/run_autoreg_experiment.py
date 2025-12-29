import logging
import warnings
import copy
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import wandb
from data_utils import load_and_split_data
from run_scripts import (
    run_training_loop,
    
)
from data_utils import make_loader
from logging_utils import init_logging, init_run
from eval_autoreg import eval_autoreg
from model import IntraOpPredictor
from utils import get_device, get_loss_modules, apply_finetune_hypo_config_overrides,inject_hypotension_heads

import torch.nn as nn

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False"
)

logger = logging.getLogger(__name__)
def freeze_model_layers(model, config):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for key in config.get("finetune_unfrozen_layers", []):
            if key in name:
                param.requires_grad = True
                print(f"🔓 Unfrozen: {name}")

def save_yaml(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f)

def log_config_diff(orig: dict, final: dict):
    for k in sorted(set(orig) | set(final)):
        o = orig.get(k, '<MISSING>')
        n = final.get(k, '<MISSING>')
        if o != n:
            logger.info(f"⚙️ config['{k}'] changed: {o!r} → {n!r}")

def log_config_artifact(config: dict, save_dir: Path, artifact_name: str = "config"):
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / f"{artifact_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    art = wandb.Artifact(name=artifact_name, type="config")
    art.add_file(str(config_path))
    wandb.log_artifact(art)

def run_autoreg_experiment(_, __, ___, config: dict, dataset_class):
    # 1) Logging & device
    init_logging(config)
    logger.info("Starting autoreg experiment")
    device = get_device()
    
    # Enable TensorFloat32 for faster matrix multiplication on A100/RTX/etc
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        logger.info("✅ Enabled TensorFloat32 for faster matmul")

    # 2) Snapshot & overrides
    # Early config fixup
    config = apply_finetune_hypo_config_overrides(config)
    orig_cfg = copy.deepcopy(config)  # Now truly reflects overridden config
    run_name = init_run(config)

    wandb.config.update({"finetune_mode": config.get("finetune_hypo_only", False)})

    # 3) Config diffs & artifacts
    log_config_diff(orig_cfg, config)
    save_dir = Path(config['save_path']) / config['run_name']
    save_yaml(orig_cfg, save_dir / 'config_original.yaml')
    save_yaml(config,    save_dir / 'config_used.yaml')
    log_config_artifact(config, save_dir)

    # ─── Loss setup ───────────────────────────────────────────────
    loss_module, loss_weight_module = get_loss_modules(config, device)
    logger.info(
        f"LOSS_MODE={config['loss_mode']!r} → "
        f"loss_module={loss_module.__class__.__name__ if loss_module else None}, "
        f"loss_weight_module={loss_weight_module}"
    )
    logger.info(f"✅ Using loss_mode={config.get('loss_mode', 'scheduled')}")

    # ─── Data loading ─────────────────────────────────────────────
    dfs, datasets, loaders = load_and_split_data(config, dataset_class)
    if not config.get("use_batch_caching", False):
        loaders = {
            'train': make_loader(datasets['train'], shuffle=True,  config=config),
            'val':   make_loader(datasets['val'],   shuffle=False, config=config),
            'test':  make_loader(datasets['test'],  shuffle=False, config=config),
        }
    # ─── Model instantiation ──────────────────────────────────────
    model = IntraOpPredictor(config)
    
    # Hypotension head initialization is now handled in model._make_simple_hypo_head()
    # The simplified single-layer architecture initializes itself properly
    if config.get("use_hypo_onset_fused", False) or config.get("use_hypo_onset_bp", False):
        logger.info(f"🎯 Hypotension heads initialized with simplified architecture")
        logger.info(f"   Positive rate: {config.get('hypo_positive_rate', 0.043):.1%}")
        logger.info(f"   Using single-layer head with proper bias initialization")

    model.to(device)  # ✅ Everything moves together

    # 🔁 Hypo‑only fine‑tuning
    if config.get("finetune_hypo_only", False):
        ckpt = config.get("pretrained_checkpoint")
        if not ckpt or not Path(ckpt).exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt}")

        logging.info(f"\U0001f4e6 Loading pretrained model from: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model_state'], strict=False)

        with torch.no_grad():
            if config.get("zero_hypo_head_bias", False):
                if config.get("use_hypo_onset_fused", False) and hasattr(model, "hypo_fused_head"):
                    if model.hypo_fused_head and model.hypo_fused_head.bias is not None:
                        nn.init.zeros_(model.hypo_fused_head.bias)
                if config.get("use_hypo_onset_bp", False) and hasattr(model, "hypo_bp_head"):
                    if model.hypo_bp_head and model.hypo_bp_head.bias is not None:
                        nn.init.zeros_(model.hypo_bp_head.bias)

        # ✅ NEW: freeze based on config['finetune_unfrozen_layers'] only
        freeze_model_layers(model, config)

        if not (config.get("use_hypo_onset_fused", False) or config.get("use_hypo_onset_bp", False)):
            logging.warning("⚠️ No hypotension head is enabled, but finetune_hypo_only is True. Loss will be ineffective.")

    # Optional torch.compile with optimized settings
    if config.get("use_compile", True):
        try:
            # Use optimized compile settings to avoid CUDA graph warnings
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
            logger.info("✅ Model compiled with torch.compile (reduce-overhead mode)")
            
            # Optional warmup to trigger compilation early
            if config.get("compile_warmup", True):
                logger.info("🔥 Warming up torch.compile...")
                with torch.no_grad():
                    # Create dummy batch matching expected input shape with correct dtypes
                    dummy_batch = {
                        'vitals': torch.randn(2, config['max_len'], len(config['vital_cols']), device=device, dtype=torch.float32),
                        'meds': torch.randn(2, config['max_len'], len(config['med_cols']), device=device, dtype=torch.float32),
                        'gases': torch.randn(2, config['max_len'], len(config.get('gas_cols', [])), device=device, dtype=torch.float32),
                        'bolus': torch.randn(2, config['max_len'], len(config.get('bolus_cols', [])), device=device, dtype=torch.float32),
                        'attention_mask': torch.ones(2, config['max_len'], dtype=torch.bool, device=device),
                        'static_cat': {col: torch.zeros(2, dtype=torch.long, device=device) 
                                      for col in config.get('static_categoricals', [])},
                        'static_num': torch.randn(2, len(config.get('static_numericals', [])), device=device, dtype=torch.float32) 
                                     if config.get('static_numericals') else None
                    }
                    # Trigger compilation
                    _ = model(**{k: v for k, v in dummy_batch.items() if k != 'static_cat'}, 
                             static_cat=dummy_batch['static_cat'], 
                             static_num=dummy_batch['static_num'],
                             future_steps=config['future_steps'])
                logger.info("✅ Warmup completed, model ready for training")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile failed: {e}")

    # ─── Training ─────────────────────────────────────────────────
    best_state, best_epoch, best_ckpt, summary_df = run_training_loop(
        model=model,
        loaders=loaders,
        config=config,
        logger=logger,
        run_name=config['run_name'],
        test_df=dfs['test'],
        loss_module=loss_module,
        loss_weight_module=loss_weight_module
    )
    if best_state is None:
        raise RuntimeError("Training failed: no valid checkpoint.")

    # ─── Final evaluation ────────────────────────────────────────
    logger.info(f"📈 Evaluating best model at epoch {best_epoch}…")
    # load best weights
    ck = torch.load(best_ckpt, map_location=device)['model_state']
    model.load_state_dict(ck)
    model.eval()

    # autoreg + hypo‐head evaluation
    test_metrics = eval_autoreg(
        model=model,
        dataloader=loaders['test'],
        config=config,
        save_path=save_dir/"eval_csvs",   
        device=device,
        loss_module=loss_module,
        loss_weight_module=loss_weight_module,
    )
    # Extract outputs for full or hypo-only evaluation
    preds   = test_metrics.get("preds")
    targets = test_metrics.get("targets")
    masks   = test_metrics.get("mask")
    regression_available = all(x is not None for x in [preds, targets, masks])

    # if not regression_available:
    #     logger.warning("⚠️ Skipping RMSE/plotting — finetune_hypo_only mode assumed")

    # # Evaluate regression + classification outputs
    # df_sum = evaluate_and_analyze_autoreg(
    #     preds=preds if regression_available else None,
    #     targets=targets if regression_available else None,
    #     masks=masks if regression_available else None,
    #     original_indices=test_metrics.get("original_index"),
    #     case_ids=test_metrics.get("mpog_case_id"),
    #     elapsed_time=test_metrics.get("minutes_elapsed"),
    #     last5_bp_values=test_metrics.get("last5_bp_values"),
    #     last_input_bolus=test_metrics.get("last_input_bolus"),
    #     config=config,
    #     run_name=config['run_name'],
    #     save_dir=save_dir,
    #     split='test',
    #     hypo_fused_logits=test_metrics.get("hypo_fused_logits"),
    #     hypo_bp_logits=test_metrics.get("hypo_bp_logits"),
    #     hypo_onset_labels=test_metrics.get("hypo_onset_labels"),
    # )

    # # Postprocess + W&B metric logging
    # postprocess_and_log_autoreg_metrics(
    #     df_sum,
    #     dfs['test'],
    #     config['run_name'],
    #     save_dir=save_dir
    # )

    #     # Log scalar evaluation metrics to W&B
    # # W&B: Log scalar final evaluation metrics
    # wandb.define_metric("final_eval/*", step_metric="final_eval_step")
    # logd = {"final_eval_step": best_epoch}

    # for k, v in test_metrics.items():
    #     # Scalar: safe to log
    #     if isinstance(v, (int, float, np.number)):
    #         logd[f"final_eval/{k}"] = v
    #     elif isinstance(v, torch.Tensor) and v.numel() == 1:
    #         logd[f"final_eval/{k}"] = v.item()
    #     elif isinstance(v, (np.ndarray, torch.Tensor)) and v.size == 1:
    #         logd[f"final_eval/{k}"] = float(v)

    # # If hypo-only fine-tune, also pull scalar metrics from df_sum
    # if config.get("finetune_hypo_only", False):
    #     metrics_to_log = [
    #         "hypo_fused_f1", "hypo_fused_precision", "hypo_fused_recall", "hypo_fused_auprc", "hypo_fused_auroc",
    #         "hypo_bp_f1", "hypo_bp_precision", "hypo_bp_recall", "hypo_bp_auprc", "hypo_bp_auroc"
    #     ]
    #     for key in metrics_to_log:
    #         val = df_sum.attrs.get(key)  # assuming postprocess attaches metrics as attributes
    #         if val is not None:
    #             logd[f"final_eval/{key}"] = val

    # wandb.log(logd)

    # df_sum.to_feather(save_dir / "test_summary.feather")
