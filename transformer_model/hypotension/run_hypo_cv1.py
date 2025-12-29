"""
Run hypotension classifier for CV run 1.
Modified version of run_hypo_classifier.py with configurable config path.
"""
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
import yaml
import pandas as pd
from tqdm import tqdm
import sys
import os

# Use local hypo_with_gas folder
sys.path.insert(0, '/remote/home/marcgh/CV_intraop_model/hypo_with_gas')

from hypo_onset_predictor import HypoOnsetPredictor
from hypo_dataset import HypoDataset, load_datasets_for_hypo
from utils import get_device, move_batch_to_device
from eval_hypo_classifier import eval_hypo_classifier
from intraop_dataset_hypo import IntraOpDataset

pd.set_option('future.no_silent_downcasting', True)

def move_batch_to_device_fast(batch, device):
    """Optimized version with non_blocking transfer"""
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch

def main():
    print(f"🚀 Starting hypotension classification for CV run 1 at {pd.Timestamp.now()}")

    # === Load config ===
    config_path = "/remote/home/marcgh/CV_intraop_model/hypo_with_gas/config_hypo_cv1.yaml"
    print(f"📜 Loading config from {config_path}...")
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded: {config['run_name']}")
        print(f"   Train: {config['train_path']}")
        print(f"   Val:   {config['val_path']}")
        print(f"   Test:  {config['test_path']}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    # === Load datasets ===
    print("\n📦 Loading and filtering datasets...")
    print("   Filtering for: true_onset, none, ongoing")
    try:
        train_dataset, val_dataset, test_dataset = load_datasets_for_hypo(
            config, dataset_class=IntraOpDataset, filter_types=["true_onset", "none", "ongoing"]
        )
        print(f"✅ Datasets loaded:")
        print(f"   Train: {len(train_dataset):,} samples")
        print(f"   Val:   {len(val_dataset):,} samples")
        print(f"   Test:  {len(test_dataset):,} samples")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    # === Init device ===
    print("\n🔍 Initializing device...")
    try:
        device = get_device()
        print(f"✅ Device: {device}, CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ Error initializing device: {e}")
        return

    # === WandB ===
    print("\n📡 Initializing Wandb...")
    try:
        wandb.init(
            project=config["wandb_project"],
            name=config["run_name"],
            config=config,
            mode="disabled"  # Disable for initial run
        )
        print("✅ Wandb initialized (disabled mode)")
    except Exception as e:
        print(f"❌ Error initializing Wandb: {e}")
        return

    # === Create DataLoaders ===
    print("\n🚚 Creating DataLoaders...")
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size_bp"],
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("persistent_workers", True),
            prefetch_factor=config.get("prefetch_factor", 4),
            collate_fn=IntraOpDataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=IntraOpDataset.collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=IntraOpDataset.collate_fn
        )
        print(f"✅ DataLoaders created:")
        print(f"   Train: {len(train_loader)} batches (batch_size={config['batch_size_bp']})")
        print(f"   Val:   {len(val_loader)} batches")
        print(f"   Test:  {len(test_loader)} batches")
    except Exception as e:
        print(f"❌ Error creating DataLoaders: {e}")
        return

    # === Initialize Model ===
    print("\n🤖 Initializing model...")
    try:
        model = HypoOnsetPredictor(config).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model initialized with {num_params:,} trainable parameters")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return

    # === Setup Optimizer ===
    print("\n⚙️ Setting up optimizer...")
    try:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["lr_bp"],
            weight_decay=config.get("wd_bp", 0.0)
        )
        scaler = GradScaler()
        print(f"✅ Optimizer: Adam (lr={config['lr_bp']}, wd={config.get('wd_bp', 0.0)})")
    except Exception as e:
        print(f"❌ Error setting up optimizer: {e}")
        return

    # === Training Loop ===
    print(f"\n🚀 Starting training for {config['epochs_bp']} epochs...")
    print(f"   Positive weight: {config.get('pos_weight', 25.0)}")
    print(f"   Gradient clipping: {config.get('gradient_clip_val', 1.0)}")

    for epoch in range(config["epochs_bp"]):
        print(f"\n{'='*80}")
        print(f"🟢 EPOCH {epoch+1}/{config['epochs_bp']}")
        print(f"{'='*80}")

        model.train()
        total_loss, num_batches = 0.0, 0
        num_pos, num_neg = 0, 0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            try:
                optimizer.zero_grad(set_to_none=True)
                move_batch_to_device_fast(batch, device)

                # Mixed precision forward pass
                with autocast():
                    logits = model(
                        vitals=batch["vitals"],
                        meds=batch["meds"],
                        bolus=batch["bolus"],
                        attention_mask=batch["attention_mask"],
                        gases=batch.get("gases"),  # NEW: Pass gases to model
                        static_cat=batch.get("static_cat"),
                        static_num=batch.get("static_num")
                    )

                    labels = batch["hypo_onset_label"].float()
                    valid_mask = torch.ones_like(labels, dtype=torch.bool)
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    # Count positives and negatives
                    num_pos += (valid_labels == 1).sum().item()
                    num_neg += (valid_labels == 0).sum().item()

                    pos_weight = torch.tensor([config.get("pos_weight", 25.0)], device=device)
                    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels, pos_weight=pos_weight)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ Invalid loss at batch {i}")
                    continue

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get("gradient_clip_val", 1.0))
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                num_batches += 1

                # Log every 100 batches
                if i % 100 == 0 and i > 0:
                    avg_loss = total_loss / num_batches
                    print(f"   Batch {i}/{len(train_loader)}: loss={avg_loss:.4f}, pos={num_pos}, neg={num_neg}")

            except Exception as e:
                print(f"❌ Error in batch {i}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        print(f"\n✅ Epoch {epoch+1} completed:")
        print(f"   Average train loss: {avg_loss:.4f}")
        print(f"   Total positives: {num_pos:,}")
        print(f"   Total negatives: {num_neg:,}")
        print(f"   Class ratio: {num_pos/(num_pos+num_neg)*100:.2f}% positive")

        # Evaluate on validation set
        print(f"\n📊 Evaluating on validation set...")
        try:
            metrics = eval_hypo_classifier(model, val_loader, device)
            print(f"   Val metrics: AUC={metrics.get('auc', 0):.4f}, F1={metrics.get('f1', 0):.4f}")
        except Exception as e:
            print(f"   ⚠️  Evaluation error: {e}")

    # === Final Evaluation ===
    print(f"\n{'='*80}")
    print(f"📊 FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}")
    try:
        # Save predictions to CSV
        pred_save_path = os.path.join(config.get("save_path", "/remote/home/marcgh/CV_intraop_model/runs/hypo_cv1"), "test_predictions.csv")
        metrics = eval_hypo_classifier(model, test_loader, device, save_path=pred_save_path)
        print(f"\n✅ Test metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
    except Exception as e:
        print(f"❌ Error in final evaluation: {e}")

    # === Save Model ===
    print(f"\n💾 Saving model...")
    try:
        save_dir = config.get("save_path", "/remote/home/marcgh/CV_intraop_model/runs/hypo_cv1")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "hypo_classifier_cv1.pt")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to: {model_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    print(f"\n🎉 Training completed at {pd.Timestamp.now()}")

if __name__ == "__main__":
    main()
