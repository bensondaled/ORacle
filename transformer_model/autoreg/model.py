from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import logging

from encoder import VitalsMedEncoder
from fusion import CrossAttentionFusion
from decoder import ARDecoder, TransformerARDecoder
from static_embedding import get_static_embedder

logger = logging.getLogger(__name__)


class IntraOpPredictor(nn.Module):
    """Predict intra-operative mean arterial pressure and optionally classify hypotension onset."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.decoder_type = config.get("decoder_type", "gru").lower()
        self.bp_embed_dim = config["bp_embed_dim"]

        # === Static Embeddings ===
        self.static_embed = get_static_embedder(config.get("static_combine_mode", "concat"), config)
        self.static_dim = self.static_embed.output_dim if self.static_embed else 0

        # === Encoder & Fusion ===
        self.encoder = VitalsMedEncoder(config)
        self.fusion = CrossAttentionFusion(self.bp_embed_dim)

        # === Decoder ===
        if self.decoder_type == "gru":
            self.decoder = ARDecoder(context_dim=self.bp_embed_dim, static_dim=self.static_dim, output_dim=len(config["target_cols"])) # Use len(target_cols)
        elif self.decoder_type == "transformer":
            self.decoder = TransformerARDecoder(
                context_dim=self.bp_embed_dim,
                static_dim=self.static_dim,
                output_dim=len(config["target_cols"]), # Use len(target_cols)
                nhead=config.get("dec_nhead", 4),
                num_layers=config.get("dec_layers", 2),
                dropout=config.get("dec_dropout", 0.1),
            )
        else:
            raise ValueError(f"Invalid decoder_type: {self.decoder_type}")

        # === Hypotension Classifier Heads (Simplified for stability) ===
        self.use_fused = config.get("use_hypo_onset_fused", False)
        self.use_bp = config.get("use_hypo_onset_bp", False)
        self._hypo_pos_rate = config.get("hypo_positive_rate", 0.043)  # Store for initialization
        dropout = config.get("hypo_head_dropout", 0.3)

        # CRITICAL FIX: Hypotension heads expect fused features (128) + static features (static_dim) 
        hypo_input_dim = self.bp_embed_dim + self.static_dim  # Correct total input dimension
        self.hypo_fused_head = self._make_simple_hypo_head(hypo_input_dim) if self.use_fused else None
        self.hypo_bp_head = self._make_simple_hypo_head(hypo_input_dim) if self.use_bp else None

        # === Gas/Med Combination Weights (learnable) ===
        # FIXED: Use learnable weighted average instead of concatenate+project
        self.med_weight = nn.Parameter(torch.tensor(0.6))  # Learned weight for medications
        self.gas_weight = nn.Parameter(torch.tensor(0.4))  # Learned weight for gases

        logger.info(f"Model Hypotension Heads: fused={bool(self.hypo_fused_head)}, bp={bool(self.hypo_bp_head)}")

    def _make_simple_hypo_head(self, total_input_dim: int) -> nn.Module:
        """Simplified hypotension classifier - single layer for stability."""
        # CRITICAL: Use single linear layer to avoid gradient explosion
        # Complex architectures don't help with binary classification
        
        # Get positive rate from config for proper bias initialization
        pos_rate = getattr(self, '_hypo_pos_rate', 0.043)  # Store from config
        bias_init = float(torch.log(torch.tensor(pos_rate / (1 - pos_rate))))  # log odds
        
        # Single linear layer with proper initialization
        head = nn.Sequential(
            nn.LayerNorm(total_input_dim),  # Normalize inputs
            nn.Dropout(0.2),                # Light dropout before projection
            nn.Linear(total_input_dim, 1)   # Direct binary classification
        )
        
        # Proper initialization for extreme class imbalance
        # Use smaller weight initialization to prevent gradient explosion
        nn.init.xavier_uniform_(head[-1].weight, gain=0.1)  # Very small gain
        # Initialize bias to actual log odds for 4.3% positive rate
        nn.init.constant_(head[-1].bias, bias_init)  # Proper bias: ~-3.1
        
        logger.info(f"Hypotension head initialized with bias={bias_init:.3f} for {pos_rate:.1%} positive rate")
        return head

    def forward(
        self,
        vitals: torch.Tensor,
        meds: torch.Tensor,
        gases: torch.Tensor,
        bolus: torch.Tensor,
        attention_mask: torch.Tensor,
        static_cat: Dict[str, torch.Tensor],
        static_num: Optional[torch.Tensor],
        future_steps: int,
        hypo_onset_types: Optional[List[str]] = None,  # For proper masking during training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            preds: Tensor [B, T, 1]
            hypo_fused_logits: Tensor [B, 1] or None
            hypo_bp_logits: Tensor [B, 1] or None
        """
        # === Static embedding (2D for features, 3D for decoder context) ===
        static_embed_2d = self.static_embed(static_cat, static_num) if self.static_embed else None
        if static_embed_2d is None:
            static_embed_2d = torch.zeros(vitals.size(0), self.static_dim, device=vitals.device)
        
        # Task-specific static integration:
        # For decoder: expand to 3D for temporal context
        static_embed_for_decoder = static_embed_2d.unsqueeze(1) if isinstance(self.decoder, ARDecoder) else static_embed_2d

        # === Encode (triple branch: vitals, meds, gases) ===
        # FIXED: Correct argument order - bolus before attention_mask, gases as kwarg
        vit_enc, med_enc, gas_enc = self.encoder(vitals, meds, bolus, attention_mask, gases=gases)

        # === FIXED: Weighted average instead of concatenate+project (no information loss) ===
        # Normalize weights to sum to 1 (softmax-like)
        total_weight = torch.abs(self.med_weight) + torch.abs(self.gas_weight)
        norm_med_weight = torch.abs(self.med_weight) / total_weight
        norm_gas_weight = torch.abs(self.gas_weight) / total_weight

        # Weighted combination of medications and gases
        med_gas_combined = norm_med_weight * med_enc + norm_gas_weight * gas_enc  # [B, L, 128]

        # Cross-attention: vitals query against combined med+gas context
        fused, attention_weights = self.fusion(vit_enc, med_gas_combined)  # [B, L, 128]

        if not torch.compiler.is_compiling():
            logger.debug(f"[fused] shape={fused.shape}, mean={fused.mean():.4f}, std={fused.std():.4f}")
            logger.debug(f"[fused] norm (per-sample avg) = {fused.norm(dim=-1).mean():.4f}")

        # === Decode BP (using static as temporal context) ===
        dec_kwargs = {"context": fused, "static_embed": static_embed_for_decoder, "future_steps": future_steps}
        if isinstance(self.decoder, TransformerARDecoder):
            dec_kwargs["attention_mask"] = attention_mask
        preds = self.decoder(**dec_kwargs)

        # === Enhanced Pooling Strategy (inspired by HypoOnsetPredictor) ===
        # FUSED pooling: Use final timestep (most recent state) for hypotension prediction
        fused_pooled_final = fused[:, -1, :]                      # [B, D] - Current state (best for classification)
        fused_pooled_mean = fused.mean(dim=1)                     # [B, D] - Average state  
        
        # Combine final state with average for robust representation
        fused_pooled = 0.7 * fused_pooled_final + 0.3 * fused_pooled_mean     # [B, D] - Weighted combination
        
        # BP-ONLY pooling: Use only vital encoder outputs (no medication info)
        bp_pooled_final = vit_enc[:, -1, :]                       # [B, D] - Current vitals state
        bp_pooled_mean = vit_enc.mean(dim=1)                      # [B, D] - Average vitals state
        bp_pooled = 0.7 * bp_pooled_final + 0.3 * bp_pooled_mean  # [B, D] - Weighted combination
        
        # === Task-Specific Static Integration for Hypotension Classification ===
        # Use 2D static embeddings as features (not temporal context)
        if static_embed_2d is not None and static_embed_2d.numel() > 0:
            fused_pooled_with_static = torch.cat([fused_pooled, static_embed_2d], dim=1)  # [B, 128 + static_dim]
            bp_pooled_with_static = torch.cat([bp_pooled, static_embed_2d], dim=1)        # [B, 128 + static_dim]
        else:
            fused_pooled_with_static = fused_pooled
            bp_pooled_with_static = bp_pooled

        if not torch.compiler.is_compiling():
            logger.debug(f"[fused_pooled] mean={fused_pooled.mean():.4f}, std={fused_pooled.std():.4f}")
            logger.debug(f"[bp_pooled] mean={bp_pooled.mean():.4f}, std={bp_pooled.std():.4f}")

        # === Enhanced Classification Heads with Static Integration ===
        # CORRECT: hypo_fused uses fused vital+medication representations
        hypo_fused_logits = self.hypo_fused_head(fused_pooled_with_static) if self.hypo_fused_head else None
        # CORRECT: hypo_bp uses only vital/BP representations (no medication info)
        hypo_bp_logits = self.hypo_bp_head(bp_pooled_with_static) if self.hypo_bp_head else None
        
        # === NO MASKING IN MODEL - Let loss function handle it (like run_hypo_classifier.py) ===
        # Just return raw logits, the loss function will mask properly
        return preds, hypo_fused_logits, hypo_bp_logits
