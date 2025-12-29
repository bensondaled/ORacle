import torch
import torch.nn as nn
from encoder import VitalsMedEncoder
from static_embedding import get_static_embedder

class HypoOnsetPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_dim = config["bp_embed_dim"]
        self.num_heads = config["bp_nhead"]

        # FIX: Handle None static embedder properly
        self.static_embed = get_static_embedder(config.get("static_combine_mode", "concat"), config)
        self.static_dim = self.static_embed.output_dim if self.static_embed is not None else 0

        self.encoder = VitalsMedEncoder(config)

        # Replace fusion with clean MHA (batch_first=True)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        # Classifier head
        self.classifier = nn.Linear(self.embed_dim, 1)

    def forward(self, vitals, meds, bolus, attention_mask, gases=None, static_cat=None, static_num=None):
        """
        Predicts hypotension onset from [B, T, *] sequence inputs.

        Args:
            vitals: [B, T, num_vitals] - vital signs
            meds: [B, T, num_meds] - IV medications
            bolus: [B, T, num_bolus] - bolus medications
            attention_mask: [B, T] - attention mask
            gases: [B, T, num_gases] - anesthetic gases (NEW)
            static_cat: dict of categorical static features
            static_num: [B, num_static] - numerical static features
        """
        # FIX: Handle None static embedder properly
        static_embed = None
        if self.static_embed is not None:
            static_embed = self.static_embed(static_cat, static_num)

        # [B, T, D] - Pass gases to encoder
        vit_enc, med_enc, gas_enc = self.encoder(
            vitals=vitals,
            meds=meds,
            bolus=bolus,
            attn_mask=attention_mask,
            gases=gases,  # NEW: Pass gases to encoder
        )

        # Combine medication pathways (IV meds + anesthetic gases)
        combined_meds = med_enc + gas_enc  # [B, T, D]

        # Cross-attend: combined med query attends to vitals
        attn_output, _ = self.cross_attn(
            query=combined_meds,  # [B, T, D]
            key=vit_enc,          # [B, T, D]
            value=vit_enc,        # [B, T, D]
            key_padding_mask=~attention_mask  # [B, T]
        )

        # Use final time step (t0) for prediction
        final = attn_output[:, -1, :]  # [B, D]
        logits = self.classifier(final).squeeze(-1)  # [B] 

        return logits