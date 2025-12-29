# --- fusion.py ---
import torch
import torch.nn as nn

# --- fusion.py ---

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, vit_enc, med_enc):
        # FIXED: vitals attend to meds (correct causal direction: meds → vitals)
        # Vitals ask: "Which medications are affecting me?"
        attn_output, attn_weights = self.cross_attn(query=vit_enc, key=med_enc, value=med_enc)

        # Gated fusion
        gate_input = torch.cat([vit_enc, attn_output], dim=-1)
        g = self.gate(gate_input)
        fused = (1 - g) * vit_enc + g * attn_output
        fused = self.fusion_proj(fused)
        fused = self.dropout(fused)
        fused = self.norm(fused + vit_enc) # Residual connection

        return fused, attn_weights
