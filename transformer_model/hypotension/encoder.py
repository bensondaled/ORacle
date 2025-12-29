import torch
import torch.nn as nn
import math
from temporal_encoding import MedicalTemporalEncoding, AdaptiveTemporalEncoding

class VitalsMedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config['bp_embed_dim']
        self.max_len = config['max_len']
        self.pos_encoding_type = config.get('pos_encoding_type', 'learned')

        vital_dim = config['vital_latent_dim'] if config.get('use_vae', False) else len(config['vital_cols'])
        self.vital_proj = nn.Linear(vital_dim, self.embed_dim)

        self.num_meds = len(config['med_cols'])
        self.num_bolus = len(config['bolus_cols'])
        self.num_gases = len(config.get('gas_cols', []))
        
        # Triple-pathway architecture: IV meds, gases, and bolus
        self.med_proj = nn.Sequential(
            nn.Linear(self.num_meds, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.1),
        )
        
        self.gas_proj = nn.Sequential(
            nn.Linear(self.num_gases, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.1),
        ) if self.num_gases > 0 else None
        
        self.bolus_encoder = nn.Sequential(
            nn.Linear(self.num_bolus, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.1),
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config['bp_nhead'],
            dim_feedforward=config['bp_dim_ff'],
            dropout=config['bp_dropout'],
            batch_first=True,
            norm_first=True,
        )
        self.vital_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['bp_num_layers'])
        self.med_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['bp_num_layers'])

        # FIXED: Dedicated encoder for gases (different temporal dynamics than IV meds)
        gas_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config['bp_nhead'],
            dim_feedforward=config['bp_dim_ff'],
            dropout=config['bp_dropout'],
            batch_first=True,
            norm_first=True,
        )
        self.gas_encoder = nn.TransformerEncoder(gas_encoder_layer, num_layers=config['bp_num_layers'])

        if self.pos_encoding_type == 'learned':
            self.pos_embed = nn.Parameter(torch.randn(1, self.max_len, self.embed_dim) * 0.02)
        elif self.pos_encoding_type == 'sinusoidal':
            self.register_buffer('pos_embed', self._get_sinusoidal_encoding(self.max_len, self.embed_dim))
        elif self.pos_encoding_type == 'medical':
            # Enhanced medical temporal encoding
            self.medical_pos_encoder = MedicalTemporalEncoding(
                embed_dim=self.embed_dim,
                max_len=self.max_len,
                num_bolus_drugs=len(config['bolus_cols']),
                use_drug_encoding=True,
                use_physio_encoding=True,
                use_relative_encoding=True,
                learnable_frequencies=True
            )
        elif self.pos_encoding_type == 'adaptive':
            # Simplified adaptive encoding (easier to integrate)
            self.adaptive_pos_encoder = AdaptiveTemporalEncoding(
                embed_dim=self.embed_dim,
                max_len=self.max_len,
                num_drug_types=len(config['bolus_cols'])
            )
        else:
            raise ValueError(f"Unknown pos_encoding_type: {self.pos_encoding_type}")

    def _get_sinusoidal_encoding(self, max_len, embed_dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, vitals, meds, bolus, attn_mask, gases=None):
        """Encode vitals, IV meds, gases, and bolus separately then fuse all medications."""
        B, L, _ = vitals.shape

        # ─── Normalize & sanity‐check attention mask ─────────────────
        # expect a [B, L] boolean mask where True=valid
        if not isinstance(attn_mask, torch.BoolTensor):
            attn_mask = attn_mask.to(dtype=torch.bool)
        assert attn_mask.shape == (B, L), \
            f"attn_mask must be [B, L]={(B,L)}, got {tuple(attn_mask.shape)}"

        # --- Project vitals ---
        vit_proj = self.vital_proj(vitals)  # [B, L, E]

        # --- Project IV meds ---
        meds_flat = meds.view(B*L, -1)
        med_proj = self.med_proj(meds_flat)
        med_proj = med_proj.view(B, L, -1)  # [B, L, E]

        # --- Project gases ---
        if self.gas_proj is not None and gases is not None:
            gases_flat = gases.view(B*L, -1)
            gas_proj = self.gas_proj(gases_flat)
            gas_proj = gas_proj.view(B, L, -1)  # [B, L, E]
        else:
            gas_proj = torch.zeros_like(med_proj)  # [B, L, E] zeros if no gases

        # --- Project bolus ---
        bolus_flat = bolus.view(B*L, -1)
        bolus_proj = self.bolus_encoder(bolus_flat)
        bolus_proj = bolus_proj.view(B, L, -1)  # [B, L, E]

        # --- FIXED: Proper gated combination (no double-counting) ---
        # Gate learns to interpolate between infusion-only and bolus representations
        # When gate=0: output = med_proj (infusion only)
        # When gate=1: output = bolus_proj (bolus dominates)
        bolus_gate = torch.sigmoid(bolus_proj)    # [B, L, E] ➔ values in [0, 1]

        # Linear interpolation: NO residual connection to avoid double-counting infusion
        # med_gated = (1 - g) * h_inf + g * h_bol
        med_gated = (1 - bolus_gate) * med_proj + bolus_gate * bolus_proj

        # --- Enhanced Positional encoding ---
        if self.pos_encoding_type in ['learned', 'sinusoidal']:
            pos_embed = self.pos_embed[:, -L:, :]
            vit_proj = vit_proj + pos_embed
            med_gated = med_gated + pos_embed
            gas_proj = gas_proj + pos_embed
        elif self.pos_encoding_type == 'medical':
            # Use enhanced medical temporal encoding
            bolus_mask = (bolus.sum(dim=-1) > 0)  # [B, L] - detect bolus events
            dominant_drugs = torch.argmax(bolus, dim=-1)  # [B, L] - dominant drug at each step
            
            pos_embed = self.medical_pos_encoder(
                seq_len=L,
                bolus_mask=bolus_mask,
                last_bolus_drugs=bolus,
                device=vitals.device
            )  # [B, L, E]
            
            vit_proj = vit_proj + pos_embed
            med_gated = med_gated + pos_embed
            gas_proj = gas_proj + pos_embed
            
        elif self.pos_encoding_type == 'adaptive':
            # Use simplified adaptive encoding
            bolus_mask = (bolus.sum(dim=-1) > 0)  # [B, L] - detect bolus events
            dominant_drugs = torch.argmax(bolus, dim=-1)  # [B, L] - dominant drug at each step
            
            pos_embed = self.adaptive_pos_encoder(
                seq_len=L,
                bolus_mask=bolus_mask,
                dominant_drugs=dominant_drugs
            )  # [B, L, E]
            
            vit_proj = vit_proj + pos_embed
            med_gated = med_gated + pos_embed
            gas_proj = gas_proj + pos_embed

        # --- Triple Encoder Approach: Separate branches ---
        vit_enc = self.vital_encoder(vit_proj, src_key_padding_mask=~attn_mask)

        # FIXED: Separate encoders for IV medications and gases
        med_enc = self.med_encoder(med_gated, src_key_padding_mask=~attn_mask)     # IV meds encoder
        gas_enc = self.gas_encoder(gas_proj, src_key_padding_mask=~attn_mask)      # Dedicated gas encoder

        return vit_enc, med_enc, gas_enc  # Return all three separate branches
