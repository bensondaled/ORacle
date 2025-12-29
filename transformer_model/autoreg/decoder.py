# decoder.py – autoregressive decoders (GRU / Transformer)
# ---------------------------------------------------------
# Both classes share an identical forward signature so the rest
# of the pipeline can switch by *just* changing ``config['decoder_type']``.
# ---------------------------------------------------------
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional
from temporal_encoding import AdaptiveTemporalEncoding

__all__ = [
    "ARDecoder",
    "TransformerARDecoder",
]

# ───────────────────── positional encoding helper ───────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim: int, max_len: int = 5000, encoding_type: str = "sinusoidal"):
        super().__init__()
        self.encoding_type = encoding_type.lower()
        self.input_dim = input_dim
        self.max_len = max_len

        if self.encoding_type == "learned":
            self.pe = nn.Parameter(torch.zeros(1, max_len, input_dim))
            nn.init.normal_(self.pe, std=0.02)
        elif self.encoding_type == "sinusoidal":
            pos = torch.arange(max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim))
            pe = torch.zeros(max_len, input_dim)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, D]
        pe_slice = self.pe[:, :x.size(1)]
        return x + pe_slice
# ──────────────────────────── GRU version ───────────────────────────

# decoder.py — Corrected GRU Decoder Version

class ARDecoder(nn.Module):
    """
    Autoregressive GRU decoder for time series, using full encoder context at initialization.

    Parameters
    ----------
    context_dim : output dimension of encoder (C)
    static_dim  : static feature embedding size (S)
    output_dim  : target prediction dimension (usually 1 for BP)
    """

    def __init__(self, context_dim: int, static_dim: int, output_dim: int):
        super().__init__()
        self.context_dim = context_dim
        self.static_dim = static_dim
        self.output_dim = output_dim

        # Project scalar prediction back into context space
        self.tf_proj = nn.Linear(output_dim, context_dim)

        # GRU: takes (context + static) at each decoding step
        self.rnn = nn.GRU(
            input_size=context_dim + static_dim,
            hidden_size=context_dim,
            batch_first=True,
        )

        # Head to predict scalar BP from GRU output
        self.head = nn.Linear(context_dim, output_dim)

        # Memory attention to summarize full encoder context
        self.memory_attn = nn.MultiheadAttention(context_dim, num_heads=4, batch_first=True)

        # Projection to initialize GRU hidden state
        self.init_proj = nn.Linear(context_dim, context_dim)

    def forward(
        self,
        context: torch.Tensor,          # [B, Lc, C]
        static_embed: torch.Tensor,     # [B, 1, S]
        future_steps: int,
        teacher_forcing_inputs: Optional[torch.Tensor] = None,  # [B, T, output_dim] or None
    ) -> torch.Tensor:                  # Returns [B, T, output_dim]

        B, Lc, C = context.shape
        static1 = static_embed if static_embed.dim() == 3 else static_embed.unsqueeze(1)  # [B,1,S]

        preds = []

        # --- Memory attention over full sequence ---
        attn_out, attn_weights = self.memory_attn(context, context, context)  # [B, Lc, C], [B, Lc, Lc]

        # Average attention weights across heads (if multi-head)
        weights = attn_weights.mean(dim=1)  # [B, Lc, Lc] → [B, Lc]

        # Normalize weights across timesteps (optional but safer numerically)
        weights = torch.softmax(weights, dim=-1)  # [B, Lc]

        # Compute weighted sum of attn_out using weights
        attn_summary = torch.bmm(weights.unsqueeze(1), attn_out).squeeze(1)  # [B, 1, Lc] × [B, Lc, C] → [B, C]

        hidden = self.init_proj(attn_summary).unsqueeze(0)  # [1, B, C]
        # [1, B, C]

        # --- Initialize first decoder input (mean pooled context) ---
        inp = context.mean(dim=1, keepdim=True)                    # [B,1,C]

        for t in range(future_steps):
            if teacher_forcing_inputs is not None and t > 0:
                gt = teacher_forcing_inputs[:, t-1].unsqueeze(1)  # [B,1,output_dim]
                inp = self.tf_proj(gt).relu()

            # --- Run RNN step ---
            rnn_in = torch.cat([inp, static1], dim=-1)             # [B,1,C+S]
            out, hidden = self.rnn(rnn_in, hidden)                 # [B,1,C]
            pred = self.head(out)                                  # [B,1,output_dim]
            preds.append(pred.squeeze(1))                         # [B,output_dim]

            # --- Feedback for next input ---
            inp = self.tf_proj(pred).relu()

        return torch.stack(preds, dim=1)  # [B, T, output_dim]



# ───────────────────────── Transformer version ──────────────────────

def _expand_static(static: torch.Tensor, steps: int) -> torch.Tensor:
    # static: [B, D] or [B, 1, D]
    if static.dim() == 2:
        static = static.unsqueeze(1)
    return static.expand(-1, steps, -1)


class TransformerARDecoder(nn.Module):
    def __init__(
        self,
        context_dim: int,
        static_dim: int,
        output_dim: int,
        *,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 500,  # <-- Add this
        config: Optional[dict] = None  # <-- Allow config to be passed
    ):
        super().__init__()
        self.input_dim = context_dim + static_dim
        self.max_len = max_len  # ✅ Fix: define it here
        self.config = config or {}

        self.mem_proj = (
            nn.Linear(context_dim, self.input_dim)
            if context_dim != self.input_dim
            else nn.Identity()
        )

        self.tf_proj = nn.Sequential(
            nn.Linear(output_dim, context_dim),
            nn.ReLU()
        )

        self.start_token = nn.Parameter(torch.randn(1, 1, self.input_dim))

        layer = nn.TransformerDecoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=4 * self.input_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

        # Enhanced positional encoding support
        pos_encoding_type = self.config.get("pos_encoding_type", "sinusoidal")
        
        if pos_encoding_type == "adaptive":
            self.pos_enc = AdaptiveTemporalEncoding(
                embed_dim=self.input_dim,
                max_len=self.max_len,
                num_drug_types=self.config.get("num_bolus_drugs", 15)
            )
        else:
            self.pos_enc = PositionalEncoding(
                input_dim=self.input_dim,
                max_len=self.max_len,
                encoding_type=pos_encoding_type
            )

        self.head = nn.Linear(self.input_dim, output_dim)

        # final head from decoder output to prediction
        self.head = nn.Linear(self.input_dim, output_dim)

    def forward(
        self,
        context: torch.Tensor,
        static_embed: torch.Tensor,
        future_steps: int,
        teacher_forcing_inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: encoder outputs [B, L, context_dim]
            static_embed: static embedding [B, static_dim] or [B, 1, static_dim]
            future_steps: number of timesteps to predict
            teacher_forcing_inputs: ground-truth targets [B, T, output_dim]
            attention_mask: padding mask for memory [B, L] (True=keep, False=pad)

        Returns:
            preds: [B, future_steps, output_dim]
        """
        B = context.size(0)
        # prepare memory
        memory = self.mem_proj(context)  # [B,L,input_dim]
        preds = []  # list of [B, output_dim]

        for t in range(future_steps):
            if t == 0:
                # first step: just start token
                tgt = self.start_token.expand(B, 1, -1)
            else:
                # build autoregressive input
                if teacher_forcing_inputs is not None:
                    prev = teacher_forcing_inputs[:, :t]  # [B, t, output_dim]
                    ctx_in = self.tf_proj(prev)
                else:
                    prev = torch.stack(preds, dim=1)  # [B, t, output_dim]
                    ctx_in = self.tf_proj(prev)  # Only apply tf_proj to self-preds

                static_seq = _expand_static(static_embed, ctx_in.size(1))
                tgt = torch.cat([
                    self.start_token.expand(B, 1, -1),
                    torch.cat([ctx_in, static_seq], dim=-1)
                ], dim=1)  # [B, t+1, input_dim]

            # autoregressive mask to prevent attending to future in tgt
            T = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(T, T, device=tgt.device), 1).bool()

            # Apply positional encoding
            if isinstance(self.pos_enc, AdaptiveTemporalEncoding):
                # For adaptive encoding, we'd need bolus information from context
                # For now, use without drug-specific info in decoder
                tgt_with_pos = self.pos_enc(seq_len=tgt.size(1))[:, :tgt.size(1), :] + tgt
            else:
                tgt_with_pos = self.pos_enc(tgt)
            
            # decode
            dec_out = self.decoder(
                tgt_with_pos,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~attention_mask if attention_mask is not None else None,
            )  # [B, t+1, input_dim]

            # predict only the last step
            step_pred = self.head(dec_out[:, -1:])  # [B,1,output_dim]
            preds.append(step_pred.squeeze(1))  # [B, output_dim]

        return torch.stack(preds, dim=1)  # [B, T, output_dim]

