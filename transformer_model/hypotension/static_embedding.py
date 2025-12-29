# static_embedding.py
import torch
import torch.nn as nn
from typing import Dict, Optional

def get_static_embedder(mode: str, config: dict):
    """
    Returns a static embedder module based on the selected combination mode.
    """
    if mode == "tft":
        return StaticEmbeddingTFT(config)
    elif mode == "sum":
        return StaticEmbeddingSum(config)
    elif mode == "concat":
        return StaticEmbeddingConcat(config)
    elif mode == "none":
        return None
    else:
        raise ValueError(f"Unknown static_combine_mode: {mode}")


class StaticEmbeddingTFT(nn.Module):
    """
    Static embedding using projection layers + gated combination inspired by TFT.
    """
    def __init__(self, config: dict):
        super().__init__()
        embed_dim = config["static_embed_dim"]
        num_numerical = len(config.get("static_numericals", []))
        cat_vocabs = config.get("vocabs", {})

        self.num_proj = nn.Linear(num_numerical, embed_dim) if num_numerical > 0 else None
        self.cat_embeds = nn.ModuleDict({
            k: nn.Embedding(vocab_size + 1, embed_dim)
            for k, vocab_size in cat_vocabs.items()
        })

        input_dim = embed_dim * (int(num_numerical > 0) + len(cat_vocabs))
        self.context_gate = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, static_cat: dict[str, torch.Tensor], static_num: Optional[torch.Tensor]):
        out = []
        if self.num_proj and static_num is not None:
            out.append(self.num_proj(static_num))
        for k, emb in self.cat_embeds.items():
            out.append(emb(static_cat[k]))
        combined = torch.cat(out, dim=-1)
        return self.context_gate(combined)

    @property
    def output_dim(self):
        return self.context_gate[0].out_features


class StaticEmbedding(nn.Module):
    """
    Base static embedding for sum/concat modes.
    """
    def __init__(self, config: Dict, combine_mode: str="concat"):
        super().__init__()
        self.embed_dim = config["static_embed_dim"]
        num_num   = len(config.get("static_numericals", []))
        cat_vocabs= config.get("vocabs", {})

        self.combine_mode = combine_mode
        self.linear_num   = nn.Linear(num_num, self.embed_dim) if num_num > 0 else None
        self.cat_embeds   = nn.ModuleDict({
            k: nn.Embedding(vsz + 1, self.embed_dim, padding_idx=0)
            for k, vsz in cat_vocabs.items()
        })
        
        # Add patient_id embedding
        patient_vocab_size = config.get("patient_vocab_size", 50000) # Default to a large number
        self.patient_embed = nn.Embedding(patient_vocab_size + 1, self.embed_dim, padding_idx=0)

        for emb in self.cat_embeds.values():
            nn.init.normal_(emb.weight, 0, 0.02)
        if self.linear_num:
            nn.init.normal_(self.linear_num.weight, 0, 0.02)
            nn.init.zeros_(self.linear_num.bias)
        nn.init.normal_(self.patient_embed.weight, 0, 0.02)

    def forward(self, static_cat: Dict[str, torch.Tensor], static_num: Optional[torch.Tensor], patient_id: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        embeds = [emb(static_cat[k]) for k, emb in self.cat_embeds.items()]
        if self.linear_num and static_num is not None:
            embeds.append(self.linear_num(static_num))
        if patient_id is not None:
            embeds.append(self.patient_embed(patient_id))
        if not embeds:
            return None
        if self.combine_mode == 'sum':
            return torch.stack(embeds, dim=0).sum(dim=0)
        return torch.cat(embeds, dim=-1)

    @property
    def output_dim(self) -> int:
        if self.combine_mode == 'sum':
            return self.embed_dim
        num_cat = len(self.cat_embeds)
        num_num = 1 if self.linear_num is not None else 0
        return self.embed_dim * (num_cat + num_num)


class StaticEmbeddingSum(StaticEmbedding):
    def __init__(self, config: dict):
        super().__init__(config, combine_mode='sum')


class StaticEmbeddingConcat(StaticEmbedding):
    def __init__(self, config: dict):
        super().__init__(config, combine_mode='concat')
