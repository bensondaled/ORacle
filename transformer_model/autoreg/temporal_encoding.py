"""
Enhanced Temporal Position Encoding for Medical Time Series
===========================================================

This module implements advanced positional encoding techniques specifically
designed for medical time series data, particularly for intraoperative 
vital sign prediction with drug administration events.

Key Features:
- Multi-scale temporal encoding (short/medium/long-term patterns)
- Drug-aware positional encoding with bolus event markers
- Physiological rhythm encoding (heartbeat, respiratory cycles)
- Relative temporal encoding (time since last bolus)
- Learnable frequency adaptation for medical domains
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MedicalTemporalEncoding(nn.Module):
    """
    Advanced temporal position encoding for medical time series.
    
    Combines multiple encoding strategies:
    1. Multi-scale sinusoidal encoding (short, medium, long-term patterns)
    2. Drug-aware encoding with bolus event markers
    3. Physiological rhythm encoding
    4. Learnable frequency adaptation
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        max_len: int = 5000,
        num_bolus_drugs: int = 15,
        use_drug_encoding: bool = True,
        use_physio_encoding: bool = True,
        use_relative_encoding: bool = True,
        learnable_frequencies: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_bolus_drugs = num_bolus_drugs
        self.use_drug_encoding = use_drug_encoding
        self.use_physio_encoding = use_physio_encoding
        self.use_relative_encoding = use_relative_encoding
        self.learnable_frequencies = learnable_frequencies
        
        # Divide embedding dimensions across different encoding types
        self.base_dim = embed_dim // 2  # Base temporal encoding
        self.drug_dim = embed_dim // 4 if use_drug_encoding else 0  # Drug-aware encoding
        self.physio_dim = embed_dim - self.base_dim - self.drug_dim  # Physiological encoding
        
        logger.info(f"Medical Temporal Encoding: base_dim={self.base_dim}, "
                   f"drug_dim={self.drug_dim}, physio_dim={self.physio_dim}")
        
        # 1. Multi-scale Base Temporal Encoding
        self._init_base_encoding()
        
        # 2. Drug-aware Encoding
        if self.use_drug_encoding:
            self._init_drug_encoding()
            
        # 3. Physiological Rhythm Encoding
        if self.use_physio_encoding:
            self._init_physio_encoding()
            
        # 4. Relative Time Encoding
        if self.use_relative_encoding:
            self._init_relative_encoding()
    
    def _init_base_encoding(self):
        """Initialize multi-scale base temporal encoding."""
        
        # Different frequency scales for short/medium/long-term patterns
        self.short_scale = 100.0    # ~1-2 minute patterns (vital sign fluctuations)
        self.medium_scale = 1000.0  # ~10-20 minute patterns (drug effects)
        self.long_scale = 10000.0   # ~1-2 hour patterns (surgical phases)
        
        if self.learnable_frequencies:
            # Make frequency scales learnable
            self.short_freq = nn.Parameter(torch.tensor(self.short_scale))
            self.medium_freq = nn.Parameter(torch.tensor(self.medium_scale))
            self.long_freq = nn.Parameter(torch.tensor(self.long_scale))
        
        # Pre-compute sinusoidal encodings for each scale
        self.register_buffer('base_encoding', self._create_multiscale_encoding())
    
    def _create_multiscale_encoding(self):
        """Create multi-scale sinusoidal encoding."""
        
        # Split base dimensions across different scales
        short_dim = self.base_dim // 3
        medium_dim = self.base_dim // 3
        long_dim = self.base_dim - short_dim - medium_dim
        
        position = torch.arange(self.max_len, dtype=torch.float).unsqueeze(1)
        encoding = torch.zeros(self.max_len, self.base_dim)
        
        # Short-term encoding (high frequency)
        if short_dim > 0:
            div_term = torch.exp(torch.arange(0, short_dim, 2, dtype=torch.float) * 
                               (-math.log(self.short_scale) / short_dim))
            encoding[:, 0:short_dim:2] = torch.sin(position * div_term)
            if short_dim > 1:
                encoding[:, 1:short_dim:2] = torch.cos(position * div_term[:short_dim//2])
        
        # Medium-term encoding (medium frequency)  
        if medium_dim > 0:
            start_idx = short_dim
            div_term = torch.exp(torch.arange(0, medium_dim, 2, dtype=torch.float) * 
                               (-math.log(self.medium_scale) / medium_dim))
            encoding[:, start_idx:start_idx+medium_dim:2] = torch.sin(position * div_term)
            if medium_dim > 1:
                encoding[:, start_idx+1:start_idx+medium_dim:2] = torch.cos(position * div_term[:medium_dim//2])
        
        # Long-term encoding (low frequency)
        if long_dim > 0:
            start_idx = short_dim + medium_dim
            div_term = torch.exp(torch.arange(0, long_dim, 2, dtype=torch.float) * 
                               (-math.log(self.long_scale) / long_dim))
            encoding[:, start_idx:start_idx+long_dim:2] = torch.sin(position * div_term)
            if long_dim > 1:
                encoding[:, start_idx+1:start_idx+long_dim:2] = torch.cos(position * div_term[:long_dim//2])
        
        return encoding.unsqueeze(0)  # [1, max_len, base_dim]
    
    def _init_drug_encoding(self):
        """Initialize drug-aware encoding components."""
        
        # Learnable drug-specific embeddings
        self.drug_embeddings = nn.Embedding(self.num_bolus_drugs + 1, self.drug_dim)  # +1 for no-drug
        
        # Bolus event marker (special encoding for timesteps with bolus)
        self.bolus_marker = nn.Parameter(torch.randn(1, 1, self.drug_dim) * 0.02)
        
        # Time-since-bolus encoding (exponential decay)
        self.bolus_decay = nn.Parameter(torch.tensor(0.1))  # Learnable decay rate
    
    def _init_physio_encoding(self):
        """Initialize physiological rhythm encoding."""
        
        if self.physio_dim <= 0:
            return
            
        # Physiological frequency ranges (cycles per minute)
        self.heart_rate_freq = 60.0 / 60.0    # ~60 BPM = 1 Hz
        self.resp_rate_freq = 15.0 / 60.0     # ~15 breaths/min = 0.25 Hz
        self.bp_variation_freq = 0.1          # ~6 cycles/min Mayer waves
        
        if self.learnable_frequencies:
            self.heart_freq = nn.Parameter(torch.tensor(self.heart_rate_freq))
            self.resp_freq = nn.Parameter(torch.tensor(self.resp_rate_freq))
            self.bp_freq = nn.Parameter(torch.tensor(self.bp_variation_freq))
        
        # Physiological encoding projection
        self.physio_proj = nn.Linear(3, self.physio_dim)  # Project 3 physio rhythms
    
    def _init_relative_encoding(self):
        """Initialize relative time encoding components."""
        
        # Project relative time features to embedding space
        self.relative_proj = nn.Linear(3, self.embed_dim // 4)  # time_since_bolus, time_to_future, global_time
        
    def forward(
        self, 
        seq_len: int,
        bolus_mask: Optional[torch.Tensor] = None,  # [B, L] - indicates bolus at each timestep
        last_bolus_drugs: Optional[torch.Tensor] = None,  # [B, L, num_drugs] - drug amounts
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate enhanced temporal position encoding.
        
        Args:
            seq_len: Sequence length
            bolus_mask: [B, L] Boolean mask indicating bolus events
            last_bolus_drugs: [B, L, num_drugs] Drug administration amounts
            device: Target device
            
        Returns:
            pos_encoding: [B, L, embed_dim] or [1, L, embed_dim] if no batch-specific info
        """
        
        if device is None:
            device = next(self.parameters()).device
        
        batch_size = bolus_mask.size(0) if bolus_mask is not None else 1
        
        # 1. Base multi-scale temporal encoding
        base_pos = self.base_encoding[:, :seq_len, :].to(device)  # [1, L, base_dim]
        
        # Apply learnable frequency scaling if enabled
        if self.learnable_frequencies:
            # This is a simplified approach - in practice you'd recompute the encoding
            # with the learned frequencies, but for efficiency we'll use a scaling approach
            freq_scale = torch.sigmoid(torch.stack([self.short_freq, self.medium_freq, self.long_freq]))
            freq_scale = freq_scale / freq_scale.sum()  # Normalize
            # Apply frequency scaling (simplified - could be more sophisticated)
            base_pos = base_pos * freq_scale.mean().item()
        
        encoding_parts = [base_pos.expand(batch_size, -1, -1)]  # [B, L, base_dim]
        
        # 2. Drug-aware encoding
        if self.use_drug_encoding and self.drug_dim > 0:
            drug_encoding = self._compute_drug_encoding(
                seq_len, batch_size, bolus_mask, last_bolus_drugs, device
            )
            encoding_parts.append(drug_encoding)  # [B, L, drug_dim]
        
        # 3. Physiological rhythm encoding  
        if self.use_physio_encoding and self.physio_dim > 0:
            physio_encoding = self._compute_physio_encoding(seq_len, batch_size, device)
            encoding_parts.append(physio_encoding)  # [B, L, physio_dim]
        
        # 4. Relative time encoding
        if self.use_relative_encoding:
            relative_encoding = self._compute_relative_encoding(
                seq_len, batch_size, bolus_mask, device
            )
            encoding_parts.append(relative_encoding)  # [B, L, embed_dim//4]
        
        # Concatenate all encoding components
        full_encoding = torch.cat(encoding_parts, dim=-1)  # [B, L, embed_dim]
        
        # Ensure correct output dimension
        if full_encoding.size(-1) != self.embed_dim:
            # Project to correct dimension if needed
            if not hasattr(self, 'final_proj'):
                self.final_proj = nn.Linear(full_encoding.size(-1), self.embed_dim).to(device)
            full_encoding = self.final_proj(full_encoding)
        
        return full_encoding
    
    def _compute_drug_encoding(
        self, 
        seq_len: int, 
        batch_size: int,
        bolus_mask: Optional[torch.Tensor],
        last_bolus_drugs: Optional[torch.Tensor], 
        device: torch.device
    ) -> torch.Tensor:
        """Compute drug-aware positional encoding."""
        
        drug_pos = torch.zeros(batch_size, seq_len, self.drug_dim, device=device)
        
        if bolus_mask is not None and last_bolus_drugs is not None:
            # Get dominant drug at each timestep
            drug_amounts = last_bolus_drugs  # [B, L, num_drugs]
            dominant_drugs = torch.argmax(drug_amounts, dim=-1)  # [B, L]
            
            # Where no drug is given, use drug index 0 (no-drug embedding)
            has_drug = drug_amounts.sum(dim=-1) > 0  # [B, L]
            dominant_drugs = dominant_drugs * has_drug.long()  # Zero out where no drug
            
            # Get drug embeddings
            drug_emb = self.drug_embeddings(dominant_drugs)  # [B, L, drug_dim]
            
            # Add bolus event markers
            bolus_marker_expanded = self.bolus_marker.expand(batch_size, seq_len, -1)
            drug_pos = drug_emb + bolus_marker_expanded * bolus_mask.unsqueeze(-1).float()
            
            # Add time-since-bolus decay
            if hasattr(self, 'bolus_decay'):
                time_positions = torch.arange(seq_len, device=device).float()
                time_matrix = time_positions.unsqueeze(0) - time_positions.unsqueeze(1)  # [L, L]
                time_matrix = torch.clamp(time_matrix, min=0)  # Only forward time
                
                # For each timestep, find time since last bolus
                for b in range(batch_size):
                    bolus_positions = torch.where(bolus_mask[b])[0]
                    if len(bolus_positions) > 0:
                        for t in range(seq_len):
                            # Find most recent bolus before timestep t
                            recent_bolus = bolus_positions[bolus_positions <= t]
                            if len(recent_bolus) > 0:
                                time_since_bolus = t - recent_bolus[-1].item()
                                decay_factor = torch.exp(-self.bolus_decay * time_since_bolus)
                                drug_pos[b, t] *= decay_factor
        
        return drug_pos
    
    def _compute_physio_encoding(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute physiological rhythm encoding."""
        
        # Time steps (assuming 1-minute resolution)
        time_steps = torch.arange(seq_len, dtype=torch.float, device=device)
        
        # Get physiological frequencies
        heart_freq = self.heart_freq if self.learnable_frequencies else self.heart_rate_freq
        resp_freq = self.resp_freq if self.learnable_frequencies else self.resp_rate_freq  
        bp_freq = self.bp_freq if self.learnable_frequencies else self.bp_variation_freq
        
        # Generate physiological rhythms
        heart_rhythm = torch.sin(2 * math.pi * heart_freq * time_steps)     # Heart rate rhythm
        resp_rhythm = torch.sin(2 * math.pi * resp_freq * time_steps)       # Respiratory rhythm
        bp_rhythm = torch.sin(2 * math.pi * bp_freq * time_steps)           # BP variation rhythm
        
        # Stack rhythms and project to physio_dim
        physio_features = torch.stack([heart_rhythm, resp_rhythm, bp_rhythm], dim=-1)  # [L, 3]
        physio_features = physio_features.unsqueeze(0).expand(batch_size, -1, -1)     # [B, L, 3]
        
        # Project to embedding space
        physio_encoding = self.physio_proj(physio_features)  # [B, L, physio_dim]
        
        return physio_encoding
    
    def _compute_relative_encoding(
        self, 
        seq_len: int, 
        batch_size: int,
        bolus_mask: Optional[torch.Tensor], 
        device: torch.device
    ) -> torch.Tensor:
        """Compute relative time encoding (CAUSAL - no future information)."""
        
        # Global normalized time position
        global_time = torch.arange(seq_len, dtype=torch.float, device=device) / seq_len
        global_time = global_time.unsqueeze(0).expand(batch_size, -1)  # [B, L]
        
        # Time since last bolus (normalized) - CAUSAL
        time_since_bolus = torch.zeros(batch_size, seq_len, device=device)
        
        # REMOVED: time_to_bolus - THIS WOULD LEAK FUTURE INFORMATION!
        # Replace with time-to-end-of-sequence for planning horizon
        time_to_end = torch.zeros(batch_size, seq_len, device=device)
        
        if bolus_mask is not None:
            for b in range(batch_size):
                bolus_positions = torch.where(bolus_mask[b])[0]
                
                for t in range(seq_len):
                    # Time since last bolus (CAUSAL - only looks backward)
                    past_bolus = bolus_positions[bolus_positions <= t]
                    if len(past_bolus) > 0:
                        time_since_bolus[b, t] = (t - past_bolus[-1].item()) / seq_len
                    else:
                        time_since_bolus[b, t] = 1.0  # No previous bolus
                    
                    # Time remaining in sequence (planning horizon)
                    time_to_end[b, t] = (seq_len - 1 - t) / seq_len
        else:
            # If no bolus mask, just use global time patterns
            for t in range(seq_len):
                time_to_end[:, t] = (seq_len - 1 - t) / seq_len
        
        # Stack relative time features (CAUSAL ONLY)
        relative_features = torch.stack([
            time_since_bolus, time_to_end, global_time
        ], dim=-1)  # [B, L, 3]
        
        # Project to embedding space
        relative_encoding = self.relative_proj(relative_features)  # [B, L, embed_dim//4]
        
        return relative_encoding


class AdaptiveTemporalEncoding(nn.Module):
    """
    Simplified adaptive version that can be easily integrated.
    Uses learnable frequency components with drug-aware scaling.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000, num_drug_types: int = 15):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_drug_types = num_drug_types
        
        # Learnable frequency scales
        self.base_frequencies = nn.Parameter(
            torch.logspace(-2, 2, embed_dim // 2)  # Range from 0.01 to 100
        )
        
        # Drug-specific frequency modulation
        self.drug_freq_modulation = nn.Parameter(
            torch.ones(num_drug_types, embed_dim // 2) * 0.1
        )
        
        # Bolus event embedding
        self.bolus_embedding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(
        self, 
        seq_len: int, 
        bolus_mask: Optional[torch.Tensor] = None,
        dominant_drugs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            seq_len: Length of sequence
            bolus_mask: [B, L] Boolean mask for bolus events
            dominant_drugs: [B, L] Drug indices for each timestep
            
        Returns:
            encoding: [B, L, embed_dim] or [1, L, embed_dim]
        """
        
        device = self.base_frequencies.device
        batch_size = bolus_mask.size(0) if bolus_mask is not None else 1
        
        # FIXED: Causal drug frequency modulation - only use past/present information
        if dominant_drugs is not None:
            # Generate position-specific encodings with causal drug modulation
            encoding_parts = []
            
            for t in range(seq_len):
                # Only use drug information up to current timestep (CAUSAL)
                past_drugs = dominant_drugs[:, :t+1]  # [B, t+1] - only past/present
                
                if t > 0 and past_drugs.numel() > 0:
                    # Get drug modulation weights for past/present drugs only
                    past_drug_weights = self.drug_freq_modulation[past_drugs]  # [B, t+1, embed_dim//2]
                    # Average only over past/present drug effects
                    causal_drug_modulation = past_drug_weights.mean(dim=[0, 1])  # [embed_dim//2]
                else:
                    # No past drugs or first timestep - use base frequencies
                    causal_drug_modulation = torch.zeros(self.embed_dim // 2, device=device)
                
                # Position-specific frequency modulation
                pos_freqs = self.base_frequencies * (1 + causal_drug_modulation)
                pos_freqs = pos_freqs.unsqueeze(0)  # [1, embed_dim//2]
                
                # Generate encoding for this specific position
                pos_tensor = torch.tensor([float(t)], device=device).unsqueeze(1)  # [1, 1]
                angles = pos_tensor * pos_freqs  # [1, embed_dim//2]
                sin_enc = torch.sin(angles)  # [1, embed_dim//2]
                cos_enc = torch.cos(angles)  # [1, embed_dim//2]
                
                # Interleave sin and cos for this position
                pos_encoding = torch.zeros(1, self.embed_dim, device=device)
                pos_encoding[:, 0::2] = sin_enc
                pos_encoding[:, 1::2] = cos_enc[:, :pos_encoding[:, 1::2].size(1)]
                
                encoding_parts.append(pos_encoding)
            
            # Stack all position encodings
            encoding = torch.cat(encoding_parts, dim=0)  # [seq_len, embed_dim]
            
        else:
            # No drug information - use standard sinusoidal encoding
            position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
            freqs = self.base_frequencies.unsqueeze(0)  # [1, embed_dim//2]
            
            # Generate sinusoidal encoding
            angles = position * freqs  # [seq_len, embed_dim//2]
            sin_enc = torch.sin(angles)
            cos_enc = torch.cos(angles)
            
            # Interleave sin and cos
            encoding = torch.zeros(seq_len, self.embed_dim, device=device)
            encoding[:, 0::2] = sin_enc
            encoding[:, 1::2] = cos_enc[:, :encoding[:, 1::2].size(1)]
        
        # Expand for batch
        encoding = encoding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, embed_dim]
        
        # Add bolus event markers (CAUSAL - only affects current timestep)
        if bolus_mask is not None:
            bolus_contrib = self.bolus_embedding * bolus_mask.unsqueeze(-1).float()
            encoding = encoding + bolus_contrib
        
        return encoding