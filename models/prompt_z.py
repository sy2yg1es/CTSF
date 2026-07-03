"""
PromptZModulator — Drift-conditioned Gray-box Representation Modulation
========================================================================

Core equation:
    h'_l = h_l + gamma_t · mask_t ⊙ delta_h_l

Where:
    drift_state = DriftEncoder(hidden_summary, residual_stats)
    gamma_t     = ConfidenceGate(drift_state)      — scalar gate per channel
    mask_t      = SparseMaskHead(drift_state)       — per-channel sparse mask
    delta_h_l   = LowRankModulator(h_l, drift_state) — low-rank correction

Design constraints (from user feedback):
    - gamma bias init = -3  → sigmoid(-3) ≈ 0.047 → near-zero by default
    - mask bias init = -1.5 → sigmoid(-1.5) ≈ 0.18 → mostly off
    - up projection zero-init → delta starts at 0
    - scale output through tanh → bounded [-1, 1]
    - ratio clamp: ||delta|| / ||hidden|| ≤ max_delta_ratio
    - shape asserts for both BCDP and BCD layouts
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple


# ============================================================================
# Sub-modules
# ============================================================================

class DriftEncoder(nn.Module):
    """
    Encode hidden summary + residual statistics into a drift state vector.

    Inputs:
        hidden_summary : [B, C, D_hidden_in]
            PatchTST: 3*D (mean, std, last-first over patch axis)
            iTransformer: D (hidden itself)
        residual_stats : [C, 5]
            [error_mean, error_slope, error_std, signed_bias, steps_gap]
            Broadcast over batch.

    Output:
        drift_state : [B, C, D_drift]
    """

    def __init__(self, d_hidden_in: int, d_drift: int = 64, n_residual_features: int = 5):
        super().__init__()
        self.hidden_proj = nn.Linear(d_hidden_in, d_drift)
        self.residual_proj = nn.Linear(n_residual_features, d_drift)
        self.norm = nn.LayerNorm(d_drift)

    def forward(self, hidden_summary: Tensor, residual_stats: Tensor) -> Tensor:
        """
        Parameters
        ----------
        hidden_summary : [B, C, D_hidden_in]
        residual_stats : [C, 5]  — will be broadcast over batch

        Returns
        -------
        drift_state : [B, C, D_drift]
        """
        h_proj = self.hidden_proj(hidden_summary)            # [B, C, D_drift]
        r_proj = self.residual_proj(residual_stats)           # [C, D_drift]
        # Broadcast residual stats over batch
        r_proj = r_proj.unsqueeze(0).expand(h_proj.shape[0], -1, -1)
        drift_state = self.norm(h_proj + r_proj)
        return drift_state                                    # [B, C, D_drift]


class ConfidenceGate(nn.Module):
    """
    Produce a scalar confidence gate per channel.

    drift_state [B, C, D_drift] → gamma [B, C, 1]

    Initialized with bias = -4 → sigmoid(-4) ≈ 0.018.
    Prompt-Z is near-no-op by default.
    """

    def __init__(self, d_drift: int, init_bias: float = -4.0):
        super().__init__()
        self.linear = nn.Linear(d_drift, 1)
        # Zero-init weight, negative bias → default gamma ≈ 0
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, init_bias)

    def forward(self, drift_state: Tensor) -> Tensor:
        """Returns gamma [B, C, 1] in [0, 1]."""
        return torch.sigmoid(self.linear(drift_state))       # [B, C, 1]


class SparseMaskHead(nn.Module):
    """
    Produce a per-channel mask.

    drift_state [B, C, D_drift] → mask [B, C, 1]

    Channel-level sparsity: decides which channels get modulated.
    L1-regularizable via mask.abs().mean().

    Initialized with bias = -2 → sigmoid(-2) ≈ 0.12, mostly off.
    """

    def __init__(self, d_drift: int, init_bias: float = -2.0):
        super().__init__()
        self.linear = nn.Linear(d_drift, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, init_bias)

    def forward(self, drift_state: Tensor) -> Tensor:
        """Returns mask [B, C, 1] in [0, 1]."""
        return torch.sigmoid(self.linear(drift_state))       # [B, C, 1]


class LowRankModulator(nn.Module):
    """
    Low-rank representation modulation conditioned on drift state.

    hidden [*, D] → delta_h [*, D]

    Structure:
        down = Linear(D, rank)
        up = Linear(rank, D)          # zero-init
        scale = Linear(D_drift, D)    # small-init, tanh-bounded

    delta = tanh(scale(drift_state)) * up(GELU(down(hidden)))

    Parameters
    ----------
    d_model : int — hidden dimension D
    d_drift : int — drift state dimension
    rank : int — bottleneck rank (default 8)
    """

    def __init__(self, d_model: int, d_drift: int = 64, rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        self.down = nn.Linear(d_model, rank)
        self.up = nn.Linear(rank, d_model)
        self.scale_proj = nn.Linear(d_drift, d_model)

        # Zero-init up → delta starts at 0
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # Small-init scale → bounded perturbation
        nn.init.normal_(self.scale_proj.weight, std=0.01)
        nn.init.zeros_(self.scale_proj.bias)

    def forward(self, hidden: Tensor, drift_state: Tensor) -> Tensor:
        """
        Parameters
        ----------
        hidden      : [B, C, *, D]  — backbone hidden states
        drift_state : [B, C, D_drift]

        Returns
        -------
        delta_h : same shape as hidden
        """
        # Scale from drift state: [B, C, D_drift] → [B, C, D]
        scale = torch.tanh(self.scale_proj(drift_state))     # [B, C, D], bounded [-1, 1]

        # Low-rank projection
        h_down = F.gelu(self.down(hidden))                   # [B, C, *, rank]
        h_up = self.up(h_down)                               # [B, C, *, D]

        # Expand scale to match hidden shape
        # hidden may be [B,C,D] or [B,C,P,D]; scale is [B,C,D]
        while scale.dim() < h_up.dim():
            scale = scale.unsqueeze(-2)                      # [B, C, 1..., D]

        delta_h = scale * h_up
        return delta_h


# ============================================================================
# Top-level PromptZModulator
# ============================================================================

class PromptZModulator(nn.Module):
    """
    Prompt-driven gray-box internal representation modulator.

    Composes DriftEncoder + ConfidenceGate + SparseMaskHead + LowRankModulator.

    Parameters
    ----------
    d_model : int
        Backbone hidden dimension D.
    hidden_layout : str
        'BCDP' for PatchTST (4D hidden), 'BCD' for iTransformer (3D hidden).
    d_drift : int
        Drift state dimension.
    rank : int
        Low-rank bottleneck rank.
    gamma_init_bias : float
        ConfidenceGate bias init (default -4 → near-zero gate).
    mask_init_bias : float
        SparseMaskHead bias init (default -2 → mostly off).
    max_delta_ratio : float
        Ratio clamp: ||delta_h|| / ||hidden|| ≤ max_delta_ratio.
        Applied per (batch, channel) pair.
    """

    def __init__(
        self,
        d_model: int,
        hidden_layout: str = 'BCDP',
        d_drift: int = 64,
        rank: int = 8,
        gamma_init_bias: float = -3.0,
        mask_init_bias: float = -1.5,
        max_delta_ratio: float = 0.05,
    ):
        super().__init__()
        assert hidden_layout in ('BCDP', 'BCD'), \
            f"hidden_layout must be 'BCDP' or 'BCD', got '{hidden_layout}'"

        self.d_model = d_model
        self.hidden_layout = hidden_layout
        self.d_drift = d_drift
        self.max_delta_ratio = max_delta_ratio

        # Hidden summary input dimension depends on layout
        if hidden_layout == 'BCDP':
            d_hidden_in = 3 * d_model
        else:
            d_hidden_in = d_model

        self.drift_encoder = DriftEncoder(d_hidden_in, d_drift)
        self.confidence_gate = ConfidenceGate(d_drift, init_bias=gamma_init_bias)
        self.sparse_mask = SparseMaskHead(d_drift, init_bias=mask_init_bias)
        self.low_rank_mod = LowRankModulator(d_model, d_drift, rank)

    def _hidden_summary(self, hidden: Tensor) -> Tensor:
        """
        Extract summary features from hidden for DriftEncoder.

        Parameters
        ----------
        hidden : [B, C, D, P] (BCDP) or [B, C, D] (BCD)

        Returns
        -------
        summary : [B, C, D_hidden_in]
        """
        if self.hidden_layout == 'BCDP':
            h_mean = hidden.mean(dim=-1)                     # [B, C, D]
            h_std = hidden.std(dim=-1, unbiased=False)       # [B, C, D]
            h_delta = hidden[..., -1] - hidden[..., 0]      # [B, C, D]
            return torch.cat([h_mean, h_std, h_delta], dim=-1)
        else:
            return hidden

    def _ratio_clamp(self, delta_h: Tensor, hidden: Tensor) -> Tensor:
        """
        Ratio clamp: scale delta_h so that ||delta_h|| / ||hidden|| ≤ max_delta_ratio.
        Computed per (batch, channel) pair on the full [D,P] or [D] energy.

        Differentiable: uses multiplicative scaling, not hard clamp.
        """
        # Flatten spatial dims: [B, C, D, P] → [B, C, D*P] or [B, C, D] → [B, C, D]
        delta_flat = delta_h.flatten(2)                      # [B, C, *]
        hidden_flat = hidden.detach().flatten(2)             # [B, C, *]

        delta_norm = delta_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # [B, C, 1]
        hidden_norm = hidden_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8) # [B, C, 1]

        max_allowed = self.max_delta_ratio * hidden_norm     # [B, C, 1]
        # scale_factor ≤ 1.0 always; differentiable
        scale_factor = (max_allowed / delta_norm).clamp(max=1.0)

        # Reshape scale_factor to match delta_h shape
        if self.hidden_layout == 'BCDP':
            scale_factor = scale_factor.unsqueeze(-1)        # [B, C, 1, 1]

        return delta_h * scale_factor

    def forward(
        self,
        hidden: Tensor,
        residual_stats: Tensor,
        gamma_floor: float = 0.0,
        mask_floor: float = 0.0,
    ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, float]]:
        """
        Apply drift-conditioned modulation to backbone hidden states.

        Parameters
        ----------
        hidden          : [B, C, D, P] or [B, C, D] — backbone hidden states
        residual_stats  : [C, 5] — from ResidualTracker.get_stats()
        gamma_floor     : training-only lower bound mixed into gamma
        mask_floor      : training-only lower bound mixed into mask

        Returns
        -------
        hidden_mod   : same shape as hidden
        reg_tensors  : dict of LIVE (differentiable) tensors for loss computation:
            'effective_delta_ratio' : scalar — ||gamma*mask*delta|| / ||hidden||
            'gamma_mean'           : scalar — mean of gamma (for noop penalty)
            'mask_mean'            : scalar — mean of mask (for L1 sparsity)
        diagnostics  : dict of detached floats for logging
        """
        # Shape validation
        if self.hidden_layout == 'BCDP':
            assert hidden.dim() == 4, \
                f"Expected 4D hidden [B,C,D,P] for BCDP layout, got {hidden.shape}"
            B, C, D, P = hidden.shape
            assert D == self.d_model, \
                f"D mismatch: hidden D={D}, expected {self.d_model}"
        else:
            assert hidden.dim() == 3, \
                f"Expected 3D hidden [B,C,D] for BCD layout, got {hidden.shape}"
            B, C, D = hidden.shape
            assert D == self.d_model

        # 1. Drift state
        summary = self._hidden_summary(hidden)
        drift_state = self.drift_encoder(summary, residual_stats)

        # 2. Confidence gate
        gamma = self.confidence_gate(drift_state)             # [B, C, 1]
        if gamma_floor > 0:
            gamma = gamma_floor + (1.0 - gamma_floor) * gamma

        # 3. Sparse mask (channel-level)
        mask = self.sparse_mask(drift_state)                  # [B, C, 1]
        if mask_floor > 0:
            mask = mask_floor + (1.0 - mask_floor) * mask

        # 4. Low-rank modulation + ratio clamp
        if self.hidden_layout == 'BCDP':
            h_work = hidden.permute(0, 1, 3, 2)              # [B, C, P, D]
            delta_h = self.low_rank_mod(h_work, drift_state)  # [B, C, P, D]
            delta_h = delta_h.permute(0, 1, 3, 2)            # [B, C, D, P]
        else:
            delta_h = self.low_rank_mod(hidden, drift_state)  # [B, C, D]

        # Ratio clamp (differentiable)
        delta_h = self._ratio_clamp(delta_h, hidden)

        # 5. Apply gated sparse modulation
        if self.hidden_layout == 'BCDP':
            gamma_4d = gamma.unsqueeze(-1)                    # [B, C, 1, 1]
            mask_4d = mask.unsqueeze(-1)                      # [B, C, 1, 1]
            applied_delta = gamma_4d * mask_4d * delta_h
        else:
            applied_delta = gamma * mask * delta_h

        hidden_mod = hidden + applied_delta

        # Shape assertion
        assert hidden_mod.shape == hidden.shape, \
            f"Shape mismatch: hidden_mod {hidden_mod.shape} != hidden {hidden.shape}"

        # --- LIVE reg tensors (differentiable, for loss computation) ---
        applied_flat = applied_delta.flatten(2)              # [B, C, *]
        hidden_flat = hidden.detach().flatten(2)
        applied_norm = applied_flat.norm(dim=-1)             # [B, C]
        hidden_norm = hidden_flat.norm(dim=-1).clamp(min=1e-8)
        effective_delta_ratio = (applied_norm / hidden_norm).mean()  # scalar, has grad

        reg_tensors = {
            "effective_delta_ratio": effective_delta_ratio,   # ||gamma*mask*delta|| / ||h||
            "gamma_mean": gamma.mean(),                      # live scalar
            "mask_mean": mask.mean(),                         # live scalar
        }

        # --- Detached diagnostics for logging ---
        with torch.no_grad():
            raw_delta_flat = delta_h.flatten(2)
            raw_delta_norm = raw_delta_flat.norm(dim=-1).mean()
            h_norm_scalar = hidden_norm.mean()
            diag = {
                "gamma_mean": gamma.mean().item(),
                "gamma_std": gamma.std().item(),
                "gamma_min": gamma.min().item(),
                "gamma_max": gamma.max().item(),
                "mask_ratio": (mask > 0.5).float().mean().item(),
                "mask_mean": mask.mean().item(),
                "raw_delta_norm": raw_delta_norm.item(),
                "hidden_norm": h_norm_scalar.item(),
                "raw_delta_to_hidden_ratio": (raw_delta_norm / h_norm_scalar.clamp(min=1e-8)).item(),
                "effective_delta_ratio": effective_delta_ratio.item(),
                "applied_delta_norm": applied_norm.mean().item(),
                "gamma_floor": float(gamma_floor),
                "mask_floor": float(mask_floor),
            }

        return hidden_mod, reg_tensors, diag

    def get_gate_params(self):
        """Return parameters eligible for Mode 1 online calibration."""
        return [self.confidence_gate.linear.bias]

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, layout={self.hidden_layout}, "
            f"d_drift={self.d_drift}, rank={self.low_rank_mod.rank}, "
            f"max_delta_ratio={self.max_delta_ratio}"
        )
