"""
PromptZTSF — Top-level Prompt-Z Integration Module
====================================================

Architecture:
    X_t → BackboneAdapter.encode_until_hook(X_t) → hidden [B,C,D,P] or [B,C,D]
        → PromptZModulator(hidden, residual_stats)  → hidden_mod
        → BackboneAdapter.decode_from_hook(hidden_mod) → Y_hat [B,pred_len,C]

Completely independent from ContinualPromptTSF / MoE / prompt_pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

from models.backbone_adapter import BackboneAdapter
from models.prompt_z import PromptZModulator
from core.residual_tracker import ResidualTracker


class PromptZTSF(nn.Module):
    """
    Top-level Prompt-Z forecasting module.

    Components
    ----------
    backbone_adapter : BackboneAdapter (frozen)
    prompt_z         : PromptZModulator (trainable or frozen depending on mode)
    residual_tracker : ResidualTracker (stateful, no grad)
    """

    def __init__(
        self,
        backbone_adapter: BackboneAdapter,
        prompt_z: PromptZModulator,
        residual_tracker: ResidualTracker,
    ) -> None:
        super().__init__()
        self.backbone_adapter = backbone_adapter
        self.prompt_z = prompt_z
        self.residual_tracker = residual_tracker

        # Verify layout consistency
        assert backbone_adapter.hidden_layout == prompt_z.hidden_layout, (
            f"Layout mismatch: adapter={backbone_adapter.hidden_layout}, "
            f"prompt_z={prompt_z.hidden_layout}"
        )

    @torch.no_grad()
    def forward(self, X_t: Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        """
        Streaming inference path (all under no_grad).

        Parameters
        ----------
        X_t : [B, seq_len, C]

        Returns
        -------
        Y_hat       : [B, pred_len, C]
        hidden      : [B, C, D, P] or [B, C, D] — for caching
        diagnostics : dict with gamma/mask/delta stats
        """
        hidden, means, stdev = self.backbone_adapter.encode_until_hook(X_t)

        residual_stats = self.residual_tracker.get_stats()
        stats_tensor = self._pack_stats(residual_stats, hidden.device)

        hidden_mod, _reg, diagnostics = self.prompt_z(hidden, stats_tensor)

        Y_hat = self.backbone_adapter.decode_from_hook(hidden_mod, means, stdev)

        return Y_hat, hidden, diagnostics

    def forward_frozen(self, X_t: Tensor) -> Tensor:
        """
        Run backbone without Prompt-Z modulation.
        For computing frozen baseline and noop margin loss.
        """
        with torch.no_grad():
            hidden, means, stdev = self.backbone_adapter.encode_until_hook(X_t)
            return self.backbone_adapter.decode_from_hook(hidden, means, stdev)

    def forward_train(
        self,
        X_t: Tensor,
        gamma_floor: float = 0.0,
        mask_floor: float = 0.0,
    ):
        """
        Training forward pass. Backbone frozen, but PromptZ has grad.

        Returns
        -------
        Y_hat       : [B, pred_len, C] — with grad through PromptZ
        Y_frozen    : [B, pred_len, C] — frozen baseline (no grad)
        reg_tensors : dict of LIVE (differentiable) tensors:
                      'effective_delta_ratio', 'gamma_mean', 'mask_mean'
        diagnostics : dict of detached floats for logging
        """
        # Encode (frozen, detached)
        with torch.no_grad():
            hidden, means, stdev = self.backbone_adapter.encode_until_hook(X_t)
        hidden = hidden.detach()
        means = means.detach()
        stdev = stdev.detach()

        # Frozen baseline (for noop margin)
        with torch.no_grad():
            Y_frozen = self.backbone_adapter.decode_from_hook(hidden, means, stdev)

        # Prompt-Z modulation (with gradients)
        residual_stats = self.residual_tracker.get_stats()
        stats_tensor = self._pack_stats(residual_stats, hidden.device)
        hidden_mod, reg_tensors, diagnostics = self.prompt_z(
            hidden,
            stats_tensor,
            gamma_floor=gamma_floor,
            mask_floor=mask_floor,
        )

        # Decode (frozen head, but gradients flow back through hidden_mod → PromptZ)
        Y_hat = self.backbone_adapter.decode_from_hook(hidden_mod, means, stdev)

        return Y_hat, Y_frozen, reg_tensors, diagnostics

    def _pack_stats(self, stats: Dict[str, Tensor], device: torch.device) -> Tensor:
        """
        Pack residual statistics into a single [C, 5] tensor.

        Order: [error_mean, error_slope, error_std, signed_bias, steps_gap]
        steps_gap is broadcast to C channels.
        """
        C = self.residual_tracker.C
        error_mean = stats["error_mean"].to(device)
        error_slope = stats["error_slope"].to(device)
        error_std = stats["error_std"].to(device)
        signed_bias = stats["signed_bias"].to(device)
        steps_gap = stats["steps_gap"].to(device).expand(C)

        return torch.stack([
            error_mean, error_slope, error_std, signed_bias, steps_gap
        ], dim=-1)  # [C, 5]

    def __repr__(self) -> str:
        return (
            f"PromptZTSF(\n"
            f"  backbone = {self.backbone_adapter.__class__.__name__} "
            f"(layout={self.backbone_adapter.hidden_layout}),\n"
            f"  prompt_z = {self.prompt_z},\n"
            f"  residual_tracker = K={self.residual_tracker.K}\n"
            f")"
        )
