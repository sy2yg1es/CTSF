"""
ResidualTracker — Causal Rolling Error Statistics for Prompt-Z
==============================================================

Maintains a ring buffer of per-channel error statistics computed from
**delayed** labels only.  Strictly causal: at prediction time t, the
tracker only contains residuals from labels that arrived at t-H or earlier.

The streaming loop is responsible for calling update() only AFTER the
delayed label arrives — this module does NOT enforce the delay itself,
it just tracks whatever is pushed to it.

Tracked signals (all per-channel [C]):
  error_mean         — rolling mean of |residual|
  error_slope        — OLS slope over window (positive = worsening)
  error_std          — rolling std of |residual|
  signed_bias        — rolling mean of signed residual (bias direction)
  steps_since_update — scalar, how many steps since last update() call
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


class ResidualTracker(nn.Module):
    """
    Stateful ring buffer for rolling error statistics.
    No gradients — pure buffer management.

    Parameters
    ----------
    num_channels : int
        Number of forecasting channels C.
    window_K : int
        Size of the rolling window.
    """

    def __init__(self, num_channels: int, window_K: int = 24) -> None:
        super().__init__()
        self.C = num_channels
        self.K = window_K

        # Ring buffers
        self.register_buffer("abs_error_window", torch.zeros(window_K, num_channels))
        self.register_buffer("signed_error_window", torch.zeros(window_K, num_channels))
        self.register_buffer("_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_count", torch.zeros(1, dtype=torch.long))

        # Steps since last update
        self.register_buffer("_steps_gap", torch.zeros(1, dtype=torch.float32))

        # Pre-compute OLS x-vector for slope estimation
        x = torch.arange(window_K, dtype=torch.float32)
        x = x - x.mean()
        self.register_buffer("_x_norm", x / (x.pow(2).sum() + 1e-8))  # [K]

    @torch.no_grad()
    def update(self, Y_hat_cached: Tensor, Y_true: Tensor) -> None:
        """
        Push a new residual observation into the ring buffer.

        IMPORTANT: This must only be called when the delayed label arrives.
        The caller (streaming loop) is responsible for the timing.

        Parameters
        ----------
        Y_hat_cached : [B, pred_len, C] — prediction made H steps ago
        Y_true       : [B, pred_len, C] — ground truth that just arrived
        """
        residual = Y_true - Y_hat_cached                     # [B, pred_len, C]
        abs_err = residual.abs().mean(dim=(0, 1))             # [C]
        signed_err = residual.mean(dim=(0, 1))                # [C]

        ptr = int(self._ptr.item())
        self.abs_error_window[ptr] = abs_err
        self.signed_error_window[ptr] = signed_err

        self._ptr = (self._ptr + 1) % self.K
        self._count = torch.clamp(self._count + 1, max=self.K)
        self._steps_gap.zero_()

    @torch.no_grad()
    def step_no_update(self) -> None:
        """Called each streaming step where no label arrives. Increments gap."""
        self._steps_gap += 1.0

    @torch.no_grad()
    def get_stats(self) -> Dict[str, Tensor]:
        """
        Compute rolling statistics from the ring buffer.

        Returns dict of Tensors, all on the same device as buffers:
          error_mean  : [C]
          error_slope : [C]
          error_std   : [C]
          signed_bias : [C]
          steps_gap   : [1]
        """
        n = int(self._count.item())
        device = self.abs_error_window.device

        if n == 0:
            C = self.C
            return {
                "error_mean": torch.zeros(C, device=device),
                "error_slope": torch.zeros(C, device=device),
                "error_std": torch.zeros(C, device=device),
                "signed_bias": torch.zeros(C, device=device),
                "steps_gap": self._steps_gap.clone(),
            }

        # Get filled portion in chronological order
        if n < self.K:
            filled = self.abs_error_window[:n]          # [n, C]
            signed_filled = self.signed_error_window[:n]
        else:
            # Full window: reorder from oldest to newest
            ptr = int(self._ptr.item())
            idx = torch.arange(self.K, device=device)
            idx = (idx + ptr) % self.K
            filled = self.abs_error_window[idx]         # [K, C]
            signed_filled = self.signed_error_window[idx]

        error_mean = filled.mean(dim=0)                 # [C]
        error_std = filled.std(dim=0) if n > 1 else torch.zeros(self.C, device=device)

        # OLS slope: dot(x_norm, error_window)
        if n >= 3:
            if n < self.K:
                x = torch.arange(n, dtype=torch.float32, device=device)
                x = x - x.mean()
                x_norm = x / (x.pow(2).sum() + 1e-8)
                error_slope = (x_norm.unsqueeze(1) * filled).sum(dim=0)  # [C]
            else:
                error_slope = (self._x_norm.unsqueeze(1) * filled).sum(dim=0)
        else:
            error_slope = torch.zeros(self.C, device=device)

        signed_bias = signed_filled.mean(dim=0)         # [C]

        return {
            "error_mean": error_mean,
            "error_slope": error_slope,
            "error_std": error_std,
            "signed_bias": signed_bias,
            "steps_gap": self._steps_gap.clone(),
        }

    def reset(self) -> None:
        """Reset all state."""
        self.abs_error_window.zero_()
        self.signed_error_window.zero_()
        self._ptr.zero_()
        self._count.zero_()
        self._steps_gap.zero_()
