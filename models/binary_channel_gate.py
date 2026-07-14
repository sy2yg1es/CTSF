"""Final binary channel gate used after the Prompt-Z delta branch is frozen.

The gate predicts the signed benefit of applying the existing correction for
each ``(window, channel)`` pair.  Its structural decision threshold is zero:
positive score applies the full correction and non-positive score keeps the
frozen-backbone prediction.  There is no second channel mask and no gate
magnitude/sparsity regularizer.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor


PROTOCOL_VERSION = "binary_channel_gate_v1"
SUPPORTED_MODES = ("frozen", "fixed", "learned_regressor")


class BinaryChannelGate(nn.Module):
    """Small normalized signed-advantage regressor.

    The returned value is a score rather than a probability.  ``score > 0``
    is the binary gate decision and is intentionally not tuned on validation
    or test data.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 64,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        if feature_mean is None:
            feature_mean = torch.zeros(feature_dim)
        if feature_std is None:
            feature_std = torch.ones(feature_dim)
        self.register_buffer("feature_mean", feature_mean.to(torch.float32))
        self.register_buffer("feature_std", feature_std.to(torch.float32))

        if hidden > 0:
            self.network = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            output = self.network[-1]
        else:
            self.network = nn.Linear(feature_dim, 1)
            output = self.network
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, features: Tensor) -> Tensor:
        normalized = (features - self.feature_mean) / self.feature_std.clamp(min=1e-6)
        return self.network(normalized).squeeze(-1)

    def decisions(self, features: Tensor) -> Tensor:
        """Return the protocol-defined hard binary decision."""
        return self(features) > 0.0


def build_causal_gate_features(
    drift_state: Tensor,
    residual_stats: Tensor,
    frozen_prediction: Tensor,
    fixed_prediction: Tensor,
    feature_mode: str = "causal_augmented",
) -> Tensor:
    """Build the exact causal features shared by gate training and TEST.

    Inputs follow the repository convention: predictions are ``[B,H,C]``,
    drift state is ``[B,C,D]``, and residual statistics are ``[C,5]``.
    No target values are used.
    """
    if feature_mode not in {
        "drift", "drift_residual", "drift_output", "causal_augmented"
    }:
        raise ValueError(f"Unsupported gate feature mode: {feature_mode}")

    features = drift_state
    extras = []
    if feature_mode in ("drift_residual", "causal_augmented"):
        extras.append(
            residual_stats.unsqueeze(0).expand(drift_state.shape[0], -1, -1)
        )
    if feature_mode in ("drift_output", "causal_augmented"):
        frozen_c = frozen_prediction.transpose(1, 2)  # [B,C,H]
        correction_c = (fixed_prediction - frozen_prediction).transpose(1, 2)
        frozen_rms = frozen_c.pow(2).mean(dim=-1).sqrt()
        correction_rms = correction_c.pow(2).mean(dim=-1).sqrt()
        extras.append(
            torch.stack(
                [
                    frozen_c.mean(dim=-1),
                    frozen_c.abs().mean(dim=-1),
                    frozen_c.std(dim=-1, unbiased=False),
                    correction_c.mean(dim=-1),
                    correction_c.abs().mean(dim=-1),
                    correction_rms,
                    correction_rms / frozen_rms.clamp(min=1e-6),
                ],
                dim=-1,
            )
        )
    return torch.cat([features, *extras], dim=-1) if extras else features


def gate_from_checkpoint(
    checkpoint: Mapping,
    device: torch.device | str,
) -> BinaryChannelGate:
    """Construct the final regressor and strictly load its saved state."""
    state = checkpoint["regressor"]
    config = checkpoint["config"]
    feature_dim = int(state["feature_mean"].numel())
    hidden = int(config.get("probe_hidden", 64))
    gate = BinaryChannelGate(feature_dim, hidden=hidden).to(device)
    gate.load_state_dict(state, strict=True)
    gate.eval()
    return gate


def validate_gate_checkpoint(checkpoint: Mapping) -> None:
    """Reject diagnostic/legacy checkpoints that are unsafe for formal TEST."""
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            "Gate checkpoint is not a finalized binary-channel-gate artifact. "
            "Retrain it with the synchronized Gate protocol before TEST."
        )
    mode = checkpoint.get("selected_mode")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid or missing selected_mode in gate checkpoint: {mode}")
    if float(checkpoint.get("decision_threshold", float("nan"))) != 0.0:
        raise ValueError("Formal binary gate requires the structural threshold score > 0")
