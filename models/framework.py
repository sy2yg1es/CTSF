"""
Online Time Series Forecasting — Top-level Integration Module
=============================================================
ContinualPromptTSF  (v3 — Backbone-Agnostic Plugin Architecture)
-----------------------------------------------------------------

Architecture overview
~~~~~~~~~~~~~~~~~~~~~

  Input X_t : [B, seq_len, C]
        │
        ▼
  BackboneAdapter.encode(X_t)
        │
        ├──→ H_tokens : [B, C, D, S]     ← backbone-specific hidden states
        │                                    S = patch_num (PatchTST) or 1 (iTransformer)
        └──→ z_query  : [B, C, D]        ← per-channel MoE routing query
        │
        ▼
  SparsePromptMemory.retrieve_prompt(z_query)
        │
        ▼  theta : [B, C, D]             ← aggregated expert prompt, per channel
        │
        ▼ (optional)
  BottleneckAdapter(theta)                ← low-rank regularization
        │
        ▼
  BackboneAdapter.fuse_and_decode(H_tokens, theta, means, stdev)
        │                                    PatchTST: Prefix-Tuning + Truncation
        │                                    iTransformer: Additive Fusion
        ▼
  Y_hat : [B, pred_len, C]

v3 changes over v2
~~~~~~~~~~~~~~~~~~~
  - Replaced direct backbone access with BackboneAdapter interface.
  - Removed ChannelFusion (z_query now computed inside each adapter).
  - Removed _prefix_forward helper (backbone-specific logic in adapter).
  - ContinualPromptTSF is now backbone-blind: only touches adapter, prompt_memory, adapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from models.backbone_adapter import BackboneAdapter


# ============================================================================
# BottleneckAdapter  (Phase 2 — unchanged)
# ============================================================================

class BottleneckAdapter(nn.Module):
    """
    Low-rank bottleneck for prompt regularization.

    Architecture:  theta → Down(D→d) → GELU → Up(d→D) → residual add

    The Up projection is zero-initialized so that at startup the adapter
    output is zero, making the overall transform near-identity:
        theta_out = theta + adapter(theta) ≈ theta + 0 = theta
    """

    def __init__(self, d_model: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.up   = nn.Linear(bottleneck_dim, d_model)

        # Zero-init up projection → adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, theta: Tensor) -> Tensor:
        """Apply bottleneck with residual: theta + Up(GELU(Down(theta)))."""
        return theta + self.up(F.gelu(self.down(theta)))


# ============================================================================
# ContinualPromptTSF  (v3 — backbone-agnostic)
# ============================================================================

class ContinualPromptTSF(nn.Module):
    """
    Top-level continual online TSF module — backbone-agnostic edition.

    Components
    ----------
    backbone_adapter : BackboneAdapter subclass (PatchTSTAdapter, iTransformerAdapter, ...).
                       Encapsulates all backbone-specific logic.
    prompt_memory    : SparsePromptMemory; router + expert bank in [B, C, D] space.
    adapter          : Optional BottleneckAdapter for prompt regularization.

    Dual-forward design (required by BDLA asynchronous update loop)
    ---------------------------------------------------------------
    forward(X_t)
        Streaming inference path. Everything under no_grad.
        Returns (Y_hat, z_query, routing_probs, dispatch_indices).

    forward_update(X_history, dispatch_indices, z_query_history)
        Delayed-label update path (BDLA phase). Backbone frozen but
        gradients flow through frozen ops back to θ (prompt/adapter).
        Returns (Y_hat, theta_forced, routing_probs_history).
    """

    def __init__(
        self,
        backbone_adapter: BackboneAdapter,
        prompt_memory: nn.Module,
        adapter: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.backbone_adapter = backbone_adapter
        self.prompt_memory    = prompt_memory
        self.adapter          = adapter

    def _router_features(self, H_tokens: Tensor, z_query: Tensor) -> Tensor:
        """
        Build the 4D feature vector used by RichMLPRouter.

        Stage3 router training uses [mean, last, last-first, std] over the
        token axis. Streaming inference and delayed replay must use the same
        representation.
        """
        if not getattr(self.prompt_memory, "_use_rich_router", False):
            return z_query

        h_mean = H_tokens.mean(dim=-1)
        h_last = H_tokens[..., -1]
        h_first = H_tokens[..., 0]
        h_std = H_tokens.std(dim=-1, unbiased=False)
        return torch.cat([h_mean, h_last, h_last - h_first, h_std], dim=-1)

    # -------------------------------------------------------------------------
    # forward — streaming inference (all under no_grad)
    # -------------------------------------------------------------------------

    def forward(
        self,
        X_t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Streaming inference path.

        Returns
        -------
        Y_hat            : [B, pred_len, C]     — forecast.
        z_query          : [B, C, D]  detached  — per-channel query (stored in BDLABuffer).
        routing_probs    : [B, C, E]             — full softmax probabilities.
        dispatch_indices : [B, C, K]             — Top-K expert indices.
        """
        # Step 1: Encode (backbone-agnostic, frozen)
        with torch.no_grad():
            H_tokens, z_query, means, stdev = self.backbone_adapter.encode(X_t)

        # Step 2: MoE routing — per-channel prompt retrieval
        router_features = self._router_features(H_tokens, z_query)
        if getattr(self.prompt_memory, "_use_rich_router", False):
            theta, routing_probs, dispatch_indices = self.prompt_memory.retrieve_prompt(
                z_query, z_features=router_features
            )
            stored_query = router_features
        else:
            theta, routing_probs, dispatch_indices = self.prompt_memory.retrieve_prompt(z_query)
            stored_query = z_query

        # Step 3: Optional bottleneck adapter
        if self.adapter is not None:
            theta = self.adapter(theta)

        # Step 4: Fuse prompt + decode (backbone-agnostic, frozen)
        with torch.no_grad():
            Y_hat = self.backbone_adapter.fuse_and_decode(H_tokens, theta, means, stdev)

        return Y_hat, stored_query.detach(), routing_probs, dispatch_indices

    # -------------------------------------------------------------------------
    # forward_update — delayed-label update path (gradients through θ)
    # -------------------------------------------------------------------------

    def forward_update(
        self,
        X_history: Tensor,
        dispatch_indices: Tensor,
        z_query_history: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        BDLA update path. Backbone is frozen but θ retains grad_fn.

        Gradient flow:
            loss → Y_hat → (frozen backbone ops) → θ → prompt_memory / adapter

        Parameters
        ----------
        X_history         : [B, seq_len, C]  — historical input from BDLABuffer.
        dispatch_indices  : [B, C, K]        — stored Top-K expert indices.
        z_query_history   : [B, C, D]        — stored channel queries.

        Returns
        -------
        Y_hat                : [B, pred_len, C]  — with grad_fn attached.
        theta_forced         : [B, C, D]         — with grad_fn attached.
        routing_probs_history: [B, C, E]         — with grad_fn attached.
        """
        # Step 1: Encode + detach (backbone contribution is frozen, no grad)
        with torch.no_grad():
            H_tokens, z_query_replay, means, stdev = self.backbone_adapter.encode(X_history)
        H_tokens = H_tokens.detach()
        z_query_replay = z_query_replay.detach()
        means = means.detach()
        stdev = stdev.detach()

        # Step 2: Reconstruct routing with live grad_fn (force_prompt replays z_query through router)
        if getattr(self.prompt_memory, "_use_rich_router", False):
            z_query_history = self._router_features(H_tokens, z_query_replay)
        theta_forced, routing_probs_history = self.prompt_memory.force_prompt(
            dispatch_indices, z_query_history
        )

        # Step 3: Optional bottleneck adapter (must match inference path)
        if self.adapter is not None:
            theta_forced = self.adapter(theta_forced)

        # Step 4: Fuse + decode WITHOUT no_grad — gradients flow through frozen ops back to θ
        Y_hat = self.backbone_adapter.fuse_and_decode(H_tokens, theta_forced, means, stdev)

        return Y_hat, theta_forced, routing_probs_history

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ContinualPromptTSF(\n"
            f"  backbone_adapter = {self.backbone_adapter.__class__.__name__},\n"
            f"  prompt_memory    = {self.prompt_memory},\n"
            f"  adapter          = {self.adapter.__class__.__name__ if self.adapter else 'None'}\n"
            f")"
        )
