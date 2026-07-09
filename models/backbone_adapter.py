"""
BackboneAdapter — Abstract interface for backbone-agnostic prompt injection.
============================================================================

Each adapter wraps a specific frozen backbone and exposes exactly two operations:

  encode(X)                       → H_tokens, z_query, means, stdev
  fuse_and_decode(H, θ, μ, σ)    → Y_hat

The framework (ContinualPromptTSF) never touches the backbone directly.
All backbone-specific logic (patching, prefix-tuning vs additive fusion,
prediction heads, normalization) is encapsulated here.

Gradient semantics
~~~~~~~~~~~~~~~~~~
  • All backbone parameters have requires_grad=False.
  • In inference: the framework wraps both calls in torch.no_grad().
  • In update (forward_update): encode is no_grad + detach, but
    fuse_and_decode runs WITH the computation graph so gradients
    flow from loss → through frozen backbone ops → back to θ.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Tuple


class BackboneAdapter(nn.Module, ABC):
    """
    Abstract base class for backbone adapters.

    Subclasses must implement:
      encode(X)                    — tokenize + extract z_query
      fuse_and_decode(H, θ, μ, σ) — inject prompt + run encoder/head

    Prompt-Z hook API (for gray-box representation modulation):
      encode_until_hook(X)         — run up to hook point (encoder output)
      decode_from_hook(h, μ, σ)    — run from hook point (prediction head)
      hidden_layout                — 'BCDP' or 'BCD'
    """

    @abstractmethod
    def encode(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Tokenize and encode input (typically called under no_grad).

        Parameters
        ----------
        X : [B, seq_len, C]  — raw input window.

        Returns
        -------
        H_tokens : [B, C, D, S]  — backbone hidden states.
                   S = patch_num for PatchTST, 1 for iTransformer.
        z_query  : [B, C, D]     — per-channel query for MoE router.
        means    : [B, 1, C]     — instance normalization mean.
        stdev    : [B, 1, C]     — instance normalization stdev.
        """
        ...

    @abstractmethod
    def fuse_and_decode(
        self, H_tokens: Tensor, theta: Tensor,
        means: Tensor, stdev: Tensor,
    ) -> Tensor:
        """
        Inject prompt into hidden states, run encoder (if any), and decode.

        Parameters
        ----------
        H_tokens : [B, C, D, S]  — from encode().
        theta    : [B, C, D]     — prompt from MoE retrieval.
        means    : [B, 1, C]     — for denormalization.
        stdev    : [B, 1, C]     — for denormalization.

        Returns
        -------
        Y_hat : [B, pred_len, C]  — forecast.
        """
        ...

    # ------------------------------------------------------------------
    # Prompt-Z Hook API
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_until_hook(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run backbone from input up to the hook point (encoder output,
        before prediction head).

        Parameters
        ----------
        X : [B, seq_len, C]  — raw input window.

        Returns
        -------
        hidden : [B, C, D, P] (PatchTST) or [B, C, D] (iTransformer)
        means  : [B, 1, C]
        stdev  : [B, 1, C]
        """
        ...

    @abstractmethod
    def decode_from_hook(
        self, hidden: Tensor, means: Tensor, stdev: Tensor,
    ) -> Tensor:
        """
        Run from hook point (prediction head) to output.

        Parameters
        ----------
        hidden : same shape as encode_until_hook output.
        means  : [B, 1, C]
        stdev  : [B, 1, C]

        Returns
        -------
        Y_hat : [B, pred_len, C]
        """
        ...

    @property
    @abstractmethod
    def hidden_layout(self) -> str:
        """'BCDP' for PatchTST, 'BCD' for iTransformer."""
        ...


# ============================================================================
# PatchTSTAdapter
# ============================================================================

class PatchTSTAdapter(BackboneAdapter):
    """
    Adapter for PatchTST backbone.

    Prompt injection strategy: **Prefix-Tuning with Truncation**.
      θ is prepended as a virtual token along the patch axis.
      After encoding, the prefix token is surgically removed.

    Encapsulates all PatchTST-specific internals:
      - Patch embedding with CI layout and method-name shadowing workaround
      - forward_transformer_encoder (handles [B*C, S, D] reshaping)
      - FlattenHead prediction + denormalization
    """

    def __init__(self, backbone: nn.Module, use_query_mlp: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_query_mlp = use_query_mlp

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optional z_query MLP: 4D patch statistics → D
        # Features: [mean(P), last(P), last-first(P), std(P)] each [B,C,D] → cat → [B,C,4D]
        # We lazily init once we know D (first forward call).
        self._query_mlp: nn.Module | None = None
        self._d_model: int | None = None

    def _ensure_query_mlp(self, D: int, device) -> None:
        """Lazily initialize the query MLP on first call."""
        if self._query_mlp is None:
            self._d_model = D
            self._query_mlp = nn.Sequential(
                nn.Linear(4 * D, D),
                nn.GELU(),
                nn.Linear(D, D),
            ).to(device)

    @property
    def d_model(self) -> int:
        """Expose backbone d_model for prompt_memory initialization."""
        return self.backbone.encoder.layers[0].attention.inner_attention.scale \
            if hasattr(self.backbone, 'd_model') else \
            self.backbone.head.linear.in_features // self._infer_patch_num()

    def encode(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        PatchTST encode: normalize → patch_embedding → [B, C, D, P].

        z_query strategy:
          use_query_mlp=False: mean(P)  — original, fast, backward-compat
          use_query_mlp=True:  MLP([mean, last, last-first, std])  — richer
        """
        H_patches, means, stdev = self.backbone.encode_local(X)  # [B, C, D, P]

        if self.use_query_mlp:
            # H_patches: [B, C, D, P]
            h_mean = H_patches.mean(dim=-1)            # [B, C, D]
            h_last = H_patches[..., -1]                # [B, C, D]
            h_first = H_patches[..., 0]                # [B, C, D]
            h_std  = H_patches.std(dim=-1)             # [B, C, D]
            features = torch.cat([h_mean, h_last, h_last - h_first, h_std], dim=-1)  # [B, C, 4D]
            self._ensure_query_mlp(H_patches.shape[2], H_patches.device)
            z_query = self._query_mlp(features)        # [B, C, D]
        else:
            z_query = H_patches.mean(dim=-1)           # [B, C, D]

        return H_patches, z_query, means, stdev

    def fuse_and_decode(
        self, H_tokens: Tensor, theta: Tensor,
        means: Tensor, stdev: Tensor,
    ) -> Tensor:
        """
        Prefix-Tuning: prepend θ → encoder → truncate → head → denorm.
        """
        # Prepend theta as virtual prefix token
        theta_prefix = theta.unsqueeze(-1)                        # [B, C, D, 1]
        H_prefix = torch.cat([theta_prefix, H_tokens], dim=-1)   # [B, C, D, P+1]

        # Transformer encoder
        H_encoded = self.backbone.forward_transformer_encoder(H_prefix)  # [B, C, D, P+1]

        # Truncate prefix token
        H_real = H_encoded[..., 1:]                               # [B, C, D, P]

        # Prediction head + denormalization
        Y_hat = self.backbone.apply_prediction_head(H_real, means, stdev)
        return Y_hat                                               # [B, pred_len, C]

    # ------------------------------------------------------------------
    # Prompt-Z Hook API
    # ------------------------------------------------------------------

    def encode_until_hook(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        PatchTST: norm → patch_embed → encoder → [B, C, D, P].
        Hook is at encoder output, before prediction head.
        """
        H_patches, means, stdev = self.backbone.encode_local(X)   # [B, C, D, P]
        B, C, D, P = H_patches.shape
        enc_in = H_patches.permute(0, 1, 3, 2).reshape(B * C, P, D)
        enc_out, _ = self.backbone.encoder(enc_in)
        enc_out = enc_out.reshape(B, C, P, D).permute(0, 1, 3, 2)  # [B, C, D, P]
        return enc_out, means, stdev

    def decode_from_hook(
        self, hidden: Tensor, means: Tensor, stdev: Tensor,
    ) -> Tensor:
        """PatchTST: prediction_head([B, C, D, P]) → [B, pred_len, C]."""
        return self.backbone.apply_prediction_head(hidden, means, stdev)

    @property
    def hidden_layout(self) -> str:
        return 'BCDP'

    @torch.no_grad()
    def forward_frozen(self, X: Tensor) -> Tensor:
        """
        Run backbone completely without prompt injection.
        Used by oracle label generation as the no-op baseline.
        """
        hidden, means, stdev = self.encode_until_hook(X)
        return self.decode_from_hook(hidden, means, stdev)


# ============================================================================
# iTransformerAdapter (Phase 3 P1 — stub for now)
# ============================================================================

class iTransformerAdapter(BackboneAdapter):
    """
    Adapter for iTransformer backbone.

    iTransformer treats each CHANNEL as a token (inverted Transformer).
    Self-Attention operates over the channel dimension, not the time dimension.

    Prompt injection strategy: **Additive Fusion**.
      θ is added to the channel embedding before the encoder.
      This modulates each channel's representation before cross-channel attention.

    Internal representation: [B, C, D] (no patch axis).
    We add a dummy S=1 axis for BackboneAdapter interface compliance.
    """

    def __init__(self, backbone: nn.Module, use_query_mlp: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.pred_len = backbone.pred_len
        self.use_query_mlp = use_query_mlp

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._query_mlp: nn.Module | None = None

    def _ensure_query_mlp(self, D: int, device) -> None:
        if self._query_mlp is None:
            # iTransformer has no patch axis, so we use time-axis statistics
            # from the raw normalized input instead
            self._query_mlp = nn.Sequential(
                nn.Linear(4 * D, D),
                nn.GELU(),
                nn.Linear(D, D),
            ).to(device)

    def encode(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        iTransformer encode: normalize → inverted embedding → [B, C, D].
        z_query = channel embedding directly (natural per-channel representation).
        """
        # Instance normalization
        means = X.mean(1, keepdim=True).detach()             # [B, 1, C]
        x_norm = X - means
        stdev = torch.sqrt(
            torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()                                            # [B, 1, C]
        x_norm = x_norm / stdev

        # iTransformer inverted embedding: [B, seq_len, C] → [B, C, D]
        enc_out = self.backbone.enc_embedding(x_norm, None)   # [B, C, D]

        z_query = enc_out                                      # [B, C, D]
        H_tokens = enc_out.unsqueeze(-1)                       # [B, C, D, 1]

        return H_tokens, z_query, means, stdev

    def fuse_and_decode(
        self, H_tokens: Tensor, theta: Tensor,
        means: Tensor, stdev: Tensor,
    ) -> Tensor:
        """
        Additive Fusion: enc_out + θ → encoder → projection → denorm.
        """
        # Additive fusion: inject prompt into channel embeddings
        enc_out = H_tokens.squeeze(-1)                         # [B, C, D]
        enc_fused = enc_out + theta                            # [B, C, D]

        # iTransformer encoder (attention over channels)
        enc_out, _ = self.backbone.encoder(enc_fused)          # [B, C, D]

        # Projection: [B, C, D] → [B, C, pred_len] → [B, pred_len, C]
        dec_out = self.backbone.projection(enc_out)            # [B, C, pred_len]
        C = enc_out.shape[1]
        dec_out = dec_out.permute(0, 2, 1)[:, :, :C]          # [B, pred_len, C]

        # Denormalization
        dec_out = dec_out * stdev[:, 0, :C].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :C].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out                                         # [B, pred_len, C]

    # ------------------------------------------------------------------
    # Prompt-Z Hook API
    # ------------------------------------------------------------------

    def encode_until_hook(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        iTransformer: norm → embedding → encoder → [B, C, D].
        Hook is at encoder output, before projection head.
        Delegates to backbone.encode_local() for consistency.
        """
        # backbone.encode_local() handles normalization, embedding, encoder
        enc_out, means, stdev = self.backbone.encode_local(X)  # [B, C, D], [B,1,C], [B,1,C]
        return enc_out, means, stdev

    def decode_from_hook(
        self, hidden: Tensor, means: Tensor, stdev: Tensor,
    ) -> Tensor:
        """iTransformer: projection([B, C, D]) → [B, pred_len, C].
        Channel slice [:,:,:C] matches original forecast() behavior.
        """
        C = hidden.shape[1]
        dec_out = self.backbone.projection(hidden)             # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1)[:, :, :C]          # [B, pred_len, C]
        dec_out = dec_out * stdev[:, 0, :C].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :C].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    @property
    def hidden_layout(self) -> str:
        return 'BCD'

    @torch.no_grad()
    def forward_frozen(self, X: Tensor) -> Tensor:
        """iTransformer frozen baseline: no prompt injection."""
        hidden, means, stdev = self.encode_until_hook(X)
        return self.decode_from_hook(hidden, means, stdev)
