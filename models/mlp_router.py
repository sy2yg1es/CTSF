"""
RichMLPRouter — Stage 1 MLP-based Router with No-op Expert
===========================================================

替代 SparsePromptMemory 中的 nn.Linear router。

输入：
  z_features  : [B, C, 4*D]  — 当前窗口形态特征 (mean/last/slope/std)
  err_history : [B, C, K]    — 最近 K 步的每通道预测误差 (可选)

输出：
  logits : [B, C, E+1]  — 其中第 E 列是 no-op expert 的 logit

No-op Expert (index = num_experts)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
No-op expert 对应的 prompt 向量固定为 zeros，在 SparsePromptMemory 中注册为
非可训练的 register_buffer。当 router 选中 no-op 时，weighted theta 中该
expert 的贡献为 0，等价于保持 frozen baseline 的预测。

这防止了短视界场景下 router 被迫选一个"不如 frozen"的 expert。

Temporal Smoothness Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training 阶段可通过 get_smoothness_loss() 获取相邻步路由的 KL 散度。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class RichMLPRouter(nn.Module):
    """
    两分支 MLP Router：

      Window branch  : [B, C, 4D]  → Linear(4D, hidden) → LayerNorm → GELU
      History branch : [B, C, K]   → linear summary → [B, C, hist_hidden] → GELU
      Merge          : concat → Linear(hidden+hist_hidden, hidden) → GELU → Linear(hidden, E+1)

    E+1 outputs: E 个 expert logits + 1 个 no-op logit (最后一列)

    Parameters
    ----------
    d_model      : int — backbone embedding dimension (D)
    num_experts  : int — number of real experts E (not counting no-op)
    hist_K       : int — length of error history window
    hidden       : int — MLP hidden dimension
    hist_hidden  : int — history branch hidden dimension
    dropout      : float
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        hist_K: int = 12,
        hidden: int = 256,
        hist_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.hist_K = hist_K
        self.n_out = num_experts + 1  # +1 for no-op

        # ---- Window branch: processes 4D patch statistics ----
        self.window_branch = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- History branch: processes K-step error history ----
        # Extracts: mean, slope, std, bias-from-ref  (4 summary stats)
        # Then projects to hist_hidden
        self.hist_branch = nn.Sequential(
            nn.Linear(4, hist_hidden),  # 4 summary stats
            nn.GELU(),
        )

        # ---- Merge and classify ----
        self.merge = nn.Sequential(
            nn.Linear(hidden + hist_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_out),
        )

        # ---- Temporal smoothness: cache previous logits ----
        # Not a parameter — just a transient buffer for training
        self._prev_logits: Optional[Tensor] = None

        # ---- Linear regression weights for slope (fixed) ----
        # x = [0, 1, ..., K-1], centered and normalized
        x = torch.arange(hist_K, dtype=torch.float32)
        x = x - x.mean()
        self.register_buffer('_x_slope', x / (x.pow(2).sum() + 1e-8))  # [K]

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init the final linear → router starts near uniform
        nn.init.zeros_(self.merge[-1].weight)
        nn.init.zeros_(self.merge[-1].bias)

    def _history_summary(self, err_history: Tensor) -> Tensor:
        """
        Compute 4 summary statistics from per-channel error history.

        Parameters
        ----------
        err_history : [B, C, K]  — recent prediction errors per channel

        Returns
        -------
        summary : [B, C, 4]  — [mean, slope, std, signed_bias]
        """
        h_mean  = err_history.mean(dim=-1, keepdim=True)     # [B, C, 1]
        # slope via precomputed normalized x
        slope   = (self._x_slope * err_history).sum(dim=-1, keepdim=True)  # [B, C, 1]
        h_std   = err_history.std(dim=-1, keepdim=True)      # [B, C, 1]
        # bias: how much current error exceeds the half-window mean
        half = self.hist_K // 2
        bias    = (err_history[..., half:].mean(dim=-1, keepdim=True) -
                   err_history[..., :half].mean(dim=-1, keepdim=True))  # [B, C, 1]
        return torch.cat([h_mean, slope, h_std, bias], dim=-1)  # [B, C, 4]

    def forward(
        self,
        z_features: Tensor,
        err_history: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        z_features  : [B, C, 4*D]  — window features (mean/last/diff/std)
        err_history : [B, C, K]    — error history (None → use zeros)

        Returns
        -------
        logits : [B, C, E+1]  — last column is no-op logit
        """
        B, C, _ = z_features.shape

        # Window branch
        w_feat = self.window_branch(z_features)  # [B, C, hidden]

        # History branch
        if err_history is None:
            err_history = torch.zeros(B, C, self.hist_K, device=z_features.device)
        h_summary = self._history_summary(err_history)       # [B, C, 4]
        h_feat = self.hist_branch(h_summary)                  # [B, C, hist_hidden]

        # Merge
        merged = torch.cat([w_feat, h_feat], dim=-1)          # [B, C, hidden+hist_hidden]
        logits = self.merge(merged)                            # [B, C, E+1]

        # Cache for temporal smoothness loss
        self._prev_logits = logits.detach()

        return logits  # [B, C, E+1]

    def get_smoothness_loss(self, current_logits: Tensor) -> Tensor:
        """
        KL divergence between current and previous routing distribution.
        Encourages temporally stable routing (important for Traffic H=1).

        Returns scalar tensor (0 if no previous logits cached).
        """
        if self._prev_logits is None:
            return torch.tensor(0.0, device=current_logits.device)
        p_prev = F.softmax(self._prev_logits, dim=-1).detach()
        p_curr = F.softmax(current_logits, dim=-1)
        # KL(p_prev || p_curr) averaged over B, C
        kl = F.kl_div(p_curr.log(), p_prev, reduction='batchmean')
        return kl

    def reset_cache(self) -> None:
        """Call at the start of each epoch / streaming episode."""
        self._prev_logits = None
