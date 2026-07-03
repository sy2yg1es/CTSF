"""
ImprovedDriftDetector — 多条件漂移检测器
==========================================

改进点（相对于 ActualDriftDetector）：

旧方法：
  误差滑动均值 > 阈值 AND abs_degradation → update

新方法（四重条件门控）：
  error_mean > tau           — 当前误差均值超过基线（原有条件）
  AND error_slope > 0        — 误差趋势向上（排除突刺后恢复）
  AND error_std < std_cap    — 误差不是剧烈抖动（排除高噪声步骤）
  AND consecutive >= p       — 已连续 p 步满足以上所有条件

error_slope 用最小二乘法在窗口上估计，计算量很低。
error_std 用窗口内标准差，防止偶发噪声触发。

额外参数（相比旧版）：
  slope_thresh : float = 0.0   — 要求斜率 > 此值（0 = 任意正斜率）
  std_cap_mult : float = 2.0   — std_cap = base_sigma * std_cap_mult
                                  超过此倍数 sigma 认为当前在抖动，不更新
"""

import torch
import torch.nn as nn
import math


class ImprovedDriftDetector(nn.Module):
    def __init__(
        self,
        num_channels: int,
        window_K: int = 12,
        threshold_tau: float = 0.1,
        patience_C: int = 3,
        alpha_tol: float = 1.0,
        e_min: float = 1e-4,
        slope_thresh: float = 0.0,
        std_cap_mult: float = 2.0,
    ):
        super().__init__()
        self.C = num_channels
        self.window_K = window_K
        self.threshold_tau = threshold_tau
        self.patience_C = patience_C
        self.alpha_tol = alpha_tol
        self.e_min = e_min
        self.slope_thresh = slope_thresh
        self.std_cap_mult = std_cap_mult

        # 环形窗口
        self.register_buffer("history_window", torch.zeros(window_K, num_channels))
        self.register_buffer("window_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_warmed_up", torch.zeros(1, dtype=torch.bool))
        self._ptr: int = 0
        self._warmed: bool = False

        # 基线
        self.register_buffer("R_ref", torch.zeros(num_channels))
        self.register_buffer("sigma_0", torch.zeros(num_channels))

        # 连续报警计数器
        self.register_buffer("consecutive_alerts", torch.zeros(num_channels, dtype=torch.long))

        # 预计算线性回归 x 向量: [K]，用于 slope 估计
        x = torch.arange(window_K, dtype=torch.float32)
        x = x - x.mean()
        self.register_buffer("_x_norm", x / (x.pow(2).sum() + 1e-8))

    def update_and_check(self, Y_hat: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:
        B, pred_len, C = Y_hat.shape
        device = Y_hat.device
        assert C == self.C

        # 误差 [C]
        err_t = torch.abs(Y_hat - Y_true).mean(dim=(0, 1))

        # 更新窗口
        ptr = self._ptr
        self.history_window[ptr] = err_t
        new_ptr = (ptr + 1) % self.window_K
        self.window_ptr[0] = new_ptr
        self._ptr = new_ptr

        if not self._warmed:
            if new_ptr == 0:
                self.is_warmed_up[0] = True
                self._warmed = True
                self.R_ref.copy_(torch.quantile(self.history_window, 0.9, dim=0))
                self.sigma_0.copy_(torch.std(self.history_window, dim=0))
            return torch.zeros(B, C, device=device)

        # ---- 条件 1: 误差均值超过基线 ----
        R_t = torch.quantile(self.history_window, 0.9, dim=0)  # [C]
        rel_degrad = (R_t - self.R_ref) / (self.R_ref + 1e-5) > self.threshold_tau
        abs_degrad = R_t > (self.R_ref + self.alpha_tol * self.sigma_0 + self.e_min)
        cond_mean = rel_degrad & abs_degrad

        # ---- 条件 2: 误差斜率 > slope_thresh ----
        # x_norm: [K], history_window: [K, C] → slope: [C]
        slope = (self._x_norm.unsqueeze(1) * self.history_window).sum(dim=0)  # [C]
        cond_slope = slope > self.slope_thresh

        # ---- 条件 3: 误差标准差不超过 std_cap ----
        err_std = self.history_window.std(dim=0)  # [C]
        std_cap = self.std_cap_mult * self.sigma_0
        cond_stable = err_std < std_cap          # 当前窗口不太抖 → 允许更新

        # ---- 四重门控 ----
        is_alert = cond_mean & cond_slope & cond_stable

        # ---- 连续 patience 计数 ----
        self.consecutive_alerts = torch.where(
            is_alert,
            self.consecutive_alerts + 1,
            torch.zeros_like(self.consecutive_alerts),
        )

        drift_mask_1d = (self.consecutive_alerts >= self.patience_C).float()  # [C]

        # 漂移后重置基线
        self.R_ref = torch.where(drift_mask_1d > 0.5, R_t, self.R_ref)
        self.consecutive_alerts = torch.where(
            drift_mask_1d > 0.5,
            torch.zeros_like(self.consecutive_alerts),
            self.consecutive_alerts,
        )

        return drift_mask_1d.unsqueeze(0).expand(B, -1)  # [B, C]
