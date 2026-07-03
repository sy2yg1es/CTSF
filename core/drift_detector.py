import torch
import torch.nn as nn

class ActualDriftDetector(nn.Module):
    """
    Channel-Independent Actual Drift Trigger (CI-ADT)
    全面向量化实现，独立维护 C 个通道的漂移状态机。
    """
    def __init__(
        self, 
        num_channels: int, 
        window_K: int = 12, 
        threshold_tau: float = 0.2, 
        patience_C: int = 3,
        alpha_tol: float = 1.0,
        e_min: float = 1e-4
    ):
        super().__init__()
        self.C = num_channels
        self.window_K = window_K
        self.threshold_tau = threshold_tau
        self.patience_C = patience_C
        self.alpha_tol = alpha_tol
        self.e_min = e_min

        # 核心状态机 (使用 register_buffer 确保它们能随模型 save/load 并在正确的 device 上)
        # 1. 历史残差滑动窗口: [window_K, C]
        self.register_buffer("history_window", torch.zeros(window_K, self.C))
        self.register_buffer("window_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_warmed_up", torch.zeros(1, dtype=torch.bool))

        # Python-side shadows to avoid GPU→CPU sync from .item() in the hot loop
        self._ptr: int = 0
        self._warmed: bool = False

        # 2. 参考基线与容忍度: [C]
        self.register_buffer("R_ref", torch.zeros(self.C))
        self.register_buffer("sigma_0", torch.zeros(self.C))
        
        # 3. 连续报警计数器: [C]
        self.register_buffer("consecutive_alerts", torch.zeros(self.C, dtype=torch.long))

    def update_and_check(self, Y_hat: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:
        """
        接收预测与真实值，返回 [B, C] 的漂移掩码。
        Y_hat, Y_true: [B, pred_len, C]
        """
        B, pred_len, C = Y_hat.shape
        device = Y_hat.device

        assert C == self.C, f"Input channels {C} mismatch initialized {self.C}"

        # 1. 计算当前时间步 t 的通道级残差绝对值均值 (Mean over prediction horizon) -> [C]
        # 假设 Batch size = 1 (严格流式)
        err_t = torch.abs(Y_hat - Y_true).mean(dim=(0, 1))

        # 2. 更新环形滑动窗口
        ptr = self._ptr
        self.history_window[ptr] = err_t
        new_ptr = (ptr + 1) % self.window_K
        self.window_ptr[0] = new_ptr
        self._ptr = new_ptr

        # 3. 预热期判定：窗口未填满前，不触发漂移
        if not self._warmed:
            if new_ptr == 0:  # 窗口首次被填满
                self.is_warmed_up[0] = True
                self._warmed = True
                # 初始化基线 R_ref (第 90 百分位数) 与 sigma_0
                self.R_ref.copy_(torch.quantile(self.history_window, 0.9, dim=0))
                self.sigma_0.copy_(torch.std(self.history_window, dim=0))
            return torch.zeros(B, C, device=device) # [B, C] 全 0 掩码

        # 4. 计算当前窗口的 R_t (第 90 百分位数) -> [C]
        R_t = torch.quantile(self.history_window, 0.9, dim=0)

        # 5. 双重门控判定 (Relative & Absolute Degradation) -> 布尔张量 [C]
        rel_degrad = (R_t - self.R_ref) / (self.R_ref + 1e-5) > self.threshold_tau
        abs_degrad = R_t > (self.R_ref + self.alpha_tol * self.sigma_0 + self.e_min)
        
        is_alert = rel_degrad & abs_degrad

        # 6. 更新连续报警计数器
        self.consecutive_alerts = torch.where(
            is_alert, 
            self.consecutive_alerts + 1, 
            torch.zeros_like(self.consecutive_alerts) # 只要中断一次就清零
        )

        # 7. 生成漂移掩码 (达到 patience) -> float 张量 [C]
        drift_mask_1d = (self.consecutive_alerts >= self.patience_C).float()

        # 8. 漂移后状态重置 (至关重要：发生漂移的通道，其基线需更新为当前 R_t，并清零计数器)
        self.R_ref = torch.where(drift_mask_1d > 0.5, R_t, self.R_ref)
        self.consecutive_alerts = torch.where(drift_mask_1d > 0.5, torch.zeros_like(self.consecutive_alerts), self.consecutive_alerts)

        # 9. 扩维以匹配 streaming_loop 的预期输出 -> [B, C]
        drift_mask = drift_mask_1d.unsqueeze(0).expand(B, -1)
        
        return drift_mask