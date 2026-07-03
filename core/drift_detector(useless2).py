"""
Online Time Series Forecasting — Per-prompt Dynamic Drift Detector
===================================================================
ActualDriftDetector: 
Identifies Stable, Virtual Drift, and Actual Drift using joint criteria:
1. P(X) distribution shift (MMD on features/representations)
2. P(Y|X) structural degradation (P90 of residuals with absolute tolerance)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Literal

# 定义 Drift 状态枚举
DriftStatus = Literal["STABLE", "VIRTUAL_DRIFT", "ACTUAL_DRIFT"]

class ActualDriftDetector:
    def __init__(
        self,
        window_K: int,
        threshold_tau: float,
        patience_C: int,
        alpha_stable: float = 0.05,        # 暴露为超参：平稳期 EMA 衰减率
        alpha_reset: float = 0.30,         # 暴露为超参：漂移后 EMA 衰减率
        alpha_tol: float = 0.5,            # 绝对误差底线的方差乘子
        min_abs_error: float = 1e-4,       # 传感器/数据本底噪声容忍下限
        mmd_threshold: float = 0.05        # P(X) 漂移阈值
    ) -> None:
        """
        参数说明:
            window_K      : 滑动窗口大小 (需与特征提取窗口一致)
            threshold_tau : 相对恶化阈值 (如 0.2 表示相对 R_ref 恶化 20%)
            patience_C    : 确认 Actual Drift 的耐心窗口大小
            alpha_tol     : 结合初始窗口标准差的绝对误差放大系数
            mmd_threshold : 判断 Virtual Drift 的 MMD 距离阈值
        """
        assert window_K >= 2, "window_K 必须大于等于 2 以计算统计量"
        
        self.window_K = window_K
        self.threshold_tau = threshold_tau
        self.patience_C = patience_C
        
        self.alpha_stable = alpha_stable
        self.alpha_reset = alpha_reset
        self.alpha_tol = alpha_tol
        self.min_abs_error = min_abs_error
        self.mmd_threshold = mmd_threshold

        # 每 prompt 的状态容器
        self._err_windows: Dict[int, deque] = {}         # 残差滑动窗口
        self._feat_windows: Dict[int, deque] = {}        # 特征 X_t 滑动窗口
        
        self._R_ref: Dict[int, Optional[float]] = {}     # 动态参考残差
        self._X_ref: Dict[int, Optional[np.ndarray]] = {} # 参考特征分布 (用于计算 MMD)
        self._sigma_0: Dict[int, Optional[float]] = {}   # 冷启动时的残差标准差
        
        self._patience: Dict[int, int] = {}              # 连续恶化计数器

    def update_and_check(
        self, 
        residual_e: float, 
        x_t: np.ndarray, 
        prompt_idx: int
    ) -> DriftStatus:
        """
        摄入单个时刻的残差与特征，进行联合判别。
        
        返回:
            STABLE, VIRTUAL_DRIFT, 或 ACTUAL_DRIFT
        """
        if prompt_idx not in self._err_windows:
            self._init_prompt(prompt_idx)

        # 1. 更新滑动窗口
        self._err_windows[prompt_idx].append(residual_e)
        self._feat_windows[prompt_idx].append(x_t)
        
        err_win = list(self._err_windows[prompt_idx])
        feat_win = np.array(self._feat_windows[prompt_idx])

        # 窗口未满，保持 STABLE，累积数据
        if len(err_win) < self.window_K:
            return "STABLE"

        # 2. 计算当前窗口的鲁棒统计量 (90th Percentile) 替代 Mean
        R_t = np.percentile(err_win, 90)

        # 3. 冷启动逻辑：初始化参考分布与绝对误差基线
        if self._R_ref[prompt_idx] is None:
            self._R_ref[prompt_idx] = R_t
            self._X_ref[prompt_idx] = feat_win.copy()
            self._sigma_0[prompt_idx] = np.std(err_win) + 1e-8
            return "STABLE"

        R_ref = self._R_ref[prompt_idx]
        X_ref = self._X_ref[prompt_idx]
        sigma_0 = self._sigma_0[prompt_idx]

        # 4. 计算 P(X) 漂移度量 (简化版 RBF MMD 或直接欧氏距离)
        px_shifted = self._check_distribution_shift(X_ref, feat_win)

        # 5. 评估 P(Y|X) 残差恶化 (引入双重门控：相对恶化 + 绝对误差底线)
        degradation = (R_t - R_ref) / (R_ref + 1e-5)
        rel_condition = degradation > self.threshold_tau
        abs_condition = R_t > (R_ref + self.alpha_tol * sigma_0 + self.min_abs_error)
        
        py_x_degraded = rel_condition and abs_condition

        # 6. 核心路由与判别逻辑
        if py_x_degraded:
            self._patience[prompt_idx] += 1
            if self._patience[prompt_idx] >= self.patience_C:
                # Actual Drift 触发：重置状态，锚定新 regime
                self._patience[prompt_idx] = 0
                self._R_ref[prompt_idx] = self._ema(R_ref, R_t, self.alpha_reset)
                self._X_ref[prompt_idx] = feat_win.copy()
                self._sigma_0[prompt_idx] = np.std(err_win) + 1e-8
                return "ACTUAL_DRIFT"
            
            # 耐心期内，P(X) 变化则报 Virtual Drift，否则报 Stable
            return "VIRTUAL_DRIFT" if px_shifted else "STABLE"
        else:
            # 预测残差正常，清空耐心槽并平滑更新 R_ref
            self._patience[prompt_idx] = 0
            self._R_ref[prompt_idx] = self._ema(R_ref, R_t, self.alpha_stable)
            
            # 若输入分布改变但残差未恶化 -> Virtual Drift
            if px_shifted:
                # 缓慢更新参考特征分布以适应稳态演化
                self._X_ref[prompt_idx] = feat_win.copy()
                return "VIRTUAL_DRIFT"
                
            return "STABLE"

    def _check_distribution_shift(self, X_ref: np.ndarray, X_curr: np.ndarray) -> bool:
        mu_ref = np.mean(X_ref, axis=0).flatten()
        mu_curr = np.mean(X_curr, axis=0).flatten()
    
        std_ref = np.std(X_ref, axis=0).flatten()
        std_curr = np.std(X_curr, axis=0).flatten()
    
        # 结合均值偏移与方差偏移
        dist_mu = np.linalg.norm(mu_curr - mu_ref) / (np.linalg.norm(mu_ref) + 1e-5)
        dist_std = np.linalg.norm(std_curr - std_ref) / (np.linalg.norm(std_ref) + 1e-5)
    
        return (dist_mu + dist_std) > self.mmd_threshold

    def _init_prompt(self, prompt_idx: int) -> None:
        self._err_windows[prompt_idx]  = deque(maxlen=self.window_K)
        self._feat_windows[prompt_idx] = deque(maxlen=self.window_K)
        self._R_ref[prompt_idx]    = None
        self._X_ref[prompt_idx]    = None
        self._sigma_0[prompt_idx]  = None
        self._patience[prompt_idx] = 0

    @staticmethod
    def _ema(old: float, new: float, alpha: float) -> float:
        return alpha * new + (1.0 - alpha) * old