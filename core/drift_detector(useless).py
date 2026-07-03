"""
Online Time Series Forecasting — Per-prompt Dynamic Drift Detection
===================================================================
ActualDriftDetector: sliding-window EMA drift detector per prompt slot.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional


class ActualDriftDetector:
    """
    Per-prompt dynamic drift detector for online TSF.

    Maintains a sliding window of the last window_K residuals per prompt
    to compute the current error rate R_t.  Compares R_t against a
    per-prompt dynamic reference R_ref (exponential moving average of
    stable-regime errors).  Drift is confirmed only after patience_C
    consecutive steps where the relative degradation exceeds threshold_tau.

    Design rationale for R_ref:
      - Initialized to the first observed R_t for a new prompt (graceful cold start).
      - Updated via slow EMA (alpha=0.05) when no degradation is detected,
        so R_ref tracks long-term stable performance without chasing noise.
      - NOT updated while the patience counter is incrementing, so R_ref
        stays anchored to the pre-drift baseline during the detection window.
      - After drift is confirmed, updated with a faster EMA (alpha=0.3) to
        re-anchor to the new regime.
    """

    # EMA learning rates
    _ALPHA_STABLE: float = 0.05   # slow adaptation during stable regime
    _ALPHA_RESET:  float = 0.30   # faster re-anchor after confirmed drift

    def __init__(
        self,
        window_K: int,
        threshold_tau: float,
        patience_C: int,
    ) -> None:
        """
        Args:
            window_K      : Sliding window size for computing R_t.
            threshold_tau : Relative degradation ratio to flag a step as
                            degraded.  E.g., 0.2 → 20 % worse than R_ref.
            patience_C    : Consecutive degraded steps required to confirm
                            actual drift and return True.
        """
        assert window_K >= 1,     "window_K must be >= 1"
        assert threshold_tau > 0, "threshold_tau must be positive"
        assert patience_C >= 1,   "patience_C must be >= 1"

        self.window_K      = window_K
        self.threshold_tau = threshold_tau
        self.patience_C    = patience_C

        # Per-prompt state containers
        self._windows:  Dict[int, deque]           = {}  # sliding residual windows
        self._R_ref:    Dict[int, Optional[float]] = {}  # dynamic reference errors
        self._patience: Dict[int, int]             = {}  # consecutive-degradation counters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_and_check(self, residual_e: float, prompt_idx: int) -> bool:
        """
        Ingest one residual observation and check for actual drift.

        Steps:
          1. Append residual_e to the sliding window for prompt_idx.
          2. Compute R_t = mean of the window.
          3. If R_ref is uninitialized (new prompt), set R_ref = R_t and
             return False (insufficient history).
          4. Evaluate degradation condition:
               (R_t - R_ref) / (R_ref + 1e-5) > threshold_tau
          5. If condition holds, increment patience counter.
             If patience counter reaches patience_C → drift confirmed:
               - Reset patience counter.
               - Re-anchor R_ref toward the new error level.
               - Return True.
          6. If condition does not hold, reset patience counter and
             slowly update R_ref via EMA.
             Return False.

        Args:
            residual_e : Absolute residual |y_hat - y_true| for this step.
            prompt_idx : Index of the active prompt / regime.

        Returns:
            True  — actual drift confirmed (patience_C consecutive degraded steps).
            False — no drift (yet).
        """
        if prompt_idx not in self._windows:
            self._init_prompt(prompt_idx)

        # 1. Update sliding window
        self._windows[prompt_idx].append(residual_e)
        window = self._windows[prompt_idx]

        # 2. Compute current sliding mean R_t
        R_t = sum(window) / len(window)

        # 3. Cold-start: initialize R_ref on first observation
        if self._R_ref[prompt_idx] is None:
            self._R_ref[prompt_idx] = R_t
            return False

        R_ref = self._R_ref[prompt_idx]

        # 4. Evaluate relative degradation condition
        degradation = (R_t - R_ref) / (R_ref + 1e-5)
        condition_met = degradation > self.threshold_tau

        if condition_met:
            # 5a. Increment patience counter
            self._patience[prompt_idx] += 1

            if self._patience[prompt_idx] >= self.patience_C:
                # Drift confirmed
                self._patience[prompt_idx] = 0
                self._R_ref[prompt_idx] = self._ema(R_ref, R_t, self._ALPHA_RESET)
                return True
            # Patience not yet exhausted — hold R_ref fixed
        else:
            # 6. Stable step: reset patience, slowly update R_ref
            self._patience[prompt_idx] = 0
            self._R_ref[prompt_idx] = self._ema(R_ref, R_t, self._ALPHA_STABLE)

        return False

    def get_R_ref(self, prompt_idx: int) -> Optional[float]:
        """Return the current dynamic reference error for a prompt."""
        return self._R_ref.get(prompt_idx)

    def get_patience(self, prompt_idx: int) -> int:
        """Return the current patience counter for a prompt."""
        return self._patience.get(prompt_idx, 0)

    def reset_prompt(self, prompt_idx: int) -> None:
        """
        Fully reset the state for a prompt (e.g., after an explicit regime switch).
        The next call to update_and_check will re-initialize it gracefully.
        """
        self._windows.pop(prompt_idx, None)
        self._R_ref.pop(prompt_idx, None)
        self._patience.pop(prompt_idx, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_prompt(self, prompt_idx: int) -> None:
        """Lazily initialize per-prompt state for a newly seen prompt."""
        self._windows[prompt_idx]  = deque(maxlen=self.window_K)
        self._R_ref[prompt_idx]    = None
        self._patience[prompt_idx] = 0

    @staticmethod
    def _ema(old: float, new: float, alpha: float) -> float:
        """Exponential moving average:  alpha * new + (1 - alpha) * old."""
        return alpha * new + (1.0 - alpha) * old

    def __repr__(self) -> str:
        return (
            f"ActualDriftDetector("
            f"window_K={self.window_K}, "
            f"threshold_tau={self.threshold_tau}, "
            f"patience_C={self.patience_C}, "
            f"active_prompts_tracked={len(self._windows)})"
        )
