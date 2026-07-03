"""
Online Time Series Forecasting — Buffer-based Delayed Label Alignment
=====================================================================
BDLABuffer  (v3 — Channel-wise MoE / Stateless BDLA)
-----------------------------------------------------

Design contract
~~~~~~~~~~~~~~~
In online TSF with forecast horizon H, the ground-truth label Y_{t+H}
only becomes available H steps after the prediction at step t.

This buffer stores prediction events in a dict-keyed FIFO structure
so that when Y arrives at step current_t, we can retrieve the
corresponding prediction from step t = current_t - H.

What is stored per timestep
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  X_t              : [B, seq_len, C]  — input window (for forward_update)
  y_hat_future     : [B, pred_len, C] — frozen prediction (for metric eval)
  dispatch_indices : [B, C, K]        — Top-K expert indices (for force_prompt)
  z_t              : [B, C, D]        — per-channel query (for Router replay)

What is NOT stored (v3 change)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  routing_probs  — removed.  The Router is replayed from z_t at update
                   time to reconstruct a fresh computation graph.
                   Storing routing_probs would sever the graph.

Memory safety
~~~~~~~~~~~~~
All stored tensors are detached and moved to CPU immediately on push,
breaking the autograd graph and preventing GPU OOM over the H-step
horizon window.
"""

from __future__ import annotations

from torch import Tensor
from typing import Dict, Optional, Tuple


class BDLABuffer:
    """
    Buffer-based Delayed Label Alignment (BDLA).

    Public API
    ----------
    push(t, X_t, y_hat_future, dispatch_indices, z_t)
        Store a prediction event at timestep t.
    pop_and_align(current_t, Y_current) → 4-tuple or None
        Retrieve the historical record aligned with the arriving label.
    get_stored_prediction(current_t) → Tensor or None
        Return y_hat_future for the record at current_t - horizon_H
        WITHOUT removing it from the buffer (non-destructive read).
    """

    def __init__(self, horizon_H: int) -> None:
        """
        Parameters
        ----------
        horizon_H : int
            Forecast horizon.  A prediction pushed at step t will be
            aligned when ``pop_and_align`` is called at step t + H.
        """
        assert horizon_H > 0, "horizon_H must be a positive integer"
        self.horizon_H = horizon_H
        # Dict[timestep → record] — sparse FIFO keyed by integer timestep
        self._buffer: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def push(
        self,
        t: int,
        X_t: Tensor,
        y_hat_future: Tensor,
        dispatch_indices: Tensor,
        z_t: Tensor,
    ) -> None:
        """
        Store a prediction event made at timestep t.

        All tensors are immediately detached and moved to CPU to:
          • Break the autograd computation graph (no retained grads).
          • Prevent GPU memory accumulation over the H-step window.

        Parameters
        ----------
        t                : int
            Timestep at which the prediction was made.
        X_t              : Tensor [B, seq_len, C]
            Input window — needed by forward_update at step t+H.
        y_hat_future     : Tensor [B, pred_len, C]
            Frozen model prediction for timestep t+H.
            Stored for pre-update metric evaluation.
        dispatch_indices : Tensor [B, C, K]
            Top-K expert indices from the MoE Router at step t.
            Stored so force_prompt can reproduce the exact routing
            decision at step t+H without any internal caches.
        z_t              : Tensor [B, C, D]
            Per-channel query embedding from ChannelFusion at step t.
            Passed to force_prompt at step t+H to reconstruct the
            Router's computation graph (stateless design).
        """
        self._buffer[t] = {
            "X_t":              X_t.detach(),
            "y_hat_future":     y_hat_future.detach(),
            "dispatch_indices": dispatch_indices.detach(),
            "z_t":              z_t.detach(),
        }

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def pop_and_align(
        self,
        current_t: int,
        Y_current: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        Align the newly arrived ground-truth label with its historical
        prediction and pop the record from the buffer.

        When Y_current arrives at step current_t, the corresponding
        prediction was made at step ``history_t = current_t - horizon_H``.
        The record is removed after retrieval (FIFO pop semantics) to
        immediately free CPU memory.

        Parameters
        ----------
        current_t : int
            The timestep at which Y_current has just become available.
        Y_current : Tensor [B, pred_len, C]
            Ground-truth label for the forecast window ending at current_t.

        Returns
        -------
        4-tuple ``(X_history, Y_current_cpu, dispatch_indices_hist, z_query_hist)``
        if a record exists for ``current_t - horizon_H``, otherwise ``None``.

        Return shapes
        -------------
        X_history             : [B, seq_len, C]  — historical input window
        Y_current_cpu         : [B, pred_len, C] — ground-truth (CPU copy)
        dispatch_indices_hist : [B, C, K]         — Top-K experts at step t-H
        z_query_hist          : [B, C, D]         — query embedding at step t-H
        """
        history_t = current_t - self.horizon_H

        if history_t not in self._buffer:
            return None

        # Pop the record immediately to free memory
        record = self._buffer.pop(history_t)

        Y_current_detached = Y_current.detach()

        return (
            record["X_t"],               # [B, seq_len, C]
            Y_current_detached,          # [B, pred_len, C]
            record["dispatch_indices"],   # [B, C, K]
            record["z_t"],               # [B, C, D]
        )

    def get_stored_prediction(self, current_t: int) -> Optional[Tensor]:
        """
        Non-destructive read: return the stored ``y_hat_future`` for the
        record at ``current_t - horizon_H`` WITHOUT removing it.

        Used by the streaming loop to retrieve the frozen prediction for
        pre-update metric evaluation BEFORE ``pop_and_align`` consumes
        the record.

        Parameters
        ----------
        current_t : int
            The current timestep (same value passed to pop_and_align).

        Returns
        -------
        y_hat_future : Tensor [B, pred_len, C] on CPU, or None if no
                       record exists for ``current_t - horizon_H``.
        """
        history_t = current_t - self.horizon_H
        record = self._buffer.get(history_t, None)
        if record is None:
            return None
        return record["y_hat_future"]     # [B, pred_len, C] — detached, CPU

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_residual(y_hat: Tensor, y_true: Tensor) -> Tensor:
        """
        Element-wise absolute residual  e = |y_hat − y_true|.

        Parameters
        ----------
        y_hat  : Tensor — predicted values (any shape).
        y_true : Tensor — ground-truth values (same shape as y_hat).

        Returns
        -------
        Tensor of the same shape as inputs.
        """
        return (y_hat - y_true).abs()

    def __len__(self) -> int:
        """Number of unaligned prediction records currently in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"BDLABuffer(horizon_H={self.horizon_H}, "
            f"buffered_steps={len(self._buffer)})"
        )
