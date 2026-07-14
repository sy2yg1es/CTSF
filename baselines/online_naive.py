"""Naive online gradient descent baseline."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from .online_common import OnlineBaseline


class NaiveOnlineBaseline(OnlineBaseline):
    method = "naive"

    def update(self, x: Tensor, y: Tensor) -> Optional[float]:
        loss = self._step(x.to(self.device), y.to(self.device))
        return float(loss.item())
