"""Experience Replay, preserving OnlineTSF's ER mechanism."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .online_common import OnlineBaseline, ReservoirBuffer, forecast_patchtst


class ERBaseline(OnlineBaseline):
    method = "er"

    def reset_online_state(self) -> None:
        super().reset_online_state()
        self.buffer = ReservoirBuffer(self.config.buffer_size, self.config.seed)

    def update(self, x: Tensor, y: Tensor) -> Optional[float]:
        assert self.optimizer is not None
        x, y = x.to(self.device), y.to(self.device)
        self.backbone.train()
        last_loss = torch.zeros((), device=self.device)
        for _ in range(self.config.update_steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self._loss(x, y)
            if len(self.buffer):
                replay_x, replay_y, _ = self.buffer.sample(
                    self.config.replay_batch_size, self.device
                )
                replay_out = forecast_patchtst(self.backbone, replay_x)
                loss = loss + self.config.replay_weight * F.mse_loss(replay_out, replay_y)
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.config.grad_clip)
            self.optimizer.step()
            last_loss = loss.detach()
        self.buffer.add(x, y)
        return float(last_loss.item())
