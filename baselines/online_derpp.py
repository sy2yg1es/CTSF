"""OnlineTSF DERpp: reservoir replay of historical prediction logits."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .online_common import OnlineBaseline, ReservoirBuffer, forecast_patchtst


class DERppBaseline(OnlineBaseline):
    method = "derpp"

    def reset_online_state(self) -> None:
        super().reset_online_state()
        self.buffer = ReservoirBuffer(self.config.buffer_size, self.config.seed)

    def update(self, x: Tensor, y: Tensor) -> Optional[float]:
        assert self.optimizer is not None
        x, y = x.to(self.device), y.to(self.device)
        self.backbone.train()
        last_loss = torch.zeros((), device=self.device)
        stored_logits = None
        for _ in range(self.config.update_steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss, outputs = self._loss(x, y)
            if stored_logits is None:
                stored_logits = outputs.detach()
            if len(self.buffer):
                replay_x, _replay_y, replay_logits = self.buffer.sample(
                    self.config.replay_batch_size, self.device
                )
                assert replay_logits is not None
                replay_out = forecast_patchtst(self.backbone, replay_x)
                loss = loss + self.config.distill_weight * F.mse_loss(
                    replay_out, replay_logits
                )
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.config.grad_clip)
            self.optimizer.step()
            last_loss = loss.detach()
        assert stored_logits is not None
        self.buffer.add(x, y, stored_logits)
        return float(last_loss.item())
