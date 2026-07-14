"""Shared, leakage-free interface for CTSF online baselines.

The data stream and evaluator live in ``scripts/eval_online_baselines.py``.
Method-specific differences are deliberately restricted to ``update``,
``predict`` and ``reset_online_state``.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def forecast_patchtst(model: nn.Module, x: Tensor) -> Tensor:
    """The exact PatchTST forward path used by CTSF pretraining."""
    patches, means, stdev = model.encode_local(x)
    batch, channels, dim, patch_num = patches.shape
    encoded_in = patches.permute(0, 1, 3, 2).reshape(
        batch * channels, patch_num, dim
    )
    encoded, _ = model.encoder(encoded_in)
    encoded = encoded.reshape(batch, channels, patch_num, dim).permute(0, 1, 3, 2)
    return model.apply_prediction_head(encoded, means, stdev)


@dataclass(frozen=True)
class OnlineBaselineConfig:
    online_lr: float = 1e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clip: float = 0.0
    amp: bool = False
    update_steps: int = 1
    buffer_size: int = 500
    replay_batch_size: int = 8
    replay_weight: float = 0.2
    distill_weight: float = 0.2
    seed: int = 2025

    def to_log_dict(self) -> dict:
        result = asdict(self)
        result.update(
            optimizer="Adam",
            derpp_replay_label_weight=0.0,
            derpp_note="OnlineTSF source parity: current loss + logits distillation only",
        )
        return result


class ReservoirBuffer:
    """CPU reservoir buffer matching OnlineTSF's sampling policy."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self._rng = random.Random(seed)
        self.clear()

    def clear(self) -> None:
        self.items: list[tuple[Tensor, Tensor, Optional[Tensor]]] = []
        self.num_seen = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, x: Tensor, y: Tensor, logits: Optional[Tensor] = None) -> None:
        item = (
            x.detach().cpu().clone(),
            y.detach().cpu().clone(),
            None if logits is None else logits.detach().cpu().clone(),
        )
        index = self.num_seen
        self.num_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replacement = self._rng.randint(0, index)
        if replacement < self.capacity:
            self.items[replacement] = item

    def sample(self, size: int, device: torch.device):
        chosen = self._rng.sample(self.items, min(size, len(self.items)))
        x = torch.cat([v[0] for v in chosen], dim=0).to(device)
        y = torch.cat([v[1] for v in chosen], dim=0).to(device)
        logits = None
        if chosen and all(v[2] is not None for v in chosen):
            logits = torch.cat([v[2] for v in chosen if v[2] is not None], dim=0).to(device)
        return x, y, logits


class OnlineBaseline:
    method = "base"

    def __init__(self, backbone: nn.Module, config: OnlineBaselineConfig, device: torch.device):
        self.backbone = backbone.to(device)
        self.config = config
        self.device = device
        self._initial_state = copy.deepcopy(self.backbone.state_dict())
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.reset_online_state()

    def _new_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.backbone.parameters(),
            lr=self.config.online_lr,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )

    def reset_online_state(self) -> None:
        self.backbone.load_state_dict(self._initial_state, strict=True)
        self.backbone.eval()
        self.optimizer = None if self.method == "frozen" else self._new_optimizer()

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        self.backbone.eval()
        return forecast_patchtst(self.backbone, x.to(self.device))

    def _loss(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        outputs = forecast_patchtst(self.backbone, x)
        return F.mse_loss(outputs, y), outputs

    def _step(self, x: Tensor, y: Tensor) -> Tensor:
        assert self.optimizer is not None
        last_loss = torch.zeros((), device=self.device)
        self.backbone.train()
        for _ in range(self.config.update_steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self._loss(x, y)
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.config.grad_clip)
            self.optimizer.step()
            last_loss = loss.detach()
        return last_loss

    def update(self, x: Tensor, y: Tensor) -> Optional[float]:
        raise NotImplementedError


class FrozenBaseline(OnlineBaseline):
    method = "frozen"

    def update(self, x: Tensor, y: Tensor) -> Optional[float]:
        return None


def build_online_baseline(
    method: str,
    backbone: nn.Module,
    config: OnlineBaselineConfig,
    device: torch.device,
) -> OnlineBaseline:
    method = method.lower().replace("++", "pp")
    if method == "frozen":
        return FrozenBaseline(backbone, config, device)
    if method in {"naive", "online", "gd"}:
        from .online_naive import NaiveOnlineBaseline

        return NaiveOnlineBaseline(backbone, config, device)
    if method == "er":
        from .online_er import ERBaseline

        return ERBaseline(backbone, config, device)
    if method in {"derpp", "der"}:
        from .online_derpp import DERppBaseline

        return DERppBaseline(backbone, config, device)
    raise ValueError(f"Unknown baseline method: {method}")
