"""
train_gamma_final.py — Two-Phase Delta Warmup + Gamma Release
==============================================================

目的：在 mask 恒为 1 的条件下，先验证 delta 是否能学到稳定修正方向，
再验证 gamma 是否能在此基础上学习动态开关。

实验口径：
  Phase 1 — Delta warmup
    * gamma 固定为 1
    * 只训练 drift_encoder + low_rank_mod
    * lambda_delta = 0
    * best checkpoint 仅按验证集 full-on(gamma=1) MSE 选择

  Phase 2 — Gamma release
    * 加载 Phase-1 validation-best checkpoint
    * gamma = confidence_gate(drift_state)
    * 训练 drift_encoder + confidence_gate + low_rank_mod
    * mask 仍恒为 1；lambda_mask = lambda_noop = 0
    * best checkpoint 仅按验证集 learned-gamma MSE 选择
    * 同时评估 fixed gamma=1，判断 gate 是否真正带来价值

重要：
  * frozen backbone 权重必须完整严格匹配，否则直接报错
  * 默认使用完整 validation split；--val_max_windows 仅用于 smoke test
  * validation 用于开发与 checkpoint 选择，最终论文结果必须在未参与调参的 test 上单独评估

示例：
    python scripts/train_gamma_final.py \
        --root_path ./dataset \
        --data_path ETTm1.csv \
        --forecast_H 1 \
        --pretrained_weights weights/patchtst_pretrained_ETTm1_H1.pth \
        --backbone patchtst \
        --phase1_steps 1000 \
        --phase2_steps 2000 \
        --max_delta_ratio 0.02 \
        --val_max_windows 2000 \
        --experiment_tag smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from core.residual_tracker import ResidualTracker
from data_provider.data_loader import data_provider
from models.backbone_adapter import PatchTSTAdapter, iTransformerAdapter
from models.prompt_z import PromptZModulator


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Checkpoint helpers
# ============================================================================

def safe_torch_load(path: str, device: torch.device) -> Any:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_state_dict(state: Any) -> dict[str, Tensor]:
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint 必须是 dict，实际为 {type(state).__name__}")

    for key in (
        "state_dict",
        "model_state_dict",
        "backbone_state_dict",
        "prompt_z_state_dict",
        "model",
        "backbone",
        "prompt_z",
    ):
        value = state.get(key)
        if isinstance(value, dict) and value:
            return value
    return state


def extract_matching_state_dict(
    state: Any,
    target_state: dict[str, Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, Tensor]:
    source = unwrap_state_dict(state)
    matched: dict[str, Tensor] = {}

    for raw_key, value in source.items():
        if not isinstance(raw_key, str) or not isinstance(value, Tensor):
            continue

        candidates = [raw_key]
        if raw_key.startswith("module."):
            candidates.append(raw_key[len("module."):])

        expanded = list(candidates)
        for candidate in candidates:
            for prefix in prefixes:
                if candidate.startswith(prefix):
                    expanded.append(candidate[len(prefix):])

        for candidate in expanded:
            if (
                candidate in target_state
                and target_state[candidate].shape == value.shape
            ):
                matched[candidate] = value
                break

    return matched


def load_prompt_z_state(
    prompt_z: PromptZModulator,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    state = unwrap_state_dict(safe_torch_load(checkpoint_path, device))
    missing, unexpected = prompt_z.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Prompt-Z checkpoint 与当前结构不一致: "
            f"missing={missing[:10]} (共 {len(missing)})，"
            f"unexpected={unexpected[:10]} (共 {len(unexpected)})"
        )


# ============================================================================
# Backbone — strict loading
# ============================================================================

def build_backbone(args: argparse.Namespace, device: torch.device):
    if args.backbone == "patchtst":
        from models.backbones.PatchTST import Model as PatchTST

        class Cfg:
            pass

        cfg = Cfg()
        cfg.task_name = "long_term_forecast"
        cfg.seq_len = args.seq_len
        cfg.pred_len = args.forecast_H
        cfg.d_model = args.D_model
        cfg.d_ff = args.d_ff
        cfg.n_heads = 8
        cfg.e_layers = args.e_layers
        cfg.dropout = 0.1
        cfg.activation = "gelu"
        cfg.factor = 1
        cfg.enc_in = args.enc_in
        backbone = PatchTST(cfg).to(device)
        adapter = PatchTSTAdapter(backbone).to(device)

    elif args.backbone == "itransformer":
        from models.backbones.iTransformer import Model as iTransformer

        class Cfg:
            pass

        cfg = Cfg()
        cfg.task_name = "long_term_forecast"
        cfg.seq_len = args.seq_len
        cfg.pred_len = args.forecast_H
        cfg.d_model = args.D_model
        cfg.d_ff = args.d_ff
        cfg.n_heads = 8
        cfg.e_layers = args.e_layers
        cfg.dropout = 0.1
        cfg.activation = "gelu"
        cfg.factor = 1
        cfg.enc_in = args.enc_in
        cfg.output_attention = False
        cfg.embed = "timeF"
        cfg.freq = "h"
        backbone = iTransformer(cfg).to(device)
        adapter = iTransformerAdapter(backbone).to(device)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    state = safe_torch_load(args.pretrained_weights, device)
    target_state = adapter.backbone.state_dict()
    backbone_state = extract_matching_state_dict(
        state,
        target_state,
        prefixes=(
            "backbone_adapter.backbone.",
            "adapter.backbone.",
            "backbone.",
            "model.backbone.",
            "model.",
            "network.",
        ),
    )

    if not backbone_state:
        raise RuntimeError(
            "没有从 checkpoint 中找到任何匹配的 backbone 参数。"
            "请检查 backbone、H、D_model、e_layers、enc_in 和权重路径。"
        )

    missing, unexpected = adapter.backbone.load_state_dict(
        backbone_state,
        strict=False,
    )
    matched_ratio = len(backbone_state) / max(1, len(target_state))
    print(
        f"[*] 已加载 backbone 权重: {args.pretrained_weights} "
        f"({len(backbone_state)}/{len(target_state)} keys, {matched_ratio:.1%})"
    )

    if missing or unexpected or len(backbone_state) != len(target_state):
        raise RuntimeError(
            "Backbone 权重与当前结构不完全一致："
            f"matched={len(backbone_state)}/{len(target_state)}，"
            f"missing={missing[:10]} (共 {len(missing)})，"
            f"unexpected={unexpected[:10]} (共 {len(unexpected)})。"
            "实验必须使用与预训练完全一致的 frozen backbone。"
        )

    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()
    return adapter


# ============================================================================
# Data
# ============================================================================

def build_dataloaders(args: argparse.Namespace):
    class DPArgs:
        pass

    dp = DPArgs()
    dp.root_path = args.root_path
    dp.data_path = args.data_path
    dp.features = args.features
    dp.seq_len = args.seq_len
    dp.pred_len = args.forecast_H
    dp.target = "OT"
    dp.num_workers = args.num_workers
    dp.train_ratio = args.train_ratio
    dp.val_ratio = args.val_ratio

    dataset, _ = data_provider(dp)

    train_start, train_end = 0, dataset.train_size
    val_start, val_end = dataset.val_start, dataset.test_start
    if args.val_max_windows > 0:
        val_end = min(val_end, val_start + args.val_max_windows)

    if train_end <= train_start:
        raise ValueError("训练集为空。")
    if val_end <= val_start:
        raise ValueError(
            f"验证集范围无效: [{val_start}, {val_end})，"
            f"dataset.test_start={dataset.test_start}"
        )

    train_loader = DataLoader(
        Subset(dataset, range(train_start, train_end)),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, range(val_start, val_end)),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f"[*] Train windows: [{train_start}, {train_end}) = {train_end-train_start}")
    print(
        f"[*] Val windows  : [{val_start}, {val_end}) = {val_end-val_start}"
        + (
            " (limited by --val_max_windows; smoke only)"
            if args.val_max_windows > 0
            else " (full validation split)"
        )
    )
    return train_loader, val_loader


# ============================================================================
# Streaming helpers
# ============================================================================

def pack_residual_stats(
    tracker: ResidualTracker,
    device: torch.device,
) -> Tensor:
    stats = tracker.get_stats()
    return torch.stack(
        [
            stats["error_mean"].to(device),
            stats["error_slope"].to(device),
            stats["error_std"].to(device),
            stats["signed_bias"].to(device),
            stats["steps_gap"].to(device).expand(tracker.C),
        ],
        dim=-1,
    )


def advance_tracker(
    tracker: ResidualTracker,
    residual_cache: deque,
    frozen_pred: Tensor,
    y_true: Tensor,
    forecast_h: int,
) -> None:
    residual_cache.append((frozen_pred.detach(), y_true.detach()))
    if len(residual_cache) > forecast_h:
        old_pred, old_true = residual_cache.popleft()
        tracker.update(old_pred, old_true)
    else:
        tracker.step_no_update()


# ============================================================================
# Forward
# ============================================================================

def prompt_forward(
    X: Tensor,
    adapter,
    prompt_z: PromptZModulator,
    stats_tensor: Tensor,
    use_learned_gamma: bool,
) -> dict[str, Tensor]:
    """
    mask 始终等价于 1。

    use_learned_gamma=False: Phase 1，实际预测使用 gamma=1。
    use_learned_gamma=True : Phase 2，实际预测使用 confidence gate。

    同时返回 full_on_pred，供 validation 中与 learned gamma 公平比较。
    """
    with torch.no_grad():
        hidden, means, stdev = adapter.encode_until_hook(X)
        hidden = hidden.detach()
        means = means.detach()
        stdev = stdev.detach()
        frozen_pred = adapter.decode_from_hook(hidden, means, stdev)

    summary = prompt_z._hidden_summary(hidden)
    drift_state = prompt_z.drift_encoder(summary, stats_tensor)

    if use_learned_gamma:
        gamma = prompt_z.confidence_gate(drift_state)
    else:
        gamma = torch.ones(
            (*drift_state.shape[:2], 1),
            device=drift_state.device,
            dtype=drift_state.dtype,
        )

    layout = prompt_z.hidden_layout
    if layout == "BCDP":
        h_work = hidden.permute(0, 1, 3, 2)
        delta_h = prompt_z.low_rank_mod(h_work, drift_state)
        delta_h = delta_h.permute(0, 1, 3, 2)
    else:
        delta_h = prompt_z.low_rank_mod(hidden, drift_state)

    delta_h = prompt_z._ratio_clamp(delta_h, hidden)

    if layout == "BCDP":
        applied = gamma.unsqueeze(-1) * delta_h
    else:
        applied = gamma * delta_h

    pred = adapter.decode_from_hook(hidden + applied, means, stdev)
    full_on_pred = adapter.decode_from_hook(hidden + delta_h, means, stdev)

    delta_flat = delta_h.flatten(2)
    hidden_flat = hidden.flatten(2)
    hidden_norm = hidden_flat.norm(dim=-1).clamp(min=1e-8)
    raw_ratio_per_channel = delta_flat.norm(dim=-1) / hidden_norm
    effective_ratio_per_channel = applied.flatten(2).norm(dim=-1) / hidden_norm

    return {
        "pred": pred,
        "full_on_pred": full_on_pred,
        "frozen_pred": frozen_pred,
        "gamma": gamma,
        "delta_h": delta_h,
        "hidden": hidden,
        "raw_ratio_per_channel": raw_ratio_per_channel,
        "effective_ratio_per_channel": effective_ratio_per_channel,
    }


# ============================================================================
# Validation — always computes learned gamma and fixed gamma=1
# ============================================================================

@torch.no_grad()
def evaluate_validation(
    args: argparse.Namespace,
    adapter,
    prompt_z: PromptZModulator,
    val_loader: DataLoader,
    device: torch.device,
    phase: int,
    step: int,
) -> dict[str, float]:
    was_training = prompt_z.training
    prompt_z.eval()

    tracker = ResidualTracker(
        num_channels=args.enc_in,
        window_K=args.residual_window_K,
    ).to(device)
    tracker.reset()
    residual_cache: deque = deque()

    squared_error_frozen = 0.0
    squared_error_learned = 0.0
    squared_error_full_on = 0.0
    n_elements = 0
    n_windows = 0

    gamma_sum = 0.0
    gamma_sq_sum = 0.0
    gamma_count = 0
    gamma_min = float("inf")
    gamma_max = float("-inf")
    channel_gamma_lt01 = 0
    channel_gamma_gt09 = 0
    window_gamma_means: list[float] = []
    within_window_spread_sum = 0.0

    raw_ratio_sum = 0.0
    effective_ratio_sum = 0.0
    ratio_count = 0
    raw_at_cap_count = 0

    for X, Y in val_loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        stats_tensor = pack_residual_stats(tracker, device)

        # Validation always evaluates the learned gate and the full-on control.
        out = prompt_forward(
            X,
            adapter,
            prompt_z,
            stats_tensor,
            use_learned_gamma=True,
        )

        frozen_pred = out["frozen_pred"]
        learned_pred = out["pred"]
        full_on_pred = out["full_on_pred"]
        gamma = out["gamma"].detach()
        raw_ratio = out["raw_ratio_per_channel"].detach()
        effective_ratio = out["effective_ratio_per_channel"].detach()

        squared_error_frozen += (frozen_pred - Y).pow(2).sum().item()
        squared_error_learned += (learned_pred - Y).pow(2).sum().item()
        squared_error_full_on += (full_on_pred - Y).pow(2).sum().item()
        n_elements += Y.numel()
        n_windows += X.shape[0]

        gamma_sum += gamma.sum().item()
        gamma_sq_sum += gamma.pow(2).sum().item()
        gamma_count += gamma.numel()
        gamma_min = min(gamma_min, gamma.min().item())
        gamma_max = max(gamma_max, gamma.max().item())
        channel_gamma_lt01 += int((gamma < 0.1).sum().item())
        channel_gamma_gt09 += int((gamma > 0.9).sum().item())
        window_gamma_means.append(gamma.mean().item())
        within_window_spread_sum += gamma.std(dim=1, unbiased=False).mean().item()

        raw_ratio_sum += raw_ratio.sum().item()
        effective_ratio_sum += effective_ratio.sum().item()
        ratio_count += raw_ratio.numel()
        raw_at_cap_count += int(
            (raw_ratio >= args.max_delta_ratio * 0.999).sum().item()
        )

        advance_tracker(
            tracker,
            residual_cache,
            frozen_pred,
            Y,
            args.forecast_H,
        )

    if n_elements == 0 or n_windows == 0 or gamma_count == 0:
        raise RuntimeError("验证集为空，无法评估。")

    frozen_mse = squared_error_frozen / n_elements
    learned_mse = squared_error_learned / n_elements
    full_on_mse = squared_error_full_on / n_elements

    def improvement_pct(baseline: float, value: float) -> float:
        return (baseline - value) / max(baseline, 1e-12) * 100.0

    gamma_mean = gamma_sum / gamma_count
    gamma_variance = max(gamma_sq_sum / gamma_count - gamma_mean**2, 0.0)
    window_gamma_tensor = torch.tensor(window_gamma_means, dtype=torch.float32)

    metrics = {
        "phase": float(phase),
        "step": float(step),
        "val_windows": float(n_windows),
        "frozen_mse": frozen_mse,
        "learned_gamma_mse": learned_mse,
        "full_on_mse": full_on_mse,
        "learned_improvement_pct": improvement_pct(frozen_mse, learned_mse),
        "full_on_improvement_pct": improvement_pct(frozen_mse, full_on_mse),
        "gamma_mean_all_channels": gamma_mean,
        "gamma_std_all_channels": gamma_variance**0.5,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "channel_fraction_gamma_lt01": channel_gamma_lt01 / gamma_count,
        "channel_fraction_gamma_gt09": channel_gamma_gt09 / gamma_count,
        "window_gamma_mean": window_gamma_tensor.mean().item(),
        "window_gamma_std": window_gamma_tensor.std(unbiased=False).item(),
        "window_fraction_mean_gamma_lt01": (
            (window_gamma_tensor < 0.1).float().mean().item()
        ),
        "window_fraction_mean_gamma_gt09": (
            (window_gamma_tensor > 0.9).float().mean().item()
        ),
        "mean_within_window_channel_spread": (
            within_window_spread_sum / n_windows
        ),
        "mean_raw_d2h": raw_ratio_sum / ratio_count,
        "mean_effective_d2h": effective_ratio_sum / ratio_count,
        "channel_fraction_raw_d2h_at_cap": raw_at_cap_count / ratio_count,
    }

    print("\n" + "=" * 76)
    print(f"  Validation | Phase {phase} | step {step} | {n_windows} windows")
    print("=" * 76)
    print(f"  Frozen MSE                         : {frozen_mse:.6f}")
    print(f"  Learned-gamma MSE                 : {learned_mse:.6f}")
    print(f"  Learned-gamma improvement         : {metrics['learned_improvement_pct']:+.2f}%")
    print(f"  Full-on gamma=1 MSE               : {full_on_mse:.6f}")
    print(f"  Full-on improvement               : {metrics['full_on_improvement_pct']:+.2f}%")
    print()
    print(
        "  Gamma mean/std over channels      : "
        f"{metrics['gamma_mean_all_channels']:.4f} / "
        f"{metrics['gamma_std_all_channels']:.4f}"
    )
    print(f"  Gamma min/max                     : {gamma_min:.4f} / {gamma_max:.4f}")
    print(
        "  Channel gate values <0.1 / >0.9  : "
        f"{metrics['channel_fraction_gamma_lt01']*100:.1f}% / "
        f"{metrics['channel_fraction_gamma_gt09']*100:.1f}%"
    )
    print(
        "  Window mean-gamma <0.1 / >0.9    : "
        f"{metrics['window_fraction_mean_gamma_lt01']*100:.1f}% / "
        f"{metrics['window_fraction_mean_gamma_gt09']*100:.1f}%"
    )
    print(f"  Std of window mean-gamma          : {metrics['window_gamma_std']:.4f}")
    print(
        "  Mean within-window channel spread: "
        f"{metrics['mean_within_window_channel_spread']:.4f}"
    )
    print(f"  Mean raw d/h                      : {metrics['mean_raw_d2h']:.5f}")
    print(f"  Mean effective d/h                : {metrics['mean_effective_d2h']:.5f}")
    print(
        "  Channel raw d/h at clamp cap     : "
        f"{metrics['channel_fraction_raw_d2h_at_cap']*100:.1f}%"
    )
    print("=" * 76)

    if was_training:
        prompt_z.train()
    return metrics


def print_phase2_diagnosis(metrics: dict[str, float]) -> None:
    learned_impr = metrics["learned_improvement_pct"]
    full_on_impr = metrics["full_on_improvement_pct"]
    channel_closed = metrics["channel_fraction_gamma_lt01"]
    channel_open = metrics["channel_fraction_gamma_gt09"]
    window_closed = metrics["window_fraction_mean_gamma_lt01"]
    window_open = metrics["window_fraction_mean_gamma_gt09"]
    window_std = metrics["window_gamma_std"]
    spread = metrics["mean_within_window_channel_spread"]
    at_cap = metrics["channel_fraction_raw_d2h_at_cap"]

    print("\n── Validation-based Phase-2 diagnosis ─────────────────────────")
    if learned_impr > 0.5:
        print(f"  ✓ Learned gamma 在验证集提升 {learned_impr:+.2f}%。")
    elif learned_impr >= 0.0:
        print(f"  △ Learned gamma 在验证集仅小幅提升 {learned_impr:+.2f}%。")
    else:
        print(f"  ✗ Learned gamma 在验证集退化 {learned_impr:+.2f}%。")

    if learned_impr > full_on_impr + 0.2:
        print("  ✓ Learned gamma 明显优于固定 gamma=1，gate 具有实际选择价值。")
    elif abs(learned_impr - full_on_impr) <= 0.2:
        print("  △ Learned gamma 与固定 gamma=1 接近，gate 尚未产生明显价值。")
    else:
        print("  ✗ 固定 gamma=1 更好，当前 gate 抑制了有效 delta。")

    if channel_open > 0.9 and window_open > 0.9:
        print("  ✗ Gamma 几乎 always-on：通道级和窗口级统计都接近全开。")
    elif channel_closed > 0.9 and window_closed > 0.9:
        print("  ✗ Gamma collapse：通道级和窗口级统计都接近关闭。")
    elif window_std > 0.05 or spread > 0.05:
        print("  ✓ Gamma 存在窗口间或通道间动态差异。")
    else:
        print("  △ Gamma 动态性较弱。")

    if at_cap > 0.8:
        print("  ✗ Delta 超过 80% 通道长期触及 max_delta_ratio 上限。")
    elif at_cap > 0.3:
        print("  △ 较多通道的 delta 触及幅度上限。")
    print("────────────────────────────────────────────────────────────────")


# ============================================================================
# Logging / optimizer checks
# ============================================================================

def write_jsonl(file_obj, row: dict[str, Any]) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_obj.flush()


def unique_parameters(modules: Iterable[torch.nn.Module]) -> list[Tensor]:
    result: list[Tensor] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
    return result


def assert_sparse_mask_excluded(
    optimizer_params: list[Tensor],
    prompt_z: PromptZModulator,
) -> None:
    optimizer_ids = {id(parameter) for parameter in optimizer_params}
    mask_ids = {id(parameter) for parameter in prompt_z.sparse_mask.parameters()}
    overlap = optimizer_ids.intersection(mask_ids)
    if overlap:
        raise RuntimeError("SparseMaskHead 参数意外进入 optimizer。")


# ============================================================================
# Phase 1 — delta warmup, validation-best by full-on MSE
# ============================================================================

def run_phase1(
    args: argparse.Namespace,
    adapter,
    prompt_z: PromptZModulator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    paths: dict[str, str],
) -> tuple[str, dict[str, float]]:
    print("\n" + "=" * 76)
    print("  Phase 1 — Delta warmup")
    print(f"  steps={args.phase1_steps}, gamma=1, max_delta_ratio={args.max_delta_ratio}")
    print("  train: drift_encoder + low_rank_mod; lambda_delta=0")
    print("  best: validation full-on(gamma=1) MSE")
    print("=" * 76)

    params = unique_parameters([
        prompt_z.drift_encoder,
        prompt_z.low_rank_mod,
    ])
    assert_sparse_mask_excluded(params, prompt_z)
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    tracker = ResidualTracker(
        num_channels=args.enc_in,
        window_K=args.residual_window_K,
    ).to(device)
    tracker.reset()
    residual_cache: deque = deque()

    best_val = float("inf")
    best_metrics: dict[str, float] | None = None
    step = 0
    epoch = 0
    t0 = time.time()

    with open(paths["p1_train_log"], "w", encoding="utf-8") as train_log, open(
        paths["p1_val_log"], "w", encoding="utf-8"
    ) as val_log:
        while step < args.phase1_steps:
            tracker.reset()
            residual_cache.clear()

            for X, Y in train_loader:
                if step >= args.phase1_steps:
                    break

                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                stats_tensor = pack_residual_stats(tracker, device)

                optimizer.zero_grad(set_to_none=True)
                out = prompt_forward(
                    X,
                    adapter,
                    prompt_z,
                    stats_tensor,
                    use_learned_gamma=False,
                )
                forecast_loss = F.mse_loss(out["pred"], Y)
                forecast_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
                optimizer.step()

                with torch.no_grad():
                    frozen_loss = F.mse_loss(out["frozen_pred"], Y).item()
                    raw_ratio = out["raw_ratio_per_channel"].mean().item()
                    improvement = (
                        (frozen_loss - forecast_loss.item())
                        / max(frozen_loss, 1e-12)
                        * 100.0
                    )
                    row = {
                        "phase": 1,
                        "step": step + 1,
                        "epoch": epoch,
                        "forecast_loss": forecast_loss.item(),
                        "frozen_loss": frozen_loss,
                        "step_improvement_pct": improvement,
                        "raw_d2h": raw_ratio,
                    }
                    write_jsonl(train_log, row)

                advance_tracker(
                    tracker,
                    residual_cache,
                    out["frozen_pred"],
                    Y,
                    args.forecast_H,
                )

                step += 1
                if step == 1 or step % args.log_interval == 0:
                    print(
                        f"[P1|{step:5d}] loss={forecast_loss.item():.6f} "
                        f"frozen={frozen_loss:.6f} "
                        f"step_impr={improvement:+.2f}% raw_d/h={raw_ratio:.5f}"
                    )

                should_validate = (
                    step % args.phase1_val_interval == 0
                    or step == args.phase1_steps
                )
                if should_validate:
                    metrics = evaluate_validation(
                        args,
                        adapter,
                        prompt_z,
                        val_loader,
                        device,
                        phase=1,
                        step=step,
                    )
                    write_jsonl(val_log, metrics)
                    selection_mse = metrics["full_on_mse"]
                    if selection_mse < best_val:
                        best_val = selection_mse
                        best_metrics = metrics
                        torch.save(prompt_z.state_dict(), paths["p1_best"])
                        print(
                            f"[*] Phase-1 new validation best: {best_val:.6f} "
                            f"-> {paths['p1_best']}"
                        )

            epoch += 1

    torch.save(prompt_z.state_dict(), paths["p1_final"])
    if best_metrics is None or not os.path.isfile(paths["p1_best"]):
        raise RuntimeError("Phase 1 未产生 validation-best checkpoint。")

    load_prompt_z_state(prompt_z, paths["p1_best"], device)
    confirmed_metrics = evaluate_validation(
        args,
        adapter,
        prompt_z,
        val_loader,
        device,
        phase=1,
        step=step,
    )
    if not math.isclose(
        confirmed_metrics["full_on_mse"],
        best_val,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            "Phase-1 best checkpoint 复核不一致，可能存在保存或加载错误。"
        )

    print(
        f"[P1] done in {time.time()-t0:.0f}s | "
        f"best full-on val MSE={confirmed_metrics['full_on_mse']:.6f} | "
        f"improvement={confirmed_metrics['full_on_improvement_pct']:+.2f}%"
    )
    return paths["p1_best"], confirmed_metrics


# ============================================================================
# Phase 2 — learned gamma, validation-best by learned-gamma MSE
# ============================================================================

def run_phase2(
    args: argparse.Namespace,
    adapter,
    prompt_z: PromptZModulator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    paths: dict[str, str],
    p1_checkpoint: str,
) -> dict[str, float]:
    print("\n" + "=" * 76)
    print("  Phase 2 — Gamma release")
    print(f"  steps={args.phase2_steps}, learned gamma, lambda_delta={args.lambda_delta}")
    print("  train: drift_encoder + confidence_gate + low_rank_mod")
    print("  best: validation learned-gamma MSE")
    print("=" * 76)

    load_prompt_z_state(prompt_z, p1_checkpoint, device)
    params = unique_parameters([
        prompt_z.drift_encoder,
        prompt_z.confidence_gate,
        prompt_z.low_rank_mod,
    ])
    assert_sparse_mask_excluded(params, prompt_z)
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    tracker = ResidualTracker(
        num_channels=args.enc_in,
        window_K=args.residual_window_K,
    ).to(device)
    tracker.reset()
    residual_cache: deque = deque()

    best_val = float("inf")
    best_metrics: dict[str, float] | None = None
    step = 0
    epoch = 0
    t0 = time.time()

    with open(paths["p2_train_log"], "w", encoding="utf-8") as train_log, open(
        paths["p2_val_log"], "w", encoding="utf-8"
    ) as val_log:
        while step < args.phase2_steps:
            tracker.reset()
            residual_cache.clear()

            for X, Y in train_loader:
                if step >= args.phase2_steps:
                    break

                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                stats_tensor = pack_residual_stats(tracker, device)

                optimizer.zero_grad(set_to_none=True)
                out = prompt_forward(
                    X,
                    adapter,
                    prompt_z,
                    stats_tensor,
                    use_learned_gamma=True,
                )
                forecast_loss = F.mse_loss(out["pred"], Y)
                raw_ratio_live = out["raw_ratio_per_channel"].mean()
                loss = forecast_loss + args.lambda_delta * raw_ratio_live
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
                optimizer.step()

                with torch.no_grad():
                    frozen_loss = F.mse_loss(out["frozen_pred"], Y).item()
                    gamma = out["gamma"].detach()
                    raw_ratio = out["raw_ratio_per_channel"].mean().item()
                    effective_ratio = (
                        out["effective_ratio_per_channel"].mean().item()
                    )
                    improvement = (
                        (frozen_loss - forecast_loss.item())
                        / max(frozen_loss, 1e-12)
                        * 100.0
                    )
                    row = {
                        "phase": 2,
                        "step": step + 1,
                        "epoch": epoch,
                        "forecast_loss": forecast_loss.item(),
                        "frozen_loss": frozen_loss,
                        "total_loss": loss.item(),
                        "step_improvement_pct": improvement,
                        "gamma_mean": gamma.mean().item(),
                        "gamma_std": gamma.std(unbiased=False).item(),
                        "gamma_min": gamma.min().item(),
                        "gamma_max": gamma.max().item(),
                        "channel_fraction_gamma_lt01": (
                            (gamma < 0.1).float().mean().item()
                        ),
                        "channel_fraction_gamma_gt09": (
                            (gamma > 0.9).float().mean().item()
                        ),
                        "raw_d2h": raw_ratio,
                        "effective_d2h": effective_ratio,
                    }
                    write_jsonl(train_log, row)

                advance_tracker(
                    tracker,
                    residual_cache,
                    out["frozen_pred"],
                    Y,
                    args.forecast_H,
                )

                step += 1
                if step == 1 or step % args.log_interval == 0:
                    print(
                        f"[P2|{step:5d}] loss={forecast_loss.item():.6f} "
                        f"frozen={frozen_loss:.6f} step_impr={improvement:+.2f}% "
                        f"gamma={gamma.mean().item():.4f} "
                        f"[{gamma.min().item():.3f},{gamma.max().item():.3f}] "
                        f"raw_d/h={raw_ratio:.5f} eff_d/h={effective_ratio:.5f}"
                    )

                should_validate = (
                    step % args.phase2_val_interval == 0
                    or step == args.phase2_steps
                )
                if should_validate:
                    metrics = evaluate_validation(
                        args,
                        adapter,
                        prompt_z,
                        val_loader,
                        device,
                        phase=2,
                        step=step,
                    )
                    write_jsonl(val_log, metrics)
                    selection_mse = metrics["learned_gamma_mse"]
                    if selection_mse < best_val:
                        best_val = selection_mse
                        best_metrics = metrics
                        torch.save(prompt_z.state_dict(), paths["p2_best"])
                        print(
                            f"[*] Phase-2 new validation best: {best_val:.6f} "
                            f"-> {paths['p2_best']}"
                        )

            epoch += 1

    torch.save(prompt_z.state_dict(), paths["p2_final"])
    if best_metrics is None or not os.path.isfile(paths["p2_best"]):
        raise RuntimeError("Phase 2 未产生 validation-best checkpoint。")

    load_prompt_z_state(prompt_z, paths["p2_best"], device)
    confirmed_metrics = evaluate_validation(
        args,
        adapter,
        prompt_z,
        val_loader,
        device,
        phase=2,
        step=step,
    )
    if not math.isclose(
        confirmed_metrics["learned_gamma_mse"],
        best_val,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            "Phase-2 best checkpoint 复核不一致，可能存在保存或加载错误。"
        )

    print(
        f"[P2] done in {time.time()-t0:.0f}s | "
        f"best learned-gamma val MSE={confirmed_metrics['learned_gamma_mse']:.6f} | "
        f"improvement={confirmed_metrics['learned_improvement_pct']:+.2f}%"
    )
    print_phase2_diagnosis(confirmed_metrics)
    return confirmed_metrics


# ============================================================================
# Main training flow
# ============================================================================

def build_paths(args: argparse.Namespace) -> dict[str, str]:
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    dataset_name = args.data_path.replace(".csv", "")
    tag = f"gamma_final_{dataset_name}_H{args.forecast_H}"
    if args.experiment_tag:
        tag += f"_{args.experiment_tag}"

    return {
        "tag": tag,
        "p1_best": os.path.join(args.save_dir, f"{tag}_p1_best.pth"),
        "p1_final": os.path.join(args.save_dir, f"{tag}_p1_final.pth"),
        "p2_best": os.path.join(args.save_dir, f"{tag}_p2_best.pth"),
        "p2_final": os.path.join(args.save_dir, f"{tag}_p2_final.pth"),
        "p1_train_log": os.path.join(args.log_dir, f"{tag}_p1_train.jsonl"),
        "p1_val_log": os.path.join(args.log_dir, f"{tag}_p1_val.jsonl"),
        "p2_train_log": os.path.join(args.log_dir, f"{tag}_p2_train.jsonl"),
        "p2_val_log": os.path.join(args.log_dir, f"{tag}_p2_val.jsonl"),
    }


def train(args: argparse.Namespace, device: torch.device) -> None:
    adapter = build_backbone(args, device)
    prompt_z = PromptZModulator(
        d_model=args.D_model,
        hidden_layout=adapter.hidden_layout,
        d_drift=args.d_drift,
        rank=args.rank,
        gamma_init_bias=args.gamma_init_bias,
        mask_init_bias=-1.5,
        max_delta_ratio=args.max_delta_ratio,
    ).to(device)

    train_loader, val_loader = build_dataloaders(args)
    paths = build_paths(args)

    print(f"[*] Tag: {paths['tag']}")
    print("[*] Effective mask=1; SparseMaskHead excluded from all optimizers")
    print("[*] lambda_mask=0; lambda_noop=0")
    print(f"[*] Phase-1 best: {paths['p1_best']}")
    print(f"[*] Phase-2 best: {paths['p2_best']}")

    if args.skip_phase1:
        if not args.phase1_ckpt:
            raise ValueError("--skip_phase1 必须同时提供 --phase1_ckpt")
        load_prompt_z_state(prompt_z, args.phase1_ckpt, device)
        p1_checkpoint = args.phase1_ckpt
        p1_metrics = evaluate_validation(
            args,
            adapter,
            prompt_z,
            val_loader,
            device,
            phase=1,
            step=0,
        )
        print(
            f"[*] Loaded Phase-1 checkpoint full-on improvement: "
            f"{p1_metrics['full_on_improvement_pct']:+.2f}%"
        )
    else:
        p1_checkpoint, p1_metrics = run_phase1(
            args,
            adapter,
            prompt_z,
            train_loader,
            val_loader,
            device,
            paths,
        )

    p1_improvement = p1_metrics["full_on_improvement_pct"]
    print("\n" + "=" * 76)
    print(
        f"[DECISION] Phase-1 full-on validation improvement="
        f"{p1_improvement:+.2f}%"
    )
    if p1_improvement > args.phase1_pass_threshold:
        proceed = True
        print(
            f"[DECISION] PASS: > {args.phase1_pass_threshold:+.2f}%; "
            "delta 在 validation 上有正向修正能力。"
        )
    else:
        proceed = False
        print(
            f"[DECISION] STOP: <= {args.phase1_pass_threshold:+.2f}%; "
            "没有证据证明 delta 在 validation 上有效。"
        )

    if args.force_phase2:
        proceed = True
        print("[DECISION] --force_phase2: 强制进入 Phase 2。")
    print("=" * 76)

    if not proceed:
        print("[*] Phase 2 skipped. 不应把该结果解释为 gamma 失败；当前只说明 delta warmup 未通过。")
        return

    final_metrics = run_phase2(
        args,
        adapter,
        prompt_z,
        train_loader,
        val_loader,
        device,
        paths,
        p1_checkpoint,
    )

    print("\n" + "=" * 76)
    print("  Final validation conclusion")
    print("=" * 76)
    print(f"  Frozen MSE             : {final_metrics['frozen_mse']:.6f}")
    print(f"  Learned-gamma MSE      : {final_metrics['learned_gamma_mse']:.6f}")
    print(f"  Fixed gamma=1 MSE      : {final_metrics['full_on_mse']:.6f}")
    print(
        f"  Learned improvement    : "
        f"{final_metrics['learned_improvement_pct']:+.2f}%"
    )
    print(
        f"  Full-on improvement    : "
        f"{final_metrics['full_on_improvement_pct']:+.2f}%"
    )
    print("  注意：以上仅是 validation 结论；最终论文数值需使用未调参 test。")
    print("=" * 76)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-Phase Delta Warmup + Gamma Release (validated)"
    )

    # Data
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--forecast_H", type=int, required=True)
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--val_max_windows",
        type=int,
        default=0,
        help="0=完整 validation；正数仅建议 smoke test",
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        type=str,
        default="patchtst",
        choices=["patchtst", "itransformer"],
    )
    parser.add_argument("--D_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        required=True,
        help="与 Prompt-Z 实验完全一致的 frozen backbone 权重",
    )

    # Prompt-Z
    parser.add_argument("--d_drift", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gamma_init_bias", type=float, default=0.0)
    parser.add_argument("--max_delta_ratio", type=float, default=0.02)
    parser.add_argument("--residual_window_K", type=int, default=24)

    # Two phases
    parser.add_argument("--phase1_steps", type=int, default=1000)
    parser.add_argument("--phase2_steps", type=int, default=2000)
    parser.add_argument("--phase1_val_interval", type=int, default=200)
    parser.add_argument("--phase2_val_interval", type=int, default=200)
    parser.add_argument(
        "--phase1_pass_threshold",
        type=float,
        default=0.0,
        help="Phase-1 full-on validation improvement 必须高于该百分比",
    )
    parser.add_argument("--force_phase2", action="store_true")
    parser.add_argument("--skip_phase1", action="store_true")
    parser.add_argument("--phase1_ckpt", type=str, default=None)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lambda_delta",
        type=float,
        default=2e-4,
        help="Phase-2 raw delta ratio coefficient; Phase 1 固定为 0",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Misc
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="weights/prompt_z")
    parser.add_argument("--log_dir", type=str, default="logs/prompt_z")
    parser.add_argument("--experiment_tag", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    if args.phase1_steps <= 0 and not args.skip_phase1:
        raise ValueError("phase1_steps 必须 > 0，或使用 --skip_phase1")
    if args.phase2_steps <= 0:
        raise ValueError("phase2_steps 必须 > 0")
    if args.phase1_val_interval <= 0 or args.phase2_val_interval <= 0:
        raise ValueError("validation interval 必须 > 0")
    if not (0.0 < args.max_delta_ratio <= 1.0):
        raise ValueError("max_delta_ratio 必须位于 (0, 1]")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Seed: {args.seed}")

    if args.enc_in is None:
        import pandas as pd

        csv_path = os.path.join(args.root_path, args.data_path)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"数据文件不存在: {csv_path}")
        dataframe = pd.read_csv(csv_path)
        args.enc_in = len(
            [column for column in dataframe.columns if column.lower() != "date"]
        )
        print(f"[*] Auto enc_in={args.enc_in}")

    print("[*] mask=1; lambda_mask=0; lambda_noop=0")
    print(
        f"[*] Phase 1: gamma=1, steps={args.phase1_steps}, "
        f"val_interval={args.phase1_val_interval}"
    )
    print(
        f"[*] Phase 2: learned gamma, steps={args.phase2_steps}, "
        f"val_interval={args.phase2_val_interval}, "
        f"lambda_delta={args.lambda_delta}"
    )

    train(args, device)


if __name__ == "__main__":
    main()
