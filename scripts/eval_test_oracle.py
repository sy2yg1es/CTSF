"""
eval_test_oracle.py — Test Set Evaluation + Oracle Gate Upper Bound
====================================================================

在 test split 上评估基线，并计算两种 oracle gate 的理论上界：

  Frozen             : backbone 直接预测，无任何修正
  Fixed gamma=1      : delta 全开（load P1 or P2 checkpoint）
  Learned gamma      : confidence_gate 输出（需要 P2 checkpoint）
  Window oracle      : 每个窗口共享一个事后最优 gamma ∈ {0,1}
  Channel oracle     : 每个 (window, channel) 使用事后最优 gamma ∈ {0,1}

后者与 Prompt-Z confidence_gate 的输出粒度 [B,C,1] 一致。脚本同时记录
advantage = MSE(frozen) - MSE(delta)；oracle 上界始终使用 advantage > 0，
可训练诊断标签可通过 --oracle_target_margin_pct 忽略接近打平的噪声样本。

重要：这里的标签来自 test 真值，只能用于 oracle/可学习性诊断，绝不能用于训练
或选择 gate。训练标签必须在 train split 生成，阈值只能在 validation 上确定。

Oracle 上界的意义：
  - 若 oracle 比 fixed_gamma=1 提升 < 0.1%  → gamma 本来就没有多少选择空间，直接去掉
  - 若 oracle 比 fixed_gamma=1 提升 1%～3%   → 动态选择理论上有价值，当前 gate 学不出来

Tracker warmup 策略：
  从 warmup_start（默认 val_start）开始顺序跑到 test_start，
  建立接近 test 起点的 drift 状态，再在 test 上评估。

使用示例：
    # 评估 P1 checkpoint（fixed gamma=1）
    python scripts/eval_test_oracle.py \\
        --root_path ./dataset \\
        --data_path ETTm1.csv \\
        --forecast_H 1 \\
        --pretrained_weights weights/patchtst_pretrained_ETTm1_H1.pth \\
        --p1_ckpt weights/prompt_z/gfv2_ETTm1_H1_v2_exc1_p1.pth \\
        --experiment_tag v2_exc1

    # 同时评估 P2 checkpoint（learned + fixed gamma）
    python scripts/eval_test_oracle.py \\
        --root_path ./dataset \\
        --data_path ETTm1.csv \\
        --forecast_H 1 \\
        --pretrained_weights weights/patchtst_pretrained_ETTm1_H1.pth \\
        --p1_ckpt weights/prompt_z/gfv2_ETTm1_H1_v2_exc1_p1.pth \\
        --p2_ckpt weights/prompt_z/gfv2_ETTm1_H1_v2_exc1_p2_best.pth \\
        --experiment_tag v2_exc1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from data_provider.data_loader import data_provider
from models.backbone_adapter import PatchTSTAdapter, iTransformerAdapter
from models.prompt_z import PromptZModulator
from core.residual_tracker import ResidualTracker


# ============================================================================
# Backbone
# ============================================================================

def build_backbone(args, device):
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

    if args.pretrained_weights:
        state = torch.load(args.pretrained_weights, map_location=device)
        if "backbone_adapter.backbone" in str(list(state.keys())[:3]):
            prefix = "backbone_adapter.backbone."
            backbone_state = {
                k[len(prefix):]: v for k, v in state.items()
                if k.startswith(prefix)
            }
        else:
            backbone_state = state

        missing, unexpected = adapter.backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            raise RuntimeError(
                f"Backbone missing keys: {missing[:5]} ({len(missing)} total). "
                "Check --pretrained_weights and backbone config."
            )
        if unexpected:
            print(f"[!] Backbone: {len(unexpected)} unexpected keys (benign): {unexpected[:3]}")
        print(f"[*] Backbone loaded: {args.pretrained_weights}")

    for p in adapter.parameters():
        p.requires_grad = False
    adapter.eval()
    return adapter


def build_prompt_z(args, device):
    prompt_z = PromptZModulator(
        d_model=args.D_model,
        hidden_layout=args.hidden_layout,
        d_drift=args.d_drift,
        rank=args.rank,
        gamma_init_bias=0.0,
        mask_init_bias=-1.5,
        max_delta_ratio=args.max_delta_ratio,
    ).to(device)
    return prompt_z


# ============================================================================
# Dataset
# ============================================================================

def get_splits(args):
    class DPArgs:
        pass

    dp = DPArgs()
    dp.root_path   = args.root_path
    dp.data_path   = args.data_path
    dp.features    = args.features
    dp.seq_len     = args.seq_len
    dp.pred_len    = args.forecast_H
    dp.target      = "OT"
    dp.num_workers = args.num_workers
    dp.train_ratio = args.train_ratio
    dp.val_ratio   = args.val_ratio

    dataset, _ = data_provider(dp)
    print(f"[*] Dataset splits:")
    print(f"    train: [0, {dataset.train_size})")
    print(f"    val  : [{dataset.val_start}, {dataset.test_start})")
    print(f"    test : [{dataset.test_start}, {dataset.n_windows})")

    # Warmup windows: val split（在 test 之前）
    warmup_start = getattr(args, "warmup_start", dataset.val_start)
    warmup_end   = dataset.test_start
    warmup_subset = Subset(dataset, range(warmup_start, warmup_end))

    test_subset = Subset(dataset, range(dataset.test_start, dataset.n_windows))

    warmup_loader = DataLoader(warmup_subset, batch_size=1, shuffle=False,
                               num_workers=args.num_workers, drop_last=False)
    test_loader   = DataLoader(test_subset,   batch_size=1, shuffle=False,
                               num_workers=args.num_workers, drop_last=False)

    print(f"    warmup: [{warmup_start}, {warmup_end})  "
          f"({warmup_end - warmup_start} windows)")
    return warmup_loader, test_loader, dataset


def pack_stats(tracker, device):
    s = tracker.get_stats()
    return torch.stack([
        s["error_mean"].to(device),
        s["error_slope"].to(device),
        s["error_std"].to(device),
        s["signed_bias"].to(device),
        s["steps_gap"].to(device).expand(tracker.C),
    ], dim=-1)


# ============================================================================
# Per-window forward：返回 mse_frozen, mse_fixed1, mse_learned
# ============================================================================

def _per_channel_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Return MSE per (sample, channel), preserving first/last dimensions."""
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")
    if pred.dim() < 2:
        raise ValueError(f"Expected predictions with a channel dimension, got {pred.shape}")
    reduce_dims = tuple(range(1, pred.dim() - 1))
    squared_error = (pred - target).pow(2)
    return squared_error.mean(dim=reduce_dims) if reduce_dims else squared_error


def _margin_target(relative_advantage_pct: Tensor, margin_pct: float):
    """Build a binary target plus a validity mask for near-tie filtering.

    target=1 means the full-on delta is better. Samples with
    |relative_advantage_pct| < margin_pct are marked invalid so a future gate
    trainer can drop or down-weight labels dominated by numerical/noise ties.
    The exact oracle itself never uses this margin.
    """
    if margin_pct < 0:
        raise ValueError(f"oracle_target_margin_pct must be >= 0, got {margin_pct}")
    target = (relative_advantage_pct > margin_pct).to(torch.int64)
    valid = relative_advantage_pct.abs() >= margin_pct
    return target, valid


def compute_oracle_supervision(
    Y_frozen: Tensor,
    Y_fixed1: Tensor,
    Y_learned: Tensor,
    Y_true: Tensor,
    *,
    target_margin_pct: float = 0.0,
):
    """Compute exact oracle bounds and gate-aligned supervision diagnostics.

    Window targets have one decision per sample window. Channel targets have
    one decision per output channel and therefore align with gamma [B,C,1].
    """
    mse_frozen_c = _per_channel_mse(Y_frozen, Y_true)
    mse_fixed1_c = _per_channel_mse(Y_fixed1, Y_true)
    mse_learned_c = _per_channel_mse(Y_learned, Y_true)

    advantage_c = mse_frozen_c - mse_fixed1_c
    relative_advantage_c_pct = (
        advantage_c / mse_frozen_c.clamp(min=1e-12) * 100.0
    )

    # Exact channel oracle: independent binary decision for each sample/channel.
    oracle_gamma_c = (advantage_c > 0).to(torch.int64)
    channel_view = [Y_true.shape[0]] + [1] * (Y_true.dim() - 2) + [Y_true.shape[-1]]
    oracle_gate_c = oracle_gamma_c.to(torch.bool).view(*channel_view)
    Y_oracle_channel = torch.where(oracle_gate_c, Y_fixed1, Y_frozen)
    mse_oracle_channel = F.mse_loss(Y_oracle_channel, Y_true)

    # Continuous channel oracle in [0,1]. Because both current adapters decode
    # channel-separably after the Prompt-Z hook, output interpolation exactly
    # matches applying a continuous per-channel gamma in hidden space.
    residual = Y_true - Y_frozen
    direction = Y_fixed1 - Y_frozen
    reduce_dims = tuple(range(1, Y_true.dim() - 1))
    numerator = (residual * direction).sum(dim=reduce_dims)
    denominator = direction.pow(2).sum(dim=reduce_dims).clamp(min=1e-12)
    oracle_gamma_continuous_c = (numerator / denominator).clamp(0.0, 1.0)
    continuous_gate_c = oracle_gamma_continuous_c.view(*channel_view)
    Y_oracle_continuous_channel = Y_frozen + continuous_gate_c * direction
    mse_oracle_continuous_channel = F.mse_loss(
        Y_oracle_continuous_channel, Y_true
    )

    mse_frozen = F.mse_loss(Y_frozen, Y_true)
    mse_fixed1 = F.mse_loss(Y_fixed1, Y_true)
    advantage_window = mse_frozen - mse_fixed1
    relative_advantage_window_pct = (
        advantage_window / mse_frozen.clamp(min=1e-12) * 100.0
    )

    # Exact window oracle: one shared binary decision for all channels.
    oracle_gamma_window = (advantage_window > 0).to(torch.int64)
    mse_oracle_window = torch.minimum(mse_frozen, mse_fixed1)

    target_gamma_c, target_valid_c = _margin_target(
        relative_advantage_c_pct, target_margin_pct
    )
    target_gamma_window, target_valid_window = _margin_target(
        relative_advantage_window_pct, target_margin_pct
    )

    return {
        "mse_frozen_channel": mse_frozen_c,
        "mse_fixed1_channel": mse_fixed1_c,
        "mse_learned_channel": mse_learned_c,
        "advantage_channel": advantage_c,
        "relative_advantage_channel_pct": relative_advantage_c_pct,
        "oracle_gamma_channel": oracle_gamma_c,
        "target_gamma_channel": target_gamma_c,
        "target_valid_channel": target_valid_c,
        "mse_oracle_channel": mse_oracle_channel,
        "oracle_gamma_continuous_channel": oracle_gamma_continuous_c,
        "mse_oracle_continuous_channel": mse_oracle_continuous_channel,
        "advantage_window": advantage_window,
        "relative_advantage_window_pct": relative_advantage_window_pct,
        "oracle_gamma_window": oracle_gamma_window,
        "target_gamma_window": target_gamma_window,
        "target_valid_window": target_valid_window,
        "mse_oracle_window": mse_oracle_window,
    }


@torch.no_grad()
def forward_all_modes(X, Y, adapter, prompt_z, stats_tensor, layout,
                      target_margin_pct=0.0):
    """
    一次 backbone encode，三种 gamma 模式下的 MSE。
    同时返回逐通道 oracle 信息（用于后续分析）。

    Returns:
        mse_frozen   : scalar
        mse_fixed1   : scalar  (gamma=1)
        mse_learned  : scalar  (gamma=confidence_gate output)
        Y_frozen     : tensor
        Y_fixed1     : tensor
        Y_learned    : tensor
    """
    with torch.no_grad():
        hidden, means, stdev = adapter.encode_until_hook(X)
    hidden = hidden.detach()

    Y_frozen = adapter.decode_from_hook(hidden, means, stdev)

    # drift state
    summary     = prompt_z._hidden_summary(hidden)
    drift_state = prompt_z.drift_encoder(summary, stats_tensor)

    # gamma values
    gamma_learned = prompt_z.confidence_gate(drift_state)  # [B,C,1]
    gamma_ones    = torch.ones_like(gamma_learned)

    # mask always ones
    prompt_z.sparse_mask(drift_state)  # run but discard

    # delta
    if layout == "BCDP":
        h_w  = hidden.permute(0, 1, 3, 2)
        dh   = prompt_z.low_rank_mod(h_w, drift_state)
        dh   = dh.permute(0, 1, 3, 2)
    else:
        dh = prompt_z.low_rank_mod(hidden, drift_state)
    dh = prompt_z._ratio_clamp(dh, hidden)

    if layout == "BCDP":
        applied_fixed1  = gamma_ones.unsqueeze(-1) * dh
        applied_learned = gamma_learned.unsqueeze(-1) * dh
    else:
        applied_fixed1  = gamma_ones   * dh
        applied_learned = gamma_learned * dh

    Y_fixed1  = adapter.decode_from_hook(hidden + applied_fixed1,  means, stdev)
    Y_learned = adapter.decode_from_hook(hidden + applied_learned, means, stdev)

    supervision = compute_oracle_supervision(
        Y_frozen, Y_fixed1, Y_learned, Y,
        target_margin_pct=target_margin_pct,
    )
    mse_frozen  = F.mse_loss(Y_frozen,  Y).item()
    mse_fixed1  = F.mse_loss(Y_fixed1,  Y).item()
    mse_learned = F.mse_loss(Y_learned, Y).item()

    # unclamped ratio for diagnostics
    if layout == "BCDP":
        df_raw = dh.flatten(2)
    else:
        df_raw = dh.flatten(2)
    h_flat   = hidden.detach().flatten(2)
    h_norm   = h_flat.norm(dim=-1).clamp(min=1e-8)
    d2h_clmp = (df_raw.norm(dim=-1) / h_norm).mean().item()
    g_mean   = gamma_learned.mean().item()
    g_channel = gamma_learned.mean(dim=0).squeeze(-1)

    return dict(
        mse_frozen=mse_frozen,
        mse_fixed1=mse_fixed1,
        mse_learned=mse_learned,
        Y_frozen=Y_frozen,
        Y_fixed1=Y_fixed1,
        Y_learned=Y_learned,
        gamma_mean=g_mean,
        gamma_channel=g_channel.detach().cpu().tolist(),
        clamped_d2h=d2h_clmp,
        oracle_supervision=supervision,
    )


# ============================================================================
# Main evaluation
# ============================================================================

@torch.no_grad()
def evaluate(adapter, prompt_z, warmup_loader, test_loader, args, device):
    """
    Tracker warmup → test evaluation。
    Returns per-window lists and aggregated stats.
    """
    layout = prompt_z.hidden_layout

    # ── Tracker warmup（val split）──────────────────────────────────────
    tracker = ResidualTracker(args.enc_in, args.residual_window_K).to(device)
    tracker.reset()
    rc = deque()

    prompt_z.eval()
    print(f"[*] Warming tracker on {len(warmup_loader)} val windows ...")
    for X, Y in warmup_loader:
        X = X.to(device); Y = Y.to(device)
        st = pack_stats(tracker, device)

        with torch.no_grad():
            hidden, means, stdev = adapter.encode_until_hook(X)
        Y_frozen = adapter.decode_from_hook(hidden.detach(), means, stdev)

        rc.append((Y_frozen.detach(), Y.detach()))
        if len(rc) > args.forecast_H:
            op, ot = rc.popleft(); tracker.update(op, ot)
        else:
            tracker.step_no_update()
    print(f"[*] Tracker warmed: count={int(tracker._count.item())}  pending_rc={len(rc)}")

    # ── Test evaluation ──────────────────────────────────────────────────
    results_per_window = []
    n = 0

    for X, Y in test_loader:
        X = X.to(device); Y = Y.to(device)
        st = pack_stats(tracker, device)

        out = forward_all_modes(
            X, Y, adapter, prompt_z, st, layout,
            target_margin_pct=args.oracle_target_margin_pct,
        )
        sup = out["oracle_supervision"]

        results_per_window.append({
            "mse_frozen":    out["mse_frozen"],
            "mse_fixed1":    out["mse_fixed1"],
            "mse_learned":   out["mse_learned"],
            "mse_oracle_window":  sup["mse_oracle_window"].item(),
            "mse_oracle_channel": sup["mse_oracle_channel"].item(),
            "mse_oracle_continuous_channel":
                sup["mse_oracle_continuous_channel"].item(),
            "advantage_window": sup["advantage_window"].item(),
            "relative_advantage_window_pct":
                sup["relative_advantage_window_pct"].item(),
            "oracle_gamma_window": sup["oracle_gamma_window"].item(),
            "target_gamma_window": sup["target_gamma_window"].item(),
            "target_valid_window": bool(sup["target_valid_window"].item()),
            "mse_frozen_channel": sup["mse_frozen_channel"].cpu().flatten().tolist(),
            "mse_fixed1_channel": sup["mse_fixed1_channel"].cpu().flatten().tolist(),
            "mse_learned_channel": sup["mse_learned_channel"].cpu().flatten().tolist(),
            "advantage_channel": sup["advantage_channel"].cpu().flatten().tolist(),
            "relative_advantage_channel_pct":
                sup["relative_advantage_channel_pct"].cpu().flatten().tolist(),
            "oracle_gamma_channel":
                sup["oracle_gamma_channel"].cpu().flatten().tolist(),
            "oracle_gamma_continuous_channel":
                sup["oracle_gamma_continuous_channel"].cpu().flatten().tolist(),
            "target_gamma_channel":
                sup["target_gamma_channel"].cpu().flatten().tolist(),
            "target_valid_channel":
                sup["target_valid_channel"].cpu().flatten().tolist(),
            "gamma_mean":    out["gamma_mean"],
            "gamma_channel": out["gamma_channel"],
            "clamped_d2h":   out["clamped_d2h"],
        })

        # Tracker update
        rc.append((out["Y_frozen"].detach(), Y.detach()))
        if len(rc) > args.forecast_H:
            op, ot = rc.popleft(); tracker.update(op, ot)
        else:
            tracker.step_no_update()

        n += 1

    N = max(n, 1)
    avg = lambda key: sum(r[key] for r in results_per_window) / N

    mse_frozen  = avg("mse_frozen")
    mse_fixed1  = avg("mse_fixed1")
    mse_learned = avg("mse_learned")
    mse_oracle_window = avg("mse_oracle_window")
    mse_oracle_channel = avg("mse_oracle_channel")
    mse_oracle_continuous_channel = avg("mse_oracle_continuous_channel")
    oracle_window_frac = avg("oracle_gamma_window")
    channel_labels = [
        value
        for row in results_per_window
        for value in row["oracle_gamma_channel"]
    ]
    target_channel_valid = [
        value
        for row in results_per_window
        for value in row["target_valid_channel"]
    ]
    oracle_channel_frac = sum(channel_labels) / max(len(channel_labels), 1)
    target_channel_valid_frac = (
        sum(target_channel_valid) / max(len(target_channel_valid), 1)
    )
    target_window_valid_frac = (
        sum(r["target_valid_window"] for r in results_per_window) / N
    )
    gamma_mean  = avg("gamma_mean")
    d2h_mean    = avg("clamped_d2h")

    if mse_oracle_channel > mse_oracle_window + 1e-7:
        raise RuntimeError(
            "Channel oracle must not be worse than window oracle: "
            f"{mse_oracle_channel:.8f} > {mse_oracle_window:.8f}"
        )
    if mse_oracle_continuous_channel > mse_oracle_channel + 1e-7:
        raise RuntimeError(
            "Continuous channel oracle must not be worse than binary channel oracle: "
            f"{mse_oracle_continuous_channel:.8f} > {mse_oracle_channel:.8f}"
        )

    def impr(mse_ref, mse_new):
        return (mse_ref - mse_new) / max(mse_ref, 1e-12) * 100

    continuous_gain = mse_frozen - mse_oracle_continuous_channel
    binary_gain = mse_frozen - mse_oracle_channel
    continuous_routing_gain = mse_fixed1 - mse_oracle_continuous_channel
    binary_routing_gain = mse_fixed1 - mse_oracle_channel

    return {
        "n_test_windows":       n,
        "mse_frozen":           mse_frozen,
        "mse_fixed_gamma1":     mse_fixed1,
        "mse_learned_gamma":    mse_learned,
        "mse_oracle_window":    mse_oracle_window,
        "mse_oracle_channel":   mse_oracle_channel,
        "mse_oracle_continuous_channel": mse_oracle_continuous_channel,
        # Backward-compatible alias: the old oracle was window-level.
        "mse_oracle":           mse_oracle_window,
        "impr_fixed_vs_frozen":   impr(mse_frozen, mse_fixed1),
        "impr_learned_vs_frozen": impr(mse_frozen, mse_learned),
        "impr_oracle_window_vs_frozen":
            impr(mse_frozen, mse_oracle_window),
        "impr_oracle_window_vs_fixed1":
            impr(mse_fixed1, mse_oracle_window),
        "impr_oracle_channel_vs_frozen":
            impr(mse_frozen, mse_oracle_channel),
        "impr_oracle_channel_vs_fixed1":
            impr(mse_fixed1, mse_oracle_channel),
        "impr_oracle_continuous_channel_vs_frozen":
            impr(mse_frozen, mse_oracle_continuous_channel),
        "impr_oracle_continuous_channel_vs_fixed1":
            impr(mse_fixed1, mse_oracle_continuous_channel),
        # Backward-compatible aliases for existing result parsers.
        "impr_oracle_vs_frozen": impr(mse_frozen, mse_oracle_window),
        "impr_oracle_vs_fixed1": impr(mse_fixed1, mse_oracle_window),
        "gate_delta_mse":          mse_learned - mse_fixed1,  # <0 => learned better
        "oracle_window_helpful_frac": oracle_window_frac,
        "oracle_channel_helpful_frac": oracle_channel_frac,
        "oracle_helpful_frac": oracle_window_frac,
        "oracle_target_margin_pct": args.oracle_target_margin_pct,
        "target_window_valid_frac": target_window_valid_frac,
        "target_channel_valid_frac": target_channel_valid_frac,
        "binary_capture_of_continuous_gain":
            binary_gain / max(continuous_gain, 1e-12),
        "binary_capture_of_continuous_routing_gain":
            binary_routing_gain / max(continuous_routing_gain, 1e-12),
        "gamma_mean_test":        gamma_mean,
        "clamped_d2h_mean":       d2h_mean,
        "per_window":             results_per_window,
    }


# ============================================================================
# Print & save
# ============================================================================

def print_results(r, label="Test"):
    N   = r["n_test_windows"]
    d   = r["gate_delta_mse"]
    gate_sign = ("✓ learned<fixed" if d < -1e-7 else
                 "✗ learned>fixed" if d > 1e-7 else "≈ equal")

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {label} Evaluation  (N={N} windows)")
    print(sep)
    print(f"  {'Metric':<28s}  {'MSE':>10s}  {'vs Frozen':>10s}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}")
    print(f"  {'Frozen':<28s}  {r['mse_frozen']:>10.6f}  {'—':>10s}")
    print(f"  {'Fixed gamma=1':<28s}  {r['mse_fixed_gamma1']:>10.6f}  "
          f"{r['impr_fixed_vs_frozen']:>+9.2f}%")
    print(f"  {'Learned gamma':<28s}  {r['mse_learned_gamma']:>10.6f}  "
          f"{r['impr_learned_vs_frozen']:>+9.2f}%")
    print(f"  {'Window oracle {0,1}':<28s}  {r['mse_oracle_window']:>10.6f}  "
          f"{r['impr_oracle_window_vs_frozen']:>+9.2f}%")
    print(f"  {'Channel oracle {0,1}':<28s}  {r['mse_oracle_channel']:>10.6f}  "
          f"{r['impr_oracle_channel_vs_frozen']:>+9.2f}%")
    print(f"  {'Channel oracle [0,1]':<28s}  "
          f"{r['mse_oracle_continuous_channel']:>10.6f}  "
          f"{r['impr_oracle_continuous_channel_vs_frozen']:>+9.2f}%")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}")
    print(f"  Gate Δ (learned - fixed): {d:+.7f}  [{gate_sign}]")
    print(f"  Window oracle vs fixed:   {r['impr_oracle_window_vs_fixed1']:+.2f}%")
    print(f"  Channel oracle vs fixed:  {r['impr_oracle_channel_vs_fixed1']:+.2f}%")
    print(f"  Continuous vs fixed:      "
          f"{r['impr_oracle_continuous_channel_vs_fixed1']:+.2f}%")
    print(f"  Binary capture (total):   "
          f"{r['binary_capture_of_continuous_gain']*100:.1f}%")
    print(f"  Binary capture (routing): "
          f"{r['binary_capture_of_continuous_routing_gain']*100:.1f}%")
    print()
    print(f"  window helpful frac : {r['oracle_window_helpful_frac']:.3f}  "
          f"({r['oracle_window_helpful_frac']*100:.1f}% windows)")
    print(f"  channel helpful frac: {r['oracle_channel_helpful_frac']:.3f}  "
          f"({r['oracle_channel_helpful_frac']*100:.1f}% window-channel pairs)")
    print(f"  target valid frac   : window={r['target_window_valid_frac']:.3f}, "
          f"channel={r['target_channel_valid_frac']:.3f}  "
          f"(margin={r['oracle_target_margin_pct']:.3f}%)")
    print(f"  gamma_mean (test)   : {r['gamma_mean_test']:.4f}")
    print(f"  clamped_d/h (test)  : {r['clamped_d2h_mean']:.5f}")
    print()

    # Interpretation
    print("  ── 自动解读 ─────────────────────────────────────────────────────")
    oracle_gap = r["impr_oracle_channel_vs_fixed1"]
    fixed_impr = r["impr_fixed_vs_frozen"]
    learned_impr = r["impr_learned_vs_frozen"]

    if fixed_impr > 0.2:
        print(f"  ✓ Fixed gamma=1 有效: {fixed_impr:+.2f}% vs frozen（test 上可信）")
    elif fixed_impr > -0.1:
        print(f"  △ Fixed gamma=1 微弱: {fixed_impr:+.2f}%，接近 frozen（不显著）")
    else:
        print(f"  ✗ Fixed gamma=1 退化: {fixed_impr:+.2f}%（delta 方向在 test 上有害）")

    if d < -1e-5:
        print(f"  ✓ Learned gamma 优于 fixed: {d:+.7f}（gate 有选择价值）")
    elif d < 1e-5:
        print(f"  ≈ Learned gamma ≈ fixed（gate 无额外价值，差异 < 1e-5）")
    else:
        print(f"  ✗ Learned gamma 差于 fixed: {d:+.7f}（gate 减弱了有效 delta）")

    if oracle_gap < 0.1:
        print(f"  → Channel oracle 上界 {oracle_gap:+.2f}% vs fixed_gamma=1：选择空间极小，"
              f"gamma gate 可直接移除")
    elif oracle_gap < 1.0:
        print(f"  → Channel oracle 上界 {oracle_gap:+.2f}% vs fixed_gamma=1：存在选择空间，"
              f"但收益较小")
    else:
        print(f"  → Channel oracle 上界 {oracle_gap:+.2f}% vs fixed_gamma=1：动态选择理论上有价值，"
              f"当前 gate 学不出来")
    print("=" * 70)


def save_results(r, out_path):
    """保存完整结果（不含 per_window 大数组）到 JSON。"""
    summary = {k: v for k, v in r.items() if k != "per_window"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[*] Saved: {out_path}")


def save_oracle_targets(r, out_path):
    """Save label diagnostics as JSONL; never use test targets for training."""
    with open(out_path, "w", encoding="utf-8") as f:
        for window_index, row in enumerate(r["per_window"]):
            payload = {"window_index": window_index, **row}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"[!] Saved TEST-only oracle diagnostics: {out_path}")
    print("[!] Do not train or select a gate with this file; generate training labels on train split.")


# ============================================================================
# Entry
# ============================================================================

def main():
    parser = argparse.ArgumentParser("Test Set + Oracle Gate Evaluation")

    # 数据
    parser.add_argument("--root_path",   default="./data")
    parser.add_argument("--data_path",   required=True)
    parser.add_argument("--features",    default="M")
    parser.add_argument("--seq_len",     type=int, default=96)
    parser.add_argument("--forecast_H", type=int, required=True)
    parser.add_argument("--enc_in",      type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio",   type=float, default=0.1)

    # Backbone
    parser.add_argument("--backbone",           default="patchtst",
                        choices=["patchtst", "itransformer"])
    parser.add_argument("--D_model",            type=int, default=512)
    parser.add_argument("--d_ff",               type=int, default=512)
    parser.add_argument("--e_layers",           type=int, default=3)
    parser.add_argument("--pretrained_weights", default=None)

    # PromptZ
    parser.add_argument("--d_drift",           type=int,   default=64)
    parser.add_argument("--rank",              type=int,   default=8)
    parser.add_argument("--max_delta_ratio",   type=float, default=0.02)
    parser.add_argument("--residual_window_K", type=int,   default=24)

    # Checkpoints
    parser.add_argument("--p1_ckpt", type=str, default=None,
                        help="Phase-1 checkpoint（drift+delta 训练后）")
    parser.add_argument("--p2_ckpt", type=str, default=None,
                        help="Phase-2 checkpoint（gate 训练后）。若为空则只评估 P1。")

    # Misc
    parser.add_argument("--num_workers",    type=int, default=0)
    parser.add_argument("--experiment_tag", type=str, default="")
    parser.add_argument("--out_dir",        type=str, default="logs/prompt_z")
    parser.add_argument(
        "--oracle_target_margin_pct", type=float, default=0.0,
        help=("Relative improvement margin for trainable-label diagnostics. "
              "Exact oracle metrics always use zero margin."),
    )
    parser.add_argument(
        "--save_oracle_targets", action="store_true", default=False,
        help=("Save per-window/per-channel TEST labels for diagnostics only. "
              "Never use them to train or select the gate."),
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # enc_in
    if args.enc_in is None:
        import pandas as pd
        csv_candidates = [
            os.path.join(args.root_path, args.data_path),
            os.path.join("./dataset", args.data_path),
            os.path.join("./data", args.data_path),
            args.data_path,
        ]
        for c in csv_candidates:
            if os.path.exists(c):
                df = pd.read_csv(c)
                args.enc_in = len([col for col in df.columns if col.lower() != "date"])
                print(f"[*] enc_in={args.enc_in} (from {c})")
                break
        else:
            raise FileNotFoundError(
                f"Cannot find '{args.data_path}'. Pass --enc_in directly."
            )

    if args.p1_ckpt is None and args.p2_ckpt is None:
        raise ValueError("At least one of --p1_ckpt or --p2_ckpt must be specified.")

    adapter  = build_backbone(args, device)
    # hidden_layout from adapter
    args.hidden_layout = adapter.hidden_layout

    os.makedirs(args.out_dir, exist_ok=True)
    ds  = args.data_path.replace(".csv", "")
    tag = f"eval_test_{ds}_H{args.forecast_H}"
    if args.experiment_tag:
        tag = f"{tag}_{args.experiment_tag}"

    warmup_loader, test_loader, dataset = get_splits(args)

    # ── 评估 P1 checkpoint ───────────────────────────────────────────────
    if args.p1_ckpt:
        print(f"\n[*] Loading P1 checkpoint: {args.p1_ckpt}")
        prompt_z_p1 = build_prompt_z(args, device)
        prompt_z_p1.load_state_dict(torch.load(args.p1_ckpt, map_location=device))
        prompt_z_p1.eval()

        r_p1 = evaluate(adapter, prompt_z_p1, warmup_loader, test_loader, args, device)
        print_results(r_p1, label=f"P1 Ckpt Test ({os.path.basename(args.p1_ckpt)})")
        p1_out = os.path.join(args.out_dir, f"{tag}_p1.json")
        save_results(r_p1, p1_out)
        if args.save_oracle_targets:
            save_oracle_targets(r_p1, p1_out.replace(".json", "_oracle_targets.jsonl"))

    # ── 评估 P2 checkpoint ───────────────────────────────────────────────
    if args.p2_ckpt:
        print(f"\n[*] Loading P2 checkpoint: {args.p2_ckpt}")
        prompt_z_p2 = build_prompt_z(args, device)
        prompt_z_p2.load_state_dict(torch.load(args.p2_ckpt, map_location=device))
        prompt_z_p2.eval()

        r_p2 = evaluate(adapter, prompt_z_p2, warmup_loader, test_loader, args, device)
        print_results(r_p2, label=f"P2 Ckpt Test ({os.path.basename(args.p2_ckpt)})")
        p2_out = os.path.join(args.out_dir, f"{tag}_p2.json")
        save_results(r_p2, p2_out)
        if args.save_oracle_targets:
            save_oracle_targets(r_p2, p2_out.replace(".json", "_oracle_targets.jsonl"))

    # ── 若两个 checkpoint 都有，打印横向对比 ─────────────────────────────
    if args.p1_ckpt and args.p2_ckpt:
        print("\n── P1 vs P2 横向对比 ────────────────────────────────────────────")
        print(f"  {'Metric':<35s}  {'P1':>10s}  {'P2':>10s}")
        print(f"  {'-'*35}  {'-'*10}  {'-'*10}")
        for k in ["mse_frozen", "mse_fixed_gamma1", "mse_learned_gamma",
                  "mse_oracle_window", "mse_oracle_channel",
                  "mse_oracle_continuous_channel",
                  "impr_fixed_vs_frozen", "impr_oracle_window_vs_fixed1",
                  "impr_oracle_channel_vs_fixed1",
                  "oracle_window_helpful_frac", "oracle_channel_helpful_frac"]:
            v1 = r_p1.get(k, float("nan"))
            v2 = r_p2.get(k, float("nan"))
            is_pct = "impr" in k or "frac" in k
            fmt = f"{v1:>10.2f}" if is_pct else f"{v1:>10.6f}"
            fmt2 = f"{v2:>10.2f}" if is_pct else f"{v2:>10.6f}"
            print(f"  {k:<35s}  {fmt}  {fmt2}")
        print("─────────────────────────────────────────────────────────────────")
        print("  注：P1 fixed_gamma=1 ≈ P2 fixed_gamma=1（P2 只训练 gate，delta 相同）")

    print(f"\n[*] Results saved to: {args.out_dir}/{tag}_p*.json")


if __name__ == "__main__":
    main()
