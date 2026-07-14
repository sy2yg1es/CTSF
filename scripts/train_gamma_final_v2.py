"""
train_gamma_final_v2.py — Clean Two-Phase Delta → Gate Ablation
================================================================

修订核心：

  [1] Phase 2 只训练 confidence_gate，冻结 drift_encoder + low_rank_mod
      → 排除"Phase 2 提升来自 delta 继续训练"的混淆

  [2] 不重叠时间窗口：
        Phase 1: train windows [0,           phase1_steps)
        Phase 2: train windows [phase1_steps, phase1_steps + phase2_steps)
        Validation: dataset val split（独立）
      Phase 2 开始前先 warmup tracker 至 Phase 1 结束状态

  [3] Excess penalty 替代 raw_ratio 正则：
        excess  = relu(unclamped_ratio - max_delta_ratio)^2  per channel
        penalty = lambda_excess * excess.mean()
      目标：unclamped d/h ≈ max_delta_ratio，clamped-at-cap < 30%

  [4] Phase 2 val 评估同时报告 learned_gamma 和 fixed_gamma=1 的 MSE
      只有 learned < fixed 才说明 gate 有选择价值

  [5] phase1_pass_threshold 默认 0.2%

使用示例：
    python scripts/train_gamma_final_v2.py \\
        --data_path ETTm1.csv \\
        --forecast_H 1 \\
        --pretrained_weights weights/patchtst_pretrained_ETTm1_H1.pth \\
        --phase1_steps 1000 \\
        --phase2_steps 2000 \\
        --max_delta_ratio 0.02 \\
        --lambda_excess 1.0 \\
        --experiment_tag v2_test

    # 跳过阶段一：
    python scripts/train_gamma_final_v2.py \\
        --data_path ETTm1.csv --forecast_H 1 \\
        --skip_phase1 --phase1_ckpt weights/prompt_z/gfv2_ETTm1_H1_v2_test_p1.pth \\
        --phase2_steps 2000 --experiment_tag v2_test

输出：
    weights/prompt_z/gfv2_<tag>_p1.pth
    weights/prompt_z/gfv2_<tag>_p2_best.pth
    weights/prompt_z/gfv2_<tag>_p2_final.pth
    logs/prompt_z/gfv2_<tag>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

        missing, unexpected = adapter.backbone.load_state_dict(
            backbone_state, strict=False
        )
        if missing:
            raise RuntimeError(
                f"Backbone load FAILED — missing keys: {missing[:5]}... "
                f"({len(missing)} total). Check --pretrained_weights path and backbone config."
            )
        if unexpected:
            print(f"[!] Backbone: {len(unexpected)} unexpected keys (benign if loading subset): "
                  f"{unexpected[:3]}")
        print(f"[*] Backbone loaded: {args.pretrained_weights}")

    for p in adapter.parameters():
        p.requires_grad = False
    adapter.eval()
    return adapter


# ============================================================================
# Data
# ============================================================================

def get_datasets(args):
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
    train_n = dataset.train_size

    # 从训练集尾部切分，保证 P1→P2→Val 时间连续
    p2_end   = train_n
    p2_start = p2_end   - args.phase2_steps
    p1_end   = p2_start
    p1_start = p1_end   - args.phase1_steps

    if p1_start < 0:
        raise ValueError(
            f"phase1_steps({args.phase1_steps}) + phase2_steps({args.phase2_steps}) "
            f"= {args.phase1_steps + args.phase2_steps} exceeds train_size={train_n}."
        )

    # 保存实际 dataset 索引，供各阶段打印用
    args.p1_start = p1_start
    args.p1_end   = p1_end
    args.p2_start = p2_start
    args.p2_end   = p2_end
    args.val_start  = dataset.val_start
    args.test_start = dataset.test_start

    p1_subset  = Subset(dataset, range(p1_start, p1_end))
    p2_subset  = Subset(dataset, range(p2_start, p2_end))
    val_subset = Subset(dataset, range(dataset.val_start, dataset.test_start))
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)

    if args.phase1_selection_source == "train_tail":
        selection_steps = max(1, int(args.phase1_steps * args.phase1_selection_fraction))
        fit_end = p1_end - selection_steps
        if fit_end <= p1_start:
            raise ValueError(
                "phase1_selection_fraction leaves no Phase-1 fitting windows"
            )
        p1_subset = Subset(dataset, range(p1_start, fit_end))
        selection_subset = Subset(dataset, range(fit_end, p1_end))
        phase1_selection_loader = DataLoader(
            selection_subset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        args.p1_fit_end = fit_end
        args.p1_selection_start = fit_end
        print(
            f"[*] P1 clean split: fit=[{p1_start},{fit_end})  "
            f"selection=[{fit_end},{p1_end})  validation untouched"
        )
    else:
        phase1_selection_loader = val_loader
        args.p1_fit_end = p1_end
        args.p1_selection_start = dataset.val_start

    print(f"[*] Train total: {train_n}  |  "
          f"P1: [{p1_start},{p1_end})  P2: [{p2_start},{p2_end})  "
          f"Val: [{dataset.val_start},{dataset.test_start})")
    return dataset, p1_subset, p2_subset, val_loader, phase1_selection_loader


def make_loader(subset, args):
    return DataLoader(subset, batch_size=1, shuffle=False,
                      num_workers=args.num_workers, drop_last=False)


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
# Core forward
# ============================================================================

def pz_forward(X, adapter, prompt_z, stats_tensor, *, gamma_mode):
    """
    gamma_mode:
      "ones"        : gamma=1，confidence_gate 脱离计算图
      "learned"     : gamma = confidence_gate(drift_state)，可学
      "ones_nograd" : gamma=1，全程 no_grad（tracker warmup 专用）

    mask 始终为 ones，sparse_mask 不进任何计算图。

    Returns dict: Y_hat, Y_frozen, delta_h_raw, delta_h_clamped,
                  hidden, unclamped_ratio_live (or None), diag
    """
    layout = prompt_z.hidden_layout

    with torch.no_grad():
        hidden, means, stdev = adapter.encode_until_hook(X)
    hidden = hidden.detach()
    means  = means.detach()
    stdev  = stdev.detach()

    with torch.no_grad():
        Y_frozen = adapter.decode_from_hook(hidden, means, stdev)

    if gamma_mode == "ones_nograd":
        with torch.no_grad():
            summary     = prompt_z._hidden_summary(hidden)
            drift_state = prompt_z.drift_encoder(summary, stats_tensor)
            if layout == "BCDP":
                h_w         = hidden.permute(0, 1, 3, 2)
                delta_h_raw = prompt_z.low_rank_mod(h_w, drift_state)
                delta_h_raw = delta_h_raw.permute(0, 1, 3, 2)
            else:
                delta_h_raw = prompt_z.low_rank_mod(hidden, drift_state)
            delta_h_clamped = prompt_z._ratio_clamp(delta_h_raw, hidden)
            hidden_mod = hidden + delta_h_clamped
            Y_hat = adapter.decode_from_hook(hidden_mod, means, stdev)
            unclamped_ratio_live = None
    else:
        summary     = prompt_z._hidden_summary(hidden)
        drift_state = prompt_z.drift_encoder(summary, stats_tensor)

        if gamma_mode == "ones":
            with torch.no_grad():
                gamma_raw = prompt_z.confidence_gate(drift_state)
            gamma = torch.ones_like(gamma_raw)
        else:
            gamma = prompt_z.confidence_gate(drift_state)

        with torch.no_grad():
            prompt_z.sparse_mask(drift_state)   # run but discard (mask=ones)

        if layout == "BCDP":
            h_w         = hidden.permute(0, 1, 3, 2)
            delta_h_raw = prompt_z.low_rank_mod(h_w, drift_state)
            delta_h_raw = delta_h_raw.permute(0, 1, 3, 2)
        else:
            delta_h_raw = prompt_z.low_rank_mod(hidden, drift_state)

        # unclamped ratio (live) for excess penalty
        d_flat    = delta_h_raw.flatten(2)
        h_flat    = hidden.detach().flatten(2)
        d_norm    = d_flat.norm(dim=-1)                    # [B,C]
        h_norm    = h_flat.norm(dim=-1).clamp(min=1e-8)
        unclamped_ratio_live = d_norm / h_norm             # [B,C], live

        delta_h_clamped = prompt_z._ratio_clamp(delta_h_raw, hidden)

        if layout == "BCDP":
            applied = gamma.unsqueeze(-1) * delta_h_clamped
        else:
            applied = gamma * delta_h_clamped

        hidden_mod = hidden + applied
        Y_hat = adapter.decode_from_hook(hidden_mod, means, stdev)

    # Diagnostics (detached)
    with torch.no_grad():
        df_raw  = delta_h_raw.detach().flatten(2)
        df_clmp = delta_h_clamped.detach().flatten(2)
        h_flat2 = hidden.flatten(2)
        h_n     = h_flat2.norm(dim=-1).clamp(min=1e-8)

        unclamp_d2h = (df_raw.norm(dim=-1) / h_n).mean().item()
        clamp_d2h   = (df_clmp.norm(dim=-1) / h_n).mean().item()
        max_allowed = prompt_z.max_delta_ratio * h_n
        at_cap = (df_raw.norm(dim=-1) > max_allowed + 1e-6).float().mean().item()

        if gamma_mode != "ones_nograd":
            g = gamma.detach()
            gd = {
                "gamma_mean":      g.mean().item(),
                "gamma_std":       g.std().item(),
                "gamma_min":       g.min().item(),
                "gamma_max":       g.max().item(),
                "gamma_spread":    g.std(dim=1).mean().item(),
                "frac_gamma_lt01": (g < 0.1).float().mean().item(),
                "frac_gamma_gt09": (g > 0.9).float().mean().item(),
            }
        else:
            gd = dict(gamma_mean=1.0, gamma_std=0.0, gamma_min=1.0,
                      gamma_max=1.0, gamma_spread=0.0,
                      frac_gamma_lt01=0.0, frac_gamma_gt09=1.0)

        diag = {**gd,
                "unclamped_d2h": unclamp_d2h,
                "clamped_d2h":   clamp_d2h,
                "at_cap_frac":   at_cap}

    return dict(Y_hat=Y_hat, Y_frozen=Y_frozen,
                delta_h_raw=delta_h_raw, delta_h_clamped=delta_h_clamped,
                hidden=hidden, unclamped_ratio_live=unclamped_ratio_live,
                diag=diag)


def excess_penalty_fn(unclamped_ratio_live, max_delta_ratio):
    """relu(ratio - cap)^2 mean, live."""
    return torch.relu(unclamped_ratio_live - max_delta_ratio).pow(2).mean()


# ============================================================================
# Val evaluation
# ============================================================================

@torch.no_grad()
def evaluate_val_p1(adapter, prompt_z, val_loader, args, device):
    """Phase 1 专用 val 评估：只评估 fixed_gamma=1，不运行 confidence_gate。
    Phase 1 没有训练 gate，报 'learned_gamma' 会产生误导（实际是 init-gamma≈0.5）。
    """
    tracker = ResidualTracker(args.enc_in, args.residual_window_K).to(device)
    tracker.reset()
    rc = deque()
    total_fixed = total_frozen = 0.0
    n = 0

    prompt_z.eval()
    for X, Y in val_loader:
        X = X.to(device); Y = Y.to(device)
        st  = pack_stats(tracker, device)
        out = pz_forward(X, adapter, prompt_z, st, gamma_mode="ones")

        total_fixed  += F.mse_loss(out["Y_hat"],    Y).item()
        total_frozen += F.mse_loss(out["Y_frozen"], Y).item()
        n += 1

        rc.append((out["Y_frozen"].detach(), Y.detach()))
        if len(rc) > args.forecast_H:
            op, ot = rc.popleft(); tracker.update(op, ot)
        else:
            tracker.step_no_update()

    prompt_z.train()
    af  = total_fixed  / max(n, 1)
    afr = total_frozen / max(n, 1)
    return {
        "val_mse_frozen":       afr,
        "val_mse_fixed_gamma1": af,
        "val_impr_fixed_pct":   (afr - af) / max(afr, 1e-12) * 100,
        "n_val_windows":        n,
    }


@torch.no_grad()
def evaluate_val_dual(adapter, prompt_z, val_loader, args, device):
    """Phase 2 val 评估：同时报告 fixed_gamma=1 和 learned_gamma（gate 已训练）。"""
    tracker = ResidualTracker(args.enc_in, args.residual_window_K).to(device)
    tracker.reset()
    rc = deque()
    total_fixed = total_learned = total_frozen = 0.0
    n = 0

    prompt_z.eval()
    for X, Y in val_loader:
        X = X.to(device)
        Y = Y.to(device)
        st = pack_stats(tracker, device)

        out1 = pz_forward(X, adapter, prompt_z, st, gamma_mode="ones")
        out2 = pz_forward(X, adapter, prompt_z, st, gamma_mode="learned")

        total_fixed   += F.mse_loss(out1["Y_hat"],    Y).item()
        total_learned += F.mse_loss(out2["Y_hat"],    Y).item()
        total_frozen  += F.mse_loss(out1["Y_frozen"], Y).item()
        n += 1

        rc.append((out1["Y_frozen"].detach(), Y.detach()))
        if len(rc) > args.forecast_H:
            op, ot = rc.popleft(); tracker.update(op, ot)
        else:
            tracker.step_no_update()

    prompt_z.train()
    af = total_fixed / max(n, 1)
    al = total_learned / max(n, 1)
    afr = total_frozen / max(n, 1)
    return {
        "val_mse_frozen":        afr,
        "val_mse_fixed_gamma1":  af,
        "val_mse_learned_gamma": al,
        "val_impr_fixed_pct":    (afr - af) / max(afr, 1e-12) * 100,
        "val_impr_learned_pct":  (afr - al) / max(afr, 1e-12) * 100,
        "gate_delta_mse":        al - af,    # <0 => learned beats fixed
        "n_val_windows":         n,
    }


def print_val(vr, label="Val"):
    d = vr["gate_delta_mse"]
    sign = ("✓ learned<fixed" if d < -1e-7 else
            "✗ learned>fixed" if d > 1e-7 else "≈ equal")
    print(f"[{label}]  frozen={vr['val_mse_frozen']:.6f}  "
          f"fixed_γ=1={vr['val_mse_fixed_gamma1']:.6f} ({vr['val_impr_fixed_pct']:+.2f}%)  "
          f"learned_γ={vr['val_mse_learned_gamma']:.6f} ({vr['val_impr_learned_pct']:+.2f}%)  "
          f"Δ={d:+.7f} [{sign}]")


# ============================================================================
# Phase 1: Delta Warmup
# ============================================================================

def run_phase1(adapter, prompt_z, p1_subset, val_loader,
               args, device, log_f, tag, save_dir):
    print("\n" + "=" * 70)
    print("  Phase 1: Delta Warmup")
    print(f"  Trainable: drift_encoder + low_rank_mod")
    print(f"  gamma=1 (fixed)  |  lambda_excess={args.lambda_excess}  |  lambda_delta=0")
    selection_name = (
        "train-tail selection"
        if args.phase1_selection_source == "train_tail"
        else "validation"
    )
    print(
        f"  Checkpoint selection: {selection_name} fixed_gamma=1 MSE  "
        f"(every {args.val_interval} steps)"
    )
    print("=" * 70)

    p1_params = (list(prompt_z.drift_encoder.parameters())
                 + list(prompt_z.low_rank_mod.parameters()))
    print(f"[P1] Trainable: {sum(p.numel() for p in p1_params):,} params")

    opt     = torch.optim.AdamW(p1_params, lr=args.lr, weight_decay=args.weight_decay)
    loader  = make_loader(p1_subset, args)
    tracker = ResidualTracker(args.enc_in, args.residual_window_K).to(device)
    tracker.reset()
    rc = deque()

    ckpt     = os.path.join(save_dir, f"{tag}_p1.pth")
    best_val = float("inf")  # 按 val fixed_gamma=1 MSE 选 checkpoint
    step     = 0
    t0       = time.time()

    prompt_z.train()
    for X, Y in loader:
        X = X.to(device); Y = Y.to(device)
        opt.zero_grad()
        st = pack_stats(tracker, device)

        out = pz_forward(X, adapter, prompt_z, st, gamma_mode="ones")
        forecast_loss = F.mse_loss(out["Y_hat"], Y)
        exc           = excess_penalty_fn(out["unclamped_ratio_live"], args.max_delta_ratio)
        loss          = forecast_loss + args.lambda_excess * exc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p1_params, max_norm=1.0)
        opt.step()

        with torch.no_grad():
            mse_fr = F.mse_loss(out["Y_frozen"], Y).item()
            impr   = (mse_fr - forecast_loss.item()) / max(mse_fr, 1e-12) * 100
            d = out["diag"]
            row = {"phase": 1, "step": step,
                   "forecast_loss": forecast_loss.item(), "frozen_loss": mse_fr,
                   "improvement": impr, "excess_penalty": exc.item(), **d}
        log_f.write(json.dumps(row) + "\n"); log_f.flush()

        if step % args.log_interval == 0:
            print(f"[P1|{step:5d}]  loss={forecast_loss.item():.6f}  "
                  f"frozen={mse_fr:.6f}  impr={impr:+.2f}%  "
                  f"excess={exc.item():.6f}  "
                  f"unclamp={d['unclamped_d2h']:.5f}  "
                  f"clamp={d['clamped_d2h']:.5f}  at_cap={d['at_cap_frac']:.2f}")

        # 定期 val 评估，按 fixed_gamma=1 MSE 保存 best checkpoint
        # Phase 1 没有训练 gate，不报 learned_gamma（避免误读）
        if (step + 1) % args.val_interval == 0:
            vr_snap = evaluate_val_p1(adapter, prompt_z, val_loader, args, device)
            tag_s   = f"P1 Val@{step+1}"
            print(f"[{tag_s}]  frozen={vr_snap['val_mse_frozen']:.6f}  "
                  f"fixed_γ=1={vr_snap['val_mse_fixed_gamma1']:.6f} "
                  f"({vr_snap['val_impr_fixed_pct']:+.2f}%)"
                  + ("  → new best" if vr_snap["val_mse_fixed_gamma1"] < best_val else ""))
            if vr_snap["val_mse_fixed_gamma1"] < best_val:
                best_val = vr_snap["val_mse_fixed_gamma1"]
                torch.save(prompt_z.state_dict(), ckpt)
            log_f.write(json.dumps(
                {"phase": 1, "step": step, "val_snapshot": True, **vr_snap}
            ) + "\n"); log_f.flush()

        with torch.no_grad():
            rc.append((out["Y_frozen"].detach(), Y.detach()))
            if len(rc) > args.forecast_H:
                op, ot = rc.popleft(); tracker.update(op, ot)
            else:
                tracker.step_no_update()

        step += 1

    # 若从未触发 val（steps < val_interval），做一次最终评估
    if best_val == float("inf"):
        vr_snap = evaluate_val_p1(adapter, prompt_z, val_loader, args, device)
        best_val = vr_snap["val_mse_fixed_gamma1"]
        torch.save(prompt_z.state_dict(), ckpt)

    dt = time.time() - t0
    print(f"\n[P1] Done {dt:.0f}s  best_val_fixed_gamma1={best_val:.6f}  ckpt={ckpt}")

    # 加载最佳 checkpoint 做最终 val 报告
    prompt_z.load_state_dict(torch.load(ckpt, map_location=device))
    vr_final = evaluate_val_p1(adapter, prompt_z, val_loader, args, device)
    print(f"[P1 Final Selection]  frozen={vr_final['val_mse_frozen']:.6f}  "
          f"fixed_γ=1={vr_final['val_mse_fixed_gamma1']:.6f} "
          f"({vr_final['val_impr_fixed_pct']:+.2f}%)  "
          f"(n={vr_final['n_val_windows']})")
    return ckpt, vr_final


# ============================================================================
# Phase 2: Gate Only
# ============================================================================

def warmup_tracker(adapter, prompt_z, p1_subset, args, device):
    """No-grad pass through Phase-1 windows to rebuild tracker + residual cache state.
    返回 (tracker, rc)，Phase 2 直接接续，不重置 rc，保持 P1→P2 流式连续。
    """
    print("[P2] Tracker warmup through Phase-1 windows ...")
    tracker = ResidualTracker(args.enc_in, args.residual_window_K).to(device)
    tracker.reset()
    rc = deque()
    prompt_z.eval()
    with torch.no_grad():
        for X, Y in make_loader(p1_subset, args):
            X = X.to(device); Y = Y.to(device)
            st  = pack_stats(tracker, device)
            out = pz_forward(X, adapter, prompt_z, st, gamma_mode="ones_nograd")
            rc.append((out["Y_frozen"].detach(), Y.detach()))
            if len(rc) > args.forecast_H:
                op, ot = rc.popleft(); tracker.update(op, ot)
            else:
                tracker.step_no_update()
    prompt_z.train()
    print(f"[P2] Tracker ready: count={int(tracker._count.item())}  "
          f"pending_rc={len(rc)} (will be flushed in first {args.forecast_H} P2 steps)")
    return tracker, rc  # rc 一并返回，保持 pending residuals 连续


def run_phase2(adapter, prompt_z, p1_subset, p2_subset, val_loader,
               args, device, log_f, tag, save_dir, p1_ckpt):
    print("\n" + "=" * 70)
    print("  Phase 2: Gate Training (confidence_gate ONLY)")
    print(f"  Windows [{args.p2_start},{args.p2_end})  "
          f"({args.phase2_steps} steps, dataset indices)")
    print(f"  Frozen: drift_encoder + low_rank_mod")
    print(f"  Trainable: confidence_gate  |  lr_gate={args.lr_gate}")
    print("=" * 70)

    prompt_z.load_state_dict(torch.load(p1_ckpt, map_location=device))

    # 冻结 drift + delta
    for p in prompt_z.drift_encoder.parameters():
        p.requires_grad = False
    for p in prompt_z.low_rank_mod.parameters():
        p.requires_grad = False

    gate_params = list(prompt_z.confidence_gate.parameters())
    print(f"[P2] Gate params: {sum(p.numel() for p in gate_params):,}")

    opt = torch.optim.AdamW(gate_params, lr=args.lr_gate, weight_decay=args.weight_decay)

    tracker, rc = warmup_tracker(adapter, prompt_z, p1_subset, args, device)
    # 不重新 deque()：直接接续 P1 末尾的 pending residuals

    best_path  = os.path.join(save_dir, f"{tag}_p2_best.pth")
    final_path = os.path.join(save_dir, f"{tag}_p2_final.pth")
    best_val   = float("inf")
    step       = 0
    t0         = time.time()

    prompt_z.train()
    for X, Y in make_loader(p2_subset, args):
        X = X.to(device); Y = Y.to(device)
        opt.zero_grad()
        st = pack_stats(tracker, device)

        out           = pz_forward(X, adapter, prompt_z, st, gamma_mode="learned")
        forecast_loss = F.mse_loss(out["Y_hat"], Y)
        # delta frozen → excess penalty 无意义，只用 forecast loss
        forecast_loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
        opt.step()

        with torch.no_grad():
            mse_fr = F.mse_loss(out["Y_frozen"], Y).item()
            impr   = (mse_fr - forecast_loss.item()) / max(mse_fr, 1e-12) * 100
            d = out["diag"]
            row = {"phase": 2, "step": step,
                   "forecast_loss": forecast_loss.item(), "frozen_loss": mse_fr,
                   "improvement": impr, **d}
        log_f.write(json.dumps(row) + "\n"); log_f.flush()

        if step % args.log_interval == 0:
            print(f"[P2|{step:5d}]  loss={forecast_loss.item():.6f}  "
                  f"frozen={mse_fr:.6f}  impr={impr:+.2f}%  "
                  f"γ={d['gamma_mean']:.4f}±{d['gamma_std']:.4f}"
                  f"[{d['gamma_min']:.3f},{d['gamma_max']:.3f}]  "
                  f"spread={d['gamma_spread']:.4f}  "
                  f"lt0.1={d['frac_gamma_lt01']:.2f}  gt0.9={d['frac_gamma_gt09']:.2f}")

        if step > 0 and step % args.val_interval == 0:
            vr = evaluate_val_dual(adapter, prompt_z, val_loader, args, device)
            print_val(vr, label=f"P2 Val@{step}")
            if vr["val_mse_learned_gamma"] < best_val:
                best_val = vr["val_mse_learned_gamma"]
                torch.save(prompt_z.state_dict(), best_path)
                print(f"  → val best={best_val:.6f}")
            log_f.write(json.dumps({
                "phase": 2, "step": step, "val_snapshot": True, **vr
            }) + "\n"); log_f.flush()

        with torch.no_grad():
            rc.append((out["Y_frozen"].detach(), Y.detach()))
            if len(rc) > args.forecast_H:
                op, ot = rc.popleft(); tracker.update(op, ot)
            else:
                tracker.step_no_update()

        step += 1

    # Final val
    vr = evaluate_val_dual(adapter, prompt_z, val_loader, args, device)
    print_val(vr, label="P2 Val Final")
    if vr["val_mse_learned_gamma"] < best_val:
        best_val = vr["val_mse_learned_gamma"]
        torch.save(prompt_z.state_dict(), best_path)
    torch.save(prompt_z.state_dict(), final_path)

    print(f"\n[P2] Done {time.time()-t0:.0f}s  best_val={best_val:.6f}")
    print(f"[P2] Best : {best_path}")
    print(f"[P2] Final: {final_path}")

    # 恢复 grad
    for p in prompt_z.drift_encoder.parameters(): p.requires_grad = True
    for p in prompt_z.low_rank_mod.parameters():  p.requires_grad = True

    _print_tail_stats(log_f.name, phase=2)


# ============================================================================
# Tail stats + gate diagnosis
# ============================================================================

def _print_tail_stats(log_path, phase, tail_n=200):
    try:
        lines = open(log_path, encoding="utf-8").readlines()
    except FileNotFoundError:
        return
    rows = [json.loads(l) for l in lines
            if json.loads(l).get("phase") == phase
            and not json.loads(l).get("val_snapshot")]
    tail = rows[-tail_n:]
    if not tail:
        return

    keys = ["forecast_loss", "frozen_loss", "improvement",
            "unclamped_d2h", "clamped_d2h", "at_cap_frac",
            "gamma_mean", "gamma_std", "gamma_spread",
            "frac_gamma_lt01", "frac_gamma_gt09"]
    print(f"\n── Phase {phase} 最后 {len(tail)} 步均值 ────────────────────────────")
    avgs = {}
    for k in keys:
        vals = [r.get(k, 0.0) for r in tail]
        avgs[k] = sum(vals) / len(vals)
        print(f"  {k:<28s}: {avgs[k]:.6f}")

    if phase == 1:
        print("\n── Delta Health ──────────────────────────────────────────────────")
        cap = avgs["at_cap_frac"]
        ratio = avgs["unclamped_d2h"] / max(avgs["clamped_d2h"], 1e-8)
        status = "✓" if cap < 0.3 else ("△" if cap < 0.6 else "✗")
        print(f"  {status} at_cap_frac={cap:.2f}  "
              f"unclamped/clamped={ratio:.1f}x  (target cap<30%, ratio≈1~3x)")

    if phase == 2:
        print("\n── Gate Diagnosis ────────────────────────────────────────────────")
        print("  Key: compare learned_gamma vs fixed_gamma=1 in Val snapshots above.")
        spread = avgs["gamma_spread"]
        s = "✓" if spread > 0.08 else "△"
        print(f"  {s} gamma_spread={spread:.4f}  "
              f"lt0.1={avgs['frac_gamma_lt01']:.2f}  gt0.9={avgs['frac_gamma_gt09']:.2f}")
        print("─────────────────────────────────────────────────────────────────")


# ============================================================================
# Main
# ============================================================================

def run(args, device):
    adapter  = build_backbone(args, device)
    prompt_z = PromptZModulator(
        d_model=args.D_model, hidden_layout=adapter.hidden_layout,
        d_drift=args.d_drift, rank=args.rank,
        gamma_init_bias=args.gamma_init_bias,
        mask_init_bias=-1.5,
        max_delta_ratio=args.max_delta_ratio,
    ).to(device)
    print(f"[*] PromptZ: {sum(p.numel() for p in prompt_z.parameters()):,} params  "
          f"(SparseMaskHead always excluded from optimizer)")

    dataset, p1_sub, p2_sub, val_loader, phase1_selection_loader = get_datasets(args)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", "prompt_z"), exist_ok=True)
    ds  = args.data_path.replace(".csv", "")
    tag = f"gfv2_{ds}_H{args.forecast_H}"
    if args.experiment_tag:
        tag = f"{tag}_{args.experiment_tag}"
    log_path = os.path.join("logs", "prompt_z", f"{tag}.jsonl")
    print(f"[*] Tag: {tag}  Log: {log_path}")

    log_f = open(log_path, "w", encoding="utf-8")

    # Phase 1
    if args.skip_phase1:
        if not args.phase1_ckpt:
            raise ValueError("--skip_phase1 requires --phase1_ckpt")
        p1_ckpt = args.phase1_ckpt
        prompt_z.load_state_dict(torch.load(p1_ckpt, map_location=device))
        vr1 = evaluate_val_dual(adapter, prompt_z, val_loader, args, device)
        print_val(vr1, label="P1 ckpt Val")
    else:
        p1_ckpt, vr1 = run_phase1(
            adapter, prompt_z, p1_sub, phase1_selection_loader,
            args, device, log_f, tag, args.save_dir,
        )

    if args.stop_after_phase1:
        print(
            "[DECISION] stop_after_phase1=true; validation and Phase 2 remain untouched."
        )
        log_f.close()
        return

    # Decision
    print(f"\n{'='*70}")
    impr1 = vr1["val_impr_fixed_pct"]
    go    = args.force_phase2 or (impr1 >= args.phase1_pass_threshold)
    if go:
        src = "force" if args.force_phase2 else f"{impr1:+.2f}% >= {args.phase1_pass_threshold}%"
        print(f"[DECISION] ✓ Entering Phase 2 ({src})")
    else:
        print(f"[DECISION] ✗ Phase-1 improvement {impr1:+.2f}% < {args.phase1_pass_threshold}%")
        print("           Add --force_phase2 to proceed anyway.")
        log_f.close()
        return
    print(f"{'='*70}\n")

    # Phase 2
    run_phase2(
        adapter, prompt_z, p1_sub, p2_sub, val_loader,
        args, device, log_f, tag, args.save_dir, p1_ckpt,
    )
    log_f.close()
    print(f"\n[*] Done. Log: {log_path}")


def main():
    p = argparse.ArgumentParser("Two-Phase Delta Warmup + Gate-Only Training (v2)")
    p.add_argument("--root_path",   default="./data")
    p.add_argument("--data_path",   required=True)
    p.add_argument("--features",    default="M")
    p.add_argument("--seq_len",     type=int, default=96)
    p.add_argument("--forecast_H", type=int, required=True)
    p.add_argument("--enc_in",      type=int, default=None)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio",   type=float, default=0.1)

    p.add_argument("--backbone",           default="patchtst",
                   choices=["patchtst", "itransformer"])
    p.add_argument("--D_model",            type=int,   default=512)
    p.add_argument("--d_ff",               type=int,   default=512)
    p.add_argument("--e_layers",           type=int,   default=3)
    p.add_argument("--pretrained_weights", default=None)

    p.add_argument("--d_drift",           type=int,   default=64)
    p.add_argument("--rank",              type=int,   default=8)
    p.add_argument("--gamma_init_bias",   type=float, default=0.0)
    p.add_argument("--max_delta_ratio",   type=float, default=0.02,
                   help="Delta clamp cap. Also target for excess penalty.")
    p.add_argument("--residual_window_K", type=int,   default=24)

    p.add_argument("--phase1_steps",          type=int,   default=1000)
    p.add_argument("--phase2_steps",          type=int,   default=2000)
    p.add_argument("--phase1_pass_threshold", type=float, default=0.2,
                   help="Min val improvement %% for Phase-1 to pass (recommend 0.2~0.5)")
    p.add_argument(
        "--phase1_selection_source",
        choices=["validation", "train_tail"],
        default="validation",
        help="Use train_tail for a clean delta checkpoint without validation labels",
    )
    p.add_argument("--phase1_selection_fraction", type=float, default=0.2)
    p.add_argument("--stop_after_phase1", action="store_true", default=False)
    p.add_argument("--force_phase2",  action="store_true", default=False)
    p.add_argument("--skip_phase1",   action="store_true", default=False)
    p.add_argument("--phase1_ckpt",   default=None)
    p.add_argument("--val_interval",  type=int, default=200)

    p.add_argument("--lr",           type=float, default=1e-3,
                   help="Phase-1 lr (drift+delta)")
    p.add_argument("--lr_gate",      type=float, default=5e-4,
                   help="Phase-2 lr (gate only)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_excess", type=float, default=1.0,
                   help="Excess penalty coeff. relu(unclamp_ratio - cap)^2")

    p.add_argument("--num_workers",    type=int, default=0)
    p.add_argument("--save_dir",       default="weights/prompt_z")
    p.add_argument("--experiment_tag", default="")
    p.add_argument("--log_interval",   type=int, default=100)

    args = p.parse_args()
    if not (0.0 < args.phase1_selection_fraction < 0.5):
        p.error("--phase1_selection_fraction must be in (0,0.5)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    if args.enc_in is None:
        import pandas as pd
        csv_path = os.path.join(args.root_path, args.data_path)
        if not os.path.exists(csv_path):
            # 尝试常见备选路径
            candidates = [
                os.path.join("./dataset", args.data_path),
                os.path.join("./datasets", args.data_path),
                os.path.join("./data", args.data_path),
                args.data_path,
            ]
            for c in candidates:
                if os.path.exists(c):
                    csv_path = c
                    print(f"[*] Data found at: {csv_path}")
                    break
            else:
                raise FileNotFoundError(
                    f"Cannot find '{args.data_path}' under root_path='{args.root_path}'.\n"
                    f"Tried: {candidates}\n"
                    f"Please pass --root_path <correct_path>  or  --enc_in <N> directly."
                )
        df = pd.read_csv(csv_path)
        args.enc_in = len([c for c in df.columns if c.lower() != "date"])
        print(f"[*] enc_in={args.enc_in}")

    print(f"[*] P1: gamma=1, drift+delta, lambda_excess={args.lambda_excess}, "
          f"max_delta_ratio={args.max_delta_ratio}")
    print(f"[*] P2: gamma=learned, gate_only, lr_gate={args.lr_gate}, "
          f"pass_threshold={args.phase1_pass_threshold}%")

    run(args, device)


if __name__ == "__main__":
    main()
