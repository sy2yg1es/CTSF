"""
validate_itransformer.py — iTransformer Frozen Backbone Validation
==================================================================

对所有数据集 × H ∈ {1, 12, 24, 48} 跑 frozen baseline 评估，
汇总 MSE / MAE 输出到 stdout 和 summary.tsv。

用法:
    python scripts/validate_itransformer.py
    python scripts/validate_itransformer.py --horizons 1 12
    python scripts/validate_itransformer.py --datasets ECL.csv ETTh1.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ── 确保项目根在 sys.path ──────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_loader import data_provider
from data_provider.streaming_env import StreamingEnvironment
from models.backbone_adapter import iTransformerAdapter
from models.prompt_z import PromptZModulator
from models.prompt_z_framework import PromptZTSF
from core.residual_tracker import ResidualTracker


# ── 默认配置 ──────────────────────────────────────────────────────────────
DEFAULT_DATASETS = [
    "ECL.csv", "Traffic.csv", "ETTh1.csv", "WTH.csv",
    "ETTm2.csv", "ETTm1.csv", "ETTh2.csv",
]
DEFAULT_HORIZONS = [1, 12, 24, 48]

BACKBONE = "itransformer"
SEQ_LEN  = 96
D_MODEL  = 512
D_FF     = 512
E_LAYERS = 3
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.1
NUM_WORKERS = 4
D_DRIFT = 64
RANK = 8
GAMMA_INIT_BIAS = -3.0
MASK_INIT_BIAS = -1.5
MAX_DELTA_RATIO = 0.05
RESIDUAL_WINDOW_K = 24

EPOCHS = 3
BATCH_SIZE = 1
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_DELTA = 2e-4
LAMBDA_MASK = 1e-4
LAMBDA_NOOP = 0.005
NOOP_EPSILON = 1e-4
TARGET_MASK_RATIO = 0.10
REG_WARMUP_STEPS = 2000
NOOP_WARMUP_STEPS = 6000
NOOP_RAMP_STEPS = 2000
NOOP_MIN_EFF_RATIO = 1e-4
GAMMA_FLOOR = 0.1
GAMMA_FLOOR_STEPS = 8000
MASK_FLOOR = 0.05
MASK_FLOOR_STEPS = 12000
DELAYED_RESIDUAL = True
TRAIN_LOG_INTERVAL = 200
LOG_INTERVAL = 500          # 每隔多少 step 打印一次滚动 MSE


# ── 工具函数 ──────────────────────────────────────────────────────────────
def auto_detect_enc_in(root_path: str, data_path: str) -> int:
    import pandas as pd
    df = pd.read_csv(os.path.join(root_path, data_path), nrows=5)
    return len([c for c in df.columns if c.lower() != "date"])


def find_backbone_weights(ds: str, h: int, weight_dir: str = "weights") -> str | None:
    candidates = [
        f"{weight_dir}/{BACKBONE}_pretrained_{ds}_H{h}_GA1.pth",
        f"{weight_dir}/{BACKBONE}_pretrained_{ds}_H{h}.pth",
        f"{weight_dir}/backbone/{BACKBONE}_pretrained_{ds}_H{h}_GA1.pth",
        f"{weight_dir}/backbone/{BACKBONE}_pretrained_{ds}_H{h}.pth",
        f"{weight_dir}/pretrained/{BACKBONE}_pretrained_{ds}_H{h}_GA1.pth",
        f"{weight_dir}/pretrained/{BACKBONE}_pretrained_{ds}_H{h}.pth",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def build_backbone(enc_in: int, forecast_H: int, device: torch.device) -> iTransformerAdapter:
    from models.backbones.iTransformer import Model as iTransformer

    class Cfg:
        pass

    cfg = Cfg()
    cfg.task_name        = "long_term_forecast"
    cfg.seq_len          = SEQ_LEN
    cfg.pred_len         = forecast_H
    cfg.d_model          = D_MODEL
    cfg.d_ff             = D_FF
    cfg.n_heads          = 8
    cfg.e_layers         = E_LAYERS
    cfg.dropout          = 0.1
    cfg.activation       = "gelu"
    cfg.factor           = 1
    cfg.enc_in           = enc_in
    cfg.output_attention = False
    cfg.embed            = "timeF"
    cfg.freq             = "h"

    model   = iTransformer(cfg).to(device)
    adapter = iTransformerAdapter(model).to(device)
    return adapter


def load_weights(adapter: iTransformerAdapter, weight_path: str, device: torch.device) -> None:
    state = torch.load(weight_path, map_location=device, weights_only=False)
    prefix_candidates = ["backbone_adapter.backbone.", "model.", ""]
    loaded = False
    for prefix in prefix_candidates:
        if prefix and not any(k.startswith(prefix) for k in state.keys()):
            continue
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)} if prefix else state
        try:
            adapter.backbone.load_state_dict(sub, strict=False)
            loaded = True
            break
        except RuntimeError:
            continue
    if not loaded:
        adapter.backbone.load_state_dict(state, strict=False)
    print(f"    [*] Loaded backbone from {weight_path}")


def build_full_model(
    adapter: iTransformerAdapter,
    enc_in: int,
    device: torch.device,
) -> PromptZTSF:
    prompt_z = PromptZModulator(
        d_model=D_MODEL,
        hidden_layout=adapter.hidden_layout,
        d_drift=D_DRIFT,
        rank=RANK,
        gamma_init_bias=GAMMA_INIT_BIAS,
        mask_init_bias=MASK_INIT_BIAS,
        max_delta_ratio=MAX_DELTA_RATIO,
    ).to(device)

    residual_tracker = ResidualTracker(
        num_channels=enc_in,
        window_K=RESIDUAL_WINDOW_K,
    ).to(device)

    model = PromptZTSF(adapter, prompt_z, residual_tracker).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    [*] Model: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    return model


def get_dataloader(root_path: str, data_path: str, forecast_H: int):
    class DPArgs:
        pass
    args = DPArgs()
    args.root_path   = root_path
    args.data_path   = data_path
    args.features    = "M"
    args.seq_len     = SEQ_LEN
    args.pred_len    = forecast_H
    args.target      = "OT"
    args.num_workers = NUM_WORKERS
    args.train_ratio = TRAIN_RATIO
    args.val_ratio   = VAL_RATIO
    dataset, loader = data_provider(args)
    return dataset, loader


def get_train_dataloader(root_path: str, data_path: str, forecast_H: int):
    dataset, _ = get_dataloader(root_path, data_path, forecast_H)
    train_size = dataset.train_size
    subset = Subset(dataset, range(train_size))
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    print(f"    [*] Train subset: {train_size}/{len(dataset)} windows")
    return subset, loader


def train_one_epoch(model, dataloader, optimizer, epoch, device, total_steps_before, forecast_H):
    model.prompt_z.train()
    model.backbone_adapter.eval()

    total_loss = total_forecast = total_noop_active = 0.0
    n_steps = 0
    diag_accum = {k: 0.0 for k in [
        "gamma_mean", "mask_ratio", "mask_mean", "hidden_norm",
        "raw_delta_norm", "applied_delta_norm",
        "raw_delta_to_hidden_ratio", "effective_delta_ratio",
    ]}

    model.residual_tracker.reset()
    residual_cache: deque = deque()

    for step, (X, Y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        optimizer.zero_grad()

        global_step = total_steps_before + step
        gamma_floor_current = (
            GAMMA_FLOOR * (1.0 - global_step / GAMMA_FLOOR_STEPS)
            if GAMMA_FLOOR_STEPS > 0 and global_step < GAMMA_FLOOR_STEPS else 0.0
        )
        mask_floor_current = (
            MASK_FLOOR * (1.0 - global_step / MASK_FLOOR_STEPS)
            if MASK_FLOOR_STEPS > 0 and global_step < MASK_FLOOR_STEPS else 0.0
        )
        reg_scale = (
            min(1.0, max(0.0, (global_step + 1) / REG_WARMUP_STEPS))
            if REG_WARMUP_STEPS > 0 else 1.0
        )
        if global_step < NOOP_WARMUP_STEPS:
            noop_scale = 0.0
        else:
            noop_scale = min(1.0, (global_step - NOOP_WARMUP_STEPS) / max(1, NOOP_RAMP_STEPS))

        Y_hat, Y_frozen, reg_tensors, diagnostics = model.forward_train(
            X,
            gamma_floor=gamma_floor_current,
            mask_floor=mask_floor_current,
        )

        forecast_loss = nn.functional.mse_loss(Y_hat, Y)
        delta_reg = reg_tensors["effective_delta_ratio"]
        mask_mean = reg_tensors["mask_mean"]
        mask_reg = torch.relu(mask_mean - TARGET_MASK_RATIO)

        with torch.no_grad():
            mse_frozen = nn.functional.mse_loss(Y_frozen, Y)

        noop_penalty = forecast_loss.new_zeros(())
        noop_active = 0.0
        eff_ratio = reg_tensors["effective_delta_ratio"].detach().item()
        if (global_step >= NOOP_WARMUP_STEPS
                and eff_ratio >= NOOP_MIN_EFF_RATIO
                and forecast_loss.item() >= mse_frozen.item() - NOOP_EPSILON):
            noop_penalty = reg_tensors["gamma_mean"]
            noop_active = 1.0

        loss = (forecast_loss
                + reg_scale * LAMBDA_DELTA * delta_reg
                + reg_scale * LAMBDA_MASK * mask_reg
                + noop_scale * LAMBDA_NOOP * noop_penalty)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.prompt_z.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            if DELAYED_RESIDUAL:
                residual_cache.append((Y_frozen.detach(), Y.detach()))
                if len(residual_cache) > forecast_H:
                    old_pred, old_true = residual_cache.popleft()
                    model.residual_tracker.update(old_pred, old_true)
                else:
                    model.residual_tracker.step_no_update()
            else:
                model.residual_tracker.update(Y_frozen.detach(), Y.detach())

            residual_cache_len = len(residual_cache)
            tracker_count = int(model.residual_tracker._count.item())
            residual_tracker_warmed = int(tracker_count >= model.residual_tracker.K)

        total_loss += loss.item()
        total_forecast += forecast_loss.item()
        total_noop_active += noop_active
        n_steps += 1
        for k in diag_accum:
            diag_accum[k] += diagnostics.get(k, 0.0)

        if step % TRAIN_LOG_INTERVAL == 0:
            print(
                f"  [Epoch {epoch} Step {step}] "
                f"loss={loss.item():.6f} forecast={forecast_loss.item():.6f} "
                f"frozen={mse_frozen.item():.6f} "
                f"gamma={diagnostics.get('gamma_mean', 0):.6f} "
                f"mask_mean={diagnostics.get('mask_mean', 0):.6f} "
                f"mask_ratio={diagnostics.get('mask_ratio', 0):.4f} "
                f"reg_scale={reg_scale:.3f} noop={noop_active:.0f} "
                f"noop_scale={noop_scale:.3f} "
                f"gamma_floor={gamma_floor_current:.4f} "
                f"residual_cache_len={residual_cache_len} "
                f"residual_tracker_warmed={residual_tracker_warmed} "
                f"raw_d/h={diagnostics.get('raw_delta_to_hidden_ratio', 0):.6f} "
                f"eff_d/h={diagnostics.get('effective_delta_ratio', 0):.6f}"
            )

    if n_steps == 0:
        return {}
    return {
        "loss": total_loss / n_steps,
        "forecast": total_forecast / n_steps,
        "noop_rate": total_noop_active / n_steps,
        **{k: v / n_steps for k, v in diag_accum.items()},
    }


def train_prompt_z(
    model: PromptZTSF,
    root_path: str,
    data_path: str,
    forecast_H: int,
    save_path: str,
    device: torch.device,
    force: bool = False,
) -> str:
    if os.path.isfile(save_path) and not force:
        print(f"    [*] Skip training, weights exist: {save_path}")
        state = torch.load(save_path, map_location=device, weights_only=True)
        model.prompt_z.load_state_dict(state)
        return save_path

    _, dataloader = get_train_dataloader(root_path, data_path, forecast_H)
    optimizer = torch.optim.AdamW(
        model.prompt_z.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )

    epoch_steps = len(dataloader)
    for epoch in range(EPOCHS):
        t0 = time.time()
        metrics = train_one_epoch(
            model, dataloader, optimizer, epoch, device,
            total_steps_before=epoch * epoch_steps,
            forecast_H=forecast_H,
        )
        elapsed = time.time() - t0
        print(
            f"    [Epoch {epoch}] {elapsed:.1f}s | "
            f"loss={metrics.get('loss', 0):.6f} | "
            f"forecast={metrics.get('forecast', 0):.6f} | "
            f"gamma={metrics.get('gamma_mean', 0):.4f} | "
            f"mask_ratio={metrics.get('mask_ratio', 0):.4f} | "
            f"eff_d/h={metrics.get('effective_delta_ratio', 0):.6f}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.prompt_z.state_dict(), save_path)
    print(f"    [*] Prompt-Z saved to {save_path}")
    return save_path


def run_streaming_eval_splits(
    model: PromptZTSF,
    root_path: str,
    data_path: str,
    forecast_H: int,
    device: torch.device,
    mode: str = "frozen",
    calibration_lr: float = 1e-4,
    label: str = "",
    splits: dict[str, tuple[int, int | None]] | None = None,
) -> dict:
    dataset, base_loader = get_dataloader(root_path, data_path, forecast_H)
    streaming_loader = StreamingEnvironment(base_loader, forecast_H=forecast_H)
    n_total = len(dataset)
    if splits is None:
        splits = {
            "val": (dataset.val_start, dataset.test_start),
            "test": (dataset.test_start, n_total),
        }

    split_desc = ", ".join(
        f"{name}=[{start},{end if end is not None else n_total})"
        for name, (start, end) in splits.items()
    )
    print(f"    [*] [{label}] mode={mode} | total={n_total} | {split_desc} | "
          f"raw_train_end={dataset.raw_train_end} | raw_val_end={dataset.raw_val_end}")

    model.eval()
    model.residual_tracker.reset()

    accum = {
        name: {
            "mse_sum": torch.zeros(1, device=device),
            "mae_sum": torch.zeros(1, device=device),
            "n": 0,
        }
        for name in splits
    }
    y_hat_cache: deque = deque()
    x_cache: deque = deque()
    calibration_start = min(start for start, _end in splits.values())

    calib_optimizer = None
    if mode == "mode1":
        gate_params = model.prompt_z.get_gate_params()
        calib_optimizer = torch.optim.SGD(gate_params, lr=calibration_lr)

    t0 = time.time()
    for t, (X_t, Y_arrived) in enumerate(streaming_loader):
        X_t = X_t.to(device, non_blocking=True)

        with torch.no_grad():
            if mode == "frozen":
                Y_hat = model.forward_frozen(X_t)
            else:
                Y_hat, _hidden, _diag = model(X_t)

        y_hat_cache.append((t, Y_hat.detach().cpu()))
        x_cache.append(X_t.detach().cpu())

        if Y_arrived is None:
            model.residual_tracker.step_no_update()
            continue

        Y_arrived = Y_arrived.to(device, non_blocking=True)
        pred_idx, Y_hat_cached_cpu = y_hat_cache.popleft()
        Y_hat_cached = Y_hat_cached_cpu.to(device, non_blocking=True)
        X_cached = x_cache.popleft()

        with torch.no_grad():
            diff = Y_hat_cached - Y_arrived
            step_mse = diff.pow(2).mean()
            step_mae = diff.abs().mean()

        for name, (start, end) in splits.items():
            end = n_total if end is None else end
            if start <= pred_idx < end:
                accum[name]["mse_sum"] += step_mse
                accum[name]["mae_sum"] += step_mae
                accum[name]["n"] += 1

        if mode == "mode1" and pred_idx >= calibration_start and calib_optimizer is not None:
            calib_stats = model.residual_tracker.get_stats()
            calib_stats_tensor = model._pack_stats(calib_stats, device)
        else:
            calib_stats_tensor = None

        model.residual_tracker.update(Y_hat_cached, Y_arrived)

        if calib_stats_tensor is not None:
            X_dev = X_cached.to(device, non_blocking=True)
            for p in model.prompt_z.parameters():
                p.requires_grad_(False)
            for p in model.prompt_z.get_gate_params():
                p.requires_grad_(True)
            calib_optimizer.zero_grad()
            with torch.no_grad():
                hc, mc, sc = model.backbone_adapter.encode_until_hook(X_dev)
            hc = hc.detach()
            mc = mc.detach()
            sc = sc.detach()
            hm, _, _ = model.prompt_z(hc, calib_stats_tensor)
            calib_loss = nn.functional.mse_loss(
                model.backbone_adapter.decode_from_hook(hm, mc, sc), Y_arrived
            )
            calib_loss.backward()
            calib_optimizer.step()
            for p in model.prompt_z.parameters():
                p.requires_grad_(False)

        if t % LOG_INTERVAL == 0:
            pieces = []
            for name, state in accum.items():
                if state["n"] > 0:
                    cur_mse = (state["mse_sum"] / state["n"]).item()
                    pieces.append(f"{name}_MSE={cur_mse:.6f} n={state['n']}")
            if pieces:
                print(f"    [Step {t}] pred_idx={pred_idx} " + " | ".join(pieces))

    elapsed = time.time() - t0
    results = {}
    for name, state in accum.items():
        n_aligned = state["n"]
        if n_aligned == 0:
            print(f"    [!] [{label}:{name}] No aligned steps!")
            results[name] = {
                "MSE": float("inf"),
                "MAE": float("inf"),
                "RMSE": float("inf"),
                "n_aligned": 0,
            }
            continue

        final_mse = (state["mse_sum"] / n_aligned).item()
        final_mae = (state["mae_sum"] / n_aligned).item()
        final_rmse = final_mse ** 0.5
        print(f"    [*] [{label}:{name}] n={n_aligned} | MSE={final_mse:.6f} | "
              f"MAE={final_mae:.6f} | RMSE={final_rmse:.6f} | elapsed={elapsed:.1f}s")
        results[name] = {
            "MSE": final_mse,
            "MAE": final_mae,
            "RMSE": final_rmse,
            "n_aligned": n_aligned,
        }
    return results


def run_frozen_eval(
    adapter: iTransformerAdapter,
    root_path: str,
    data_path: str,
    forecast_H: int,
    device: torch.device,
) -> dict:
    """跑 frozen baseline streaming 评估，返回 {MSE, MAE, n_aligned}。"""
    dataset, base_loader = get_dataloader(root_path, data_path, forecast_H)
    streaming_loader = StreamingEnvironment(base_loader, forecast_H=forecast_H)

    test_start = dataset.test_start
    N_total    = len(dataset)

    print(f"    [*] total_windows={N_total} | test_start={test_start} | "
          f"raw_train_end={dataset.raw_train_end} | raw_val_end={dataset.raw_val_end}")

    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    mse_sum = torch.zeros(1, device=device)
    mae_sum = torch.zeros(1, device=device)
    n_aligned = 0
    y_hat_cache: deque = deque()

    t0 = time.time()
    for t, (X_t, Y_arrived) in enumerate(streaming_loader):
        X_t = X_t.to(device, non_blocking=True)

        with torch.no_grad():
            Y_hat = adapter(X_t)

        y_hat_cache.append(Y_hat.detach().cpu())

        if Y_arrived is None:
            continue

        Y_arrived = Y_arrived.to(device, non_blocking=True)
        Y_hat_cached = y_hat_cache.popleft().to(device, non_blocking=True)

        if t >= test_start:
            diff = Y_hat_cached - Y_arrived
            mse_sum += diff.pow(2).mean()
            mae_sum += diff.abs().mean()
            n_aligned += 1

        if t % LOG_INTERVAL == 0 and n_aligned > 0:
            cur_mse = (mse_sum / n_aligned).item()
            print(f"    [Step {t}] MSE={cur_mse:.6f} n={n_aligned}")

    elapsed = time.time() - t0

    if n_aligned == 0:
        print("    [!] No aligned test steps!")
        return {"MSE": float("inf"), "MAE": float("inf"), "n_aligned": 0}

    final_mse  = (mse_sum / n_aligned).item()
    final_mae  = (mae_sum / n_aligned).item()
    final_rmse = final_mse ** 0.5

    print(f"    [*] n_aligned={n_aligned} | MSE={final_mse:.6f} | "
          f"MAE={final_mae:.6f} | RMSE={final_rmse:.6f} | elapsed={elapsed:.1f}s")
    return {"MSE": final_mse, "MAE": final_mae, "RMSE": final_rmse, "n_aligned": n_aligned}


# ── 主逻辑 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="iTransformer Frozen Validation")
    parser.add_argument("--root_path", type=str, default=None,
                        help="数据根目录 (默认自动检测 ./data 或 ./dataset)")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        metavar="DS", help="数据集文件名列表")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS,
                        metavar="H", help="预测视角列表")
    parser.add_argument("--weight_dir", type=str, default="weights",
                        help="backbone 权重搜索根目录")
    parser.add_argument("--logdir", type=str,
                        default="logs/validate/itransformer_h1_12_24_48",
                        help="summary.tsv 输出目录")
    parser.add_argument("--skip_missing", action="store_true",
                        help="找不到权重时跳过而不是退出")
    parser.add_argument("--pz_dir", type=str, default="weights/prompt_z")
    parser.add_argument("--pz_mode", type=str, default="mode0", choices=["mode0", "mode1"])
    parser.add_argument("--force_train", action="store_true")
    args = parser.parse_args()
    force_train = args.force_train or os.environ.get("FORCE_TRAIN", "0") == "1"

    # 自动检测 root_path
    if args.root_path is None:
        args.root_path = "./data" if os.path.isdir("./data") else "./dataset"

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.pz_dir, exist_ok=True)
    summary_path = os.path.join(args.logdir, "summary.tsv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"  iTransformer Train Prompt-Z -> Frozen Eval -> {args.pz_mode.upper()} Eval")
    print("=" * 65)
    print(f"  root_path  : {args.root_path}")
    print(f"  datasets   : {args.datasets}")
    print(f"  horizons   : {args.horizons}")
    print(f"  weight_dir : {args.weight_dir}")
    print(f"  pz_mode    : {args.pz_mode}")
    print(f"  pz_dir     : {args.pz_dir}")
    print(f"  logdir     : {args.logdir}")
    print(f"  device     : {device}")
    print("=" * 65)

    rows = []
    header = (
        "case\tbackbone\t"
        "frozen_val_MSE\tpz_val_MSE\tval_improve_pct\t"
        "frozen_test_MSE\tpz_test_MSE\ttest_improve_pct\t"
        "frozen_val_MAE\tpz_val_MAE\tfrozen_test_MAE\tpz_test_MAE\t"
        "val_n\ttest_n\tpz_weights\tbackbone_weights"
    )

    for data_path in args.datasets:
        ds = data_path.replace(".csv", "")
        enc_in = auto_detect_enc_in(args.root_path, data_path)

        for h in args.horizons:
            case = f"{ds}_H{h}"
            print(f"\n[CASE] {case}")

            wpath = find_backbone_weights(ds, h, args.weight_dir)
            if wpath is None:
                print(f"  [MISSING] No backbone weights found for {case}")
                if args.skip_missing:
                    rows.append(f"{case}\t{BACKBONE}" + "\tMISSING" * (len(header.split('\t')) - 2))
                    continue
                sys.exit(1)

            adapter = build_backbone(enc_in, h, device)
            load_weights(adapter, wpath, device)
            for p in adapter.parameters():
                p.requires_grad_(False)
            model = build_full_model(adapter, enc_in, device)

            pz_save = os.path.join(
                args.pz_dir,
                f"prompt_z_{BACKBONE}_{ds}_H{h}_final.pth",
            )
            print(f"\n--- [1/3] Train Prompt-Z ---")
            train_prompt_z(
                model, args.root_path, data_path, h, pz_save, device, force=force_train,
            )

            print(f"\n--- [2/3] Frozen Eval ---")
            frozen_res = run_streaming_eval_splits(
                model, args.root_path, data_path, h, device,
                mode="frozen", label="FROZEN",
            )

            print(f"\n--- [3/3] {args.pz_mode.upper()} Eval ---")
            adapter2 = build_backbone(enc_in, h, device)
            load_weights(adapter2, wpath, device)
            for p in adapter2.parameters():
                p.requires_grad_(False)
            model2 = build_full_model(adapter2, enc_in, device)
            state = torch.load(pz_save, map_location=device, weights_only=True)
            model2.prompt_z.load_state_dict(state)
            print(f"    [*] Loaded PromptZ weights from {pz_save}")

            pz_res = run_streaming_eval_splits(
                model2, args.root_path, data_path, h, device,
                mode=args.pz_mode, label=args.pz_mode.upper(),
            )

            frozen_val = frozen_res["val"]
            frozen_test = frozen_res["test"]
            pz_val = pz_res["val"]
            pz_test = pz_res["test"]

            def improvement_pct(base_mse: float, new_mse: float) -> float:
                if base_mse in (0.0, float("inf")):
                    return float("nan")
                return (base_mse - new_mse) / base_mse * 100.0

            val_improve = improvement_pct(frozen_val["MSE"], pz_val["MSE"])
            test_improve = improvement_pct(frozen_test["MSE"], pz_test["MSE"])

            print(
                f"\n  [COMPARE] {case} "
                f"val: frozen={frozen_val['MSE']:.6f} pz={pz_val['MSE']:.6f} "
                f"improve={val_improve:+.2f}% | "
                f"test: frozen={frozen_test['MSE']:.6f} pz={pz_test['MSE']:.6f} "
                f"improve={test_improve:+.2f}%"
            )

            rows.append(
                f"{case}\t{BACKBONE}\t"
                f"{frozen_val['MSE']:.6f}\t{pz_val['MSE']:.6f}\t{val_improve:+.2f}\t"
                f"{frozen_test['MSE']:.6f}\t{pz_test['MSE']:.6f}\t{test_improve:+.2f}\t"
                f"{frozen_val['MAE']:.6f}\t{pz_val['MAE']:.6f}\t"
                f"{frozen_test['MAE']:.6f}\t{pz_test['MAE']:.6f}\t"
                f"{pz_val['n_aligned']}\t{pz_test['n_aligned']}\t{pz_save}\t{wpath}"
            )

    # 写 summary
    with open(summary_path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(header)
    for row in rows:
        print(row)
    print(f"\n[*] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
