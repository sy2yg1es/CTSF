"""Validate binary-vs-continuous channel gating across datasets.

This is a structural diagnostic. Existing Prompt-Z checkpoints are used only
to supply a non-zero delta direction; their end-to-end scores are not treated
as fair model comparisons or as Phase-1 delta results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.residual_tracker import ResidualTracker
from scripts.eval_test_oracle import (
    build_backbone,
    build_prompt_z,
    compute_oracle_supervision,
    pack_stats,
)
from scripts.experiment_gate_learnability import (
    _fixed_delta_forward,
    _frozen_forward,
    _tracker_step,
    get_dataset,
    make_loader,
)


DEFAULT_CASES = [
    "ETTh1:1:2000",
    "ETTh2:1:2000",
    "ETTm2:1:2000",
    "WTH:1:2000",
    "ECL:1:500",
    "Traffic:1:300",
    "ETTh1:24:1500",
    "ETTm2:24:1500",
    "WTH:24:1500",
    "ECL:24:400",
    "Traffic:24:250",
]


def parse_case(spec):
    dataset, horizon, windows = spec.split(":")
    return dataset, int(horizon), int(windows)


def make_args(base, dataset_name, horizon):
    data_path = dataset_name + ".csv"
    import pandas as pd

    frame = pd.read_csv(os.path.join(base.root_path, data_path), nrows=2)
    enc_in = len([c for c in frame.columns if c.lower() != "date"])
    return SimpleNamespace(
        root_path=base.root_path,
        data_path=data_path,
        features="M",
        seq_len=base.seq_len,
        forecast_H=horizon,
        enc_in=enc_in,
        train_ratio=base.train_ratio,
        val_ratio=base.val_ratio,
        num_workers=base.num_workers,
        backbone="patchtst",
        D_model=base.D_model,
        d_ff=base.d_ff,
        e_layers=base.e_layers,
        pretrained_weights=os.path.join(
            base.weights_dir, f"patchtst_pretrained_{dataset_name}_H{horizon}.pth"
        ),
        d_drift=base.d_drift,
        rank=base.rank,
        max_delta_ratio=base.max_delta_ratio,
        residual_window_K=base.residual_window_K,
        target_margin_pct=0.0,
        log_interval=base.log_interval,
    )


@torch.no_grad()
def evaluate_case(base, dataset_name, horizon, requested_windows, device):
    args = make_args(base, dataset_name, horizon)
    checkpoint = os.path.join(
        base.promptz_dir,
        f"prompt_z_{dataset_name}_H{horizon}_pzB_v1_final.pth",
    )
    for path in (args.pretrained_weights, checkpoint):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    dataset = get_dataset(args)
    train_end = dataset.train_size
    val_start = dataset.val_start
    val_end = min(dataset.test_start, val_start + requested_windows)
    warm_start = max(0, train_end - base.warmup_steps)

    adapter = build_backbone(args, device)
    args.hidden_layout = adapter.hidden_layout
    prompt_z = build_prompt_z(args, device)
    prompt_z.load_state_dict(torch.load(checkpoint, map_location=device))
    prompt_z.eval()

    tracker = ResidualTracker(args.enc_in, args.residual_window_K).to(device)
    tracker.reset()
    residual_cache = deque()
    for X, Y in make_loader(dataset, warm_start, train_end, args):
        X, Y = X.to(device), Y.to(device)
        frozen = _frozen_forward(adapter, X)
        _tracker_step(tracker, residual_cache, frozen, Y, horizon)

    totals = dict(frozen=0.0, fixed=0.0, binary=0.0, continuous=0.0)
    helpful = mid = total_decisions = 0
    gamma_sum = 0.0
    n = 0
    print(f"[{dataset_name}-H{horizon}] val [{val_start},{val_end})")
    for X, Y in make_loader(dataset, val_start, val_end, args):
        X, Y = X.to(device), Y.to(device)
        stats = pack_stats(tracker, device)
        frozen, fixed, _ = _fixed_delta_forward(adapter, prompt_z, X, stats)
        sup = compute_oracle_supervision(frozen, fixed, fixed, Y)
        totals["frozen"] += torch.nn.functional.mse_loss(frozen, Y).item()
        totals["fixed"] += torch.nn.functional.mse_loss(fixed, Y).item()
        totals["binary"] += sup["mse_oracle_channel"].item()
        totals["continuous"] += sup["mse_oracle_continuous_channel"].item()
        binary_gamma = sup["oracle_gamma_channel"]
        continuous_gamma = sup["oracle_gamma_continuous_channel"]
        helpful += int(binary_gamma.sum().item())
        mid += int(
            ((continuous_gamma > 0.05) & (continuous_gamma < 0.95)).sum().item()
        )
        total_decisions += continuous_gamma.numel()
        gamma_sum += continuous_gamma.sum().item()
        _tracker_step(tracker, residual_cache, frozen, Y, horizon)
        n += 1
        if n % base.log_interval == 0:
            print(f"[{dataset_name}-H{horizon}] {n}/{val_end-val_start}")

    avg = {k: v / max(n, 1) for k, v in totals.items()}
    continuous_gain = avg["frozen"] - avg["continuous"]
    binary_gain = avg["frozen"] - avg["binary"]
    continuous_routing_gain = avg["fixed"] - avg["continuous"]
    binary_routing_gain = avg["fixed"] - avg["binary"]

    result = {
        "case": f"{dataset_name}_H{horizon}",
        "dataset": dataset_name,
        "horizon": horizon,
        "n_windows": n,
        "n_channels": args.enc_in,
        "checkpoint": checkpoint,
        "frozen_mse": avg["frozen"],
        "fixed_gamma1_mse": avg["fixed"],
        "binary_oracle_mse": avg["binary"],
        "continuous_oracle_mse": avg["continuous"],
        "binary_excess_mse": avg["binary"] - avg["continuous"],
        "binary_capture_total": binary_gain / max(continuous_gain, 1e-12),
        "binary_capture_routing":
            binary_routing_gain / max(continuous_routing_gain, 1e-12),
        "binary_helpful_frac": helpful / max(total_decisions, 1),
        "continuous_gamma_mean": gamma_sum / max(total_decisions, 1),
        "continuous_gamma_mid_frac": mid / max(total_decisions, 1),
        "diagnostic_only": True,
    }
    print(
        f"[{result['case']}] capture_total={result['binary_capture_total']*100:.2f}% "
        f"capture_routing={result['binary_capture_routing']*100:.2f}% "
        f"mid={result['continuous_gamma_mid_frac']*100:.2f}%"
    )
    del prompt_z, adapter, tracker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser("Cross-dataset binary oracle structural diagnostic")
    p.add_argument("--root_path", default="./dataset")
    p.add_argument("--weights_dir", default="./weights")
    p.add_argument("--promptz_dir", default="./weights/prompt_z")
    p.add_argument("--case", action="append", dest="cases",
                   help="DATASET:HORIZON:VAL_WINDOWS; repeatable")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--D_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--e_layers", type=int, default=3)
    p.add_argument("--d_drift", type=int, default=64)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--max_delta_ratio", type=float, default=0.05)
    p.add_argument("--residual_window_K", type=int, default=24)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--out", default="logs/prompt_z/binary_oracle_multidata.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = args.cases or DEFAULT_CASES
    results = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def save_progress():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "note": (
                        "Structural diagnostic using existing joint-training checkpoints; "
                        "not a fair model-performance comparison."
                    ),
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    for spec in cases:
        dataset_name, horizon, windows = parse_case(spec)
        try:
            results.append(
                evaluate_case(args, dataset_name, horizon, windows, device)
            )
        except Exception as exc:
            print(f"[FAILED] {spec}: {type(exc).__name__}: {exc}")
            results.append({"case_spec": spec, "error": f"{type(exc).__name__}: {exc}"})
        save_progress()
    print(f"[*] saved {args.out}")


if __name__ == "__main__":
    main()
