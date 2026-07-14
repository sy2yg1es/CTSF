#!/usr/bin/env python3
"""ETTm1/H=1/PatchTST Frozen alignment gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import OnlineBaselineConfig, build_online_baseline
from models.backbone_adapter import PatchTSTAdapter
from scripts.eval_online_baselines import (
    build_backbone,
    build_dataset,
    checkpoint_path,
    configure_exact_cuda_math,
    describe_windows,
    load_checkpoint_strict,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="dataset")
    parser.add_argument("--weight_dir", default="weights")
    parser.add_argument("--output_dir", default="logs/online_baselines/alignment_ETTm1_H1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_windows", type=int, default=None, help="Smoke-test only; omit for the formal alignment gate")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_exact_cuda_math()
    set_seed(2025)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    data_path, horizon = "ETTm1.csv", 1
    dataset = build_dataset(args, data_path, horizon)
    end = len(dataset)
    if args.max_windows is not None:
        end = min(end, dataset.test_start + args.max_windows)
        print("[WARNING] partial smoke test only; not a formal full-test alignment result")
    print("[windows] " + json.dumps(describe_windows(dataset, dataset.test_start, end), sort_keys=True))

    ctsf_model = build_backbone(dataset.data_x.shape[1], horizon, args)
    baseline_model = build_backbone(dataset.data_x.shape[1], horizon, args)
    weight_path = checkpoint_path(args, data_path, horizon)
    load_checkpoint_strict(ctsf_model, weight_path)
    load_checkpoint_strict(baseline_model, weight_path)

    max_parameter_diff = max(
        (a.detach().cpu() - b.detach().cpu()).abs().max().item()
        for a, b in zip(ctsf_model.parameters(), baseline_model.parameters())
    )
    if max_parameter_diff != 0.0:
        raise AssertionError(f"Initial parameters differ: {max_parameter_diff}")
    print(f"[alignment] max_parameter_diff={max_parameter_diff}")

    ctsf_adapter = PatchTSTAdapter(ctsf_model).to(device).eval()
    frozen = build_online_baseline(
        "frozen", baseline_model, OnlineBaselineConfig(online_lr=0.0), device
    )
    print("[alignment] optimizer_state=0 replay_state=0 validation_adaptation=false")

    pred_ctsf, pred_baseline = [], []
    sq_ctsf = sq_baseline = 0.0
    abs_ctsf = abs_baseline = 0.0
    total = 0
    for offset, index in enumerate(range(dataset.test_start, end)):
        x, y = dataset[index]
        x_dev = x.unsqueeze(0).to(device)
        with torch.no_grad():
            a = ctsf_adapter.forward_frozen(x_dev).cpu()
            b = frozen.predict(x.unsqueeze(0)).cpu()
        target = y.unsqueeze(0).double()
        da, db = a.double() - target, b.double() - target
        sq_ctsf += da.square().sum().item()
        sq_baseline += db.square().sum().item()
        abs_ctsf += da.abs().sum().item()
        abs_baseline += db.abs().sum().item()
        total += target.numel()
        pred_ctsf.append(a.numpy())
        pred_baseline.append(b.numpy())

        if offset < 5:
            origin = index + dataset.seq_len
            update_index = index - horizon
            print("[delay-check] " + json.dumps({
                "current_forecast_origin": origin,
                "current_target_range": [origin, origin + horizon],
                "update_sample_input_range": None if update_index < dataset.test_start else [update_index, update_index + dataset.seq_len],
                "update_sample_target_range": None if update_index < dataset.test_start else [update_index + dataset.seq_len, update_index + dataset.seq_len + horizon],
                "latest_observable_timestamp": origin - 1,
            }, sort_keys=True))

    pred_ctsf_np = np.concatenate(pred_ctsf, axis=0)
    pred_baseline_np = np.concatenate(pred_baseline, axis=0)
    prediction_diff = np.abs(pred_ctsf_np.astype(np.float64) - pred_baseline_np.astype(np.float64))
    metrics = {
        "ctsf_frozen_MSE": sq_ctsf / total,
        "adapter_frozen_MSE": sq_baseline / total,
        "ctsf_frozen_MAE": abs_ctsf / total,
        "adapter_frozen_MAE": abs_baseline / total,
        "mse_abs_diff": abs(sq_ctsf - sq_baseline) / total,
        "max_abs_prediction_diff": float(prediction_diff.max()),
        "mean_abs_prediction_diff": float(prediction_diff.mean()),
        "num_test_windows": end - dataset.test_start,
        "num_evaluated_elements": total,
        "max_parameter_diff": max_parameter_diff,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pred_ctsf.npy", pred_ctsf_np)
    np.save(output_dir / "pred_baseline_adapter.npy", pred_baseline_np)
    with (output_dir / "alignment.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("[alignment-result] " + json.dumps(metrics, sort_keys=True))

    if metrics["mse_abs_diff"] >= 1e-6:
        raise SystemExit("FAIL: Frozen MSE mismatch")
    if metrics["max_abs_prediction_diff"] >= 1e-5:
        raise SystemExit("FAIL: max prediction mismatch")
    if metrics["mean_abs_prediction_diff"] >= 1e-6:
        raise SystemExit("FAIL: mean prediction mismatch")
    print("ALIGNMENT PASS")


if __name__ == "__main__":
    main()
