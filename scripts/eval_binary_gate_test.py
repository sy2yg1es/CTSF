"""One-shot TEST evaluation for the finalized binary Prompt-Z channel gate.

This entry point never computes oracle targets and never selects a mode,
threshold, checkpoint, or hyperparameter from TEST labels.  The train-only
safety decision embedded in the gate checkpoint is treated as immutable.
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.residual_tracker import ResidualTracker
from data_provider.data_loader import data_provider
from models.binary_channel_gate import (
    build_causal_gate_features,
    gate_from_checkpoint,
    validate_gate_checkpoint,
)
from scripts.eval_test_oracle import build_backbone, build_prompt_z, pack_stats


def _dataset(args):
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
    return dataset


def _loader(dataset, start, end, args):
    return DataLoader(
        Subset(dataset, range(start, end)),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def _tracker_update(tracker, pending, frozen, target, horizon):
    pending.append((frozen.detach(), target.detach()))
    if len(pending) > horizon:
        old_prediction, old_target = pending.popleft()
        tracker.update(old_prediction, old_target)
    else:
        tracker.step_no_update()


@torch.no_grad()
def _frozen_prediction(adapter, inputs):
    hidden, means, stdev = adapter.encode_until_hook(inputs)
    hidden = hidden.detach()
    return adapter.decode_from_hook(hidden, means, stdev)


@torch.no_grad()
def _fixed_delta_outputs(adapter, prompt_z, inputs, stats):
    hidden, means, stdev = adapter.encode_until_hook(inputs)
    hidden = hidden.detach()
    frozen = adapter.decode_from_hook(hidden, means, stdev)
    drift_state = prompt_z.drift_encoder(prompt_z._hidden_summary(hidden), stats)
    if prompt_z.hidden_layout == "BCDP":
        delta = prompt_z.low_rank_mod(hidden.permute(0, 1, 3, 2), drift_state)
        delta = delta.permute(0, 1, 3, 2)
    else:
        delta = prompt_z.low_rank_mod(hidden, drift_state)
    delta = prompt_z._ratio_clamp(delta, hidden)
    fixed = adapter.decode_from_hook(hidden + delta, means, stdev)
    hidden_norm = hidden.flatten(2).norm(dim=-1).clamp(min=1e-8)
    delta_ratio = (delta.flatten(2).norm(dim=-1) / hidden_norm).mean()
    return frozen, fixed, drift_state, delta_ratio


def _assert_checkpoint_matches(args, checkpoint):
    config = checkpoint["config"]
    expected = {
        "data_path": Path(args.data_path).name,
        "forecast_H": args.forecast_H,
        "backbone": args.backbone,
        "D_model": args.D_model,
        "d_ff": args.d_ff,
        "e_layers": args.e_layers,
        "d_drift": args.d_drift,
        "rank": args.rank,
        "residual_window_K": args.residual_window_K,
    }
    mismatches = []
    for key, value in expected.items():
        saved = config.get(key)
        if key == "data_path" and saved is not None:
            saved = Path(saved).name
        if saved != value:
            mismatches.append(f"{key}: checkpoint={saved!r}, requested={value!r}")
    saved_ratio = float(config.get("max_delta_ratio", float("nan")))
    if abs(saved_ratio - args.max_delta_ratio) > 1e-12:
        mismatches.append(
            f"max_delta_ratio: checkpoint={saved_ratio}, requested={args.max_delta_ratio}"
        )
    artifact_paths = {
        "pretrained_weights": args.pretrained_weights,
        "p1_ckpt": args.p1_ckpt,
    }
    for key, requested_path in artifact_paths.items():
        saved_path = config.get(key)
        if saved_path is None or Path(saved_path).name != Path(requested_path).name:
            mismatches.append(
                f"{key}: checkpoint={saved_path!r}, requested={requested_path!r}"
            )

    fixed_protocol = {
        "validation_protocol": "fixed_zero_blocked",
        "feature_mode": "causal_augmented",
        "target_margin_pct": 0.0,
        "probe_hidden": 64,
        "train_selection_fraction": 0.2,
        "safe_min_improvement_pct": 0.2,
        "safe_min_positive_block_frac": 0.75,
    }
    for key, required in fixed_protocol.items():
        saved = config.get(key)
        if saved != required:
            mismatches.append(f"{key}: checkpoint={saved!r}, required={required!r}")
    expected_history = {1: 2000, 12: 8000, 24: 2000, 48: 2000}
    required_steps = expected_history.get(args.forecast_H)
    if required_steps is not None and config.get("train_steps") != required_steps:
        mismatches.append(
            f"train_steps: checkpoint={config.get('train_steps')!r}, "
            f"required={required_steps!r} for H={args.forecast_H}"
        )
    if mismatches:
        raise ValueError("Gate checkpoint/config mismatch:\n  " + "\n  ".join(mismatches))


@torch.no_grad()
def evaluate(args, device):
    checkpoint = torch.load(args.gate_ckpt, map_location="cpu", weights_only=False)
    validate_gate_checkpoint(checkpoint)
    _assert_checkpoint_matches(args, checkpoint)
    mode = checkpoint["selected_mode"]
    feature_mode = checkpoint["config"]["feature_mode"]

    adapter = build_backbone(args, device)
    args.hidden_layout = adapter.hidden_layout
    if checkpoint["config"].get("hidden_layout") != adapter.hidden_layout:
        raise ValueError(
            "Gate checkpoint hidden layout does not match the loaded backbone: "
            f"{checkpoint['config'].get('hidden_layout')!r} vs {adapter.hidden_layout!r}"
        )
    prompt_z = build_prompt_z(args, device)
    prompt_z.load_state_dict(
        torch.load(args.p1_ckpt, map_location=device, weights_only=True), strict=True
    )
    prompt_z.eval()
    for parameter in prompt_z.parameters():
        parameter.requires_grad = False

    gate = None
    if mode == "learned_regressor":
        gate = gate_from_checkpoint(checkpoint, device)

    dataset = _dataset(args)
    warmup_loader = _loader(dataset, dataset.val_start, dataset.test_start, args)
    test_loader = _loader(dataset, dataset.test_start, dataset.n_windows, args)
    tracker = ResidualTracker(args.enc_in, args.residual_window_K).to(device)
    tracker.reset()
    pending = deque()

    warmup_start = time.perf_counter()
    for inputs, target in warmup_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        frozen = _frozen_prediction(adapter, inputs)
        _tracker_update(tracker, pending, frozen, target, args.forecast_H)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start

    frozen_sse = torch.zeros((), device=device, dtype=torch.float64)
    selected_sse = torch.zeros((), device=device, dtype=torch.float64)
    gate_on = torch.zeros((), device=device, dtype=torch.float64)
    gate_total = 0
    delta_ratio_sum = torch.zeros((), device=device, dtype=torch.float64)
    value_count = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_start = time.perf_counter()
    for inputs, target in test_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        stats = pack_stats(tracker, device)

        if mode == "frozen":
            frozen = _frozen_prediction(adapter, inputs)
            selected = frozen
            decisions = torch.zeros(
                (inputs.shape[0], args.enc_in), device=device, dtype=torch.bool
            )
            delta_ratio = torch.zeros((), device=device)
        else:
            frozen, fixed, drift_state, delta_ratio = _fixed_delta_outputs(
                adapter, prompt_z, inputs, stats
            )
            if mode == "fixed":
                decisions = torch.ones(
                    (inputs.shape[0], args.enc_in), device=device, dtype=torch.bool
                )
            else:
                features = build_causal_gate_features(
                    drift_state, stats, frozen, fixed, feature_mode
                )
                decisions = gate.decisions(features)
            gate_view = decisions.unsqueeze(1)
            # This is the same channel-separable interpolation used by the
            # validation protocol, avoiding a redundant third decoder pass.
            selected = torch.where(gate_view, fixed, frozen)

        frozen_sse += (frozen - target).to(torch.float64).pow(2).sum()
        selected_sse += (selected - target).to(torch.float64).pow(2).sum()
        gate_on += decisions.to(torch.float64).sum()
        gate_total += decisions.numel()
        delta_ratio_sum += delta_ratio.to(torch.float64)
        value_count += target.numel()
        _tracker_update(tracker, pending, frozen, target, args.forecast_H)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_start

    frozen_mse = (frozen_sse / max(value_count, 1)).item()
    selected_mse = (selected_sse / max(value_count, 1)).item()
    n_windows = len(test_loader)
    return {
        "protocol": checkpoint["protocol_version"],
        "selected_mode": mode,
        "decision_threshold": 0.0,
        "n_test_windows": n_windows,
        "frozen_mse": frozen_mse,
        "gate_prompt_z_mse": selected_mse,
        "relative_improvement_pct": (
            (frozen_mse - selected_mse) / max(frozen_mse, 1e-12) * 100.0
        ),
        "gate_usage_ratio": (gate_on / max(gate_total, 1)).item(),
        "effective_correction_ratio": (
            delta_ratio_sum / max(n_windows, 1)
        ).item(),
        "runtime": {
            "warmup_seconds": warmup_seconds,
            "test_seconds": evaluation_seconds,
            "milliseconds_per_test_window": (
                evaluation_seconds * 1000.0 / max(n_windows, 1)
            ),
        },
        "artifacts": {
            "backbone": args.pretrained_weights,
            "prompt_z_p1": args.p1_ckpt,
            "binary_gate": args.gate_ckpt,
        },
    }


def main():
    parser = argparse.ArgumentParser("Final one-shot binary Gate TEST")
    parser.add_argument("--root_path", default="./dataset")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features", default="M")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--forecast_H", type=int, required=True)
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--backbone", default="patchtst", choices=["patchtst", "itransformer"]
    )
    parser.add_argument("--D_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--pretrained_weights", required=True)
    parser.add_argument("--p1_ckpt", required=True)
    parser.add_argument("--gate_ckpt", required=True)
    parser.add_argument("--d_drift", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max_delta_ratio", type=float, default=0.02)
    parser.add_argument("--residual_window_K", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", default="logs/prompt_z/test")
    parser.add_argument("--experiment_tag", default="")
    args = parser.parse_args()

    if args.enc_in is None:
        import pandas as pd

        frame = pd.read_csv(os.path.join(args.root_path, args.data_path))
        args.enc_in = len([c for c in frame.columns if c.lower() != "date"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate(args, device)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = (
        f"binary_gate_test_{Path(args.data_path).stem}_H{args.forecast_H}_"
        f"{args.backbone}"
    )
    if args.experiment_tag:
        stem += f"_{args.experiment_tag}"
    output_path = os.path.join(args.out_dir, stem + ".json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("\nBinary Gate TEST summary")
    print(f"  selected mode       : {result['selected_mode']}")
    print(f"  Frozen MSE          : {result['frozen_mse']:.8f}")
    print(f"  Gate/Prompt-Z MSE   : {result['gate_prompt_z_mse']:.8f}")
    print(f"  relative improvement: {result['relative_improvement_pct']:+.4f}%")
    print(f"  gate usage ratio    : {result['gate_usage_ratio']:.4f}")
    print(
        "  runtime             : "
        f"{result['runtime']['test_seconds']:.2f}s, "
        f"{result['runtime']['milliseconds_per_test_window']:.3f} ms/window"
    )
    print(f"  saved               : {output_path}")


if __name__ == "__main__":
    main()
