#!/usr/bin/env python3
"""Run CTSF-aligned Frozen/Naive/ER/DERpp baselines.

This entry point owns the single shared stream and evaluator.  Validation is
used only for LR selection; every test run reconstructs the backbone,
optimizer and replay state from scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import OnlineBaselineConfig, build_online_baseline
from data_provider.data_loader import Dataset_Custom
from models.backbones.PatchTST import Model as PatchTST


DEFAULT_DATASETS = ["ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv", "ECL.csv", "Traffic.csv", "WTH.csv"]
DEFAULT_HORIZONS = [1, 12, 24, 48]
DEFAULT_LR_GRID = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_exact_cuda_math() -> None:
    """Use full FP32 math for cross-path alignment checks."""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def dataset_name(data_path: str) -> str:
    return Path(data_path).stem


def weight_dataset_name(data_path: str) -> str:
    name = dataset_name(data_path)
    return "WTH" if name.lower() in {"weather", "wth"} else name


def build_dataset(args, data_path: str, horizon: int) -> Dataset_Custom:
    return Dataset_Custom(
        root_path=args.root_path,
        data_path=data_path,
        seq_len=args.seq_len,
        pred_len=horizon,
        features="M",
        target="OT",
        train_ratio=0.6,
        val_ratio=0.1,
    )


def build_backbone(channels: int, horizon: int, args) -> PatchTST:
    config = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=args.seq_len,
        pred_len=horizon,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        dropout=args.dropout,
        activation="gelu",
        factor=1,
        enc_in=channels,
    )
    return PatchTST(config, patch_len=args.patch_len, stride=args.stride)


def checkpoint_path(args, data_path: str, horizon: int) -> Path:
    return Path(args.weight_dir) / f"patchtst_pretrained_{weight_dataset_name(data_path)}_H{horizon}.pth"


def architecture_config(dataset: Dataset_Custom, horizon: int, args) -> dict:
    """Explicit checkpoint architecture contract recorded with every run."""
    channels = int(dataset.data_x.shape[1])
    return {
        "seq_len": args.seq_len,
        "pred_len": horizon,
        "enc_in": channels,
        "c_out": channels,
        "patch_len": args.patch_len,
        "stride": args.stride,
        "padding_patch": "end",
        "padding": args.stride,
        "e_layers": args.e_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "fc_dropout": 0.0,
        "head_dropout": args.dropout,
        "RevIN": "internal_nonparametric_instance_normalization",
        "affine": False,
        "subtract_last": False,
        "decomposition": False,
        "individual": False,
    }


def load_checkpoint_strict(model: torch.nn.Module, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must be a state_dict mapping: {path}")
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(f"[checkpoint] strict=True missing=0 unexpected=0 path={path}")


def fresh_baseline(method: str, dataset: Dataset_Custom, horizon: int, lr: float, args, device):
    set_seed(args.seed)
    model = build_backbone(dataset.data_x.shape[1], horizon, args)
    load_checkpoint_strict(model, checkpoint_path(args, dataset.data_path, horizon))
    config = OnlineBaselineConfig(
        online_lr=lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        grad_clip=args.grad_clip,
        amp=False,
        update_steps=args.update_steps,
        buffer_size=args.buffer_size,
        replay_batch_size=args.replay_batch_size,
        replay_weight=args.replay_weight,
        distill_weight=args.distill_weight,
        seed=args.seed,
    )
    baseline = build_online_baseline(method, model, config, device)
    optimizer_state = 0 if baseline.optimizer is None else len(baseline.optimizer.state)
    if optimizer_state != 0:
        raise AssertionError("Online optimizer inherited non-empty state")
    print(f"[reset] method={method} optimizer_state={optimizer_state} replay_state=0")
    print("[architecture] " + json.dumps(architecture_config(dataset, horizon, args), sort_keys=True))
    print("[config] " + json.dumps(config.to_log_dict(), sort_keys=True))
    return baseline, config


def describe_windows(dataset: Dataset_Custom, start: int, end: int) -> dict:
    seq_len, horizon = dataset.seq_len, dataset.pred_len
    return {
        "raw_dataset_length": dataset.total_rows,
        "train_end_index": dataset.raw_train_end,
        "val_start_index": dataset.val_start,
        "val_end_index": dataset.val_end,
        "raw_val_end_index": dataset.raw_val_end,
        "test_start_index": dataset.test_start,
        "test_end_index": len(dataset),
        "num_windows": max(0, end - start),
        "first_input_range": [start, start + seq_len],
        "first_target_range": [start + seq_len, start + seq_len + horizon],
        "last_input_range": [end - 1, end - 1 + seq_len],
        "last_target_range": [end - 1 + seq_len, end - 1 + seq_len + horizon],
    }


def run_stream(
    baseline,
    dataset: Dataset_Custom,
    start: int,
    end: int,
    device: torch.device,
    max_windows: int | None = None,
    save_predictions: Path | None = None,
) -> dict:
    if max_windows is not None:
        end = min(end, start + max_windows)
    print("[windows] " + json.dumps(describe_windows(dataset, start, end), sort_keys=True))

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    total_elements = 0
    predictions = [] if save_predictions is not None else None

    for offset, current_idx in enumerate(range(start, end)):
        update_idx = current_idx - dataset.pred_len
        if update_idx >= start:
            update_x, update_y = dataset[update_idx]
            baseline.update(update_x.unsqueeze(0), update_y.unsqueeze(0))
        else:
            update_x = update_y = None

        current_x, current_y = dataset[current_idx]
        prediction = baseline.predict(current_x.unsqueeze(0)).cpu()
        target = current_y.unsqueeze(0)
        difference = prediction.double() - target.double()
        squared_error_sum += difference.square().sum().item()
        absolute_error_sum += difference.abs().sum().item()
        total_elements += difference.numel()
        if predictions is not None:
            predictions.append(prediction.numpy())

        if offset < 5:
            origin = current_idx + dataset.seq_len
            delay_log = {
                "current_forecast_origin": origin,
                "current_target_range": [origin, origin + dataset.pred_len],
                "update_sample_input_range": None if update_x is None else [update_idx, update_idx + dataset.seq_len],
                "update_sample_target_range": None if update_y is None else [update_idx + dataset.seq_len, update_idx + dataset.seq_len + dataset.pred_len],
                "latest_observable_timestamp": origin - 1,
            }
            print("[delay-check] " + json.dumps(delay_log, sort_keys=True))

    if total_elements == 0:
        raise RuntimeError("No windows were evaluated")
    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_predictions, np.concatenate(predictions, axis=0))
    return {
        "MSE": squared_error_sum / total_elements,
        "MAE": absolute_error_sum / total_elements,
        "num_test_windows": end - start,
        "num_evaluated_elements": total_elements,
    }


def select_lr(method: str, dataset: Dataset_Custom, horizon: int, args, device, case_dir: Path):
    if method == "frozen":
        return 0.0, []
    trials = []
    for lr in args.online_lr_grid:
        print(f"[validation] method={method} lr={lr:g}")
        baseline, _ = fresh_baseline(method, dataset, horizon, lr, args, device)
        metrics = run_stream(
            baseline,
            dataset,
            dataset.val_start,
            dataset.val_end,
            device,
            max_windows=args.max_val_windows,
        )
        trials.append({"online_lr": lr, **metrics})
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / f"{method}_validation_lr.json").open("w", encoding="utf-8") as handle:
        json.dump(trials, handle, indent=2)
    best = min(trials, key=lambda item: item["MSE"])
    print(f"[validation-best] method={method} lr={best['online_lr']:g} MSE={best['MSE']:.10g}")
    return best["online_lr"], trials


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="dataset")
    parser.add_argument("--weight_dir", default="weights")
    parser.add_argument("--output_dir", default="logs/online_baselines")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--methods", nargs="+", default=["frozen"])
    parser.add_argument("--allow_adaptive", action="store_true", help="Required for naive/ER/DERpp after Frozen alignment passes")
    parser.add_argument("--online_lr_grid", nargs="+", type=float, default=DEFAULT_LR_GRID)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--update_steps", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=500)
    parser.add_argument("--replay_batch_size", type=int, default=8)
    parser.add_argument("--replay_weight", type=float, default=0.2)
    parser.add_argument("--distill_weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--max_val_windows", type=int, default=None, help="Smoke-test only; omit for formal runs")
    parser.add_argument("--max_test_windows", type=int, default=None, help="Smoke-test only; omit for formal runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.lower().replace("++", "pp") for m in args.methods]
    if any(method != "frozen" for method in methods) and not args.allow_adaptive:
        raise SystemExit("Adaptive baselines are gated. Run Frozen alignment first, then pass --allow_adaptive.")
    if args.max_val_windows is not None or args.max_test_windows is not None:
        print("[WARNING] max_*_windows is set: this is a smoke test, not a formal result")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"

    print("[protocol] split=60/10/30 scaler_fit=train-only validation_adaptation_to_test=false")
    print("[protocol] order=update(i-H)->predict(i)->evaluate(i) batch=1 shuffle=false stride=1")
    with summary_path.open("a", encoding="utf-8") as summary:
        for data_path in args.datasets:
            for horizon in args.horizons:
                dataset = build_dataset(args, data_path, horizon)
                case = f"{dataset_name(data_path)}_H{horizon}"
                case_dir = output_dir / case
                for method in methods:
                    best_lr, validation_trials = select_lr(method, dataset, horizon, args, device, case_dir)
                    # Fresh model, optimizer and buffer: validation state never reaches test.
                    baseline, config = fresh_baseline(method, dataset, horizon, best_lr, args, device)
                    pred_path = case_dir / f"{method}_pred.npy" if args.save_predictions else None
                    metrics = run_stream(
                        baseline,
                        dataset,
                        dataset.test_start,
                        len(dataset),
                        device,
                        max_windows=args.max_test_windows,
                        save_predictions=pred_path,
                    )
                    record = {
                        "case": case,
                        "dataset": data_path,
                        "horizon": horizon,
                        "method": method,
                        "selected_online_lr": best_lr,
                        "validation_trials": validation_trials,
                        "architecture": architecture_config(dataset, horizon, args),
                        "config": config.to_log_dict(),
                        **metrics,
                    }
                    print("[result] " + json.dumps(record, sort_keys=True))
                    summary.write(json.dumps(record, sort_keys=True) + "\n")
                    summary.flush()


if __name__ == "__main__":
    main()
