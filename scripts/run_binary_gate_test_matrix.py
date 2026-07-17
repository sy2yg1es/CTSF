"""Resumable server runner for the finalized Binary Channel Gate matrix.

The default mode is deliberately ``preflight``.  Formal TEST is only entered
after every requested case has a protocol-valid gate checkpoint.  ``all`` runs
the complete prepare pass first and the complete TEST pass second.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.binary_channel_gate import validate_gate_checkpoint


HORIZONS = (1, 12, 24, 48)
BACKBONES = ("patchtst", "itransformer")
GATE_HISTORY = {1: 2000, 12: 8000, 24: 2000, 48: 2000}
DEFAULT_PROTOCOL_TAG = "binarygate_v1"


def short_backbone(backbone: str) -> str:
    return "patch" if backbone == "patchtst" else "itrans"


def discover_datasets(dataset_dir: Path) -> list[str]:
    return sorted(path.stem for path in dataset_dir.glob("*.csv"))


def backbone_path(dataset: str, horizon: int, backbone: str) -> Path:
    return ROOT / "weights" / f"{backbone}_pretrained_{dataset}_H{horizon}.pth"


def preferred_p1_path(
    dataset: str, horizon: int, backbone: str, protocol_tag: str
) -> Path:
    short = short_backbone(backbone)
    return ROOT / "weights" / "prompt_z" / (
        f"gfv2_{dataset}_H{horizon}_{short}_{protocol_tag}_p1.pth"
    )


def compatible_p1_candidates(
    dataset: str, horizon: int, backbone: str, protocol_tag: str
) -> list[Path]:
    """Return P1 candidates whose fit range can be disjoint from gate history."""
    short = short_backbone(backbone)
    candidates = [preferred_p1_path(dataset, horizon, backbone, protocol_tag)]
    if protocol_tag != DEFAULT_PROTOCOL_TAG:
        return candidates
    if horizon == 12:
        # The 8k-tagged checkpoints were trained before an 8k gate tail.
        candidates.append(
            ROOT / "weights" / "prompt_z" /
            f"gfv2_{dataset}_H12_{short}_gate8k_v1_p1.pth"
        )
    else:
        candidates.append(
            ROOT / "weights" / "prompt_z" /
            f"gfv2_{dataset}_H{horizon}_{short}_cleanp1_v1_p1.pth"
        )
    return candidates


def gate_path(
    dataset: str, horizon: int, backbone: str, seed: int, protocol_tag: str
) -> Path:
    short = short_backbone(backbone)
    return ROOT / "weights" / "prompt_z" / "gate_probe" / (
        f"gate_probe_{dataset}_H{horizon}_causal_augmented_s{seed}_"
        f"{short}_{protocol_tag}.pth"
    )


def test_result_path(
    dataset: str, horizon: int, backbone: str, out_dir: Path, protocol_tag: str
) -> Path:
    short = short_backbone(backbone)
    return out_dir / (
        f"binary_gate_test_{dataset}_H{horizon}_{backbone}_"
        f"{short}_{protocol_tag}.json"
    )


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_gate(path: Path):
    if not path.exists():
        return None, "missing"
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validate_gate_checkpoint(checkpoint)
        return checkpoint, "valid"
    except Exception as error:  # preflight must report every bad artifact
        return None, f"invalid: {error}"


def load_test_result(path: Path, expected_gate: Path):
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("protocol") != "binary_channel_gate_v1":
            return None
        if result.get("decision_threshold") != 0.0:
            return None
        saved_gate = result.get("artifacts", {}).get("binary_gate")
        if saved_gate is None or Path(saved_gate).name != expected_gate.name:
            return None
        return result
    except (OSError, ValueError, TypeError):
        return None


def is_valid_p1_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return isinstance(state, dict) and bool(state)
    except Exception:
        return False


def resolve_p1(
    dataset: str, horizon: int, backbone: str, protocol_tag: str, checkpoint=None
):
    if checkpoint is not None:
        saved = checkpoint.get("config", {}).get("p1_ckpt")
        if saved:
            saved_path = Path(saved)
            if not saved_path.is_absolute():
                saved_path = ROOT / saved_path
            if is_valid_p1_checkpoint(saved_path):
                return saved_path
    for candidate in compatible_p1_candidates(
        dataset, horizon, backbone, protocol_tag
    ):
        if is_valid_p1_checkpoint(candidate):
            return candidate
    return None


def case_key(dataset: str, horizon: int, backbone: str) -> str:
    return f"{dataset}_H{horizon}_{backbone}"


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_command(command: list[str], label: str, dry_run: bool) -> None:
    print("\n" + "=" * 96, flush=True)
    print(f"[RUN] {label}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    print("=" * 96, flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def p1_command(args, dataset, horizon, backbone, output: Path) -> list[str]:
    short = short_backbone(backbone)
    return [
        args.python,
        "scripts/train_gamma_final_v2.py",
        "--root_path", relative(args.dataset_dir),
        "--data_path", f"{dataset}.csv",
        "--forecast_H", str(horizon),
        "--backbone", backbone,
        "--pretrained_weights", relative(backbone_path(dataset, horizon, backbone)),
        "--d_drift", "64",
        "--rank", "8",
        "--gamma_init_bias", "0",
        "--max_delta_ratio", str(args.max_delta_ratio),
        "--residual_window_K", "24",
        "--phase1_steps", str(args.p1_steps),
        # Reserve the complete gate tail so P1 and gate labels never overlap.
        "--phase2_steps", str(GATE_HISTORY[horizon]),
        "--phase1_selection_source", "train_tail",
        "--phase1_selection_fraction", "0.2",
        "--stop_after_phase1",
        "--val_interval", str(args.p1_val_interval),
        "--lr", "0.001",
        "--weight_decay", "0.0001",
        "--lambda_excess", "1.0",
        "--num_workers", str(args.num_workers),
        "--log_interval", str(args.log_interval),
        "--save_dir", relative(output.parent),
        "--experiment_tag", f"{short}_{args.protocol_tag}",
    ]


def gate_command(args, dataset, horizon, backbone, p1: Path) -> list[str]:
    short = short_backbone(backbone)
    return [
        args.python,
        "scripts/train_binary_gate.py",
        "--root_path", relative(args.dataset_dir),
        "--data_path", f"{dataset}.csv",
        "--forecast_H", str(horizon),
        "--backbone", backbone,
        "--pretrained_weights", relative(backbone_path(dataset, horizon, backbone)),
        "--p1_ckpt", relative(p1),
        "--d_drift", "64",
        "--rank", "8",
        "--max_delta_ratio", str(args.max_delta_ratio),
        "--residual_window_K", "24",
        "--train_steps", str(GATE_HISTORY[horizon]),
        "--val_steps", str(args.val_steps),
        "--warmup_steps", str(args.warmup_steps),
        "--target_margin_pct", "0",
        "--feature_mode", "causal_augmented",
        "--probe_hidden", "64",
        "--final_regressor_only",
        "--probe_epochs", "100",
        "--probe_lr", "0.001",
        "--probe_weight_decay", "0.0001",
        "--batch_size", "512",
        "--patience", "12",
        "--seed", str(args.seed),
        "--validation_protocol", "fixed_zero_blocked",
        "--train_selection_fraction", "0.2",
        "--validation_blocks", "4",
        "--safe_min_improvement_pct", "0.2",
        "--safe_min_positive_block_frac", "0.75",
        "--num_workers", str(args.num_workers),
        "--log_interval", str(args.log_interval),
        "--experiment_tag", f"{short}_{args.protocol_tag}",
    ]


def test_command(args, dataset, horizon, backbone, p1: Path, gate: Path) -> list[str]:
    short = short_backbone(backbone)
    return [
        args.python,
        "scripts/eval_binary_gate_test.py",
        "--root_path", relative(args.dataset_dir),
        "--data_path", f"{dataset}.csv",
        "--forecast_H", str(horizon),
        "--backbone", backbone,
        "--pretrained_weights", relative(backbone_path(dataset, horizon, backbone)),
        "--p1_ckpt", relative(p1),
        "--gate_ckpt", relative(gate),
        "--d_drift", "64",
        "--rank", "8",
        "--max_delta_ratio", str(args.max_delta_ratio),
        "--residual_window_K", "24",
        "--num_workers", str(args.num_workers),
        "--out_dir", relative(args.out_dir),
        "--experiment_tag", f"{short}_{args.protocol_tag}",
    ]


def requested_cases(args):
    combinations = [
        (dataset, horizon, backbone)
        for dataset in args.datasets
        for horizon in args.horizons
        for backbone in args.backbones
    ]
    return combinations[: args.limit_cases] if args.limit_cases else combinations


def preflight(args, cases):
    rows = []
    for dataset, horizon, backbone in cases:
        backbone_file = backbone_path(dataset, horizon, backbone)
        gate_file = gate_path(
            dataset, horizon, backbone, args.seed, args.protocol_tag
        )
        checkpoint, gate_status = load_gate(gate_file)
        p1 = resolve_p1(
            dataset, horizon, backbone, args.protocol_tag, checkpoint
        )
        result_file = test_result_path(
            dataset, horizon, backbone, args.out_dir, args.protocol_tag
        )
        result = load_test_result(result_file, gate_file)
        rows.append({
            "case": case_key(dataset, horizon, backbone),
            "backbone": "ready" if backbone_file.exists() else "missing",
            "p1": relative(p1) if p1 else "missing",
            "gate": gate_status,
            "test": "complete" if result is not None else "pending",
        })
    return rows


def print_preflight(rows) -> None:
    print("\nPreflight")
    print(f"{'case':40s} {'backbone':9s} {'gate':10s} {'test':9s} p1")
    print("-" * 110)
    for row in rows:
        print(
            f"{row['case']:40s} {row['backbone']:9s} "
            f"{row['gate'][:10]:10s} {row['test']:9s} {row['p1']}"
        )
    ready = sum(
        row["backbone"] == "ready" and row["p1"] != "missing"
        and row["gate"] == "valid" for row in rows
    )
    print(f"\nProtocol-ready: {ready}/{len(rows)}")


def prepare_all(args, cases, manifest):
    for dataset, horizon, backbone in cases:
        key = case_key(dataset, horizon, backbone)
        state = manifest["cases"].setdefault(key, {})
        backbone_file = backbone_path(dataset, horizon, backbone)
        if not backbone_file.exists():
            state.update(status="missing_backbone", backbone=relative(backbone_file))
            save_manifest(args.manifest, manifest)
            continue

        gate_file = gate_path(
            dataset, horizon, backbone, args.seed, args.protocol_tag
        )
        checkpoint, gate_status = load_gate(gate_file)
        p1 = resolve_p1(
            dataset, horizon, backbone, args.protocol_tag, checkpoint
        )
        p1_rebuilt = False
        if args.force_prepare or p1 is None:
            p1 = preferred_p1_path(
                dataset, horizon, backbone, args.protocol_tag
            )
            p1_rebuilt = True
            state["status"] = "p1_running"
            save_manifest(args.manifest, manifest)
            run_command(
                p1_command(args, dataset, horizon, backbone, p1),
                f"{key} clean non-overlap P1",
                args.dry_run,
            )
            if args.dry_run:
                state["status"] = "p1_dry_run"
                save_manifest(args.manifest, manifest)
            elif not p1.exists():
                raise FileNotFoundError(f"P1 command did not create {p1}")

        if args.force_prepare or p1_rebuilt or gate_status != "valid":
            state["status"] = "gate_running"
            state["p1"] = relative(p1)
            save_manifest(args.manifest, manifest)
            run_command(
                gate_command(args, dataset, horizon, backbone, p1),
                f"{key} final binary gate",
                args.dry_run,
            )
            if args.dry_run:
                state["status"] = "gate_dry_run"
                save_manifest(args.manifest, manifest)
                continue

        checkpoint, gate_status = load_gate(gate_file)
        if gate_status != "valid":
            raise RuntimeError(f"{key}: final gate is not protocol-valid: {gate_status}")
        state.update(
            status="prepared",
            backbone=relative(backbone_file),
            p1=relative(resolve_p1(
                dataset, horizon, backbone, args.protocol_tag, checkpoint
            )),
            gate=relative(gate_file),
            selected_mode=checkpoint["selected_mode"],
        )
        save_manifest(args.manifest, manifest)


def test_all(args, cases, manifest):
    missing = []
    ready = []
    for dataset, horizon, backbone in cases:
        key = case_key(dataset, horizon, backbone)
        state = manifest["cases"].setdefault(key, {})
        backbone_file = backbone_path(dataset, horizon, backbone)
        gate_file = gate_path(
            dataset, horizon, backbone, args.seed, args.protocol_tag
        )
        checkpoint, gate_status = load_gate(gate_file)
        p1 = resolve_p1(
            dataset, horizon, backbone, args.protocol_tag, checkpoint
        )
        if not backbone_file.exists() or p1 is None or gate_status != "valid":
            reason = {
                "backbone": backbone_file.exists(),
                "p1": p1 is not None,
                "gate": gate_status,
            }
            state.update(status="not_test_ready", reason=reason)
            save_manifest(args.manifest, manifest)
            missing.append((key, reason))
            continue
        state.update(
            status="ready_for_test",
            backbone=relative(backbone_file),
            p1=relative(p1),
            gate=relative(gate_file),
            selected_mode=checkpoint["selected_mode"],
        )
        save_manifest(args.manifest, manifest)
        ready.append((dataset, horizon, backbone, p1, gate_file))

    # Do not touch any TEST split until the whole requested matrix is ready.
    if missing:
        return missing

    for dataset, horizon, backbone, p1, gate_file in ready:
        key = case_key(dataset, horizon, backbone)
        state = manifest["cases"].setdefault(key, {})
        result_file = test_result_path(
            dataset, horizon, backbone, args.out_dir, args.protocol_tag
        )
        result = load_test_result(result_file, gate_file)
        if args.force_test or result is None:
            state["status"] = "test_running"
            save_manifest(args.manifest, manifest)
            run_command(
                test_command(args, dataset, horizon, backbone, p1, gate_file),
                f"{key} one-shot TEST",
                args.dry_run,
            )
            if args.dry_run:
                state["status"] = "test_dry_run"
                save_manifest(args.manifest, manifest)
                continue
            result = load_test_result(result_file, gate_file)
            if result is None:
                raise RuntimeError(f"TEST command did not create a valid result: {result_file}")
        state.update(status="test_complete", result=relative(result_file), **result)
        save_manifest(args.manifest, manifest)
    return missing


def write_summary(args, manifest) -> None:
    completed = []
    for key, state in sorted(manifest["cases"].items()):
        if state.get("status") != "test_complete":
            continue
        completed.append({
            "case": key,
            "selected_mode": state["selected_mode"],
            "frozen_mse": state["frozen_mse"],
            "gate_prompt_z_mse": state["gate_prompt_z_mse"],
            "relative_improvement_pct": state["relative_improvement_pct"],
            "gate_usage_ratio": state["gate_usage_ratio"],
            "test_seconds": state["runtime"]["test_seconds"],
            "ms_per_window": state["runtime"]["milliseconds_per_test_window"],
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "binary_gate_test_summary.json"
    csv_path = args.out_dir / "binary_gate_test_summary.csv"
    json_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    if completed:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
            writer.writeheader()
            writer.writerows(completed)
    print(f"[*] summary: {json_path} ({len(completed)} completed cases)")


def parse_args():
    parser = argparse.ArgumentParser("Binary Gate full TEST matrix")
    parser.add_argument(
        "--mode", choices=["preflight", "prepare", "test", "all"],
        default="preflight",
    )
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument(
        "--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES)
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_delta_ratio", type=float, default=0.02)
    parser.add_argument("--protocol_tag", default=DEFAULT_PROTOCOL_TAG)
    parser.add_argument("--p1_steps", type=int, default=2000)
    parser.add_argument("--p1_val_interval", type=int, default=200)
    parser.add_argument("--val_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--manifest", type=Path, default=None,
    )
    parser.add_argument(
        "--out_dir", type=Path, default=None
    )
    parser.add_argument("--force_prepare", action="store_true")
    parser.add_argument("--force_test", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit_cases", type=int, default=0)
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir.resolve()
    if not (0.0 < args.max_delta_ratio <= 1.0):
        parser.error("--max_delta_ratio must be in (0, 1]")
    allowed_tag_chars = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    )
    if any(char not in allowed_tag_chars for char in args.protocol_tag):
        parser.error("--protocol_tag may contain only letters, digits, '_' and '-'")
    if (
        args.max_delta_ratio != 0.02
        and args.protocol_tag == DEFAULT_PROTOCOL_TAG
    ):
        parser.error(
            "A non-default --max_delta_ratio requires a distinct --protocol_tag "
            "so existing 0.02 artifacts cannot be overwritten"
        )
    if args.manifest is None:
        suffix = (
            "" if args.protocol_tag == DEFAULT_PROTOCOL_TAG
            else f"_{args.protocol_tag}"
        )
        args.manifest = ROOT / f"logs/prompt_z/test_matrix_manifest{suffix}.json"
    if args.out_dir is None:
        suffix = (
            "" if args.protocol_tag == DEFAULT_PROTOCOL_TAG
            else f"_{args.protocol_tag}"
        )
        args.out_dir = ROOT / f"logs/prompt_z/test_matrix{suffix}"
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.datasets is None:
        args.datasets = discover_datasets(args.dataset_dir)
    if args.seed != 2026:
        parser.error(
            "The frozen formal TEST matrix uses seed=2026; a different seed "
            "requires a new validation protocol and separate result namespace"
        )
    unknown_horizons = sorted(set(args.horizons) - set(HORIZONS))
    if unknown_horizons:
        parser.error(f"Unsupported horizons: {unknown_horizons}")
    if not args.datasets:
        parser.error(f"No CSV datasets found under {args.dataset_dir}")
    return args


def main():
    args = parse_args()
    cases = requested_cases(args)
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.manifest.exists()
        else {
            "protocol": "binary_channel_gate_v1",
            "protocol_tag": args.protocol_tag,
            "max_delta_ratio": args.max_delta_ratio,
            "seed": args.seed,
            "cases": {},
        }
    )
    if manifest.get("protocol") != "binary_channel_gate_v1":
        raise ValueError(f"Refusing incompatible manifest: {args.manifest}")
    if manifest.get("seed") != args.seed:
        raise ValueError(
            f"Manifest seed={manifest.get('seed')} does not match requested seed={args.seed}"
        )
    if manifest.get("protocol_tag", DEFAULT_PROTOCOL_TAG) != args.protocol_tag:
        raise ValueError(
            f"Manifest protocol_tag={manifest.get('protocol_tag')} does not match "
            f"requested protocol_tag={args.protocol_tag}"
        )
    saved_max_delta_ratio = float(manifest.get("max_delta_ratio", 0.02))
    if abs(saved_max_delta_ratio - args.max_delta_ratio) > 1e-12:
        raise ValueError(
            f"Manifest max_delta_ratio={manifest.get('max_delta_ratio')} does not "
            f"match requested max_delta_ratio={args.max_delta_ratio}"
        )

    print(
        f"[*] mode={args.mode} datasets={len(args.datasets)} cases={len(cases)} "
        f"seed={args.seed} max_delta_ratio={args.max_delta_ratio} "
        f"protocol_tag={args.protocol_tag} dry_run={args.dry_run}"
    )
    print_preflight(preflight(args, cases))
    started = time.perf_counter()

    if args.mode in ("prepare", "all"):
        prepare_all(args, cases, manifest)
    missing = []
    if args.mode in ("test", "all"):
        missing = test_all(args, cases, manifest)
        if not missing:
            write_summary(args, manifest)
    save_manifest(args.manifest, manifest)
    print(f"[*] elapsed={time.perf_counter() - started:.1f}s manifest={args.manifest}")
    if missing and not args.dry_run:
        details = "\n".join(f"  {key}: {reason}" for key, reason in missing)
        raise SystemExit(
            f"{len(missing)} cases were not TEST-ready. Run --mode prepare first:\n{details}"
        )


if __name__ == "__main__":
    main()
