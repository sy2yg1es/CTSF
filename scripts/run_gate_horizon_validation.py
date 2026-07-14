"""Resumable clean-P1 and binary-gate validation across forecast horizons.

The runner deliberately executes one case at a time so GPU memory and log
ownership stay simple. Existing checkpoints/results are skipped. A manifest is
updated after every command, making an interrupted matrix safe to resume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp.replace(path)


def run_command(command: list[str], label: str) -> None:
    print("\n" + "=" * 88, flush=True)
    print(f"[RUN] {label}", flush=True)
    print(" ".join(command), flush=True)
    print("=" * 88, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def result_path(dataset: str, horizon: int, backbone: str, seed: int) -> Path:
    tag = "patch_binarygate_v1" if backbone == "patchtst" else "itrans_binarygate_v1"
    return ROOT / "logs" / "prompt_z" / "gate_probe" / (
        f"gate_probe_{dataset}_H{horizon}_causal_augmented_s{seed}_{tag}.json"
    )


def p1_path(dataset: str, horizon: int, backbone: str) -> Path:
    short = "patch" if backbone == "patchtst" else "itrans"
    protocol_tag = "binarygate_h12_v1" if horizon == 12 else "cleanp1_v1"
    return ROOT / "weights" / "prompt_z" / (
        f"gfv2_{dataset}_H{horizon}_{short}_{protocol_tag}_p1.pth"
    )


def main() -> None:
    parser = argparse.ArgumentParser("Resumable multi-horizon binary-gate validation")
    parser.add_argument("--datasets", nargs="+", default=["ETTm1", "ETTm2"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 12, 24, 48])
    parser.add_argument(
        "--backbones", nargs="+", default=["patchtst", "itransformer"],
        choices=["patchtst", "itransformer"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--root_path", default="./dataset")
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--h12_train_steps", type=int, default=8000)
    parser.add_argument("--val_steps", type=int, default=2000)
    parser.add_argument("--p1_steps", type=int, default=2000)
    parser.add_argument("--selection_fraction", type=float, default=0.2)
    parser.add_argument(
        "--manifest", default="logs/prompt_z/gate_horizon_manifest_binary_v1.json"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"protocol": "binary_channel_gate_v1", "cases": {}}

    for dataset in args.datasets:
        data_path = f"{dataset}.csv"
        for horizon in args.horizons:
            gate_train_steps = (
                args.h12_train_steps if horizon == 12 else args.train_steps
            )
            for backbone in args.backbones:
                case = f"{dataset}_H{horizon}_{backbone}"
                state = manifest["cases"].setdefault(case, {})
                backbone_weights = ROOT / "weights" / (
                    f"{backbone}_pretrained_{dataset}_H{horizon}.pth"
                )
                if not backbone_weights.exists():
                    state.update(status="missing_backbone", path=str(backbone_weights))
                    save_manifest(manifest_path, manifest)
                    print(f"[SKIP] {case}: missing {backbone_weights}", flush=True)
                    continue

                clean_p1 = p1_path(dataset, horizon, backbone)
                if args.force or not clean_p1.exists():
                    short = "patch" if backbone == "patchtst" else "itrans"
                    p1_tag = (
                        f"{short}_binarygate_h12_v1"
                        if horizon == 12
                        else f"{short}_cleanp1_v1"
                    )
                    command = [
                        sys.executable,
                        "scripts/train_gamma_final_v2.py",
                        "--root_path", args.root_path,
                        "--data_path", data_path,
                        "--forecast_H", str(horizon),
                        "--backbone", backbone,
                        "--pretrained_weights", str(backbone_weights.relative_to(ROOT)),
                        "--d_drift", "64",
                        "--rank", "8",
                        "--gamma_init_bias", "0",
                        "--max_delta_ratio", "0.02",
                        "--residual_window_K", "24",
                        "--phase1_steps", str(args.p1_steps),
                        "--phase2_steps", str(gate_train_steps),
                        "--phase1_selection_source", "train_tail",
                        "--phase1_selection_fraction", str(args.selection_fraction),
                        "--stop_after_phase1",
                        "--val_interval", "200",
                        "--lr", "0.001",
                        "--weight_decay", "0.0001",
                        "--lambda_excess", "1.0",
                        "--num_workers", "0",
                        "--experiment_tag", p1_tag,
                    ]
                    state["status"] = "p1_running"
                    save_manifest(manifest_path, manifest)
                    try:
                        run_command(command, f"{case} clean P1")
                    except subprocess.CalledProcessError as error:
                        state.update(status="p1_failed", returncode=error.returncode)
                        save_manifest(manifest_path, manifest)
                        raise
                state.update(status="p1_complete", p1_path=str(clean_p1))
                save_manifest(manifest_path, manifest)

                for seed in args.seeds:
                    output = result_path(dataset, horizon, backbone, seed)
                    seed_state = state.setdefault("seeds", {}).setdefault(str(seed), {})
                    if args.force or not output.exists():
                        tag = (
                            "patch_binarygate_v1"
                            if backbone == "patchtst"
                            else "itrans_binarygate_v1"
                        )
                        command = [
                            sys.executable,
                            "scripts/train_binary_gate.py",
                            "--root_path", args.root_path,
                            "--data_path", data_path,
                            "--forecast_H", str(horizon),
                            "--backbone", backbone,
                            "--pretrained_weights", str(backbone_weights.relative_to(ROOT)),
                            "--p1_ckpt", str(clean_p1.relative_to(ROOT)),
                            "--d_drift", "64",
                            "--rank", "8",
                            "--max_delta_ratio", "0.02",
                            "--residual_window_K", "24",
                            "--train_steps", str(gate_train_steps),
                            "--val_steps", str(args.val_steps),
                            "--warmup_steps", "1000",
                            "--target_margin_pct", "0",
                            "--feature_mode", "causal_augmented",
                            "--probe_hidden", "64",
                            "--probe_epochs", "100",
                            "--probe_lr", "0.001",
                            "--probe_weight_decay", "0.0001",
                            "--batch_size", "512",
                            "--patience", "12",
                            "--seed", str(seed),
                            "--validation_protocol", "fixed_zero_blocked",
                            "--train_selection_fraction", str(args.selection_fraction),
                            "--validation_blocks", "4",
                            "--experiment_tag", tag,
                        ]
                        seed_state["status"] = "running"
                        save_manifest(manifest_path, manifest)
                        try:
                            run_command(command, f"{case} gate seed={seed}")
                        except subprocess.CalledProcessError as error:
                            seed_state.update(status="failed", returncode=error.returncode)
                            save_manifest(manifest_path, manifest)
                            raise
                    result = json.loads(output.read_text(encoding="utf-8"))
                    seed_state.update(
                        status="complete",
                        result_path=str(output),
                        selected_mode=result["safety_selection_block"]["selected_mode"],
                        regressor_vs_fixed_pct=result["holdout_validation"]
                        ["regressor"]["binary_improvement_vs_fixed_pct"],
                        regressor_vs_frozen_pct=result["holdout_validation"]
                        ["regressor"]["binary_improvement_vs_frozen_pct"],
                    )
                    save_manifest(manifest_path, manifest)

                state["status"] = "complete"
                save_manifest(manifest_path, manifest)

    print(f"\n[DONE] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
