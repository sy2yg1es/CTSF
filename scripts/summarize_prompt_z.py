#!/usr/bin/env python
"""Summarize Prompt-Z JSON logs into a compact CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_summary(logdir: Path, tag: str) -> dict[tuple[str, str], dict[str, str]]:
    path = logdir / f"summary_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")

    rows = {}
    suffix = f"_{tag}"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            case = row["case"]
            if case.endswith(suffix):
                case = case[:-len(suffix)]
            rows[(case, row["method"])] = row
    return rows


def write_compare_rows(rows: list[dict[str, object]], out: Path | None) -> None:
    if not rows:
        print("[!] No comparison rows.")
        return

    fields = list(rows[0].keys())
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[*] Comparison saved to {out}")

    print(" | ".join(fields))
    print("-" * min(140, 3 + sum(len(f) + 3 for f in fields)))
    for row in rows:
        print(" | ".join(str(row.get(f, "")) for f in fields))


def compare_ratio(logdir: Path, base_tag: str, probe_tag: str, out: Path | None) -> None:
    base = read_summary(logdir, base_tag)
    probe = read_summary(logdir, probe_tag)
    cases = sorted({case for case, _method in probe})

    rows = []
    for case in cases:
        base_frozen = base.get((case, "frozen"), {})
        probe_frozen = probe.get((case, "frozen"), {})
        base_mode0 = base.get((case, "pz_mode0"), {})
        probe_mode0 = probe.get((case, "pz_mode0"), {})
        base_sel = base.get((case, "pz_selected"), {})
        probe_sel = probe.get((case, "pz_selected"), {})
        rows.append({
            "case": case,
            "frozen": base_frozen.get("MSE", probe_frozen.get("MSE", "")),
            f"{base_tag}_mode0": base_mode0.get("MSE", ""),
            f"{probe_tag}_mode0": probe_mode0.get("MSE", ""),
            f"{base_tag}_selected": base_sel.get("MSE", ""),
            f"{probe_tag}_selected": probe_sel.get("MSE", ""),
            "delta_005": base_mode0.get("delta_vs_frozen_pct", ""),
            "delta_002": probe_mode0.get("delta_vs_frozen_pct", ""),
            "eff_005": base_mode0.get("effective_delta_ratio", ""),
            "eff_002": probe_mode0.get("effective_delta_ratio", ""),
        })
    write_compare_rows(rows, out)


def compare_margin(logdir: Path, base_tag: str, probe_tag: str, out: Path | None) -> None:
    base = read_summary(logdir, base_tag)
    probe = read_summary(logdir, probe_tag)
    cases = sorted({case for case, method in probe if method == "pz_selected"})

    rows = []
    for case in cases:
        base_sel = base.get((case, "pz_selected"), {})
        probe_sel = probe.get((case, "pz_selected"), {})
        rows.append({
            "case": case,
            f"{base_tag}_selected_MSE": base_sel.get("MSE", ""),
            f"{probe_tag}_selected_MSE": probe_sel.get("MSE", ""),
            "promptz_enabled_005": base_sel.get("promptz_enabled", ""),
            "promptz_enabled_002": probe_sel.get("promptz_enabled", ""),
            "val_delta_percent_005": base_sel.get("val_delta_percent", ""),
            "val_delta_percent_002": probe_sel.get("val_delta_percent", ""),
            "test_promptz_raw_mse_005": base_sel.get("test_promptz_raw_mse", ""),
            "test_promptz_raw_mse_002": probe_sel.get("test_promptz_raw_mse", ""),
            "test_promptz_selected_mse_005": base_sel.get("test_promptz_selected_mse", ""),
            "test_promptz_selected_mse_002": probe_sel.get("test_promptz_selected_mse", ""),
        })
    write_compare_rows(rows, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument("--logdir", default="logs/prompt_z")
    parser.add_argument("--compare_ratio", nargs=2, metavar=("BASE_TAG", "PROBE_TAG"))
    parser.add_argument("--compare_margin", nargs=2, metavar=("BASE_TAG", "PROBE_TAG"))
    parser.add_argument("--out")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    out = Path(args.out) if args.out else None
    if args.compare_ratio:
        compare_ratio(logdir, args.compare_ratio[0], args.compare_ratio[1], out)
        return
    if args.compare_margin:
        compare_margin(logdir, args.compare_margin[0], args.compare_margin[1], out)
        return
    if not args.tag:
        raise SystemExit("--tag is required unless --compare_ratio or --compare_margin is used")

    tag = args.tag
    out = logdir / f"summary_{tag}.csv"

    rows = []
    for path in sorted(logdir.glob(f"*_{tag}_*.json")):
        name = path.stem
        if name.startswith("train_"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "MSE" not in data:
            continue
        rows.append({
            "experiment": name,
            "MSE": data.get("MSE"),
            "MAE": data.get("MAE"),
            "RMSE": data.get("RMSE"),
            "calibration_updates": data.get("calibration_updates", 0),
            "gamma_mean": data.get("gamma_mean_mean", ""),
            "mask_ratio": data.get("mask_ratio_mean", ""),
            "raw_delta_to_hidden_ratio": data.get("raw_delta_to_hidden_ratio_mean", ""),
            "effective_delta_ratio": data.get("effective_delta_ratio_mean", ""),
            "val_frozen_mse": data.get("val_frozen_mse", ""),
            "val_promptz_mse": data.get("val_promptz_mse", ""),
            "val_delta_percent": data.get("val_delta_percent", ""),
            "fallback_margin": data.get("fallback_margin", ""),
            "promptz_enabled": data.get("promptz_enabled", ""),
            "test_frozen_mse": data.get("test_frozen_mse", ""),
            "test_promptz_raw_mse": data.get("test_promptz_raw_mse", ""),
            "test_promptz_selected_mse": data.get("test_promptz_selected_mse", ""),
        })

    frozen_by_case = {}
    suffixes = ("_frozen", "_pz_random_mode0", "_pz_mode0", "_pz_mode1", "_pz_selected")
    for row in rows:
        exp = row["experiment"]
        if exp.endswith("_frozen"):
            frozen_by_case[exp[:-len("_frozen")]] = float(row["MSE"])

    for row in rows:
        exp = row["experiment"]
        case = None
        for suffix in suffixes:
            if exp.endswith(suffix):
                case = exp[:-len(suffix)]
                row["method"] = suffix[1:]
                break
        row["case"] = case or exp
        row["method"] = row.get("method", "unknown")
        if row["method"] == "pz_selected" and row.get("test_frozen_mse") not in ("", None):
            base = float(row["test_frozen_mse"])
        else:
            base = frozen_by_case.get(row["case"])
        row["delta_vs_frozen_pct"] = (
            (float(row["MSE"]) / base - 1.0) * 100.0
            if base is not None and row["method"] != "frozen" else 0.0
        )

    rows.sort(key=lambda r: (r["case"], r.get("method", "")))
    fields = [
        "case", "method", "experiment", "MSE", "MAE", "RMSE",
        "delta_vs_frozen_pct", "calibration_updates",
        "gamma_mean", "mask_ratio", "raw_delta_to_hidden_ratio", "effective_delta_ratio",
        "val_frozen_mse", "val_promptz_mse", "val_delta_percent",
        "fallback_margin", "promptz_enabled",
        "test_frozen_mse", "test_promptz_raw_mse", "test_promptz_selected_mse",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"[*] Summary saved to {out}")
    print(f"{'case':<24} {'method':<18} {'MSE':>10} {'delta%':>10}")
    print("-" * 66)
    for row in rows:
        print(
            f"{row['case']:<24} {row['method']:<18} "
            f"{float(row['MSE']):>10.6f} {float(row['delta_vs_frozen_pct']):>10.2f}"
        )


if __name__ == "__main__":
    main()
