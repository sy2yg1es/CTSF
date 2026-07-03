#!/usr/bin/env python
"""Audit and normalize Prompt-Z/backbone weight names.

Canonical names used by the current scripts:

Backbone:
    weights/patchtst_pretrained_<Dataset>_H<horizon>.pth

Prompt-Z B baseline:
    weights/prompt_z/prompt_z_<Dataset>_H<horizon>_pzB_v1_final.pth

Prompt-Z 0.02 probe:
    weights/prompt_z/prompt_z_<Dataset>_H<horizon>_pzB002_v1_final.pth
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(".")
WEIGHTS = ROOT / "weights"
PZ_DIR = WEIGHTS / "prompt_z"

MAIN_12 = [
    ("ECL", 1), ("ECL", 24), ("ECL", 48), ("ECL", 96),
    ("Traffic", 1), ("Traffic", 24), ("Traffic", 48), ("Traffic", 96),
    ("ETTh1", 1), ("ETTh1", 24), ("ETTh1", 48), ("ETTh1", 96),
]
MAIN_16 = [
    ("WTH", 1), ("WTH", 24), ("WTH", 48), ("WTH", 96),
    ("ETTm2", 1), ("ETTm2", 24), ("ETTm2", 48), ("ETTm2", 96),
    ("ETTm1", 1), ("ETTm1", 24), ("ETTm1", 48), ("ETTm1", 96),
    ("ETTh2", 1), ("ETTh2", 24), ("ETTh2", 48), ("ETTh2", 96),
]
ALL_CASES = MAIN_12 + MAIN_16
PROBE_002 = [
    ("ECL", 24), ("ECL", 96),
    ("ETTh2", 96),
    ("WTH", 48), ("WTH", 96),
    ("Traffic", 24), ("Traffic", 48),
    ("ETTh1", 1),
]


def backbone_candidates(ds: str, h: int) -> list[Path]:
    names = [
        f"patchtst_pretrained_{ds}_H{h}.pth",
        f"patchtst_pretrained_{ds}_H{h}_GA1.pth",
        f"pretrain_{ds}_H{h}.pth",
        f"pretrained_{ds}_H{h}.pth",
        f"{ds}_H{h}_best.pth",
        f"{ds}_H{h}.pth",
    ]
    dirs = [WEIGHTS, WEIGHTS / "backbone", WEIGHTS / "pretrained"]
    return [d / name for d in dirs for name in names]


def find_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def pz_path(ds: str, h: int, tag: str) -> Path:
    return PZ_DIR / f"prompt_z_{ds}_H{h}_{tag}_final.pth"


def pz_source_for_b(ds: str, h: int) -> Path | None:
    current = pz_path(ds, h, "pzB_v1")
    if current.exists():
        return current
    source_tag = "main_B_12cases" if (ds, h) in MAIN_12 else "main_B_16cases"
    source = pz_path(ds, h, source_tag)
    if source.exists():
        return source
    return None


def copy_canonical(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def audit(args: argparse.Namespace) -> None:
    print("=================================================================")
    print(" Backbone weights")
    print("=================================================================")
    missing_backbone = []
    for ds, h in ALL_CASES:
        found = find_existing(backbone_candidates(ds, h))
        if found is None:
            missing_backbone.append((ds, h))
            print(f"[MISSING] {ds}_H{h}: expected weights/patchtst_pretrained_{ds}_H{h}.pth")
        else:
            canonical = WEIGHTS / f"patchtst_pretrained_{ds}_H{h}.pth"
            status = "canonical" if found == canonical else f"alias={found}"
            print(f"[OK]      {ds}_H{h}: {status}")

    print("")
    print("=================================================================")
    print(" Prompt-Z pzB_v1 canonical weights")
    print("=================================================================")
    missing_pzb = []
    for ds, h in ALL_CASES:
        dst = pz_path(ds, h, "pzB_v1")
        src = pz_source_for_b(ds, h)
        if src is None:
            missing_pzb.append((ds, h))
            print(f"[MISSING] {ds}_H{h}: no pzB_v1/main_B source")
            continue
        if args.copy_canonical:
            action = copy_canonical(src, dst)
        else:
            action = "would_copy" if not dst.exists() else "exists"
        print(f"[OK]      {ds}_H{h}: {dst.name} ({action}; source={src.name})")

    print("")
    print("=================================================================")
    print(" Prompt-Z pzB002_v1 probe weights")
    print("=================================================================")
    missing_002 = []
    for ds, h in PROBE_002:
        dst = pz_path(ds, h, "pzB002_v1")
        if dst.exists():
            print(f"[OK]      {ds}_H{h}: {dst.name}")
        else:
            missing_002.append((ds, h))
            print(f"[MISSING] {ds}_H{h}: {dst.name}")

    canonical_keep = {pz_path(ds, h, "pzB_v1") for ds, h in ALL_CASES}
    canonical_keep.update(pz_path(ds, h, "pzB002_v1") for ds, h in PROBE_002)
    all_promptz = sorted(PZ_DIR.glob("prompt_z_*_final.pth"))
    unused = [p for p in all_promptz if p not in canonical_keep]

    print("")
    print("=================================================================")
    print(" Unused Prompt-Z final weights under current comparison plan")
    print("=================================================================")
    if not unused:
        print("[OK] none")
    for path in unused:
        if args.delete_unused_promptz:
            path.unlink()
            print(f"[DELETED] {path}")
        else:
            print(f"[UNUSED]  {path.name}")

    print("")
    print("=================================================================")
    print(" Summary")
    print("=================================================================")
    print(f"missing_backbone={len(missing_backbone)}")
    print(f"missing_pzB_v1={len(missing_pzb)}")
    print(f"missing_pzB002_v1={len(missing_002)}")
    print(f"unused_promptz_final={len(unused)}")
    if missing_backbone:
        print("[ACTION] Upload/copy backbone weights using canonical names, e.g.")
        print("         weights/patchtst_pretrained_ECL_H1.pth")
    if args.delete_unused_promptz:
        print("[!] Unused Prompt-Z final weights were deleted.")
    else:
        print("[SAFE] No files deleted. Use --delete-unused-promptz to remove UNUSED final weights.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-canonical", action="store_true",
                        help="Copy main_B_* final Prompt-Z weights to pzB_v1 canonical names.")
    parser.add_argument("--delete-unused-promptz", action="store_true",
                        help="Delete non-canonical Prompt-Z *_final.pth files listed as UNUSED.")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
