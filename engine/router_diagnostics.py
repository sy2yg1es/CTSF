"""
Router Learnability Diagnostics
===============================

Offline diagnostics for Stage 3 routers.  The goal is not to improve the
router during this run, but to answer whether posterior oracle expert labels
are learnable from the current causal router inputs.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from core.buffer import BDLABuffer
from core.drift_detector import ActualDriftDetector
from engine.oracle_experiments import _expert_candidate_stats
from models.framework import ContinualPromptTSF


def _safe_tag(tag: str) -> str:
    tag = tag or "unnamed"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _quantiles(values: Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            "p_noop_mean": float("nan"),
            "p_noop_std": float("nan"),
            "p_noop_p10": float("nan"),
            "p_noop_p50": float("nan"),
            "p_noop_p90": float("nan"),
        }
    values = values.float()
    qs = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
    return {
        "p_noop_mean": values.mean().item(),
        "p_noop_std": values.std(unbiased=False).item(),
        "p_noop_p10": qs[0].item(),
        "p_noop_p50": qs[1].item(),
        "p_noop_p90": qs[2].item(),
    }


def _router_probs_from_history(
    model: ContinualPromptTSF,
    z_query_hist: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Replay current router on stored z_query and return probs/top-k."""
    _, routing_probs, dispatch_indices = model.prompt_memory.retrieve_prompt(z_query_hist)
    return routing_probs, dispatch_indices


def run_router_learnability_diagnostics(
    model: ContinualPromptTSF,
    dataloader,
    buffer: BDLABuffer,
    detector: ActualDriftDetector,
    optimizer: torch.optim.Optimizer,
    l_aux_weight: float = 0.0,
    train_size: int = 0,
    experiment_tag: str = "router_learnability",
    output_dir: str = "logs/stage3_diagnostics",
) -> Dict[str, float]:
    """
    Compare posterior oracle best expert labels with the current router outputs.

    Outputs:
      logs/stage3_diagnostics/router_learnability_<tag>.json
      logs/stage3_diagnostics/router_learnability_<tag>.csv
    """
    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)

    num_experts = model.prompt_memory.prompts.shape[0]
    noop_idx = num_experts
    n_router_slots = getattr(model.prompt_memory, "n_router_out", num_experts)
    router_has_noop = n_router_slots > num_experts

    mae_sum = torch.zeros(1, device=device)
    mse_sum = torch.zeros(1, device=device)
    n_aligned = 0

    oracle_counts: Counter[int] = Counter()
    top1_correct = 0
    top3_hit = 0
    total_labels = 0
    switch_count = 0
    switch_total = 0
    entropy_sum = 0.0
    entropy_total = 0
    oracle_noop_count = 0
    router_noop_count = 0
    noop_tp = 0
    p_noop_values = []
    rows = []
    prev_oracle_best: Optional[Tensor] = None

    with torch.no_grad():
        _ = model.prompt_memory.retrieve_prompt(
            torch.zeros(1, 1, model.prompt_memory.prompt_dim, device=device)
        )

    print(f"[*] Router learnability diagnostics | tag={experiment_tag}")
    print(f"[*] Experts: real={num_experts} noop_idx={noop_idx} router_has_noop={router_has_noop}")

    for t, (X_t, Y_t) in enumerate(dataloader):
        model.eval()
        with torch.no_grad():
            X_t = X_t.to(device, non_blocking=True)
            Y_hat_future, z_channel, routing_probs, dispatch_indices = model(X_t)

        buffer.push(t=t, X_t=X_t, y_hat_future=Y_hat_future,
                    dispatch_indices=dispatch_indices, z_t=z_channel)

        if Y_t is None:
            continue
        Y_hat_history = buffer.get_stored_prediction(t)
        if Y_hat_history is None:
            continue
        Y_hat_history = Y_hat_history.to(device, non_blocking=True)

        aligned = buffer.pop_and_align(t, Y_t)
        if aligned is None:
            continue

        X_history, Y_current, _, z_query_hist = aligned
        X_history = X_history.to(device, non_blocking=True)
        Y_current = Y_current.to(device, non_blocking=True)
        z_query_hist = z_query_hist.to(device, non_blocking=True)

        if t < train_size:
            continue

        detector.update_and_check(Y_hat_history, Y_current)

        with torch.no_grad():
            mse_by_expert, mae_by_expert = _expert_candidate_stats(
                model, X_history, Y_current
            )
            oracle_best = mse_by_expert.argmin(dim=0)  # [B, C], includes no-op
            selected_mse = torch.gather(
                mse_by_expert, 0, oracle_best.unsqueeze(0)
            ).squeeze(0)
            selected_mae = torch.gather(
                mae_by_expert, 0, oracle_best.unsqueeze(0)
            ).squeeze(0)

            router_probs, _ = _router_probs_from_history(model, z_query_hist)
            router_top1 = router_probs.argmax(dim=-1)
            topk_k = min(3, router_probs.shape[-1])
            router_top3 = torch.topk(router_probs, k=topk_k, dim=-1).indices

            if router_probs.shape[-1] <= noop_idx:
                p_noop = torch.zeros_like(oracle_best, dtype=router_probs.dtype)
                router_noop = torch.zeros_like(oracle_best, dtype=torch.bool)
            else:
                p_noop = router_probs[..., noop_idx]
                router_noop = router_top1 == noop_idx

            oracle_noop = oracle_best == noop_idx
            top1 = router_top1 == oracle_best
            top3 = (router_top3 == oracle_best.unsqueeze(-1)).any(dim=-1)
            entropy = -(router_probs.clamp_min(1e-12) * router_probs.clamp_min(1e-12).log()).sum(dim=-1)

            if prev_oracle_best is None:
                switch_step = float("nan")
            else:
                switch_tensor = oracle_best != prev_oracle_best
                switch_count += int(switch_tensor.sum().item())
                switch_total += switch_tensor.numel()
                switch_step = switch_tensor.float().mean().item()
            prev_oracle_best = oracle_best.detach().clone()

            flat_labels = oracle_best.reshape(-1).detach().cpu().tolist()
            oracle_counts.update(int(x) for x in flat_labels)

            n_items = oracle_best.numel()
            top1_correct += int(top1.sum().item())
            top3_hit += int(top3.sum().item())
            total_labels += n_items
            oracle_noop_count += int(oracle_noop.sum().item())
            router_noop_count += int(router_noop.sum().item())
            noop_tp += int((oracle_noop & router_noop).sum().item())
            entropy_sum += float(entropy.sum().item())
            entropy_total += entropy.numel()
            p_noop_values.append(p_noop.reshape(-1).detach().cpu())

            mae_sum += selected_mae.mean()
            mse_sum += selected_mse.mean()
            n_aligned += 1

            rows.append({
                "t": t,
                "oracle_noop_ratio": oracle_noop.float().mean().item(),
                "router_noop_ratio": router_noop.float().mean().item(),
                "router_top1_acc": top1.float().mean().item(),
                "router_top3_recall": top3.float().mean().item(),
                "router_entropy": entropy.mean().item(),
                "oracle_label_switch_rate_step": switch_step,
                "p_noop_mean": p_noop.float().mean().item(),
                "p_noop_std": p_noop.float().std(unbiased=False).item(),
                "posterior_oracle_mse_step": selected_mse.mean().item(),
            })

        if t % 500 == 0:
            print(
                f"[Step {t}] top1={rows[-1]['router_top1_acc']:.3f} "
                f"top3={rows[-1]['router_top3_recall']:.3f} "
                f"oracle_noop={rows[-1]['oracle_noop_ratio']*100:.1f}% "
                f"router_noop={rows[-1]['router_noop_ratio']*100:.1f}%"
            )

    p_noop_all = torch.cat(p_noop_values) if p_noop_values else torch.empty(0)
    p_noop_stats = _quantiles(p_noop_all)
    oracle_distribution = {
        str(i): oracle_counts.get(i, 0)
        for i in range(num_experts + 1)
    }

    router_top1_acc = top1_correct / max(total_labels, 1)
    router_top3_recall = top3_hit / max(total_labels, 1)
    oracle_label_switch_rate = switch_count / max(switch_total, 1)
    router_entropy = entropy_sum / max(entropy_total, 1)
    oracle_noop_ratio = oracle_noop_count / max(total_labels, 1)
    router_noop_ratio = router_noop_count / max(total_labels, 1)
    no_op_precision = noop_tp / max(router_noop_count, 1)
    no_op_recall = noop_tp / max(oracle_noop_count, 1)
    mse_val = (mse_sum / max(n_aligned, 1)).item()

    output_tag = experiment_tag
    if output_tag.endswith("_router_learnability"):
        output_tag = output_tag[: -len("_router_learnability")]

    summary = {
        "experiment_tag": experiment_tag,
        "output_tag": output_tag,
        "n_aligned": n_aligned,
        "num_experts": num_experts,
        "noop_idx": noop_idx,
        "oracle_best_expert_distribution": oracle_distribution,
        "oracle_label_switch_rate": oracle_label_switch_rate,
        "router_top1_acc": router_top1_acc,
        "router_top3_recall": router_top3_recall,
        "router_entropy": router_entropy,
        "oracle_no_op_ratio": oracle_noop_ratio,
        "router_no_op_ratio": router_noop_ratio,
        "oracle_noop_ratio": oracle_noop_ratio,
        "router_noop_ratio": router_noop_ratio,
        "no_op_precision": no_op_precision,
        "no_op_recall": no_op_recall,
        **p_noop_stats,
        "posterior_oracle_MSE": mse_val,
        "posterior_oracle_RMSE": mse_val ** 0.5,
        "posterior_oracle_MAE": (mae_sum / max(n_aligned, 1)).item(),
    }

    tag = _safe_tag(output_tag)
    json_path = os.path.join(output_dir, f"router_learnability_{tag}.json")
    csv_path = os.path.join(output_dir, f"router_learnability_{tag}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "t", "oracle_noop_ratio", "router_noop_ratio",
            "router_top1_acc", "router_top3_recall", "router_entropy",
            "oracle_label_switch_rate_step", "p_noop_mean", "p_noop_std",
            "posterior_oracle_mse_step",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[*] Router diagnostics saved: {json_path}")
    print(f"[*] Router diagnostics saved: {csv_path}")
    print(
        "[*] Router diagnostics summary: "
        f"top1={router_top1_acc:.4f} top3={router_top3_recall:.4f} "
        f"switch={oracle_label_switch_rate:.4f} entropy={router_entropy:.4f}"
    )
    print(
        "[*] no-op: "
        f"oracle={oracle_noop_ratio*100:.2f}% router={router_noop_ratio*100:.2f}% "
        f"precision={no_op_precision:.4f} recall={no_op_recall:.4f}"
    )
    print(
        "[*] p_noop: "
        f"mean={p_noop_stats['p_noop_mean']:.4f} "
        f"std={p_noop_stats['p_noop_std']:.4f} "
        f"p10={p_noop_stats['p_noop_p10']:.4f} "
        f"p50={p_noop_stats['p_noop_p50']:.4f} "
        f"p90={p_noop_stats['p_noop_p90']:.4f}"
    )

    if n_aligned == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan"), "n": 0}
    return {
        "MAE": summary["posterior_oracle_MAE"],
        "MSE": summary["posterior_oracle_MSE"],
        "RMSE": summary["posterior_oracle_RMSE"],
        "n": n_aligned,
        "total_aligned_steps": n_aligned,
        "update_triggered_steps": 0,
        "avg_channel_update_ratio": 0.0,
        "router_top1_acc": router_top1_acc,
        "router_top3_recall": router_top3_recall,
        "oracle_label_switch_rate": oracle_label_switch_rate,
    }
