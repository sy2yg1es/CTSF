"""
Oracle Streaming Experiments
============================
四个 oracle 实验，用于诊断系统的性能上界。

oracle_detector:
    只有当本次 update 能让下一步 loss 下降时才保留更新，否则回滚。
    上界：如果漂移检测完全正确，最多能提升多少？

oracle_channel:
    对每个通道独立尝试更新，只保留让下一步 loss 下降的通道。
    上界：完美的 channel selection 能提升多少？

oracle_routing:
    对所有 32 个 expert 暴力枚举，选让当前 loss 最小的 top-k 组合。
    上界：完美的 routing 能提升多少？

segment_adapt:
    把 test set 切成若干 segment，每段前半段做离线 SGD，后半段评估。
    上界：如果允许分段适应，数据集本身有多少可适应空间？

实现原则：
  - 所有 oracle 模式与正式 streaming 评估使用相同的 test-then-train 延迟对齐
  - oracle 操作仅发生在 Step 5（update phase），Step 4（metrics）永远是 pre-update
  - oracle 实验不影响 streaming_loop.py 的正式代码
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple, List
from collections import deque

from models.framework import ContinualPromptTSF
from core.buffer import BDLABuffer
from core.drift_detector import ActualDriftDetector


# ==============================================================================
# Shared helpers
# ==============================================================================

def _do_update(
    model: ContinualPromptTSF,
    optimizer: torch.optim.Optimizer,
    X_history: Tensor,
    Y_current: Tensor,
    dispatch_indices_hist: Tensor,
    z_query_hist: Tensor,
    update_mask: Tensor,
    l_aux_weight: float,
) -> float:
    """Single gradient step. Returns loss_task scalar."""
    model.train()
    optimizer.zero_grad()

    Y_hat_update, _, routing_probs_hist = model.forward_update(
        X_history, dispatch_indices_hist, z_query_hist,
    )

    sq_err = (Y_hat_update - Y_current) ** 2
    mse_per_ch = sq_err.mean(dim=1)
    masked_mse = mse_per_ch * update_mask
    n_active = update_mask.sum().clamp(min=1.0)
    loss_task = masked_mse.sum() / n_active

    loss_aux = model.prompt_memory.compute_load_balancing_loss(
        routing_probs=routing_probs_hist,
        drift_mask=update_mask,
        alpha=l_aux_weight,
    )

    (loss_task + loss_aux).backward()
    optimizer.step()
    return loss_task.item()


def _eval_loss(
    model: ContinualPromptTSF,
    X: Tensor,
    Y: Tensor,
) -> float:
    """Compute MSE of model(X) vs Y without updating."""
    model.eval()
    with torch.no_grad():
        Y_hat, _, _, _ = model(X)
        return (Y_hat - Y).pow(2).mean().item()


def _snapshot(model: ContinualPromptTSF) -> dict:
    """Deep copy of trainable parameters for rollback."""
    return {
        n: p.data.clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def _restore(model: ContinualPromptTSF, snap: dict) -> None:
    """Restore trainable parameters from snapshot."""
    for n, p in model.named_parameters():
        if n in snap:
            p.data.copy_(snap[n])


@torch.no_grad()
def _expert_candidate_stats(
    model: ContinualPromptTSF,
    X_history: Tensor,
    Y_current: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Evaluate every real expert plus the synthetic no-op/frozen expert.

    Returns
    -------
    mse_by_expert : [E+1, B, C]
    mae_by_expert : [E+1, B, C]
        Last row is no-op, implemented as backbone_adapter.forward_frozen().
    """
    H_tokens, _, means, stdev = model.backbone_adapter.encode(X_history)
    H_tokens = H_tokens.detach()
    means = means.detach()
    stdev = stdev.detach()

    B = X_history.shape[0]
    C = Y_current.shape[-1]
    prompts = model.prompt_memory.prompts
    E, D = prompts.shape

    mse_list: List[Tensor] = []
    mae_list: List[Tensor] = []

    for e_idx in range(E):
        theta_e = prompts[e_idx].view(1, 1, D).expand(B, C, D)
        Y_hat_e = model.backbone_adapter.fuse_and_decode(
            H_tokens, theta_e, means, stdev
        )
        diff_e = Y_hat_e - Y_current
        mse_list.append(diff_e.pow(2).mean(dim=1))
        mae_list.append(diff_e.abs().mean(dim=1))

    if hasattr(model.backbone_adapter, "forward_frozen"):
        Y_hat_noop = model.backbone_adapter.forward_frozen(X_history)
    else:
        theta_zero = torch.zeros(B, C, D, device=X_history.device)
        Y_hat_noop = model.backbone_adapter.fuse_and_decode(
            H_tokens, theta_zero, means, stdev
        )
    diff_noop = Y_hat_noop - Y_current
    mse_list.append(diff_noop.pow(2).mean(dim=1))
    mae_list.append(diff_noop.abs().mean(dim=1))

    return torch.stack(mse_list, dim=0), torch.stack(mae_list, dim=0)


# ==============================================================================
# Oracle 1: Oracle Detector
# ==============================================================================

def run_oracle_detector(
    model: ContinualPromptTSF,
    dataloader,
    buffer: BDLABuffer,
    detector: ActualDriftDetector,
    optimizer: torch.optim.Optimizer,
    l_aux_weight: float = 0.0,
    train_size: int = 0,
    experiment_tag: str = 'oracle_detector',
) -> Dict[str, float]:
    """
    Oracle Detector: lookahead decides whether to keep update.

    At each aligned step t:
      1. Snapshot model state.
      2. Do update (all channels, no mask).
      3. Compute loss on CURRENT step after update.
      4. If loss improved vs pre-update → keep update.
         Else → roll back to snapshot.

    Note: uses CURRENT step loss as proxy (true lookahead needs t+1 label,
    which isn't available yet). This is an optimistic oracle.
    """
    device = next(model.parameters()).device
    mae_sum = torch.zeros(1, device=device)
    mse_sum = torch.zeros(1, device=device)
    n_aligned = 0
    total_updates = 0
    kept_updates = 0
    channel_ratio_sum = 0.0

    # Cold start
    with torch.no_grad():
        _ = model.prompt_memory.retrieve_prompt(
            torch.zeros(1, 1, model.prompt_memory.prompt_dim, device=device)
        )

    print(f"[*] Oracle mode: oracle_detector")

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

        X_history, Y_current, dispatch_indices_hist, z_query_hist = aligned
        X_history = X_history.to(device, non_blocking=True)
        Y_current = Y_current.to(device, non_blocking=True)
        dispatch_indices_hist = dispatch_indices_hist.to(device, non_blocking=True)
        z_query_hist = z_query_hist.to(device, non_blocking=True)

        if t < train_size:
            continue

        # Drift detection (for logging only in oracle mode)
        drift_mask = detector.update_and_check(Y_hat_history, Y_current)
        drift_mask = drift_mask.to(device, non_blocking=True)

        # Metric (pre-update)
        with torch.no_grad():
            diff = Y_hat_history - Y_current
            mae_sum += diff.abs().mean()
            mse_sum += diff.pow(2).mean()
            n_aligned += 1

        # Oracle: try update on ALL channels, keep if it helps
        all_channels_mask = torch.ones_like(drift_mask)

        # Pre-update loss (on current step)
        pre_loss = (Y_hat_history - Y_current).pow(2).mean().item()

        # Snapshot
        snap = _snapshot(model)

        # Try update
        _do_update(
            model, optimizer,
            X_history, Y_current,
            dispatch_indices_hist, z_query_hist,
            all_channels_mask, l_aux_weight,
        )
        total_updates += 1

        # Post-update loss on same step
        with torch.no_grad():
            Y_hat_post, _, _, _ = model(X_history)
            # align shapes: Y_hat_post [B, pred_len, C], Y_current [B, pred_len, C]
            post_loss = (Y_hat_post - Y_current).pow(2).mean().item()

        if post_loss >= pre_loss:
            # Rollback
            _restore(model, snap)
        else:
            kept_updates += 1
            channel_ratio_sum += 1.0  # all channels were attempted

        if t % 500 == 0:
            keep_rate = kept_updates / max(total_updates, 1) * 100
            print(f"[Step {t}] tried={total_updates} kept={kept_updates} ({keep_rate:.1f}%)")

    avg_channel_ratio = channel_ratio_sum / max(kept_updates, 1)
    print(f"[*] Oracle Detector Stats: tried={total_updates} kept={kept_updates} "
          f"({kept_updates/max(total_updates,1)*100:.1f}% kept) | avg_ch_ratio={avg_channel_ratio*100:.1f}%")

    if n_aligned == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan"), "n": 0}
    mse_val = (mse_sum / n_aligned).item()
    return {
        "MAE": (mae_sum / n_aligned).item(),
        "MSE": mse_val,
        "RMSE": mse_val ** 0.5,
        "n": n_aligned,
        "total_aligned_steps": n_aligned,
        "update_triggered_steps": kept_updates,
        "avg_channel_update_ratio": avg_channel_ratio,
    }


# ==============================================================================
# Oracle 2: Oracle Channel Mask
# ==============================================================================

def run_oracle_channel(
    model: ContinualPromptTSF,
    dataloader,
    buffer: BDLABuffer,
    detector: ActualDriftDetector,
    optimizer: torch.optim.Optimizer,
    l_aux_weight: float = 0.0,
    train_size: int = 0,
    experiment_tag: str = 'oracle_channel',
) -> Dict[str, float]:
    """
    Oracle Channel Mask: per-channel lookahead update.

    For each channel c:
      1. Snapshot → try update on channel c only → evaluate loss.
      2. If better → keep, else rollback.
      3. The final update uses only the channels that passed.

    Approximation: we evaluate channel benefit independently (ignoring
    interaction between channels). True oracle would be exponential.
    """
    device = next(model.parameters()).device
    mae_sum = torch.zeros(1, device=device)
    mse_sum = torch.zeros(1, device=device)
    n_aligned = 0
    total_updates = 0
    channel_ratio_sum = 0.0
    C = None

    with torch.no_grad():
        _ = model.prompt_memory.retrieve_prompt(
            torch.zeros(1, 1, model.prompt_memory.prompt_dim, device=device)
        )

    print(f"[*] Oracle mode: oracle_channel")

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

        X_history, Y_current, dispatch_indices_hist, z_query_hist = aligned
        X_history = X_history.to(device, non_blocking=True)
        Y_current = Y_current.to(device, non_blocking=True)
        dispatch_indices_hist = dispatch_indices_hist.to(device, non_blocking=True)
        z_query_hist = z_query_hist.to(device, non_blocking=True)

        if t < train_size:
            continue

        detector.update_and_check(Y_hat_history, Y_current)

        with torch.no_grad():
            diff = Y_hat_history - Y_current
            mae_sum += diff.abs().mean()
            mse_sum += diff.pow(2).mean()
            n_aligned += 1

        B, pred_len, C_num = Y_current.shape
        if C is None:
            C = C_num

        pre_loss = (Y_hat_history - Y_current).pow(2).mean(dim=1)  # [B, C]

        # Per-channel oracle: find which channels benefit from update
        beneficial_mask = torch.zeros(B, C_num, device=device)

        # Do a full-channel update first (cheapest proxy)
        snap_base = _snapshot(model)
        _do_update(model, optimizer, X_history, Y_current,
                   dispatch_indices_hist, z_query_hist,
                   torch.ones(B, C_num, device=device), l_aux_weight)

        with torch.no_grad():
            Y_hat_post, _, _, _ = model(X_history)
            post_loss = (Y_hat_post - Y_current).pow(2).mean(dim=1)  # [B, C]

        # Channel is beneficial if its loss decreased
        beneficial_mask = (post_loss < pre_loss).float()
        _restore(model, snap_base)

        # If any channel benefits, do the real update with beneficial mask only
        if beneficial_mask.sum() > 0:
            _do_update(model, optimizer, X_history, Y_current,
                       dispatch_indices_hist, z_query_hist,
                       beneficial_mask, l_aux_weight)
            total_updates += 1
            ch_ratio = beneficial_mask.sum().item() / C_num
            channel_ratio_sum += ch_ratio

            if t % 500 == 0:
                print(f"[Step {t}] beneficial_ch={int(beneficial_mask.sum().item())}/{C_num} "
                      f"({ch_ratio*100:.1f}%)")

    avg_channel_ratio = channel_ratio_sum / max(total_updates, 1)
    print(f"[*] Oracle Channel Stats: update_steps={total_updates}/{n_aligned} "
          f"avg_ch_ratio={avg_channel_ratio*100:.1f}%")

    if n_aligned == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan"), "n": 0}
    mse_val = (mse_sum / n_aligned).item()
    return {
        "MAE": (mae_sum / n_aligned).item(),
        "MSE": mse_val,
        "RMSE": mse_val ** 0.5,
        "n": n_aligned,
        "total_aligned_steps": n_aligned,
        "update_triggered_steps": total_updates,
        "avg_channel_update_ratio": avg_channel_ratio,
    }


# ==============================================================================
# Oracle 3: Oracle Routing (brute-force expert selection)
# ==============================================================================

def run_oracle_routing(
    model: ContinualPromptTSF,
    dataloader,
    buffer: BDLABuffer,
    detector: ActualDriftDetector,
    optimizer: torch.optim.Optimizer,
    l_aux_weight: float = 0.0,
    train_size: int = 0,
    experiment_tag: str = 'oracle_routing',
) -> Dict[str, float]:
    """
    Oracle Routing: inference-only, brute-force expert selection.

    At each step, instead of the learned router, try each expert and
    pick the one that minimizes MSE on Y_current (the label available
    via delayed alignment). No parameter updates — pure routing oracle.

    Answers: if routing were perfect, how much better would frozen+oracle_routing be?
    """
    device = next(model.parameters()).device
    mae_sum = torch.zeros(1, device=device)
    mse_sum = torch.zeros(1, device=device)
    n_aligned = 0
    noop_count = 0
    total_labels = 0

    num_experts = model.prompt_memory.prompts.shape[0]
    noop_idx = num_experts
    top_k = model.prompt_memory.top_k

    with torch.no_grad():
        _ = model.prompt_memory.retrieve_prompt(
            torch.zeros(1, 1, model.prompt_memory.prompt_dim, device=device)
        )

    print(f"[*] Oracle mode: posterior_oracle_routing | E={num_experts} top_k={top_k} noop_idx={noop_idx}")
    print(f"[*] NOTE: oracle_routing is inference-only (no parameter updates)")

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

        X_history, Y_current, dispatch_indices_hist, z_query_hist = aligned
        X_history = X_history.to(device, non_blocking=True)
        Y_current = Y_current.to(device, non_blocking=True)
        dispatch_indices_hist = dispatch_indices_hist.to(device, non_blocking=True)
        z_query_hist = z_query_hist.to(device, non_blocking=True)

        if t < train_size:
            continue

        detector.update_and_check(Y_hat_history, Y_current)

        # Posterior oracle routing: try each expert plus no-op, pick best.
        with torch.no_grad():
            mse_by_expert, mae_by_expert = _expert_candidate_stats(
                model, X_history, Y_current
            )  # [E+1, B, C]
            best_idx = mse_by_expert.argmin(dim=0)  # [B, C]
            selected_mse = torch.gather(
                mse_by_expert, 0, best_idx.unsqueeze(0)
            ).squeeze(0)
            selected_mae = torch.gather(
                mae_by_expert, 0, best_idx.unsqueeze(0)
            ).squeeze(0)

            mae_sum += selected_mae.mean()
            mse_sum += selected_mse.mean()
            n_aligned += 1
            noop_count += int((best_idx == noop_idx).sum().item())
            total_labels += best_idx.numel()

        if t % 500 == 0:
            print(f"[Step {t}] oracle_routing running...")

    oracle_noop_ratio = noop_count / max(total_labels, 1)
    print(f"[*] Oracle Routing: n_aligned={n_aligned}")
    print(f"[*] Oracle no-op ratio: {oracle_noop_ratio*100:.2f}% ({noop_count}/{total_labels})")

    if n_aligned == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan"), "n": 0}
    mse_val = (mse_sum / n_aligned).item()
    return {
        "MAE": (mae_sum / n_aligned).item(),
        "MSE": mse_val,
        "RMSE": mse_val ** 0.5,
        "n": n_aligned,
        "total_aligned_steps": n_aligned,
        "update_triggered_steps": 0,
        "avg_channel_update_ratio": 0.0,
        "oracle_noop_ratio": oracle_noop_ratio,
    }


def run_causal_oracle_routing(
    model: ContinualPromptTSF,
    dataloader,
    buffer: BDLABuffer,
    detector: ActualDriftDetector,
    optimizer: torch.optim.Optimizer,
    l_aux_weight: float = 0.0,
    train_size: int = 0,
    experiment_tag: str = 'causal_oracle_routing',
    history_K: int = 12,
) -> Dict[str, float]:
    """
    Causal Oracle Routing: choose expert from past per-expert losses only.

    At aligned historical step s, the selected expert is:
        argmin_k mean(loss_k over previous K aligned historical steps)
    The current label is used only after selection, to update the history.
    """
    assert history_K > 0, "history_K must be positive"

    device = next(model.parameters()).device
    mae_sum = torch.zeros(1, device=device)
    mse_sum = torch.zeros(1, device=device)
    n_aligned = 0

    num_experts = model.prompt_memory.prompts.shape[0]
    noop_idx = num_experts
    loss_history: deque = deque(maxlen=history_K)
    noop_count = 0
    total_labels = 0

    with torch.no_grad():
        _ = model.prompt_memory.retrieve_prompt(
            torch.zeros(1, 1, model.prompt_memory.prompt_dim, device=device)
        )

    print(f"[*] Oracle mode: causal_oracle_routing | K={history_K} E={num_experts} noop_idx={noop_idx}")
    print("[*] NOTE: causal oracle uses only previous aligned-step expert losses")

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

        X_history, Y_current, _, _ = aligned
        X_history = X_history.to(device, non_blocking=True)
        Y_current = Y_current.to(device, non_blocking=True)

        if t < train_size:
            continue

        detector.update_and_check(Y_hat_history, Y_current)

        with torch.no_grad():
            mse_by_expert, mae_by_expert = _expert_candidate_stats(
                model, X_history, Y_current
            )  # [E+1, B, C]

            if loss_history:
                hist_losses = torch.stack(list(loss_history), dim=0)  # [T, E+1, B, C]
                avg_hist_loss = hist_losses.mean(dim=0)               # [E+1, B, C]
                selected_idx = avg_hist_loss.argmin(dim=0)            # [B, C]
            else:
                selected_idx = torch.full_like(mse_by_expert[0].long(), noop_idx)

            selected_mse = torch.gather(
                mse_by_expert, 0, selected_idx.unsqueeze(0)
            ).squeeze(0)
            selected_mae = torch.gather(
                mae_by_expert, 0, selected_idx.unsqueeze(0)
            ).squeeze(0)

            mae_sum += selected_mae.mean()
            mse_sum += selected_mse.mean()
            n_aligned += 1
            noop_count += int((selected_idx == noop_idx).sum().item())
            total_labels += selected_idx.numel()

            # Current labels become causal evidence only after the prediction.
            loss_history.append(mse_by_expert.detach())

        if t % 500 == 0:
            noop_ratio_so_far = noop_count / max(total_labels, 1) * 100
            print(f"[Step {t}] causal_oracle_K{history_K} running... noop={noop_ratio_so_far:.2f}%")

    oracle_noop_ratio = noop_count / max(total_labels, 1)
    print(f"[*] Causal Oracle Routing K={history_K}: n_aligned={n_aligned}")
    print(f"[*] Causal oracle no-op ratio: {oracle_noop_ratio*100:.2f}% ({noop_count}/{total_labels})")

    if n_aligned == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan"), "n": 0}
    mse_val = (mse_sum / n_aligned).item()
    return {
        "MAE": (mae_sum / n_aligned).item(),
        "MSE": mse_val,
        "RMSE": mse_val ** 0.5,
        "n": n_aligned,
        "total_aligned_steps": n_aligned,
        "update_triggered_steps": 0,
        "avg_channel_update_ratio": 0.0,
        "oracle_noop_ratio": oracle_noop_ratio,
        "causal_oracle_K": history_K,
    }


# ==============================================================================
# Oracle 4: Segment-wise Offline Adaptation
# ==============================================================================

def run_segment_adapt(
    model: ContinualPromptTSF,
    dataloader,
    buffer: BDLABuffer,
    detector: ActualDriftDetector,
    optimizer: torch.optim.Optimizer,
    l_aux_weight: float = 0.0,
    train_size: int = 0,
    segment_size: int = 500,
    adapt_steps: int = 10,
    experiment_tag: str = 'segment_adapt',
) -> Dict[str, float]:
    """
    Segment-wise Offline Adaptation.

    Split test stream into segments of `segment_size` aligned steps.
    For each segment:
      1. Collect all (X_history, Y_current) pairs.
      2. Do `adapt_steps` passes of SGD on the first half of the segment.
      3. Evaluate on the second half.

    Answers: if we could batch-adapt per segment, how much would we gain?
    """
    device = next(model.parameters()).device
    mae_sum = torch.zeros(1, device=device)
    mse_sum = torch.zeros(1, device=device)
    n_aligned = 0
    total_updates = 0

    with torch.no_grad():
        _ = model.prompt_memory.retrieve_prompt(
            torch.zeros(1, 1, model.prompt_memory.prompt_dim, device=device)
        )

    print(f"[*] Oracle mode: segment_adapt | seg={segment_size} adapt_steps={adapt_steps}")

    # Collect all aligned samples first
    all_samples = []

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

        aligned = buffer.pop_and_align(t, Y_t)
        if aligned is None:
            continue

        X_history, Y_current, dispatch_indices_hist, z_query_hist = aligned

        if t < train_size:
            continue

        all_samples.append((
            t,
            X_history.to(device),
            Y_current.to(device),
            dispatch_indices_hist.to(device),
            z_query_hist.to(device),
        ))

    print(f"[*] Collected {len(all_samples)} aligned samples. Processing segments...")

    # Save initial model state
    initial_snap = _snapshot(model)

    # Process segments
    for seg_start in range(0, len(all_samples), segment_size):
        seg = all_samples[seg_start: seg_start + segment_size]
        if len(seg) < 4:
            continue

        split = len(seg) // 2
        adapt_set = seg[:split]
        eval_set  = seg[split:]

        # Restore model to initial state for each segment (fresh adaptation)
        _restore(model, initial_snap)

        # Adapt on first half: multiple passes
        all_channels_mask = None
        for _ in range(adapt_steps):
            for _, X_h, Y_c, d_hist, z_hist in adapt_set:
                if all_channels_mask is None or all_channels_mask.shape[0] != X_h.shape[0]:
                    all_channels_mask = torch.ones(X_h.shape[0], Y_c.shape[2], device=device)
                _do_update(model, optimizer, X_h, Y_c, d_hist, z_hist,
                           all_channels_mask, l_aux_weight)
                total_updates += 1

        # Evaluate on second half
        model.eval()
        with torch.no_grad():
            for _, X_h, Y_c, d_hist, z_hist in eval_set:
                Y_hat, _, _, _ = model(X_h)
                diff = Y_hat - Y_c
                mae_sum += diff.abs().mean()
                mse_sum += diff.pow(2).mean()
                n_aligned += 1

        print(f"[Seg {seg_start//segment_size}] adapt={len(adapt_set)} eval={len(eval_set)}")

    print(f"[*] Segment Adapt: n_eval={n_aligned} total_updates={total_updates}")

    if n_aligned == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan"), "n": 0}
    mse_val = (mse_sum / n_aligned).item()
    return {
        "MAE": (mae_sum / n_aligned).item(),
        "MSE": mse_val,
        "RMSE": mse_val ** 0.5,
        "n": n_aligned,
        "total_aligned_steps": n_aligned,
        "update_triggered_steps": total_updates,
        "avg_channel_update_ratio": 1.0,
    }
