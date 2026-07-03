"""
Online Time Series Forecasting — Streaming Evaluation & Update Engine
======================================================================
run_streaming_eval  (v4 — Unified 3-Mode Evaluation)
-----------------------------------------------------

Three streaming modes (controlled by `streaming_mode` parameter):
  'frozen'   — Streaming-Frozen baseline. Identical time-stepping and
               delay-alignment as other modes, but ZERO parameter updates.
               This is the proper anchor for measuring online adaptation.
  'full_ft'  — Full Fine-Tuning (GD baseline). Every aligned step triggers
               a gradient update on ALL channels (no drift mask filtering).
  'ours'     — Channel-Independent selective update. Only drifting channels
               receive gradient signal (drift_mask gating).

All three modes share the EXACT same:
  • Time-step progression
  • BDLA delay alignment (predict at t, label arrives at t+H)
  • Metric accumulation (pre-update frozen predictions)
  • train_size skip logic

Pipeline overview (per timestep t)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Step 1 — Inference: model(X_t) → Y_hat_future
  Step 2 — Buffer Push
  Step 3 — Delayed Alignment: pop_and_align(t, Y_t)
  Step 4 — Metric accumulation (BEFORE any update)
  Step 5 — Update (mode-dependent):
           frozen:  skip
           full_ft: update all channels
           ours:    update drifting channels only
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple
from collections import deque
import json

from models.framework import ContinualPromptTSF
from core.buffer import BDLABuffer
from core.drift_detector import ActualDriftDetector


def run_streaming_eval(
    model: ContinualPromptTSF,
    dataloader,
    buffer: BDLABuffer,
    detector: ActualDriftDetector,
    optimizer: torch.optim.Optimizer,
    l_aux_weight: float = 1e-3,
    train_size: int = 0,
    experiment_tag: str = '',
    streaming_mode: str = 'ours',
) -> Dict[str, float]:
    """
    Execute the streaming pipeline over every timestep in dataloader.

    Parameters
    ----------
    streaming_mode : str, one of {'frozen', 'full_ft', 'ours'}
        'frozen'  — Zero updates (Streaming-Frozen baseline).
        'full_ft' — Online GD on all channels every step.
        'ours'    — Selective channel-wise update (drift-masked).
    """
    assert streaming_mode in ('frozen', 'full_ft', 'ours'), \
        f"Invalid streaming_mode: {streaming_mode}. Must be 'frozen', 'full_ft', or 'ours'."

    print(f"[*] Streaming mode: {streaming_mode}")

    # ------------------------------------------------------------------
    # Metric accumulators
    # ------------------------------------------------------------------
    device = next(model.parameters()).device
    mae_sum = torch.zeros(1, device=device)
    mse_sum = torch.zeros(1, device=device)
    n_aligned: int = 0

    # ------------------------------------------------------------------
    # Cold-Start Guard
    # ------------------------------------------------------------------
    with torch.no_grad():
        _ = model.prompt_memory.retrieve_prompt(
            torch.zeros(1, 1, model.prompt_memory.prompt_dim, device=device)
        )

    total_updates = 0
    channel_ratio_sum = 0.0   # accumulate update_mask.sum()/C per update step
    router_noop_count = 0
    router_noop_total = 0
    p_noop_values = []

    # ==========================================================
    # Monitoring State
    # ==========================================================
    loss_window: deque = deque(maxlen=50)
    _prev_prompt_params = None
    _prev_adapter_params = None
    monitor_log = []

    # ==================================================================
    # Main streaming loop
    # ==================================================================
    for t, (X_t, Y_t) in enumerate(dataloader):

        # ==============================================================
        # STEP 1 — Inference
        # ==============================================================
        model.eval()
        with torch.no_grad():
            X_t = X_t.to(device, non_blocking=True)
            Y_hat_future, z_channel, routing_probs, dispatch_indices = model(X_t)

        # ==============================================================
        # STEP 2 — Buffer Push
        # ==============================================================
        buffer.push(
            t=t, X_t=X_t, y_hat_future=Y_hat_future,
            dispatch_indices=dispatch_indices, z_t=z_channel,
        )

        # ==============================================================
        # STEP 3 — Delayed Alignment
        # ==============================================================
        if Y_t is None:
            continue

        Y_hat_history = buffer.get_stored_prediction(t)
        if Y_hat_history is None:
            continue
        Y_hat_history = Y_hat_history.to(device, non_blocking=True)

        aligned: Optional[Tuple[Tensor, ...]] = buffer.pop_and_align(t, Y_t)
        if aligned is None:
            continue

        X_history, Y_current, dispatch_indices_hist, z_query_hist = aligned
        X_history = X_history.to(device, non_blocking=True)
        Y_current = Y_current.to(device, non_blocking=True)
        dispatch_indices_hist = dispatch_indices_hist.to(device, non_blocking=True)
        z_query_hist = z_query_hist.to(device, non_blocking=True)

        # ==============================================================
        # Skip train set
        # ==============================================================
        if t < train_size:
            continue

        # ==============================================================
        # STEP 4 — Drift Detection (used by 'ours' and for logging)
        # ==============================================================
        drift_mask: Tensor = detector.update_and_check(
            Y_hat_history, Y_current
        )
        drift_mask = drift_mask.to(device, non_blocking=True)
        active_count = int(drift_mask.sum().item())

        # Router no-op diagnostics on the historical decision being evaluated.
        noop_idx = getattr(model.prompt_memory, "num_experts", None)
        n_router_out = getattr(model.prompt_memory, "n_router_out", 0)
        if noop_idx is not None and n_router_out > noop_idx:
            with torch.no_grad():
                _, routing_probs_hist_eval, _ = model.prompt_memory.retrieve_prompt(z_query_hist)
                router_top1_hist = routing_probs_hist_eval.argmax(dim=-1)
                p_noop_hist = routing_probs_hist_eval[..., noop_idx]
                router_noop_count += int((router_top1_hist == noop_idx).sum().item())
                router_noop_total += router_top1_hist.numel()
                p_noop_values.append(p_noop_hist.reshape(-1).detach().cpu())

        if t % 500 == 0 or active_count > 10:
            print(f"[Step {t}] Drifting Channels: {active_count} / {drift_mask.shape[1]}")

        # ------------------------------------------------------------------
        # Metric accumulation — BEFORE any update
        # ------------------------------------------------------------------
        with torch.no_grad():
            diff = Y_hat_history - Y_current
            mae_sum += diff.abs().mean()
            mse_sum += diff.pow(2).mean()
            n_aligned += 1

        # ==============================================================
        # STEP 5 — Update (mode-dependent)
        # ==============================================================

        # --- FROZEN: no updates ever ---
        if streaming_mode == 'frozen':
            continue

        # --- Determine update mask ---
        if streaming_mode == 'full_ft':
            # GD baseline: update ALL channels every step
            update_mask = torch.ones_like(drift_mask)
        else:
            # 'ours': only drifting channels
            if drift_mask.sum() == 0:
                continue
            update_mask = drift_mask

        # ==============================================================
        # Gradient Update
        # ==============================================================
        model.train()
        optimizer.zero_grad()

        # Snapshot params for delta tracking
        _prev_prompt_params = {
            n: p.data.clone() for n, p in model.prompt_memory.named_parameters()
            if p.requires_grad
        }
        if model.adapter is not None:
            _prev_adapter_params = {
                n: p.data.clone() for n, p in model.adapter.named_parameters()
                if p.requires_grad
            }

        # 5a) Delayed forward pass
        Y_hat_update, _, routing_probs_hist = model.forward_update(
            X_history, dispatch_indices_hist, z_query_hist,
        )

        # 5b) Channel-wise masked MSE
        sq_err_update = (Y_hat_update - Y_current) ** 2
        mse_per_channel = sq_err_update.mean(dim=1)           # [B, C]
        masked_mse = mse_per_channel * update_mask            # apply mode-specific mask
        n_active = update_mask.sum().clamp(min=1.0)
        loss_task = masked_mse.sum() / n_active

        # 5c) Aux loss
        loss_aux = model.prompt_memory.compute_load_balancing_loss(
            routing_probs=routing_probs_hist,
            drift_mask=update_mask,
            alpha=l_aux_weight,
        )

        # 5d) Backward + step
        loss_total = loss_task + loss_aux
        loss_total.backward()

        # Gradient Norm
        grad_norm_prompt = 0.0
        for p in model.prompt_memory.parameters():
            if p.grad is not None:
                grad_norm_prompt += p.grad.data.norm(2).item() ** 2
        grad_norm_prompt = grad_norm_prompt ** 0.5

        grad_norm_adapter = 0.0
        if model.adapter is not None:
            for p in model.adapter.parameters():
                if p.grad is not None:
                    grad_norm_adapter += p.grad.data.norm(2).item() ** 2
            grad_norm_adapter = grad_norm_adapter ** 0.5

        optimizer.step()
        total_updates += 1

        # Param Delta
        param_delta_prompt = 0.0
        if _prev_prompt_params:
            for n, p in model.prompt_memory.named_parameters():
                if n in _prev_prompt_params:
                    param_delta_prompt += (p.data - _prev_prompt_params[n]).norm().item() ** 2
            param_delta_prompt = param_delta_prompt ** 0.5

        param_delta_adapter = 0.0
        if model.adapter is not None and _prev_adapter_params:
            for n, p in model.adapter.named_parameters():
                if n in _prev_adapter_params:
                    param_delta_adapter += (p.data - _prev_adapter_params[n]).norm().item() ** 2
            param_delta_adapter = param_delta_adapter ** 0.5

        # Online Loss Variance
        loss_val = loss_task.item()
        loss_window.append(loss_val)
        if len(loss_window) >= 10:
            loss_list = list(loss_window)
            loss_mean = sum(loss_list) / len(loss_list)
            loss_var = sum((x - loss_mean) ** 2 for x in loss_list) / len(loss_list)
        else:
            loss_var = 0.0

        # Store monitoring record
        monitor_log.append({
            'update': total_updates,
            'step': t,
            'loss_task': loss_val,
            'loss_var_50': loss_var,
            'grad_norm_prompt': grad_norm_prompt,
            'grad_norm_adapter': grad_norm_adapter,
            'param_delta_prompt': param_delta_prompt,
            'param_delta_adapter': param_delta_adapter,
            'drifting_channels': active_count,
            'updating_channels': int(update_mask.sum().item()),
        })

        if total_updates % 50 == 0:
            print(f"[Update {total_updates}] loss={loss_val:.6f} var50={loss_var:.6f} "
                  f"grad_prompt={grad_norm_prompt:.4f} grad_adapter={grad_norm_adapter:.4f} "
                  f"delta_prompt={param_delta_prompt:.6f} delta_adapter={param_delta_adapter:.6f} "
                  f"drifting={active_count} updating={int(update_mask.sum().item())}")

        model.prompt_memory.update_usage(dispatch_indices_hist)

        # Track channel update ratio for this step
        total_channels = update_mask.shape[1]
        channel_ratio_sum += update_mask.sum().item() / total_channels

    # ==================================================================
    # Aggregate and return metrics
    # ==================================================================
    avg_channel_ratio = channel_ratio_sum / max(total_updates, 1)

    print(f"[*] === Streaming Stats ===")
    print(f"[*] Total Aligned Steps:      {n_aligned}")
    print(f"[*] Update Triggered Steps:   {total_updates}")
    print(f"[*] Update Ratio:             {total_updates}/{n_aligned} = {total_updates/max(n_aligned,1)*100:.1f}%")
    print(f"[*] Avg Channel Update Ratio: {avg_channel_ratio*100:.1f}%")
    if router_noop_total > 0:
        p_noop_all = torch.cat(p_noop_values) if p_noop_values else torch.empty(0)
        p_noop_q = torch.quantile(
            p_noop_all.float(),
            torch.tensor([0.1, 0.5, 0.9], device=p_noop_all.device),
        )
        print(f"[*] Router no-op ratio:       {router_noop_count/router_noop_total*100:.2f}% "
              f"({router_noop_count}/{router_noop_total})")
        print(f"[*] p_noop: mean={p_noop_all.mean().item():.4f} "
              f"std={p_noop_all.std(unbiased=False).item():.4f} "
              f"p10={p_noop_q[0].item():.4f} "
              f"p50={p_noop_q[1].item():.4f} "
              f"p90={p_noop_q[2].item():.4f}")

    if n_aligned == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MSE": float("nan"),
                "n": 0, "total_updates": 0, "avg_channel_ratio": 0.0}

    # Dump monitoring log
    if monitor_log:
        try:
            import os
            os.makedirs('logs/monitor', exist_ok=True)
            monitor_path = f'logs/monitor/monitor_{experiment_tag}.json' if experiment_tag else f'logs/monitor/streaming_monitor_{total_updates}updates.json'
            with open(monitor_path, 'w') as f:
                json.dump(monitor_log, f)
            print(f"[*] Monitoring log saved to {monitor_path}")
        except Exception as e:
            print(f"[!] Failed to save monitoring log: {e}")

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
