"""
Streaming Prompt-Z Evaluation Engine
======================================

Three modes:
  'mode0'  — PromptZ frozen, only ResidualTracker updates online
  'mode1'  — Mode 0 + gamma bias calibration on delayed labels
  'frozen' — No Prompt-Z at all (frozen backbone baseline)

IMPORTANT: dataloader must be wrapped in StreamingEnvironment before calling.
Each iteration yields (X_t, Y_arrived_or_None).
Y_arrived is the label that just arrived via delay alignment (from H steps ago).
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.prompt_z_framework import PromptZTSF


def _mean_metric(sum_tensor: Tensor, count: int) -> float:
    if count <= 0:
        return float('inf')
    return (sum_tensor / count).item()


def _summarize_diagnostics(diag_accum: Dict[str, list]) -> Dict[str, float]:
    diag_summary = {}
    for k, vals in diag_accum.items():
        if vals:
            t_vals = torch.tensor(vals)
            diag_summary[k + "_mean"] = t_vals.mean().item()
            diag_summary[k + "_std"] = t_vals.std().item()
            if k == "gamma_mean":
                diag_summary["gamma_p10"] = t_vals.quantile(0.1).item()
                diag_summary["gamma_p50"] = t_vals.quantile(0.5).item()
                diag_summary["gamma_p90"] = t_vals.quantile(0.9).item()
    return diag_summary


def _write_results(experiment_tag: str, results: Dict[str, float]) -> None:
    log_dir = os.path.join("logs", "prompt_z")
    os.makedirs(log_dir, exist_ok=True)
    if experiment_tag:
        log_path = os.path.join(log_dir, f"{experiment_tag}.json")
        with open(log_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[*] Diagnostics saved to {log_path}")


def run_prompt_z_streaming(
    model: PromptZTSF,
    dataloader,
    train_size: int = 0,
    mode: str = 'mode0',
    calibration_lr: float = 1e-4,
    experiment_tag: str = '',
) -> Dict[str, float]:
    """
    Run Prompt-Z streaming evaluation.

    Parameters
    ----------
    model : PromptZTSF
    dataloader : StreamingEnvironment — yields (X_t, Y_arrived_or_None)
    train_size : skip first N steps for metric accumulation
    mode : 'frozen', 'mode0', 'mode1'
    calibration_lr : learning rate for mode1 gamma calibration
    experiment_tag : for logging
    """
    assert mode in ('frozen', 'mode0', 'mode1'), \
        f"Invalid mode: {mode}. Must be 'frozen', 'mode0', or 'mode1'."

    print(f"[*] Prompt-Z streaming mode: {mode}")

    device = next(model.parameters()).device

    # Metric accumulators
    mae_sum = torch.zeros(1, device=device)
    mse_sum = torch.zeros(1, device=device)
    n_aligned = 0

    # Cache: store (Y_hat, X_t) for each step so we can use them when label arrives
    y_hat_cache: deque = deque()
    x_cache: deque = deque()

    # Diagnostics accumulators
    diag_accum = {
        "gamma_mean": [], "gamma_std": [], "gamma_min": [], "gamma_max": [],
        "mask_ratio": [], "mask_mean": [],
        "raw_delta_norm": [], "applied_delta_norm": [], "hidden_norm": [],
        "raw_delta_to_hidden_ratio": [], "effective_delta_ratio": [],
    }
    calibration_updates = 0

    # Calibration optimizer (mode1 only)
    calib_optimizer = None
    if mode == 'mode1':
        gate_params = model.prompt_z.get_gate_params()
        calib_optimizer = torch.optim.SGD(gate_params, lr=calibration_lr)

    model.eval()
    model.residual_tracker.reset()

    for t, (X_t, Y_arrived) in enumerate(dataloader):
        X_t = X_t.to(device, non_blocking=True)

        # ==============================================================
        # STEP 1 — Inference
        # ==============================================================
        with torch.no_grad():
            if mode == 'frozen':
                Y_hat = model.forward_frozen(X_t)
                diagnostics = {}
            else:
                Y_hat, hidden, diagnostics = model(X_t)

        # Cache prediction and input for when label arrives
        y_hat_cache.append(Y_hat.detach().cpu())
        x_cache.append(X_t.detach().cpu())

        # ==============================================================
        # STEP 2 — Handle delayed label (if arrived)
        # ==============================================================
        if Y_arrived is None:
            model.residual_tracker.step_no_update()
            continue

        Y_arrived = Y_arrived.to(device, non_blocking=True)

        # Pop the oldest cached prediction (this is the one made H steps ago)
        Y_hat_cached = y_hat_cache.popleft().to(device, non_blocking=True)
        X_cached = x_cache.popleft()

        # ==============================================================
        # STEP 3 — Metric Accumulation (BEFORE any update)
        # ==============================================================
        if t >= train_size:
            with torch.no_grad():
                diff = Y_hat_cached - Y_arrived
                mae_sum += diff.abs().mean()
                mse_sum += diff.pow(2).mean()
                n_aligned += 1

            # Accumulate diagnostics
            for k in diag_accum:
                if k in diagnostics:
                    diag_accum[k].append(diagnostics[k])

        # ==============================================================
        # STEP 4 — Residual Tracker Update (causal: label just arrived)
        # ==============================================================
        model.residual_tracker.update(Y_hat_cached, Y_arrived)

        # ==============================================================
        # STEP 5 — Mode 1: Calibrate gamma bias
        # ==============================================================
        if mode == 'mode1' and t >= train_size and calib_optimizer is not None:
            X_cached_dev = X_cached.to(device, non_blocking=True)

            # Enable grad only for gate params
            for p in model.prompt_z.parameters():
                p.requires_grad_(False)
            for p in model.prompt_z.get_gate_params():
                p.requires_grad_(True)

            calib_optimizer.zero_grad()

            with torch.no_grad():
                hidden_c, means_c, stdev_c = model.backbone_adapter.encode_until_hook(X_cached_dev)
            hidden_c = hidden_c.detach()
            means_c = means_c.detach()
            stdev_c = stdev_c.detach()

            stats = model.residual_tracker.get_stats()
            stats_tensor = model._pack_stats(stats, device)
            hidden_mod, _, _ = model.prompt_z(hidden_c, stats_tensor)
            Y_hat_calib = model.backbone_adapter.decode_from_hook(hidden_mod, means_c, stdev_c)

            calib_loss = nn.functional.mse_loss(Y_hat_calib, Y_arrived)
            calib_loss.backward()
            calib_optimizer.step()
            calibration_updates += 1

            # Restore
            for p in model.prompt_z.parameters():
                p.requires_grad_(False)

        # Logging
        if t % 500 == 0:
            if n_aligned > 0:
                cur_mse = (mse_sum / n_aligned).item()
                gamma_str = f"gamma={diagnostics.get('gamma_mean', 0):.4f}" if diagnostics else ""
                print(f"[Step {t}] MSE={cur_mse:.6f} n={n_aligned} {gamma_str}")

    # ==================================================================
    # Final Results
    # ==================================================================
    if n_aligned == 0:
        print("[!] No aligned steps found!")
        return {"MSE": float('inf'), "MAE": float('inf')}

    final_mae = (mae_sum / n_aligned).item()
    final_mse = (mse_sum / n_aligned).item()
    final_rmse = final_mse ** 0.5

    print(f"[*] === Prompt-Z Streaming Stats ===")
    print(f"[*] Mode: {mode}")
    print(f"[*] Total Aligned Steps: {n_aligned}")
    print(f"[*] Calibration Updates: {calibration_updates}")

    # Diagnostics summary
    diag_summary = _summarize_diagnostics(diag_accum)

    for k, v in diag_summary.items():
        print(f"[*] {k}: {v:.6f}")

    print(f"[*] Streaming Evaluation Completed! MAE: {final_mae:.4f}, "
          f"MSE: {final_mse:.4f}, RMSE: {final_rmse:.4f}")

    results = {
        "MSE": final_mse,
        "MAE": final_mae,
        "RMSE": final_rmse,
        "mode": mode,
        "n_aligned": n_aligned,
        "calibration_updates": calibration_updates,
        **diag_summary,
    }

    _write_results(experiment_tag, results)

    return results


def run_prompt_z_validation_fallback(
    model: PromptZTSF,
    dataloader,
    train_size: int = 0,
    mode: str = 'mode0',
    calibration_lr: float = 1e-4,
    fallback_margin: float = 0.005,
    validation_steps: Optional[int] = None,
    experiment_tag: str = '',
    # Keep for backwards compat but unused in strict-split callers
    validation_ratio: float = 0.1,
) -> Dict[str, float]:
    """
    Run a single streaming pass that compares frozen and Prompt-Z on an
    early validation segment, then reports raw Prompt-Z and selected test MSE.

    Strict split protocol:
        train_size  = window index where offline train ends (60%)
        validation_steps = number of aligned-label steps in validation (10%)
        test starts after validation_steps aligned labels have been consumed

    Selection rule:
        enable Prompt-Z iff val_promptz <= val_frozen * (1 - fallback_margin).
    If disabled, selected test output is the frozen test metric.
    """
    assert mode in ('mode0', 'mode1'), \
        f"Fallback expects a Prompt-Z mode, got {mode}."

    print(f"[*] Prompt-Z validation fallback mode: {mode}")
    print(f"[*] train_size={train_size} "
          f"fallback_margin={fallback_margin:.6f} "
          f"validation_steps={validation_steps}")

    device = next(model.parameters()).device

    # Estimate validation_steps if not provided (backwards compat)
    if validation_steps is None:
        total_steps = None
        try:
            total_steps = len(dataloader.dataloader)
        except Exception:
            pass
        if total_steps is not None:
            first_aligned_t = getattr(dataloader, "forecast_H", 0)
            eligible = max(0, total_steps - max(first_aligned_t, train_size))
            validation_steps = max(1, int(eligible * validation_ratio)) if eligible > 1 else 0
        else:
            validation_steps = 0

    frozen_cache: deque = deque()
    pz_cache: deque = deque()
    x_cache: deque = deque()

    zero = torch.zeros(1, device=device)
    val_frozen_mse_sum = zero.clone()
    val_promptz_mse_sum = zero.clone()
    test_frozen_mse_sum = zero.clone()
    test_promptz_mse_sum = zero.clone()
    test_selected_mse_sum = zero.clone()
    test_selected_mae_sum = zero.clone()
    val_n = 0
    test_n = 0
    eligible_n = 0

    diag_accum = {
        "gamma_mean": [], "gamma_std": [], "gamma_min": [], "gamma_max": [],
        "mask_ratio": [], "mask_mean": [],
        "raw_delta_norm": [], "applied_delta_norm": [], "hidden_norm": [],
        "raw_delta_to_hidden_ratio": [], "effective_delta_ratio": [],
    }

    promptz_enabled: Optional[bool] = None
    calibration_updates = 0

    calib_optimizer = None
    if mode == 'mode1':
        gate_params = model.prompt_z.get_gate_params()
        calib_optimizer = torch.optim.SGD(gate_params, lr=calibration_lr)

    model.eval()
    model.residual_tracker.reset()

    for t, (X_t, Y_arrived) in enumerate(dataloader):
        X_t = X_t.to(device, non_blocking=True)

        with torch.no_grad():
            Y_frozen = model.forward_frozen(X_t)
            Y_promptz, _hidden, diagnostics = model(X_t)

        frozen_cache.append(Y_frozen.detach().cpu())
        pz_cache.append(Y_promptz.detach().cpu())
        x_cache.append(X_t.detach().cpu())

        if Y_arrived is None:
            model.residual_tracker.step_no_update()
            continue

        Y_arrived = Y_arrived.to(device, non_blocking=True)
        Y_frozen_cached = frozen_cache.popleft().to(device, non_blocking=True)
        Y_promptz_cached = pz_cache.popleft().to(device, non_blocking=True)
        X_cached = x_cache.popleft()

        frozen_mse = (Y_frozen_cached - Y_arrived).pow(2).mean()
        promptz_mse = (Y_promptz_cached - Y_arrived).pow(2).mean()

        if t >= train_size:
            eligible_n += 1
            if eligible_n <= validation_steps:
                val_frozen_mse_sum += frozen_mse
                val_promptz_mse_sum += promptz_mse
                val_n += 1
            else:
                if promptz_enabled is None:
                    val_frozen_mse = _mean_metric(val_frozen_mse_sum, val_n)
                    val_promptz_mse = _mean_metric(val_promptz_mse_sum, val_n)
                    promptz_enabled = (
                        val_n > 0 and
                        val_promptz_mse <= val_frozen_mse * (1.0 - fallback_margin)
                    )
                    print(f"[*] Validation decision: "
                          f"val_frozen_mse={val_frozen_mse:.6f} "
                          f"val_promptz_mse={val_promptz_mse:.6f} "
                          f"enabled={promptz_enabled}")

                test_frozen_mse_sum += frozen_mse
                test_promptz_mse_sum += promptz_mse
                if promptz_enabled:
                    test_selected_mse_sum += promptz_mse
                    test_selected_mae_sum += (Y_promptz_cached - Y_arrived).abs().mean()
                else:
                    test_selected_mse_sum += frozen_mse
                    test_selected_mae_sum += (Y_frozen_cached - Y_arrived).abs().mean()
                test_n += 1

                for k in diag_accum:
                    if k in diagnostics:
                        diag_accum[k].append(diagnostics[k])

        # Keep Prompt-Z raw tracker causal and comparable to normal mode0/mode1.
        model.residual_tracker.update(Y_promptz_cached, Y_arrived)

        if mode == 'mode1' and t >= train_size and calib_optimizer is not None:
            X_cached_dev = X_cached.to(device, non_blocking=True)

            for p in model.prompt_z.parameters():
                p.requires_grad_(False)
            for p in model.prompt_z.get_gate_params():
                p.requires_grad_(True)

            calib_optimizer.zero_grad()

            with torch.no_grad():
                hidden_c, means_c, stdev_c = model.backbone_adapter.encode_until_hook(X_cached_dev)
            hidden_c = hidden_c.detach()
            means_c = means_c.detach()
            stdev_c = stdev_c.detach()

            stats = model.residual_tracker.get_stats()
            stats_tensor = model._pack_stats(stats, device)
            hidden_mod, _, _ = model.prompt_z(hidden_c, stats_tensor)
            Y_hat_calib = model.backbone_adapter.decode_from_hook(hidden_mod, means_c, stdev_c)

            calib_loss = nn.functional.mse_loss(Y_hat_calib, Y_arrived)
            calib_loss.backward()
            calib_optimizer.step()
            calibration_updates += 1

            for p in model.prompt_z.parameters():
                p.requires_grad_(False)

        if t % 500 == 0 and test_n > 0:
            print(f"[Step {t}] test_raw_pz={_mean_metric(test_promptz_mse_sum, test_n):.6f} "
                  f"test_selected={_mean_metric(test_selected_mse_sum, test_n):.6f} "
                  f"enabled={promptz_enabled}")

    if promptz_enabled is None:
        val_frozen_mse = _mean_metric(val_frozen_mse_sum, val_n)
        val_promptz_mse = _mean_metric(val_promptz_mse_sum, val_n)
        promptz_enabled = (
            val_n > 0 and
            val_promptz_mse <= val_frozen_mse * (1.0 - fallback_margin)
        )
    else:
        val_frozen_mse = _mean_metric(val_frozen_mse_sum, val_n)
        val_promptz_mse = _mean_metric(val_promptz_mse_sum, val_n)

    test_frozen_mse = _mean_metric(test_frozen_mse_sum, test_n)
    test_promptz_raw_mse = _mean_metric(test_promptz_mse_sum, test_n)
    test_promptz_selected_mse = _mean_metric(test_selected_mse_sum, test_n)
    val_delta_percent = (
        (val_promptz_mse / val_frozen_mse - 1.0) * 100.0
        if val_frozen_mse not in (0.0, float('inf')) else float('inf')
    )

    print("[*] === Prompt-Z Validation Fallback Stats ===")
    print(f"[*] val_frozen_mse: {val_frozen_mse:.6f}")
    print(f"[*] val_promptz_mse: {val_promptz_mse:.6f}")
    print(f"[*] val_delta_percent: {val_delta_percent:.4f}")
    print(f"[*] fallback_margin: {fallback_margin:.6f}")
    print(f"[*] promptz_enabled: {promptz_enabled}")
    print(f"[*] test_frozen_mse: {test_frozen_mse:.6f}")
    print(f"[*] test_promptz_raw_mse: {test_promptz_raw_mse:.6f}")
    print(f"[*] test_promptz_selected_mse: {test_promptz_selected_mse:.6f}")

    diag_summary = _summarize_diagnostics(diag_accum)
    for k, v in diag_summary.items():
        print(f"[*] {k}: {v:.6f}")

    results = {
        "MSE": test_promptz_selected_mse,
        "MAE": _mean_metric(test_selected_mae_sum, test_n),
        "RMSE": test_promptz_selected_mse ** 0.5
                if test_promptz_selected_mse != float('inf') else float('inf'),
        "mode": f"{mode}_selected",
        "validation_fallback": True,
        "fallback_margin": fallback_margin,
        "validation_ratio": validation_ratio,
        "validation_steps": validation_steps,
        "val_n": val_n,
        "test_n": test_n,
        "val_frozen_mse": val_frozen_mse,
        "val_promptz_mse": val_promptz_mse,
        "val_delta_percent": val_delta_percent,
        "promptz_enabled": bool(promptz_enabled),
        "test_frozen_mse": test_frozen_mse,
        "test_promptz_raw_mse": test_promptz_raw_mse,
        "test_promptz_selected_mse": test_promptz_selected_mse,
        "calibration_updates": calibration_updates,
        **diag_summary,
    }

    _write_results(experiment_tag, results)
    return results
