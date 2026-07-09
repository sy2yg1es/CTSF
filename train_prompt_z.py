"""
train_prompt_z.py — Stage A Offline Training for PromptZModulator
==================================================================

Freeze backbone + prediction head. Train PromptZModulator only.
Data is processed sequentially (not shuffled) because ResidualTracker
is stateful and must see data in chronological order.

Loss:
    forecast_loss  = MSE(Y_hat, Y_true)
    delta_reg      = lambda_delta * ||delta_h||_2
    mask_sparsity  = lambda_mask  * ||mask||_1
    noop_margin    = lambda_noop  * gamma.mean()
                     IF MSE_mod >= MSE_frozen - epsilon
                     (penalize gate opening when correction doesn't help)

Usage:
    python train_prompt_z.py --data_path ECL.csv --forecast_H 96 --epochs 3
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
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Project imports
from data_provider.data_loader import data_provider
from models.backbone_adapter import PatchTSTAdapter, iTransformerAdapter
from models.prompt_z import PromptZModulator
from models.prompt_z_framework import PromptZTSF
from core.residual_tracker import ResidualTracker


def build_backbone(args, device):
    """Load pretrained backbone and wrap in adapter."""
    if args.backbone == 'patchtst':
        from models.backbones.PatchTST import Model as PatchTST
        class Cfg:
            pass
        cfg = Cfg()
        cfg.task_name = 'long_term_forecast'
        cfg.seq_len = args.seq_len
        cfg.pred_len = args.forecast_H
        cfg.d_model = args.D_model
        cfg.d_ff = args.d_ff
        cfg.n_heads = 8
        cfg.e_layers = args.e_layers
        cfg.dropout = 0.1
        cfg.activation = 'gelu'
        cfg.factor = 1
        cfg.enc_in = args.enc_in
        backbone = PatchTST(cfg).to(device)
        adapter = PatchTSTAdapter(backbone).to(device)
    elif args.backbone == 'itransformer':
        from models.backbones.iTransformer import Model as iTransformer
        class Cfg:
            pass
        cfg = Cfg()
        cfg.task_name = 'long_term_forecast'
        cfg.seq_len = args.seq_len
        cfg.pred_len = args.forecast_H
        cfg.d_model = args.D_model
        cfg.d_ff = args.d_ff
        cfg.n_heads = 8
        cfg.e_layers = args.e_layers
        cfg.dropout = 0.1
        cfg.activation = 'gelu'
        cfg.factor = 1
        cfg.enc_in = args.enc_in
        cfg.output_attention = False
        cfg.embed = 'timeF'
        cfg.freq = 'h'
        backbone = iTransformer(cfg).to(device)
        adapter = iTransformerAdapter(backbone).to(device)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # Load pretrained weights
    if args.pretrained_weights:
        state = torch.load(args.pretrained_weights, map_location=device)
        if 'backbone_adapter.backbone' in str(list(state.keys())[:3]):
            # Saved from ContinualPromptTSF — extract backbone keys
            prefix = 'backbone_adapter.backbone.'
            backbone_state = {
                k[len(prefix):]: v for k, v in state.items()
                if k.startswith(prefix)
            }
            adapter.backbone.load_state_dict(backbone_state, strict=False)
        else:
            adapter.backbone.load_state_dict(state, strict=False)
        print(f"[*] Loaded pretrained weights from {args.pretrained_weights}")

    # Freeze everything in adapter
    for p in adapter.parameters():
        p.requires_grad = False

    return adapter


def build_model(args, adapter, device):
    """Build PromptZTSF model."""
    prompt_z = PromptZModulator(
        d_model=args.D_model,
        hidden_layout=adapter.hidden_layout,
        d_drift=args.d_drift,
        rank=args.rank,
        gamma_init_bias=args.gamma_init_bias,
        mask_init_bias=args.mask_init_bias,
        max_delta_ratio=args.max_delta_ratio,
    ).to(device)

    residual_tracker = ResidualTracker(
        num_channels=args.enc_in,
        window_K=args.residual_window_K,
    ).to(device)

    model = PromptZTSF(adapter, prompt_z, residual_tracker).to(device)

    # Verify only PromptZ params are trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[*] Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    return model


def get_dataloader(args, train_ratio=0.6, val_ratio=0.1):
    """Get sequential dataloader (no shuffle), limited to train windows only.
    Train window = window whose label falls entirely within the train raw-row zone.
    Determined by dataset.train_size (label-timestamp boundary), not window count.
    """
    class DPArgs:
        pass
    dp_args = DPArgs()
    dp_args.root_path = args.root_path
    dp_args.data_path = args.data_path
    dp_args.features = args.features
    dp_args.seq_len = args.seq_len
    dp_args.pred_len = args.forecast_H
    dp_args.target = 'OT'
    dp_args.num_workers = args.num_workers
    dp_args.train_ratio = train_ratio
    dp_args.val_ratio   = val_ratio

    dataset, _ = data_provider(dp_args)

    # Use label-timestamp boundary from dataset (strict causal)
    train_size = dataset.train_size
    train_subset = Subset(dataset, range(train_size))
    print(f"[*] Train subset: {train_size}/{len(dataset)} windows "
          f"(label fully in train raw rows, raw_train_end={dataset.raw_train_end})")
    dataloader = DataLoader(
        train_subset,
        batch_size=getattr(dp_args, 'batch_size', 1),
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return train_subset, dataloader


def train_epoch(model, dataloader, optimizer, args, device, epoch):
    """Train one epoch sequentially."""
    model.prompt_z.train()
    model.backbone_adapter.eval()

    total_loss = 0.0
    total_forecast = 0.0
    total_delta_reg = 0.0
    total_mask_reg = 0.0
    total_noop_pen = 0.0
    total_reg_scale = 0.0
    total_noop_scale = 0.0
    total_gamma_floor = 0.0
    total_mask_floor = 0.0
    total_noop_active = 0.0
    n_steps = 0

    diag_accum = {
        "gamma_mean": 0.0, "mask_ratio": 0.0,
        "mask_mean": 0.0, "hidden_norm": 0.0,
        "raw_delta_norm": 0.0, "applied_delta_norm": 0.0,
        "raw_delta_to_hidden_ratio": 0.0,
        "effective_delta_ratio": 0.0,
    }

    # Reset residual tracker and delayed-label cache at start of each epoch.
    model.residual_tracker.reset()
    residual_cache = deque()

    for step, (X, Y) in enumerate(dataloader):
        if args.max_steps and step >= args.max_steps:
            break

        X = X.to(device, non_blocking=True)  # [B, seq_len, C]
        Y = Y.to(device, non_blocking=True)  # [B, pred_len, C]

        optimizer.zero_grad()

        try:
            global_step = epoch * len(dataloader) + step
        except TypeError:
            global_step = step
        if args.gamma_floor_steps > 0 and global_step < args.gamma_floor_steps:
            gamma_floor_current = args.gamma_floor * (
                1.0 - global_step / args.gamma_floor_steps
            )
        else:
            gamma_floor_current = 0.0
        if args.mask_floor_steps > 0 and global_step < args.mask_floor_steps:
            mask_floor_current = args.mask_floor * (
                1.0 - global_step / args.mask_floor_steps
            )
        else:
            mask_floor_current = 0.0
        if args.reg_warmup_steps > 0:
            reg_scale = min(1.0, max(0.0, (global_step + 1) / args.reg_warmup_steps))
        else:
            reg_scale = 1.0
        if global_step < args.noop_warmup_steps:
            noop_scale = 0.0
        else:
            noop_scale = min(
                1.0,
                (global_step - args.noop_warmup_steps) / max(1, args.noop_ramp_steps),
            )

        # Forward (backbone frozen, PromptZ with grad)
        Y_hat, Y_frozen, reg_tensors, diagnostics = model.forward_train(
            X,
            gamma_floor=gamma_floor_current,
            mask_floor=mask_floor_current,
        )

        # 1. Forecast loss
        forecast_loss = nn.functional.mse_loss(Y_hat, Y)

        # 2. Delta regularization
        delta_reg = reg_tensors["effective_delta_ratio"]

        # 3. Mask budget regularization.
        # Allow a small active mask budget; only penalize usage above target.
        mask_mean = reg_tensors["mask_mean"]
        mask_reg = torch.relu(mask_mean - args.target_mask_ratio)

        # 4. No-op margin: penalize gamma when correction doesn't help
        with torch.no_grad():
            mse_frozen = nn.functional.mse_loss(Y_frozen, Y)
        # If correction is not clearly better, penalize gate opening
        # This encourages gamma → 0 when there's no benefit
        noop_penalty = forecast_loss.new_zeros(())
        noop_active = 0.0
        effective_ratio = reg_tensors["effective_delta_ratio"].detach().item()
        if (
            global_step >= args.noop_warmup_steps
            and effective_ratio >= args.noop_min_effective_ratio
            and forecast_loss.item() >= mse_frozen.item() - args.noop_epsilon
        ):
            # Correction didn't help → penalize gamma magnitude
            # gamma is in diagnostics as a scalar mean
            noop_penalty = reg_tensors["gamma_mean"]
            noop_active = 1.0

        loss = (forecast_loss
                + reg_scale * args.lambda_delta * delta_reg
                + reg_scale * args.lambda_mask * mask_reg
                + noop_scale * args.lambda_noop * noop_penalty)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.prompt_z.parameters(), max_norm=1.0
        )

        optimizer.step()

        # Update residual tracker.  In delayed mode, the current label is not
        # visible to the current prediction; it can only enter the tracker
        # after H newer windows have passed, matching streaming eval timing.
        with torch.no_grad():
            if args.delayed_residual_training:
                residual_cache.append((Y_frozen.detach(), Y.detach()))
                if len(residual_cache) > args.forecast_H:
                    old_pred, old_true = residual_cache.popleft()
                    model.residual_tracker.update(old_pred, old_true)
                else:
                    model.residual_tracker.step_no_update()
            else:
                model.residual_tracker.update(Y_frozen.detach(), Y.detach())

            residual_cache_len = len(residual_cache)
            tracker_count = int(model.residual_tracker._count.item())
            residual_tracker_warmed = float(tracker_count >= model.residual_tracker.K)

        # Accumulate
        total_loss += loss.item()
        total_forecast += forecast_loss.item()
        total_delta_reg += delta_reg.item() if isinstance(delta_reg, torch.Tensor) else delta_reg
        total_mask_reg += mask_reg.item() if isinstance(mask_reg, torch.Tensor) else mask_reg
        total_noop_pen += noop_penalty.item() if isinstance(noop_penalty, torch.Tensor) else noop_penalty
        total_reg_scale += reg_scale
        total_noop_scale += noop_scale
        total_gamma_floor += gamma_floor_current
        total_mask_floor += mask_floor_current
        total_noop_active += noop_active
        n_steps += 1

        for k in diag_accum:
            diag_accum[k] += diagnostics.get(k, 0.0)

        if step % 200 == 0:
            print(f"  [Epoch {epoch} Step {step}] "
                  f"loss={loss.item():.6f} forecast={forecast_loss.item():.6f} "
                  f"frozen={mse_frozen.item():.6f} "
                  f"gamma={diagnostics.get('gamma_mean', 0):.6f} "
                  f"mask_mean={diagnostics.get('mask_mean', 0):.6f} "
                  f"mask_ratio={diagnostics.get('mask_ratio', 0):.4f} "
                  f"mask_budget={mask_reg.item():.6f} "
                  f"reg_scale={reg_scale:.3f} noop={noop_active:.0f} "
                  f"noop_scale={noop_scale:.3f} "
                  f"gamma_floor={gamma_floor_current:.4f} "
                  f"mask_floor={mask_floor_current:.4f} "
                  f"residual_cache_len={residual_cache_len} "
                  f"residual_tracker_warmed={int(residual_tracker_warmed)} "
                  f"raw_d/h={diagnostics.get('raw_delta_to_hidden_ratio', 0):.6f} "
                  f"eff_d/h={diagnostics.get('effective_delta_ratio', 0):.6f}")

    if n_steps == 0:
        return {}

    return {
        "loss": total_loss / n_steps,
        "forecast_loss": total_forecast / n_steps,
        "delta_reg": total_delta_reg / n_steps,
        "mask_reg": total_mask_reg / n_steps,
        "noop_penalty": total_noop_pen / n_steps,
        "reg_scale": total_reg_scale / n_steps,
        "noop_scale": total_noop_scale / n_steps,
        "gamma_floor": total_gamma_floor / n_steps,
        "mask_floor": total_mask_floor / n_steps,
        "noop_active_ratio": total_noop_active / n_steps,
        "delayed_residual_training": args.delayed_residual_training,
        "residual_cache_len": len(residual_cache),
        "residual_tracker_warmed": bool(
            int(model.residual_tracker._count.item()) >= model.residual_tracker.K
        ),
        **{k: v / n_steps for k, v in diag_accum.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Prompt-Z Offline Training (Stage A)")

    # Data
    parser.add_argument('--root_path', type=str, default='./data')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--forecast_H', type=int, required=True)
    parser.add_argument('--enc_in', type=int, default=None,
                        help='Number of channels. Auto-detected if None.')

    # Backbone
    parser.add_argument('--backbone', type=str, default='patchtst',
                        choices=['patchtst', 'itransformer'])
    parser.add_argument('--D_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--pretrained_weights', type=str, default=None)

    # PromptZ architecture
    parser.add_argument('--d_drift', type=int, default=64)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--gamma_init_bias', type=float, default=-3.0)
    parser.add_argument('--mask_init_bias', type=float, default=-1.5)
    parser.add_argument('--max_delta_ratio', type=float, default=0.05,
                        help='Ratio clamp: raw delta_h norm may be at most this fraction of hidden norm.')
    parser.add_argument('--residual_window_K', type=int, default=24)

    # Training
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--delayed_residual_training',
                        dest='delayed_residual_training',
                        action='store_true',
                        default=True,
                        help='Train residual stats with H-step delayed label feedback.')
    parser.add_argument('--no_delayed_residual_training',
                        dest='delayed_residual_training',
                        action='store_false',
                        help='Use immediate residual updates during training.')

    # Loss weights
    parser.add_argument('--lambda_delta', type=float, default=2e-4)
    parser.add_argument('--lambda_mask', type=float, default=1e-4)
    parser.add_argument('--lambda_noop', type=float, default=0.005)
    parser.add_argument('--noop_epsilon', type=float, default=1e-4)
    parser.add_argument('--target_mask_ratio', type=float, default=0.10,
                        help='Mask budget target; only mask usage above this mean is penalized.')
    parser.add_argument('--reg_warmup_steps', type=int, default=2000,
                        help='Linearly ramp delta/mask regularization over this many global steps.')
    parser.add_argument('--noop_warmup_steps', type=int, default=6000,
                        help='Disable no-op gamma penalty before this global step.')
    parser.add_argument('--noop_ramp_steps', type=int, default=2000,
                        help='Linearly ramp no-op gamma penalty after warmup over this many steps.')
    parser.add_argument('--noop_min_effective_ratio', type=float, default=1e-4,
                        help='Only apply no-op penalty after Prompt-Z makes a non-trivial correction.')
    parser.add_argument('--gamma_floor', type=float, default=0.1,
                        help='Training-only additive floor for gamma to help zero-init modulation start.')
    parser.add_argument('--gamma_floor_steps', type=int, default=4000,
                        help='Linearly decay gamma_floor to zero over this many global steps.')
    parser.add_argument('--mask_floor', type=float, default=0.05,
                        help='Training-only additive floor for mask to prevent early mask collapse.')
    parser.add_argument('--mask_floor_steps', type=int, default=4000,
                        help='Linearly decay mask_floor to zero over this many global steps.')

    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='weights/prompt_z')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Fraction of dataset windows used for Prompt-Z training (60/10/30 split).')
    parser.add_argument('--experiment_tag', type=str, default='')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")
    print(f"[*] delayed_residual_training={args.delayed_residual_training}")

    # Auto-detect enc_in
    if args.enc_in is None:
        import pandas as pd
        df = pd.read_csv(os.path.join(args.root_path, args.data_path))
        # Exclude 'date' column
        args.enc_in = len([c for c in df.columns if c.lower() != 'date'])
        print(f"[*] Auto-detected enc_in={args.enc_in}")

    # Build
    adapter = build_backbone(args, device)
    model = build_model(args, adapter, device)
    _, dataloader = get_dataloader(args, train_ratio=args.train_ratio,
                                   val_ratio=getattr(args, 'val_ratio', 0.1))

    # Optimizer (only PromptZ params)
    optimizer = torch.optim.AdamW(
        model.prompt_z.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Train
    os.makedirs(args.save_dir, exist_ok=True)
    ds_name = args.data_path.replace('.csv', '')
    tag = f"{ds_name}_H{args.forecast_H}"
    if args.experiment_tag:
        tag = f"{tag}_{args.experiment_tag}"

    best_loss = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        metrics = train_epoch(model, dataloader, optimizer, args, device, epoch)
        dt = time.time() - t0
        print(f"[Epoch {epoch}] {dt:.1f}s | "
              f"loss={metrics.get('loss', 0):.6f} | "
              f"forecast={metrics.get('forecast_loss', 0):.6f} | "
              f"gamma={metrics.get('gamma_mean', 0):.4f} | "
              f"mask_ratio={metrics.get('mask_ratio', 0):.4f} | "
              f"mask_floor={metrics.get('mask_floor', 0):.4f} | "
              f"residual_cache_len={metrics.get('residual_cache_len', 0)} | "
              f"residual_tracker_warmed={int(metrics.get('residual_tracker_warmed', False))} | "
              f"raw_d/h={metrics.get('raw_delta_to_hidden_ratio', 0):.6f} | "
              f"eff_d/h={metrics.get('effective_delta_ratio', 0):.6f}")

        if metrics.get('loss', float('inf')) < best_loss:
            best_loss = metrics['loss']
            save_path = os.path.join(args.save_dir, f"prompt_z_{tag}.pth")
            torch.save(model.prompt_z.state_dict(), save_path)
            print(f"  → Saved best to {save_path}")

    # Save final
    save_path = os.path.join(args.save_dir, f"prompt_z_{tag}_final.pth")
    torch.save(model.prompt_z.state_dict(), save_path)
    print(f"[*] Final weights saved to {save_path}")

    # Save training log
    log_dir = os.path.join("logs", "prompt_z")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{tag}.json")
    with open(log_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[*] Training log saved to {log_path}")


if __name__ == '__main__':
    main()
