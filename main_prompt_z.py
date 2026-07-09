"""
main_prompt_z.py — CLI Entry for Prompt-Z Training and Evaluation
===================================================================

Modes:
  train  — Stage A offline training
  eval   — Streaming evaluation (mode0 / mode1 / frozen)
"""

from __future__ import annotations

import argparse
import os
import sys
import json

import torch

from data_provider.data_loader import data_provider
from data_provider.streaming_env import StreamingEnvironment
from models.backbone_adapter import PatchTSTAdapter, iTransformerAdapter
from models.prompt_z import PromptZModulator
from models.prompt_z_framework import PromptZTSF
from core.residual_tracker import ResidualTracker
from engine.streaming_prompt_z import (
    run_prompt_z_streaming,
    run_prompt_z_validation_fallback,
)


def auto_detect_enc_in(root_path, data_path):
    """Detect number of channels from CSV."""
    import pandas as pd
    df = pd.read_csv(os.path.join(root_path, data_path), nrows=5)
    return len([c for c in df.columns if c.lower() != 'date'])


def build_backbone_adapter(args, device):
    """Build and load pretrained backbone adapter."""
    if args.backbone == 'patchtst':
        from models.backbones.PatchTST import Model as PatchTST

        class Cfg:
            task_name = 'long_term_forecast'
            seq_len = args.seq_len
            pred_len = args.forecast_H
            d_model = args.D_model
            d_ff = args.d_ff
            n_heads = 8
            e_layers = args.e_layers
            dropout = 0.1
            activation = 'gelu'
            factor = 1
            enc_in = args.enc_in

        backbone = PatchTST(Cfg()).to(device)
        adapter = PatchTSTAdapter(backbone).to(device)
    elif args.backbone == 'itransformer':
        from models.backbones.iTransformer import Model as iTransformer

        class Cfg:
            task_name = 'long_term_forecast'
            seq_len = args.seq_len
            pred_len = args.forecast_H
            d_model = args.D_model
            d_ff = args.d_ff
            n_heads = 8
            e_layers = args.e_layers
            dropout = 0.1
            activation = 'gelu'
            factor = 1
            enc_in = args.enc_in
            output_attention = False
            embed = 'timeF'
            freq = 'h'

        backbone = iTransformer(Cfg()).to(device)
        adapter = iTransformerAdapter(backbone).to(device)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # Load pretrained weights
    if args.pretrained_weights:
        state = torch.load(args.pretrained_weights, map_location=device,
                          weights_only=False)
        # Handle different save formats
        prefix_candidates = ['backbone_adapter.backbone.', 'model.', '']
        loaded = False
        for prefix in prefix_candidates:
            if prefix and any(k.startswith(prefix) for k in state.keys()):
                sub_state = {k[len(prefix):]: v for k, v in state.items()
                            if k.startswith(prefix)}
                try:
                    adapter.backbone.load_state_dict(sub_state, strict=False)
                    loaded = True
                    break
                except RuntimeError:
                    continue
        if not loaded:
            adapter.backbone.load_state_dict(state, strict=False)
        print(f"[*] Loaded backbone from {args.pretrained_weights}")

    # Freeze backbone
    for p in adapter.parameters():
        p.requires_grad = False

    return adapter


def build_prompt_z_model(args, adapter, device):
    """Build PromptZTSF with PromptZModulator + ResidualTracker."""
    prompt_z = PromptZModulator(
        d_model=args.D_model,
        hidden_layout=adapter.hidden_layout,
        d_drift=args.d_drift,
        rank=args.rank,
        gamma_init_bias=args.gamma_init_bias,
        mask_init_bias=args.mask_init_bias,
        max_delta_ratio=args.max_delta_ratio,
    ).to(device)

    # Load trained PromptZ weights if specified
    if args.prompt_z_weights:
        state = torch.load(args.prompt_z_weights, map_location=device,
                          weights_only=True)
        prompt_z.load_state_dict(state)
        print(f"[*] Loaded PromptZ weights from {args.prompt_z_weights}")

    residual_tracker = ResidualTracker(
        num_channels=args.enc_in,
        window_K=args.residual_window_K,
    ).to(device)

    model = PromptZTSF(adapter, prompt_z, residual_tracker).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[*] Model built: {n_trainable:,} trainable / {n_total:,} total "
          f"({100*n_trainable/n_total:.2f}%)")
    print(f"[*] Layout: {adapter.hidden_layout}")

    return model


def get_streaming_dataloader(args):
    """Get streaming dataloader (sequential, batch=1)."""
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
    dp_args.train_ratio = getattr(args, 'train_ratio', 0.6)
    dp_args.val_ratio   = getattr(args, 'val_ratio', 0.1)

    dataset, dataloader = data_provider(dp_args)
    return dataset, dataloader


def main():
    parser = argparse.ArgumentParser(description="Prompt-Z CLI")

    # Data
    parser.add_argument('--root_path', type=str, default='./data')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--forecast_H', type=int, required=True)
    parser.add_argument('--enc_in', type=int, default=None)

    # Backbone
    parser.add_argument('--backbone', type=str, default='patchtst',
                        choices=['patchtst', 'itransformer'])
    parser.add_argument('--D_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--pretrained_weights', type=str, default=None)

    # PromptZ
    parser.add_argument('--d_drift', type=int, default=64)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--gamma_init_bias', type=float, default=-3.0)
    parser.add_argument('--mask_init_bias', type=float, default=-1.5)
    parser.add_argument('--max_delta_ratio', type=float, default=0.05,
                        help='Ratio clamp: raw delta_h norm may be at most this fraction of hidden norm.')
    parser.add_argument('--residual_window_K', type=int, default=24)
    parser.add_argument('--prompt_z_weights', type=str, default=None)

    # Streaming eval
    parser.add_argument('--streaming_mode', type=str, default='mode0',
                        choices=['frozen', 'mode0', 'mode1'])
    parser.add_argument('--calibration_lr', type=float, default=1e-4)
    parser.add_argument('--experiment_tag', type=str, default='')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Override: skip first N windows for metric accumulation. '
                             'Auto-computed from strict split if None.')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Fraction of dataset windows used for offline training.')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of dataset windows used for validation.')
    parser.add_argument('--enable_validation_fallback', action='store_true',
                        help='Use a validation streaming segment to select Prompt-Z vs frozen.')
    parser.add_argument('--fallback_margin', type=float, default=0.005,
                        help='Enable Prompt-Z only if validation MSE beats frozen by this fraction.')
    parser.add_argument('--validation_steps', type=int, default=None,
                        help='Override validation fallback length in aligned label steps. '
                             'Auto-computed from val_ratio if None.')

    # Misc
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-detect enc_in
    if args.enc_in is None:
        args.enc_in = auto_detect_enc_in(args.root_path, args.data_path)
        print(f"[*] Auto-detected enc_in={args.enc_in}")

    # Build model
    adapter = build_backbone_adapter(args, device)
    model = build_prompt_z_model(args, adapter, device)

    # Get dataloader and wrap in StreamingEnvironment for delay alignment
    dataset, base_loader = get_streaming_dataloader(args)
    streaming_loader = StreamingEnvironment(base_loader, forecast_H=args.forecast_H)

    # --- Strict split by label timestamp (from dataset) ---
    # dataset attributes are computed from raw row boundaries, not window count
    # ratios, so they correctly reflect physical time boundaries.
    train_size  = dataset.train_size   # windows with label fully in train zone
    val_start   = dataset.val_start    # first window with label starting in val
    test_start  = dataset.test_start   # first window with label starting in test
    N_total     = len(dataset)

    # If user explicitly overrides train_size (legacy), respect it
    if args.train_size is not None:
        test_start = args.train_size

    # Validation segment = windows [val_start, test_start)
    val_steps = test_start - val_start
    if args.validation_steps is not None:
        val_steps = args.validation_steps

    print(f"[*] STRICT_SPLIT (label-timestamp based)")
    print(f"[*]   total_windows={N_total} | raw_train_end={dataset.raw_train_end} | raw_val_end={dataset.raw_val_end}")
    print(f"[*]   train=[0,{train_size}) — label in train raw rows")
    print(f"[*]   val=[{val_start},{test_start}) — label in val raw rows")
    print(f"[*]   test=[{test_start},{N_total}) — label in test raw rows")
    print(f"[*]   val_steps={val_steps}")
    if args.pretrained_weights:
        print(f"[*]   loaded_backbone_weights={args.pretrained_weights}")
    if args.prompt_z_weights:
        print(f"[*]   loaded_promptz_weights={args.prompt_z_weights}")

    # Run streaming eval
    if args.enable_validation_fallback:
        if args.streaming_mode == 'frozen':
            raise ValueError("validation fallback requires mode0 or mode1, not frozen")
        results = run_prompt_z_validation_fallback(
            model=model,
            dataloader=streaming_loader,
            train_size=val_start,
            mode=args.streaming_mode,
            calibration_lr=args.calibration_lr,
            fallback_margin=args.fallback_margin,
            validation_steps=val_steps,
            experiment_tag=args.experiment_tag,
        )
    else:
        results = run_prompt_z_streaming(
            model=model,
            dataloader=streaming_loader,
            train_size=test_start,
            mode=args.streaming_mode,
            calibration_lr=args.calibration_lr,
            experiment_tag=args.experiment_tag,
        )

    print(f"\n[*] Final MSE: {results['MSE']:.6f}")


if __name__ == '__main__':
    main()
