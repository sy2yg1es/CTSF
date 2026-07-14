"""Causal gate-learnability probe for a frozen Prompt-Z delta branch.

The experiment never trains on validation/test labels:
  1. Freeze backbone, drift encoder, and the Phase-1 delta branch.
  2. Generate causal per-(window, channel) correction advantages on train tail.
  3. Fit three tiny probes on causal features:
       - weighted binary classifier for sign(advantage)
       - Huber regressor for normalized advantage
       - exact advantage-weighted logistic regret classifier
  4. In the default protocol, select checkpoints on a temporal holdout from
     the training tail, use the structural threshold logit > 0, and report
     every contiguous validation block. Test is intentionally untouched.

The same implementation is exposed by ``train_binary_gate.py`` for formal
Phase-2 training, keeping validation and deployment semantics identical.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.residual_tracker import ResidualTracker
from data_provider.data_loader import data_provider
from models.binary_channel_gate import (
    PROTOCOL_VERSION,
    BinaryChannelGate,
    build_causal_gate_features,
)
from scripts.eval_test_oracle import (
    build_backbone,
    build_prompt_z,
    pack_stats,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(args):
    class DPArgs:
        pass

    dp = DPArgs()
    dp.root_path = args.root_path
    dp.data_path = args.data_path
    dp.features = args.features
    dp.seq_len = args.seq_len
    dp.pred_len = args.forecast_H
    dp.target = "OT"
    dp.num_workers = args.num_workers
    dp.train_ratio = args.train_ratio
    dp.val_ratio = args.val_ratio
    dataset, _ = data_provider(dp)
    return dataset


def make_loader(dataset, start: int, end: int, args):
    return DataLoader(
        Subset(dataset, range(start, end)),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def _frozen_forward(adapter, X):
    hidden, means, stdev = adapter.encode_until_hook(X)
    hidden = hidden.detach()
    return adapter.decode_from_hook(hidden, means, stdev)


def _fixed_delta_forward(adapter, prompt_z, X, stats):
    hidden, means, stdev = adapter.encode_until_hook(X)
    hidden = hidden.detach()
    Y_frozen = adapter.decode_from_hook(hidden, means, stdev)

    summary = prompt_z._hidden_summary(hidden)
    drift_state = prompt_z.drift_encoder(summary, stats)
    if prompt_z.hidden_layout == "BCDP":
        delta = prompt_z.low_rank_mod(hidden.permute(0, 1, 3, 2), drift_state)
        delta = delta.permute(0, 1, 3, 2)
    else:
        delta = prompt_z.low_rank_mod(hidden, drift_state)
    delta = prompt_z._ratio_clamp(delta, hidden)
    Y_fixed = adapter.decode_from_hook(hidden + delta, means, stdev)
    return Y_frozen, Y_fixed, drift_state


def _tracker_step(tracker, residual_cache, frozen, true, horizon):
    residual_cache.append((frozen.detach(), true.detach()))
    if len(residual_cache) > horizon:
        old_pred, old_true = residual_cache.popleft()
        tracker.update(old_pred, old_true)
    else:
        tracker.step_no_update()


@torch.no_grad()
def collect_causal_records(
    adapter,
    prompt_z,
    dataset,
    warmup_range,
    sample_range,
    args,
    device,
    label,
):
    """Collect frozen drift features and output-space oracle targets."""
    tracker = ResidualTracker(args.enc_in, args.residual_window_K).to(device)
    tracker.reset()
    residual_cache = deque()
    prompt_z.eval()

    warm_start, warm_end = warmup_range
    print(f"[{label}] tracker warmup [{warm_start},{warm_end})")
    for X, Y in make_loader(dataset, warm_start, warm_end, args):
        X = X.to(device)
        Y = Y.to(device)
        Y_frozen = _frozen_forward(adapter, X)
        _tracker_step(
            tracker, residual_cache, Y_frozen, Y, args.forecast_H
        )

    features = []
    advantages = []
    relative_advantages = []
    labels = []
    valid = []
    frozen_mse_channels = []
    fixed_mse_channels = []
    correction_cross_channels = []
    correction_energy_channels = []
    continuous_mse_channels = []
    continuous_gammas = []

    start, end = sample_range
    print(f"[{label}] collect [{start},{end}) ({end-start} windows)")
    for local_step, (X, Y) in enumerate(make_loader(dataset, start, end, args)):
        X = X.to(device)
        Y = Y.to(device)
        stats = pack_stats(tracker, device)
        Y_frozen, Y_fixed, drift_state = _fixed_delta_forward(
            adapter, prompt_z, X, stats
        )
        residual = Y - Y_frozen
        correction = Y_fixed - Y_frozen
        reduce_dims = tuple(range(1, Y.dim() - 1))
        frozen_mse_c = (Y_frozen - Y).pow(2).mean(dim=reduce_dims)
        fixed_mse_c = (Y_fixed - Y).pow(2).mean(dim=reduce_dims)
        correction_cross_c = (residual * correction).mean(dim=reduce_dims)
        correction_energy_c = correction.pow(2).mean(dim=reduce_dims)
        advantage_c = frozen_mse_c - fixed_mse_c
        relative_advantage_c = (
            advantage_c / frozen_mse_c.clamp(min=1e-12) * 100.0
        )
        target_c = (relative_advantage_c > args.target_margin_pct).to(torch.float32)
        valid_c = relative_advantage_c.abs() >= args.target_margin_pct
        continuous_gamma_c = (
            correction_cross_c / correction_energy_c.clamp(min=1e-12)
        ).clamp(0.0, 1.0)
        continuous_mse_c = (
            frozen_mse_c
            - 2.0 * continuous_gamma_c * correction_cross_c
            + continuous_gamma_c.pow(2) * correction_energy_c
        )

        gate_features = build_causal_gate_features(
            drift_state, stats, Y_frozen, Y_fixed, args.feature_mode
        )
        feature_dim = gate_features.shape[-1]
        packed = torch.cat(
            [
                gate_features.squeeze(0),
                torch.stack(
                    [
                        advantage_c.squeeze(0),
                        relative_advantage_c.squeeze(0),
                        target_c.squeeze(0),
                        valid_c.squeeze(0).to(torch.float32),
                        frozen_mse_c.squeeze(0),
                        fixed_mse_c.squeeze(0),
                        correction_cross_c.squeeze(0),
                        correction_energy_c.squeeze(0),
                        continuous_mse_c.squeeze(0),
                        continuous_gamma_c.squeeze(0),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        ).cpu()
        features.append(packed[:, :feature_dim])
        stats_columns = packed[:, feature_dim:]
        advantages.append(stats_columns[:, 0])
        relative_advantages.append(stats_columns[:, 1])
        labels.append(stats_columns[:, 2])
        valid.append(stats_columns[:, 3].to(torch.bool))
        frozen_mse_channels.append(stats_columns[:, 4])
        fixed_mse_channels.append(stats_columns[:, 5])
        correction_cross_channels.append(stats_columns[:, 6])
        correction_energy_channels.append(stats_columns[:, 7])
        continuous_mse_channels.append(stats_columns[:, 8])
        continuous_gammas.append(stats_columns[:, 9])

        _tracker_step(
            tracker, residual_cache, Y_frozen, Y, args.forecast_H
        )
        if (local_step + 1) % args.log_interval == 0:
            print(f"[{label}] collected {local_step+1}/{end-start}")

    return {
        "features": torch.stack(features),              # [N,C,D]
        "advantage": torch.stack(advantages),           # [N,C]
        "relative_advantage_pct": torch.stack(relative_advantages),
        "label": torch.stack(labels).to(torch.float32), # [N,C]
        "valid": torch.stack(valid).to(torch.bool),     # [N,C]
        "frozen_mse_channel": torch.stack(frozen_mse_channels),
        "fixed_mse_channel": torch.stack(fixed_mse_channels),
        "correction_cross_channel": torch.stack(correction_cross_channels),
        "correction_energy_channel": torch.stack(correction_energy_channels),
        "continuous_mse_channel": torch.stack(continuous_mse_channels),
        "continuous_gamma_channel": torch.stack(continuous_gammas),
    }


GateProbe = BinaryChannelGate  # Backward-compatible research-script alias.


def flatten_valid(records):
    x = records["features"].reshape(-1, records["features"].shape[-1])
    y = records["label"].reshape(-1)
    rel = records["relative_advantage_pct"].reshape(-1)
    valid = records["valid"].reshape(-1)
    return x[valid], y[valid], rel[valid]


def flatten_exact(records):
    """Flatten exact oracle decisions without dropping near-tie examples."""
    x = records["features"].reshape(-1, records["features"].shape[-1])
    rel = records["relative_advantage_pct"].reshape(-1)
    y = (records["advantage"].reshape(-1) > 0).to(torch.float32)
    return x, y, rel


def binary_auc(scores, labels):
    labels = labels.to(torch.bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = torch.unique(
        scores, sorted=True, return_inverse=True, return_counts=True
    )
    cumulative = counts.cumsum(0).to(torch.float32)
    average_rank = cumulative - (counts.to(torch.float32) - 1.0) / 2.0
    ranks = average_rank[inverse]
    rank_sum = ranks[labels].sum().item()
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(scores, labels):
    labels = labels.to(torch.bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_y = labels[order].to(torch.float32)
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    positives = torch.zeros(len(counts), dtype=torch.float32)
    positives.scatter_add_(0, inverse, sorted_y)
    cumulative_positives = positives.cumsum(0)
    cumulative_total = counts.cumsum(0).to(torch.float32)
    precision = cumulative_positives / cumulative_total
    recall_increment = positives / n_pos
    return (precision * recall_increment).sum().item()


def balanced_accuracy(scores, labels, threshold):
    pred = scores > threshold
    truth = labels.to(torch.bool)
    pos = truth.sum().clamp(min=1)
    neg = (~truth).sum().clamp(min=1)
    tpr = (pred & truth).sum().to(torch.float32) / pos
    tnr = ((~pred) & (~truth)).sum().to(torch.float32) / neg
    return ((tpr + tnr) / 2).item()


def pearson(x, y):
    x = x.to(torch.float32) - x.to(torch.float32).mean()
    y = y.to(torch.float32) - y.to(torch.float32).mean()
    denom = x.norm() * y.norm()
    return (x @ y / denom.clamp(min=1e-12)).item()


def train_classifier(train, calib, args, device):
    train_x, train_y, train_rel = flatten_valid(train)
    calib_x, calib_y, calib_rel = flatten_valid(calib)
    severity_scale = train_rel.abs().median().clamp(min=1e-4)
    weights = (train_rel.abs() / severity_scale).clamp(0.1, 10.0)

    dataset = TensorDataset(train_x, train_y, weights)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp(min=1e-6)
    model = GateProbe(
        train_x.shape[-1], args.probe_hidden, feature_mean, feature_std
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay
    )
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(args.probe_epochs):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            per_item = F.binary_cross_entropy_with_logits(
                model(xb), yb, reduction="none"
            )
            loss = (per_item * wb).sum() / wb.sum().clamp(min=1e-12)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(calib_x.to(device)).cpu()
            calib_w = (calib_rel.abs() / severity_scale).clamp(0.1, 10.0)
            per_item = F.binary_cross_entropy_with_logits(
                logits, calib_y, reduction="none"
            )
            val_loss = (per_item * calib_w).sum() / calib_w.sum().clamp(min=1e-12)
        if val_loss.item() < best_loss - 1e-6:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model.cpu(), {"best_calib_loss": best_loss, "epochs": epoch + 1}


def train_regressor(train, calib, args, device):
    train_x, _, train_rel = flatten_valid(train)
    calib_x, _, calib_rel = flatten_valid(calib)
    target_scale = train_rel.abs().quantile(0.75).clamp(min=1e-3)
    train_target = (train_rel / target_scale).clamp(-5.0, 5.0)
    calib_target = (calib_rel / target_scale).clamp(-5.0, 5.0)
    dataset = TensorDataset(train_x, train_target)
    generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp(min=1e-6)
    model = GateProbe(
        train_x.shape[-1], args.probe_hidden, feature_mean, feature_std
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay
    )
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(args.probe_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.smooth_l1_loss(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.smooth_l1_loss(
                model(calib_x.to(device)).cpu(), calib_target
            )
        if val_loss.item() < best_loss - 1e-6:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model.cpu(), {
        "best_calib_loss": best_loss,
        "epochs": epoch + 1,
        "target_scale_pct": target_scale.item(),
    }


def train_regret_classifier(train, selection, args, device):
    """Train a zero-threshold gate with an advantage-weighted logistic loss.

    For signed label s in {-1,+1}, softplus(-s*z) is a calibrated surrogate
    for choosing the wrong binary action. Weighting by |advantage| aligns
    checkpoint selection with forecast regret while near-ties contribute
    almost no gradient. No sparsity or gate-magnitude regularizer is used.
    """
    train_x, train_y, train_rel = flatten_exact(train)
    select_x, select_y, select_rel = flatten_exact(selection)
    severity_scale = train_rel.abs().median().clamp(min=1e-4)
    train_w = (train_rel.abs() / severity_scale).clamp(max=20.0)
    select_w = (select_rel.abs() / severity_scale).clamp(max=20.0)

    dataset = TensorDataset(train_x, train_y, train_w)
    generator = torch.Generator().manual_seed(args.seed + 2)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp(min=1e-6)
    model = GateProbe(
        train_x.shape[-1], args.probe_hidden, feature_mean, feature_std
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay
    )
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(args.probe_epochs):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            sign = yb.mul(2.0).sub(1.0)
            opt.zero_grad()
            per_item = F.softplus(-sign * model(xb))
            loss = (per_item * wb).sum() / wb.sum().clamp(min=1e-12)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(select_x.to(device)).cpu()
            sign = select_y.mul(2.0).sub(1.0)
            per_item = F.softplus(-sign * logits)
            val_loss = (
                (per_item * select_w).sum() / select_w.sum().clamp(min=1e-12)
            )
        if val_loss.item() < best_loss - 1e-6:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model.cpu(), {
        "best_selection_regret_logistic": best_loss,
        "epochs": epoch + 1,
        "severity_scale_pct": severity_scale.item(),
    }


@torch.no_grad()
def probe_scores(model, records):
    return model(records["features"]).to(torch.float32)


def mse_with_gate(records, gamma):
    if gamma.dtype == torch.bool:
        return torch.where(
            gamma,
            records["fixed_mse_channel"],
            records["frozen_mse_channel"],
        ).mean().item()
    gamma = gamma.to(torch.float32)
    mse = (
        records["frozen_mse_channel"]
        - 2.0 * gamma * records["correction_cross_channel"]
        + gamma.pow(2) * records["correction_energy_channel"]
    )
    return mse.mean().item()


def tune_threshold(scores, records):
    flat = scores.flatten()
    quantiles = torch.linspace(0.0, 1.0, 101)
    candidates = torch.unique(torch.quantile(flat, quantiles))
    best_threshold = 0.0
    best_mse = float("inf")
    for threshold in candidates:
        mse = mse_with_gate(records, scores > threshold)
        if mse < best_mse:
            best_mse = mse
            best_threshold = threshold.item()
    return best_threshold, best_mse


def split_records(records, split_at):
    return (
        {k: v[:split_at] for k, v in records.items()},
        {k: v[split_at:] for k, v in records.items()},
    )


def select_safe_mode(
    records,
    regressor,
    min_improvement_pct=0.2,
    min_positive_block_frac=0.75,
    n_blocks=4,
):
    """Conservatively choose Frozen, Fixed, or learned on train-only data.

    A non-frozen mode must clear both an aggregate improvement margin and a
    contiguous-block consistency check. Otherwise the branch defaults off.
    """
    reg_scores = probe_scores(regressor, records)
    candidates = {
        "frozen": records["frozen_mse_channel"].mean().item(),
        "fixed": records["fixed_mse_channel"].mean().item(),
        "learned_regressor": mse_with_gate(records, reg_scores > 0.0),
    }
    n_windows = len(records["features"])
    block_improvements = {"fixed": [], "learned_regressor": []}
    for block_index in range(n_blocks):
        start = n_windows * block_index // n_blocks
        end = n_windows * (block_index + 1) // n_blocks
        if end <= start:
            continue
        block = {key: value[start:end] for key, value in records.items()}
        frozen_mse = block["frozen_mse_channel"].mean().item()
        block_values = {
            "fixed": block["fixed_mse_channel"].mean().item(),
            "learned_regressor": mse_with_gate(
                block, probe_scores(regressor, block) > 0.0
            ),
        }
        for mode, value in block_values.items():
            block_improvements[mode].append(
                (frozen_mse - value) / max(frozen_mse, 1e-12) * 100.0
            )

    frozen_mse = candidates["frozen"]
    eligible = ["frozen"]
    diagnostics = {}
    for mode in ("fixed", "learned_regressor"):
        aggregate_improvement = (
            (frozen_mse - candidates[mode]) / max(frozen_mse, 1e-12) * 100.0
        )
        blocks = block_improvements[mode]
        positive_fraction = (
            sum(value > 0.0 for value in blocks) / max(len(blocks), 1)
        )
        is_eligible = (
            aggregate_improvement >= min_improvement_pct
            and positive_fraction >= min_positive_block_frac
        )
        diagnostics[mode] = {
            "aggregate_improvement_vs_frozen_pct": aggregate_improvement,
            "block_improvement_vs_frozen_pct": blocks,
            "positive_block_fraction": positive_fraction,
            "eligible": is_eligible,
        }
        if is_eligible:
            eligible.append(mode)
    selected = min(eligible, key=lambda mode: candidates[mode])
    return selected, candidates, diagnostics


def evaluate_safe_mode(records, selected_mode, regressor):
    frozen_mse = records["frozen_mse_channel"].mean().item()
    fixed_mse = records["fixed_mse_channel"].mean().item()
    if selected_mode == "frozen":
        selected_mse = frozen_mse
    elif selected_mode == "fixed":
        selected_mse = fixed_mse
    elif selected_mode == "learned_regressor":
        selected_mse = mse_with_gate(
            records, probe_scores(regressor, records) > 0.0
        )
    else:
        raise ValueError(f"Unknown safe mode: {selected_mode}")
    return {
        "selected_mode": selected_mode,
        "mse": selected_mse,
        "improvement_vs_frozen_pct": (
            (frozen_mse - selected_mse) / max(frozen_mse, 1e-12) * 100.0
        ),
        "improvement_vs_fixed_pct": (
            (fixed_mse - selected_mse) / max(fixed_mse, 1e-12) * 100.0
        ),
    }


def evaluate_records(
    records,
    classifier,
    regressor,
    regret_classifier,
    cls_threshold,
    reg_threshold,
    regret_threshold=0.0,
):
    cls_scores = probe_scores(classifier, records)
    reg_scores = probe_scores(regressor, records)
    regret_scores = probe_scores(regret_classifier, records)
    x, y, rel = flatten_valid(records)
    valid_flat = records["valid"].reshape(-1)
    cls_valid = cls_scores.reshape(-1)[valid_flat]
    reg_valid = reg_scores.reshape(-1)[valid_flat]
    regret_valid = regret_scores.reshape(-1)[valid_flat]

    frozen_mse = records["frozen_mse_channel"].mean().item()
    fixed_mse = records["fixed_mse_channel"].mean().item()
    binary_oracle_mse = torch.minimum(
        records["frozen_mse_channel"], records["fixed_mse_channel"]
    ).mean().item()
    continuous_oracle_mse = records["continuous_mse_channel"].mean().item()
    cls_mse = mse_with_gate(records, cls_scores > cls_threshold)
    cls_soft_mse = mse_with_gate(records, torch.sigmoid(cls_scores))
    reg_mse = mse_with_gate(records, reg_scores > reg_threshold)
    regret_mse = mse_with_gate(records, regret_scores > regret_threshold)
    regret_soft_mse = mse_with_gate(records, torch.sigmoid(regret_scores))

    def improvement(reference, value):
        return (reference - value) / max(reference, 1e-12) * 100.0

    def recovery(value):
        denom = fixed_mse - binary_oracle_mse
        return (fixed_mse - value) / max(denom, 1e-12)

    continuous_gain = frozen_mse - continuous_oracle_mse
    binary_gain = frozen_mse - binary_oracle_mse
    continuous_gamma = records["continuous_gamma_channel"]
    return {
        "n_windows": int(records["features"].shape[0]),
        "n_channels": int(records["features"].shape[1]),
        "valid_label_frac": records["valid"].float().mean().item(),
        "positive_label_frac": y.mean().item(),
        "frozen_mse": frozen_mse,
        "fixed_gamma1_mse": fixed_mse,
        "binary_oracle_mse": binary_oracle_mse,
        "continuous_oracle_mse": continuous_oracle_mse,
        "continuous_gamma_mean": continuous_gamma.mean().item(),
        "continuous_gamma_mid_frac": (
            (continuous_gamma > 0.05) & (continuous_gamma < 0.95)
        ).float().mean().item(),
        "binary_capture_of_continuous_gain":
            binary_gain / max(continuous_gain, 1e-12),
        "classifier": {
            "auc": binary_auc(cls_valid, y),
            "average_precision": average_precision(cls_valid, y),
            "balanced_accuracy": balanced_accuracy(cls_valid, y, cls_threshold),
            "threshold": cls_threshold,
            "binary_mse": cls_mse,
            "soft_mse": cls_soft_mse,
            "binary_improvement_vs_frozen_pct": improvement(frozen_mse, cls_mse),
            "binary_improvement_vs_fixed_pct": improvement(fixed_mse, cls_mse),
            "binary_oracle_recovery": recovery(cls_mse),
        },
        "regressor": {
            "auc": binary_auc(reg_valid, y),
            "average_precision": average_precision(reg_valid, y),
            "balanced_accuracy": balanced_accuracy(reg_valid, y, reg_threshold),
            "pearson_advantage": pearson(reg_valid, rel),
            "threshold": reg_threshold,
            "binary_mse": reg_mse,
            "binary_improvement_vs_frozen_pct": improvement(frozen_mse, reg_mse),
            "binary_improvement_vs_fixed_pct": improvement(fixed_mse, reg_mse),
            "binary_oracle_recovery": recovery(reg_mse),
        },
        "regret_classifier": {
            "auc": binary_auc(regret_valid, y),
            "average_precision": average_precision(regret_valid, y),
            "balanced_accuracy": balanced_accuracy(
                regret_valid, y, regret_threshold
            ),
            "threshold": regret_threshold,
            "binary_mse": regret_mse,
            "soft_mse": regret_soft_mse,
            "binary_improvement_vs_frozen_pct": improvement(
                frozen_mse, regret_mse
            ),
            "binary_improvement_vs_fixed_pct": improvement(
                fixed_mse, regret_mse
            ),
            "binary_oracle_recovery": recovery(regret_mse),
        },
    }


def print_summary(result):
    h = result["holdout_validation"]
    safe = result["safe_holdout_validation"]
    print("\n" + "=" * 78)
    print("Gate learnability — held-out validation")
    print("=" * 78)
    print(f"windows={h['n_windows']} channels={h['n_channels']} "
          f"positive={h['positive_label_frac']:.3f} valid={h['valid_label_frac']:.3f}")
    print(f"Frozen={h['frozen_mse']:.6f}  Fixed={h['fixed_gamma1_mse']:.6f}  "
          f"BinaryOracle={h['binary_oracle_mse']:.6f}  "
          f"ContinuousOracle={h['continuous_oracle_mse']:.6f}")
    print(f"Binary captures {h['binary_capture_of_continuous_gain']*100:.1f}% "
          f"of continuous oracle gain; mid-gamma={h['continuous_gamma_mid_frac']*100:.2f}%")
    for name in ("classifier", "regressor", "regret_classifier"):
        m = h[name]
        print(f"{name:10s}: AUC={m['auc']:.3f} AP={m['average_precision']:.3f} "
              f"BalAcc={m['balanced_accuracy']:.3f} MSE={m['binary_mse']:.6f} "
              f"vsFrozen={m['binary_improvement_vs_frozen_pct']:+.3f}% "
              f"vsFixed={m['binary_improvement_vs_fixed_pct']:+.3f}% "
              f"oracleRecovery={m['binary_oracle_recovery']*100:+.1f}%")
    print(
        f"safe mode={safe['selected_mode']}  MSE={safe['mse']:.6f}  "
        f"vsFrozen={safe['improvement_vs_frozen_pct']:+.3f}%  "
        f"vsFixed={safe['improvement_vs_fixed_pct']:+.3f}%"
    )
    print("=" * 78)


def main():
    p = argparse.ArgumentParser("Frozen-delta causal gate learnability probe")
    p.add_argument("--root_path", default="./dataset")
    p.add_argument("--data_path", required=True)
    p.add_argument("--features", default="M")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--forecast_H", type=int, required=True)
    p.add_argument("--enc_in", type=int, default=None)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--backbone", default="patchtst", choices=["patchtst", "itransformer"])
    p.add_argument("--D_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--e_layers", type=int, default=3)
    p.add_argument("--pretrained_weights", required=True)
    p.add_argument("--p1_ckpt", required=True)
    p.add_argument("--d_drift", type=int, default=64)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--max_delta_ratio", type=float, default=0.02)
    p.add_argument("--residual_window_K", type=int, default=24)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--val_steps", type=int, default=2000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--calibration_fraction", type=float, default=0.5)
    p.add_argument(
        "--validation_protocol",
        choices=["fixed_zero_blocked", "calibrated_holdout"],
        default="fixed_zero_blocked",
        help=(
            "fixed_zero_blocked selects checkpoints on the training tail and "
            "evaluates all validation blocks at logit threshold zero"
        ),
    )
    p.add_argument("--train_selection_fraction", type=float, default=0.2)
    p.add_argument("--validation_blocks", type=int, default=4)
    p.add_argument("--safe_min_improvement_pct", type=float, default=0.2)
    p.add_argument("--safe_min_positive_block_frac", type=float, default=0.75)
    p.add_argument("--target_margin_pct", type=float, default=0.0)
    p.add_argument(
        "--feature_mode",
        choices=["drift", "drift_residual", "drift_output", "causal_augmented"],
        default="causal_augmented",
    )
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_weight_decay", type=float, default=1e-4)
    p.add_argument("--probe_hidden", type=int, default=64,
                   help="0 for linear probe; >0 for a one-hidden-layer MLP")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--out_dir", default="logs/prompt_z/gate_probe")
    p.add_argument("--save_dir", default="weights/prompt_z/gate_probe")
    p.add_argument("--experiment_tag", default="")
    args = p.parse_args()

    if not (0.0 < args.calibration_fraction < 1.0):
        p.error("--calibration_fraction must be in (0,1)")
    if not (0.0 < args.train_selection_fraction < 0.5):
        p.error("--train_selection_fraction must be in (0,0.5)")
    if args.validation_blocks < 1:
        p.error("--validation_blocks must be >= 1")
    if args.target_margin_pct < 0:
        p.error("--target_margin_pct must be >= 0")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device={device}")

    if args.enc_in is None:
        import pandas as pd

        frame = pd.read_csv(os.path.join(args.root_path, args.data_path))
        args.enc_in = len([c for c in frame.columns if c.lower() != "date"])
    dataset = get_dataset(args)
    train_end = dataset.train_size
    gate_train_start = train_end - args.train_steps
    train_warm_start = max(0, gate_train_start - args.warmup_steps)
    val_start = dataset.val_start
    val_end = min(dataset.test_start, val_start + args.val_steps)
    val_warm_start = max(0, train_end - args.warmup_steps)
    if gate_train_start < 0 or val_end <= val_start:
        raise ValueError("Invalid temporal ranges for requested train/validation steps")

    adapter = build_backbone(args, device)
    args.hidden_layout = adapter.hidden_layout
    prompt_z = build_prompt_z(args, device)
    prompt_z.load_state_dict(torch.load(args.p1_ckpt, map_location=device))
    prompt_z.eval()
    for parameter in prompt_z.parameters():
        parameter.requires_grad = False
    print(f"[*] loaded frozen P1 delta: {args.p1_ckpt}")

    train_records = collect_causal_records(
        adapter,
        prompt_z,
        dataset,
        (train_warm_start, gate_train_start),
        (gate_train_start, train_end),
        args,
        device,
        "train",
    )
    val_records = collect_causal_records(
        adapter,
        prompt_z,
        dataset,
        (val_warm_start, train_end),
        (val_start, val_end),
        args,
        device,
        "validation",
    )
    if args.validation_protocol == "fixed_zero_blocked":
        train_split = int(
            len(train_records["features"]) * (1.0 - args.train_selection_fraction)
        )
        fit_records, selection_pool = split_records(train_records, train_split)
        safety_split = max(1, len(selection_pool["features"]) // 2)
        checkpoint_selection_records, safety_selection_records = split_records(
            selection_pool, safety_split
        )
        evaluation_records = val_records
    else:
        split_at = int(len(val_records["features"]) * args.calibration_fraction)
        checkpoint_selection_records, evaluation_records = split_records(
            val_records, split_at
        )
        safety_selection_records = checkpoint_selection_records
        fit_records = train_records

    classifier, cls_train = train_classifier(
        fit_records, checkpoint_selection_records, args, device
    )
    regressor, reg_train = train_regressor(
        fit_records, checkpoint_selection_records, args, device
    )
    regret_classifier, regret_train = train_regret_classifier(
        fit_records, checkpoint_selection_records, args, device
    )
    if args.validation_protocol == "fixed_zero_blocked":
        cls_threshold = 0.0
        reg_threshold = 0.0
        cls_selection_mse = mse_with_gate(
            checkpoint_selection_records,
            probe_scores(classifier, checkpoint_selection_records) > cls_threshold,
        )
        reg_selection_mse = mse_with_gate(
            checkpoint_selection_records,
            probe_scores(regressor, checkpoint_selection_records) > reg_threshold,
        )
    else:
        cls_selection_scores = probe_scores(
            classifier, checkpoint_selection_records
        )
        reg_selection_scores = probe_scores(
            regressor, checkpoint_selection_records
        )
        cls_threshold, cls_selection_mse = tune_threshold(
            cls_selection_scores, checkpoint_selection_records
        )
        reg_threshold, reg_selection_mse = tune_threshold(
            reg_selection_scores, checkpoint_selection_records
        )

    safe_mode, safe_selection_candidates, safe_selection_diagnostics = select_safe_mode(
        safety_selection_records,
        regressor,
        min_improvement_pct=args.safe_min_improvement_pct,
        min_positive_block_frac=args.safe_min_positive_block_frac,
    )

    block_results = []
    n_val = len(evaluation_records["features"])
    for block_index in range(args.validation_blocks):
        block_start = n_val * block_index // args.validation_blocks
        block_end = n_val * (block_index + 1) // args.validation_blocks
        block = {k: v[block_start:block_end] for k, v in evaluation_records.items()}
        block_results.append(
            evaluate_records(
                block,
                classifier,
                regressor,
                regret_classifier,
                cls_threshold,
                reg_threshold,
                0.0,
            )
        )

    result = {
        "config": vars(args),
        "ranges": {
            "train_warmup": [train_warm_start, gate_train_start],
            "gate_train": [gate_train_start, train_end],
            "validation_warmup": [val_warm_start, train_end],
            "model_selection_source": (
                "train_tail" if args.validation_protocol == "fixed_zero_blocked"
                else "validation_prefix"
            ),
            "validation_evaluation": (
                [val_start, val_end]
                if args.validation_protocol == "fixed_zero_blocked"
                else [val_start + split_at, val_end]
            ),
        },
        "classifier_training": {
            **cls_train,
            "selected_threshold": cls_threshold,
            "selection_mse": cls_selection_mse,
        },
        "regressor_training": {
            **reg_train,
            "selected_threshold": reg_threshold,
            "selection_mse": reg_selection_mse,
        },
        "regret_classifier_training": {
            **regret_train,
            "selected_threshold": 0.0,
        },
        "training_labels": {
            "n_windows": int(train_records["features"].shape[0]),
            "feature_dim": int(train_records["features"].shape[-1]),
            "valid_label_frac": train_records["valid"].float().mean().item(),
            "positive_label_frac": (
                train_records["label"][train_records["valid"]].mean().item()
            ),
        },
        "model_selection_block": evaluate_records(
            checkpoint_selection_records,
            classifier,
            regressor,
            regret_classifier,
            cls_threshold,
            reg_threshold,
            0.0,
        ),
        "safety_selection_block": {
            "n_windows": int(safety_selection_records["features"].shape[0]),
            "selected_mode": safe_mode,
            "candidate_mse": safe_selection_candidates,
            "diagnostics": safe_selection_diagnostics,
        },
        "safe_holdout_validation": evaluate_safe_mode(
            evaluation_records, safe_mode, regressor
        ),
        "holdout_validation": evaluate_records(
            evaluation_records,
            classifier,
            regressor,
            regret_classifier,
            cls_threshold,
            reg_threshold,
            0.0,
        ),
        "validation_blocks": block_results,
        "note": (
            "Binary gate checkpoint is selected entirely on train-tail blocks with a fixed "
            "zero threshold. Validation is report-only and TEST remains untouched."
        ),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    stem = (
        f"gate_probe_{args.data_path.replace('.csv','')}_H{args.forecast_H}_"
        f"{args.feature_mode}_s{args.seed}"
    )
    if args.experiment_tag:
        stem += f"_{args.experiment_tag}"
    out_path = os.path.join(args.out_dir, stem + ".json")
    model_path = os.path.join(args.save_dir, stem + ".pth")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    torch.save(
        {
            "classifier": classifier.state_dict(),
            "regressor": regressor.state_dict(),
            "regret_classifier": regret_classifier.state_dict(),
            "classifier_threshold": cls_threshold,
            "regressor_threshold": reg_threshold,
            "decision_threshold": 0.0,
            "selected_mode": safe_mode,
            "protocol_version": PROTOCOL_VERSION,
            "safety_selection": {
                "candidate_mse": safe_selection_candidates,
                "diagnostics": safe_selection_diagnostics,
            },
            "config": vars(args),
        },
        model_path,
    )
    print_summary(result)
    print(f"[*] result={out_path}")
    print(f"[*] probes={model_path}")


if __name__ == "__main__":
    main()
