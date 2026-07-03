"""
Expert Bank & Oracle Label Diagnostics
=======================================

回答一个关键问题：oracle label 本身是否有可学信号？
如果 expert bank 的 prompts 都是 0.02*randn 噪声，所有 expert 产生几乎相同 MSE，
那 oracle soft label 就是近均匀分布，KL 训练不可能收敛。

诊断项：
1. Expert diversity: cosine similarity matrix, mean pairwise sim
2. Expert norm: 每个 expert 的 L2 norm
3. Per-window oracle label statistics:
   - entropy of oracle soft labels (vs max entropy log(E+1))
   - argmax concentration (top-1 expert 占多大比例)
   - loss variance across experts (同一个 window 不同 expert 的 loss 差距)
4. No-op dominance: oracle 多少步认为 noop 是最好的
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from data_provider.data_loader import data_provider
from data_provider.streaming_env import StreamingEnvironment
from models.backbones.PatchTST import Model as PatchTSTModel
from models.backbone_adapter import PatchTSTAdapter
from core.buffer import BDLABuffer


def expert_diversity(prompts):
    """Cosine similarity matrix between all expert pairs."""
    E, D = prompts.shape
    norms = prompts.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = prompts / norms
    sim_matrix = normed @ normed.T  # [E, E]
    mask = ~torch.eye(E, dtype=torch.bool)
    mean_sim = sim_matrix[mask].mean().item()
    std_sim = sim_matrix[mask].std().item()
    max_sim = sim_matrix[mask].max().item()
    norms_flat = norms.squeeze().tolist()
    return {
        'mean_pairwise_cosine_sim': mean_sim,
        'std_pairwise_cosine_sim': std_sim,
        'max_pairwise_cosine_sim': max_sim,
        'expert_norms': {
            'mean': float(np.mean(norms_flat)),
            'std': float(np.std(norms_flat)),
            'min': float(np.min(norms_flat)),
            'max': float(np.max(norms_flat)),
        },
    }


@torch.no_grad()
def oracle_label_signal_check(
    backbone_adapter,
    prompts,       # [E, D]
    X_batch,       # [B, seq_len, C]
    Y_batch,       # [B, pred_len, C]
    temperature=1.0,
):
    """
    Compute per-expert losses and oracle soft labels for one window.
    Returns statistics about signal quality.
    """
    device = X_batch.device
    E, D = prompts.shape
    B, pred_len, C = Y_batch.shape

    H_tokens, z_query, means, stdev = backbone_adapter.encode(X_batch)

    losses = []  # [E] each [B, C]
    for e_idx in range(E):
        theta_e = prompts[e_idx].unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        Y_hat_e = backbone_adapter.fuse_and_decode(H_tokens, theta_e, means, stdev)
        L_e = (Y_hat_e - Y_batch).pow(2).mean(dim=1)  # [B, C]
        losses.append(L_e.cpu())
        del Y_hat_e

    # Noop
    Y_hat_frozen = backbone_adapter.forward_frozen(X_batch)
    L_noop = (Y_hat_frozen - Y_batch).pow(2).mean(dim=1).cpu()
    losses.append(L_noop)

    # [B, C, E+1]
    loss_stack = torch.stack(losses, dim=-1)

    # Oracle soft labels
    soft_labels = F.softmax(-loss_stack / temperature, dim=-1)

    # Entropy per (b, c)
    log_probs = (soft_labels + 1e-10).log()
    entropy = -(soft_labels * log_probs).sum(dim=-1)
    max_entropy = float(np.log(E + 1))

    # Argmax stats
    argmax_idx = soft_labels.argmax(dim=-1)
    noop_is_best = (argmax_idx == E).float().mean().item()

    # Top-1 oracle expert concentration
    top1_probs = soft_labels.max(dim=-1).values

    # Loss variance across experts
    loss_var = loss_stack.var(dim=-1)
    loss_mean = loss_stack.mean(dim=-1)
    loss_cv = loss_stack.std(dim=-1) / loss_mean.clamp(min=1e-8)

    return {
        'entropy_mean': float(entropy.mean()),
        'entropy_std': float(entropy.std()),
        'max_entropy': max_entropy,
        'entropy_ratio': float(entropy.mean() / max_entropy),
        'noop_is_best_ratio': noop_is_best,
        'top1_oracle_prob_mean': float(top1_probs.mean()),
        'top1_oracle_prob_std': float(top1_probs.std()),
        'loss_cv_mean': float(loss_cv.mean()),
        'loss_cv_std': float(loss_cv.std()),
        'loss_var_mean': float(loss_var.mean()),
        'loss_mean_mean': float(loss_mean.mean()),
        'expert_argmax_histogram': torch.bincount(
            argmax_idx.view(-1), minlength=E+1
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='./data')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default='M')
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--forecast_H", type=int, default=96)
    parser.add_argument("--D_model", type=int, default=512)
    parser.add_argument("--num_experts", type=int, default=32)
    parser.add_argument("--target", type=str, default='OT')
    parser.add_argument("--oracle_temp", type=float, default=1.0)
    parser.add_argument("--n_windows", type=int, default=200,
                        help="Number of test windows to evaluate")
    parser.add_argument("--pm_weights", type=str, default=None,
                        help="Path to prompt_memory state dict (stage1 or stage2)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Bridges
    args.pred_len = args.forecast_H
    args.d_model = args.D_model
    args.task_name = 'long_term_forecast'
    args.dropout = 0.1; args.factor = 3; args.n_heads = 8
    args.d_ff = 512; args.e_layers = 3; args.activation = 'gelu'
    args.num_class = 1; args.embed = 'timeF'; args.freq = 'h'
    args.batch_size = 1; args.accum_steps = 1; args.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    dataset, base_loader = data_provider(args)
    actual_features = dataset.data_x.shape[1]
    args.enc_in = actual_features
    train_size = int(len(dataset) * 0.5)

    streaming_env = StreamingEnvironment(base_loader, forecast_H=args.forecast_H)

    backbone = PatchTSTModel(configs=args, patch_len=16, stride=8).to(device)
    file_name = args.data_path.split('.')[0]
    weight_path = f"./weights/patchtst_pretrained_{file_name}_H{args.forecast_H}.pth"
    try:
        backbone.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"[*] Backbone loaded: {weight_path}")
    except FileNotFoundError:
        print(f"[!] No backbone weights: {weight_path}")
    backbone_adapter = PatchTSTAdapter(backbone, use_query_mlp=False)

    # Load expert bank
    if args.pm_weights:
        pm_state = torch.load(args.pm_weights, map_location=device)
        if 'prompts' in pm_state:
            prompts = pm_state['prompts'].to(device)
            print(f"[*] Loaded expert bank from: {args.pm_weights}")
            print(f"    prompts shape: {prompts.shape}")
        else:
            print(f"[!] No 'prompts' key. Keys: {list(pm_state.keys())[:10]}")
            prompts = torch.randn(args.num_experts, args.D_model, device=device) * 0.02
    else:
        prompts = torch.randn(args.num_experts, args.D_model, device=device) * 0.02
        print(f"[*] Using random init expert bank (0.02 * randn)")

    E, D = prompts.shape
    print(f"\n{'='*60}")
    print(f"Expert Bank Diagnostics: {file_name} H={args.forecast_H}")
    print(f"{'='*60}")

    # 1. Expert diversity
    div = expert_diversity(prompts.cpu())
    print(f"\n--- Expert Diversity ({E} experts, D={D}) ---")
    print(f"  Mean pairwise cosine sim: {div['mean_pairwise_cosine_sim']:.4f}")
    print(f"  Max  pairwise cosine sim: {div['max_pairwise_cosine_sim']:.4f}")
    print(f"  Expert norm mean: {div['expert_norms']['mean']:.4f}")
    print(f"  Expert norm range: [{div['expert_norms']['min']:.4f}, {div['expert_norms']['max']:.4f}]")

    # 2. Oracle label signal
    print(f"\n--- Oracle Label Signal ({args.n_windows} windows, T={args.oracle_temp}) ---")
    all_stats = []
    buffer = BDLABuffer(horizon_H=args.forecast_H)
    n_collected = 0

    for t, (X_t, Y_t) in enumerate(streaming_env):
        X_t = X_t.to(device)
        with torch.no_grad():
            H_tokens, z_query, means, stdev = backbone_adapter.encode(X_t)
            theta_zero = torch.zeros(1, actual_features, D, device=device)
            Y_hat = backbone_adapter.fuse_and_decode(H_tokens, theta_zero, means, stdev)

        buffer.push(t=t, X_t=X_t, y_hat_future=Y_hat,
                    dispatch_indices=torch.zeros(1, actual_features, 2, dtype=torch.long, device=device),
                    z_t=z_query)

        if Y_t is None:
            continue
        if buffer.get_stored_prediction(t) is None:
            continue
        aligned = buffer.pop_and_align(t, Y_t)
        if aligned is None:
            continue
        if t < train_size:
            continue

        X_hist, Y_curr, _, _ = aligned
        stats = oracle_label_signal_check(
            backbone_adapter, prompts,
            X_hist.to(device), Y_curr.to(device),
            temperature=args.oracle_temp,
        )
        all_stats.append(stats)
        n_collected += 1

        if n_collected % 50 == 0:
            print(f"  [{n_collected}/{args.n_windows}] "
                  f"entropy_ratio={stats['entropy_ratio']:.3f} "
                  f"noop_best={stats['noop_is_best_ratio']:.2f} "
                  f"loss_cv={stats['loss_cv_mean']:.4f}")

        if n_collected >= args.n_windows:
            break

    # Aggregate
    agg = {}
    for key in ['entropy_mean', 'entropy_ratio', 'noop_is_best_ratio',
                'top1_oracle_prob_mean', 'loss_cv_mean', 'loss_var_mean',
                'loss_mean_mean']:
        vals = [s[key] for s in all_stats]
        agg[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    hist_total = np.zeros(E + 1)
    for s in all_stats:
        hist_total += np.array(s['expert_argmax_histogram'])
    hist_total /= hist_total.sum()

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS ({n_collected} windows)")
    print(f"{'='*60}")
    print(f"  Entropy ratio (1.0=uniform): {agg['entropy_ratio']['mean']:.4f} ± {agg['entropy_ratio']['std']:.4f}")
    print(f"  Top-1 oracle prob:           {agg['top1_oracle_prob_mean']['mean']:.4f} ± {agg['top1_oracle_prob_mean']['std']:.4f}")
    print(f"  Loss CV (higher=more signal):{agg['loss_cv_mean']['mean']:.6f} ± {agg['loss_cv_mean']['std']:.6f}")
    print(f"  Loss variance:               {agg['loss_var_mean']['mean']:.8f}")
    print(f"  Noop is best:                {agg['noop_is_best_ratio']['mean']*100:.1f}%")
    print(f"  Expert argmax distribution:  top3 = {np.argsort(hist_total)[-3:][::-1]} "
          f"with probs {np.sort(hist_total)[-3:][::-1]}")
    print(f"  Max entropy = {float(np.log(E+1)):.3f}")

    er = agg['entropy_ratio']['mean']
    lcv = agg['loss_cv_mean']['mean']
    if er > 0.95:
        verdict = "FATAL: Oracle labels near-uniform. Expert bank has NO differentiation."
        advice = "Experts need pretraining/diversification BEFORE router distillation."
    elif er > 0.8:
        verdict = "WEAK: Some signal but very noisy. Router training will be very slow."
        advice = "Consider increasing oracle temperature or diversifying experts."
    elif er > 0.5:
        verdict = "MODERATE: Reasonable signal. Router should be learnable."
        advice = "Check gradient flow and learning rate."
    else:
        verdict = "STRONG: Clear expert specialization. Router should learn well."
        advice = "If router still fails, it's a training bug."

    print(f"\n  VERDICT: {verdict}")
    print(f"  ADVICE:  {advice}")

    results = {
        'dataset': file_name,
        'horizon': args.forecast_H,
        'num_experts': E,
        'd_model': D,
        'pm_weights': args.pm_weights,
        'expert_diversity': div,
        'aggregated': agg,
        'expert_argmax_distribution': hist_total.tolist(),
        'verdict': verdict,
        'n_windows': n_collected,
    }

    out_path = args.output or f"logs/expert_diagnostics_{file_name}_H{args.forecast_H}.json"
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Saved to {out_path}")


if __name__ == '__main__':
    main()
