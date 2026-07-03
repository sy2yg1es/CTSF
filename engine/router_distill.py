"""
Router Oracle Distillation — Stage 1 Training
==============================================

Stage 1 目标：冻结 expert bank，让 RichMLPRouter 学会模仿 oracle_routing 的选择。

OOM 修复（v2）
--------------
1. oracle label 生成：逐 expert 串行评估，每次只保留 [B,C] 标量 loss，
   不同时保留 32 个 Y_hat 张量。
2. samples 全部存 CPU，训练时按需 .to(device)，避免 ECL C=321 的 OOM。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional
from core.buffer import BDLABuffer


# ============================================================================
# Oracle Label Generation (OOM-safe: one expert at a time)
# ============================================================================

@torch.no_grad()
def generate_oracle_labels(
    backbone_adapter,
    prompts: Tensor,            # [E, D]  on device
    X_history: Tensor,          # [B, seq_len, C]  on device
    Y_current: Tensor,          # [B, pred_len, C] on device
    temperature: float = 1.0,
    noop_as_extra: bool = True,
) -> Tensor:
    """
    Per-channel oracle soft labels. OOM-safe: evaluates one expert at a time,
    accumulates scalar [B,C] losses in CPU, stacks at the end.

    Returns
    -------
    soft_labels : [B, C, E+1] CPU tensor
    """
    device = X_history.device
    E, D = prompts.shape
    B, pred_len, C = Y_current.shape

    # Encode once — shared across all experts
    H_tokens, z_query, means, stdev = backbone_adapter.encode(X_history)
    # H_tokens: [B, C, D, P]  — stays on GPU for fuse_and_decode

    loss_per_expert_cpu = []  # accumulate [B, C] on CPU

    # --- Real experts ---
    for e_idx in range(E):
        theta_e = prompts[e_idx].unsqueeze(0).unsqueeze(0).expand(B, C, -1)  # [B, C, D]
        Y_hat_e = backbone_adapter.fuse_and_decode(H_tokens, theta_e, means, stdev)
        L_e = (Y_hat_e - Y_current).pow(2).mean(dim=1).cpu()  # [B, C] on CPU
        loss_per_expert_cpu.append(L_e)
        del Y_hat_e, L_e

    # --- No-op expert ---
    if noop_as_extra:
        Y_hat_frozen = backbone_adapter.forward_frozen(X_history)
        L_noop = (Y_hat_frozen - Y_current).pow(2).mean(dim=1).cpu()
        loss_per_expert_cpu.append(L_noop)
        del Y_hat_frozen, L_noop

    # Stack on CPU: [B, C, E(+1)]
    loss_stack = torch.stack(loss_per_expert_cpu, dim=-1)  # CPU
    soft_labels = F.softmax(-loss_stack / temperature, dim=-1)
    return soft_labels  # CPU tensor


# ============================================================================
# Training Losses
# ============================================================================

def oracle_kl_loss(router_logits: Tensor, soft_labels: Tensor,
                   temperature: float = 1.0) -> Tensor:
    log_probs = F.log_softmax(router_logits / temperature, dim=-1)
    return F.kl_div(log_probs, soft_labels.to(router_logits.device), reduction='batchmean')


def noop_margin_loss(router_logits: Tensor, soft_labels: Tensor,
                     noop_idx: int = -1, margin: float = 0.1) -> Tensor:
    sl = soft_labels.to(router_logits.device)
    oracle_noop_best = (sl[..., noop_idx] >= sl[..., :noop_idx].max(dim=-1).values)
    if not oracle_noop_best.any():
        return torch.tensor(0.0, device=router_logits.device)
    logit_noop = router_logits[..., noop_idx]
    logit_best_real = router_logits[..., :noop_idx].max(dim=-1).values
    deficit = logit_best_real - logit_noop + margin
    return (F.relu(deficit) * oracle_noop_best.float()).mean()


def load_balance_loss(routing_probs: Tensor, noop_idx: Optional[int] = None,
                      alpha: float = 1e-2) -> Tensor:
    probs_real = routing_probs[..., :noop_idx] if noop_idx is not None else routing_probs
    E_real = probs_real.shape[-1]
    P_e = probs_real.mean(dim=(0, 1))                                     # [E_real]
    am = probs_real.argmax(dim=-1)                                        # [B, C]
    arange = torch.arange(E_real, device=probs_real.device)               # [E_real]
    f_e = (am.unsqueeze(-1) == arange.view(1, 1, E_real)).float().mean(dim=(0, 1))
    return alpha * (f_e * P_e).sum()



# ============================================================================
# Shared: collect aligned samples (stored on CPU)
# ============================================================================

def _collect_samples(model, backbone_adapter, dataloader, train_size, device):
    """
    Run one pass through the streaming dataloader to collect aligned
    (X_history, Y_current, z_features) on CPU.

    z_features = [mean, last, last-first, std] of H_patches: [B, C, 4D]
    """
    pred_len = getattr(backbone_adapter.backbone, 'pred_len',
                       getattr(backbone_adapter.backbone, 'forecast_H', 96))
    buffer = BDLABuffer(horizon_H=pred_len)
    samples = []  # list of (X_cpu, Y_cpu, z_feat_cpu)

    print("[Collect] Streaming through data...")
    model.eval()
    with torch.no_grad():
        for t, (X_t, Y_t) in enumerate(dataloader):
            X_t = X_t.to(device)
            Y_hat_future, z_channel, routing_probs, dispatch_indices = model(X_t)
            buffer.push(t=t, X_t=X_t, y_hat_future=Y_hat_future,
                        dispatch_indices=dispatch_indices, z_t=z_channel)

            if Y_t is None:
                continue
            if buffer.get_stored_prediction(t) is None:
                continue
            aligned = buffer.pop_and_align(t, Y_t)
            if aligned is None:
                continue
            if t < train_size:
                continue

            X_history, Y_current, _, z_query = aligned

            # Compute z_features from H_patches (on GPU, then move to CPU)
            H_tokens, _, means, stdev = backbone_adapter.encode(X_history)
            h_mean  = H_tokens.mean(dim=-1)
            h_last  = H_tokens[..., -1]
            h_first = H_tokens[..., 0]
            h_std   = H_tokens.std(dim=-1, unbiased=False)
            z_features = torch.cat([h_mean, h_last, h_last - h_first, h_std], dim=-1)

            samples.append((
                X_history.cpu(),
                Y_current.cpu(),
                z_features.cpu(),   # [B, C, 4D]
                means.cpu(),
                stdev.cpu(),
            ))

    print(f"[Collect] {len(samples)} aligned samples collected (stored on CPU).")
    return samples


# ============================================================================
# Stage 1: Oracle Distillation
# ============================================================================

def run_stage1_distillation(
    model,
    backbone_adapter,
    dataloader,
    optimizer,
    train_size: int,
    device,
    lambda_forecast: float = 1.0,
    lambda_kl: float = 2.0,
    lambda_bal: float = 0.01,
    lambda_smooth: float = 0.1,
    lambda_noop: float = 0.5,
    oracle_temperature: float = 1.0,
    router_temperature: float = 1.0,
    epochs: int = 3,
    log_interval: int = 200,
) -> Dict[str, List[float]]:
    """Stage 1: freeze expert bank + backbone, train only MLP-router via distillation."""

    # Freeze everything except router
    for name, p in model.named_parameters():
        p.requires_grad = ('router' in name)
    for p in backbone_adapter.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[Stage 1] Trainable params: {sum(p.numel() for p in trainable)}")

    prompts = model.prompt_memory.prompts.detach()   # [E, D], no grad
    E = prompts.shape[0]
    noop_idx = E

    # Collect samples on CPU
    samples = _collect_samples(model, backbone_adapter, dataloader, train_size, device)

    history = {'loss_total': [], 'loss_kl': [], 'loss_forecast': [], 'noop_ratio': []}

    for epoch in range(epochs):
        total_loss = total_kl = total_fc = total_noop = 0.0
        n = 0

        if hasattr(model.prompt_memory.router, 'reset_cache'):
            model.prompt_memory.router.reset_cache()

        for step_idx, (X_cpu, Y_cpu, z_feat_cpu, means_cpu, stdev_cpu) in enumerate(samples):
            model.train()
            optimizer.zero_grad()

            # Move only what's needed to GPU
            X_history  = X_cpu.to(device)
            Y_current  = Y_cpu.to(device)
            z_features = z_feat_cpu.to(device)   # [B, C, 4D]
            means      = means_cpu.to(device)
            stdev      = stdev_cpu.to(device)

            # Oracle labels (one expert at a time, returns CPU tensor)
            with torch.no_grad():
                # Re-encode for H_tokens (needed by oracle + forecast)
                H_tokens, _, _, _ = backbone_adapter.encode(X_history)
                soft_labels = generate_oracle_labels(
                    backbone_adapter, prompts,
                    X_history, Y_current,
                    temperature=oracle_temperature,
                    noop_as_extra=True,
                )  # [B, C, E+1] CPU

            # Router forward (GPU)
            router_logits = model.prompt_memory.router(z_features, None)   # [B, C, E+1]
            router_probs  = F.softmax(router_logits / router_temperature, dim=-1)

            # KL loss (soft_labels moved to GPU inside oracle_kl_loss)
            loss_kl = oracle_kl_loss(router_logits, soft_labels, router_temperature)

            # Forecast loss (top-k expert, detached prompts → no grad through bank)
            with torch.no_grad():
                top_k = model.prompt_memory.top_k
                _, top_idx = torch.topk(router_probs.detach(), k=top_k, dim=-1)
                top_w = torch.gather(router_probs.detach(), -1, top_idx)
                top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                all_p = torch.cat([prompts, model.prompt_memory.noop_prompt], dim=0) \
                        if model.prompt_memory.use_noop else prompts
                gathered = all_p[top_idx]
                theta = torch.einsum('bck,bckd->bcd', top_w, gathered)
            Y_hat = backbone_adapter.fuse_and_decode(H_tokens.detach(), theta, means, stdev)
            loss_forecast = (Y_hat - Y_current).pow(2).mean()

            # Temporal smoothness
            loss_smooth = (model.prompt_memory.router.get_smoothness_loss(router_logits)
                           if hasattr(model.prompt_memory.router, 'get_smoothness_loss')
                           else torch.tensor(0.0, device=device))

            # No-op margin
            loss_noop_m = noop_margin_loss(router_logits, soft_labels,
                                           noop_idx=noop_idx, margin=0.1)

            # Load balance (real experts only)
            loss_bal = load_balance_loss(router_probs, noop_idx=noop_idx, alpha=1.0)

            loss = (lambda_forecast * loss_forecast
                    + lambda_kl     * loss_kl
                    + lambda_bal    * loss_bal
                    + lambda_smooth * loss_smooth
                    + lambda_noop   * loss_noop_m)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            noop_sel = (router_probs.argmax(dim=-1) == noop_idx).float().mean().item()
            total_loss += loss.item(); total_kl += loss_kl.item()
            total_fc   += loss_forecast.item(); total_noop += noop_sel
            n += 1

            if (step_idx + 1) % log_interval == 0:
                a = lambda x: x / n
                print(f"  [E{epoch+1} S{step_idx+1}] loss={a(total_loss):.4f} "
                      f"kl={a(total_kl):.4f} fc={a(total_fc):.4f} "
                      f"noop={a(total_noop)*100:.1f}%")

        a = lambda x: x / max(n, 1)
        print(f"[Stage 1 Epoch {epoch+1}] "
              f"loss={a(total_loss):.4f} kl={a(total_kl):.4f} "
              f"fc={a(total_fc):.4f} noop_ratio={a(total_noop)*100:.1f}%")
        history['loss_total'].append(a(total_loss))
        history['loss_kl'].append(a(total_kl))
        history['loss_forecast'].append(a(total_fc))
        history['noop_ratio'].append(a(total_noop))

    return history


def run_router_sanity_overfit(
    model,
    backbone_adapter,
    dataloader,
    optimizer,
    train_size: int,
    device,
    max_samples: int = 512,
    epochs: int = 20,
    oracle_temperature: float = 1.0,
    log_interval: int = 1,
) -> Dict[str, List[float]]:
    """
    Small-subset sanity check: can the router overfit posterior oracle labels?

    If this cannot drive entropy down and top-1 accuracy up on 512/2048 windows,
    the issue is likely implementation, checkpoint, gradients, or loss wiring
    rather than generalization.
    """
    for name, p in model.named_parameters():
        p.requires_grad = ('router' in name)
    for p in backbone_adapter.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[Sanity] Trainable router params: {sum(p.numel() for p in trainable)}")

    samples = _collect_samples(model, backbone_adapter, dataloader, train_size, device)
    samples = samples[:max_samples]
    print(f"[Sanity] Overfit samples: {len(samples)}")

    prompts = model.prompt_memory.prompts.detach()
    noop_idx = prompts.shape[0]
    history = {'loss_ce': [], 'top1_acc': [], 'top3_recall': [], 'entropy': [], 'noop_ratio': []}

    for epoch in range(epochs):
        total_loss = total_acc = total_top3 = total_entropy = total_noop = 0.0
        total_items = 0

        if hasattr(model.prompt_memory.router, 'reset_cache'):
            model.prompt_memory.router.reset_cache()

        for X_cpu, Y_cpu, z_feat_cpu, _, _ in samples:
            model.train()
            optimizer.zero_grad()

            X_history = X_cpu.to(device)
            Y_current = Y_cpu.to(device)
            z_features = z_feat_cpu.to(device)

            with torch.no_grad():
                soft_labels = generate_oracle_labels(
                    backbone_adapter, prompts,
                    X_history, Y_current,
                    temperature=oracle_temperature,
                    noop_as_extra=True,
                )
                hard_labels = soft_labels.argmax(dim=-1).to(device)  # [B, C]

            logits = model.prompt_memory.router(z_features, None)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                hard_labels.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                top1 = probs.argmax(dim=-1)
                top3 = torch.topk(probs, k=min(3, probs.shape[-1]), dim=-1).indices
                entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
                n_items = hard_labels.numel()
                total_loss += loss.item() * n_items
                total_acc += (top1 == hard_labels).sum().item()
                total_top3 += (top3 == hard_labels.unsqueeze(-1)).any(dim=-1).sum().item()
                total_entropy += entropy.sum().item()
                total_noop += (top1 == noop_idx).sum().item()
                total_items += n_items

        avg_loss = total_loss / max(total_items, 1)
        top1_acc = total_acc / max(total_items, 1)
        top3_recall = total_top3 / max(total_items, 1)
        entropy_avg = total_entropy / max(total_items, 1)
        noop_ratio = total_noop / max(total_items, 1)

        history['loss_ce'].append(avg_loss)
        history['top1_acc'].append(top1_acc)
        history['top3_recall'].append(top3_recall)
        history['entropy'].append(entropy_avg)
        history['noop_ratio'].append(noop_ratio)

        if (epoch + 1) % log_interval == 0:
            print(f"[Sanity E{epoch+1}] ce={avg_loss:.4f} "
                  f"top1={top1_acc:.4f} top3={top3_recall:.4f} "
                  f"entropy={entropy_avg:.4f} noop={noop_ratio*100:.2f}%")

    return history


# ============================================================================
# Stage 2: Small-LR Joint Fine-tuning
# ============================================================================

def run_stage2_joint(
    model,
    backbone_adapter,
    dataloader,
    optimizer_router,
    optimizer_experts,
    train_size: int,
    device,
    lambda_forecast: float = 1.0,
    lambda_kl: float = 1.0,
    lambda_bal: float = 0.01,
    lambda_smooth: float = 0.1,
    lambda_noop: float = 0.5,
    oracle_temperature: float = 1.0,
    router_temperature: float = 1.0,
    epochs: int = 2,
    log_interval: int = 200,
) -> Dict[str, List[float]]:
    """Stage 2: unfreeze expert bank (small LR), keep router trainable."""

    for name, p in model.named_parameters():
        if name == 'prompt_memory.prompts':
            p.requires_grad = True
            print(f"[Stage 2] Unfrozen: {name}")
        elif 'router' in name:
            p.requires_grad = True
        elif 'noop_prompt' in name:
            p.requires_grad = False

    E = model.prompt_memory.prompts.shape[0]
    noop_idx = E
    history = {'loss_total': [], 'loss_kl': []}

    samples = _collect_samples(model, backbone_adapter, dataloader, train_size, device)
    print(f"[Stage 2] {len(samples)} samples. Joint training...")

    for epoch in range(epochs):
        total_loss = 0.0; n = 0

        if hasattr(model.prompt_memory.router, 'reset_cache'):
            model.prompt_memory.router.reset_cache()

        for step_idx, (X_cpu, Y_cpu, z_feat_cpu, means_cpu, stdev_cpu) in enumerate(samples):
            model.train()
            optimizer_router.zero_grad()
            optimizer_experts.zero_grad()

            X_history  = X_cpu.to(device)
            Y_current  = Y_cpu.to(device)
            z_features = z_feat_cpu.to(device)
            means      = means_cpu.to(device)
            stdev      = stdev_cpu.to(device)
            prompts    = model.prompt_memory.prompts  # grad enabled

            with torch.no_grad():
                H_tokens, _, _, _ = backbone_adapter.encode(X_history)
                soft_labels = generate_oracle_labels(
                    backbone_adapter, prompts.detach(),
                    X_history, Y_current,
                    temperature=oracle_temperature, noop_as_extra=True,
                )  # CPU

            router_logits = model.prompt_memory.router(z_features, None)
            router_probs  = F.softmax(router_logits / router_temperature, dim=-1)

            loss_kl     = oracle_kl_loss(router_logits, soft_labels, router_temperature)
            loss_smooth = (model.prompt_memory.router.get_smoothness_loss(router_logits)
                           if hasattr(model.prompt_memory.router, 'get_smoothness_loss')
                           else torch.tensor(0.0, device=device))
            loss_noop_m = noop_margin_loss(router_logits, soft_labels, noop_idx=noop_idx)
            loss_bal    = load_balance_loss(router_probs, noop_idx=noop_idx, alpha=1.0)

            # Forecast: grad flows through expert bank this time
            top_k = model.prompt_memory.top_k
            _, top_idx = torch.topk(router_probs, k=top_k, dim=-1)
            top_w = torch.gather(router_probs, -1, top_idx)
            top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            all_p = torch.cat([prompts, model.prompt_memory.noop_prompt], dim=0) \
                    if model.prompt_memory.use_noop else prompts
            theta = torch.einsum('bck,bckd->bcd', top_w, all_p[top_idx])
            Y_hat = backbone_adapter.fuse_and_decode(H_tokens.detach(), theta, means, stdev)
            loss_forecast = (Y_hat - Y_current).pow(2).mean()

            loss = (lambda_forecast * loss_forecast
                    + lambda_kl     * loss_kl
                    + lambda_bal    * loss_bal
                    + lambda_smooth * loss_smooth
                    + lambda_noop   * loss_noop_m)

            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer_router.step()
            optimizer_experts.step()

            total_loss += loss.item(); n += 1

            if (step_idx + 1) % log_interval == 0:
                print(f"  [Stage2 E{epoch+1} S{step_idx+1}] loss={total_loss/n:.4f}")

        print(f"[Stage 2 Epoch {epoch+1}] avg_loss={total_loss/max(n,1):.4f}")
        history['loss_total'].append(total_loss / max(n, 1))

    return history
