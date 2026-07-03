"""
pretrain_router.py — Stage 1 & 2 Router Training Entry Point
============================================================

用法：
  Stage 1 (oracle distillation, router only):
    python pretrain_router.py \\
        --root_path ./data --data_path ECL.csv \\
        --forecast_H 96 --D_model 512 \\
        --stage 1 --epochs 3 \\
        --lambda_kl 2.0 --lambda_noop 0.5

  Stage 2 (joint fine-tuning, router + expert bank):
    python pretrain_router.py \\
        --root_path ./data --data_path ECL.csv \\
        --forecast_H 96 --D_model 512 \\
        --stage 2 --epochs 2 \\
        --router_lr 1e-4 --expert_lr 1e-5 \\
        --router_weights ./weights/router_stage1_ECL_H96.pth

  Stage 12 (1 then 2 in one shot):
    python pretrain_router.py ... --stage 12
"""

import argparse
import os
import torch
import torch.optim as optim

from data_provider.data_loader import data_provider
from data_provider.streaming_env import StreamingEnvironment
from models.backbones.PatchTST import Model as PatchTSTModel
from models.backbone_adapter import PatchTSTAdapter
from models.prompts.prompt_pool import SparsePromptMemory
from models.mlp_router import RichMLPRouter
from models.framework import ContinualPromptTSF
from engine.router_distill import (
    run_stage1_distillation, run_stage2_joint, run_router_sanity_overfit,
)


def build_model(args, device, router_weights_path=None, prompt_memory_weights_path=None):
    """Build ContinualPromptTSF with RichMLPRouter."""
    # Backbone
    backbone = PatchTSTModel(configs=args, patch_len=16, stride=8).to(device)
    file_name = args.data_path.split('.')[0]
    weight_path = f"./weights/patchtst_pretrained_{file_name}_H{args.forecast_H}.pth"
    try:
        backbone.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"[*] Backbone weights loaded: {weight_path}")
    except FileNotFoundError:
        print(f"[!] Backbone weights not found: {weight_path}. Using random init.")

    backbone_adapter = PatchTSTAdapter(backbone, use_query_mlp=False)

    # Rich MLP Router
    rich_router = RichMLPRouter(
        d_model=args.D_model,
        num_experts=args.num_experts,
        hist_K=args.window_K,
        hidden=args.router_hidden,
        hist_hidden=args.router_hist_hidden,
        dropout=args.router_dropout,
    ).to(device)

    if router_weights_path and os.path.exists(router_weights_path):
        rich_router.load_state_dict(torch.load(router_weights_path, map_location=device))
        print(f"[*] Router weights loaded: {router_weights_path}")

    # SparsePromptMemory with injectable router + noop
    prompt_memory = SparsePromptMemory(
        prompt_dim=args.D_model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        temperature=args.temp_T,
        load_balancing_alpha=args.l_aux_weight,
        rich_router=rich_router,
        use_noop=True,
    ).to(device)

    if prompt_memory_weights_path:
        file_name = args.data_path.split('.')[0]
        if prompt_memory_weights_path == 'auto':
            prompt_memory_weights_path = (
                f"./weights/prompt_memory_stage{args.router_stage}_"
                f"{file_name}_H{args.forecast_H}.pth"
            )
        print(f"[*] prompt_memory checkpoint path: {prompt_memory_weights_path}")
        if os.path.exists(prompt_memory_weights_path):
            state = torch.load(prompt_memory_weights_path, map_location=device)
            incompatible = prompt_memory.load_state_dict(state, strict=False)
            print(f"[*] prompt_memory loaded: {prompt_memory_weights_path}")
            print(f"[*] missing_keys: {list(incompatible.missing_keys)}")
            print(f"[*] unexpected_keys: {list(incompatible.unexpected_keys)}")
        else:
            print(f"[!] prompt_memory weights not found: {prompt_memory_weights_path}")

    model = ContinualPromptTSF(
        backbone_adapter=backbone_adapter,
        prompt_memory=prompt_memory,
        adapter=None,
    ).to(device)

    return model, backbone_adapter


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[*] seed: {args.seed}")

    # Data
    dataset, base_loader = data_provider(args)
    actual_features = dataset.data_x.shape[1]
    args.enc_in = actual_features
    train_size = int(len(dataset) * 0.5)

    streaming_env = StreamingEnvironment(base_loader, forecast_H=args.forecast_H)

    def dl_gen():
        for x, y in streaming_env:
            yield x, y

    # Build model
    router_weights = getattr(args, 'router_weights', None)
    pm_weights = getattr(args, 'prompt_memory_weights', None)
    model, backbone_adapter = build_model(args, device, router_weights, pm_weights)

    file_name = args.data_path.split('.')[0]

    # ---- Sanity overfit ----
    if args.stage == 'sanity':
        print("\n" + "="*60)
        print(f"[Sanity] Router Oracle-Label Overfit - {file_name} H={args.forecast_H}")
        print("="*60)
        print(f"[*] max_samples={args.sanity_samples} epochs={args.epochs}")
        print(f"[*] router_lr={args.router_lr} temp_T={args.temp_T} top_k={args.top_k}")

        router_params = [p for p in model.prompt_memory.router.parameters()]
        opt_sanity = optim.Adam(router_params, lr=args.router_lr)
        history = run_router_sanity_overfit(
            model=model,
            backbone_adapter=backbone_adapter,
            dataloader=dl_gen(),
            optimizer=opt_sanity,
            train_size=train_size,
            device=device,
            max_samples=args.sanity_samples,
            epochs=args.epochs,
            oracle_temperature=args.oracle_temp,
            log_interval=args.log_interval,
        )
        print("[Sanity] Final: "
              f"ce={history['loss_ce'][-1]:.4f} "
              f"top1={history['top1_acc'][-1]:.4f} "
              f"top3={history['top3_recall'][-1]:.4f} "
              f"entropy={history['entropy'][-1]:.4f} "
              f"noop={history['noop_ratio'][-1]*100:.2f}%")
        return

    # ---- Stage 1 ----
    if '1' in args.stage:
        print("\n" + "="*60)
        print(f"[Stage 1] Oracle Distillation — {file_name} H={args.forecast_H}")
        print("="*60)

        router_params = [p for p in model.prompt_memory.router.parameters()]
        opt_stage1 = optim.Adam(router_params, lr=args.router_lr)

        history = run_stage1_distillation(
            model=model,
            backbone_adapter=backbone_adapter,
            dataloader=dl_gen(),
            optimizer=opt_stage1,
            train_size=train_size,
            device=device,
            lambda_forecast=args.lambda_forecast,
            lambda_kl=args.lambda_kl,
            lambda_bal=args.lambda_bal,
            lambda_smooth=args.lambda_smooth,
            lambda_noop=args.lambda_noop,
            oracle_temperature=args.oracle_temp,
            router_temperature=args.temp_T,
            epochs=args.epochs,
            log_interval=args.log_interval,
        )

        save_router = f"./weights/router_stage1_{file_name}_H{args.forecast_H}.pth"
        os.makedirs('./weights', exist_ok=True)
        torch.save(model.prompt_memory.router.state_dict(), save_router)

        # Save full prompt_memory (router + prompts) for streaming eval
        save_pm = f"./weights/prompt_memory_stage1_{file_name}_H{args.forecast_H}.pth"
        torch.save(model.prompt_memory.state_dict(), save_pm)
        print(f"[*] Stage 1 done. Router: {save_router}")
        print(f"    noop_ratio_final: {history['noop_ratio'][-1]*100:.1f}%")

    # ---- Stage 2 ----
    if '2' in args.stage:
        print("\n" + "="*60)
        print(f"[Stage 2] Joint Fine-tuning — {file_name} H={args.forecast_H}")
        print("="*60)

        # Reload stage1 router if doing 12 as separate run
        if '1' not in args.stage and getattr(args, 'router_weights', None):
            print(f"[*] Loading Stage 1 router from: {args.router_weights}")

        opt_router = optim.Adam(
            model.prompt_memory.router.parameters(), lr=args.router_lr
        )
        opt_experts = optim.Adam(
            [model.prompt_memory.prompts], lr=args.expert_lr
        )

        run_stage2_joint(
            model=model,
            backbone_adapter=backbone_adapter,
            dataloader=dl_gen(),
            optimizer_router=opt_router,
            optimizer_experts=opt_experts,
            train_size=train_size,
            device=device,
            lambda_forecast=args.lambda_forecast,
            lambda_kl=args.lambda_kl,
            lambda_bal=args.lambda_bal,
            lambda_smooth=args.lambda_smooth,
            lambda_noop=args.lambda_noop,
            oracle_temperature=args.oracle_temp,
            router_temperature=args.temp_T,
            epochs=max(1, args.epochs // 2),
            log_interval=args.log_interval,
        )

        save_pm2 = f"./weights/prompt_memory_stage2_{file_name}_H{args.forecast_H}.pth"
        torch.save(model.prompt_memory.state_dict(), save_pm2)
        print(f"[*] Stage 2 done. Saved: {save_pm2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default='M')
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--forecast_H", type=int, default=96)
    parser.add_argument("--D_model", type=int, default=512)
    parser.add_argument("--target", type=str, default='OT')
    parser.add_argument("--num_workers", type=int, default=4)
    # MoE
    parser.add_argument("--num_experts", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--temp_T", type=float, default=1.0)
    parser.add_argument("--l_aux_weight", type=float, default=0.0)
    parser.add_argument("--window_K", type=int, default=12)
    # Router architecture
    parser.add_argument("--router_hidden", type=int, default=256)
    parser.add_argument("--router_hist_hidden", type=int, default=64)
    parser.add_argument("--router_dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--stage", type=str, default='1', choices=['1', '2', '12', 'sanity'],
                        help="1=distill only, 2=joint only, 12=both")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--router_lr", type=float, default=1e-3)
    parser.add_argument("--expert_lr", type=float, default=1e-4,
                        help="Stage 2 expert bank LR (should be 10x smaller than router_lr)")
    parser.add_argument("--router_weights", type=str, default=None,
                        help="Path to pre-trained router weights (for Stage 2 standalone)")
    parser.add_argument("--prompt_memory_weights", type=str, default=None,
                        help="Path to full prompt_memory state dict, or auto")
    parser.add_argument("--router_stage", type=int, default=2, choices=[1, 2],
                        help="Stage used when --prompt_memory_weights auto")
    parser.add_argument("--log_interval", type=int, default=200)
    # Loss weights
    parser.add_argument("--lambda_forecast", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=2.0)
    parser.add_argument("--lambda_bal", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_noop", type=float, default=0.5)
    parser.add_argument("--oracle_temp", type=float, default=1.0,
                        help="Temperature for oracle soft label softmax")
    parser.add_argument("--sanity_samples", type=int, default=512,
                        help="Number of aligned windows for router sanity overfit")
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    # Parameter bridges
    args.pred_len = args.forecast_H
    args.d_model = args.D_model
    args.task_name = 'long_term_forecast'
    args.dropout = 0.1
    args.factor = 3
    args.n_heads = 8
    args.d_ff = 512
    args.e_layers = 3
    args.activation = 'gelu'
    args.num_class = 1
    args.embed = 'timeF'
    args.freq = 'h'
    args.batch_size = 1
    args.accum_steps = 1

    main(args)
