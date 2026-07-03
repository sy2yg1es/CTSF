import argparse
import torch
import torch.optim as optim
import os
import random

from models.framework import ContinualPromptTSF, BottleneckAdapter
from models.backbone_adapter import PatchTSTAdapter, iTransformerAdapter
from models.mlp_router import RichMLPRouter
from core.buffer import BDLABuffer
from core.drift_detector import ActualDriftDetector
from core.improved_drift_detector import ImprovedDriftDetector
from models.prompts.prompt_pool import SparsePromptMemory
from engine.streaming_loop import run_streaming_eval
from engine.oracle_experiments import (
    run_oracle_detector, run_oracle_channel,
    run_oracle_routing, run_causal_oracle_routing,
    run_segment_adapt,
)
from engine.router_diagnostics import run_router_learnability_diagnostics
from data_provider.data_loader import data_provider
from data_provider.streaming_env import StreamingEnvironment
from models.backbones.PatchTST import Model as PatchTSTModel
from models.backbones.iTransformer import Model as iTransformerModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _router_param_norm(router: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for p in router.parameters():
            total += p.detach().float().pow(2).sum().item()
    return total ** 0.5


def _print_stage3_router_report(model, args, sample_batch, device) -> None:
    if not getattr(model.prompt_memory, "_use_rich_router", False):
        print("[*] Router diagnostics: linear router (no Stage3 checkpoint)")
        return

    router = model.prompt_memory.router
    print("[*] === Stage3 Router Load/Effect Diagnostics ===")
    print(f"[*] router checkpoint path: {getattr(args, '_resolved_prompt_memory_path', None)}")
    print(f"[*] missing_keys: {getattr(args, '_pm_missing_keys', [])}")
    print(f"[*] unexpected_keys: {getattr(args, '_pm_unexpected_keys', [])}")
    print(f"[*] router type: {router.__class__.__name__}")
    print(f"[*] router weight norm: {_router_param_norm(router):.6f}")
    print(f"[*] temperature: {model.prompt_memory.temperature}")
    print(f"[*] top_k: {model.prompt_memory.top_k}")
    print(f"[*] expert bank: {model.prompt_memory.num_experts} real + "
          f"{1 if model.prompt_memory.use_noop else 0} no-op")

    try:
        X_sample = sample_batch[0] if isinstance(sample_batch, (tuple, list)) else sample_batch
        X_sample = X_sample.to(device, non_blocking=True)
        model.eval()
        with torch.no_grad():
            H_tokens, z_query, _, _ = model.backbone_adapter.encode(X_sample)
            z_features = model._router_features(H_tokens, z_query)
            if getattr(model.prompt_memory, "_use_rich_router", False):
                logits = router(z_features, None)
            else:
                logits = model.prompt_memory._linear_router_logits(z_query)
            probs = torch.softmax(logits / model.prompt_memory.temperature, dim=-1)
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
            top1 = probs.argmax(dim=-1).reshape(-1)
            counts = torch.bincount(top1.cpu(), minlength=probs.shape[-1])
            noop_idx = model.prompt_memory.num_experts
            noop_logit_mean = (
                logits[..., noop_idx].mean().item()
                if logits.shape[-1] > noop_idx else float("nan")
            )

        print(f"[*] router logits mean/std/min/max: "
              f"{logits.mean().item():.6f} / {logits.std(unbiased=False).item():.6f} / "
              f"{logits.min().item():.6f} / {logits.max().item():.6f}")
        print(f"[*] router output entropy: mean={entropy.mean().item():.6f} "
              f"std={entropy.std(unbiased=False).item():.6f}")
        print(f"[*] top1 expert distribution: {counts.tolist()}")
        print(f"[*] no-op logit mean: {noop_logit_mean:.6f}")
    except Exception as e:
        print(f"[!] Failed to print Stage3 router effect diagnostics: {e}")

def main(args):
    set_seed(getattr(args, "seed", 2026))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
    print(f"[*] seed: {getattr(args, 'seed', 2026)}")

    # --- 模块 1: 实例化真实数据集与流式环境 ---
    print(f"[*] Loading Dataset: {args.data_path}")
    dataset, base_loader = data_provider(args)
    streaming_dataloader = StreamingEnvironment(base_loader, forecast_H=args.forecast_H)
    
    # 动态获取实际特征数 (对应 ETTh1 / ECL 等多变量数据集)
    actual_features = dataset.data_x.shape[1] 
    print(f"[*] Detected Channels (C): {actual_features}")
    args.enc_in = actual_features  # 喂给 PatchTST 的 FlattenHead

    # --- 模块 2: 初始化 BackboneAdapter (冻结在 adapter 内部完成) ---
    backbone_name = getattr(args, 'backbone', 'patchtst')
    file_name = args.data_path.split('.')[0]
    accum_steps = getattr(args, 'accum_steps', 1)

    if backbone_name == 'itransformer':
        backbone = iTransformerModel(configs=args).to(device)
        weight_path = f"./weights/itransformer_pretrained_{file_name}_H{args.forecast_H}.pth"
        backbone_adapter = iTransformerAdapter(backbone, use_query_mlp=getattr(args, 'use_query_mlp', False))
    else:
        backbone = PatchTSTModel(configs=args, patch_len=16, stride=8).to(device)
        if accum_steps > 1:
            weight_path = f"./weights/patchtst_pretrained_{file_name}_H{args.forecast_H}_GA{accum_steps}.pth"
        else:
            weight_path = f"./weights/patchtst_pretrained_{file_name}_H{args.forecast_H}.pth"
        backbone_adapter = PatchTSTAdapter(backbone, use_query_mlp=getattr(args, 'use_query_mlp', False))

    try:
        backbone.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"[*] Loaded pretrained weights: {weight_path}")
    except FileNotFoundError:
        print(f"[!] No pretrained weights at {weight_path}. Random init!")

    print(f"[*] Backbone: {backbone_name} | Adapter: {backbone_adapter.__class__.__name__}")

    # --- 模块 3: 初始化核心状态与记忆 ---
    load_pm = getattr(args, 'load_prompt_memory', None)
    use_rich_router = (load_pm is not None)

    if use_rich_router:
        # Stage 3: load pretrained MLP-router + expert bank
        rich_router = RichMLPRouter(
            d_model=args.D_model,
            num_experts=args.num_experts,
            hist_K=args.window_K,
            hidden=getattr(args, 'router_hidden', 256),
            hist_hidden=getattr(args, 'router_hist_hidden', 64),
        ).to(device)
        prompt_memory = SparsePromptMemory(
            prompt_dim=args.D_model,
            num_experts=args.num_experts,
            top_k=args.top_k,
            temperature=args.temp_T,
            load_balancing_alpha=args.l_aux_weight,
            rich_router=rich_router,
            use_noop=True,
        )
        # Determine which stage weights to load
        router_stage = getattr(args, 'router_stage', 2)
        file_name = args.data_path.split('.')[0]
        if load_pm == 'auto':
            load_pm = (f"./weights/prompt_memory_stage{router_stage}_"
                       f"{file_name}_H{args.forecast_H}.pth")
        args._resolved_prompt_memory_path = load_pm
        args._pm_missing_keys = []
        args._pm_unexpected_keys = []
        print(f"[*] router checkpoint path: {load_pm}")
        try:
            pm_state = torch.load(load_pm, map_location=device)
            incompatible = prompt_memory.load_state_dict(pm_state, strict=False)
            args._pm_missing_keys = list(incompatible.missing_keys)
            args._pm_unexpected_keys = list(incompatible.unexpected_keys)
            print(f"[*] Loaded prompt_memory from: {load_pm}")
            print(f"[*] missing_keys: {args._pm_missing_keys}")
            print(f"[*] unexpected_keys: {args._pm_unexpected_keys}")
        except FileNotFoundError:
            print(f"[!] prompt_memory weights not found: {load_pm}. Using random init.")
    else:
        prompt_memory = SparsePromptMemory(
            prompt_dim=args.D_model,
            num_experts=args.num_experts,
            top_k=args.top_k,
            temperature=args.temp_T,
            load_balancing_alpha=args.l_aux_weight
        )

    buffer = BDLABuffer(horizon_H=args.forecast_H)

    # --- Drift Detector: improved (default) or legacy ---
    use_legacy = getattr(args, 'legacy_detector', False)
    DetectorClass = ActualDriftDetector if use_legacy else ImprovedDriftDetector
    detector_kwargs = dict(
        num_channels=actual_features,
        window_K=args.window_K,
        threshold_tau=args.threshold_tau,
        patience_C=args.patience_C,
    )
    if not use_legacy:
        detector_kwargs['slope_thresh'] = getattr(args, 'slope_thresh', 0.0)
        detector_kwargs['std_cap_mult'] = getattr(args, 'std_cap_mult', 2.0)
        print(f"[*] Using ImprovedDriftDetector (slope+std gating) [default]")
        print("[*] Detector: improved")
    else:
        print(f"[*] Using LegacyDriftDetector (single-threshold)")
        print("[*] Detector: original")
    detector = DetectorClass(**detector_kwargs).to(device)


    # --- 模块 3.5: 可选 BottleneckAdapter (Phase 2) ---
    adapter = None
    if getattr(args, 'bottleneck_dim', 0) > 0:
        adapter = BottleneckAdapter(
            d_model=args.D_model,
            bottleneck_dim=args.bottleneck_dim,
        ).to(device)
        print(f"[*] BottleneckAdapter enabled: {args.D_model} → {args.bottleneck_dim} → {args.D_model}")

    # --- 模块 4: 实例化拼装工厂 ---
    model = ContinualPromptTSF(
        backbone_adapter=backbone_adapter,
        prompt_memory=prompt_memory,
        adapter=adapter,
    ).to(device)

    # --- 模块 5: 定义优化器 ---
    if getattr(args, 'freeze_prompt', False):
        for param in model.prompt_memory.parameters():
            param.requires_grad = False
        print("[*] prompt_memory FROZEN (no online updates)")

    # Stage 3A: router frozen, only prompt params online-updatable
    # Stage 3B: router calibration (only bias/temperature)
    if use_rich_router:
        router_calibration = getattr(args, 'router_calibration', False)
        if not router_calibration:
            # 3A: freeze entire router
            for p in model.prompt_memory.router.parameters():
                p.requires_grad = False
            print("[*] RichMLPRouter FROZEN (Stage 3A: router-frozen online)")
        else:
            # 3B: freeze all router except last bias
            for name, p in model.prompt_memory.router.named_parameters():
                p.requires_grad = name.endswith('bias') and 'merge' in name
            n_cal = sum(p.numel() for p in model.prompt_memory.router.parameters()
                        if p.requires_grad)
            print(f"[*] RichMLPRouter calibration mode: {n_cal} params trainable")

    online_update = (
        "stage3b_calibration" if use_rich_router and getattr(args, 'router_calibration', False)
        else "stage3a_router_frozen" if use_rich_router
        else getattr(args, 'streaming_mode', 'ours')
    )
    print("[*] === Eval Configuration ===")
    print(f"[*] checkpoint: {getattr(args, '_resolved_prompt_memory_path', None)}")
    print(f"[*] detector: {'original' if use_legacy else 'improved'}")
    print(f"[*] router type: {model.prompt_memory.router.__class__.__name__}")
    print(f"[*] top_k: {args.top_k}")
    print(f"[*] temperature: {args.temp_T}")
    print(f"[*] seed: {getattr(args, 'seed', 2026)}")
    print(f"[*] expert bank: {args.num_experts}")
    print(f"[*] no-op setting: {getattr(model.prompt_memory, 'use_noop', False)}")
    print(f"[*] online update setting: {online_update}")

    try:
        sample_batch = next(iter(base_loader))
        _print_stage3_router_report(model, args, sample_batch, device)
    except Exception as e:
        print(f"[!] Failed to sample batch for router diagnostics: {e}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[*] Trainable params: {n_trainable}")

    if n_trainable == 0:
        # P1: 完全无在线更新，给一个 dummy optimizer
        print("[!] No trainable params — streaming will run inference-only (no updates)")
        optimizer = optim.Adam([torch.zeros(1, device=device, requires_grad=True)], lr=args.learning_rate)
    else:
        optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
    
    # 注意：流式循环内部使用了 MSELoss(reduction='none')，此处无需传入 criterion

    # --- 模块 6: 进入流式引擎 ---
    def device_dataloader_wrapper(loader):
        for X_t, Y_t in loader:
            X_t = X_t.to(device, non_blocking=True)
            if Y_t is not None:
                Y_t = Y_t.to(device, non_blocking=True)
            yield X_t, Y_t

    print("[*] Starting Streaming Evaluation...")
    train_size = int(len(dataset) * 0.5)
    streaming_mode = getattr(args, 'streaming_mode', 'ours')
    tag = getattr(args, 'experiment_tag', '')
    dl = device_dataloader_wrapper(streaming_dataloader)

    ORACLE_MODES = (
        'oracle_detector', 'oracle_channel', 'oracle_routing',
        'posterior_oracle_routing', 'causal_oracle_routing',
        'router_learnability', 'segment_adapt',
    )

    if streaming_mode in ORACLE_MODES:
        oracle_kwargs = dict(
            model=model, dataloader=dl, buffer=buffer, detector=detector,
            optimizer=optimizer, l_aux_weight=args.l_aux_weight,
            train_size=train_size, experiment_tag=tag,
        )
        if streaming_mode == 'oracle_detector':
            metrics = run_oracle_detector(**oracle_kwargs)
        elif streaming_mode == 'oracle_channel':
            metrics = run_oracle_channel(**oracle_kwargs)
        elif streaming_mode in ('oracle_routing', 'posterior_oracle_routing'):
            metrics = run_oracle_routing(**oracle_kwargs)
        elif streaming_mode == 'causal_oracle_routing':
            metrics = run_causal_oracle_routing(
                **oracle_kwargs,
                history_K=getattr(args, 'causal_oracle_K', 12),
            )
        elif streaming_mode == 'router_learnability':
            metrics = run_router_learnability_diagnostics(
                **oracle_kwargs,
                output_dir=getattr(args, 'diagnostics_dir', 'logs/stage3_diagnostics'),
            )
        elif streaming_mode == 'segment_adapt':
            metrics = run_segment_adapt(
                **oracle_kwargs,
                segment_size=getattr(args, 'segment_size', 500),
                adapt_steps=getattr(args, 'adapt_steps', 10),
            )
    else:
        metrics = run_streaming_eval(
            model=model, dataloader=dl, buffer=buffer, detector=detector,
            optimizer=optimizer, l_aux_weight=args.l_aux_weight,
            train_size=train_size, experiment_tag=tag,
            streaming_mode=streaming_mode,
        )

    mse = metrics.get('MSE', metrics['RMSE'] ** 2)
    updates = metrics.get('update_triggered_steps', 0)
    aligned = metrics.get('total_aligned_steps', 0)
    ch_ratio = metrics.get('avg_channel_update_ratio', 0.0)
    print(f"[*] Streaming Evaluation Completed! MAE: {metrics['MAE']:.4f}, MSE: {mse:.4f}, RMSE: {metrics['RMSE']:.4f}")
    print(f"[*] Updates: {updates}/{aligned} ({updates/max(aligned,1)*100:.1f}%) | Avg Ch Ratio: {ch_ratio*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集参数 (与 data_loader.py 对齐)
    parser.add_argument("--root_path", type=str, required=True, help="root path of the data file")
    parser.add_argument("--data_path", type=str, required=True, help="data file name (e.g. ETTh1.csv)")
    parser.add_argument("--features", type=str, default='M', help="forecasting task, options:[M, S, MS]")
    parser.add_argument("--target", type=str, default='OT', help="target feature in S or MS task")
    parser.add_argument("--num_workers", type=int, default=4, help="data loader num workers")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    
    # 核心架构参数
    parser.add_argument("--seq_len", type=int, default=96, help="Input sequence length L")
    parser.add_argument("--forecast_H", type=int, default=24, help="Forecast horizon H (pred_len)")
    parser.add_argument("--D_model", type=int, default=512, help="Backbone output dimension")
    
    # 漂移检测参数 
    parser.add_argument("--window_K", type=int, default=12)
    parser.add_argument("--threshold_tau", type=float, default=0.2)
    parser.add_argument("--patience_C", type=int, default=3)
    
    # MoE 记忆库与路由参数 (清理了旧参，引入新参)
    parser.add_argument("--num_experts", type=int, default=10, help="Number of experts in MoE bank")
    parser.add_argument("--top_k", type=int, default=2, help="Top-K sparse routing")
    parser.add_argument("--temp_T", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--l_aux_weight", type=float, default=1e-3, help="Load balancing loss weight alpha")
    
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--accum_steps", type=int, default=1, help="must match pretrain accum_steps for weight loading")
    parser.add_argument("--bottleneck_dim", type=int, default=0,
                        help="Adapter bottleneck dimension. 0 = no adapter. e.g. 32 for 512->32->512 compression.")
    parser.add_argument("--experiment_tag", type=str, default='',
                        help="Tag for monitor log filename")
    parser.add_argument("--freeze_prompt", action='store_true',
                        help="Freeze prompt_memory during streaming.")
    parser.add_argument("--streaming_mode", type=str, default='ours',
                        choices=['frozen', 'full_ft', 'ours',
                                 'oracle_detector', 'oracle_channel',
                                 'oracle_routing', 'posterior_oracle_routing',
                                 'causal_oracle_routing', 'router_learnability',
                                 'segment_adapt'],
                        help="Streaming mode")
    parser.add_argument("--backbone", type=str, default='patchtst',
                        choices=['patchtst', 'itransformer'],
                        help="Backbone architecture to use")
    parser.add_argument("--legacy_detector", action='store_true',
                        help="Use legacy single-threshold detector (not recommended)")
    # --- Improved detector params ---
    parser.add_argument("--improved_detector", action='store_true',
                        help="[deprecated] now default; kept for backward compat")
    parser.add_argument("--slope_thresh", type=float, default=0.0)
    parser.add_argument("--std_cap_mult", type=float, default=2.0)
    # --- Improved z_query ---
    parser.add_argument("--use_query_mlp", action='store_true')
    # --- segment_adapt ---
    parser.add_argument("--segment_size", type=int, default=500)
    parser.add_argument("--adapt_steps", type=int, default=10)
    # --- Pretrained router (Stage 3) ---
    parser.add_argument("--load_prompt_memory", type=str, default=None,
                        help="Path to prompt_memory state dict, or 'auto' to auto-resolve from dataset/H")
    parser.add_argument("--router_stage", type=int, default=2, choices=[1, 2],
                        help="Which stage weights to load when --load_prompt_memory auto")
    parser.add_argument("--router_calibration", action='store_true',
                        help="Stage 3B: allow only router bias calibration online")
    parser.add_argument("--router_hidden", type=int, default=256)
    parser.add_argument("--router_hist_hidden", type=int, default=64)
    parser.add_argument("--causal_oracle_K", type=int, default=12,
                        help="History window for causal_oracle_routing")
    parser.add_argument("--diagnostics_dir", type=str, default="logs/stage3_diagnostics",
                        help="Output directory for router learnability diagnostics")

    args = parser.parse_args()
    
    # ==========================================
    # 参数桥接与补齐 (适配 PatchTST 和 DataLoader 的官方契约)
    # ==========================================
    args.pred_len = args.forecast_H  # 映射 forecast_H 到 pred_len
    args.d_model = args.D_model      # 映射 D_model 到 d_model
    
    # 补齐 backbone 必需的默认网络结构参数
    args.task_name = 'long_term_forecast'
    args.dropout = 0.1
    args.factor = 3
    args.n_heads = 8
    args.d_ff = 512
    args.e_layers = 3
    args.activation = 'gelu'
    args.num_class = 1
    # iTransformer 需要的额外参数
    args.embed = 'timeF'
    args.freq = 'h'
    
    main(args)
    
