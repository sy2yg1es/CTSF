import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

from data_provider.data_loader import data_provider
from models.backbones.PatchTST import Model as PatchTSTModel
from models.backbones.iTransformer import Model as iTransformerModel

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Pre-training on device: {device}")

    # 1. 加载全量数据
    args.val_ratio = getattr(args, 'val_ratio', 0.1)   # ensure val_ratio is set
    dataset, _ = data_provider(args)
    actual_features = dataset.data_x.shape[1]
    args.enc_in = actual_features

    # Pretrain uses the earliest windows only (label fully in train zone).
    # Use label-timestamp based boundary, not window count ratio.
    train_size = dataset.train_size
    train_size = max(1, min(train_size, len(dataset)))
    train_subset = Subset(dataset, range(train_size))

    # 划分前 60% 作为离线训练集
    
    # 动态计算 batch_size，防止通道数（C）太多导致 CI (Channel Independence) 机制下 Transformer 爆显存
    # 目标是保持实际送入 Transformer 的 Effective Batch Size (batch_size * C) 不超过 max_effective_bs
    # 显存充足时（如 RTX 5090）可通过 --max_effective_bs 调高上限，直接用大 BS 替代 GA
    adaptive_batch_size = max(1, min(args.batch_size, args.max_effective_bs // actual_features))
    effective_bs = adaptive_batch_size * args.accum_steps
    print(f"[*] Total channels: {actual_features}, Adaptive Batch Size: {adaptive_batch_size} (Requested: {args.batch_size})")
    print(f"[*] Gradient Accumulation Steps: {args.accum_steps}, Effective Batch Size: {effective_bs}")
    
    train_loader = DataLoader(train_subset, batch_size=adaptive_batch_size, shuffle=True)
    
    print(f"[*] Pretrain train_ratio: {train_ratio:.3f}")
    print(f"[*] Total samples: {len(dataset)}, Training on first {train_size} samples")

    # 3. 初始化 Backbone
    backbone_name = getattr(args, 'backbone', 'patchtst')
    if backbone_name == 'itransformer':
        model = iTransformerModel(configs=args).to(device)
    else:
        model = PatchTSTModel(configs=args, patch_len=16, stride=8).to(device)
    
    # 4. 定义优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 5. 预训练循环 (支持 Gradient Accumulation)
    epochs = args.epochs
    total_optimizer_steps = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step_idx, (batch_x, batch_y) in enumerate(pbar):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if backbone_name == 'itransformer':
                # iTransformer: normalize → inv_embedding → encoder → projection
                means = batch_x.mean(1, keepdim=True).detach()
                x_norm = batch_x - means
                stdev = torch.sqrt(
                    torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
                ).detach()
                x_norm = x_norm / stdev
                enc_out = model.enc_embedding(x_norm, None)       # [B, C, D]
                enc_out, _ = model.encoder(enc_out)               # [B, C, D]
                dec_out = model.projection(enc_out).permute(0, 2, 1)  # [B, pred_len, C]
                dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, args.pred_len, 1)
                outputs = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, args.pred_len, 1)
            else:
                # PatchTST: encode_local → encoder → prediction_head
                H_patches, means, stdev = model.encode_local(batch_x)
                B, C, D, P = H_patches.shape
                enc_in = H_patches.permute(0, 1, 3, 2).reshape(B * C, P, D)
                enc_out, _ = model.encoder(enc_in)
                enc_out = torch.reshape(enc_out, (B, C, P, D)).permute(0, 1, 3, 2)
                outputs = model.apply_prediction_head(enc_out, means, stdev)
            
            loss = criterion(outputs, batch_y)
            # GA: scale loss by accum_steps for correct gradient magnitude
            scaled_loss = loss / args.accum_steps
            scaled_loss.backward()
            
            total_loss += loss.item()
            batch_count += 1

            # Gradient accumulation: step every accum_steps batches
            if (step_idx + 1) % args.accum_steps == 0:
                # --- Grad Norm 监控 ---
                grad_norm = torch.norm(
                    torch.stack([
                        p.grad.norm()
                        for p in model.parameters()
                        if p.grad is not None
                    ])
                ).item()
                
                optimizer.step()
                optimizer.zero_grad()
                total_optimizer_steps += 1
                
                # 每 100 个 optimizer step 打印 grad norm
                if total_optimizer_steps % 100 == 0:
                    print(f"  [Step {total_optimizer_steps}] grad_norm={grad_norm:.4f} loss={loss.item():.4f}")
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 处理尾部 (不足 accum_steps 的残余 batch)
        if (step_idx + 1) % args.accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            total_optimizer_steps += 1
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | Optimizer Steps this epoch: {total_optimizer_steps}")

    print(f"[*] Total Optimizer Steps: {total_optimizer_steps}")

    # 6. 保存预训练权重
    # 修改文件名生成逻辑，加入 args.forecast_H
    file_name = args.data_path.split('.')[0]
    prefix = f"{backbone_name}_pretrained"
    if args.accum_steps > 1:
        save_path = f"./weights/{prefix}_{file_name}_H{args.forecast_H}_GA{args.accum_steps}.pth"
    else:
        save_path = f"./weights/{prefix}_{file_name}_H{args.forecast_H}.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[*] Pre-training completed! Weights saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 保持与 main.py 完全一致的参数
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default='M')
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--forecast_H", type=int, default=24)
    parser.add_argument("--D_model", type=int, default=512)
    parser.add_argument("--target", type=str, default='OT')
    parser.add_argument("--batch_size", type=int, default=32, help="pre-training batch size")
    parser.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps for effective BS = batch_size * accum_steps")
    parser.add_argument("--epochs", type=int, default=10, help="number of pre-training epochs")
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="fraction of earliest dataset windows used for backbone pre-training")
    parser.add_argument("--max_effective_bs", type=int, default=1000,
                        help="max allowed (batch_size * C) sent to transformer per step. "
                             "Raise for large-VRAM GPUs (e.g. --max_effective_bs 20000 for RTX 5090) "
                             "to bypass adaptive BS cap without gradient accumulation.")
    parser.add_argument("--backbone", type=str, default='patchtst',
                        choices=['patchtst', 'itransformer'],
                        help="Backbone to pretrain")
    args = parser.parse_args()

    # 参数桥接
    args.num_workers = 0
    args.pred_len = args.forecast_H
    args.d_model = args.D_model
    args.task_name = 'long_term_forecast'
    args.dropout = 0.1
    args.factor = 1
    args.n_heads = 8
    args.d_ff = 512
    args.e_layers = 3
    args.activation = 'gelu'
    args.num_class = 1
    args.embed = 'timeF'
    args.freq = 'h'

    main(args)
