import argparse
import torch
from core.buffer import BDLABuffer
from data_provider.data_loader import data_provider
from data_provider.streaming_env import StreamingEnvironment
from models.backbones.PatchTST import Model as PatchTSTModel

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Static Baseline on device: {device}")

    # 1. 数据加载
    dataset, base_loader = data_provider(args)
    streaming_dataloader = StreamingEnvironment(base_loader, forecast_H=args.forecast_H)
    actual_features = dataset.data_x.shape[1] 
    args.enc_in = actual_features

    # 2. 初始化 Backbone 并注入灵魂
    backbone = PatchTSTModel(configs=args, patch_len=16, stride=8).to(device)
    file_name = args.data_path.split('.')[0]
    accum_steps = getattr(args, 'accum_steps', 1)
    if accum_steps > 1:
        weight_path = f"./weights/patchtst_pretrained_{file_name}_H{args.forecast_H}_GA{accum_steps}.pth"
    else:
        weight_path = f"./weights/patchtst_pretrained_{file_name}_H{args.forecast_H}.pth"
    
    try:
        backbone.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"[*] Loaded pre-trained weights: {weight_path}")
    except FileNotFoundError:
        print(f"[!] Weights not found at {weight_path}! Baseline is meaningless without correct weights.")
        return

    backbone.eval()

    # 3. 极简流式评测 (没有 Prompt，没有 MoE，只有原始的预测)
    buffer = BDLABuffer(horizon_H=args.forecast_H)
    mae_sum = 0.0
    mse_sum = 0.0
    n_aligned = 0

    print("[*] Starting Static Baseline Evaluation...")
    train_size = int(len(dataset) * 0.5)
    with torch.no_grad():
        for t, (X_t, Y_t) in enumerate(streaming_dataloader):
            X_t = X_t.to(device)
            
            # 直接使用刚刚重构的纯净 API 获取预测值！
            H_patches, means, stdev = backbone.encode_local(X_t)
            
            # 直接过 Encoder 和 Head (无需 Prefix Token)
            B, C, D, P = H_patches.shape
            enc_in = H_patches.permute(0, 1, 3, 2).reshape(B * C, P, D)
            enc_out, _ = backbone.encoder(enc_in)
            enc_out = torch.reshape(enc_out, (B, C, P, D)).permute(0, 1, 3, 2)
            Y_hat_future = backbone.apply_prediction_head(enc_out, means, stdev)

            # 存入 Buffer 等待标签到达
            buffer.push(t, X_t, Y_hat_future, torch.zeros(1), torch.zeros(1))

            if Y_t is None:
                continue
            
            # 取出对齐的预测值进行算分
            Y_hat_history = buffer.get_stored_prediction(t)
            if Y_hat_history is None:
                continue
                
            aligned = buffer.pop_and_align(t, Y_t)
            if aligned is None:
                continue
                
            # ====== 新增：跳过训练集阶段的评估 ======
            if t < train_size:
                continue
            # ============================================

            Y_hat_history = Y_hat_history.to(device)
            Y_current = aligned[1].to(device)

            err = Y_hat_history - Y_current
            mae_sum += err.abs().mean().item()
            mse_sum += err.pow(2).mean().item()
            n_aligned += 1

    if n_aligned > 0:
        final_mae = mae_sum / n_aligned
        final_mse = mse_sum / n_aligned
        print(f"[*] Static Baseline Completed! MAE: {final_mae:.4f}, MSE: {final_mse:.4f}, RMSE: {final_mse ** 0.5:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default='M')
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--forecast_H", type=int, default=24)
    parser.add_argument("--D_model", type=int, default=512)
    parser.add_argument("--target", type=str, default='OT')
    parser.add_argument("--accum_steps", type=int, default=1, help="must match pretrain accum_steps for weight loading")
    args = parser.parse_args()

    # 参数桥接
    args.num_workers = 0
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
    
    main(args)