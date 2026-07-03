# Prompt-Z 首轮实验分析 + 修复方案

## 实验结果总览

| 数据集 | frozen MSE | pz_mode0 MSE | pz_mode1 MSE | Δ% mode0 | Δ% mode1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ECL H=96** | 1.0977 | 1.1689 | 1.1498 | **+6.5% ❌** | **+4.7% ❌** |
| **ETTh1 H=24** | 0.6886 | 0.7133 | 0.7990 | **+3.6% ❌** | **+16.0% ❌** |
| **Traffic H=1** | 1.1525 | 1.0288 | 0.9299 | **-10.7% ✅** | **-19.3% ✅** |

> [!CAUTION]
> ECL 和 ETTh1 退化严重，只有 Traffic 有效。

---

## 诊断：4 个致命问题

### 1. delta_to_hidden_ratio 爆炸（1.77 → 6.89）

```
ECL_H96:   delta_norm=5.77,  hidden_norm=3.26,  ratio=1.77
Traffic_H1: delta_norm=19.13, hidden_norm=3.26,  ratio=5.86
ETTh1_H24: delta_norm=11.29, hidden_norm=3.26,  ratio=3.46
```

**即使 gamma ≈ 0.015-0.108，实际修正量 = gamma × ratio ≈ 0.03-0.19**。这不是"微调"，是在严重扭曲 hidden。

**根因**: `LowRankModulator.down` 投影没有任何约束，把 D=512 压到 rank=8 再放大回来，scale 虽然经过 tanh 但仍能放大整体 norm。而训练时 **delta_reg 是从 diagnostics dict 拿的 detached float**，不参与梯度计算！

### 2. mask_ratio = 1.0（完全没有稀疏性）

所有实验 mask_mean ≈ 0.96-1.0。SparseMaskHead 的 L1 惩罚 `mask_reg` 也是从 diagnostics 拿的 **detached scalar**，同样不参与梯度。

### 3. 训练 loss 发散

```
ECL:     Epoch0=0.251 → Epoch1=0.281 → Epoch2=0.344  （越来越差）
Traffic: Epoch0=0.262 → Epoch1=0.228 → Epoch2=0.237
ETTh1:   Epoch0=0.345 → Epoch1=0.385 → Epoch2=0.410  （越来越差）
```

ECL 和 ETTh1 的 forecast_loss 在下降（0.229→0.165），但 total_loss 在上升——因为 **delta_reg 和 noop_penalty 在增长但没有梯度信号来抑制它们**。

### 4. noop_penalty 无效

```python
# train_prompt_z.py 中的问题代码：
noop_penalty = torch.tensor(diagnostics.get("gamma_mean", 0.0), device=device)
```

这创建了一个 **detached tensor**，`loss.backward()` 时梯度不会流回 ConfidenceGate。整个 noop margin 机制是死代码。

---

## 修复方案

### Fix 1: Loss 必须有梯度

当前 `delta_reg`、`mask_reg`、`noop_penalty` 都是从 diagnostics dict 读的 detached float。
需要从 forward pass 的实际 tensor 计算 loss。

```diff
- delta_reg = torch.tensor(diagnostics.get("delta_norm", 0.0), device=device)
- mask_reg = torch.tensor(diagnostics.get("mask_mean", 0.0), device=device)
+ # 从 forward_train 返回实际 tensor
+ delta_reg = (hidden_mod - hidden).norm(dim=-1).mean()  # 有梯度！
+ mask_reg = model.prompt_z.sparse_mask.last_output.mean()  # 有梯度！
```

修改 `PromptZModulator.forward()` 返回可求导的 regularization tensors。

### Fix 2: delta norm 硬上限

```python
# LowRankModulator 输出后 clamp
delta_h = delta_h.clamp(-delta_clamp, delta_clamp)  # e.g. delta_clamp=0.1
```

或者更优雅地：normalize delta 使得 `||delta|| / ||hidden|| ≤ max_ratio`

### Fix 3: 统一入口到 main.py

不再维护 `main_prompt_z.py`。把 Prompt-Z 作为 `--streaming_mode prompt_z` 集成到 main.py。

```
main.py --streaming_mode prompt_z \
        --prompt_z_weights weights/prompt_z/prompt_z_ECL_H96.pth \
        --prompt_z_mode mode0  # or mode1
```

关键对齐点：
- `train_size = int(len(dataset) * 0.5)` — 与 main.py 一致
- backbone 构建逻辑完全复用 main.py
- 权重路径格式 `weights/patchtst_pretrained_{name}_H{H}.pth` 复用
- 输出 metrics 格式与 `run_streaming_eval` 一致

### Fix 4: PatchTST 参数不匹配

main.py: `d_ff=512, factor=3`
main_prompt_z.py: `d_ff=D_model*4=2048, factor=1`

这意味着两边构建的 backbone **结构不同**，pretrained weights 加载会 silent mismatch。

---

## 要修改的文件

| 文件 | 修改 |
|:---|:---|
| `models/prompt_z.py` | forward() 返回 reg tensors; delta clamp |
| `models/prompt_z_framework.py` | forward_train() 返回 reg tensors |
| `engine/streaming_prompt_z.py` | 适配新接口，改为可被 main.py 调用 |
| `main.py` | 新增 `prompt_z` streaming mode |
| `train_prompt_z.py` | 修复 loss 计算，使用有梯度的 reg tensors |
| ~~`main_prompt_z.py`~~ | 废弃，功能合并到 main.py |

---

## Open Questions

> [!IMPORTANT]
> **delta clamp 策略？**
> - 硬 clamp（`clamp(-0.1, 0.1)`）简单粗暴
> - ratio clamp（`delta * min(1, max_ratio * ||h|| / ||delta||)`）更精准
> - 建议先用 ratio clamp，max_ratio=0.05 (delta 最多是 hidden 的 5%)
