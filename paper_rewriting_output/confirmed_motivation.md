# Confirmed Motivation

## User-Confirmed Core

假设在线分布变化主要引起隐藏表征偏移，而非主干模型知识整体失效。项目通过历史预测残差识别漂移时间、强度和受影响通道，并在冻结主干的基础上，对Z-space表征实施低秩、门控、近恒等的局部修正；无可靠漂移时保持No-op，从而兼顾在线适应能力、预测稳定性和计算效率。

## English Working Form

The working hypothesis of CTSF is that many online distribution shifts in time series forecasting first appear as local displacement in hidden representations, not as wholesale failure of the pretrained forecasting backbone. CTSF should use causal residual histories to infer when drift appears, how strong it is, and which channels are affected. It should then apply a low-rank, gated, near-identity correction in Z-space while keeping the backbone frozen. If the residual evidence is weak or unreliable, the system should remain a no-op.

## Claim Boundary

This motivation supports a method paper and an experimental scaffold. It does not yet support final benchmark superiority claims because the user requested the experimental result section to remain unfilled in the draft.
