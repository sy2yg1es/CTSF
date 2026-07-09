# PaperSpine Configuration

| Field | Value |
|---|---|
| Workflow | build_from_materials |
| Target scene | conference |
| Tier | pro |
| Output language | en |
| Target name | CTSF |
| Materials directory | `D:\model\paper_templates\CTSF` |
| Draft path | `D:\model\paper` |
| Reference mode | local_first |
| Reference paths | `D:\model\references` |
| Citation target count | 20 |
| Word output | none |
| Translation package | none |
| Humanize tier | heavy |
| Detection platform | general |
| UI language | zh |

## User Motivation

假设在线分布变化主要引起隐藏表征偏移，而非主干模型知识整体失效。项目通过历史预测残差识别漂移时间、强度和受影响通道，并在冻结主干的基础上，对Z-space表征实施低秩、门控、近恒等的局部修正；无可靠漂移时保持No-op，从而兼顾在线适应能力、预测稳定性和计算效率。

## Special Requirements

- 必须输出 `final_paper/main.tex`；如果本机有 LaTeX 编译器则编译 `paper.pdf`。
- 必须生成详细 `writing_rationale_matrix.md`，逐段解释写作逻辑。
- 从素材从零构筑，不把技术说明当成初稿润色。
