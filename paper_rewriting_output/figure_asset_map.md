# Figure Asset Map

| Figure ID | Planned Figure | Source | Status | Claim It Should Support |
|---|---|---|---|---|
| F1 | CTSF architecture: input window, frozen backbone, Z-space hook, residual tracker, gate/mask, low-rank correction, prediction head. | Code structure in `PromptZTSF`, `PromptZModulator`, `ResidualTracker`. | Text-described in draft; graphic not rendered yet. | CTSF modifies hidden states while freezing the backbone. |
| F2 | Delayed-label streaming timeline showing prediction at time t and residual update after horizon H. | `StreamingEnvironment`, DSOF/PROCEED PDFs. | Planned. | CTSF only uses causal residuals. |
| F3 | Conservation controls: negative gate bias, zero-init up projection, ratio clamp, validation fallback. | `prompt_z.py`, `streaming_prompt_z.py`. | Planned. | CTSF remains no-op under weak evidence. |
| T1 | Main quantitative benchmark. | Final experiment run. | Empty in this draft. | Accuracy under leak-free online evaluation. |
| T2 | Ablation table for residual features, gate/mask, rank, clamp, fallback. | Final ablation run. | Empty in this draft. | Each mechanism's contribution. |
| T3 | Efficiency table. | Final profiling run. | Empty in this draft. | Parameter and compute cost. |
