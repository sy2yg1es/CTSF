# Evidence Bank

| Evidence ID | Source | Extracted Fact | Supports Claim | Use In Manuscript |
|---|---|---|---|---|
| E1 | `models/prompt_z.py` | `PromptZModulator` applies `hidden_mod = hidden + gamma * mask * delta_h`, with `delta_h` produced by a low-rank module. | CTSF performs local hidden-state correction rather than full backbone update. | Method equation |
| E2 | `models/prompt_z.py` | Gate and mask heads use negative bias initialization, and the low-rank up projection is zero-initialized. | CTSF is near no-op at initialization. | Conservation principle |
| E3 | `models/prompt_z.py` | `_ratio_clamp` limits correction norm to `max_delta_ratio` times the hidden norm. | CTSF bounds perturbation magnitude. | Method and ablation plan |
| E4 | `core/residual_tracker.py` | Residual state includes error mean, error slope, error std, signed bias, and steps since update. | Residual histories encode drift timing, strength, and channel effect. | Residual state subsection |
| E5 | `models/prompt_z_framework.py` | Backbone encode/decode operations are frozen; Prompt-Z receives gradients during training. | CTSF freezes the main forecasting backbone. | Architecture description |
| E6 | `engine/streaming_prompt_z.py` | Streaming evaluation uses caches and only updates residuals when delayed labels arrive. | The implementation respects causal delayed feedback. | Problem setup and protocol |
| E7 | `engine/streaming_prompt_z.py` | Validation fallback selects Prompt-Z only when validation beats frozen by a margin. | No-op behavior is operationalized in evaluation. | Experiment plan |
| E8 | `logs/prompt_z/summary_strict_v1.csv` | Existing strict-split logs include frozen, random Prompt-Z, mode0, mode1, and selected fallback rows. | Pilot evidence exists, but final result tables are not inserted in this draft. | Results validation note |
| E9 | ADAPT-Z PDF | The local paper argues that distribution shift can be addressed in feature/Z-space due to changes in latent factors. | Supports CTSF's representation-shift motivation. | Introduction and related work |
| E10 | DSOF PDF | The local paper identifies information leakage in online time series forecasting and redefines evaluation around unknown future steps. | Supports leak-free protocol. | Problem setup |
| E11 | PROCEED PDF | The local paper frames horizon-induced feedback delay as a temporal gap that can cause drift between training samples and the test sample. | Supports causal residual and proactive correction motivation. | Introduction |
| E12 | SOLID PDF | The local paper uses residual-context dependence to detect context-driven distribution shift. | Supports residual history as diagnostic signal. | Related work |
| E13 | PETSA PDF | The local paper uses low-rank adapters and dynamic gating for parameter-efficient TTA. | Supports design relevance of low-rank gated adaptation. | Related work |
| E14 | TAFAS PDF | The local paper freezes a source forecaster while adapting calibration modules with partial/delayed ground truth. | Supports frozen source forecaster and delayed supervision norm. | Related work |
