# SOTA Gap Map

| Line of Work | Representative Sources | What It Solves | Gap Left For CTSF | Writing Consequence |
|---|---|---|---|---|
| Non-stationary normalization | RevIN, Dish-TS, SAN, DDN, FAN | Handles changing input statistics, phases, or frequency patterns. | Mainly targets covariate/statistical shift and may not decide when an online hidden representation should be locally corrected. | Treat as background robustness methods, not direct competitors. |
| Online TSF parameter updates | FSNet, OneNet, D3A, DSOF, PROCEED | Updates weights, ensembles, or adaptation coefficients under streaming feedback. | The update locus is often model-parameter or output-level, and delayed labels can make updates chase outdated concepts. | Define delayed-label causality before method. |
| Residual/context calibration | SOLID/Reconditionor | Uses residual-context dependence to detect context-driven shift and calibrate prediction layers. | Does not make a channel-local Z-space correction inside a frozen backbone. | Use residual diagnostics as a cited motivation for CTSF's residual state. |
| Feature/Z-space adaptation | ADAPT-Z | Argues that distribution shift can be addressed by feature adjustment in latent space. | Does not center the no-op/near-identity conservation principle in the same way as CTSF. | Position CTSF as conservative residual-conditioned Z-space correction. |
| Test-time adaptation for TSF | TAFAS, PETSA | Uses partial/delayed labels, low-rank modules, and gates for efficient adaptation. | Often calibrates input/output or source forecaster modules rather than a per-channel hidden-state displacement governed by residual reliability. | Emphasize correction locus and no-op behavior. |
| Lightweight online feedback for foundation models | ELF | Avoids full online retraining and exploits feedback efficiently. | Mostly adapts forecasts or ensembling weights, not the frozen backbone's hidden representation. | Use as efficiency and deployment motivation. |

## CTSF Gap Statement

CTSF targets a narrow deployment problem: in leak-free online forecasting, delayed residuals may reveal that only some channels and hidden directions need correction, while the frozen backbone remains useful. Existing methods motivate online adaptation, but the local corpus leaves room for an adaptation rule that is conservative by construction: no reliable residual evidence, no correction; reliable residual evidence, a bounded low-rank Z-space adjustment.
