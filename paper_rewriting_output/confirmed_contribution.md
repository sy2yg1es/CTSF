# Confirmed Contribution

## Core Contribution

| Field | Content |
|---|---|
| Main contribution statement | CTSF introduces a conservative Z-space correction framework for online time series forecasting. It freezes the forecasting backbone and applies only residual-conditioned, low-rank, gated, near-identity hidden-state corrections when causal residual evidence indicates representation drift. |
| Contribution type | method |
| Reviewer payoff | Reviewers get a bounded adaptation mechanism that is easier to audit than full online fine-tuning: the paper exposes when adaptation opens, how large the correction can be, and why no-op behavior is a first-class design goal. |

## Why This Contribution Is Needed

| Field | Content |
|---|---|
| Field problem | Online time series forecasting faces non-stationarity, concept drift, and horizon-induced label delay. Existing adaptation can leak future information, chase outdated concepts, or perturb model parameters even when the backbone remains useful. |
| Specific gap | The local SOTA covers parameter updates, contextual calibration, feature adjustment, and TTA modules, but it leaves room for a conservative hidden-representation correction governed by residual reliability and channel-local drift evidence. |
| Concrete challenge | The method must adapt fast enough for online streams without using unavailable labels, while also avoiding harmful updates when the residual signal is noisy, absent, or not predictive of current hidden-state displacement. |
| Why prior work leaves it unresolved | DSOF and PROCEED sharpen the delayed-feedback protocol; SOLID uses residuals for contextual calibration; ADAPT-Z motivates feature adjustment; TAFAS and PETSA use gates and lightweight modules. CTSF combines these pressures at a different correction locus: bounded per-channel Z-space displacement in a frozen backbone. |

## How This Paper Responds

| Field | Content |
|---|---|
| Design response | CTSF builds a residual state from delayed errors, encodes it with hidden summaries, opens a confidence gate and sparse channel mask, and applies a low-rank correction whose norm is clamped relative to the hidden representation. |
| Evidence required | A full submission should report leak-free comparisons against frozen backbones, random/no-op Prompt-Z, mode0, mode1, selected fallback, and online/TTA baselines; it should also include ablations for residual features, gate, mask, rank, and ratio clamp. |
| Evidence available | Code evidence supports the architecture and conservation mechanisms; local logs supply pilot strict-split results; local PDFs support the research gap and evaluation protocol. The final manuscript intentionally leaves result tables unfilled. |
| Evidence missing | Final benchmark tables, statistical comparisons, complete compute measurements, and cross-backbone ablations remain to be inserted before submission. |

## Claim Boundary

| Field | Content |
|---|---|
| Strong claims allowed | CTSF is residual-conditioned, frozen-backbone, low-rank, gated, sparse, near-identity, and ratio-clamped by implementation. The method is designed for delayed-label online forecasting. |
| Claims to soften or avoid | Avoid claiming state-of-the-art accuracy, universal improvement, or proven superiority over ADAPT-Z, DSOF, TAFAS, PETSA, or ELF until final experiments are filled. |
| Novelty risk | ADAPT-Z already claims feature-space online adaptation, so CTSF must foreground the conservative no-op policy, residual-state conditioning, per-channel gate/mask, and bounded local hidden correction. |
| Significance risk | Without final results, significance must be framed as a plausible and testable design contribution rather than as empirically settled performance. |
