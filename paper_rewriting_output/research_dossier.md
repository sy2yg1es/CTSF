# Research Dossier

## Target Scene

The saved target is an English conference paper with ICLR as the official venue reference. The ICLR 2026 author guide emphasizes anonymous, double-blind submissions, a 9-page main-text limit at submission, use of the official LaTeX template, optional ethics and reproducibility statements, and code/supplementary material for reproducibility. The paper should therefore read as a compact machine-learning conference paper: a narrow technical gap, a named mechanism, a formal problem setup, precise claims, and experiments that validate each contribution rather than only report aggregate accuracy.

## Local Corpus Read

The local CTSF corpus is coherent. ADAPT-Z argues that online distribution shift can be addressed in feature space, not only by updating ordinary parameters. DSOF and PROCEED sharpen the deployment setting: multi-step forecasting creates delayed labels, and update/evaluation protocols can leak information if they reuse time points already touched by backpropagation. SOLID uses residuals to detect context-driven distribution shift and calibrates only when the shift appears meaningful. TAFAS and PETSA show the recent test-time-adaptation trend toward frozen source forecasters, partial or delayed supervision, lightweight calibration modules, low-rank adapters, and gates. ELF adds an efficiency argument: online feedback is useful, but updating large backbones during deployment is often too costly.

## Field Gap

The direct gap for CTSF is not simply that online forecasters need to adapt. The materials already cover adapter updates, two-stream updates, proactive parameter rescaling, feature adjustment, and test-time calibration. The narrower gap is that these methods often choose an adaptation locus first and then decide how to update it. CTSF instead starts from a diagnosis: when the backbone still encodes useful temporal knowledge, distribution shift may appear as a localized displacement in hidden representations. This diagnosis motivates a conservative Z-space correction that is residual-conditioned, channel-aware, low-rank, gated, and near-identity.

## Venue-Norm Implications

An ICLR-style paper needs the method to be stated as a principle, not only as code. The principle is conservation under uncertainty: if residual evidence does not support drift, the adaptation path should remain a no-op. This makes the method different from approaches that always fine-tune, always calibrate input/output, or always rescale model parameters. It also gives reviewers concrete axes to test: delayed-label causality, representation-local correction, no-op behavior, perturbation size, channel selectivity, and computational overhead.

## Evidence Available

The implementation supplies stronger evidence for the method design than for final quantitative claims. `PromptZModulator` has zero-initialized low-rank up projection, negative gate and mask biases, bounded scaling, and a ratio clamp. `ResidualTracker` only consumes delayed prediction errors and exposes five causal statistics per channel. `PromptZTSF` keeps the backbone frozen and permits gradient flow only through Prompt-Z during training. `streaming_prompt_z.py` records frozen, mode0, mode1, and validation-fallback evaluation paths. Existing logs in `summary_strict_v1.csv` can be used as pilot evidence, but the manuscript draft keeps the experimental result cells empty so no unsupported final benchmark claim is made.
