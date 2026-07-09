# Style Profile

## Target Voice

The paper should use a compact ICLR-style voice: direct problem framing, restrained novelty claims, and mechanism-first method description. Paragraphs should avoid broad claims such as "time series forecasting is important" unless the sentence immediately narrows to the online deployment problem. The preferred rhythm is observation, gap, design consequence, and evidence plan.

## Naming And Throughline

Use `CTSF` as the paper-level method and `Z-space correction` as the main mechanism. The implementation name `Prompt-Z` may appear as an instantiation detail, but the manuscript should not read as a code manual. The throughline is: delayed feedback makes online adaptation hard; residual histories are causal and channel-local; hidden states can drift while backbone knowledge remains useful; conservative Z-space corrections can adapt without opening unnecessary update paths.

## Claim Strength

Strong claims allowed: CTSF freezes the backbone; CTSF uses delayed residual statistics; CTSF applies a low-rank gated near-identity correction; the implementation includes no-op-biased initialization and a ratio clamp. Claims to avoid until final experiments are inserted: state-of-the-art performance, consistent improvement across all datasets, superiority over ADAPT-Z or DSOF, or final compute savings. The experiments section is intentionally left as a fill-in scaffold for final numbers.

## Sentence-Level Guidance

Use varied sentence lengths. Keep method equations precise. Avoid repetitive connectors such as "furthermore" and "moreover". Use citations where they carry a specific role: protocol, shift diagnosis, normalization, adapter efficiency, or online-learning baseline. Avoid process language about drafts, supervisors, or rewriting; the reader-facing paper should stand as a technical argument.
