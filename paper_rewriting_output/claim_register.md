# Claim Register

| Claim ID | Claim | Evidence Anchor | Strength | Manuscript Location | Boundary |
|---|---|---|---|---|---|
| C1 | CTSF freezes the forecasting backbone and applies correction in hidden Z-space. | E1, E5 | strong | Method | This is an implementation claim, not an accuracy claim. |
| C2 | CTSF uses causal delayed residual statistics to condition adaptation. | E4, E6 | strong | Problem setup and Method | It does not imply perfect current drift detection. |
| C3 | CTSF is conservative by construction through near-no-op initialization and bounded correction. | E2, E3, E7 | strong | Method | Final harm-avoidance rates must be measured later. |
| C4 | Existing online/TTA methods motivate but do not exhaust CTSF's correction locus. | E9-E14 | moderate | Related work | Avoid overstating novelty over ADAPT-Z without final comparisons. |
| C5 | CTSF can be evaluated under a leak-free delayed-label protocol. | E6, E8, E10 | strong | Experimental protocol | Final result values are blank in this draft. |
| C6 | Existing logs indicate that the implementation already records diagnostics needed for stability analysis. | E8 | moderate | Evidence notes | Do not report as final benchmark unless the user approves insertion. |
