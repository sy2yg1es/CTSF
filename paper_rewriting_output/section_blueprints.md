# Section Blueprints

| Section | Purpose | Core Claims | Evidence/Citations | Drafting Notes |
|---|---|---|---|---|
| Abstract | State the problem, CTSF mechanism, and evaluation scaffold without claiming final results. | C1-C5 | DSOF, PROCEED, SOLID, ADAPT-Z | Do not include numeric performance claims because experiments are blank. |
| Introduction | Move from delayed online shift to representation drift and conservative correction. | C1-C4 | DSOF, PROCEED, ADAPT-Z, SOLID | End with contribution bullets that are concrete and bounded. |
| Related Work | Contrast normalization, online updates, residual calibration, feature adjustment, and TTA. | C4 | RevIN, Dish-TS, SAN, DSOF, PROCEED, ADAPT-Z, TAFAS, PETSA, ELF | Each paragraph should explain what prior work leaves open. |
| Problem Setup | Define rolling windows, horizon delay, residual availability, and frozen-backbone objective. | C2, C5 | DSOF, PROCEED | Must prevent leakage ambiguity before method. |
| Method | Present residual state, drift encoder, gate/mask, low-rank correction, clamp, and training loss. | C1-C3 | Code evidence, PETSA for low-rank/gate context | Equations should be enough to implement the method. |
| Experimental Protocol | Specify datasets, baselines, metrics, diagnostics, and result tables to fill. | C5-C6 | Code/log evidence | Keep result cells empty; no fabricated numbers. |
| Discussion and Reproducibility | State experiment boundaries and identify code paths needed to reproduce. | C1-C6 | Claim register and local repo paths | Be honest that final benchmark evidence is pending, then name reproducible protocol artifacts. |
