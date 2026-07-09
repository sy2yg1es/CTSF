# Source Map

| Source ID | Source | Type | What It Contributes | Used In |
|---|---|---|---|---|
| SRC-CFG | `paper_spine_config.json` | user configuration | Fixes the workflow as `build_from_materials`, English conference output, CTSF target, no Word package, and a conservative Z-space motivation. | confirmed_motivation, contribution, full paper |
| SRC-MAT-1 | `D:\model\paper_templates\CTSF\4690_Online_time_series_predic (1).pdf` | local paper PDF | ADAPT-Z motivates feature-space / Z-space adaptation and delayed feedback in online time series forecasting. | research dossier, related work, contribution boundary |
| SRC-MAT-2 | `ICLR-2025-fast-and-slow-streams-for-online-time-series-forecasting-without-information-leakage-Paper-Conference.pdf` | local paper PDF | DSOF supplies the leak-free rolling-window and delayed-label evaluation norm. | problem setup, experimental protocol |
| SRC-MAT-3 | `3690624.3709210.pdf` | local paper PDF | PROCEED shows that the horizon-induced temporal gap can make online updates adapt to outdated concepts. | introduction, related work |
| SRC-MAT-4 | `3637528.3671926.pdf` | local paper PDF | SOLID/Reconditionor supports residual-based shift detection and contextual calibration. | motivation, residual-state design |
| SRC-MAT-5 | `07575-KimH.pdf` | local paper PDF | TAFAS supports frozen source forecasters, delayed partial labels, and gated calibration under non-stationarity. | related work, design contrast |
| SRC-MAT-6 | `2506.23424v1.pdf` | local paper PDF | PETSA supports low-rank, parameter-efficient, dynamically gated test-time adaptation. | method design contrast |
| SRC-MAT-7 | `lee25ag.pdf` | local paper PDF | ELF supports lightweight online feedback use while avoiding full foundation model updates. | efficiency motivation |
| SRC-REF | `D:\model\references\references.bib` | local bibliography | Supplies verified local BibTeX seeds for online TSF, concept drift, normalization, continual learning, and representation adaptation. | citation bank, references.bib |
| SRC-CODE-1 | `models\prompt_z.py` | implementation | Defines DriftEncoder, ConfidenceGate, SparseMaskHead, LowRankModulator, ratio clamp, and near-no-op initialization. | method section |
| SRC-CODE-2 | `models\prompt_z_framework.py` | implementation | Shows frozen backbone encode/decode, residual-stat packing, mode-specific train/inference paths. | method section |
| SRC-CODE-3 | `core\residual_tracker.py` | implementation | Defines causal rolling residual statistics: error mean, slope, std, signed bias, and update gap. | residual-state subsection |
| SRC-CODE-4 | `engine\streaming_prompt_z.py` | implementation | Provides strict streaming evaluation, mode0/mode1, delayed label handling, and validation fallback. | experimental protocol |
| SRC-LOG | `logs\prompt_z\summary_strict_v1.csv` | experiment log | Gives optional pilot evidence for strict-split runs; final manuscript leaves experimental results blank per draft requirement. | evidence bank, results validation |
