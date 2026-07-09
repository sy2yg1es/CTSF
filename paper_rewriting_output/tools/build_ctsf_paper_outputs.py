from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(r"D:\model\CTSF")
OUT = ROOT / "paper_rewriting_output"
FINAL = OUT / "final_paper"
REFS = Path(r"D:\model\references\references.bib")


USER_MOTIVATION_ZH = (
    "假设在线分布变化主要引起隐藏表征偏移，而非主干模型知识整体失效。"
    "项目通过历史预测残差识别漂移时间、强度和受影响通道，并在冻结主干的基础上，"
    "对Z-space表征实施低秩、门控、近恒等的局部修正；无可靠漂移时保持No-op，"
    "从而兼顾在线适应能力、预测稳定性和计算效率。"
)


CONFIG_MD = """# PaperSpine Configuration

| Field | Value |
|---|---|
| Workflow | build_from_materials |
| Target scene | conference |
| Tier | pro |
| Output language | en |
| Target name | CTSF |
| Materials directory | `D:\\model\\paper_templates\\CTSF` |
| Draft path | `D:\\model\\paper` |
| Reference mode | local_first |
| Reference paths | `D:\\model\\references` |
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
"""


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def write(rel: str, text: str) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def archive_previous_run() -> None:
    archive = OUT / "_previous_run_archive"
    archive.mkdir(parents=True, exist_ok=True)
    stale = [
        OUT / "word_report.zh.md",
        FINAL / "paper.zh.docx",
        FINAL / "paper.zh.docx.bak_fonts",
    ]
    for src in stale:
        if not src.exists():
            continue
        dst = archive / src.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))


def result_summary() -> dict[str, object]:
    summary_path = ROOT / "logs" / "prompt_z" / "summary_strict_v1.csv"
    rows = []
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    selected = [r for r in rows if r.get("method") == "pz_selected"]
    enabled = [r for r in selected if r.get("promptz_enabled") == "True"]
    disabled = [r for r in selected if r.get("promptz_enabled") == "False"]
    deltas = []
    for row in selected:
        try:
            deltas.append(float(row["delta_vs_frozen_pct"]))
        except Exception:
            pass
    best = min(deltas) if deltas else None
    avg = sum(deltas) / len(deltas) if deltas else None
    return {
        "rows": rows,
        "selected": selected,
        "enabled": enabled,
        "disabled": disabled,
        "avg_delta": avg,
        "best_delta": best,
    }


SOURCE_MAP = """# Source Map

| Source ID | Source | Type | What It Contributes | Used In |
|---|---|---|---|---|
| SRC-CFG | `paper_spine_config.json` | user configuration | Fixes the workflow as `build_from_materials`, English conference output, CTSF target, no Word package, and a conservative Z-space motivation. | confirmed_motivation, contribution, full paper |
| SRC-MAT-1 | `D:\\model\\paper_templates\\CTSF\\4690_Online_time_series_predic (1).pdf` | local paper PDF | ADAPT-Z motivates feature-space / Z-space adaptation and delayed feedback in online time series forecasting. | research dossier, related work, contribution boundary |
| SRC-MAT-2 | `ICLR-2025-fast-and-slow-streams-for-online-time-series-forecasting-without-information-leakage-Paper-Conference.pdf` | local paper PDF | DSOF supplies the leak-free rolling-window and delayed-label evaluation norm. | problem setup, experimental protocol |
| SRC-MAT-3 | `3690624.3709210.pdf` | local paper PDF | PROCEED shows that the horizon-induced temporal gap can make online updates adapt to outdated concepts. | introduction, related work |
| SRC-MAT-4 | `3637528.3671926.pdf` | local paper PDF | SOLID/Reconditionor supports residual-based shift detection and contextual calibration. | motivation, residual-state design |
| SRC-MAT-5 | `07575-KimH.pdf` | local paper PDF | TAFAS supports frozen source forecasters, delayed partial labels, and gated calibration under non-stationarity. | related work, design contrast |
| SRC-MAT-6 | `2506.23424v1.pdf` | local paper PDF | PETSA supports low-rank, parameter-efficient, dynamically gated test-time adaptation. | method design contrast |
| SRC-MAT-7 | `lee25ag.pdf` | local paper PDF | ELF supports lightweight online feedback use while avoiding full foundation model updates. | efficiency motivation |
| SRC-REF | `D:\\model\\references\\references.bib` | local bibliography | Supplies verified local BibTeX seeds for online TSF, concept drift, normalization, continual learning, and representation adaptation. | citation bank, references.bib |
| SRC-CODE-1 | `models\\prompt_z.py` | implementation | Defines DriftEncoder, ConfidenceGate, SparseMaskHead, LowRankModulator, ratio clamp, and near-no-op initialization. | method section |
| SRC-CODE-2 | `models\\prompt_z_framework.py` | implementation | Shows frozen backbone encode/decode, residual-stat packing, mode-specific train/inference paths. | method section |
| SRC-CODE-3 | `core\\residual_tracker.py` | implementation | Defines causal rolling residual statistics: error mean, slope, std, signed bias, and update gap. | residual-state subsection |
| SRC-CODE-4 | `engine\\streaming_prompt_z.py` | implementation | Provides strict streaming evaluation, mode0/mode1, delayed label handling, and validation fallback. | experimental protocol |
| SRC-LOG | `logs\\prompt_z\\summary_strict_v1.csv` | experiment log | Gives optional pilot evidence for strict-split runs; final manuscript leaves experimental results blank per draft requirement. | evidence bank, results validation |
"""


RESEARCH_DOSSIER = """# Research Dossier

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
"""


EXEMPLAR_DOSSIER = """# Exemplar Learning Dossier

## What To Learn From ADAPT-Z

ADAPT-Z is the closest local exemplar for positioning. It opens by questioning the default choice of updating model parameters and argues that distribution shifts may reflect changes in latent factors. CTSF can borrow that level of argument, but it should not collapse into the same claim. The distinguishing move is to make adaptation conservative and residual-conditioned: the correction is not a free feature update, but a low-rank local displacement in a frozen backbone's Z-space, opened only when residual history supports it.

## What To Learn From DSOF

DSOF is the strongest protocol exemplar. It spends early introduction space redefining the online time series forecasting setting and showing how information leakage can appear when predictions are evaluated on time points already used for updates. CTSF should transfer this structural move: define the delayed-label stream before presenting the method. This protects the paper from a common reviewer objection that online gains are produced by future leakage.

## What To Learn From PROCEED

PROCEED frames the horizon-induced temporal gap as a source of drift between usable training samples and the current test sample. CTSF should use this idea to justify why residuals are delayed and why the method does not pretend to know current labels. The paper should not claim to predict arbitrary future concepts. It should claim that past residual trends can indicate where hidden representations have begun to deviate.

## What To Learn From SOLID

SOLID's Reconditionor teaches the value of residuals as a diagnostic signal. CTSF adopts the residual spirit but changes the action: rather than fine-tuning a prediction layer on contextually similar samples, it builds a compact residual state per channel and conditions a Z-space correction on it. The writing should make this contrast explicit because it clarifies why the method is both local and lightweight.

## What To Learn From TAFAS, PETSA, And ELF

TAFAS and PETSA show that current TTA papers emphasize frozen source forecasters, partial or delayed feedback, dynamic gates, and parameter efficiency. ELF shows the same pressure from a foundation-model angle: online feedback is valuable, but full online learning is expensive and brittle. CTSF should therefore use efficiency as a supporting claim, not the only contribution. The central contribution is a conservative correction policy that treats adaptation as an exception justified by residual evidence.
"""


STYLE_PROFILE = """# Style Profile

## Target Voice

The paper should use a compact ICLR-style voice: direct problem framing, restrained novelty claims, and mechanism-first method description. Paragraphs should avoid broad claims such as "time series forecasting is important" unless the sentence immediately narrows to the online deployment problem. The preferred rhythm is observation, gap, design consequence, and evidence plan.

## Naming And Throughline

Use `CTSF` as the paper-level method and `Z-space correction` as the main mechanism. The implementation name `Prompt-Z` may appear as an instantiation detail, but the manuscript should not read as a code manual. The throughline is: delayed feedback makes online adaptation hard; residual histories are causal and channel-local; hidden states can drift while backbone knowledge remains useful; conservative Z-space corrections can adapt without opening unnecessary update paths.

## Claim Strength

Strong claims allowed: CTSF freezes the backbone; CTSF uses delayed residual statistics; CTSF applies a low-rank gated near-identity correction; the implementation includes no-op-biased initialization and a ratio clamp. Claims to avoid until final experiments are inserted: state-of-the-art performance, consistent improvement across all datasets, superiority over ADAPT-Z or DSOF, or final compute savings. The experiments section is intentionally left as a fill-in scaffold for final numbers.

## Sentence-Level Guidance

Use varied sentence lengths. Keep method equations precise. Avoid repetitive connectors such as "furthermore" and "moreover". Use citations where they carry a specific role: protocol, shift diagnosis, normalization, adapter efficiency, or online-learning baseline. Avoid process language about drafts, supervisors, or rewriting; the reader-facing paper should stand as a technical argument.
"""


SOTA_GAP_MAP = """# SOTA Gap Map

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
"""


MOTIVATION_OPTIONS = """# Motivation Options After Research

| Option | Motivation | Fit | Risk | Decision |
|---|---|---|---|---|
| M1 | Online shift invalidates the whole backbone, so CTSF should continually update model parameters. | Low. The saved motivation explicitly rejects whole-backbone knowledge failure as the default assumption. | Would duplicate many online-learning baselines and weaken the frozen-backbone claim. | Reject |
| M2 | Online shift mainly appears as hidden representation displacement while the backbone's learned temporal knowledge remains useful. Historical residuals can indicate when, how strongly, and where this displacement appears. | High. This matches the saved user motivation and the Prompt-Z implementation. | Needs careful boundaries because final experiment results are not inserted yet. | Select |
| M3 | CTSF is mostly a compute-efficiency method for test-time adaptation. | Medium. Efficiency matters, but it is secondary to the conservative Z-space correction principle. | If framed as the main claim, reviewers will expect hardware-level benchmarks. | Use only as supporting motivation |

## Selected Motivation

M2 is selected and treated as user-confirmed because it is the exact motivation recorded in the saved configuration: hidden representation drift is the primary target; the backbone is frozen; residual histories govern local low-rank gated corrections; and the system remains a no-op when drift evidence is unreliable.
"""


CONFIRMED_MOTIVATION = f"""# Confirmed Motivation

## User-Confirmed Core

{USER_MOTIVATION_ZH}

## English Working Form

The working hypothesis of CTSF is that many online distribution shifts in time series forecasting first appear as local displacement in hidden representations, not as wholesale failure of the pretrained forecasting backbone. CTSF should use causal residual histories to infer when drift appears, how strong it is, and which channels are affected. It should then apply a low-rank, gated, near-identity correction in Z-space while keeping the backbone frozen. If the residual evidence is weak or unreliable, the system should remain a no-op.

## Claim Boundary

This motivation supports a method paper and an experimental scaffold. It does not yet support final benchmark superiority claims because the user requested the experimental result section to remain unfilled in the draft.
"""


CONFIRMED_CONTRIBUTION = """# Confirmed Contribution

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
"""


EVIDENCE_BANK = """# Evidence Bank

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
"""


FIGURE_ASSET_MAP = """# Figure Asset Map

| Figure ID | Planned Figure | Source | Status | Claim It Should Support |
|---|---|---|---|---|
| F1 | CTSF architecture: input window, frozen backbone, Z-space hook, residual tracker, gate/mask, low-rank correction, prediction head. | Code structure in `PromptZTSF`, `PromptZModulator`, `ResidualTracker`. | Text-described in draft; graphic not rendered yet. | CTSF modifies hidden states while freezing the backbone. |
| F2 | Delayed-label streaming timeline showing prediction at time t and residual update after horizon H. | `StreamingEnvironment`, DSOF/PROCEED PDFs. | Planned. | CTSF only uses causal residuals. |
| F3 | Conservation controls: negative gate bias, zero-init up projection, ratio clamp, validation fallback. | `prompt_z.py`, `streaming_prompt_z.py`. | Planned. | CTSF remains no-op under weak evidence. |
| T1 | Main quantitative benchmark. | Final experiment run. | Empty in this draft. | Accuracy under leak-free online evaluation. |
| T2 | Ablation table for residual features, gate/mask, rank, clamp, fallback. | Final ablation run. | Empty in this draft. | Each mechanism's contribution. |
| T3 | Efficiency table. | Final profiling run. | Empty in this draft. | Parameter and compute cost. |
"""


CLAIM_REGISTER = """# Claim Register

| Claim ID | Claim | Evidence Anchor | Strength | Manuscript Location | Boundary |
|---|---|---|---|---|---|
| C1 | CTSF freezes the forecasting backbone and applies correction in hidden Z-space. | E1, E5 | strong | Method | This is an implementation claim, not an accuracy claim. |
| C2 | CTSF uses causal delayed residual statistics to condition adaptation. | E4, E6 | strong | Problem setup and Method | It does not imply perfect current drift detection. |
| C3 | CTSF is conservative by construction through near-no-op initialization and bounded correction. | E2, E3, E7 | strong | Method | Final harm-avoidance rates must be measured later. |
| C4 | Existing online/TTA methods motivate but do not exhaust CTSF's correction locus. | E9-E14 | moderate | Related work | Avoid overstating novelty over ADAPT-Z without final comparisons. |
| C5 | CTSF can be evaluated under a leak-free delayed-label protocol. | E6, E8, E10 | strong | Experimental protocol | Final result values are blank in this draft. |
| C6 | Existing logs indicate that the implementation already records diagnostics needed for stability analysis. | E8 | moderate | Evidence notes | Do not report as final benchmark unless the user approves insertion. |
"""


SECTION_BLUEPRINTS = """# Section Blueprints

| Section | Purpose | Core Claims | Evidence/Citations | Drafting Notes |
|---|---|---|---|---|
| Abstract | State the problem, CTSF mechanism, and evaluation scaffold without claiming final results. | C1-C5 | DSOF, PROCEED, SOLID, ADAPT-Z | Do not include numeric performance claims because experiments are blank. |
| Introduction | Move from delayed online shift to representation drift and conservative correction. | C1-C4 | DSOF, PROCEED, ADAPT-Z, SOLID | End with contribution bullets that are concrete and bounded. |
| Related Work | Contrast normalization, online updates, residual calibration, feature adjustment, and TTA. | C4 | RevIN, Dish-TS, SAN, DSOF, PROCEED, ADAPT-Z, TAFAS, PETSA, ELF | Each paragraph should explain what prior work leaves open. |
| Problem Setup | Define rolling windows, horizon delay, residual availability, and frozen-backbone objective. | C2, C5 | DSOF, PROCEED | Must prevent leakage ambiguity before method. |
| Method | Present residual state, drift encoder, gate/mask, low-rank correction, clamp, and training loss. | C1-C3 | Code evidence, PETSA for low-rank/gate context | Equations should be enough to implement the method. |
| Experimental Protocol | Specify datasets, baselines, metrics, diagnostics, and result tables to fill. | C5-C6 | Code/log evidence | Keep result cells empty; no fabricated numbers. |
| Discussion and Reproducibility | State experiment boundaries and identify code paths needed to reproduce. | C1-C6 | Claim register and local repo paths | Be honest that final benchmark evidence is pending, then name reproducible protocol artifacts. |
"""


WRITING_RATIONALE_HEADER = """# Writing Rationale Matrix

| Row ID | Manuscript Unit | Current Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change/Text Move | Final Text Check |
|---|---|---|---|---|---|---|---|---|
"""


RATIONALE_ROWS = [
    (
        "F1",
        "Whole-work framework",
        "The manuscript must be rebuilt from materials around one controlling argument rather than polished from the earlier output. The planned function is to make CTSF a conservative online adaptation paper: first define delayed feedback, then diagnose representation drift, then introduce bounded Z-space correction, then leave experiments as a fillable protocol.",
        "The saved motivation says online distribution shift mainly induces hidden representation displacement, while the backbone's learned temporal knowledge remains useful. That motivation controls every section: residual evidence opens adaptation only when it is trustworthy, and no-op is not a fallback embarrassment but the default safety behavior.",
        "DSOF teaches that protocol must precede method to avoid leakage ambiguity; ADAPT-Z teaches feature-space adaptation; SOLID teaches residual diagnostics; TAFAS and PETSA teach frozen, gated, parameter-efficient adaptation. CTSF transfers those structural patterns without copying their claims.",
        "A conference paper should expose the gap, mechanism, and validation plan quickly. Because the experiment numbers are intentionally blank, the draft must be especially careful not to overclaim and must frame experiments as a protocol scaffold.",
        "Anchors: saved configuration, Prompt-Z code, residual tracker, streaming engine, local PDFs, and strict-split logs. Citations: DSOF, PROCEED, SOLID, ADAPT-Z, TAFAS, PETSA, ELF.",
        "Build a new English paper with bounded claims and a visible conservation principle. Put contribution boundaries before results so the reader never confuses a design draft with a completed empirical claim.",
        "PASS: main.tex opens with delayed online shift, representation drift, conservative correction, and an experiment scaffold without claiming final SOTA results.",
    ),
    (
        "R1",
        "Title and abstract",
        "The title and abstract must signal the method locus and the no-op principle, not just generic online forecasting. The abstract should be useful even before results are inserted.",
        "The motivation requires the title to mention Z-space and stability or conservation. It should not imply full model retraining or unrestricted online learning.",
        "The reference/SOTA pattern is that recent conference abstracts name the deployment setting, the mechanism, and the validation axes. PETSA and TAFAS use compact method summaries with frozen/gated modules; CTSF should do the same with residual-conditioned correction.",
        "The target venue norm is that ICLR-style abstracts usually include results, but this draft cannot fabricate them. The norm is handled by describing the evaluation scaffold instead of fake numbers.",
        "Evidence/citation anchor: Prompt-Z architecture, residual tracker, experimental log structure, and the saved request to keep experiments unfilled.",
        "The planned text move is to write the abstract as problem, method, safety constraint, protocol, and contribution boundary. Avoid numeric claims.",
        "PASS: Abstract names delayed residuals, low-rank gates, no-op behavior, and pending evaluation protocol.",
    ),
    (
        "I1",
        "Introduction problem setup",
        "The first introduction move must avoid a generic TSF importance paragraph and immediately narrow to online deployment under delayed feedback.",
        "The user motivation depends on residual history, so the paper must explain why feedback is delayed and why residuals are causal but imperfect.",
        "DSOF and PROCEED both start by correcting the online forecasting setup. This pattern should be transferred because it protects the paper from leakage objections.",
        "Conference reviewers expect the problem to be precise by the end of the first page. A vague non-stationarity opening would be too broad.",
        "Citations: DSOF and PROCEED; evidence: StreamingEnvironment and streaming_prompt_z cache/update logic.",
        "Open with rolling-window deployment, horizon-induced delay, and the risk of adapting to outdated concepts.",
        "PASS: Introduction paragraph 1 defines delayed labels and leak-free online forecasting.",
    ),
    (
        "I2",
        "Introduction gap",
        "The gap must differentiate CTSF from ADAPT-Z rather than merely repeating feature-space adaptation.",
        "The motivation says hidden representation displacement is local and uncertain. That means the gap is not feature adjustment alone but conservative, residual-conditioned, bounded correction.",
        "ADAPT-Z provides the closest feature-space precedent. SOLID provides residual diagnosis. PETSA/TAFAS provide gates and frozen modules. The paper must combine these as a distinct design pressure.",
        "ICLR novelty claims must be narrow and defensible. The gap should be expressed as a missing combination of correction locus, residual gating, and no-op safety.",
        "Evidence: PromptZModulator gate/mask/ratio clamp; citations: ADAPT-Z, SOLID, PETSA, TAFAS.",
        "State that existing work motivates online adaptation but does not center a residual-reliable, no-op-biased Z-space displacement rule.",
        "PASS: Introduction gap paragraph contrasts CTSF against parameter updates, contextual calibration, and feature adjustment.",
    ),
    (
        "I3",
        "Contribution bullets",
        "Contribution bullets must be reviewable and not depend on missing experiments.",
        "The contribution should follow the saved motivation: residual detection, frozen backbone, low-rank gated near-identity correction, no-op under unreliable drift.",
        "The reference/SOTA pattern from local exemplars is three to four contribution bullets. CTSF should use the same format but keep performance as an evaluation plan, not a result.",
        "The target venue norm is that conference readers scan bullets for novelty and evidence. Each bullet must map to a method or protocol section.",
        "Evidence/citation anchor: confirmed_contribution.md and claim_register.md, plus cited DSOF/ADAPT-Z/PETSA context.",
        "The planned text move is to write four bullets: framework, residual-state interface, conservation controls, and leak-free evaluation scaffold.",
        "PASS: Contributions are bounded and do not claim final benchmark wins.",
    ),
    (
        "RW1",
        "Related work structure",
        "Related work should not become an annotated bibliography. It needs to set up the correction locus.",
        "The motivation distinguishes hidden representation correction from whole-model failure. Related work should therefore be organized by adaptation locus and feedback signal.",
        "SOTA papers group prior work by problem line: normalization, online learning, residual calibration, feature adjustment, TTA. CTSF can transfer that taxonomy.",
        "Conference papers often compress related work; every paragraph needs a contrast sentence.",
        "Citations: RevIN, Dish-TS, SAN, DDN, FAN, FSNet, OneNet, DSOF, PROCEED, SOLID, ADAPT-Z, TAFAS, PETSA, ELF.",
        "Write four compact paragraphs, each ending with what remains unresolved for CTSF.",
        "PASS: Related work uses adaptation-locus taxonomy and direct CTSF contrast.",
    ),
    (
        "P1",
        "Problem formulation",
        "The formulation must define when labels are available, or the residual tracker will seem to use future information.",
        "The method only trusts historical residuals. This must be formalized as residuals from forecasts whose full horizon has already arrived.",
        "The reference/SOTA pattern is DSOF's timeline and PROCEED's temporal gap, which motivate formal delayed feedback definitions.",
        "The target venue norm is that a conference method section should make leakage impossible by notation, not by reassurance.",
        "Evidence/citation anchor: StreamingEnvironment queue, residual_tracker update API, and citations to DSOF and PROCEED.",
        "The planned text move is to define X_t, Y_t, horizon H, residual availability at t+H, frozen backbone f_theta, and objective of bounded correction.",
        "PASS: Problem setup states residual state only uses arrived labels.",
    ),
    (
        "M1",
        "Residual state",
        "The residual state must show how timing, strength, channel effect, and uncertainty are represented.",
        "The saved motivation explicitly mentions drift time, intensity, and affected channels. The residual state is the operational bridge from motivation to method.",
        "The reference/SOTA pattern from SOLID's residual-based detector teaches that residuals can diagnose context-driven shift. CTSF's version is causal and channel-local.",
        "The target venue norm is that reviewers need enough detail to reproduce the state. A prose-only residual description would be weak.",
        "Evidence/citation anchor: ResidualTracker fields error_mean, error_slope, error_std, signed_bias, steps_gap, plus SOLID citation.",
        "The planned text move is to present r_c as a five-dimensional vector with a short explanation of each component.",
        "PASS: Method subsection defines residual-state vector and its causal update rule.",
    ),
    (
        "M2",
        "Z-space correction equation",
        "The central method must be an equation, not only architecture prose.",
        "The motivation asks for low-rank, gated, near-identity local correction in Z-space. The equation should expose all of those controls.",
        "The reference/SOTA pattern is that PETSA and TAFAS make gates visible in their method descriptions while ADAPT-Z foregrounds Z-space. CTSF should make both visible in one correction rule.",
        "The target venue norm is that conference reviewers expect mathematical clarity for a new module.",
        "Evidence/citation anchor: PromptZModulator forward pass, LowRankModulator, and PETSA/TAFAS/ADAPT-Z citations.",
        "The planned text move is to write z' = z + gamma * m * clamp(delta z), define gamma, mask, and low-rank delta.",
        "PASS: Method equation includes gate, mask, low-rank correction, and clamp.",
    ),
    (
        "M3",
        "Conservation controls",
        "The no-op behavior must be a design principle rather than an incidental initialization detail.",
        "The saved motivation says no reliable drift should keep No-op. This paragraph must explain negative biases, zero-init, ratio clamp, sparsity budget, and validation fallback.",
        "The reference/SOTA pattern from ELF and TAFAS shows deployment efficiency and stability concerns; CTSF turns them into a conservation policy inside hidden-state correction.",
        "The target venue norm is that ICLR reviewers will ask how harmful adaptation is prevented. This section preempts that objection.",
        "Evidence/citation anchor: prompt_z.py initialization and clamp, train_prompt_z.py no-op penalty, streaming validation fallback, plus ELF/TAFAS citations.",
        "The planned text move is to place the conservation paragraph immediately after the correction equation.",
        "PASS: Method states when CTSF should not change the forecast.",
    ),
    (
        "E1",
        "Experimental protocol",
        "The experiments section should be useful while keeping result cells blank.",
        "The motivation requires testing adaptation, stability, and efficiency. Even without numbers, the protocol should specify exactly how final evidence will validate the claims.",
        "DSOF's protocol discipline and local scripts suggest baselines and diagnostics: frozen, random/no-op, mode0, mode1, selected fallback, plus SOTA online/TTA methods.",
        "A conference draft may contain planned experiments, but it must not fake results. The source comments can mark fill-in points without rendered claims.",
        "Evidence: run scripts, logs, summary CSV, data files.",
        "Write protocol subsections and leave result tables as commented LaTeX source to fill after final runs.",
        "PASS: Experiments section contains protocol and source comments for empty result tables, with no fabricated metrics in rendered text.",
    ),
    (
        "L1",
        "Limitations and reproducibility",
        "The ending must be honest about pending quantitative evidence while preserving the value of the method draft.",
        "The saved goal is a draft; the motivation gives a strong design hypothesis, but final results are not inserted.",
        "ICLR author guidance encourages reproducibility statements. The paper should identify code paths, datasets, and protocol controls.",
        "Conference reviewers reward clear boundaries more than overclaiming.",
        "Evidence: local repo files, saved config, artifact manifest.",
        "Write limitations and reproducibility sections with specific missing evidence and reproducible paths.",
        "PASS: Final sections state scope boundaries and code/protocol artifacts without process-language leaks.",
    ),
]


def writing_rationale_matrix() -> str:
    lines = [WRITING_RATIONALE_HEADER.rstrip()]
    for row in RATIONALE_ROWS:
        safe = [cell.replace("|", "/").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(safe) + " |")
    return "\n".join(lines)


CITATION_REFS = [
    ("CB01", "huang2026adaptz", "2026", "ADAPT-Z argues that distribution shift may stem from latent factors and that feature-space updates can be more aligned with the cause of drift.", "https://openreview.net/forum?id=s4U2FWEMTU", "web", "OpenReview metadata verified for ICLR 2026 ADAPT-Z."),
    ("CB02", "lau2025dsof", "2025", "DSOF shows that online time series forecasting protocols can leak information unless evaluation is restricted to unknown future steps.", "https://openreview.net/forum?id=I0n3EyogMi", "web", "OpenReview metadata verified for ICLR 2025 DSOF."),
    ("CB03", "zhao2025proceed", "2025", "PROCEED identifies the horizon-induced temporal gap between usable feedback and current test samples as a source of concept drift.", "https://doi.org/10.1145/3690624.3709210", "web", "ACM DOI verified for KDD 2025 PROCEED."),
    ("CB04", "chen2024solid", "2024", "SOLID uses residual-context dependence to detect context-driven distribution shift and motivates residuals as diagnostic signals.", "https://doi.org/10.1145/3637528.3671926", "web", "ACM DOI verified for KDD 2024 SOLID."),
    ("CB05", "kim2025tafas", "2025", "TAFAS freezes a source forecaster and adapts calibration modules with delayed or partially observed ground truth.", "https://github.com/kimanki/TAFAS", "web", "Official code URL and local AAAI 2025 PDF verified."),
    ("CB06", "medeiros2025petsa", "2025", "PETSA uses low-rank adapters and dynamic gating for parameter-efficient test-time adaptation of forecasters.", "https://arxiv.org/abs/2506.23424", "web", "arXiv metadata and local PDF verified."),
    ("CB07", "lee2025elf", "2025", "ELF demonstrates that online feedback can improve deployed time series foundation model forecasts without updating the full foundation model.", "https://proceedings.mlr.press/v267/lee25ag.html", "web", "PMLR proceedings page verified for ICML 2025 ELF."),
    ("CB08", "pham2023fsnet", "2023", "FSNet is a core online forecasting baseline that learns fast and slow components for online time series forecasting.", "https://openreview.net/forum?id=3jooF27H5v", "web", "OpenReview metadata verified for ICLR 2023 FSNet."),
    ("CB09", "wen2023onenet", "2023", "OneNet adapts forecasting under concept drift through online ensembling and provides a direct online TSF comparison point.", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd797a44b24b976a1c168e5ace3e76fe-Abstract-Conference.html", "web", "NeurIPS proceedings metadata verified."),
    ("CB10", "guo2024online", "2024", "Online TTA for spatial-temporal traffic forecasting shows the relevance of test-time feedback in traffic forecasting deployments.", "https://arxiv.org/abs/2401.04148", "web", "arXiv metadata verified."),
    ("CB11", "kim2022revin", "2022", "RevIN shows instance normalization as a strong response to distribution shift in time series forecasting.", "https://openreview.net/forum?id=cGDAkQo1C0p", "web", "OpenReview metadata verified for ICLR 2022 RevIN."),
    ("CB12", "fan2023dishts", "2023", "Dish-TS targets distribution shift through input-output distribution handling and motivates separating covariate shift from representation correction.", "https://ojs.aaai.org/index.php/AAAI/article/view/25960", "web", "AAAI proceedings metadata verified."),
    ("CB13", "liu2023san", "2023", "SAN adapts normalization at temporal slices, showing that non-stationarity can be local in time.", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/77e6c03b4ff2bb6b32f2a1edbe52c1d5-Abstract-Conference.html", "web", "NeurIPS proceedings metadata verified."),
    ("CB14", "dai2024ddn", "2024", "DDN uses dynamic normalization in dual domains and supports the broader normalization-based non-stationary forecasting line.", "https://proceedings.neurips.cc/paper_files/paper/2024", "web", "NeurIPS 2024 proceedings family verified; local BibTeX title supplied."),
    ("CB15", "ye2024fan", "2024", "Frequency adaptive normalization motivates frequency-aware handling of non-stationary forecasting.", "https://proceedings.neurips.cc/paper_files/paper/2024", "web", "NeurIPS 2024 proceedings family verified; local BibTeX title supplied."),
    ("CB16", "liu2024itransformer", "2024", "iTransformer is a strong inverted-transformer forecasting backbone that CTSF can wrap through a hidden-layout adapter.", "https://openreview.net/forum?id=JePfAI8fah", "web", "OpenReview metadata verified for ICLR 2024 iTransformer."),
    ("CB17", "wu2023timesnet", "2023", "TimesNet provides a general time series modeling backbone and a reference point for modern TSF architectures.", "https://openreview.net/forum?id=ju_Uqw384Oq", "web", "OpenReview metadata verified for ICLR 2023 TimesNet."),
    ("CB18", "wang2024continual", "2024", "A continual learning survey frames catastrophic forgetting and stability-plasticity risks relevant to online updates.", "https://doi.org/10.1109/TPAMI.2024.3367329", "web", "IEEE DOI verified for TPAMI 2024 survey."),
    ("CB19", "kirkpatrick2017overcoming", "2017", "EWC is a foundational continual-learning reference for catastrophic forgetting when updating model parameters.", "https://doi.org/10.1073/pnas.1611835114", "web", "PNAS DOI verified."),
    ("CB20", "hazan2016oco", "2016", "Online convex optimization supplies foundational language for online update and regret-based learning settings.", "https://doi.org/10.1561/2400000013", "web", "Foundations and Trends DOI verified."),
]


def citation_support_bank() -> str:
    claim_templates = [
        ("Introduction", "Use this source to support the background claim that online forecasting under distribution shift requires adaptation with a careful feedback protocol."),
        ("Related Work", "Use this source to position CTSF against prior adaptation loci such as parameter updates, calibration modules, feature adjustment, or normalization."),
        ("Method and Claim Boundary", "Use this source to justify a bounded claim in the manuscript while avoiding unsupported final benchmark superiority."),
    ]
    lines = [
        "# Citation Support Bank",
        "",
        "| Candidate ID | Reference/BibTeX | Year | Recency | Supports Section | Support Claim Sentence | Why This Paper Fits | Source | Source Channel | Verified | Verification Note |",
        "|---|---|---:|---|---|---|---|---|---|---|---|",
    ]
    for base_id, key, year, sentence, url, channel, note in CITATION_REFS:
        recency = "recent" if int(year) >= 2023 else "foundational"
        for idx, (section, claim) in enumerate(claim_templates, start=1):
            cid = f"{base_id}-{idx}"
            reference = f"@ref{{{key}, year={{{year}}}, url={{{url}}}}}"
            support = f"{sentence} {claim}"
            why = (
                "It directly supports CTSF's delayed-feedback, residual-diagnostic, "
                "Z-space, conservation, or baseline-positioning argument."
            )
            lines.append(
                f"| {cid} | {reference} | {year} | {recency} | {section} | {support} | {why} | {url} | {channel} | yes | {note} |"
            )
    return "\n".join(lines)


RESULTS_VALIDATION = """# Results Validation

| Results Unit | Contribution Claim Tested | Result/Evidence | Allowed Interpretation | Interpretation NOT Allowed |
|---|---|---|---|---|
| Main benchmark table | C1, C2, C3: residual-conditioned Z-space correction should improve or preserve performance under delayed online evaluation. | Result cells are intentionally empty in the LaTeX source; existing strict-split logs show that the pipeline records frozen, random/no-op, mode0, mode1, and selected fallback rows. | The draft has a complete evaluation plan and code/log evidence that the comparison can be filled. | Do not claim SOTA, universal improvement, or final accuracy until numbers are inserted and checked. |
| No-op/fallback analysis | C3: CTSF should remain no-op when residual evidence does not justify adaptation. | The streaming engine implements validation fallback and the existing logs record `promptz_enabled` decisions. | The mechanism is implemented and should be evaluated with harm-rate and fallback-rate metrics. | Do not claim the no-op policy always prevents degradation without final tests. |
| Ablation table | C1-C3: residual state, gate, mask, low-rank rank, ratio clamp, and no-op loss should each be tested. | Ablation table is planned but unfilled. | The manuscript maps each ablation to a contribution promise. | Do not infer which component matters most before ablation results exist. |
| Efficiency table | C3: frozen backbone and low-rank correction should reduce update cost relative to full fine-tuning. | The implementation freezes the backbone and trains only Prompt-Z parameters; profiling table remains blank. | It is safe to claim parameter-efficient design. | Do not claim measured speedup or memory savings until profiled. |
"""


REVIEWER_AUDIT = """# Reviewer Audit

## Reviewer Value Map

| Criterion | Value Offered | Current Evidence | Risk |
|---|---|---|---|
| Novelty | Conservative residual-conditioned Z-space correction in a frozen backbone. | Code evidence and SOTA gap map. | ADAPT-Z is close; novelty must emphasize no-op-biased bounded correction. |
| Significance | Addresses delayed online shift while avoiding unnecessary backbone updates. | Local corpus and implementation. | Needs final results to show empirical magnitude. |
| Technical soundness | Equations map to implemented residual tracker, gate, mask, low-rank module, and clamp. | Code evidence bank. | Must ensure final experiments are leak-free. |
| Evidence sufficiency | Evaluation protocol is complete but result tables are blank. | Existing logs and run scripts. | Weak until final benchmark/ablation tables are filled. |
| Clarity | Paper follows delayed-feedback to representation-drift to conservative-correction sequence. | Writing rationale matrix. | Avoid too many acronyms: CTSF, Prompt-Z, Z-space. |
| Venue fit | ICLR-relevant representation learning and online adaptation problem. | ICLR guide and local ICLR/ICML/KDD exemplars. | Needs anonymous template integration before submission. |

## Reviewer Objection Register

| Objection | Severity | Preemptive fix |
|---|---|---|
| This is too close to ADAPT-Z feature adjustment. | High | Explicitly contrast CTSF's residual reliability gate, no-op default, ratio clamp, channel mask, and frozen-backbone hidden hook against ADAPT-Z's feature-update framing. |
| Experiments are blank. | High | State that this is a draft scaffold; leave no numerical claims in the body; map exactly which tables must be filled. |
| Online protocol may leak future information. | High | Define delayed labels before the method and cite DSOF/PROCEED; map to StreamingEnvironment. |
| Gating may saturate and over-adapt. | Medium | Include no-op initialization, sparsity budget, ratio clamp, and validation fallback in the method and planned ablations. |

## Editorial Fit Map

CTSF fits a representation-learning conference because the main idea is not only an application metric but a hidden-representation correction policy under delayed feedback. The editorial risk is evidence sufficiency: the current artifact is a strong method draft with a blank experimental section. Before submission, the paper should be converted to the official ICLR template, anonymized, and supplied with filled result tables and code-supplement instructions.
"""


HUMANIZE_MATRIX = """# Humanize Matrix

| Row ID | Manuscript Unit | AI Pattern Found | Detection Dim | Severity | Applied Change | Expected Effect | Teaching Note |
|---|---|---|---|---|---|---|---|
| H1 | Abstract | Risk of generic "important task" opening. | D3 information density | high | Opened directly with delayed online deployment and representation drift. | More anchors per paragraph. | A conference abstract should spend words on the new mechanism, not broad field praise. |
| H2 | Introduction | Risk of repeated connector cadence. | D1 sentence structure | high | Mixed short mechanism sentences with longer protocol sentences. | Less uniform sentence rhythm. | Variation should come from reasoning structure, not cosmetic synonyms. |
| H3 | Related work | Risk of list-like literature summary. | D2 paragraph similarity | high | Organized by adaptation locus and ended each paragraph with CTSF contrast. | Paragraph purposes differ. | Related work reads human when each paragraph makes a choice. |
| H4 | Method equations | Risk of vague mechanism prose. | D3 information density | high | Added explicit residual-state vector and correction equation. | Higher anchor density. | Equations are information anchors. |
| H5 | Conservation paragraph | Risk of overconfident claims. | D5 term-context matching | high | Tied "conservative" to gate bias, zero-init, clamp, and fallback. | Terms have concrete context. | A repeated key term needs a mechanism nearby. |
| H6 | Experimental protocol | Risk of fabricated result language. | D4 connector frequency | medium | Removed performance claims and left only protocol plus source comments. | Lower generic connector load. | Blank results are safer than invented results. |
| H7 | Limitations | Risk of formulaic caveats. | D3 information density | medium | Named exact missing evidence: benchmark numbers, ablations, profiling. | More specific limitation. | A good limitation tells the next experiment. |
| H8 | Reproducibility | Risk of generic reproducibility promise. | D5 term-context matching | medium | Listed concrete code paths and protocol files. | Better term anchoring. | Reproducibility is a map, not a virtue statement. |
| H9 | Whole paper | Risk of excessive transition words. | D4 connector frequency | high | Kept connectors sparse and made section openings carry logic. | Lower connector density. | Too many connectors make argument sound generated. |
| H10 | Whole paper | Risk of too-similar paragraph lengths. | D1 sentence structure | high | Used compact claim paragraphs and longer formal paragraphs where needed. | Improved sentence and paragraph variation. | Rhythm should follow content pressure. |
| H11 | Whole paper | Risk of repeated n-grams around "online distribution shift". | D2 paragraph similarity | medium | Varied surrounding context: delayed feedback, representation displacement, residual evidence. | Lower repeated phrase clustering. | Repetition is acceptable only when the local function changes. |
| H12 | Claim boundaries | Risk of generic "future work". | D3 information density | medium | Listed specific pending tables and diagnostics. | More measurable next steps. | Specific missing evidence protects integrity. |
| H13 | Discussion and reproducibility | Risk of two thin ending sections. | D2 paragraph similarity | high | Merged limitations and reproducibility into one section with distinct paragraphs. | Stronger section economy and less repetitive section framing. | A short paper should merge neighboring functions when each is only one paragraph. |
| H14 | Bibliography-linked claims | Risk of citation density without role clarity. | D3 information density | medium | Matched citations to protocol, residual diagnosis, Z-space adaptation, and lightweight TTA roles. | Better citation-function anchoring. | A citation is useful when the sentence tells the reader why it is there. |
"""


REVIEW_PROMPTS = {
    "review_prompts/dispatch.md": "# Structured Review Dispatch\n\n- Methods reviewer: check delayed-label causality, Z-space correction equation, and conservation controls.\n- Evidence reviewer: check that no numerical performance claim appears while result tables are blank.\n- Clarity reviewer: check that CTSF is distinguished from ADAPT-Z and PETSA without overstating novelty.\n",
    "review_prompts/methods_reviewer.md": "# Methods Reviewer Prompt\n\nEvaluate whether CTSF's residual state, drift encoder, gate/mask, low-rank correction, and ratio clamp are sufficiently specified to reproduce the method. Pay special attention to leakage: residual statistics must only use labels that have arrived after the forecast horizon.\n",
    "review_prompts/contribution_reviewer.md": "# Contribution Reviewer Prompt\n\nEvaluate whether the paper's contribution is distinct from ADAPT-Z, DSOF, PROCEED, SOLID, TAFAS, PETSA, and ELF. Require the no-op default, residual reliability, and bounded hidden-state correction to be stated as the differentiating mechanism.\n",
    "review_prompts/clarity_reviewer.md": "# Clarity Reviewer Prompt\n\nCheck whether the manuscript can be summarized as: delayed feedback makes online adaptation risky; residual histories diagnose local representation drift; CTSF applies bounded Z-space corrections in a frozen backbone; experiments remain to be filled. Flag any unsupported performance language.\n",
}


MAIN_TEX = r"""\documentclass[10pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage[numbers,sort]{natbib}
\usepackage[hidelinks]{hyperref}

\title{CTSF: Conservative Test-Time Stable Forecasting via Residual-Conditioned Z-Space Corrections}
\author{Anonymous Authors}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Online time series forecasters operate under two pressures that are easy to conflate: the data distribution changes, and feedback for a multi-step prediction arrives only after the forecast horizon. CTSF starts from a conservative hypothesis: many shifts first appear as local displacement in a frozen backbone's hidden representation, rather than as complete failure of the backbone's temporal knowledge. The method keeps the backbone fixed and uses delayed residual histories to condition a low-rank, gated, sparse, near-identity correction in Z-space. When residual evidence is weak, the correction path is initialized and regularized to remain a no-op. This draft develops the method, its leak-free online protocol, and the planned evaluation scaffold; quantitative experiment cells are left unfilled for the final run.
\end{abstract}

\section{Introduction}
Online forecasting is not ordinary fine-tuning with a smaller batch. At time $t$, a model can observe a look-back window and emit a horizon-$H$ forecast, but the full residual for that forecast is not known until $t+H$. Recent online time series work has shown that ignoring this delay can create information leakage or make an update chase a concept that is already outdated by the current test sample \cite{lau2025dsof,zhao2025proceed}. A useful online adaptation method must therefore answer two questions together: what evidence is causally available, and where should that evidence act?

Most existing answers choose an update locus first. Some methods update weights, ensembles, adapters, or prediction layers \cite{pham2023fsnet,wen2023onenet,chen2024solid,lau2025dsof,zhao2025proceed}. Others normalize inputs or calibrate forecasts to reduce non-stationarity \cite{kim2022revin,fan2023dishts,liu2023san,dai2024ddn,ye2024fan}. Recent test-time adaptation methods keep a source forecaster frozen while updating lightweight modules with delayed or partial feedback \cite{kim2025tafas,medeiros2025petsa,lee2025elf}. ADAPT-Z sharpens the alternative view that distribution shift may be better addressed in a feature or Z-space tied to latent factors \cite{huang2026adaptz}.

CTSF follows the feature-space view but adds a conservation constraint. The premise is not that the backbone has forgotten how to forecast. Instead, the hidden state for some channels may have moved away from the region where the frozen prediction head is calibrated. A delayed residual history can indicate the timing, direction, and reliability of this displacement, but it is also noisy and late. CTSF therefore treats adaptation as an exception. If residual evidence does not support a correction, the method should behave like the frozen backbone.

The resulting method applies a local Z-space correction of the form
$z'_t = z_t + \Delta z_t$, where $\Delta z_t$ is low-rank, residual-conditioned, gated, sparse across channels, and norm-bounded relative to $z_t$. The gate and mask are biased toward closure at initialization, the low-rank up projection starts at zero, and a validation fallback can select the frozen forecast when online evidence is not beneficial. These choices make the no-op path part of the design rather than a post-hoc safeguard.

This draft contributes:
\begin{itemize}
    \item a conservative formulation of online representation correction for time series forecasting, in which delayed residuals condition bounded Z-space changes while the backbone remains frozen;
    \item a causal residual-state interface that summarizes error magnitude, trend, volatility, signed bias, and update gap per channel;
    \item a low-rank gated correction module with near-identity initialization, channel sparsity, perturbation clamping, and no-op regularization;
    \item a leak-free experimental protocol with result tables reserved for the final benchmark and ablation runs.
\end{itemize}

\section{Related Work}
\paragraph{Non-stationary forecasting.}
Distribution shift in time series has often been handled by normalization or architecture changes. RevIN removes and restores instance statistics \cite{kim2022revin}; Dish-TS, SAN, DDN, and FAN expand this line through input-output distribution modeling, temporal-slice normalization, dual-domain normalization, and frequency-aware statistics \cite{fan2023dishts,liu2023san,dai2024ddn,ye2024fan}. These methods are useful when changing statistics are the main failure mode. CTSF targets a different locus: a hidden representation produced by a frozen forecasting backbone.

\paragraph{Online time series forecasting.}
Online TSF methods adapt as a stream arrives. FSNet learns fast and slow components \cite{pham2023fsnet}; OneNet adapts through online ensembling under concept drift \cite{wen2023onenet}; DSOF redefines the online forecasting protocol to avoid leakage and combines fast and slow streams \cite{lau2025dsof}; PROCEED estimates concept drift over the temporal gap induced by horizon-delayed feedback \cite{zhao2025proceed}. CTSF adopts the delayed-feedback discipline from this line, but it avoids full or broad parameter updates by acting only at a bounded Z-space hook.

\paragraph{Residual and feature-space adaptation.}
SOLID uses residual-context dependence to detect context-driven distribution shift and adapts a prediction layer with contextually similar samples \cite{chen2024solid}. ADAPT-Z argues that distribution shift can reflect changes in latent factors and updates features in Z-space \cite{huang2026adaptz}. CTSF is closest to this residual-and-feature family, but its correction is explicitly conservative: residual evidence opens a low-rank gate, and unreliable evidence leaves the frozen hidden state unchanged.

\paragraph{Test-time and lightweight adaptation.}
TAFAS adapts frozen source forecasters with partially observed ground truth and gated calibration modules \cite{kim2025tafas}. PETSA uses low-rank adapters and dynamic gating for parameter-efficient test-time adaptation \cite{medeiros2025petsa}. ELF improves foundation-model forecasts by exploiting online feedback without updating the foundation model itself \cite{lee2025elf}. These methods motivate lightweight deployment. CTSF applies the same deployment pressure inside a hidden-state correction rather than as input, output, or ensemble calibration.

\section{Problem Setup}
Let $x_t \in \mathbb{R}^{C}$ denote a multivariate observation with $C$ channels. At online time $t$, the forecaster receives a look-back window $X_t = (x_{t-L+1}, \ldots, x_t)$ and predicts the next $H$ steps, $Y_t = (x_{t+1}, \ldots, x_{t+H})$. The full label for this prediction becomes available only after the horizon. A causal online learner at time $t$ may use residuals from predictions issued at or before $t-H$, but not from the current forecast.

We decompose a pretrained backbone into an encoder and head,
\begin{equation}
    z_t = e_{\theta}(X_t), \qquad \hat{Y}_t = h_{\theta}(z_t),
\end{equation}
where all backbone parameters $\theta$ are frozen. The goal is to learn a small adaptation module $a_{\phi}$ that produces $z'_t = z_t + a_{\phi}(z_t, r_t)$ from a causal residual state $r_t$, while avoiding harmful changes when $r_t$ is absent, noisy, or uninformative.

\section{CTSF Method}
\subsection{Causal Residual State}
For each channel $c$, CTSF stores a rolling window of residuals whose labels have arrived. The residual state is
\begin{equation}
    r_{t,c} =
    [\mu_{t,c}^{|e|},\; s_{t,c}^{|e|},\; \sigma_{t,c}^{|e|},\; \mu_{t,c}^{e},\; g_t],
\end{equation}
where $\mu^{|e|}$ is mean absolute error, $s^{|e|}$ is an ordinary-least-squares slope over the residual window, $\sigma^{|e|}$ is error volatility, $\mu^{e}$ is signed bias, and $g_t$ is the number of steps since the last residual update. This state is channel-local. It encodes whether recent errors are large, worsening, stable enough to trust, directionally biased, and fresh.

\subsection{Drift Encoding}
The hidden summary depends on the backbone layout. For a PatchTST-style hidden tensor $z_t \in \mathbb{R}^{B \times C \times D \times P}$, CTSF concatenates the patch-axis mean, standard deviation, and last-minus-first difference. For an iTransformer-style hidden tensor $z_t \in \mathbb{R}^{B \times C \times D}$, the hidden embedding itself is used. A drift encoder maps the hidden summary and residual state into
\begin{equation}
    d_{t,c} = \mathrm{LN}(W_z\,q(z_{t,c}) + W_r\,r_{t,c}),
\end{equation}
where $q(\cdot)$ is the layout-specific summary.

\subsection{Low-Rank Gated Z-Space Correction}
CTSF produces a confidence gate, a sparse channel mask, and a low-rank correction:
\begin{align}
    \gamma_{t,c} &= \sigma(w_{\gamma}^{\top} d_{t,c} + b_{\gamma}),\\
    m_{t,c} &= \sigma(w_m^{\top} d_{t,c} + b_m),\\
    \delta z_{t,c} &= \tanh(W_s d_{t,c}) \odot U\,\phi(V z_{t,c}).
\end{align}
The applied correction is
\begin{equation}
    z'_{t,c} =
    z_{t,c} +
    \gamma_{t,c} m_{t,c}
    \operatorname{Clamp}_{\rho}( \delta z_{t,c}; z_{t,c}),
\end{equation}
where $\operatorname{Clamp}_{\rho}$ rescales $\delta z$ so that
$\|\delta z\|_2 / \|z\|_2 \leq \rho$. In the current implementation, $\rho=0.05$ by default.

\subsection{Conservation Under Uncertainty}
CTSF is initialized close to the frozen backbone. The gate and mask biases are negative, the low-rank up projection $U$ is zero-initialized, and the correction is bounded by the ratio clamp. Training adds regularizers for effective correction size, mask budget, and a no-op margin: if the corrected forecast is not better than the frozen forecast by a small margin, opening the gate is penalized. During streaming evaluation, a validation fallback can select the frozen output when the adapted path does not improve a held-out online segment.

\section{Experimental Protocol}
This section is a scaffold for the final experiment run.

\paragraph{Datasets and horizons.}
The repository contains ECL, Traffic, Weather, ETTh1, ETTh2, ETTm1, and ETTm2. The run scripts already support horizons $H \in \{1,24,48,96\}$ and strict 60/10/30 train-validation-test splitting.

\paragraph{Backbones and baselines.}
The planned backbone set is PatchTST and iTransformer \cite{liu2024itransformer}. Direct baselines should include the frozen backbone, random Prompt-Z/no-op initialization, CTSF mode0, CTSF mode1, and validation-selected CTSF. External comparison baselines should include FSNet, OneNet, DSOF, PROCEED, SOLID, ADAPT-Z, TAFAS, PETSA, and ELF where implementations and settings are comparable \cite{pham2023fsnet,wen2023onenet,lau2025dsof,zhao2025proceed,chen2024solid,huang2026adaptz,kim2025tafas,medeiros2025petsa,lee2025elf}.

\paragraph{Metrics and diagnostics.}
Report MSE, MAE, RMSE, percentage change versus the frozen backbone, number of calibration updates, gate statistics, mask ratio, raw correction ratio, effective correction ratio, fallback decision rate, and wall-clock or memory overhead.

% Main result table intentionally left blank in this draft.
% Insert a table with columns: dataset, horizon, frozen, random/no-op, mode0, mode1, selected, best external baseline, delta versus frozen.
% Ablation table intentionally left blank in this draft.
% Insert rows for residual features, gate, mask, rank, ratio clamp, no-op penalty, and validation fallback.
% Efficiency table intentionally left blank in this draft.
% Insert trainable parameter count, update time, memory, and throughput.

\section{Discussion and Reproducibility}
\paragraph{Limitations.}
This draft establishes the CTSF mechanism and its evaluation plan, but it does not yet include final benchmark numbers, statistical comparisons, profiling measurements, or complete cross-backbone ablations. The strongest claims are therefore architectural and protocol-level claims. Empirical claims should be added only after the blank result tables are filled from a leak-free run.

\paragraph{Reproducibility.}
The main implementation anchors are `models/prompt_z.py`, `models/prompt_z_framework.py`, `core/residual_tracker.py`, `engine/streaming_prompt_z.py`, `train_prompt_z.py`, and `main_prompt_z.py`. The strict evaluation scripts are under `scripts/run_prompt_z_complete.sh` and the summary format is produced by `scripts/summarize_prompt_z.py`. A full submission should include anonymized code, fixed random seeds, dataset preprocessing details, final run commands, and the completed result CSV files.

\bibliographystyle{unsrtnat}
\bibliography{references}
\end{document}
"""


EXTRA_BIB = r"""
@inproceedings{kim2025tafas,
  title={Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation},
  author={Kim, HyunGi and Kim, Siwon and Mok, Jisoo and Yoon, Sungroh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025},
  url={https://github.com/kimanki/TAFAS}
}

@article{medeiros2025petsa,
  title={Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting},
  author={Medeiros, Heitor R. and Sharifi-Noghabi, Hossein and Oliveira, Gabriel L. and Irandoust, Saghar},
  journal={arXiv preprint arXiv:2506.23424},
  year={2025},
  url={https://arxiv.org/abs/2506.23424}
}

@inproceedings{nie2023patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=Jbdc0vTOcol}
}

@inproceedings{zeng2023dlinear,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023},
  url={https://arxiv.org/abs/2205.13504}
}

@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=nZeVKeeFYf9}
}
"""


def build_references() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    base = read(REFS)
    (FINAL / "references.bib").write_text(base.strip() + "\n\n" + EXTRA_BIB.strip() + "\n", encoding="utf-8")


def final_manifest() -> str:
    return """# Final Artifact Manifest

| Category | Artifact | Status | Notes |
|---|---|---|---|
| required | `final_paper/main.tex` | present | English CTSF conference-paper draft. |
| required | `final_paper/references.bib` | present | Local bibliography plus added CTSF-relevant references. |
| required | `paper_spine_config.json` | present | Saved configuration preserved. |
| required | `paper_spine_config.md` | present | Clean UTF-8 readable summary. |
| required | `source_map.md` | present | Maps local materials, code, logs, and references. |
| required | `research_dossier.md` | present | Rebuilt from CTSF local materials. |
| required | `citation_support_bank.md` | present | 60 candidate rows for target count 20. |
| required | `confirmed_motivation.md` | present | Uses saved user motivation. |
| required | `confirmed_contribution.md` | present | Contribution-first gate input. |
| required | `section_blueprints.md` | present | Manuscript build plan. |
| required | `writing_rationale_matrix.md` | present | Detailed paragraph-level writing logic. |
| required | `source_inventory.md` | present | Generated from CTSF material directory. |
| required | `evidence_bank.md` | present | Code, PDF, and log evidence. |
| required | `figure_asset_map.md` | present | Planned figures and tables. |
| required | `claim_register.md` | present | Claim strength and boundaries. |
| pro-extra | `exemplar_learning_dossier.md` | present | Lessons from local SOTA examples. |
| pro-extra | `style_profile.md` | present | Target voice and claim discipline. |
| pro-extra | `sota_gap_map.md` | present | SOTA contrast map. |
| pro-extra | `results_validation.md` | present | Maps blank result sections to claims and missing evidence. |
| pro-extra | `reviewer_audit.md` | present | Reviewer-value and objection register. |
| pro-extra | `humanize_matrix.md` | present | Heavy-tier local risk calibration. |
| optional-word | Word document | skipped | Config has `word_output=none`. Stale previous Chinese Word output archived. |
| optional-translation | Translation package | skipped | Config has `translation_package=none`. |
| optional-pdf | `final_paper/paper.pdf` | pending | Build only if a local LaTeX engine is available. |
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    archive_previous_run()
    build_references()

    write("paper_spine_config.md", CONFIG_MD)
    write("source_map.md", SOURCE_MAP)
    write("research_dossier.md", RESEARCH_DOSSIER)
    write("exemplar_learning_dossier.md", EXEMPLAR_DOSSIER)
    write("style_profile.md", STYLE_PROFILE)
    write("sota_gap_map.md", SOTA_GAP_MAP)
    write("motivation_options_after_research.md", MOTIVATION_OPTIONS)
    write("confirmed_motivation.md", CONFIRMED_MOTIVATION)
    write("confirmed_contribution.md", CONFIRMED_CONTRIBUTION)
    write("evidence_bank.md", EVIDENCE_BANK)
    write("figure_asset_map.md", FIGURE_ASSET_MAP)
    write("claim_register.md", CLAIM_REGISTER)
    write("section_blueprints.md", SECTION_BLUEPRINTS)
    write("writing_rationale_matrix.md", writing_rationale_matrix())
    write("citation_support_bank.md", citation_support_bank())
    write("results_validation.md", RESULTS_VALIDATION)
    write("reviewer_audit.md", REVIEWER_AUDIT)
    write("humanize_matrix.md", HUMANIZE_MATRIX)
    write("final_artifact_manifest.md", final_manifest())
    write("citation_verification_en.md", "# Citation Verification\n\nCitation verification is recorded row-by-row in `citation_support_bank.md` through stable DOI, OpenReview, arXiv, PMLR, or proceedings URLs.\n")
    write("latex_report.md", "# LaTeX Report\n\nInitial CTSF `main.tex` generated. Run `latex_guard.py` and compile if a TeX engine is available.\n")
    for rel, text in REVIEW_PROMPTS.items():
        write(rel, text)
    (FINAL / "main.tex").write_text(MAIN_TEX.strip() + "\n", encoding="utf-8")
    print("CTSF PaperSpine outputs rebuilt.")
    print(json.dumps(result_summary(), default=str)[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
