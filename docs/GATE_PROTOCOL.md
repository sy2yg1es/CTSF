# Binary Channel Gate protocol

This document freezes the protocol used by `GateChangeTest`. Changes to these
items require a new validation cycle before TEST.

## Model and training stages

1. Load the pretrained PatchTST or iTransformer backbone and keep it frozen.
2. Train Prompt-Z Phase 1 with the correction fully enabled (`gamma=1`).
   Select the delta checkpoint on a chronological train-tail split; validation
   labels are not used for checkpoint selection.
3. Freeze the backbone, drift encoder, and Prompt-Z low-rank delta branch.
4. Generate causal per-window/per-channel signed correction advantages on the
   training tail and fit the normalized Huber regressor.
5. Apply the correction exactly when `regressor_score > 0`. This threshold is
   structural and is never tuned on validation or TEST.
6. On an independent train-only safety block, choose one deployment mode from
   `frozen`, `fixed`, and `learned_regressor`. A non-frozen mode must improve
   aggregate MSE by at least 0.2% and improve at least three of four contiguous
   blocks. Otherwise deployment defaults to `frozen`.

The binary gate has shape `[B,C,1]` and is the channel selector. The legacy
continuous confidence gate and the legacy `[B,C,1]` sparse mask are not used in
the formal Phase-2 or TEST path. They remain in old Prompt-Z state dictionaries
only so existing Phase-1 checkpoints load strictly.

## Causal features and warm-up

The finalized feature set is `causal_augmented`: frozen Prompt-Z drift state,
five residual-tracker statistics, and seven frozen/correction output summary
statistics. Target values are used only to form delayed training labels; TEST
features never contain target or oracle information.

Tracker warm-up initializes causal residual statistics and preserves the same
delayed residual queue into the following split. It is not a gamma/mask
regularization warm-up. The gate needs no floor, entropy, L1, or sparsity
regularizer.

Gate-history lengths are fixed to:

- H=1: 2,000 train-tail windows
- H=12: 8,000 train-tail windows
- H=24: 2,000 train-tail windows
- H=48: 2,000 train-tail windows

## Validation and TEST

Validation reports all contiguous blocks but cannot change the threshold,
safety mode, feature set, or checkpoint. `eval_test_oracle.py` is an oracle
diagnostic and is not the formal evaluator.

`eval_binary_gate_test.py` is the one-shot formal TEST entry point. It rejects
legacy checkpoints without the embedded protocol version, zero threshold, and
train-only safety mode. It reports Frozen MSE, Gate/Prompt-Z MSE, relative
improvement, gate usage, effective correction ratio, and runtime. It never
computes or saves TEST oracle targets.

## Logging and online-order invariant

Training logs are summary-based. They retain current step, train loss,
validation MSE, best validation MSE, checkpoint updates, gate activation or
selected mode, supervised target positive ratio, necessary gate validation
metrics, and correction magnitude. Per-window JSON/CSV logs are not part of
the formal protocol.

Windows remain sequential. The residual tracker and delayed target queue are
updated in chronological order. Engineering optimizations must not batch or
parallelize state-dependent windows.
