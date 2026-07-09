# Motivation Options After Research

| Option | Motivation | Fit | Risk | Decision |
|---|---|---|---|---|
| M1 | Online shift invalidates the whole backbone, so CTSF should continually update model parameters. | Low. The saved motivation explicitly rejects whole-backbone knowledge failure as the default assumption. | Would duplicate many online-learning baselines and weaken the frozen-backbone claim. | Reject |
| M2 | Online shift mainly appears as hidden representation displacement while the backbone's learned temporal knowledge remains useful. Historical residuals can indicate when, how strongly, and where this displacement appears. | High. This matches the saved user motivation and the Prompt-Z implementation. | Needs careful boundaries because final experiment results are not inserted yet. | Select |
| M3 | CTSF is mostly a compute-efficiency method for test-time adaptation. | Medium. Efficiency matters, but it is secondary to the conservative Z-space correction principle. | If framed as the main claim, reviewers will expect hardware-level benchmarks. | Use only as supporting motivation |

## Selected Motivation

M2 is selected and treated as user-confirmed because it is the exact motivation recorded in the saved configuration: hidden representation drift is the primary target; the backbone is frozen; residual histories govern local low-rank gated corrections; and the system remains a no-op when drift evidence is unreliable.
