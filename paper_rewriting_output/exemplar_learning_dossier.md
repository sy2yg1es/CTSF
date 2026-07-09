# Exemplar Learning Dossier

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
