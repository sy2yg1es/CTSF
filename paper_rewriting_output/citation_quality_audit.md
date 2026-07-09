# Citation Quality Audit

- Output directory: `paper_rewriting_output`
- Scene: conference
- Target citation count: 20
- Entries analyzed: 30
- Verified: 24 | Mismatched: 0 | Dead: 6
- Overall quality score: 75/100
- Status: PASS

> Each entry below includes a teaching note explaining *why* the citation quality matters.

## Per-Citation Analysis

| ID | DOI | Type | Resolves | Title Match | Year Match | Score | Status |
|---|---|---|---|---|---|---|---|
| CB01-1 | https://openreview.net/forum?i | survey | yes | 0% | no | 90 | verified |
| CB01-2 | https://openreview.net/forum?i | survey | yes | 0% | no | 90 | verified |
| CB01-3 | https://openreview.net/forum?i | survey | yes | 0% | no | 90 | verified |
| CB02-1 | https://openreview.net/forum?i | survey | yes | 0% | no | 85 | verified |
| CB02-2 | https://openreview.net/forum?i | survey | yes | 0% | no | 85 | verified |
| CB02-3 | https://openreview.net/forum?i | survey | yes | 0% | no | 85 | verified |
| CB03-1 | 10.1145/3690624.3709210}} | sota | no | 0% | no | 45 | dead |
| CB03-2 | 10.1145/3690624.3709210}} | sota | no | 0% | no | 45 | dead |
| CB03-3 | 10.1145/3690624.3709210}} | sota | no | 0% | no | 45 | dead |
| CB04-1 | 10.1145/3637528.3671926}} | sota | no | 0% | no | 45 | dead |
| CB04-2 | 10.1145/3637528.3671926}} | sota | no | 0% | no | 45 | dead |
| CB04-3 | 10.1145/3637528.3671926}} | sota | no | 0% | no | 45 | dead |
| CB05-1 | https://github.com/kimanki/TAF | sota | yes | 0% | no | 85 | verified |
| CB05-2 | https://github.com/kimanki/TAF | sota | yes | 0% | no | 85 | verified |
| CB05-3 | https://github.com/kimanki/TAF | sota | yes | 0% | no | 85 | verified |
| CB06-1 | https://arxiv.org/abs/2506.234 | sota | yes | 0% | no | 85 | verified |
| CB06-2 | https://arxiv.org/abs/2506.234 | sota | yes | 0% | no | 85 | verified |
| CB06-3 | https://arxiv.org/abs/2506.234 | sota | yes | 0% | no | 85 | verified |
| CB07-1 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 85 | verified |
| CB07-2 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 85 | verified |
| CB07-3 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 85 | verified |
| CB08-1 | https://openreview.net/forum?i | survey | yes | 0% | no | 75 | verified |
| CB08-2 | https://openreview.net/forum?i | survey | yes | 0% | no | 75 | verified |
| CB08-3 | https://openreview.net/forum?i | survey | yes | 0% | no | 75 | verified |
| CB09-1 | https://proceedings.neurips.cc | sota | yes | 0% | no | 75 | verified |
| CB09-2 | https://proceedings.neurips.cc | sota | yes | 0% | no | 75 | verified |
| CB09-3 | https://proceedings.neurips.cc | sota | yes | 0% | no | 75 | verified |
| CB10-1 | https://arxiv.org/abs/2401.041 | sota | yes | 0% | no | 85 | verified |
| CB10-2 | https://arxiv.org/abs/2401.041 | sota | yes | 0% | no | 85 | verified |
| CB10-3 | https://arxiv.org/abs/2401.041 | sota | yes | 0% | no | 85 | verified |

### CB03-1 — 10.1145/3690624.3709210}}

Status: **dead**

- DOI 10.1145/3690624.3709210}} does not resolve via Crossref

> Dead DOIs suggest the citation was hallucinated or the paper was retracted. Replace with a verified alternative or remove the citation.

### CB03-2 — 10.1145/3690624.3709210}}

Status: **dead**

- DOI 10.1145/3690624.3709210}} does not resolve via Crossref

> Dead DOIs suggest the citation was hallucinated or the paper was retracted. Replace with a verified alternative or remove the citation.

### CB03-3 — 10.1145/3690624.3709210}}

Status: **dead**

- DOI 10.1145/3690624.3709210}} does not resolve via Crossref

> Dead DOIs suggest the citation was hallucinated or the paper was retracted. Replace with a verified alternative or remove the citation.

### CB04-1 — 10.1145/3637528.3671926}}

Status: **dead**

- DOI 10.1145/3637528.3671926}} does not resolve via Crossref

> Dead DOIs suggest the citation was hallucinated or the paper was retracted. Replace with a verified alternative or remove the citation.

### CB04-2 — 10.1145/3637528.3671926}}

Status: **dead**

- DOI 10.1145/3637528.3671926}} does not resolve via Crossref

> Dead DOIs suggest the citation was hallucinated or the paper was retracted. Replace with a verified alternative or remove the citation.

### CB04-3 — 10.1145/3637528.3671926}}

Status: **dead**

- DOI 10.1145/3637528.3671926}} does not resolve via Crossref

> Dead DOIs suggest the citation was hallucinated or the paper was retracted. Replace with a verified alternative or remove the citation.

## Citation Diversity Gaps

**Missing foundational method or theory papers.** Only 0 of 30 entries (0%). Cite the 2-3 methods you build on. Be specific about what you inherit vs. change. Consider adding 1-3 foundational method or theory paper references.

**Missing dataset, benchmark, or evaluation protocol papers.** Only 0 of 30 entries (0%). Cite all datasets used. Standard benchmarks are expected. Consider adding 1-3 dataset, benchmark, or evaluation protocol paper references.

**Missing domain-application or impact papers.** Only 0 of 30 entries (0%). Optional. Consider adding 1-3 domain-application or impact paper references.


## Replacement Recommendations

- 6 dead DOIs detected. For each: (1) verify the paper exists via Google Scholar, (2) find the correct DOI, (3) update the citation bank, (4) re-run this audit.

## Scene-Specific Citation Strategy

For **conference** papers, your citation strategy should:

- **direct task or state-of-the-art paper**: Cite the 5-8 most recent competing methods. Conference reviewers check recency aggressively.
- **foundational method or theory paper**: Cite the 2-3 methods you build on. Be specific about what you inherit vs. change.
- **dataset, benchmark, or evaluation protocol paper**: Cite all datasets used. Standard benchmarks are expected.
- **survey, review, or meta-analysis**: Cite 1 recent survey if it helps position your work concisely.
- **domain-application or impact paper**: Optional.
- **limitation, robustness, reproducibility, or ethics paper**: Optional but helpful for discussion section.

## Citation Strategy Principles

- **Diversity over density.** A narrow citation pool makes your Introduction read as insular. Mix SOTA, foundational, benchmark, survey, and application papers.
- **Recency signals engagement.** Most citations should be from the last 3 years. Older citations are fine for foundational work, but they need a reason to be there.
- **Verifiability is non-negotiable.** Every DOI must resolve. A dead DOI in your final paper is a credibility failure that reviewers notice immediately.
- **Type matters by venue.** Journals expect deep SOTA coverage. Reports expect broad survey coverage. Competitions expect benchmark and leaderboard coverage. Match your strategy to your scene.
