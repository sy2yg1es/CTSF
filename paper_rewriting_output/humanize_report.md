# Humanize Check Report

- Matrix path: `paper_rewriting_output\humanize_matrix.md`
- Humanize tier: heavy
- Matrix rows: 14
- Manuscript paragraphs: 26
- Coverage: 54%
- Sentence length stddev: 61.97
- Connector density: 0.23/1k chars
- Status: PASS

## Dimension Scores

### D1 sentence structure: WARNING [required]
- Metrics: sentence_count=164, length_stddev=62.24, sentence_length_cv=0.813, repeated_start_ratio=0.21, uniform_length_runs=1, short_sentence_ratio=0.15, long_sentence_ratio=0.37
- Affected units: S154-S156
- D1 consecutive sentences have near-identical lengths: ['S154-S156'].

### D2 paragraph similarity: PASS [required]
- Metrics: paragraph_count=26, max_4gram_count=3, repeated_4gram_ratio=0.0303, paragraph_length_stddev=203.09, repeated_opening_ratio=0.0, min_paragraph_length=69, max_paragraph_length=855, adjacent_paragraph_similarity_mean=0.077, adjacent_paragraph_similarity_max=0.253
- No dimension-specific risk found.

### D3 information density: PASS [required]
- Metrics: generic_phrase_density=0.0, information_anchor_density=8.35, generic_phrase_count=0, anchor_count=92, mechanism_term_count=15, ttr=0.3458, token_count=1949, unique_token_count=674
- No dimension-specific risk found.

### D4 connector frequency: PASS [required]
- Metrics: connector_count=3, connector_density=0.23, max_paragraph_connector_density=2.23
- No dimension-specific risk found.

### D5 term-context matching: PASS [required]
- Metrics: frequent_terms_checked=12, contexts_checked=96, generic_context_ratio=0.0, mechanism_contexts=19, risky_terms=
- No dimension-specific risk found.

## Required Findings

- None

## Advisory Findings

- D1 consecutive sentences have near-identical lengths: ['S154-S156'].

## Threshold Profile

- adjacent_similarity_max_fail: 0.65
- adjacent_similarity_mean_warning: 0.45
- max_4gram_count_warning: 5
- max_connector_density: 8
- max_generic_density: 7
- max_paragraph_connector_density: 14
- max_repeated_start_ratio: 0.35
- max_term_generic_context_ratio: 0.45
- min_info_anchor_density: 2.5
- min_paragraph_length_stddev: 25
- min_sentence_length_stddev: 6
- repeated_4gram_ratio_fail: 0.15
- repeated_4gram_ratio_warning: 0.08
- sentence_length_cv_fail: 0.25
- sentence_length_cv_warning: 0.35
- ttr_fail_en: 0.25
- ttr_fail_zh: 0.35
- ttr_warning_en: 0.32
- ttr_warning_zh: 0.42
