# Frozen protocol

## Corpora and splits

| Corpus | Source | Design split | Reported split |
|---|---|---|---|
| Advertising | Upworthy Research Archive, https://doi.org/10.17605/OSF.IO/JD64P (CC BY 4.0) | exploratory, 4,873 tests | confirmatory, 22,743 tests |
| Crowdfunding | https://huggingface.co/datasets/james-burton/kick_starter_funding_all_text | validation, 12,976 projects | test, 21,626 projects |

The holdout split of the advertising archive (4,871 tests) was not used.

## Inclusion rules (advertising)

A test enters the evaluation when all of the following hold:

1. every arm carries the same `eyecatcher_id`;
2. the test has between two and six arms;
3. all headline texts within the test are distinct;
4. every arm received at least 1,000 impressions;
5. the test accumulated at least ten clicks in total.

Applied to the confirmatory split this yields 9,932 tests and 44,341 packages,
covering 159,699,889 impressions and 2,119,865 clicks.

## Conditions

- `AGG` — one judgment per test; the model estimates the share of readers who
  would tap each headline; no persona.
- `IND` — one judgment per test; the model reports how likely one specific
  reader is to tap each headline; no persona.
- `PANEL` — one judgment per test per persona, eight personas built from
  published consumer-research constructs, aggregated by the mean.
- `RANDOM` — analytic lower bound.
- `ORACLE` — analytic upper bound.

## Model and parameters

DeepSeek chat-completions API. Model identifier `deepseek-flash`
(DeepSeek-V4.1-Flash), resolved by the provider from that alias on the
execution date; the alias `deepseek-v4-flash` was redirected to the same
weights during the study period. Non-thinking mode
(`thinking: {"type": "disabled"}`), temperature 0, `max_tokens` 512,
`response_format: {"type": "json_object"}`. The model-size comparison used
`deepseek-v4-pro` (DeepSeek-V4-Pro-0813) under otherwise identical settings.

Executions: first execution 29 September 2026, reported execution
30 September 2026 (Asia/Shanghai).

## Metrics

- Pairwise accuracy within a test, ties in the outcome excluded and counted.
- Top-1 accuracy against the realized winner.
- Decision lift, the click-through rate of the selected creative relative to
  the test mean.
- Share of oracle lift.
- Crowdfunding: pairwise accuracy over funded / not-funded pairs, equivalent to
  the area under the ROC curve, ties counted as one half.

Uncertainty: 95% intervals from 2,000 bootstrap resamples over tests.
