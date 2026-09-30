# Consumer digital twin screening — release package

This archive supports the manuscript *Do Consumer Digital Twins Improve
Pre-Launch Screening? Evidence from 9,932 Randomized Advertising Tests and
21,626 Real Crowdfunding Launches*.

## What is here

- `runs/` — the raw model responses of the reported execution, one JSON record
  per API call. Each record carries the call identifier, the condition, the
  model, the returned scores, the finish reason and token usage.
- `code/` — the full pipeline: corpus download, task construction, prompt
  definitions, the API runner, the metric implementation and the figure
  scripts.
- `protocol.md` — the frozen protocol: corpora, splits, inclusion rules,
  conditions, prompts, model identifiers and parameters.
- `checksums.txt` — SHA-256 digests of every released file.

## What is not here

The two source corpora are not redistributed because they are already public
and openly licensed:

- Upworthy Research Archive, https://doi.org/10.17605/OSF.IO/JD64P (CC BY 4.0)
- Kickstarter corpus, https://huggingface.co/datasets/james-burton/kick_starter_funding_all_text

The per-item responses of the first execution are not included; that execution
was retained only in aggregate and its records are no longer available. Its
aggregate values are reported in Table 5 of the manuscript.

## How to rebuild the evaluation

```bash
python code/fetch_upworthy.py            # download the two Upworthy splits
python code/build_tasks.py               # apply the inclusion rules, build prompts
python code/run_llm.py --tasks ... --out ...   # call the model API
python code/metrics.py --domain ad --results ... --tasks ...
```

The runner reads API keys from the environment file named in `common.py`.
Analysis is deterministic given the released responses.

## Environment

Python 3.11, pandas, numpy, scikit-learn, lightgbm, transformers, torch,
httpx, matplotlib. Model access: DeepSeek chat-completions API, model
identifier `deepseek-flash` (DeepSeek-V4.1-Flash), non-thinking mode,
temperature 0, max 512 output tokens, JSON response format.
