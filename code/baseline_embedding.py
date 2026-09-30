"""Frozen sentence-embedding baseline (all-MiniLM-L6-v2 + ridge).

This is the strongest cheap non-LLM comparator: it captures semantics rather
than surface form.  Fit on the exploratory split, evaluate on confirmatory.
"""

import os
import pathlib
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from baseline_text import load_ad, score_grouped  # noqa: E402

MODEL_DIR = pathlib.Path(os.environ["USERPROFILE"]) / ".paper-work" / "models" / "all-MiniLM-L6-v2"


def embed(texts: list[str], batch: int = 256) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            out.append(pooled.cpu().numpy())
            if start % (batch * 20) == 0:
                print(f"  embedded {start}/{len(texts)}", flush=True)
    return np.vstack(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    train = load_ad("exploratory")
    test = load_ad("confirmatory")
    log_ctr = np.log(train["ctr"].clip(lower=1e-5))
    train = train.assign(y=log_ctr - log_ctr.groupby(train["clickability_test_id"]).transform("mean"))

    print(f"embedding {len(train):,} train headlines", flush=True)
    x_train = embed(train["headline"].tolist())
    print(f"embedding {len(test):,} test headlines", flush=True)
    x_test = embed(test["headline"].tolist())

    for alpha in (1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(x_train, train["y"])
        pred = pd.Series(model.predict(x_test), index=test.index)
        c, n = score_grouped(pred, test)
        print(f"minilm_ridge_alpha{alpha:<6} {c / n:.4f}  (pairs {n:,})", flush=True)


if __name__ == "__main__":
    main()
