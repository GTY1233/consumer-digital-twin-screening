"""Non-LLM text baselines for the advertising domain (reviewer point A6).

Protocol
--------
Fit on the archive's exploratory split, evaluate once on the confirmatory
split. The target is the within-test relative log click rate, because the
reported metric is a within-test comparison.

Three comparators are reported:
  1. trivial heuristics (shorter headline wins)
  2. hand-crafted surface features + LightGBM
  3. surface features + word/char TF-IDF + LightGBM
"""

import argparse
import pathlib
import re
import sys
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATA  # noqa: E402
from build_tasks import keep_tests  # noqa: E402

SUPERLATIVES = (
    "best worst amazing incredible awesome shocking unbelievable perfect greatest "
    "ultimate devastating brilliant stunning"
).split()


def load_ad(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"upworthy-{split}.csv", engine="python", on_bad_lines="skip")
    df["headline"] = df["headline"].astype(str).str.strip()
    df["eyecatcher_id"] = df["eyecatcher_id"].astype(str)
    df["ctr"] = df["clicks"] / df["impressions"]
    return df[df["clickability_test_id"].isin(keep_tests(df))].copy()


def hand_features(texts: pd.Series) -> pd.DataFrame:
    lower = texts.str.lower()
    return pd.DataFrame(
        {
            "chars": texts.str.len(),
            "words": texts.str.split().str.len(),
            "has_number": texts.str.contains(r"\d").astype(int),
            "has_question": texts.str.contains(r"\?").astype(int),
            "you": lower.str.contains(r"\byou\b").astype(int),
            "starts_number": texts.str.match(r"^\s*\d").astype(int),
            "superlative": lower.apply(
                lambda t: sum(word in t.split() for word in SUPERLATIVES)
            ),
            "capitalised": texts.apply(lambda t: sum(c.isupper() for c in t)),
            "exclaim": texts.str.count("!"),
        }
    )


def pairwise(scores, clicks) -> tuple[float, int]:
    concordant = 0.0
    n = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if clicks[i] == clicks[j]:
                continue
            n += 1
            if scores[i] > scores[j]:
                concordant += 1
            elif scores[i] == scores[j]:
                concordant += 0.5
    return concordant, n


def score_grouped(pred: pd.Series, df: pd.DataFrame) -> tuple[float, int]:
    concordant = 0.0
    pairs = 0
    for _, group in df.groupby("clickability_test_id", sort=False):
        c, n = pairwise(list(pred.loc[group.index]), list(group["ctr"]))
        concordant += c
        pairs += n
    return concordant, pairs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="exploratory")
    parser.add_argument("--test", default="confirmatory")
    args = parser.parse_args()

    train = load_ad(args.train)
    test = load_ad(args.test)
    log_ctr = np.log(train["ctr"].clip(lower=1e-5))
    train = train.assign(
        y=log_ctr - log_ctr.groupby(train["clickability_test_id"]).transform("mean")
    )

    # 1. trivial heuristics
    for name, col in (("shorter_wins", -train["headline"].str.len()),):
        pred = pd.Series(test["headline"].str.len().mul(-1).values, index=test.index)
        c, n = score_grouped(pred, test)
        print(f"{name:34s} {c / n:.4f}  (pairs {n:,})")

    # 2. hand features
    xh_tr = hand_features(train["headline"]).values
    xh_te = hand_features(test["headline"]).values
    hand = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100, verbose=-1
    ).fit(xh_tr, train["y"], sample_weight=np.sqrt(train["impressions"]))
    pred = pd.Series(hand.predict(xh_te), index=test.index)
    c, n = score_grouped(pred, test)
    print(f"{'hand_features_lightgbm':34s} {c / n:.4f}  (pairs {n:,})")

    # 3. hand + TF-IDF
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True)
    from scipy.sparse import hstack

    x_tr = hstack([word.fit_transform(train["headline"]), char.fit_transform(train["headline"]), xh_tr])
    x_te = hstack([word.transform(test["headline"]), char.transform(test["headline"]), xh_te])
    full = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=200, verbose=-1
    ).fit(x_tr, train["y"], sample_weight=np.sqrt(train["impressions"]))
    pred = pd.Series(full.predict(x_te), index=test.index)
    c, n = score_grouped(pred, test)
    print(f"{'tfidf_hand_lightgbm':34s} {c / n:.4f}  (pairs {n:,})")


if __name__ == "__main__":
    main()
