"""Reproduce the paper's inclusion rules on the Upworthy archive.

Rules (manuscript Section 4.2):
  1. every arm in the test carries the same eyecatcher_id
  2. the test has between two and six arms
  3. all headline texts within the test are distinct
  4. every arm received at least 1,000 impressions
  5. the test accumulated at least ten clicks in total

Targets reported in the paper:
  confirmatory: 9,924 tests / 44,272 packages / 159.5M impressions / 2.12M clicks
  exploratory (dev): 2,043 tests
"""

import os
import sys

import pandas as pd

DATA = os.path.join(os.environ["USERPROFILE"], ".paper-work", "data")


def load(path: str) -> pd.DataFrame:
    # The archive CSVs contain unescaped commas/newlines inside text fields,
    # so the fast C parser fails; the python engine handles them.
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df["headline"] = df["headline"].astype(str)
    df["eyecatcher_id"] = df["eyecatcher_id"].astype(str)
    return df


def summarise(name: str, df: pd.DataFrame) -> None:
    total_tests = df["clickability_test_id"].nunique()
    print(f"--- {name} ---")
    print(f"packages            : {len(df):,}")
    print(f"tests (raw)         : {total_tests:,}")

    grp = df.groupby("clickability_test_id", sort=False)

    n_arms = grp["headline"].size()
    same_image = grp["eyecatcher_id"].nunique() == 1
    distinct_heads = grp["headline"].nunique() == n_arms
    min_impressions = grp["impressions"].min() >= 1000
    total_clicks = grp["clicks"].sum() >= 10

    keep = (
        (n_arms >= 2)
        & (n_arms <= 6)
        & same_image
        & distinct_heads
        & min_impressions
        & total_clicks
    )
    kept_tests = keep[keep].index
    out = df[df["clickability_test_id"].isin(kept_tests)]

    mean_arms = kept_tests.size and len(out) / kept_tests.size
    print(f"tests (kept)        : {len(kept_tests):,}")
    print(f"packages (kept)     : {len(out):,}")
    print(f"mean arms per test  : {mean_arms:.2f}")
    print(f"impressions         : {out['impressions'].sum():,}")
    print(f"clicks              : {out['clicks'].sum():,}")
    print(f"ctr                 : {out['clicks'].sum() / out['impressions'].sum():.5f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for name, fname in (
        ("confirmatory", "upworthy-confirmatory.csv"),
        ("exploratory", "upworthy-exploratory.csv"),
    ):
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        summarise(name, load(path))


if __name__ == "__main__":
    main()
