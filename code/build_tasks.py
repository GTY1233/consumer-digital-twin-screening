"""Build prompt tasks for both domains.

Prompts are reproduced verbatim from the manuscript (Appendix A.1 / A.2 / A.3).
The user-message layout is fixed here and documented, because the manuscript
does not specify it.
"""

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATA, RUNS, ensure_dirs, seed_from, shuffled  # noqa: E402

AD_SYSTEM_AGG = (
    "You estimate how real online readers behave. You are shown several competing article "
    "headlines that were placed in front of readers of a general-interest social feed, all for "
    "the same article. For EACH headline, estimate the percentage of readers shown that headline "
    "who would tap it: an integer from 0 to 100. Base the estimate on how ordinary readers of this "
    "kind of feed actually behave, not on how good the writing is and not on what you personally "
    "would choose. Return exactly one JSON object with key scores: an array of integers, one per "
    "headline, in the order the headlines are given."
)

AD_SYSTEM_IND = (
    "You role-play one specific reader of a general-interest social feed. You are shown several "
    "competing article headlines that were placed in front of readers. For EACH one, report how "
    "likely that reader is to tap it: an integer from 0 (would never tap) to 100 (would certainly "
    "tap). Judge only from that reader's own point of view. Do not try to guess what other people "
    "did and do not reward good writing in the abstract. Return exactly one JSON object with key "
    "scores: an array of integers, one per headline, in the order the headlines are given."
)

NP_SYSTEM_AGG = (
    "You estimate how real people behave on a crowdfunding platform. You are shown a project "
    "pitch. Estimate the percentage of people who see this pitch who would end up pledging money "
    "to it: an integer from 0 to 100. Base the estimate on how ordinary backers of crowdfunding "
    "projects actually behave, not on how appealing you find the text. Return exactly one JSON "
    "object with key score."
)

NP_SYSTEM_IND = (
    "You role-play one specific backer on a crowdfunding platform. You are shown a project pitch. "
    "Report how likely that backer is to pledge money to it: an integer from 0 (would never pledge) "
    "to 100 (would certainly pledge). Judge only from that backer's own point of view. Do not try "
    "to guess what other people did. Return exactly one JSON object with key score."
)


def keep_tests(df: pd.DataFrame) -> pd.Index:
    grp = df.groupby("clickability_test_id", sort=False)
    n_arms = grp["headline"].size()
    keep = (
        (n_arms >= 2)
        & (n_arms <= 6)
        & (grp["eyecatcher_id"].nunique() == 1)
        & (grp["headline"].nunique() == n_arms)
        & (grp["impressions"].min() >= 1000)
        & (grp["clicks"].sum() >= 10)
    )
    return keep[keep].index


def build_ad(split: str) -> tuple[list[dict], list[dict]]:
    path = DATA / f"upworthy-{split}.csv"
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df["headline"] = df["headline"].astype(str).str.strip()
    df["eyecatcher_id"] = df["eyecatcher_id"].astype(str)
    df["excerpt"] = df["excerpt"].fillna("").astype(str).str.strip()
    df = df[df["clickability_test_id"].isin(keep_tests(df))].copy()

    tasks, labels = [], []
    for test_id, group in df.groupby("clickability_test_id", sort=False):
        rows = group.to_dict("records")
        order = shuffled(rows, seed_from(str(test_id)))
        excerpt = order[0]["excerpt"]
        excerpt = excerpt[:1200]
        lines = [f"Article excerpt: {excerpt}" if excerpt else "Article excerpt: (none provided)", "", "Headlines:"]
        for i, row in enumerate(order, 1):
            lines.append(f"{i}. {row['headline']}")
        user = "\n".join(lines)
        tasks.append(
            {
                "id": f"AD::{test_id}",
                "system": AD_SYSTEM_AGG,
                "user": user,
                "meta": {
                    "domain": "advertising",
                    "n_arms": len(order),
                    "headlines": [r["headline"] for r in order],
                    "impressions": [int(r["impressions"]) for r in order],
                    "clicks": [int(r["clicks"]) for r in order],
                },
            }
        )
        labels.append(
            {
                "test_id": test_id,
                "headlines": [r["headline"] for r in order],
                "impressions": [int(r["impressions"]) for r in order],
                "clicks": [int(r["clicks"]) for r in order],
            }
        )
    return tasks, labels


def build_np(split: str) -> tuple[list[dict], list[dict]]:
    df = pd.read_parquet(DATA / f"kickstarter-{split}.parquet")
    tasks, labels = [], []
    for idx, row in df.iterrows():
        name = str(row["name"]).strip()
        desc = str(row["desc"]).strip()
        user = f"Project name: {name}\n\nProject pitch: {desc}"
        pid = f"NP::{split}::{idx}"
        tasks.append(
            {
                "id": pid,
                "system": NP_SYSTEM_AGG,
                "user": user,
                "meta": {"domain": "crowdfunding", "funded": int(row["final_status"])},
            }
        )
        labels.append(
            {
                "id": pid,
                "funded": int(row["final_status"]),
                "goal": float(row["goal"]) if str(row["goal"]) not in ("", "nan") else float("nan"),
            }
        )
    return tasks, labels


def write(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(rows)} rows")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="confirmatory", choices=["confirmatory", "exploratory"])
    parser.add_argument("--domain", default="all", choices=["all", "ad", "np"])
    args = parser.parse_args()
    ensure_dirs()
    if args.domain in ("all", "ad"):
        tasks, labels = build_ad(args.split)
        write(RUNS / f"ad_{args.split}_tasks.jsonl", tasks)
        write(RUNS / f"ad_{args.split}_labels.jsonl", labels)
    if args.domain in ("all", "np"):
        tasks, labels = build_np("test")
        write(RUNS / "np_test_tasks.jsonl", tasks)
        write(RUNS / "np_test_labels.jsonl", labels)
        tasks, labels = build_np("validation")
        write(RUNS / "np_validation_tasks.jsonl", tasks)
        write(RUNS / "np_validation_labels.jsonl", labels)


if __name__ == "__main__":
    main()
