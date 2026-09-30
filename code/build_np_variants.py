"""Build Kickstarter prompt variants so the input format can be chosen on the
validation split, then frozen and reported once on the test split."""

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATA, RUNS  # noqa: E402
from prompts import NP_AGG  # noqa: E402


def render(row: pd.Series, variant: str) -> str:
    name = str(row["name"]).strip()
    desc = str(row["desc"]).strip()
    if variant == "A":
        return f"Project name: {name}\n\nProject pitch: {desc}"
    if variant == "B":
        return f"Project name: {name}\n\nProject pitch: {desc}\n\nCountry: {row['country']}"
    if variant == "C":
        return (
            f"Project name: {name}\n\nProject pitch: {desc}\n\n"
            f"Country: {row['country']}\n\nFunding goal: {row['goal']} {row['currency']}"
        )
    raise ValueError(variant)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--variant", default="B")
    args = parser.parse_args()
    df = pd.read_parquet(DATA / f"kickstarter-{args.split}.parquet")
    out = RUNS / f"np_{args.split}_{args.variant}_tasks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for idx, row in df.iterrows():
            fh.write(
                json.dumps(
                    {
                        "id": f"NP::{args.split}::{idx}",
                        "system": NP_AGG,
                        "user": render(row, args.variant),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{out.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
