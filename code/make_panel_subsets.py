"""Deterministic subsets used for the persona-panel conditions."""

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402


def subset(src: str, dst: str, n: int, seed: int) -> None:
    rows = read_jsonl(RUNS / src)
    rng = random.Random(seed)
    sample = rng.sample(rows, n)
    sample.sort(key=lambda r: r["id"])
    out = RUNS / dst
    with out.open("w", encoding="utf-8") as fh:
        import json

        for row in sample:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{dst}: {len(sample)} rows")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    subset("ad_confirmatory_tasks.jsonl", "ad_conf_panel_tasks.jsonl", 1500, 20260930)
    subset("np_test_tasks.jsonl", "np_test_panel_tasks.jsonl", 1000, 20260930)
