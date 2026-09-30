"""Paired bootstrap confidence interval for panel minus unconditioned accuracy."""

import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from metrics import pairwise_ad  # noqa: E402


def ad_difference(rounds: int = 2000, seed: int = 11) -> dict:
    by_test: dict[str, list[list[int]]] = defaultdict(list)
    for rec in read_jsonl(RUNS / "ad_panel_flash.jsonl"):
        if rec.get("scores") is None:
            continue
        parts = rec["id"].split("::")
        if len(parts) == 3:
            by_test[parts[1]].append(rec["scores"])
    base = {r["id"]: r["scores"] for r in read_jsonl(RUNS / "ad_conf_AGG_flash.jsonl") if r.get("scores")}
    tasks = {t["id"]: t for t in read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")}

    per_test = []
    for test_id, lists in by_test.items():
        task = tasks.get(f"AD::{test_id}")
        if task is None:
            continue
        n_arms = task["meta"]["n_arms"]
        good = [s for s in lists if len(s) == n_arms]
        key = f"AD::{test_id}"
        if len(good) < 2 or key not in base or len(base[key]) != n_arms:
            continue
        mean = [round(sum(col) / len(good)) for col in zip(*good)]
        _, pp, pa = pairwise_ad(mean, task["meta"]["clicks"])
        _, bp, ba = pairwise_ad(base[key], task["meta"]["clicks"])
        if pp and bp:
            per_test.append((pa * pp, pp, ba * bp, bp))

    def acc(rows):
        p = sum(r[1] for r in rows)
        b = sum(r[3] for r in rows)
        return sum(r[0] for r in rows) / p, sum(r[2] for r in rows) / b

    panel, base_acc = acc(per_test)
    rng = random.Random(seed)
    diffs = []
    n = len(per_test)
    for _ in range(rounds):
        sample = [per_test[rng.randrange(n)] for _ in range(n)]
        p, b = acc(sample)
        diffs.append(p - b)
    diffs.sort()
    return {
        "tests": n,
        "panel": round(panel * 100, 2),
        "unconditioned": round(base_acc * 100, 2),
        "difference_pp": round((panel - base_acc) * 100, 2),
        "ci95_pp": [round(diffs[int(0.025 * len(diffs))] * 100, 2),
                    round(diffs[int(0.975 * len(diffs))] * 100, 2)],
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import json

    print(json.dumps(ad_difference(), indent=2))
