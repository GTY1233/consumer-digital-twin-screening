"""Compute the manuscript's evaluation metrics from a results file."""

import argparse
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402


def pairwise_ad(scores: list[int], clicks: list[int]) -> tuple[int, int, float]:
    """Return (concordant_mass, n_pairs, accuracy) with ties excluded."""
    concordant = 0.0
    n_pairs = 0
    total = 0.0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if clicks[i] == clicks[j] or scores[i] == scores[j]:
                continue
            n_pairs += 1
            total += 1
            if (scores[i] - scores[j]) * (clicks[i] - clicks[j]) > 0:
                concordant += 1
    acc = concordant / total if total else float("nan")
    return int(concordant), n_pairs, acc


def pairwise_ad_half(scores: list[int], clicks: list[int]) -> tuple[float, int]:
    concordant = 0.0
    total = 0.0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            total += 1
            if clicks[i] == clicks[j] or scores[i] == scores[j]:
                concordant += 0.5
            elif (scores[i] - scores[j]) * (clicks[i] - clicks[j]) > 0:
                concordant += 1
    return concordant, int(total)


def evaluate_ad(results_path: pathlib.Path, tasks_path: pathlib.Path) -> dict:
    results = {r["id"]: r for r in read_jsonl(results_path) if r.get("scores") is not None}
    tasks = read_jsonl(tasks_path)
    per_test = []
    for task in tasks:
        rec = results.get(task["id"])
        if rec is None:
            continue
        scores = rec["scores"]
        meta = task["meta"]
        if len(scores) != meta["n_arms"]:
            continue
        _, pairs, acc = pairwise_ad(scores, meta["clicks"])
        imp, clk = meta["impressions"], meta["clicks"]
        ctrs = [c / i for c, i in zip(clk, imp)]
        best = max(range(len(scores)), key=lambda k: (scores[k], -k))
        mean_ctr = sum(clk) / sum(imp)
        oracle = max(ctrs)
        per_test.append(
            {
                "id": task["id"],
                "acc": acc,
                "pairs": pairs,
                "top1": int(ctrs[best] == oracle and ctrs.count(oracle) == 1),
                "top1_tie": int(ctrs[best] == oracle),
                "lift": ctrs[best] / mean_ctr - 1,
                "oracle_lift": oracle / mean_ctr - 1,
                "beats_mean": int(ctrs[best] > mean_ctr),
                "shortfall_sys": oracle and (oracle - ctrs[best]) / oracle,
                "shortfall_rand": None,
                "n_arms": meta["n_arms"],
            }
        )
    return _summarise_ad(per_test)


def _summarise_ad(per_test: list[dict]) -> dict:
    valid = [t for t in per_test if t["pairs"] > 0]
    total_pairs = sum(t["pairs"] for t in valid)
    acc = sum(t["acc"] * t["pairs"] for t in valid) / total_pairs
    return {
        "n_tests": len(per_test),
        "n_tests_with_pairs": len(valid),
        "pairs": total_pairs,
        "pairwise_accuracy": acc,
        "top1": sum(t["top1_tie"] for t in per_test) / len(per_test),
        "decision_lift": sum(t["lift"] for t in per_test) / len(per_test),
        "oracle_lift": sum(t["oracle_lift"] for t in per_test) / len(per_test),
        "share_of_oracle": (
            sum(t["lift"] for t in per_test) / sum(t["oracle_lift"] for t in per_test)
            if sum(t["oracle_lift"] for t in per_test)
            else float("nan")
        ),
        "beats_mean": sum(t["beats_mean"] for t in per_test) / len(per_test),
        "_per_test": per_test,
    }


def bootstrap_ci(per_test: list[dict], rounds: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(per_test)
    stats = []
    for _ in range(rounds):
        sample = [per_test[rng.randrange(n)] for _ in range(n)]
        pairs = sum(t["pairs"] for t in sample)
        if pairs:
            stats.append(sum(t["acc"] * t["pairs"] for t in sample) / pairs)
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def evaluate_np(results_path: pathlib.Path, labels_path: pathlib.Path) -> dict:
    results = {r["id"]: r["scores"][0] for r in read_jsonl(results_path) if r.get("scores")}
    labels = read_jsonl(labels_path)
    funded, unfunded = [], []
    same_goal: dict[float, list[list[float]]] = {}
    for row in labels:
        score = results.get(row["id"])
        if score is None:
            continue
        item = (float(score), row["goal"])
        (funded if row["funded"] == 1 else unfunded).append(item)
        if not math.isnan(item[1]):
            same_goal.setdefault(item[1], [[], []])[row["funded"]].append(float(score))

    concordant = 0.0
    total = 0.0
    for s_f, _ in funded:
        for s_u, _ in unfunded:
            total += 1
            if s_f > s_u:
                concordant += 1
            elif s_f == s_u:
                concordant += 0.5
    auc = concordant / total if total else float("nan")

    within_concordant = 0.0
    within_total = 0.0
    for _, (neg, pos) in same_goal.items():
        for s_f in pos:
            for s_u in neg:
                within_total += 1
                if s_f > s_u:
                    within_concordant += 1
                elif s_f == s_u:
                    within_concordant += 0.5

    goal_concordant = 0.0
    goal_total = 0.0
    for _, g_f in funded:
        if math.isnan(g_f):
            continue
        for _, g_u in unfunded:
            if math.isnan(g_u):
                continue
            goal_total += 1
            if g_f < g_u:
                goal_concordant += 1
            elif g_f == g_u:
                goal_concordant += 0.5

    return {
        "n_scored": len(funded) + len(unfunded),
        "n_funded": len(funded),
        "n_unfunded": len(unfunded),
        "pairwise_accuracy_auc": auc,
        "pairs": int(total),
        "goal_baseline": goal_concordant / goal_total if goal_total else float("nan"),
        "goal_pairs": int(goal_total),
        "same_goal_accuracy": within_concordant / within_total if within_total else float("nan"),
        "same_goal_pairs": int(within_total),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--labels", required=False)
    parser.add_argument("--domain", required=True, choices=["ad", "np"])
    parser.add_argument("--tasks")
    args = parser.parse_args()

    results = pathlib.Path(args.results)
    labels = pathlib.Path(args.labels) if args.labels else None
    if args.domain == "ad":
        out = evaluate_ad(results, pathlib.Path(args.tasks))
        lo, hi = bootstrap_ci([t for t in out["_per_test"] if t["pairs"] > 0])
        out["ci95"] = [lo, hi]
        out.pop("_per_test")
    else:
        out = evaluate_np(results, labels)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
