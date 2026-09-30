"""Aggregate the persona panel and compare it with the unconditioned condition
on the identical subset."""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from metrics import pairwise_ad  # noqa: E402


def panel_ad(panel_path: pathlib.Path, base_path: pathlib.Path, tasks_path: pathlib.Path) -> dict:
    by_test: dict[str, list[list[int]]] = defaultdict(list)
    for rec in read_jsonl(panel_path):
        if rec.get("scores") is None:
            continue
        parts = rec["id"].split("::")
        if len(parts) != 3:
            continue
        test_id = parts[1]
        by_test[test_id].append(rec["scores"])

    base = {r["id"]: r["scores"] for r in read_jsonl(base_path) if r.get("scores") is not None}
    tasks = {t["id"]: t for t in read_jsonl(tasks_path)}

    panel_c = panel_n = base_c = base_n = 0.0
    persona_spread: list[float] = []
    gap: list[float] = []
    for test_id, lists in by_test.items():
        task = tasks.get(f"AD::{test_id}")
        if task is None:
            continue
        n_arms = task["meta"]["n_arms"]
        good = [s for s in lists if len(s) == n_arms]
        if len(good) < 2:
            continue
        mean = [sum(col) / len(good) for col in zip(*good)]
        _, pairs, acc = pairwise_ad([round(v) for v in mean], task["meta"]["clicks"])
        if pairs:
            panel_c += acc * pairs
            panel_n += pairs
        base_key = f"AD::{test_id}"
        if base_key in base and len(base[base_key]) == n_arms:
            _, bp, bacc = pairwise_ad(base[base_key], task["meta"]["clicks"])
            if bp:
                base_c += bacc * bp
                base_n += bp
            gap.append(sum(abs(m - b) for m, b in zip(mean, base[base_key])) / n_arms)
        for arm in range(n_arms):
            values = [s[arm] for s in good]
            m = sum(values) / len(values)
            persona_spread.append((sum((v - m) ** 2 for v in values) / len(values)) ** 0.5)
    return {
        "panel_tests": len(by_test),
        "panel_pairs": int(panel_n),
        "panel_pairwise_accuracy": panel_c / panel_n if panel_n else float("nan"),
        "unconditioned_pairs": int(base_n),
        "unconditioned_pairwise_accuracy": base_c / base_n if base_n else float("nan"),
        "mean_persona_sd": sum(persona_spread) / len(persona_spread) if persona_spread else float("nan"),
        "mean_abs_gap_panel_vs_unconditioned": sum(gap) / len(gap) if gap else float("nan"),
    }


def panel_np(panel_path: pathlib.Path, base_path: pathlib.Path, labels_path: pathlib.Path) -> dict:
    by_id: dict[str, list[float]] = defaultdict(list)
    for rec in read_jsonl(panel_path):
        if rec.get("scores"):
            pid = rec["id"].split("::")[0] + "::" + rec["id"].split("::")[1] + "::" + rec["id"].split("::")[2]
            by_id[pid].append(float(rec["scores"][0]))
    panel = {k: sum(v) / len(v) for k, v in by_id.items()}
    base = {r["id"]: float(r["scores"][0]) for r in read_jsonl(base_path) if r.get("scores")}
    labels = {r["id"]: r for r in read_jsonl(labels_path)}

    def auc(scores: dict[str, float]) -> tuple[float, int]:
        pos = [s for k, s in scores.items() if labels.get(k, {}).get("funded") == 1]
        neg = [s for k, s in scores.items() if labels.get(k, {}).get("funded") == 0]
        c = 0.0
        n = 0
        for p in pos:
            for q in neg:
                n += 1
                c += 1 if p > q else (0.5 if p == q else 0)
        return c, n

    pc, pn = auc(panel)
    bc, bn = auc({k: v for k, v in base.items() if k in panel})
    return {
        "panel_items": len(panel),
        "panel_auc": pc / pn if pn else float("nan"),
        "panel_pairs": pn,
        "unconditioned_auc_same_items": bc / bn if bn else float("nan"),
        "unconditioned_pairs": bn,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["ad", "np"])
    args = parser.parse_args()
    if args.domain == "ad":
        out = panel_ad(
            RUNS / "ad_panel_flash.jsonl",
            RUNS / "ad_conf_AGG_flash.jsonl",
            RUNS / "ad_confirmatory_tasks.jsonl",
        )
    else:
        out = panel_np(
            RUNS / "np_panel_flash.jsonl",
            RUNS / "np_test_AGG_flash.jsonl",
            RUNS / "np_test_labels.jsonl",
        )
    print(json.dumps(out, indent=2, ensure_ascii=False))
