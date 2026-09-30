"""Decompose the panel penalty: compare PANEL against the condition whose
prompt wrapper it actually used (IND), and against the reported aggregate."""

import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from metrics import pairwise_ad  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    by_test: dict[str, list[list[int]]] = defaultdict(list)
    for rec in read_jsonl(RUNS / "ad_panel_flash.jsonl"):
        if rec.get("scores") is None:
            continue
        parts = rec["id"].split("::")
        if len(parts) == 3:
            by_test[parts[1]].append(rec["scores"])
    base = {
        f: {r["id"]: r["scores"] for r in read_jsonl(RUNS / f) if r.get("scores") is not None}
        for f in ("ad_conf_AGG_flash.jsonl", "ad_conf_IND_flash.jsonl")
    }
    tasks = {t["id"]: t for t in read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")}

    rows = []
    for test_id, lists in by_test.items():
        task = tasks.get(f"AD::{test_id}")
        if task is None:
            continue
        n_arms = task["meta"]["n_arms"]
        good = [s for s in lists if len(s) == n_arms]
        key = f"AD::{test_id}"
        if len(good) < 2:
            continue
        mean = [round(sum(c) / len(good)) for c in zip(*good)]
        _, pp, pa = pairwise_ad(mean, task["meta"]["clicks"])
        entry = {"panel": (pa, pp)}
        for name, store in (("agg", base["ad_conf_AGG_flash.jsonl"]),
                            ("ind", base["ad_conf_IND_flash.jsonl"])):
            if key in store and len(store[key]) == n_arms:
                _, bp, ba = pairwise_ad(store[key], task["meta"]["clicks"])
                if bp:
                    entry[name] = (ba, bp)
        if len(entry) == 3 and pp:
            rows.append(entry)

    def acc(rows, key):
        p = sum(r[key][1] for r in rows)
        return sum(r[key][0] * r[key][1] for r in rows) / p if p else float("nan")

    def diff(rows, key):
        rng = random.Random(5)
        out = []
        n = len(rows)
        for _ in range(2000):
            s = [rows[rng.randrange(n)] for _ in range(n)]
            out.append(acc(s, "panel") - acc(s, key))
        out.sort()
        return round(out[int(0.025 * len(out))] * 100, 2), round(out[int(0.975 * len(out))] * 100, 2)

    print(f"tests used: {len(rows)}")
    for key in ("agg", "ind"):
        print(f"  AGG={key} ref acc={acc(rows, key) * 100:.2f}%  panel={acc(rows, 'panel') * 100:.2f}%"
              f"  panel-ref={(acc(rows, 'panel') - acc(rows, key)) * 100:+.2f} pp  ci95={diff(rows, key)}")
    _, pp, _ = 0, 0, 0
    c = sum(r["agg"][1] for r in rows)
    a = sum(r["agg"][0] * r["agg"][1] for r in rows) / c
    c2 = sum(r["ind"][1] for r in rows)
    b = sum(r["ind"][0] * r["ind"][1] for r in rows) / c2
    print(f"  AGG - IND on the same subset = {(a - b) * 100:+.2f} pp")

    # top-1 chance level, E[1/k] vs 1/mean(k)
    ks = [t["meta"]["n_arms"] for t in tasks.values()]
    print(f"  E[1/k] = {sum(1.0 / k for k in ks) / len(ks) * 100:.2f}%   "
          f"1/mean(k) = {100 / (sum(ks) / len(ks)):.2f}%   n_tests={len(ks)}")


if __name__ == "__main__":
    main()
