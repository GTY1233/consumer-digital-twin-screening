"""Two executions of the aggregate-wrapped persona panel."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from panel_report_fast import ad_panel, load_store_factory, pair_stats  # noqa: E402


def build_rows(panel_file: str, stores: dict) -> list[dict]:
    tasks = {t["id"]: t for t in read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")}
    rows = []
    for test_id, lists in ad_panel(RUNS / panel_file).items():
        task = tasks.get(f"AD::{test_id}")
        if task is None:
            continue
        n = task["meta"]["n_arms"]
        good = [s for s in lists if len(s) == n]
        key = f"AD::{test_id}"
        if len(good) < 2 or key not in stores["agg"] or key not in stores["ind"]:
            continue
        clicks = np.array(task["meta"]["clicks"], dtype=float)
        variants = {
            "panel": np.array([round(sum(c) / len(good)) for c in zip(*good)], dtype=float),
            "agg": np.array(stores["agg"][key], dtype=float),
            "ind": np.array(stores["ind"][key], dtype=float),
        }
        stats = {k: pair_stats(v, clicks) for k, v in variants.items()}
        if all(s[1] for s in stats.values()):
            rows.append(stats)
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    loads = load_store_factory()
    stores = {"agg": loads("ad_conf_AGG_flash.jsonl"), "ind": loads("ad_conf_IND_flash.jsonl")}
    out = {}
    for label, fname in (("run1", "ad_panel_agg_flash.jsonl"), ("run2", "ad_panel_agg2_flash.jsonl")):
        rows = build_rows(fname, stores)
        n = len(rows)

        def acc(sel, key):
            p = sum(r[key][1] for r in sel)
            return sum(r[key][0] for r in sel) / p if p else float("nan")

        rng = np.random.default_rng(55)
        diffs = []
        for _ in range(1000):
            s = [rows[i] for i in rng.integers(0, n, n)]
            diffs.append(acc(s, "panel") - acc(s, "agg"))
        diffs.sort()
        out[f"ad_{label}"] = {
            "tests": n,
            "panel": round(acc(rows, "panel") * 100, 2),
            "agg": round(acc(rows, "agg") * 100, 2),
            "panel_minus_agg": round((acc(rows, "panel") - acc(rows, "agg")) * 100, 2),
            "ci95": [round(diffs[int(0.025 * len(diffs))] * 100, 2),
                     round(diffs[int(0.975 * len(diffs))] * 100, 2)],
        }
    for label, fname in (("run1", "np_panel_agg_flash.jsonl"), ("run2", "np_panel_agg2_flash.jsonl")):
        labels = {r["id"]: r for r in read_jsonl(RUNS / "np_test_labels.jsonl")}
        agg = loads("np_test_AGG_flash.jsonl")
        from collections import defaultdict

        by_id: dict[str, list[float]] = defaultdict(list)
        for rec in read_jsonl(RUNS / fname):
            if rec.get("scores"):
                by_id["::".join(rec["id"].split("::")[:3])].append(float(rec["scores"][0]))
        panel = {k: sum(v) / len(v) for k, v in by_id.items() if len(v) == 8}
        ids = [i for i in sorted(panel) if i in agg]
        f = np.array([labels[i]["funded"] for i in ids], dtype=float)
        P = np.array([panel[i] for i in ids])
        A = np.array([agg[i] for i in ids])

        def auc(s):
            d = s[f == 1][:, None] - s[f == 0][None, :]
            return (float((d > 0).sum()) + 0.5 * float((d == 0).sum())) / d.size

        rng = np.random.default_rng(59)
        diffs = []
        for _ in range(400):
            idx = rng.integers(0, len(ids), len(ids))
            ff, pp, aa = f[idx], P[idx], A[idx]
            if (ff == 1).sum() and (ff == 0).sum():
                d1 = pp[ff == 1][:, None] - pp[ff == 0][None, :]
                d2 = aa[ff == 1][:, None] - aa[ff == 0][None, :]
                diffs.append(((d1 > 0).sum() + 0.5 * (d1 == 0).sum()) / d1.size
                             - ((d2 > 0).sum() + 0.5 * (d2 == 0).sum()) / d2.size)
        diffs.sort()
        out[f"np_{label}"] = {
            "items": len(ids),
            "panel": round(auc(P) * 100, 2),
            "agg": round(auc(A) * 100, 2),
            "panel_minus_agg": round((auc(P) - auc(A)) * 100, 2),
            "ci95": [round(diffs[int(0.025 * len(diffs))] * 100, 2),
                     round(diffs[int(0.975 * len(diffs))] * 100, 2)],
        }
    (pathlib.Path(__file__).resolve().parent.parent / "panel_stability.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
