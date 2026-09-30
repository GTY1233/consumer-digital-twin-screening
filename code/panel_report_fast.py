"""Vectorised panel comparisons with the prompt wrapper held constant."""

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402


def pair_stats(scores: np.ndarray, clicks: np.ndarray) -> tuple[float, int]:
    """Concordant mass and pair count over upper-triangle pairs."""
    n = len(scores)
    if n < 2:
        return 0.0, 0
    iu = np.triu_indices(n, 1)
    ds, dc = scores[iu[0]] - scores[iu[1]], clicks[iu[0]] - clicks[iu[1]]
    valid = dc != 0
    if not valid.any():
        return 0.0, 0
    ds, dc = ds[valid], dc[valid]
    return float((ds * dc > 0).sum()) + 0.5 * float((ds == 0).sum()), int(valid.sum())


def ad_panel(path: pathlib.Path) -> dict[str, list[list[int]]]:
    by_test: dict[str, list[list[int]]] = defaultdict(list)
    for rec in read_jsonl(path):
        if rec.get("scores"):
            parts = rec["id"].split("::")
            if len(parts) == 3:
                by_test[parts[1]].append(rec["scores"])
    return by_test


def store(name: str) -> dict[str, list[int]]:
    return {r["id"]: r["scores"] for r in read_jsonl(RUNS / name) if r.get("scores") is not None}


def load_store_factory():
    return store


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    tasks = {t["id"]: t for t in read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")}
    agg, ind = store("ad_conf_AGG_flash.jsonl"), store("ad_conf_IND_flash.jsonl")
    out: dict = {}

    for label, fname in (("panel_roleplay", "ad_panel_flash.jsonl"),
                         ("panel_aggregate", "ad_panel_agg_flash.jsonl")):
        panels = ad_panel(RUNS / fname)
        recs = []
        for test_id, lists in panels.items():
            task = tasks.get(f"AD::{test_id}")
            if task is None:
                continue
            n = task["meta"]["n_arms"]
            good = [s for s in lists if len(s) == n]
            key = f"AD::{test_id}"
            if len(good) < 2 or key not in agg or key not in ind:
                continue
            clicks = np.array(task["meta"]["clicks"], dtype=float)
            variants = {
                "panel": np.array([round(sum(c) / len(good)) for c in zip(*good)], dtype=float),
                "agg": np.array(agg[key], dtype=float),
                "ind": np.array(ind[key], dtype=float),
            }
            stats = {k: pair_stats(v, clicks) for k, v in variants.items()}
            if all(s[1] for s in stats.values()):
                recs.append(stats)
        n = len(recs)

        def acc(rows, key):
            p = sum(r[key][1] for r in rows)
            return sum(r[key][0] for r in rows) / p if p else float("nan")

        rng = np.random.default_rng(31)
        block = {"tests": n}
        for key in ("agg", "ind", "panel"):
            block[f"{key}_accuracy"] = round(acc(recs, key) * 100, 2)
        for key in ("agg", "ind"):
            diffs = []
            for _ in range(1000):
                s = [recs[i] for i in rng.integers(0, n, n)]
                diffs.append(acc(s, "panel") - acc(s, key))
            diffs.sort()
            block[f"panel_minus_{key}"] = round(block["panel_accuracy"] - block[f"{key}_accuracy"], 2)
            block[f"panel_minus_{key}_ci95"] = [round(diffs[int(0.025 * len(diffs))] * 100, 2),
                                                round(diffs[int(0.975 * len(diffs))] * 100, 2)]
        block["agg_minus_ind"] = round(block["agg_accuracy"] - block["ind_accuracy"], 2)
        out[label] = block

    labels = {r["id"]: r for r in read_jsonl(RUNS / "np_test_labels.jsonl")}
    np_agg, np_ind = store("np_test_AGG_flash.jsonl"), store("np_test_IND_flash.jsonl")
    for label, fname in (("panel_roleplay", "np_panel_flash.jsonl"),
                         ("panel_aggregate", "np_panel_agg_flash.jsonl")):
        by_id: dict[str, list[float]] = defaultdict(list)
        for rec in read_jsonl(RUNS / fname):
            if rec.get("scores"):
                by_id["::".join(rec["id"].split("::")[:3])].append(float(rec["scores"][0]))
        panel = {k: sum(v) / len(v) for k, v in by_id.items() if len(v) == 8}
        ids = [i for i in sorted(panel) if i in np_agg and i in np_ind]
        funded = np.array([labels[i]["funded"] for i in ids], dtype=float)
        P = np.array([panel[i] for i in ids])
        A = np.array([np_agg[i] for i in ids])
        I = np.array([np_ind[i] for i in ids])

        def auc(s):
            d = s[funded == 1][:, None] - s[funded == 0][None, :]
            return float((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size

        block = {"items": len(ids), "panel_auc": round(auc(P) * 100, 2),
                 "agg_auc": round(auc(A) * 100, 2), "ind_auc": round(auc(I) * 100, 2)}
        rng = np.random.default_rng(37)
        for key, arr in (("agg", A), ("ind", I)):
            diffs = []
            for _ in range(500):
                idx = rng.integers(0, len(ids), len(ids))
                f, p, q = funded[idx], P[idx], arr[idx]
                if (f == 1).sum() and (f == 0).sum():
                    dp = p[f == 1][:, None] - p[f == 0][None, :]
                    dq = q[f == 1][:, None] - q[f == 0][None, :]
                    diffs.append(
                        ((dp > 0).sum() + 0.5 * (dp == 0).sum()) / dp.size
                        - ((dq > 0).sum() + 0.5 * (dq == 0).sum()) / dq.size
                    )
            diffs.sort()
            block[f"panel_minus_{key}"] = round(block["panel_auc"] - block[f"{key}_auc"], 2)
            block[f"panel_minus_{key}_ci95"] = [round(diffs[int(0.025 * len(diffs))] * 100, 2),
                                                round(diffs[int(0.975 * len(diffs))] * 100, 2)]
        out[f"np_{label}"] = block

    (pathlib.Path(__file__).resolve().parent.parent / "panel_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
