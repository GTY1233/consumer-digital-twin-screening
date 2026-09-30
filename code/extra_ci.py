"""Intervals for decision lift and for AGG minus IND, plus the random-choice
benchmark used in the text."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402


def pair_stats(s, c):
    n = len(s)
    iu = np.triu_indices(n, 1)
    ds, dc = s[iu[0]] - s[iu[1]], c[iu[0]] - c[iu[1]]
    v = dc != 0
    if not v.any():
        return 0.0, 0
    ds, dc = ds[v], dc[v]
    return float((ds * dc > 0).sum()) + 0.5 * float((ds == 0).sum()), int(v.sum())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    tasks = read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")
    agg = {r["id"]: r["scores"] for r in read_jsonl(RUNS / "ad_conf_AGG_flash.jsonl")
           if r.get("scores") is not None}
    ind = {r["id"]: r["scores"] for r in read_jsonl(RUNS / "ad_conf_IND_flash.jsonl")
           if r.get("scores") is not None}

    rows = []
    for t in tasks:
        tid, n = t["id"], t["meta"]["n_arms"]
        if tid not in agg or tid not in ind or len(agg[tid]) != n or len(ind[tid]) != n:
            continue
        clk = np.array(t["meta"]["clicks"], dtype=float)
        imp = np.array(t["meta"]["impressions"], dtype=float)
        ctr = clk / imp
        mean_ctr = clk.sum() / imp.sum()
        pick = int(np.argmax(np.array(agg[tid], dtype=float)))
        oracle = ctr.max()
        ma, pa = pair_stats(np.array(agg[tid], dtype=float), clk)
        mi, pi = pair_stats(np.array(ind[tid], dtype=float), clk)
        rand_pick = int(np.argmax(np.random.default_rng(len(tid)).random(n)))
        rows.append({
            "lift": ctr[pick] / mean_ctr - 1,
            "beats": float(ctr[pick] > mean_ctr),
            "rand_beats": float(ctr[rand_pick] > mean_ctr),
            "share": (ctr[pick] / mean_ctr - 1) / (oracle / mean_ctr - 1),
            "agg": (ma, pa), "ind": (mi, pi),
        })
    n = len(rows)

    def mean_lift(s):
        return sum(r["lift"] for r in s) / len(s)

    def beats(s, key):
        return sum(r[key] for r in s) / len(s)

    def acc(s, key):
        p = sum(r[key][1] for r in s)
        return sum(r[key][0] for r in s) / p if p else float("nan")

    rng = np.random.default_rng(303)
    lift_ci, diff_ci = [], []
    for _ in range(2000):
        s = [rows[i] for i in rng.integers(0, n, n)]
        lift_ci.append(mean_lift(s))
        diff_ci.append(acc(s, "agg") - acc(s, "ind"))
    lift_ci.sort()
    diff_ci.sort()
    out = {
        "tests": n,
        "decision_lift": round(mean_lift(rows) * 100, 2),
        "decision_lift_ci95": [round(lift_ci[int(0.025 * len(lift_ci))] * 100, 2),
                               round(lift_ci[int(0.975 * len(lift_ci))] * 100, 2)],
        "beats_mean_system": round(beats(rows, "beats") * 100, 2),
        "beats_mean_random": round(beats(rows, "rand_beats") * 100, 2),
        "agg_minus_ind": round((acc(rows, "agg") - acc(rows, "ind")) * 100, 2),
        "agg_minus_ind_ci95": [round(diff_ci[int(0.025 * len(diff_ci))] * 100, 2),
                               round(diff_ci[int(0.975 * len(diff_ci))] * 100, 2)],
    }
    (pathlib.Path(__file__).resolve().parent.parent / "extra_ci.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
