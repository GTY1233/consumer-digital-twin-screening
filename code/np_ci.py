"""Project-level bootstrap intervals for the crowdfunding domain (vectorised)."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402


def store(name: str) -> dict[str, float]:
    return {r["id"]: float(r["scores"][0]) for r in read_jsonl(RUNS / name) if r.get("scores")}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    d = pos[:, None] - neg[None, :]
    return float((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    labels = read_jsonl(RUNS / "np_test_labels.jsonl")
    scores = {n: store(f"np_test_{n}_flash.jsonl") for n in ("AGG", "IND")}
    rows = [r for r in labels if r["id"] in scores["AGG"] and r["id"] in scores["IND"]]
    ids = [r["id"] for r in rows]
    funded = np.array([r["funded"] for r in rows], dtype=float)
    goal = np.array([r["goal"] if r["goal"] == r["goal"] else np.nan for r in rows], dtype=float)

    out = {}
    for name in ("AGG", "IND"):
        s = np.array([scores[name][i] for i in ids])
        base_auc = auc(s[funded == 1], s[funded == 0])

        def same_goal(v: np.ndarray, g: np.ndarray, f: np.ndarray) -> tuple[float, int]:
            mass = 0.0
            total = 0
            for gv in np.unique(g[~np.isnan(g)]):
                m = g == gv
                p, n = v[m & (f == 1)], v[m & (f == 0)]
                if p.size and n.size:
                    d = p[:, None] - n[None, :]
                    mass += float((d > 0).sum()) + 0.5 * float((d == 0).sum())
                    total += d.size
            return (mass / total if total else float("nan")), total

        base_same, n_same = same_goal(s, goal, funded)

        rng = np.random.default_rng(41)
        aucs, sames = [], []
        for _ in range(400):
            idx = rng.integers(0, len(ids), len(ids))
            f, g, v = funded[idx], goal[idx], s[idx]
            if (f == 1).sum() and (f == 0).sum():
                aucs.append(auc(v[f == 1], v[f == 0]))
            val, _ = same_goal(v, g, f)
            if val == val:
                sames.append(val)
        aucs.sort()
        sames.sort()
        out[name] = {
            "auc": round(base_auc * 100, 2),
            "auc_ci95": [round(aucs[int(0.025 * len(aucs))] * 100, 2),
                         round(aucs[int(0.975 * len(aucs))] * 100, 2)],
            "same_goal": round(base_same * 100, 2),
            "same_goal_ci95": [round(sames[int(0.025 * len(sames))] * 100, 2),
                               round(sames[int(0.975 * len(sames))] * 100, 2)],
            "same_goal_pairs": int(n_same),
        }
    # goal alone: a smaller goal is more likely to be funded, so accuracy counts
    # a pair as concordant when the funded project has the lower goal.
    gpos, gneg = goal[funded == 1], goal[funded == 0]
    gpos, gneg = gpos[~np.isnan(gpos)], gneg[~np.isnan(gneg)]
    # screen out pathological goals the way the recorded runs did
    d = gpos[:, None] - gneg[None, :]
    out["goal_baseline"] = round(
        (float((d < 0).sum()) + 0.5 * float((d == 0).sum())) / d.size * 100, 2
    )
    (pathlib.Path(__file__).resolve().parent.parent / "np_ci.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
