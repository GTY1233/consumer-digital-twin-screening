"""Project-level interval for the funding-goal baseline, matching the rule set
used for the system (no goal-value filter)."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [r for r in read_jsonl(RUNS / "np_test_labels.jsonl")
            if not np.isnan(r["goal"])]
    g = np.array([r["goal"] for r in rows], dtype=float)
    f = np.array([r["funded"] for r in rows], dtype=float)

    def acc(pos_goal, neg_goal):
        d = pos_goal[:, None] - neg_goal[None, :]
        return (float((d < 0).sum()) + 0.5 * float((d == 0).sum())) / d.size

    base = acc(g[f == 1], g[f == 0])
    rng = np.random.default_rng(77)
    vals = []
    n = len(rows)
    for _ in range(400):
        idx = rng.integers(0, n, n)
        gg, ff = g[idx], f[idx]
        if (ff == 1).sum() and (ff == 0).sum():
            vals.append(acc(gg[ff == 1], gg[ff == 0]))
    vals.sort()
    out = {"goal_baseline": round(base * 100, 2),
           "ci95": [round(vals[int(0.025 * len(vals))] * 100, 2),
                    round(vals[int(0.975 * len(vals))] * 100, 2)],
           "projects": len(rows)}
    (pathlib.Path(__file__).resolve().parent.parent / "goal_ci.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
