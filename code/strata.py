"""Stratified pairwise accuracy for the reported advertising condition."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from metrics import pairwise_ad  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scores = {
        r["id"]: r["scores"]
        for r in read_jsonl(RUNS / "ad_conf_AGG_flash.jsonl")
        if r.get("scores") is not None
    }
    tasks = read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")

    rows = []
    for t in tasks:
        s = scores.get(t["id"])
        if s is None or len(s) != t["meta"]["n_arms"]:
            continue
        _, pairs, acc = pairwise_ad(s, t["meta"]["clicks"])
        if not pairs:
            continue
        imp = sum(t["meta"]["impressions"])
        ctrs = [c / i for c, i in zip(t["meta"]["clicks"], t["meta"]["impressions"])]
        mean_ctr = sum(t["meta"]["clicks"]) / imp
        best = max(range(len(s)), key=lambda k: (s[k], -k))
        rows.append(
            {
                "impressions": imp,
                "arms": t["meta"]["n_arms"],
                "pairs": pairs,
                "acc": acc,
                "lift": ctrs[best] / mean_ctr - 1,
            }
        )

    def report(name, subset):
        p = sum(r["pairs"] for r in subset)
        a = sum(r["acc"] * r["pairs"] for r in subset) / p
        lift = sum(r["lift"] for r in subset) / len(subset)
        return {"stratum": name, "tests": len(subset), "pairs": p,
                "accuracy": round(a * 100, 2), "lift": round(lift * 100, 2)}

    out = []
    ordered = sorted(rows, key=lambda r: r["impressions"])
    q = len(ordered) // 4
    for i in range(4):
        chunk = ordered[i * q : (i + 1) * q] if i < 3 else ordered[3 * q :]
        out.append(report(f"Total impressions, Q{i + 1}", chunk))
    for arms in (2, 3, 4, 5, 6):
        sub = [r for r in rows if r["arms"] == arms]
        if sub:
            out.append(report(f"{arms} arms", sub))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (pathlib.Path(__file__).resolve().parent.parent / "strata.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
