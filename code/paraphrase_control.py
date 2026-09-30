"""Memorisation control: paraphrase the headlines, re-score them, compare.

Usage:  python paraphrase_control.py {tasks|score|analyse}

Stage tasks   -> paraphrase task file (one call per test, all its headlines)
Stage score   -> scoring task file built from the returned paraphrases
Stage analyse -> pairwise accuracy of the paraphrased stimuli against the same
                 tests' outcomes, next to the original stimuli
"""

import json
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl, seed_from  # noqa: E402
from metrics import pairwise_ad  # noqa: E402
from prompts import AD_AGG  # noqa: E402

N_TESTS = 2000
SEED = 20260930

PARA_SYSTEM = (
    "You rewrite headlines. You will be given a numbered list of headlines. Rewrite each one in "
    "different words while keeping its meaning and its level of appeal as close to the original as "
    "you can. Do not add or remove information, do not explain, and do not comment on the "
    "headlines. Return exactly one JSON object with key headlines: an array of rewritten strings, "
    "the same length as the input and in the same order."
)


def sample_tests() -> list[dict]:
    tasks = read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")
    rng = random.Random(SEED)
    picked = rng.sample(tasks, min(N_TESTS, len(tasks)))
    picked.sort(key=lambda t: t["id"])
    return picked


def stage_tasks() -> None:
    rows = []
    for task in sample_tests():
        heads = task["meta"]["headlines"]
        user = "Headlines:\n" + "\n".join(f"{i}. {h}" for i, h in enumerate(heads, 1))
        rows.append({"id": task["id"], "system": PARA_SYSTEM, "user": user})
    out = RUNS / "pc_para_tasks.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out.name}: {len(rows)} paraphrase calls")


def parse_headlines(raw: str) -> list[str] | None:
    import re

    if not raw:
        return None
    text = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    for key in ("headlines", "rewritten", "headline"):
        if key in obj and isinstance(obj[key], list):
            return [str(x).strip() for x in obj[key]]
    return None


def stage_score() -> None:
    tasks = {t["id"]: t for t in read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")}
    rows = []
    used = 0
    for rec in read_jsonl(RUNS / "pc_para_out.jsonl"):
        para = parse_headlines(rec.get("raw") or "")
        task = tasks.get(rec["id"])
        if not para or task is None or len(para) != task["meta"]["n_arms"]:
            continue
        if any(not p for p in para):
            continue
        excerpt = task["user"].split("\n\nHeadlines:")[0]
        user = excerpt + "\n\nHeadlines:\n" + "\n".join(
            f"{i}. {h}" for i, h in enumerate(para, 1)
        )
        rows.append({"id": rec["id"], "system": AD_AGG, "user": user})
        used += 1
    out = RUNS / "pc_score_tasks.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out.name}: {used} scoring calls")


def stage_analyse() -> None:
    tasks = {t["id"]: t for t in read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")}
    orig = {r["id"]: r["scores"] for r in read_jsonl(RUNS / "ad_conf_AGG_flash.jsonl")
            if r.get("scores") is not None}
    para = {r["id"]: r["scores"] for r in read_jsonl(RUNS / "pc_score_out.jsonl")
            if r.get("scores") is not None}
    ids = [i for i in sorted(para) if i in orig and i in tasks]
    rows = []
    for i in ids:
        task = tasks[i]
        clicks = task["meta"]["clicks"]
        if len(orig[i]) != task["meta"]["n_arms"] or len(para[i]) != task["meta"]["n_arms"]:
            continue
        _, p1, a1 = pairwise_ad(orig[i], clicks)
        _, p2, a2 = pairwise_ad(para[i], clicks)
        if p1 and p2:
            rows.append({"o": (a1, p1), "p": (a2, p2)})
    n = len(rows)

    def acc(sel, key):
        p = sum(r[key][1] for r in sel)
        return sum(r[key][0] * r[key][1] for r in sel) / p if p else float("nan")

    a_o, a_p = acc(rows, "o") * 100, acc(rows, "p") * 100
    rng = np.random.default_rng(101)
    diffs = []
    for _ in range(2000):
        s = [rows[i] for i in rng.integers(0, n, n)]
        diffs.append(acc(s, "p") - acc(s, "o"))
    diffs.sort()
    out = {
        "tests": n,
        "pairs": sum(r["o"][1] for r in rows),
        "original_accuracy": round(a_o, 2),
        "paraphrased_accuracy": round(a_p, 2),
        "difference": round(a_p - a_o, 2),
        "difference_ci95": [round(diffs[int(0.025 * len(diffs))] * 100, 2),
                            round(diffs[int(0.975 * len(diffs))] * 100, 2)],
    }
    (pathlib.Path(__file__).resolve().parent.parent / "paraphrase_control.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"tasks": stage_tasks, "score": stage_score, "analyse": stage_analyse}[sys.argv[1]]()
