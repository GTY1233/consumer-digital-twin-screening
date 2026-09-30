"""Panel comparisons with the wrapper held constant, plus paired bootstrap CIs."""

import json
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from metrics import _np_accuracy, pairwise_ad  # noqa: E402


def ad_panel_scores(path: pathlib.Path) -> dict[str, list[list[int]]]:
    by_test: dict[str, list[list[int]]] = defaultdict(list)
    for rec in read_jsonl(path):
        if rec.get("scores") is None:
            continue
        parts = rec["id"].split("::")
        if len(parts) == 3:
            by_test[parts[1]].append(rec["scores"])
    return by_test


def ad_rows(panel: dict, stores: dict[str, dict]) -> list[dict]:
    tasks = {t["id"]: t for t in read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")}
    rows = []
    for test_id, lists in panel.items():
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
        if not pp:
            continue
        row = {"panel": (pa, pp)}
        for name, store in stores.items():
            if key in store and len(store[key]) == n_arms:
                _, bp, ba = pairwise_ad(store[key], task["meta"]["clicks"])
                if bp:
                    row[name] = (ba, bp)
        if all(k in row for k in ("panel", *stores)):
            rows.append(row)
    return rows


def acc(rows: list[dict], key: str) -> float:
    p = sum(r[key][1] for r in rows)
    return sum(r[key][0] * r[key][1] for r in rows) / p if p else float("nan")


def paired(rows: list[dict], a: str, b: str, rounds: int = 2000, seed: int = 21):
    rng = random.Random(seed)
    n = len(rows)
    diffs = []
    for _ in range(rounds):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        diffs.append(acc(s, a) - acc(s, b))
    diffs.sort()
    return round(diffs[int(0.025 * len(diffs))] * 100, 2), round(diffs[int(0.975 * len(diffs))] * 100, 2)


def load_store(name: str) -> dict:
    return {r["id"]: r["scores"] for r in read_jsonl(RUNS / name) if r.get("scores") is not None}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out: dict = {}

    stores = {
        "agg": load_store("ad_conf_AGG_flash.jsonl"),
        "ind": load_store("ad_conf_IND_flash.jsonl"),
    }
    panel_role = ad_panel_scores(RUNS / "ad_panel_flash.jsonl")
    panel_agg = ad_panel_scores(RUNS / "ad_panel_agg_flash.jsonl")

    for label, panel in (("panel_roleplay", panel_role), ("panel_aggregate", panel_agg)):
        rows = ad_rows(panel, stores)
        block = {"tests": len(rows)}
        for key in ("agg", "ind"):
            block[f"{key}_accuracy"] = round(acc(rows, key) * 100, 2)
        block["panel_accuracy"] = round(acc(rows, "panel") * 100, 2)
        block["panel_minus_agg"] = round((acc(rows, "panel") - acc(rows, "agg")) * 100, 2)
        block["panel_minus_agg_ci95"] = paired(rows, "panel", "agg")
        block["panel_minus_ind"] = round((acc(rows, "panel") - acc(rows, "ind")) * 100, 2)
        block["panel_minus_ind_ci95"] = paired(rows, "panel", "ind")
        block["agg_minus_ind"] = round((acc(rows, "agg") - acc(rows, "ind")) * 100, 2)
        out[label] = block

    # crowdfunding
    labels = {r["id"]: r for r in read_jsonl(RUNS / "np_test_labels.jsonl")}
    np_stores = {
        "agg": load_store("np_test_AGG_flash.jsonl"),
        "ind": load_store("np_test_IND_flash.jsonl"),
    }
    for label, fname in (("panel_roleplay", "np_panel_flash.jsonl"),
                         ("panel_aggregate", "np_panel_agg_flash.jsonl")):
        by_id: dict[str, list[float]] = defaultdict(list)
        for rec in read_jsonl(RUNS / fname):
            if rec.get("scores"):
                parts = rec["id"].split("::")
                by_id["::".join(parts[:3])].append(float(rec["scores"][0]))
        panel = {k: sum(v) / len(v) for k, v in by_id.items() if len(v) == 8}
        ids = sorted(panel)
        base = {
            key: {k: v for k, v in store.items() if k in panel} for key, store in np_stores.items()
        }
        block = {"items": len(ids)}
        for key in ("agg", "ind"):
            block[f"{key}_auc"] = round(
                _np_accuracy([(base[key][i], base[key][j])
                              for i in ids for j in ids
                              if labels[i]["funded"] == 1 and labels[j]["funded"] == 0]) * 100, 2)
        block["panel_auc"] = round(
            _np_accuracy([(panel[i], panel[j]) for i in ids for j in ids
                          if labels[i]["funded"] == 1 and labels[j]["funded"] == 0]) * 100, 2)
        for key in ("agg", "ind"):
            diffs = []
            rng = random.Random(23)
            for _ in range(500):
                s = [ids[rng.randrange(len(ids))] for _ in range(len(ids))]
                pos = [i for i in s if labels[i]["funded"] == 1]
                neg = [i for i in s if labels[i]["funded"] == 0]
                p = _np_accuracy([(panel[i], panel[j]) for i in pos for j in neg])
                q = _np_accuracy([(base[key][i], base[key][j]) for i in pos for j in neg])
                diffs.append(p - q)
            diffs.sort()
            block[f"panel_minus_{key}"] = round(
                (block["panel_auc"] - block[f"{key}_auc"]), 2)
            block[f"panel_minus_{key}_ci95"] = [round(diffs[int(0.025 * len(diffs))] * 100, 2),
                                                round(diffs[int(0.975 * len(diffs))] * 100, 2)]
        out[f"np_{label}"] = block

    print(json.dumps(out, indent=2, ensure_ascii=False))
    (pathlib.Path(__file__).resolve().parent.parent / "panel_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
