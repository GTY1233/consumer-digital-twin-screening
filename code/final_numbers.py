"""Compute every number the revised manuscript will quote, and dump them to
JSON so the text, tables and figures cannot drift apart."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from metrics import bootstrap_ci, evaluate_ad, evaluate_np  # noqa: E402
from panel_metrics import panel_ad, panel_np  # noqa: E402


def ad(condition_file: str, label: str) -> dict:
    out = evaluate_ad(RUNS / condition_file, RUNS / "ad_confirmatory_tasks.jsonl")
    lo, hi = bootstrap_ci([t for t in out.pop("_per_test") if t["pairs"] > 0])
    out["ci95"] = [round(lo, 4), round(hi, 4)]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def ad_exploratory(condition_file: str, limit_tasks: pathlib.Path) -> dict:
    out = evaluate_ad(RUNS / condition_file, limit_tasks)
    lo, hi = bootstrap_ci([t for t in out.pop("_per_test") if t["pairs"] > 0])
    out["ci95"] = [round(lo, 4), round(hi, 4)]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def np_condition(condition_file: str) -> dict:
    out = evaluate_np(RUNS / condition_file, RUNS / "np_test_labels.jsonl")
    return {
        k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    numbers = {
        "run2_advertising": {
            "AGG": ad("ad_conf_AGG_flash.jsonl", "AGG"),
            "IND": ad("ad_conf_IND_flash.jsonl", "IND"),
            "PANEL": panel_ad(
                RUNS / "ad_panel_flash.jsonl",
                RUNS / "ad_conf_AGG_flash.jsonl",
                RUNS / "ad_confirmatory_tasks.jsonl",
            ),
        },
        "run2_crowdfunding": {
            "AGG": np_condition("np_test_AGG_flash.jsonl"),
            "IND": np_condition("np_test_IND_flash.jsonl"),
            "PANEL": panel_np(
                RUNS / "np_panel_flash.jsonl",
                RUNS / "np_test_AGG_flash.jsonl",
                RUNS / "np_test_labels.jsonl",
            ),
        },
    }
    numbers["run2_crowdfunding"]["AGG"]["same_goal_accuracy"] = round(
        numbers["run2_crowdfunding"]["AGG"]["same_goal_accuracy"], 4
    )
    out = pathlib.Path(__file__).resolve().parent.parent / "final_numbers.json"
    out.write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(numbers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
