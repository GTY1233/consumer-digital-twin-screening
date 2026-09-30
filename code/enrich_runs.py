"""Attach the exact system and user message to every released run record.

The API runner stored only the returned scores, not the prompts, so the release
as first published did not contain them.  The prompts are deterministic
functions of the task files and the prompt definitions, so they can be restored
exactly.  Output is written under .paper-work/enriched/.
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from prompts import (  # noqa: E402
    AD_AGG,
    AD_IND,
    AD_PERSONAS,
    NP_AGG,
    NP_IND,
    NP_PERSONAS,
    ad_panel_agg_system,
    ad_panel_system,
    np_panel_agg_system,
    np_panel_system,
)

OUT = pathlib.Path(os.environ["USERPROFILE"]) / ".paper-work" / "enriched"

PLAN = {
    "ad_conf_AGG_flash.jsonl": ("ad_confirmatory_tasks.jsonl", "AD_AGG"),
    "ad_conf_IND_flash.jsonl": ("ad_confirmatory_tasks.jsonl", "AD_IND"),
    "ad_exp_AGG_flash.jsonl": ("ad_exploratory_tasks.jsonl", "AD_AGG"),
    "ad_exp_IND_flash.jsonl": ("ad_exploratory_tasks.jsonl", "AD_IND"),
    "ad_exp_AGG_pro.jsonl": ("ad_exploratory_tasks.jsonl", "AD_AGG"),
    "ad_panel_flash.jsonl": ("ad_conf_panel_tasks.jsonl", "AD_PANEL_ROLEPLAY"),
    "ad_panel_agg_flash.jsonl": ("ad_conf_panel_tasks.jsonl", "AD_PANEL_AGG"),
    "np_test_AGG_flash.jsonl": ("np_test_tasks.jsonl", "NP_AGG"),
    "np_test_IND_flash.jsonl": ("np_test_tasks.jsonl", "NP_IND"),
    "np_panel_flash.jsonl": ("np_test_panel_tasks.jsonl", "NP_PANEL_ROLEPLAY"),
    "np_panel_agg_flash.jsonl": ("np_test_panel_tasks.jsonl", "NP_PANEL_AGG"),
    "np_val_A_flash.jsonl": ("np_validation_tasks.jsonl", "NP_AGG"),
    "np_val_B_flash.jsonl": ("np_validation_B_tasks.jsonl", "NP_AGG"),
}


def system_for(key: str, persona: str, fallback: str) -> str:
    if key == "AD_AGG":
        return AD_AGG
    if key == "AD_IND":
        return AD_IND
    if key == "NP_AGG":
        return NP_AGG
    if key == "NP_IND":
        return NP_IND
    if key == "AD_PANEL_ROLEPLAY":
        return ad_panel_system(AD_PERSONAS[persona])
    if key == "AD_PANEL_AGG":
        return ad_panel_agg_system(AD_PERSONAS[persona])
    if key == "NP_PANEL_ROLEPLAY":
        return np_panel_system(NP_PERSONAS[persona])
    if key == "NP_PANEL_AGG":
        return np_panel_agg_system(NP_PERSONAS[persona])
    return fallback


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for out_name, (task_file, key) in PLAN.items():
        src = RUNS / out_name
        if not src.exists():
            print(f"skip {out_name} (not present)")
            continue
        tasks = {t["id"]: t for t in read_jsonl(RUNS / task_file)}
        written = 0
        with (OUT / out_name).open("w", encoding="utf-8") as fh:
            for rec in read_jsonl(src):
                rid = rec["id"]
                if rid in seen:
                    continue
                seen.add(rid)
                task = tasks.get(rid.split("::")[0] + "::" + rid.split("::")[1]) if "::" in rid else None
                parts = rid.split("::")
                persona = parts[2] if (key.endswith(("ROLEPLAY", "AGG")) and len(parts) == 3) else ""
                base_id = "::".join(parts[:3]) if len(parts) == 4 else rid
                task = tasks.get(base_id)
                if task is None:
                    continue
                rec = dict(rec)
                rec["system"] = system_for(key, persona, task.get("system", ""))
                rec["user"] = task["user"]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        print(f"{out_name}: {written} records")


if __name__ == "__main__":
    main()
