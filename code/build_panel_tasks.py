"""Build persona-panel task files (each item is repeated once per persona)."""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402
from prompts import AD_PERSONAS, NP_PERSONAS, ad_panel_system, np_panel_system  # noqa: E402


def build(src: str, dst: str, personas: dict, builder) -> None:
    rows = read_jsonl(RUNS / src)
    out = RUNS / dst
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            for name, text in personas.items():
                fh.write(
                    json.dumps(
                        {
                            "id": f"{row['id']}::{name}",
                            "system": builder(text),
                            "user": row["user"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"{dst}: {len(rows) * len(personas)} rows")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["ad", "np"])
    args = parser.parse_args()
    if args.domain == "ad":
        build("ad_conf_panel_tasks.jsonl", "ad_panel_all_tasks.jsonl", AD_PERSONAS, ad_panel_system)
    else:
        build("np_test_panel_tasks.jsonl", "np_panel_all_tasks.jsonl", NP_PERSONAS, np_panel_system)
