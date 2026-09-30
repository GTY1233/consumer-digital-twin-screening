"""Count dropped ties and report the alternative tie convention."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RUNS, read_jsonl  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scores = {
        r["id"]: r["scores"]
        for r in read_jsonl(RUNS / "ad_conf_AGG_flash.jsonl")
        if r.get("scores") is not None
    }
    tasks = read_jsonl(RUNS / "ad_confirmatory_tasks.jsonl")
    kept_conc = kept_pairs = 0
    half_conc = half_pairs = 0
    score_ties = 0
    no_pair_tests = 0
    for t in tasks:
        s = scores.get(t["id"])
        if s is None or len(s) != t["meta"]["n_arms"]:
            continue
        clicks = t["meta"]["clicks"]
        n = len(s)
        local = 0
        for i in range(n):
            for j in range(i + 1, n):
                half_pairs += 1
                if clicks[i] == clicks[j]:
                    half_conc += 0.5
                    continue
                if s[i] == s[j]:
                    score_ties += 1
                    half_conc += 0.5
                    continue
                kept_pairs += 1
                half_pairs_ok = (s[i] - s[j]) * (clicks[i] - clicks[j]) > 0
                kept_conc += 1 if half_pairs_ok else 0
                half_conc += 1 if half_pairs_ok else 0
                local += 1
        if local == 0:
            no_pair_tests += 1
    print(f"pairs excluding ties : {kept_pairs:,}  accuracy {kept_conc / kept_pairs * 100:.2f}%")
    print(f"pairs ties-as-half   : {half_pairs:,}  accuracy {half_conc / half_pairs * 100:.2f}%")
    print(f"score-tied pairs dropped under the primary convention: {score_ties:,}")
    print(f"tests with no usable pair: {no_pair_tests}")


if __name__ == "__main__":
    main()
