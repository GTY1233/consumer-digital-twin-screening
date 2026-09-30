"""Shared config and helpers for the re-run pipeline."""

import json
import os
import pathlib
import random
import re

HOME = pathlib.Path(os.environ["USERPROFILE"])
DATA = HOME / ".paper-work" / "data"
WORK = HOME / ".paper-work"
RUNS = WORK / "runs"
SECRETS = HOME / ".paper-secrets" / "deepseek.env"

API_BASE = "https://api.deepseek.com"


def load_keys() -> list[str]:
    """Read API keys from DEEPSEEK_API_KEYS, or from a local env file.

    DEEPSEEK_API_KEYS takes a comma-separated list. If it is unset, the file
    named by DEEPSEEK_ENV_FILE is read, defaulting to ~/.paper-secrets/deepseek.env,
    in which each line has the form NAME=value.
    """
    inline = os.environ.get("DEEPSEEK_API_KEYS", "").strip()
    if inline:
        keys = [k.strip() for k in inline.split(",") if k.strip()]
        if keys:
            return keys

    env_file = pathlib.Path(os.environ.get("DEEPSEEK_ENV_FILE", SECRETS))
    keys = []
    for line in env_file.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        keys.append(line.split("=", 1)[1].strip())
    if not keys:
        raise RuntimeError(
            "no API keys: set DEEPSEEK_API_KEYS or populate " + str(env_file)
        )
    return keys


def ensure_dirs() -> None:
    for d in (DATA, RUNS):
        d.mkdir(parents=True, exist_ok=True)


def shuffled(items: list, seed: int) -> list:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def seed_from(text: str) -> int:
    return int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "big") % (2**31)


JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_scores(raw: str) -> list[int] | None:
    """Extract the integer score list from a model reply, tolerating markdown fences."""
    if raw is None:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    for key in ("scores", "score", "scores_list"):
        if key in obj:
            value = obj[key]
            if isinstance(value, list):
                try:
                    return [int(round(float(v))) for v in value]
                except (TypeError, ValueError):
                    return None
            if isinstance(value, (int, float)):
                return [int(round(float(value)))]
    return None


def append_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def done_ids(path: pathlib.Path) -> set[str]:
    ids = set()
    for rec in read_jsonl(path):
        if rec.get("scores") is not None:
            ids.add(rec["id"])
    return ids
