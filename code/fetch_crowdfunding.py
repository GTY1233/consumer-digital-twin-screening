"""Download the Kickstarter corpus used in the second domain.

Without this file the released run identifiers (NP::<split>::<row index>)
cannot be aligned back to the source data, because they are row positions in
these parquet files. Hugging Face is not reachable from every network, so the
mirror is tried first and the canonical host second.
"""

import hashlib
import os
import pathlib
import sys
import time

import requests

DATA = pathlib.Path(os.environ["USERPROFILE"]) / ".paper-work" / "data"
REPO = "james-burton/kick_starter_funding_all_text"
REVISION = "main"
FILES = {
    "train": "train-00000-of-00001-ef30e64c7bdf5a77.parquet",
    "validation": "validation-00000-of-00001-92035f6540ac4b59.parquet",
    "test": "test-00000-of-00001-4e986634b901c92f.parquet",
}
HOSTS = ("https://hf-mirror.com", "https://huggingface.co")


def fetch(split: str, filename: str) -> pathlib.Path:
    target = DATA / f"kickstarter-{split}.parquet"
    if target.exists() and target.stat().st_size > 0:
        print(f"{target.name}: already present ({target.stat().st_size:,} bytes)")
        return target
    path = f"datasets/{REPO}/resolve/{REVISION}/data/{filename}"
    last = None
    for host in HOSTS:
        for attempt in range(4):
            try:
                with requests.get(f"{host}/{path}", stream=True, timeout=(30, 300)) as resp:
                    if resp.status_code != 200:
                        last = f"{host} HTTP {resp.status_code}"
                        break
                    with target.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
                print(f"{target.name}: {target.stat().st_size:,} bytes sha256[:16]={digest}")
                return target
            except Exception as exc:
                last = f"{host} {type(exc).__name__}"
                time.sleep(2 + attempt)
    raise RuntimeError(f"could not download {filename}: {last}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(parents=True, exist_ok=True)
    for split, filename in FILES.items():
        fetch(split, filename)


if __name__ == "__main__":
    main()
