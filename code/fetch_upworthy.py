"""Resumable downloader for the Upworthy archive CSVs.

The OSF mirror is slow and drops connections, and curl's -C - restarts from
zero because the redirect target does not advertise range support.  This
script probes range support once and then keeps whatever it has.
"""

import pathlib
import sys
import time

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

FILES = {
    "upworthy-confirmatory.csv": "https://osf.io/download/vy8mj/",
    "upworthy-exploratory.csv": "https://osf.io/download/3vqmp/",
    "upworthy-holdout.csv": "https://osf.io/download/ynf3k/",
}


def fetch(name: str, url: str) -> None:
    target = DATA / name
    if target.exists() and target.stat().st_size > 0:
        head = requests.head(url, allow_redirects=True, timeout=60)
        total = int(head.headers.get("Content-Length", 0))
        if total and target.stat().st_size == total:
            print(f"{name}: already complete ({total:,} bytes)")
            return

    for attempt in range(60):
        have = target.stat().st_size if target.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as resp:
                if resp.status_code not in (200, 206):
                    print(f"  {name}: HTTP {resp.status_code}, retrying")
                    time.sleep(3)
                    continue
                if have and resp.status_code == 200:
                    # server ignored the range; start over
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                total = int(resp.headers.get("Content-Length", 0)) + have
                with target.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
        except Exception as exc:
            print(f"  {name}: {type(exc).__name__} at {have:,} bytes, retrying")
            time.sleep(3)
            continue
        size = target.stat().st_size
        print(f"  {name}: {size:,} bytes (attempt {attempt + 1})")
        if total and size >= total:
            print(f"{name}: complete")
            return
    print(f"{name}: gave up")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1:] or list(FILES)
    for name in want:
        fetch(name, FILES[name])


if __name__ == "__main__":
    main()
