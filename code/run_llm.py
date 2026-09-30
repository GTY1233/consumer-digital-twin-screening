"""Run a batch of prompts through the DeepSeek API, resumably.

Usage:
  python run_llm.py --tasks tasks.jsonl --out results.jsonl \
      --model deepseek-flash --condition AGG --concurrency 64

Each task line: {"id": "...", "system": "...", "user": "..."}
Each result line: {"id", "condition", "model", "scores", "raw", "usage", "error"}
"""

import argparse
import asyncio
import pathlib
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import API_BASE, append_jsonl, done_ids, load_keys, parse_scores, read_jsonl  # noqa: E402
from prompts import (  # noqa: E402
    AD_AGG,
    AD_IND,
    AD_PERSONAS,
    NP_AGG,
    NP_IND,
    NP_PERSONAS,
    ad_panel_system,
    np_panel_system,
)

PROMPT_KEYS = ("AD_AGG", "AD_IND", "NP_AGG", "NP_IND", "AD_PANEL", "NP_PANEL")


def resolve_system(prompt_key: str, persona: str, fallback: str) -> str:
    if not prompt_key:
        return fallback
    if prompt_key == "AD_AGG":
        return AD_AGG
    if prompt_key == "AD_IND":
        return AD_IND
    if prompt_key == "NP_AGG":
        return NP_AGG
    if prompt_key == "NP_IND":
        return NP_IND
    if prompt_key == "AD_PANEL":
        return ad_panel_system(AD_PERSONAS[persona])
    if prompt_key == "NP_PANEL":
        return np_panel_system(NP_PERSONAS[persona])
    raise ValueError(f"unknown prompt key {prompt_key}")


def build_payload(
    model: str, system: str, user: str, thinking: str, effort: str, max_tokens: int
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if thinking == "disabled":
        payload["thinking"] = {"type": "disabled"}
        payload["temperature"] = 0
    else:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = effort
    return payload


async def call_one(
    client: httpx.AsyncClient,
    key: str,
    payload: dict,
    sem: asyncio.Semaphore,
    retries: int,
) -> dict:
    last_err = None
    for attempt in range(retries):
        async with sem:
            try:
                resp = await client.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=180.0,
                )
                if resp.status_code == 200:
                    return {"ok": True, "data": resp.json()}
                body = resp.text[:200]
                if resp.status_code in (408, 429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {body}"
                else:
                    return {"ok": False, "error": f"HTTP {resp.status_code}: {body}"}
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(min(2 ** attempt, 20) + 0.3 * attempt)
    return {"ok": False, "error": f"retries exhausted: {last_err}"}


async def run(args: argparse.Namespace) -> None:
    tasks = read_jsonl(pathlib.Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out_path) if args.resume else set()
    todo = [t for t in tasks if t["id"] not in skip]
    print(f"tasks={len(tasks)} already_done={len(skip)} todo={len(todo)}", flush=True)
    if not todo:
        return

    keys = load_keys()
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency,
    )
    started = time.time()
    state = {"done": 0, "errors": 0}

    async def work(index: int, task: dict) -> dict:
        key = keys[index % len(keys)]
        system = resolve_system(args.prompt_key, args.persona, task["system"])
        payload = build_payload(
            args.model, system, task["user"], args.thinking, args.effort, args.max_tokens
        )
        result = await call_one(client, key, payload, sem, args.retries)
        rec = {"id": task["id"], "condition": args.condition, "model": args.model}
        if args.persona:
            rec["persona"] = args.persona
        if result["ok"]:
            data = result["data"]
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            rec["raw"] = content
            rec["scores"] = parse_scores(content)
            rec["usage"] = data.get("usage")
            rec["finish_reason"] = choice.get("finish_reason")
            if rec["scores"] is None:
                rec["error"] = "unparsed"
                state["errors"] += 1
        else:
            rec["scores"] = None
            rec["raw"] = None
            rec["error"] = result["error"]
            state["errors"] += 1
        return rec

    # trust_env=False bypasses the local system proxy; direct connections to
    # api.deepseek.com are roughly 10x faster here.
    async with httpx.AsyncClient(limits=limits, trust_env=False, timeout=90) as client:
        running: set = set()
        buffer: list[dict] = []
        for index, task in enumerate(todo):
            running.add(asyncio.create_task(work(index, task)))
            if len(running) >= args.window:
                finished, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for fut in finished:
                    buffer.append(fut.result())
                    state["done"] += 1
                if len(buffer) >= max(20, args.window):
                    append_jsonl(out_path, buffer)
                    buffer = []
                    rate = state["done"] / max(time.time() - started, 1e-9)
                    eta = (len(todo) - state["done"]) / rate if rate else 0
                    print(
                        f"  {state['done']}/{len(todo)} errors={state['errors']} "
                        f"rate={rate:.1f}/s eta={eta / 60:.1f}min",
                        flush=True,
                    )
        for fut in asyncio.as_completed(running):
            buffer.append(await fut)
            state["done"] += 1
        if buffer:
            append_jsonl(out_path, buffer)

    elapsed = time.time() - started
    print(
        f"finished done={state['done']} errors={state['errors']} elapsed={elapsed / 60:.1f}min",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--condition", default="AGG")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--prompt-key", default="", choices=[""] + list(PROMPT_KEYS))
    parser.add_argument("--persona", default="")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
