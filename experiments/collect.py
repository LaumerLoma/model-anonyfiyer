"""Collect experiment data from local LM Studio models.

Reuses the tool's own prompt templates (FORWARD, RETURN_Z, HOP), so it tests the real system.
Calls run in stages grouped by executing model, because only one model fits in RAM at a time.
Every call is cached in data/calls.jsonl, so an interrupted run can resume.

    uv run --group experiments python experiments/collect.py [--prompts N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prompts import PROMPTS  # noqa: E402

from telephone_onion.circuit import build_circuit  # noqa: E402
from telephone_onion.nodes import LOCAL_LABS  # noqa: E402
from telephone_onion.router import FORWARD, HOP, RETURN_Z  # noqa: E402

DATA = Path(__file__).parent / "data"
CALLS = DATA / "calls.jsonl"
LMS = str(Path.home() / ".lmstudio/bin/lms")
URL = "http://localhost:1234/v1/chat/completions"
GEN = {"temperature": 0.7, "max_tokens": 400}
CONCURRENCY = 4
LABS = {l.key: l for l in LOCAL_LABS}


def units(n_prompts: int):
    for p in range(n_prompts):
        for z in LABS:
            rng = random.Random(f"unit-{p}-{z}")
            c = build_circuit(LOCAL_LABS, rng)
            para = rng.choice(LOCAL_LABS)
            yield {"p": p, "z": z, "guard": c.guard.key, "middle": c.middle.key,
                   "exit": c.exit.key, "para": para.key}


def load_cache() -> dict[str, dict]:
    if not CALLS.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, CALLS.read_text().splitlines())}


def lms(*args: str) -> None:
    subprocess.run([LMS, *args], check=True, capture_output=True, text=True)


def ensure_loaded(lab_key: str) -> None:
    lms("unload", "--all")
    lms("load", LABS[lab_key].model, "--context-length", "4096", "-y")


async def run_jobs(jobs: list[dict], cache: dict[str, dict]) -> None:
    """jobs: {id, exec, prompt}. Groups by executor, loads each model once."""
    todo = [j for j in jobs if j["id"] not in cache]
    by_exec: dict[str, list[dict]] = {}
    for j in todo:
        by_exec.setdefault(j["exec"], []).append(j)
    async with httpx.AsyncClient(timeout=600) as client:
        for lab_key, group in by_exec.items():
            t0 = time.time()
            ensure_loaded(lab_key)
            sem = asyncio.Semaphore(CONCURRENCY)
            bar = tqdm(total=len(group), desc=f"  {lab_key}", unit="call", dynamic_ncols=True)

            async def one(j):
                async with sem:
                    for attempt in range(3):
                        try:
                            r = await client.post(URL, json={
                                "model": LABS[lab_key].model,
                                "messages": [{"role": "user", "content": j["prompt"]}], **GEN})
                            r.raise_for_status()
                            out = r.json()["choices"][0]["message"]["content"].strip()
                            break
                        except (httpx.HTTPError, KeyError) as e:
                            if attempt == 2:
                                raise
                            await asyncio.sleep(2)
                rec = {**j, "output": out}
                cache[j["id"]] = rec
                with CALLS.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                bar.update()

            await asyncio.gather(*(one(j) for j in group))
            bar.close()
            print(f"  {lab_key}: {len(group)} calls in {time.time() - t0:.0f}s", flush=True)


def name(key: str) -> str:
    return LABS[key].name


async def main(n_prompts: int) -> None:
    DATA.mkdir(exist_ok=True)
    cache = load_cache()
    U = list(units(n_prompts))
    (DATA / "units.json").write_text(json.dumps(U, indent=1))
    out = lambda i: cache[i]["output"]  # noqa: E731

    print("stage 1: direct, persona_fwd, direct2 (all by Z)")
    jobs = []
    for u in U:
        q = PROMPTS[u["p"]][1]
        k = f"{u['p']}-{u['z']}"
        jobs.append({"id": f"direct/{k}", "exec": u["z"], "prompt": q})
        jobs.append({"id": f"persona_fwd/{k}", "exec": u["z"], "prompt": FORWARD.format(
            a=name(u["guard"]), b=name(u["middle"]), c=name(u["exit"]), prompt=q)})
        if list(LABS).index(u["z"]) == u["p"] % len(LABS):  # one repeat sample per prompt
            jobs.append({"id": f"direct2/{k}", "exec": u["z"], "prompt": q})
    await run_jobs(jobs, cache)

    print("stage 2: z_mode return (Z), hop1 (exit), para_1 (random lab)")
    jobs = []
    for u in U:
        k = f"{u['p']}-{u['z']}"
        fwd = out(f"persona_fwd/{k}")
        jobs.append({"id": f"z_mode/{k}", "exec": u["z"], "prompt": RETURN_Z.format(
            a=name(u["guard"]), b=name(u["middle"]), c=name(u["exit"]), answer=fwd)})
        jobs.append({"id": f"nodes_1/{k}", "exec": u["exit"], "prompt": HOP.format(answer=fwd)})
        jobs.append({"id": f"para_1/{k}", "exec": u["para"],
                     "prompt": HOP.format(answer=out(f"direct/{k}"))})
    await run_jobs(jobs, cache)

    for stage, (src, dst, role) in enumerate([("nodes_1", "nodes_2", "middle"),
                                              ("nodes_2", "nodes_3", "guard")], start=3):
        print(f"stage {stage}: {dst} ({role})")
        jobs = [{"id": f"{dst}/{u['p']}-{u['z']}", "exec": u[role],
                 "prompt": HOP.format(answer=out(f"{src}/{u['p']}-{u['z']}"))} for u in U]
        await run_jobs(jobs, cache)

    lms("unload", "--all")
    print(f"done: {len(cache)} calls cached in {CALLS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    asyncio.run(main(ap.parse_args().prompts))
