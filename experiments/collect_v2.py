"""Collect the v2 conditions (PROTOCOL_v2.md) on top of the v1 cache.

    uv run --group experiments python experiments/collect_v2.py [--prompts N]

Reuses every v1 call. New ids: direct/persona_fwd/nodes_k for prompts 80-159, and
plain_1..3, fixed_1 and plain_1_hot for all prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from collect import CALLS, DATA, LABS, load_cache, lms, name, run_jobs, units  # noqa: E402
from prompts_extra import ALL_PROMPTS  # noqa: E402

from telephone_onion.router import FORWARD, HOP  # noqa: E402

FIXED = "meta"      # fixed rewriter, chosen before data collection (PROTOCOL_v2.md)
HOT = 1.2           # temperature for plain_1_hot


async def main(n_prompts: int) -> None:
    DATA.mkdir(exist_ok=True)
    cache = load_cache()
    U = list(units(n_prompts))
    (DATA / "units_v2.json").write_text(json.dumps(U, indent=1))
    out = lambda i: cache[i]["output"]  # noqa: E731
    key = lambda u: f"{u['p']}-{u['z']}"  # noqa: E731

    print("stage 1: direct, persona_fwd (all by Z)")
    jobs = []
    for u in U:
        q = ALL_PROMPTS[u["p"]][1]
        jobs.append({"id": f"direct/{key(u)}", "exec": u["z"], "prompt": q})
        jobs.append({"id": f"persona_fwd/{key(u)}", "exec": u["z"], "prompt": FORWARD.format(
            a=name(u["guard"]), b=name(u["middle"]), c=name(u["exit"]), prompt=q)})
    await run_jobs(jobs, cache)

    print("stage 2: first hops (nodes_1, plain_1, plain_1_hot by exit; fixed_1 by the fixed lab)")
    jobs = []
    for u in U:
        k = key(u)
        jobs.append({"id": f"nodes_1/{k}", "exec": u["exit"], "prompt": HOP.format(answer=out(f"persona_fwd/{k}"))})
        jobs.append({"id": f"plain_1/{k}", "exec": u["exit"], "prompt": HOP.format(answer=out(f"direct/{k}"))})
        jobs.append({"id": f"plain_1_hot/{k}", "exec": u["exit"], "temperature": HOT,
                     "prompt": HOP.format(answer=out(f"direct/{k}"))})
        jobs.append({"id": f"fixed_1/{k}", "exec": FIXED, "prompt": HOP.format(answer=out(f"direct/{k}"))})
    await run_jobs(jobs, cache)

    for stage, (hop, role) in enumerate([(2, "middle"), (3, "guard")], start=3):
        print(f"stage {stage}: hop {hop} ({role}) for nodes and plain")
        jobs = []
        for u in U:
            k = key(u)
            for chain in ("nodes", "plain"):
                jobs.append({"id": f"{chain}_{hop}/{k}", "exec": u[role],
                             "prompt": HOP.format(answer=out(f"{chain}_{hop - 1}/{k}"))})
        await run_jobs(jobs, cache)

    lms("unload", "--all")
    print(f"done: {len(cache)} calls cached in {CALLS.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=len(ALL_PROMPTS))
    args = ap.parse_args()
    assert set(LABS) >= {FIXED}
    asyncio.run(main(args.prompts))
