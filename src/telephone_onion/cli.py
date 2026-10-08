"""Command line entry point: `telephone tui | ask | bench`."""

from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import replace

from .attack import JudgeAttacker, MarkerAttacker, run_bench
from .backends import MockBackend, RouterBackend
from .nodes import DEFAULT_LABS, Lab, lab_by_key


def _labs(overrides: list[str]) -> tuple[Lab, ...]:
    labs = list(DEFAULT_LABS)
    for o in overrides:
        key, slug = o.split("=", 1)
        labs = [replace(l, model=slug) if l.key == key else l for l in labs]
    return tuple(labs)


def _backend(args):
    if args.backend == "mock":
        return MockBackend(seed=args.seed, latency=getattr(args, "latency", 0.0))
    return RouterBackend(args.backend)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="telephone", description="Obscure which model wrote an answer.")
    p.add_argument("--backend", choices=["mock", "openrouter", "gateway"], default="mock")
    p.add_argument("--model", action="append", default=[], metavar="LAB=SLUG",
                   help="override a lab's model slug, e.g. openai=openai/gpt-5-mini")
    p.add_argument("--seed", type=int, default=None)
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("tui", help="interactive abstract TUI (default)")
    t.add_argument("--z", default=None, help="lab key Z runs on (default: random, hidden)")
    t.add_argument("--relay", choices=["z", "nodes"], default="z")
    t.add_argument("--latency", type=float, default=0.6, help="mock latency per call, seconds")

    a = sub.add_parser("ask", help="one-shot query, prints the anonymised answer")
    a.add_argument("prompt")
    a.add_argument("--z", default=None)
    a.add_argument("--relay", choices=["z", "nodes"], default="z")
    a.add_argument("--trace", action="store_true", help="print every hop to stderr")

    b = sub.add_parser("bench", help="black-box attribution test")
    b.add_argument("--trials", type=int, default=200)
    b.add_argument("--judge", default=None, help="lab key for the LLM judge (real backends)")

    args = p.parse_args(argv)
    labs = _labs(args.model)

    if args.cmd in (None, "tui"):
        from .tui import TelephoneApp
        if args.cmd is None:
            args.z, args.relay, args.latency = None, "z", 0.6
        z = lab_by_key(labs, args.z) if args.z else random.SystemRandom().choice(labs)
        TelephoneApp(_backend(args), labs, z, args.relay).run()
    elif args.cmd == "ask":
        asyncio.run(_ask(args, labs))
    elif args.cmd == "bench":
        asyncio.run(_bench(args, labs))


async def _ask(args, labs):
    import sys
    from .router import TelephoneRouter

    z = lab_by_key(labs, args.z) if args.z else random.SystemRandom().choice(labs)

    def trace(ev):
        if args.trace and ev.kind in ("circuit", "enter", "leave"):
            if ev.kind == "circuit":
                print("circuit:", " -> ".join(h.key for h in ev.circuit.hops), file=sys.stderr)
            else:
                tag = "persona" if ev.virtual else "call"
                print(f"  [{ev.phase:7}] {ev.kind:5} {ev.node:9} ({tag}) {ev.text[:90]!r}", file=sys.stderr)

    r = await TelephoneRouter(_backend(args), labs, z, args.relay).ask(args.prompt, trace)
    print(r.answer)


async def _bench(args, labs):
    seed = args.seed if args.seed is not None else 0
    backend = _backend(args)
    if args.backend == "mock":
        attacker = MarkerAttacker(labs, random.Random(seed + 1))
    else:
        attacker = JudgeAttacker(backend, lab_by_key(labs, args.judge or "openai"), labs,
                                 random.Random(seed + 1))
    rows = await run_bench(backend, labs, args.trials, seed, attacker)
    chance = 1 / len(labs)
    print(f"black-box attribution, {len(labs)} labs, chance = {chance:.0%}, backend = {backend.name}")
    print(f"{'mode':8} {'trials':>6} {'attacker acc':>13}")
    for r in rows:
        print(f"{r.mode:8} {r.trials:>6} {r.accuracy:>12.1%}")


if __name__ == "__main__":
    main()
