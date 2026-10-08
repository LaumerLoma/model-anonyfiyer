"""Black-box tests: drive only the public API and observe what each party sees.

The mock backend's call log plays the role of a wiretap at the providers:
it records exactly which lab received which prompt.
"""

import asyncio
import random
from collections import Counter

import pytest

from telephone_onion.attack import MarkerAttacker, run_bench
from telephone_onion.backends import MockBackend, mock_facts
from telephone_onion.circuit import build_circuit
from telephone_onion.nodes import DEFAULT_LABS, lab_by_key
from telephone_onion.router import TelephoneRouter

LABS = DEFAULT_LABS
PROMPT = "Why do distributed systems need consensus protocols?"


def run(coro):
    return asyncio.run(coro)


def ask(relay, z="anthropic", seed=7):
    backend = MockBackend(seed=seed)
    events = []
    router = TelephoneRouter(backend, LABS, lab_by_key(LABS, z), relay, rng=random.Random(seed))
    result = run(router.ask(PROMPT, events.append))
    return backend, events, result


def test_circuit_hops_distinct_and_uniform_including_z_lab():
    rng = random.Random(1)
    n = 10_000
    guards = Counter()
    for _ in range(n):
        c = build_circuit(LABS, rng)
        assert len({h.key for h in c.hops}) == 3
        guards[c.guard.key] += 1
    for lab in LABS:  # every lab, Z's included, is guard ~20% of the time
        assert abs(guards[lab.key] / n - 0.2) < 0.02


def test_z_only_sees_nested_framing_in_circuit_order():
    backend, _, result = ask("z")
    lab, prompt, _ = backend.calls[0]
    a, b, c = (h.name for h in result.circuit.hops)
    assert lab == "anthropic"
    assert f"what {a} would return if it asked {b} what {c} would say" in prompt
    assert PROMPT in prompt and prompt.strip() != PROMPT


@pytest.mark.parametrize("relay,expected", [("z", 2), ("nodes", 4)])
def test_call_pattern(relay, expected):
    backend, _, result = ask(relay)
    callees = [c[0] for c in backend.calls]
    assert len(callees) == expected
    if relay == "z":
        assert callees == ["anthropic", "anthropic"]
    else:  # return path runs exit -> middle -> guard on their own models
        g, m, e = (h.key for h in result.circuit.hops)
        assert callees == ["anthropic", e, m, g]


@pytest.mark.parametrize("relay", ["z", "nodes"])
def test_facts_survive_the_telephone(relay):
    _, _, result = ask(relay)
    for fact in mock_facts(PROMPT):
        assert fact in result.answer


@pytest.mark.parametrize("relay", ["z", "nodes"])
def test_answer_names_no_lab(relay):
    for seed in range(30):
        _, _, result = ask(relay, z=LABS[seed % 5].key, seed=seed)
        low = result.answer.lower()
        for lab in LABS:
            for alias in (lab.name, *lab.aliases):
                assert alias.lower() not in low, (alias, result.answer)


@pytest.mark.parametrize("relay", ["z", "nodes"])
def test_only_circuit_labs_are_ever_present(relay):
    _, events, result = ask(relay)
    on_route = {h.key for h in result.circuit.hops}
    present = Counter((e.node, e.phase) for e in events if e.kind == "enter" and e.node != "Z")
    assert {n for n, _ in present} == on_route
    assert all(v == 1 for v in present.values())  # once out, once back


def test_attribution_drops_to_chance_with_node_relay():
    rows = {r.mode: r.accuracy for r in run(
        run_bench(MockBackend(seed=3), LABS, trials=500, seed=3,
                  attacker=MarkerAttacker(LABS, random.Random(4))))}
    assert rows["direct"] > 0.9            # unrouted output is trivially attributable
    assert rows["z"] < rows["direct"]       # persona nesting helps, but Z's habits leak
    assert abs(rows["nodes"] - 0.2) < 0.06  # real hops: attacker at chance
