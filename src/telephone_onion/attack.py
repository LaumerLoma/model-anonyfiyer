"""Black-box attribution test.

An attacker sees only the final text and guesses which lab produced it.
The anonymiser succeeds when the attacker drops to chance (1 / number of labs).

* ``MarkerAttacker`` works with the mock backend: it looks for each lab's stylistic fingerprints.
* ``JudgeAttacker`` asks a real model to attribute the text (used with a real backend).
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass

from .backends import Backend, style_markers
from .nodes import Lab
from .router import TelephoneRouter, ask_direct

PROMPTS = (
    "Why do distributed systems need consensus protocols?",
    "Give advice for learning a second language as an adult.",
    "Explain how vaccines train the immune system.",
    "What makes a good code review?",
    "How should a small team approach on-call rotations?",
    "Why is the sky blue?",
    "What are the trade-offs of remote work?",
    "How does compound interest work?",
)


class MarkerAttacker:
    def __init__(self, labs: tuple[Lab, ...], rng: random.Random):
        self.labs = labs
        self.rng = rng
        # Match whole markers and also their tail clause, since scrubbing can cut the head off.
        self.markers = {
            k: [m for full in ms for m in {full, full.split(", ", 1)[-1]}]
            for k, ms in style_markers().items()
        }

    async def guess(self, text: str) -> str:
        scores = Counter({k: sum(text.count(m) for m in ms) for k, ms in self.markers.items()})
        best = max(scores.values())
        candidates = [k for k, s in scores.items() if s == best]
        return self.rng.choice(candidates)


class JudgeAttacker:
    def __init__(self, backend: Backend, judge: Lab, labs: tuple[Lab, ...], rng: random.Random):
        self.backend, self.judge, self.labs, self.rng = backend, judge, labs, rng

    async def guess(self, text: str) -> str:
        options = ", ".join(l.key for l in self.labs)
        prompt = (f"Which AI lab's model most likely wrote the text below? "
                  f"Answer with exactly one word from: {options}.\n\n<text>\n{text}\n</text>")
        reply = (await self.backend.complete(self.judge, prompt, {"user_prompt": prompt})).lower()
        for lab in self.labs:
            if re.search(rf"\b{re.escape(lab.key)}\b", reply):
                return lab.key
        return self.rng.choice(self.labs).key


@dataclass
class BenchRow:
    mode: str
    trials: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.trials if self.trials else 0.0


async def run_bench(backend: Backend, labs: tuple[Lab, ...], trials: int, seed: int,
                    attacker=None, modes=("direct", "z", "nodes")) -> list[BenchRow]:
    rng = random.Random(seed)
    attacker = attacker or MarkerAttacker(labs, random.Random(seed + 1))
    rows = []
    for mode in modes:
        correct = 0
        for i in range(trials):
            z = labs[i % len(labs)]  # every lab plays Z equally often
            prompt = rng.choice(PROMPTS)
            if mode == "direct":
                text = await ask_direct(backend, z, labs, prompt)
            else:
                router = TelephoneRouter(backend, labs, z, relay=mode, rng=rng)
                text = (await router.ask(prompt)).answer
            correct += (await attacker.guess(text)) == z.key
        rows.append(BenchRow(mode, trials, correct))
    return rows
