"""Model backends. One backend serves every lab; the router picks which lab runs a call.

* ``RouterBackend`` talks to any OpenAI-compatible multi-provider router
  (OpenRouter, Vercel AI Gateway) so all five labs need only one key.
* ``MockBackend`` simulates five labs offline, each with a stylistic
  fingerprint, so the whole pipeline and the attribution test run without keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from .nodes import Lab


class Backend(Protocol):
    name: str

    async def complete(self, lab: Lab, prompt: str, meta: dict) -> str: ...


# --------------------------------------------------------------------------- real

ROUTERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gateway": ("https://ai-gateway.vercel.sh/v1", "AI_GATEWAY_API_KEY"),
    "lmstudio": ("http://localhost:1234/v1", None),  # local, no key
}


class RouterBackend:
    def __init__(self, router: str, timeout: float = 120.0):
        base_url, env = ROUTERS[router]
        key = os.environ.get(env) if env else "local"
        if not key:
            raise RuntimeError(f"{env} is not set (needed for --backend {router})")
        self.name = router
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    async def complete(self, lab: Lab, prompt: str, meta: dict) -> str:
        resp = await self._client.post(
            "/chat/completions",
            json={"model": lab.model, "messages": [{"role": "user", "content": prompt}]},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- mock


@dataclass(frozen=True)
class Style:
    opener: str
    tic: str
    closer: str
    self_ref: str


MOCK_STYLES: dict[str, Style] = {
    "anthropic": Style("That's a thoughtful question.", "It's worth being careful here.",
                       "I hope that's helpful, and I'm happy to dig deeper.",
                       "As an AI made by Anthropic, I try to be precise."),
    "openai": Style("Great question!", "Here's the short version.",
                    "Let me know if you'd like a step-by-step breakdown.",
                    "As ChatGPT, I can summarize this."),
    "google": Style("Here's a breakdown.", "This is a nuanced topic.",
                    "You can explore more on this topic using reputable sources.",
                    "As a model trained by Google, I'll keep it factual."),
    "xai": Style("Alright, let's cut to it.", "Spicy take incoming.",
                 "No fluff, that's the deal.",
                 "As Grok, I'll be blunt."),
    "deepseek": Style("Okay, let me think about this.", "Let me reason step by step.",
                      "Hope this clarifies things!",
                      "As DeepSeek, I reason carefully."),
}

_FACT_TEMPLATES = (
    "The core of {t} is {w}.",
    "Most confusion about {t} comes from {w}.",
    "A practical rule for {t}: start with {w}.",
    "Experts disagree on {t} mainly over {w}.",
    "Historically, {t} changed once {w} became cheap.",
)
_WORDS = ("feedback loops", "incentives", "measurement", "latency", "trust",
          "scale", "tooling", "edge cases", "cost", "defaults")


def mock_facts(prompt: str, n: int = 3) -> list[str]:
    """Deterministic 'content' for a prompt, so tests can check facts survive the relay."""
    seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    topic = " ".join(re.findall(r"[A-Za-z]+", prompt)[-3:]).lower() or "this"
    return [rng.choice(_FACT_TEMPLATES).format(t=topic, w=rng.choice(_WORDS)) for _ in range(n)]


def style_markers() -> dict[str, list[str]]:
    return {k: [s.opener, s.tic, s.closer, s.self_ref] for k, s in MOCK_STYLES.items()}


def _strip_markers(text: str) -> list[str]:
    """Return only content sentences: drop every known style marker of every lab."""
    for markers in style_markers().values():
        for m in markers:
            text = text.replace(m, " ")
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip() and p.strip() not in {".", "a model"}]


class MockBackend:
    """Simulated labs.

    A lab answering in its own voice emits all of its markers. A lab asked to
    *imitate* another persona emits the persona's markers, but each of its own
    markers still leaks with probability ``leak`` (imperfect imitation).
    """

    name = "mock"

    def __init__(self, seed: int | None = None, leak: float = 0.35, latency: float = 0.0):
        self.rng = random.Random(seed)
        self.leak = leak
        self.latency = latency
        self.calls: list[tuple[str, str, dict]] = []  # (lab key, prompt, meta) for inspection

    async def complete(self, lab: Lab, prompt: str, meta: dict) -> str:
        self.calls.append((lab.key, prompt, dict(meta)))
        if self.latency:
            await asyncio.sleep(self.latency * (0.5 + self.rng.random()))
        own = MOCK_STYLES[lab.key]
        if "source_text" in meta:  # rewrite / relay request
            facts = _strip_markers(meta["source_text"])
        else:
            facts = mock_facts(meta["user_prompt"])
        persona_key = meta.get("persona")
        if persona_key and persona_key != lab.key:
            voice = MOCK_STYLES[persona_key]
            out = [voice.opener, *facts, voice.closer]
            if self.rng.random() < self.leak:
                out.insert(1, own.tic)
            if self.rng.random() < self.leak:
                out.append(own.closer)
            if self.rng.random() < self.leak / 2:
                out.insert(0, own.self_ref)
        else:
            out = [own.self_ref, own.opener, own.tic, *facts, own.closer]
        return " ".join(out)
