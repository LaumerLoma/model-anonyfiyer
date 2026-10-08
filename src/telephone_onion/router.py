"""The telephone/onion router.

Forward pass  (one call to Z):
    "Tell me a version of what A would return if it asked B what C would say about <prompt>"

Return pass, same structure but carrying the answer back C -> B -> A:
    relay="z"      one more call to Z: "...what A would return after B relayed C's answer <answer>"
    relay="nodes"  three real hops: exit C, then middle B, then guard A each restate the
                   answer with their own model, so the final voice is the guard's, not Z's.

Finally the text is scrubbed of explicit self-identification.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

from .backends import Backend
from .circuit import Circuit, build_circuit
from .nodes import Lab
from .scrub import scrub

Relay = Literal["z", "nodes"]

FORWARD = """Tell me a version of what {a} would return if it asked {b} what {c} would say about the following.

<prompt>
{prompt}
</prompt>

Reply only with the final answer {a} would give, in plain prose. Do not mention any AI lab, model, persona, or this framing."""

RETURN_Z = """{c} answered, {b} relayed that answer to {a}, and {a} now passes it on.

<answer>
{answer}
</answer>

Tell me a version of what {a} would return after {b} passed along {c}'s answer. Keep every fact and every concrete detail; change only the voice. Do not mention any AI lab, model, persona, or this framing."""

HOP = """You are one relay in a game of telephone. Restate the answer below in your own words.
Keep every fact, number, step and caveat. Do not add new content, do not comment on it,
and do not mention who wrote it, who you are, or that it was relayed.

<answer>
{answer}
</answer>"""


@dataclass
class Event:
    kind: Literal["circuit", "enter", "leave", "output", "error"]
    node: str = ""              # lab key, "Z", or "user"
    phase: str = ""             # "forward" | "return"
    virtual: bool = False       # True when the lab is only a persona inside Z's prompt
    text: str = ""
    circuit: Circuit | None = None


@dataclass
class Result:
    answer: str
    raw_z: str
    circuit: Circuit
    z: Lab
    relay: Relay
    hops: list[tuple[str, str]] = field(default_factory=list)  # (node key, text produced)


Listener = Callable[[Event], Awaitable[None] | None]


class TelephoneRouter:
    def __init__(self, backend: Backend, labs: tuple[Lab, ...], z: Lab,
                 relay: Relay = "z", rng: random.Random | None = None):
        self.backend = backend
        self.labs = labs
        self.z = z
        self.relay = relay
        self.rng = rng

    async def _emit(self, listener: Listener | None, ev: Event) -> None:
        if listener is None:
            return
        r = listener(ev)
        if r is not None:
            await r

    async def ask(self, prompt: str, listener: Listener | None = None,
                  circuit: Circuit | None = None) -> Result:
        c = circuit or build_circuit(self.labs, self.rng)
        A, B, C = c.hops
        await self._emit(listener, Event("circuit", circuit=c))

        # ---- forward: user -> guard -> middle -> exit -> Z (personas wrapped around the prompt)
        for lab in (A, B, C):
            await self._emit(listener, Event("enter", lab.key, "forward", virtual=True))
            await self._emit(listener, Event("leave", lab.key, "forward", virtual=True))
        fwd = FORWARD.format(a=A.name, b=B.name, c=C.name, prompt=prompt)
        await self._emit(listener, Event("enter", "Z", "forward", text=fwd))
        raw = await self.backend.complete(self.z, fwd, {"user_prompt": prompt, "persona": A.key})
        await self._emit(listener, Event("leave", "Z", "forward", text=raw))
        hops: list[tuple[str, str]] = [("Z", raw)]

        # ---- return: exit -> middle -> guard -> user
        text = raw
        if self.relay == "z":
            ret = RETURN_Z.format(a=A.name, b=B.name, c=C.name, answer=raw)
            await self._emit(listener, Event("enter", "Z", "return", text=ret))
            text = await self.backend.complete(self.z, ret, {"source_text": raw, "persona": A.key})
            await self._emit(listener, Event("leave", "Z", "return", text=text))
            hops.append(("Z", text))
            for lab in (C, B, A):
                await self._emit(listener, Event("enter", lab.key, "return", virtual=True))
                await self._emit(listener, Event("leave", lab.key, "return", virtual=True))
        else:
            for lab in (C, B, A):
                await self._emit(listener, Event("enter", lab.key, "return", text=text))
                text = await self.backend.complete(lab, HOP.format(answer=text), {"source_text": text})
                hops.append((lab.key, text))
                await self._emit(listener, Event("leave", lab.key, "return", text=text))

        answer = scrub(text, self.labs)
        await self._emit(listener, Event("output", "user", text=answer))
        return Result(answer, raw, c, self.z, self.relay, hops)


async def ask_direct(backend: Backend, z: Lab, labs: tuple[Lab, ...], prompt: str) -> str:
    """Baseline with no routing at all, only scrubbing. Used by the attribution benchmark."""
    return scrub(await backend.complete(z, prompt, {"user_prompt": prompt}), labs)
