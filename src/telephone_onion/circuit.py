"""Random circuit selection: guard -> middle -> exit, like a Tor circuit.

All three hops are drawn uniformly *without replacement* from every lab,
including Z's own lab. Excluding Z's lab would leak information: an observer
who sees the guard's voice in the output could rule that lab out as the origin.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .nodes import Lab

ROLES = ("guard", "middle", "exit")


@dataclass(frozen=True)
class Circuit:
    guard: Lab
    middle: Lab
    exit: Lab

    @property
    def hops(self) -> tuple[Lab, Lab, Lab]:
        return (self.guard, self.middle, self.exit)

    def role_of(self, lab: Lab) -> str | None:
        for role, hop in zip(ROLES, self.hops):
            if hop.key == lab.key:
                return role
        return None


def build_circuit(labs: tuple[Lab, ...], rng: random.Random | None = None) -> Circuit:
    if len(labs) < 3:
        raise ValueError("need at least 3 labs to build a circuit")
    rng = rng or random.SystemRandom()
    guard, middle, exit_ = rng.sample(list(labs), 3)
    return Circuit(guard, middle, exit_)
