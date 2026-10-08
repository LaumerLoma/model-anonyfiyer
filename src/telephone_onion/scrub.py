"""Last-line defence: drop sentences that self-identify, then blank out any lab or model names."""

from __future__ import annotations

import re

from .nodes import Lab

_SENTENCE = re.compile(r"(?<=[.!?])[ \t]+")
_SELF_ID = r"(\bas an? (ai|model|language model|assistant)\b|\bi(?:'m| am) an? (ai|model|large language model|language model)\b|\b(created|developed|made|trained|built) by\b)"


def scrub(text: str, labs: tuple[Lab, ...]) -> str:
    names = sorted({a for lab in labs for a in (lab.name, *lab.aliases)}, key=len, reverse=True)
    alt = "|".join(re.escape(n) for n in names)
    self_id = re.compile(_SELF_ID + rf"|^\s*as ({alt})\b", re.IGNORECASE)
    name_re = re.compile(rf"\b({alt})(?:'s)?\b", re.IGNORECASE)

    lines = []
    for line in text.splitlines():  # line by line, so markdown structure survives
        kept = [s for s in _SENTENCE.split(line) if not self_id.search(s)]
        lines.append(name_re.sub("a model", " ".join(kept)))
    return "\n".join(lines).strip()
