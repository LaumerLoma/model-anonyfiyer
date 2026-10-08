"""The lab nodes that make up the relay network."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lab:
    key: str            # short id, e.g. "anthropic"
    name: str           # display / persona name used inside prompts
    model: str          # model slug for the OpenAI-compatible router
    aliases: tuple[str, ...] = field(default_factory=tuple)  # names the scrubber removes


# Model slugs are OpenRouter-style defaults; override them with --model KEY=SLUG.
DEFAULT_LABS: tuple[Lab, ...] = (
    Lab("anthropic", "Anthropic", "anthropic/claude-sonnet-5.5",
        ("Anthropic", "Claude")),
    Lab("openai", "OpenAI", "openai/gpt-5",
        ("OpenAI", "ChatGPT", "GPT-5", "GPT-4o", "GPT-4")),
    Lab("google", "Google DeepMind", "google/gemini-2.5-pro",
        ("Google DeepMind", "DeepMind", "Gemini", "Bard")),
    Lab("xai", "xAI", "x-ai/grok-4", ("xAI", "Grok")),
    Lab("deepseek", "DeepSeek", "deepseek/deepseek-chat", ("DeepSeek",)),
)


def lab_by_key(labs: tuple[Lab, ...], key: str) -> Lab:
    for lab in labs:
        if lab.key == key:
            return lab
    raise KeyError(f"unknown lab {key!r}; choose from {[l.key for l in labs]}")
