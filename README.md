# telephone · onion

Hides which AI model wrote an answer. It mixes the game of telephone with Tor-style guard → middle → exit circuits.

```
forward   you ─▶ A ─▶ B ─▶ C ─▶ Z     one call to Z:
          "Tell me a version of what A would return if it asked B what C would say about <prompt>"
return    Z ─▶ C ─▶ B ─▶ A ─▶ you     same structure, carrying the answer back
```

* **Nodes**: five labs. For each query, A, B and C are drawn uniformly at random without replacement, and Z's own lab can be drawn too. Leaving Z's lab out would leak information, because an attacker could then rule out whichever lab voiced the guard hop.
* **Z**: the model being anonymised. The TUI hides its identity unless you press `ctrl+r`.
* **Relay modes**
  * `--relay z`: the spec as written. Z handles both the forward and the return framing (two calls), and A/B/C exist only as personas inside Z's prompt (shown as *echo in Z*).
  * `--relay nodes`: Tor-like. On the way back, exit C, then middle B, then guard A each restate the answer with their **own** model (four calls), so the final voice belongs to a random lab.
* A scrubber drops self-identifying sentences and lab or model names as a last step.

## Run

```bash
uv run telephone                       # TUI, offline mock labs
uv run telephone tui --relay nodes --z openai
uv run telephone --seed 5 ask "Why is the sky blue?" --relay nodes --trace
uv run telephone --seed 1 bench --trials 500      # black-box attribution test
uv run pytest
```

Real models use one OpenAI-compatible router key for all five labs:

```bash
export OPENROUTER_API_KEY=...   # or AI_GATEWAY_API_KEY=... with --backend gateway
uv run telephone --backend openrouter --model openai=openai/gpt-5-mini tui
uv run telephone --backend openrouter bench --trials 50 --judge openai   # LLM judge attacks
```

Check that the default model slugs in `src/telephone_onion/nodes.py` match what your router offers.

## Black-box test

An attacker sees only the final text and guesses which lab Z is. Chance is 20%.

| mode   | attacker accuracy (mock, 500 trials) |
|--------|-----|
| direct | 100% |
| z      | ~27–30% |
| nodes  | ~22% (chance) |

The mock labs are a simulation with fixed stylistic fingerprints and an imitation "leak" rate, so these numbers show that the harness works, not how real models behave. To measure real models, run `bench` with a real backend and an LLM judge.

## Limits

This hides *style*, not *content*: facts, refusals and knowledge cutoffs can still identify the source model. Each provider also sees your API traffic. The anonymity applies to whoever reads the answer, not to the providers.
