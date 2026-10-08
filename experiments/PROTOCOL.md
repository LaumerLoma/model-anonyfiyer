# Attribution experiment: protocol

Written on 2026-10-07, before any real-model data was collected. Results go in `RESULTS.md`. Any deviation from this plan is listed there.

## Question

Does routing an answer through telephone circuits stop an adversary from identifying which model (Z) produced it, and how much of the answer's content survives?

## Threat model

- The adversary sees only the final answer text.
- The adversary knows the scheme (Kerckhoffs's principle), knows the five candidate models, and can generate as much labelled training data as it wants by running the scheme itself.
- Out of scope: the providers, who see all traffic; network observers; timing.

## Models (local, LM Studio, MLX/GGUF, about 3–4B parameters each)

Five labs, one small instruct model each: Meta (Llama 3.2 3B), Google (Gemma 3 4B), Alibaba (Qwen3 4B 2507), Microsoft (Phi-4-mini), Mistral (Ministral 3B).

- Each model plays Z for every prompt.
- Relays are run by the lab's own model.
- Sampling for every call: temperature 0.7, max_tokens 400.
- **Limitation, stated up front:** these are small models. Frontier models may differ in how strongly they show their identity and how well they imitate other styles.

## Prompts

80 fixed prompts across explanation, advice, coding, maths, creative writing, opinion and factual Q&A (`prompts.py`). Each (prompt, Z) pair is one unit, so there are 400 units.

## Conditions (every final text is scrubbed with `scrub()`)

| id | what the adversary sees |
|---|---|
| `direct` | Z's answer to the plain prompt |
| `persona_fwd` | Z's answer to the nested FORWARD prompt (guard/middle/exit drawn at random) |
| `z_mode` | `persona_fwd`, then Z's RETURN_Z rewrite (the design as originally specified) |
| `nodes_1` | `persona_fwd`, then a HOP rewrite by the exit lab's model |
| `nodes_2` | ...then a HOP by the middle lab's model |
| `nodes_3` | ...then a HOP by the guard lab's model (the full `--relay nodes` design) |
| `para_1` | the `direct` answer, then one HOP rewrite by a uniformly random lab (which can be Z). This is the simple baseline. |

The `nodes_k` conditions are intermediate states of the same chain, so they cost no extra calls.

## Attackers

Every attacker is evaluated with 5-fold cross-validation **grouped by prompt**: the same prompt never appears in both training and test data.

- **A-static:** trained on `direct` answers only, then tested on each condition. This is an attacker who doesn't adapt to the scheme.
- **A-adaptive:** trained and tested within each condition. This is the attacker the threat model calls for.
- Two feature sets, reported separately:
  - (a) TF-IDF of character 2–5-grams plus word 1–2-grams, feeding logistic regression. This captures style.
  - (b) nomic-embed-text-v1.5 sentence embeddings, feeding logistic regression. This captures meaning.

## Metrics

- 5-way attribution accuracy. Chance is 20%.
- 95% confidence intervals from a bootstrap over prompts (2000 resamples).
- Confusion matrices.
- For the `nodes_*` conditions, how often the attacker predicts the guard lab instead of Z. This tests whether it has learned the final voice.
- **Utility:** cosine similarity (nomic embeddings) between the final text and Z's `direct` answer for the same prompt. Two reference points:
  - floor: similarity between `direct` answers from two different models
  - noise ceiling: similarity between `direct` and a second `direct` sample from the same Z, collected for 80 units
- Mean length in words.

## Hypotheses (fixed before data collection)

- **H1.** `direct` is highly attributable: adaptive accuracy ≥ 70%.
- **H2.** `z_mode` lowers accuracy relative to `direct`, but the adaptive attacker stays clearly above chance (the lower CI bound is above 20%).
- **H3.** `nodes_3` lowers accuracy further than `z_mode`.
- **H4.** The adaptive attacker on `nodes_3` still beats chance, so the scheme leaks residual signal from Z through content.
- **H5.** `para_1` (one random rewrite, no nesting) gives roughly the same accuracy as `nodes_3` (their CIs overlap). In other words, the extra hops and the persona framing add little beyond "rewrite with a random model".
- **H6.** Utility falls with each hop.

A hypothesis counts as supported only if the CI meets the stated condition.
