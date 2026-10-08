# Probing a Tor-network style to anonymize a model in LLM usage

Results of the attribution experiment set out in [`PROTOCOL.md`](PROTOCOL.md), run on 2026-10-07/08 with five small local models in LM Studio. The protocol was written before data collection, and its hypotheses and analysis were not changed afterwards.

## Summary

- Without the scheme, a simple style classifier identifies the source model (Z) **87.5%** of the time. Chance is 20%.
- The full Tor-style circuit (`nodes_3`: exit → middle → guard each rewrite the answer with their own model) lowers this to **28.5%**. That is close to chance but still measurably above it.
- The design as originally specified (`z_mode`, where Z voices every hop itself) makes Z **easier** to identify (93.2%), not harder.
- On the strongest attacker, the full circuit beats a single rewrite by a random model (`para_1`, 42.8%).
- Most of the remaining signal comes from one model: Mistral is still identified 49% of the time after three hops.
- Answers shrink by about half (246 → 124 words). The embedding-based utility measure is too coarse to show how much content survives.

## Setup (as in the protocol)

- **Models.** Five labs, one small instruct model each: Meta (Llama 3.2 3B), Google (Gemma 3 4B), Alibaba (Qwen3 4B 2507), Microsoft (Phi-4-mini), Mistral (Ministral 3 3B).
- **Units.** 80 prompts × 5 choices of Z = 400 units. Circuits were drawn with a fixed seed for each unit.
- **Calls.** 2,880 calls in total, with temperature 0.7 and max_tokens 400. No call failed and no output was empty.
- **Scrubbing.** Every final text passed through `scrub()` before analysis.
- **Attackers.**
  - Adaptive: trained on the same condition it is tested on.
  - Static: trained on `direct` answers only.
  - Both use logistic regression on one of two feature sets: TF-IDF of character and word n-grams (style), or nomic-embed-text-v1.5 embeddings (meaning).
  - Evaluation is 5-fold cross-validation grouped by prompt, with 95% CIs from 2,000 bootstrap resamples over prompts.

## Attribution accuracy

Chance is 20%. "→guard" is the share of predictions that name the guard lab, counted only on units where the guard is not Z.

| condition | adaptive / tfidf | static / tfidf | adaptive / embed | static / embed | →guard (adaptive / tfidf) |
|---|---|---|---|---|---|
| `direct` | **87.5%** [84.2, 90.8] | n/a | 54.8% [50.0, 59.3] | n/a | n/a |
| `persona_fwd` | 78.8% [74.8, 82.5] | 62.7% [59.0, 66.8] | 48.2% [43.8, 52.5] | 32.8% [28.7, 36.8] | 4% |
| `z_mode` | **93.2%** [90.8, 95.8] | 74.5% [71.0, 77.8] | 74.8% [70.8, 78.5] | 51.7% [47.8, 56.0] | 2% |
| `nodes_1` | 38.0% [33.8, 42.0] | 31.2% [27.8, 35.0] | 28.7% [25.2, 32.2] | 24.5% [21.5, 27.8] | 13% |
| `nodes_2` | 30.2% [26.5, 34.0] | 28.0% [24.8, 31.5] | 30.2% [26.5, 34.2] | 22.8% [19.8, 25.5] | 12% |
| `nodes_3` | **28.5%** [24.8, 32.5] | 24.2% [21.0, 27.8] | 27.0% [23.2, 30.8] | 23.2% [20.8, 26.0] | 19% |
| `para_1` | 42.8% [38.0, 47.5] | 42.0% [38.2, 46.0] | 27.5% [24.2, 31.0] | 35.2% [31.5, 39.0] | n/a |

For the static TF-IDF attacker on `nodes_3`, the →guard rate is **35%**. An attacker that hasn't adapted to the scheme mostly recognises the last relay's voice, not Z's.

### Confusion matrices (adaptive / tfidf)

Rows are the true Z and columns the predicted lab, in the order meta, google, alibaba, microsoft, mistral. Each row sums to 80.

```
direct                              nodes_3
meta       73   3   2   1   1       meta       23  19  11  13  14
google      0  73   4   1   2       google     18  13   9  14  26
alibaba     7   5  56   0  12       alibaba    15  12  16  23  14
microsoft   1   0   0  78   1       microsoft  10  14  17  23  16
mistral     2   1   7   0  70       mistral    13  12   5  11  39
```

After three hops, Meta, Alibaba and Microsoft are recognised at 20–29%, close to chance, and Google at 16%. Mistral is still recognised 39 of 80 times. Google answers are often predicted as Mistral (26 of 80).

## Utility

Utility is the mean cosine similarity (nomic embeddings) between the final text and Z's `direct` answer for the same unit.

| condition | cosine | words |
|---|---|---|
| `direct` | n/a | 246 |
| `persona_fwd` | 0.892 | 169 |
| `z_mode` | 0.858 | 200 |
| `nodes_1` | 0.881 | 138 |
| `nodes_2` | 0.874 | 126 |
| `nodes_3` | 0.868 | 124 |
| `para_1` | 0.953 | 197 |
| *floor*: two different models' `direct` answers to the same prompt | 0.914 | |
| *ceiling*: Z's `direct` answer resampled | 0.954 | |

Every condition that uses the persona framing scores below the floor. This means the nested forward prompt moves the answer further from Z's plain answer than another model's plain answer would be. The metric can't separate "same content, different wording" from "different content", so it doesn't settle how much content survives. `para_1` matches the ceiling: a plain rewrite keeps the meaning but, as shown above, leaks more of Z's style.

## Hypotheses

| | prediction | verdict |
|---|---|---|
| H1 | `direct` adaptive accuracy ≥ 70% | **Supported** (TF-IDF 87.5% [84.2, 90.8]). The embedding attacker alone would not meet it (54.8%). |
| H2 | `z_mode` lowers accuracy but stays above chance | **Rejected.** Accuracy rises to 93.2% [90.8, 95.8], above `direct`. Having Z write both the forward and the return framing appears to concentrate its style. |
| H3 | `nodes_3` < `z_mode` | **Supported** (28.5% vs 93.2%, CIs far apart). |
| H4 | `nodes_3` adaptive still above chance | **Supported** (TF-IDF lower bound 24.8%, embedding lower bound 23.2%). |
| H5 | `para_1` ≈ `nodes_3` (CIs overlap) | **Not supported** for the TF-IDF attacker: 42.8% [38.0, 47.5] vs 28.5% [24.8, 32.5]. The CIs do overlap for the embedding attacker (27.5% vs 27.0%). Against the stronger attacker, the extra hops add real protection beyond one random rewrite. |
| H6 | utility falls with each hop | **Consistent but not tested.** The mean falls monotonically from `nodes_1` to `nodes_3` (0.881 → 0.874 → 0.868), but the drops are small and no CI was computed. |

## Interpretation

1. **The circuit works mostly by replacing the voice.** Accuracy collapses after the first independent hop (`nodes_1`, 38%) and approaches chance by `nodes_3`. The static attacker's →guard rate rises to 35%, which suggests the final text mostly carries the guard's style.
2. **Personas without other models don't anonymise.** In `z_mode` the hops exist only as personas inside Z's prompt, and Z ends up more identifiable than when answering directly. The value of the design comes from routing through *other models*, not from the nested framing.
3. **Leakage is uneven.** Some of Mistral's signal survives three rewrites, so the residual risk depends on which model is Z. That signal is likely content or structure, not surface style.
4. **Content, not style, is the remaining channel.** This is in line with the README's stated limit. Because the embedding attacker reaches the same accuracy on `nodes_3` as the TF-IDF attacker (27.0% vs 28.5%), the residual signal is probably at least partly in meaning.

## Deviations from the protocol

- None in design, prompts, models, sampling or analysis.
- The data was collected across one pilot and two runs of `collect.py`: a 10-prompt pilot, a full run that stopped partway through stage 1, and a resumed run. All 2,880 stored prompts were checked, and every one matches what the final code generates byte for byte, on the assigned model. The model slugs were fixed before the pilot started. `collect.py` only gained a tqdm progress bar between runs.
- Several stretches ran slowly, at up to about 50 s per call against a normal 4–8 s, because the machine was short on memory. This affected timing only.

## Limitations

- Small models (3–4B parameters). Frontier models may show their identity more or less strongly, and may imitate other styles better.
- Only two attacker families were tested. A fine-tuned transformer classifier or an LLM judge may extract more signal, so these accuracies are lower bounds on what an adversary can achieve.
- 80 prompts give CIs about ±4 points wide, enough to separate the conditions above but not small effects.
- Utility is measured by embedding similarity only, with no task-level check (for example, correctness on the maths and coding prompts).

## Reproduce

```bash
uv run --group experiments python experiments/collect.py   # needs LM Studio with the five models
uv run --group experiments python experiments/analyze.py   # needs text-embedding-nomic-embed-text-v1.5 loaded
```

Raw outputs are in `data/calls.jsonl`, the circuit for each unit in `data/units.json`, and the numbers above in `data/results.json`.
