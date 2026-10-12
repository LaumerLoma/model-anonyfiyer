# Attribution experiment v2: protocol

Written on 2026-10-08, after the v1 results (`RESULTS.md`) and before any v2 data was collected. v2 results go in `RESULTS_v2.md`, and any deviation from this plan is listed there. v1 stays as it is.

## Why v2

A review of v1 found three problems:

1. **The attacker was weak and short of data.** v1 used TF-IDF or embeddings with logistic regression, trained on about 320 texts per fold. The threat model allows the attacker unlimited labelled data, and published attribution work uses fine-tuned transformers (Sun et al., 2025, about 97% on 5-way frontier-model attribution). So v1's 28.5% on `nodes_3` is a lower bound of unknown slack.
2. **H5 was confounded.** `nodes_3` combines the persona forward prompt with three hops, and `para_1` has neither. The comparison can't tell which part does the work.
3. **Simple baselines were missing.** v1 had no multi-hop rewrite without persona and no fixed rewriter, and it did not vary temperature.

The threat model, models, sampling defaults, scrubbing and utility measure are unchanged from `PROTOCOL.md`. The utility measure has known weaknesses but is kept on purpose.

## Data

- **Prompts.** The 80 v1 prompts plus 80 new ones (`prompts_extra.py`, same seven task types and counts), giving 160 prompts × 5 choices of Z = **800 units**.
- **Circuits.** Drawn exactly as in v1, from the seed `unit-{p}-{z}`. For the first 400 units they are identical to v1.
- **Reuse.** All v1 calls are reused unchanged. New calls are cached in the same `calls.jsonl` under new ids.

## Conditions

Every final text is scrubbed with `scrub()`. Sampling is temperature 0.7 and max_tokens 400 unless stated otherwise.

| id | what the adversary sees | units |
|---|---|---|
| `direct` | Z's answer to the plain prompt | 800 |
| `persona_fwd` | Z's answer to the nested FORWARD prompt | 800 |
| `nodes_1..3` | `persona_fwd`, then HOP rewrites by the exit, then the middle, then the guard model | 800 |
| `plain_1..3` | **new.** `direct`, then HOP rewrites by the **same** exit, middle and guard models as `nodes_k` | 800 |
| `fixed_1` | **new.** `direct`, then one HOP rewrite by a **fixed** model for every unit: Meta (Llama 3.2 3B), chosen before data collection as the fastest of the five | 800 |
| `plain_1_hot` | **new.** As `plain_1`, but the HOP is sampled at temperature 1.2 | 800 |
| `z_mode`, `para_1` | as in v1 | 400 (v1 only) |

`nodes_k` and `plain_k` use the same relay labs for the same unit. They differ only in whether the persona forward prompt came first, which removes the v1 confound. `plain_1` is a single rewrite by a uniformly drawn lab, the same distribution as v1's `para_1`.

## Attackers

All attackers use 5-fold cross-validation **grouped by prompt** over the 160 prompts, so every fold trains on 128 prompts (640 texts).

- **A1: TF-IDF + logistic regression.** As in v1.
- **A2: nomic embeddings + logistic regression.** As in v1.
- **A3: fine-tuned classifier (new).** ModernBERT-base with a 5-way classification head, fine-tuned for each fold. Its hyperparameters are fixed in advance, with no tuning on test folds and no early stopping:
  - learning rate 3e-5, weight decay 0.01, 4 epochs, batch size 16
  - max length 256 tokens, linear warmup over the first 10%, seed 0
- **Static and adaptive.** As in v1. The static attacker trains on `direct` only.
- **Learning curve.** A1 on `direct`, `nodes_3` and `plain_3`, and A3 on `nodes_3`, trained on 25%, 50% and 100% of each training fold (subsampled by prompt).

*Amendment, 2026-10-08, made before any v2 analysis was run.* A timing test showed that A3 at 384 tokens with learning curves on three conditions would need about 10 hours of fine-tuning on this machine. So the maximum length dropped from 384 to 256 tokens, and the A3 learning curve covers `nodes_3` only. Relay outputs average 120–200 words, so most of them fit within 256 tokens.

The **best attacker** for a condition is the one with the highest adaptive accuracy, reported together with the others. A paired comparison of two conditions uses the same attacker on both sides: the one with the highest mean adaptive accuracy over the two. For H8 the choice is between A1 and A3, the attackers with learning curves.

## Metrics

- Accuracy with 95% bootstrap CIs over prompts (2,000 resamples).
- **Paired differences.** Comparisons between conditions use a paired bootstrap over prompts. The same units appear in both conditions, so the paired test is the right one.
- Confusion matrices.
- "→last" rate: how often a prediction names the lab that made the final rewrite, on units where that lab is not Z.
- Utility: cosine similarity to `direct` (nomic) and mean word count, as in v1. Kept as is.

## Hypotheses (fixed before data collection)

A hypothesis is supported only if its CI meets the stated condition. Every difference below is a paired bootstrap over prompts.

- **H7.** The fine-tuned attacker is stronger: on `nodes_3`, A3 − A1 > 0.
- **H8.** The attacker is not saturated: on `nodes_3`, the best attacker trained on 100% of a fold beats the same attacker on 50% (difference > 0).
- **H9.** The protection comes from the hops, not the persona: for each k in 1–3, the CI of `nodes_k` − `plain_k` contains 0 (best attacker).
- **H10.** More hops help without the persona as well: `plain_3` − `plain_1` < 0 (best attacker).
- **H11.** A fixed rewriter beats a random one: `fixed_1` − `plain_1` < 0 (best attacker).
- **H12.** Higher temperature helps: `plain_1_hot` − `plain_1` < 0 (best attacker).
- **H13.** Some signal survives even against the strongest attacker: `nodes_3` best-attacker accuracy has a lower CI bound above 20%.
