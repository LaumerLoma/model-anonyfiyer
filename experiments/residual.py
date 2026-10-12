"""Where does the remaining attribution signal after three hops come from?

    uv run --group experiments python experiments/residual.py

Exploratory, post hoc analysis of the v2 data. It was NOT pre-registered in PROTOCOL_v2.md.
It needs no model calls: it reads calls.jsonl, units_v2.json and preds_v2.json.

1. Breakdown. Per-lab recall and prediction rates of the best v2 attacker, and accuracy split by
   task type and by whether Z's own lab was one of the relays.
2. Content-only attackers. Classifiers that never see the wording:
     L   length only (words, sentences, characters, paragraphs) + task type
     S   L + layout (lists, headers, bold, code blocks, line count) + refusal flag
     F   "facts": TF-IDF over numbers and mid-sentence capitalised words only
     SF  S and F together
   Same grouped 5-fold split as v2.
3. Surviving wording. How much of Z's original text (word n-grams) is still in the final text,
   compared with the originals of the other four labs for the same prompt, and whether the
   units the attacker gets right are the ones that kept more of it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from analyze_v2 import DATA, KEYS, boot_acc, boot_diff, load  # noqa: E402
from prompts_extra import ALL_PROMPTS  # noqa: E402

TARGETS = ["nodes_3", "plain_3"]
SOURCE = {"nodes": "persona_fwd", "plain": "direct"}   # what Z wrote before the hops
TASKS = sorted({t for t, _ in ALL_PROMPTS})
REFUSAL = re.compile(r"\b(I can(?:'|no)t|I(?:'m| am) (?:sorry|unable|not able)|as an AI)\b", re.I)


def best_preds(results: dict, preds: dict, cond: str) -> tuple[str, np.ndarray]:
    att = results["best"][cond]
    return att, np.array(preds[f"{att}|adaptive|{cond}|1.0"], dtype=object)


def ci(values: np.ndarray, groups: np.ndarray, rng) -> list[float]:
    """Bootstrap CI over prompts for the mean of any per-unit value."""
    return [float(x) for x in boot_acc(values.astype(float), groups, rng)]


# ---------- 1. breakdown ----------

def breakdown(units, y, groups, results, preds, rng) -> dict:
    out = {}
    task = np.array([ALL_PROMPTS[u["p"]][0] for u in units])
    for cond in TARGETS:
        att, p = best_preds(results, preds, cond)
        ok = p == y
        hops = int(cond.split("_")[1])
        relays = [[u["exit"], u["middle"], u["guard"]][:hops] for u in units]
        z_in = np.array([u["z"] in r for u, r in zip(units, relays)])
        z_last = np.array([r[-1] == u["z"] for u, r in zip(units, relays)])
        per_lab = {k: {"recall": float(ok[y == k].mean()), "ci": ci(ok[y == k], groups[y == k], rng),
                       "predicted_share": float((p == k).mean())} for k in KEYS}
        others = y != "mistral"
        r = {
            "attacker": att, "acc": float(ok.mean()),
            "per_lab": per_lab,
            "acc_without_mistral_units": float(ok[others].mean()),
            "acc_without_mistral_units_ci": ci(ok[others], groups[others], rng),
            "by_task": {t: float(ok[task == t].mean()) for t in TASKS},
            "z_in_relay": {"yes": float(ok[z_in].mean()), "no": float(ok[~z_in].mean()),
                           "n_yes": int(z_in.sum()), "n_no": int((~z_in).sum())},
            "z_was_last_rewriter": {"yes": float(ok[z_last].mean()), "no": float(ok[~z_last].mean()),
                                    "n_yes": int(z_last.sum())},
        }
        out[cond] = r
        print(f"\n[1] {cond}  (best attacker {att}, acc {ok.mean():.1%})")
        print("    lab        recall   95% CI          predicted share")
        for k, v in per_lab.items():
            print(f"    {k:10} {v['recall']:6.1%}  [{v['ci'][0]:5.1%}, {v['ci'][1]:5.1%}]   {v['predicted_share']:6.1%}")
        lo, hi = r["acc_without_mistral_units_ci"]
        print(f"    accuracy on non-Mistral units: {r['acc_without_mistral_units']:.1%} [{lo:.1%}, {hi:.1%}]")
        print("    by task: " + ", ".join(f"{t} {a:.0%}" for t, a in r["by_task"].items()))
        zi, zl = r["z_in_relay"], r["z_was_last_rewriter"]
        print(f"    Z's lab among relays: {zi['yes']:.1%} (n={zi['n_yes']})  vs not: {zi['no']:.1%} (n={zi['n_no']})")
        print(f"    Z's lab did the last rewrite: {zl['yes']:.1%} (n={zl['n_yes']})  vs not: {zl['no']:.1%}")
    return out


# ---------- 2. content-only attackers ----------

def shape_features(text: str, task: str, layout: bool) -> list[float]:
    words = text.split()
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p]
    f = [np.log1p(len(words)), np.log1p(len(text)), np.log1p(len(sents)), len(paras),
         len(words) / max(len(sents), 1)]
    if layout:
        lines = text.splitlines()
        f += [len(lines),
              sum(bool(re.match(r"\s*[-*•]\s", l)) for l in lines),
              sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines),
              sum(l.lstrip().startswith("#") for l in lines),
              text.count("**"), text.count("```"),
              float(bool(REFUSAL.search(text)))]
    return f + [float(task == t) for t in TASKS]


def fact_tokens(text: str) -> list[str]:
    """Numbers and capitalised words that do not start a sentence: a crude stand-in for 'which facts'."""
    toks = re.findall(r"\d+(?:[.,]\d+)?", text)
    for m in re.finditer(r"(?<![.!?\n]\s)(?<!^)\b([A-Z][a-zA-Z]+)\b", text):
        if m.group(1) != "I":
            toks.append(m.group(1).lower())
    return toks


def gb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          l2_regularization=1.0, random_state=0)


def facts_model():
    return make_pipeline(TfidfVectorizer(analyzer=fact_tokens, sublinear_tf=True, min_df=2),
                         LogisticRegressionCV(Cs=[0.1, 1, 10, 100], cv=3, max_iter=3000))


class Stack:
    """SF: gradient boosting on shape features plus the facts model's out-of-fold class probabilities."""

    def fit(self, X, y, groups):
        texts, S = X
        inner = np.empty((len(y), len(KEYS)))
        for tr, te in GroupKFold(n_splits=3).split(S, y, groups):
            inner[te] = facts_model().fit([texts[i] for i in tr], y[tr]).predict_proba([texts[i] for i in te])
        self.f = facts_model().fit(texts, y)
        self.g = gb().fit(np.hstack([S, inner]), y)
        return self

    def predict(self, X):
        texts, S = X
        return self.g.predict(np.hstack([S, self.f.predict_proba(texts)]))


def content_attackers(units, texts, y, groups, results, preds, rng) -> dict:
    n = len(units)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(n), y, groups))
    task = [ALL_PROMPTS[u["p"]][0] for u in units]
    out = {}
    for cond in ["direct", *TARGETS]:
        T = texts[cond]
        L = np.array([shape_features(t, k, False) for t, k in zip(T, task)])
        S = np.array([shape_features(t, k, True) for t, k in zip(T, task)])
        p = {a: np.empty(n, dtype=object) for a in ["L", "S", "F", "SF"]}
        for tr, te in folds:
            p["L"][te] = gb().fit(L[tr], y[tr]).predict(L[te])
            p["S"][te] = gb().fit(S[tr], y[tr]).predict(S[te])
            p["F"][te] = facts_model().fit([T[i] for i in tr], y[tr]).predict([T[i] for i in te])
            p["SF"][te] = Stack().fit(([T[i] for i in tr], S[tr]), y[tr], groups[tr]).predict(([T[i] for i in te], S[te]))
        att, bp = best_preds(results, preds, cond)
        ok_best = bp == y
        r = {"best": {"attacker": att, "acc": float(ok_best.mean())}}
        print(f"\n[2] {cond}: best wording attacker {att} {ok_best.mean():.1%}")
        for a, pa in p.items():
            ok = pa == y
            d, lo, hi = boot_diff(ok.astype(float), ok_best.astype(float), groups, rng)
            both = float((ok & ok_best).sum() / max(ok.sum(), 1))
            recall = {k: float(ok[y == k].mean()) for k in KEYS}
            r[a] = {"acc": float(ok.mean()), "ci": ci(ok, groups, rng), "minus_best": [float(d), float(lo), float(hi)],
                    "best_also_right_when_this_right": both, "recall": recall}
            print(f"    {a:3} {ok.mean():6.1%} [{r[a]['ci'][0]:5.1%}, {r[a]['ci'][1]:5.1%}]  "
                  f"vs best {d:+.1%} [{lo:+.1%}, {hi:+.1%}]  best also right on {both:.0%} of its hits  "
                  f"recall " + " ".join(f"{k[:4]} {v:.0%}" for k, v in recall.items()))
        out[cond] = r
    return out


# ---------- 3. surviving wording ----------

def grams(text: str, n: int) -> set:
    w = re.findall(r"[a-z0-9']+", text.lower())
    if n == 1:
        return {x for x in w if x not in ENGLISH_STOP_WORDS}
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def containment(final: set, src: set) -> float:
    return len(final & src) / len(final) if final else 0.0


def surviving(units, texts, y, groups, results, preds, rng) -> dict:
    by_p: dict[int, list[int]] = {}
    for i, u in enumerate(units):
        by_p.setdefault(u["p"], []).append(i)
    out = {}
    for chain in ["nodes", "plain"]:
        src = texts[SOURCE[chain]]
        for n in (1, 3):
            sg = [grams(t, n) for t in src]
            for hop in (1, 2, 3):
                cond = f"{chain}_{hop}"
                own = np.array([containment(grams(t, n), sg[i]) for i, t in enumerate(texts[cond])])
                oth = np.array([np.mean([containment(grams(texts[cond][i], n), sg[j])
                                         for j in by_p[units[i]["p"]] if j != i]) for i in range(len(units))])
                exc = own - oth
                r = {"own": float(own.mean()), "others": float(oth.mean()),
                     "excess": float(exc.mean()), "excess_ci": ci(exc, groups, rng),
                     "excess_by_lab": {k: float(exc[y == k].mean()) for k in KEYS}}
                # "trace back": would the source be found by picking the original it overlaps most?
                trace = []
                for i in range(len(units)):
                    cand = by_p[units[i]["p"]]
                    g = grams(texts[cond][i], n)
                    trace.append(max(cand, key=lambda j: containment(g, sg[j])) == i)
                r["trace_back_acc"] = float(np.mean(trace))
                if cond in TARGETS:
                    att, bp = best_preds(results, preds, cond)
                    ok = bp == y
                    q = np.quantile(exc, [0.25, 0.5, 0.75])
                    bins = np.digitize(exc, q)
                    r["excess_when_right"] = float(exc[ok].mean())
                    r["excess_when_wrong"] = float(exc[~ok].mean())
                    r["right_minus_wrong_ci"] = diff_means_ci(exc, ok, groups, rng)
                    r["acc_by_excess_quartile"] = [float(ok[bins == b].mean()) for b in range(4)]
                out[f"{cond}|n={n}"] = r
    print("\n[3] share of the final text's word n-grams found in Z's original (own) vs other labs' originals")
    print("    condition  n  own     others  excess  [95% CI]          trace-back")
    for k, r in out.items():
        cond, n = k.split("|")
        print(f"    {cond:9} {n[2:]:>2}  {r['own']:6.1%}  {r['others']:6.1%}  {r['excess']:+6.1%} "
              f"[{r['excess_ci'][0]:+5.1%}, {r['excess_ci'][1]:+5.1%}]   {r['trace_back_acc']:5.1%}")
    for k, r in out.items():
        if "excess_when_right" in r:
            lo, hi = r["right_minus_wrong_ci"]
            print(f"    {k}: excess when attacker right {r['excess_when_right']:+.1%}, wrong {r['excess_when_wrong']:+.1%}"
                  f" (diff CI [{lo:+.1%}, {hi:+.1%}]); acc by excess quartile "
                  + " ".join(f"{a:.0%}" for a in r["acc_by_excess_quartile"])
                  + "; excess by lab " + " ".join(f"{l[:4]} {v:+.1%}" for l, v in r["excess_by_lab"].items()))
    return out


def diff_means_ci(v: np.ndarray, mask: np.ndarray, groups: np.ndarray, rng, n_boot: int = 2000) -> list[float]:
    """Bootstrap over prompts of mean(v[mask]) - mean(v[~mask])."""
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(n_boot):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        m = mask[s]
        if m.any() and (~m).any():
            d.append(v[s][m].mean() - v[s][~m].mean())
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def main():
    units, texts, _ = load()
    y = np.array([u["z"] for u in units])
    groups = np.array([u["p"] for u in units])
    results = json.loads((DATA / "results_v2.json").read_text())
    preds = json.loads((DATA / "preds_v2.json").read_text())
    rng = np.random.default_rng(0)
    out = {
        "note": "Exploratory post hoc analysis, not pre-registered.",
        "breakdown": breakdown(units, y, groups, results, preds, rng),
        "content_attackers": content_attackers(units, texts, y, groups, results, preds, rng),
        "surviving_wording": surviving(units, texts, y, groups, results, preds, rng),
    }
    (DATA / "residual.json").write_text(json.dumps(out, indent=1))
    print("\nwrote data/residual.json")


if __name__ == "__main__":
    sys.exit(main())
