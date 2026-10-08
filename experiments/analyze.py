"""Analyse collected data as specified in PROTOCOL.md.

    uv run --group experiments python experiments/analyze.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

from telephone_onion.nodes import LOCAL_LABS
from telephone_onion.scrub import scrub

DATA = Path(__file__).parent / "data"
CONDITIONS = ["direct", "persona_fwd", "z_mode", "nodes_1", "nodes_2", "nodes_3", "para_1"]
KEYS = [l.key for l in LOCAL_LABS]
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
RNG = np.random.default_rng(0)


def load():
    units = json.loads((DATA / "units.json").read_text())
    calls = {r["id"]: r["output"] for r in map(json.loads, (DATA / "calls.jsonl").read_text().splitlines())}
    texts = {c: [scrub(calls[f"{c}/{u['p']}-{u['z']}"], LOCAL_LABS) for u in units] for c in CONDITIONS}
    d2 = {u["p"]: scrub(calls[f"direct2/{u['p']}-{u['z']}"], LOCAL_LABS)
          for u in units if f"direct2/{u['p']}-{u['z']}" in calls}
    return units, texts, d2


def embed(all_texts: list[str]) -> dict[str, np.ndarray]:
    cache_f = DATA / "embeddings.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    todo = sorted({t for t in all_texts if h(t) not in cache})
    with httpx.Client(timeout=300) as c:
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            r = c.post("http://localhost:1234/v1/embeddings",
                       json={"model": EMBED_MODEL, "input": [f"classification: {t}" for t in batch]})
            r.raise_for_status()
            for t, d in zip(batch, r.json()["data"]):
                cache[h(t)] = d["embedding"]
    cache_f.write_text(json.dumps(cache))
    return {t: np.array(cache[h(t)]) for t in all_texts}


def tfidf_model():
    feats = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    )
    return make_pipeline(feats, LogisticRegressionCV(Cs=[0.1, 1, 10, 100], cv=3, max_iter=3000))


def emb_model():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, max_iter=3000))


def oof_predict(make, X_train_all, X_test_all, y, groups):
    """Out-of-fold predictions: train on rows of train-fold prompts, predict test-fold prompts."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=5).split(X_test_all, y, groups):
        m = make()
        m.fit(_rows(X_train_all, tr), y[tr])
        pred[te] = m.predict(_rows(X_test_all, te))
    return pred


def _rows(X, idx):
    return [X[i] for i in idx] if isinstance(X, list) else X[idx]


def boot_ci(correct: np.ndarray, groups: np.ndarray, n=2000):
    ug = np.unique(groups)
    by_g = {g: correct[groups == g] for g in ug}
    accs = []
    for _ in range(n):
        s = RNG.choice(ug, size=len(ug))
        accs.append(np.concatenate([by_g[g] for g in s]).mean())
    return np.percentile(accs, [2.5, 97.5])


def main():
    units, texts, d2 = load()
    y = np.array([u["z"] for u in units])
    groups = np.array([u["p"] for u in units])
    guard = np.array([u["guard"] for u in units])
    all_t = [t for c in CONDITIONS for t in texts[c]] + list(d2.values())
    E = embed(all_t)
    emb = {c: np.stack([E[t] for t in texts[c]]) for c in CONDITIONS}

    results = {"n_units": len(units), "n_prompts": len(set(groups)), "rows": [], "confusion": {}}
    print(f"{len(units)} units, {len(set(groups))} prompts, chance = {1/len(KEYS):.0%}\n")
    print(f"{'condition':12} {'attacker':16} {'acc':>6} {'95% CI':>15} {'→guard':>7}")
    for c in CONDITIONS:
        for feat, make, Xc, Xd in [("tfidf", tfidf_model, texts[c], texts["direct"]),
                                   ("embed", emb_model, emb[c], emb["direct"])]:
            for mode, Xtr in [("adaptive", Xc), ("static", Xd)]:
                if c == "direct" and mode == "static":
                    continue  # identical to adaptive
                pred = oof_predict(make, Xtr, Xc, y, groups)
                ok = (pred == y).astype(float)
                lo, hi = boot_ci(ok, groups)
                g = ""
                if c.startswith(("nodes", "z_mode", "persona")):
                    m = guard != y
                    g = f"{(pred[m] == guard[m]).mean():.0%}"
                results["rows"].append({"condition": c, "features": feat, "attacker": mode,
                                        "acc": ok.mean(), "ci": [lo, hi], "to_guard": g})
                print(f"{c:12} {mode + '/' + feat:16} {ok.mean():6.1%}  [{lo:5.1%}, {hi:5.1%}] {g:>7}")
                if feat == "tfidf" and mode == "adaptive" and c in ("direct", "nodes_3"):
                    results["confusion"][c] = confusion_matrix(y, pred.astype(str), labels=KEYS).tolist()
        print()

    # utility: similarity of final text to Z's direct answer for the same unit
    def cos(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    print("utility: cosine(final, Z's direct answer)   length (words)")
    util = {}
    for c in CONDITIONS[1:]:
        s = [cos(emb[c][i], emb["direct"][i]) for i in range(len(units))]
        util[c] = float(np.mean(s))
        print(f"  {c:12} {np.mean(s):.3f}   {np.mean([len(t.split()) for t in texts[c]]):6.0f}")
    by_p = {}
    for i, u in enumerate(units):
        by_p.setdefault(u["p"], []).append(i)
    floor = np.mean([cos(emb["direct"][a], emb["direct"][b])
                     for idx in by_p.values() for a in idx for b in idx if a < b])
    ceil = np.mean([cos(E[d2[u["p"]]], emb["direct"][i]) for i, u in enumerate(units)
                    if u["p"] in d2 and list(KEYS).index(u["z"]) == u["p"] % len(KEYS)])
    print(f"  {'floor':12} {floor:.3f}   (two different models, same prompt)")
    print(f"  {'ceiling':12} {ceil:.3f}   (same model resampled)")
    print(f"  {'direct':12} {'':5}   {np.mean([len(t.split()) for t in texts['direct']]):6.0f}")
    results.update(utility=util, floor=floor, ceiling=ceil, labels=KEYS)
    (DATA / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nconfusion (rows = true Z, cols = predicted), tfidf adaptive:", KEYS)
    for c, m in results["confusion"].items():
        print(c)
        for k, row in zip(KEYS, m):
            print(f"  {k:10}", row)


if __name__ == "__main__":
    sys.exit(main())
