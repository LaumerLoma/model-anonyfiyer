"""Analyse the v2 data as specified in PROTOCOL_v2.md.

    uv run --group experiments python experiments/analyze_v2.py

Fine-tuning (attacker A3) is the slow part. Each fold's predictions are cached in
data/ft_cache.json, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from analyze import emb_model, embed, tfidf_model  # noqa: E402

from telephone_onion.nodes import LOCAL_LABS  # noqa: E402
from telephone_onion.scrub import scrub  # noqa: E402

DATA = Path(__file__).parent / "data"
CONDITIONS = ["direct", "persona_fwd", "nodes_1", "nodes_2", "nodes_3",
              "plain_1", "plain_2", "plain_3", "fixed_1", "plain_1_hot"]
KEYS = [l.key for l in LOCAL_LABS]
FIXED = "meta"
FT_MODEL = "answerdotai/ModernBERT-base"
FT = {"lr": 3e-5, "weight_decay": 0.01, "epochs": 4, "batch": 16, "max_len": 256, "warmup": 0.1, "seed": 0}
CURVE_CONDS, CURVE_FRACS = ["direct", "nodes_3", "plain_3"], [0.25, 0.5, 1.0]
N_BOOT = 2000


def last_lab(cond: str, u: dict) -> str | None:
    """The lab that produced the final rewrite, or None if Z wrote the final text."""
    if cond == "fixed_1":
        return FIXED
    if cond.startswith(("nodes_", "plain_")):
        return [u["exit"], u["middle"], u["guard"]][int(cond.split("_")[1]) - 1]
    return None


def load():
    units = json.loads((DATA / "units_v2.json").read_text())
    calls = {r["id"]: r["output"] for r in map(json.loads, (DATA / "calls.jsonl").read_text().splitlines())}
    texts = {c: [scrub(calls[f"{c}/{u['p']}-{u['z']}"], LOCAL_LABS) for u in units] for c in CONDITIONS}
    d2 = {u["p"]: scrub(calls[f"direct2/{u['p']}-{u['z']}"], LOCAL_LABS)
          for u in units if f"direct2/{u['p']}-{u['z']}" in calls}
    return units, texts, d2


# ---------- A3: fine-tuned classifier ----------

class FineTuner:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.Model = AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(FT_MODEL)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _batches(self, texts, labels, shuffle, gen):
        idx = list(range(len(texts)))
        if shuffle:
            gen.shuffle(idx)
        for i in range(0, len(idx), FT["batch"]):
            b = idx[i:i + FT["batch"]]
            enc = self.tok([texts[j] for j in b], truncation=True, max_length=FT["max_len"],
                           padding=True, return_tensors="pt").to(self.device)
            yield enc, (self.torch.tensor([labels[j] for j in b], device=self.device) if labels else None)

    def fit_predict(self, train_texts, train_y, test_sets: dict[str, list[str]]) -> dict[str, list[str]]:
        torch = self.torch
        from transformers import get_linear_schedule_with_warmup
        torch.manual_seed(FT["seed"])
        gen = random.Random(FT["seed"])
        model = self.Model.from_pretrained(FT_MODEL, num_labels=len(KEYS)).to(self.device)
        y = [KEYS.index(v) for v in train_y]
        steps = FT["epochs"] * math.ceil(len(y) / FT["batch"])
        opt = torch.optim.AdamW(model.parameters(), lr=FT["lr"], weight_decay=FT["weight_decay"])
        sched = get_linear_schedule_with_warmup(opt, int(FT["warmup"] * steps), steps)
        model.train()
        for _ in range(FT["epochs"]):
            for enc, lab in self._batches(train_texts, y, True, gen):
                loss = model(**enc, labels=lab).loss
                loss.backward()
                opt.step(), sched.step(), opt.zero_grad()
        model.eval()
        preds = {}
        with torch.no_grad():
            for name, texts in test_sets.items():
                out = []
                for enc, _ in self._batches(texts, None, False, gen):
                    out += model(**enc).logits.argmax(-1).tolist()
                preds[name] = [KEYS[i] for i in out]
        del model
        if self.device == "mps":
            torch.mps.empty_cache()
        return preds


# ---------- evaluation ----------

def boot_acc(ok: np.ndarray, groups: np.ndarray, rng) -> tuple[float, float]:
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    accs = [ok[np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])].mean() for _ in range(N_BOOT)]
    return tuple(np.percentile(accs, [2.5, 97.5]))


def boot_diff(ok_a: np.ndarray, ok_b: np.ndarray, groups: np.ndarray, rng) -> tuple[float, float, float]:
    """Paired bootstrap over prompts of accuracy(a) - accuracy(b)."""
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(N_BOOT):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        d.append(ok_a[s].mean() - ok_b[s].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return ok_a.mean() - ok_b.mean(), lo, hi


def main():
    t0 = time.time()
    units, texts, d2 = load()
    y = np.array([u["z"] for u in units])
    groups = np.array([u["p"] for u in units])
    n = len(units)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(n), y, groups))
    print(f"{n} units, {len(set(groups))} prompts, chance = {1 / len(KEYS):.0%}")

    all_t = [t for c in CONDITIONS for t in texts[c]] + list(d2.values())
    E = embed(all_t)
    emb = {c: np.stack([E[t] for t in texts[c]]) for c in CONDITIONS}

    # preds[(attacker, mode, cond, frac)] = array of predicted labels per unit
    preds: dict[tuple, np.ndarray] = {}

    def run_sklearn(att, make, X):
        for frac in ([1.0] if att == "A2" else CURVE_FRACS):
            conds = CONDITIONS if frac == 1.0 else CURVE_CONDS
            for c in conds:
                p_ad = np.empty(n, dtype=object)
                p_st = {cc: np.empty(n, dtype=object) for cc in CONDITIONS} if c == "direct" and frac == 1.0 else None
                for fi, (tr, te) in enumerate(folds):
                    tr = subsample(tr, frac, fi)
                    m = make().fit(_rows(X[c], tr), y[tr])
                    p_ad[te] = m.predict(_rows(X[c], te))
                    if p_st is not None:
                        for cc in CONDITIONS:
                            p_st[cc][te] = m.predict(_rows(X[cc], te))
                preds[(att, "adaptive", c, frac)] = p_ad
                if p_st is not None:
                    for cc in CONDITIONS:
                        preds[(att, "static", cc, 1.0)] = p_st[cc]
            print(f"  {att} frac={frac} done ({time.time() - t0:.0f}s)", flush=True)

    def subsample(tr, frac, fi):
        if frac == 1.0:
            return tr
        ps = sorted(set(groups[tr]))
        keep = set(random.Random(f"curve-{fi}-{frac}").sample(ps, round(frac * len(ps))))
        return np.array([i for i in tr if groups[i] in keep])

    print("A1 (TF-IDF) and A2 (embeddings)")
    run_sklearn("A1", tfidf_model, texts)
    run_sklearn("A2", emb_model, emb)

    print("A3 (fine-tuned ModernBERT)")
    cache_f = DATA / "ft_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    ft = FineTuner()
    jobs = [(c, 1.0) for c in CONDITIONS] + [("nodes_3", f) for f in CURVE_FRACS if f < 1.0]
    for c, frac in jobs:
        p_ad = np.empty(n, dtype=object)
        p_st = {cc: np.empty(n, dtype=object) for cc in CONDITIONS} if c == "direct" and frac == 1.0 else None
        for fi, (tr, te) in enumerate(folds):
            key = f"{c}|{frac}|{fi}"
            if key not in cache:
                trs = subsample(tr, frac, fi)
                tests = {c: [texts[c][i] for i in te]}
                if p_st is not None:
                    tests = {cc: [texts[cc][i] for i in te] for cc in CONDITIONS}
                t1 = time.time()
                cache[key] = ft.fit_predict([texts[c][i] for i in trs], list(y[trs]), tests)
                cache_f.write_text(json.dumps(cache))
                print(f"  A3 {c} frac={frac} fold {fi}: {time.time() - t1:.0f}s", flush=True)
            p_ad[te] = cache[key][c]
            if p_st is not None:
                for cc in CONDITIONS:
                    p_st[cc][te] = cache[key][cc]
        preds[("A3", "adaptive", c, frac)] = p_ad
        if p_st is not None:
            for cc in CONDITIONS:
                preds[("A3", "static", cc, 1.0)] = p_st[cc]

    # ---------- metrics ----------
    rng = np.random.default_rng(0)
    ok = {k: (v == y).astype(float) for k, v in preds.items()}
    rows = []
    print(f"\n{'condition':12} {'attacker':12} {'acc':>6} {'95% CI':>15} {'→last':>6}")
    for c in CONDITIONS:
        for att in ["A1", "A2", "A3"]:
            for mode in ["adaptive", "static"]:
                if (att, mode, c, 1.0) not in preds or (c == "direct" and mode == "static"):
                    continue
                o = ok[(att, mode, c, 1.0)]
                lo, hi = boot_acc(o, groups, rng)
                last = np.array([last_lab(c, u) for u in units], dtype=object)
                m = np.array([l is not None and l != z for l, z in zip(last, y)])
                to_last = float((preds[(att, mode, c, 1.0)][m] == last[m]).mean()) if m.any() else None
                rows.append({"condition": c, "attacker": att, "mode": mode, "acc": o.mean(), "ci": [lo, hi],
                             "to_last": to_last})
                tl = f"{to_last:.0%}" if to_last is not None else ""
                print(f"{c:12} {att + '/' + mode:12} {o.mean():6.1%}  [{lo:5.1%}, {hi:5.1%}] {tl:>6}", flush=True)
        print()

    best = {c: max(["A1", "A2", "A3"], key=lambda a: ok[(a, "adaptive", c, 1.0)].mean()) for c in CONDITIONS}
    print("best attacker per condition:", best)

    curves = []
    for att in ["A1", "A3"]:
        for c in (CURVE_CONDS if att == "A1" else ["nodes_3"]):
            for f in CURVE_FRACS:
                o = ok[(att, "adaptive", c, f)]
                lo, hi = boot_acc(o, groups, rng)
                curves.append({"attacker": att, "condition": c, "frac": f, "acc": o.mean(), "ci": [lo, hi]})

    def best_ok(c, frac=1.0):
        return ok[(best[c], "adaptive", c, frac)]

    def paired(a, b):
        """Same attacker on both sides: the one with the highest mean adaptive accuracy over a and b."""
        att = max(["A1", "A2", "A3"], key=lambda t: ok[(t, "adaptive", a, 1.0)].mean() + ok[(t, "adaptive", b, 1.0)].mean())
        d, lo, hi = boot_diff(ok[(att, "adaptive", a, 1.0)], ok[(att, "adaptive", b, 1.0)], groups, rng)
        return {"test": f"{a} - {b} ({att})", "diff": d, "ci": [lo, hi]}

    def diff(a, b, test):
        d, lo, hi = boot_diff(a, b, groups, rng)
        return {"test": test, "diff": d, "ci": [lo, hi]}

    curve_att = max(["A1", "A3"], key=lambda t: ok[(t, "adaptive", "nodes_3", 1.0)].mean())
    hyp = {
        "H7": diff(ok[("A3", "adaptive", "nodes_3", 1.0)], ok[("A1", "adaptive", "nodes_3", 1.0)], "nodes_3: A3 - A1 > 0"),
        "H8": diff(ok[(curve_att, "adaptive", "nodes_3", 1.0)], ok[(curve_att, "adaptive", "nodes_3", 0.5)],
                   f"nodes_3 ({curve_att}): 100% - 50% > 0"),
        **{f"H9_{k}": paired(f"nodes_{k}", f"plain_{k}") for k in (1, 2, 3)},
        "H10": paired("plain_3", "plain_1"),
        "H11": paired("fixed_1", "plain_1"),
        "H12": paired("plain_1_hot", "plain_1"),
    }
    lo13, hi13 = boot_acc(best_ok("nodes_3"), groups, rng)
    hyp["H13"] = {"test": "nodes_3 best attacker lower CI > 20%", "acc": best_ok("nodes_3").mean(), "ci": [lo13, hi13]}
    verdict = {
        "H7": hyp["H7"]["ci"][0] > 0, "H8": hyp["H8"]["ci"][0] > 0,
        **{f"H9_{k}": hyp[f"H9_{k}"]["ci"][0] <= 0 <= hyp[f"H9_{k}"]["ci"][1] for k in (1, 2, 3)},
        "H10": hyp["H10"]["ci"][1] < 0, "H11": hyp["H11"]["ci"][1] < 0, "H12": hyp["H12"]["ci"][1] < 0,
        "H13": lo13 > 0.2,
    }
    print("\nhypotheses (paired bootstrap over prompts):")
    for h, r in hyp.items():
        est = r.get("diff", r.get("acc"))
        print(f"  {h:6} {r['test']:44} {est:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  "
              f"{'supported' if verdict[h] else 'not supported'}")

    def cos(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    util = {c: {"cos": float(np.mean([cos(emb[c][i], emb["direct"][i]) for i in range(n)])),
                "words": float(np.mean([len(t.split()) for t in texts[c]]))} for c in CONDITIONS}
    by_p = {}
    for i, u in enumerate(units):
        by_p.setdefault(u["p"], []).append(i)
    floor = float(np.mean([cos(emb["direct"][a], emb["direct"][b])
                           for idx in by_p.values() for a in idx for b in idx if a < b]))
    ceil = float(np.mean([cos(E[d2[u["p"]]], emb["direct"][i]) for i, u in enumerate(units)
                          if u["p"] in d2 and KEYS.index(u["z"]) == u["p"] % len(KEYS)]))
    print("\nutility: cosine to direct, words")
    for c, v in util.items():
        print(f"  {c:12} {v['cos']:.3f}  {v['words']:6.0f}")
    print(f"  floor {floor:.3f}  ceiling {ceil:.3f}")

    from sklearn.metrics import confusion_matrix
    conf = {c: confusion_matrix(y, preds[(best[c], "adaptive", c, 1.0)].astype(str), labels=KEYS).tolist()
            for c in ["direct", "nodes_3", "plain_3", "fixed_1"]}
    per_unit = {f"{a}|{m}|{c}|{f}": v.tolist() for (a, m, c, f), v in preds.items()}
    out = {"n_units": n, "n_prompts": len(set(groups)), "labels": KEYS, "rows": rows, "best": best,
           "curves": curves, "hypotheses": hyp, "verdicts": verdict, "utility": util, "floor": floor,
           "ceiling": ceil, "confusion": conf, "ft": {"model": FT_MODEL, **FT}}
    (DATA / "results_v2.json").write_text(json.dumps(out, indent=1, default=float))
    (DATA / "preds_v2.json").write_text(json.dumps(per_unit))
    print(f"\nwrote results_v2.json ({time.time() - t0:.0f}s)")


def _rows(X, idx):
    return [X[i] for i in idx] if isinstance(X, list) else X[idx]


if __name__ == "__main__":
    sys.exit(main())
