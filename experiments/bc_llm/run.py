"""Stage 3: b/c decider experiments, scored with the real metric (anonymizer.evaluate_gt) on train and eval.

  E0  baseline (relevance score >= t_keep 0.5)
  E1  classifier, all features (with SapBERT)        E2  classifier, no embedding features
  E3  LLM only (P(Yes) >= theta)
  E4  classifier, all features + LLM P(Yes)          E5  classifier, no embedding features + LLM P(Yes)
  E6  cascade: E1 decides; spans with |p - theta| < delta go to the LLM

Classifiers are gradient-boosted trees on the stage-1 features, out-of-fold on train (5 folds as in
smalltalk.py / evidence.py), fit on all train for eval. theta is picked on train only (best train score over
--thetas), then eval is run once at that theta. The pipeline is not rerun: each theta replays only its
decision-dependent tail on the state saved by the features step (seconds per theta). LLM scores are read from
outputs/bc_llm/llm/, never recomputed.

python -m experiments.bc_llm.run --exp E1
python -m experiments.bc_llm.run --exp E4 --llm qwen35_4b
python -m experiments.bc_llm.run --summary          # all results as one markdown table
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .common import OUT, folds, load_json, metrics, save_json, span_cost
from .features import EMBEDDING_FEATURES, finalize_split, load_states

EXPERIMENTS = {
    "E0": "baseline: relevance score >= t_keep 0.5",
    "E1": "classifier, all features (SapBERT)",
    "E2": "classifier, no embedding features",
    "E3": "LLM only",
    "E4": "classifier, all features + LLM",
    "E5": "classifier, no embedding features + LLM",
    "E6": "cascade: E1, uncertain band -> LLM",
}
NEEDS_LLM = {"E3", "E4", "E5", "E6"}
DEFAULT_THETAS = {"E3": "0.3,0.4,0.5,0.6,0.7,0.8,0.9"}
TYPE_KEYS = ("b:CONDITION", "b:FINDING", "b:PROCEDURE", "b:MEDICATION", "b:LAB",
             "c:CONDITION", "c:FINDING", "c:PROCEDURE", "c:MEDICATION", "c:LAB")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _matrix(rows, names, llm):
    X = np.array([[r["f"][n] for n in names] for r in rows], dtype=float)
    if llm is not None:
        X = np.c_[X, _logit([llm[r["key"]] for r in rows])]
    return X


def classifier_probs(tr, ev, use_embeddings: bool, llm_tr=None, llm_ev=None):
    """OOF probabilities on train, full-train-model probabilities on eval."""
    names = sorted(n for n in tr[0]["f"] if use_embeddings or n not in EMBEDDING_FEATURES)
    # early_stopping off: sklearn turns it on only above 10k samples, i.e. for the eval model but not the folds.
    model = lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                                   l2_regularization=1.0, early_stopping=False, random_state=0)
    fold = folds()
    lab = [r for r in tr if r["y"] is not None]
    X_tr, X_lab = _matrix(tr, names, llm_tr), _matrix(lab, names, llm_tr)
    y_lab = np.array([r["y"] for r in lab])
    f_tr, f_lab = np.array([fold[r["eid"]] for r in tr]), np.array([fold[r["eid"]] for r in lab])
    p_tr = np.zeros(len(tr))
    for k in sorted(set(fold.values())):
        m = model().fit(X_lab[f_lab != k], y_lab[f_lab != k])
        p_tr[f_tr == k] = m.predict_proba(X_tr[f_tr == k])[:, 1]
    p_ev = model().fit(X_lab, y_lab).predict_proba(_matrix(ev, names, llm_ev))[:, 1]
    return dict(zip((r["key"] for r in tr), p_tr)), dict(zip((r["key"] for r in ev), p_ev)), names


def _labelled(rows, p):
    lab = [r for r in rows if r["y"] is not None]
    return np.array([r["y"] for r in lab]), np.array([p[r["key"]] for r in lab]), lab


def cascade_decisions(tr, ev, p_tr, p_ev, llm_tr, llm_ev, theta, delta):
    """E6 at one theta: tau for the LLM inside the band is fit on train labelled spans (span cost)."""
    y, pc, lab = _labelled(tr, p_tr)
    pl = np.array([llm_tr[r["key"]] for r in lab])
    band = np.abs(pc - theta) < delta
    tau = min(np.linspace(0.02, 0.98, 49),
              key=lambda t: span_cost(y, np.where(band, pl >= t, pc >= theta)))

    def decide(rows, p, llm):
        return {r["key"]: bool(llm[r["key"]] >= tau) if abs(p[r["key"]] - theta) < delta else bool(p[r["key"]] >= theta)
                for r in rows}
    k_tr, k_ev = decide(tr, p_tr, llm_tr), decide(ev, p_ev, llm_ev)
    calls = float(np.mean([abs(p_ev[r["key"]] - theta) < delta for r in ev]))
    return k_tr, k_ev, {"tau": round(float(tau), 2), "llm_calls_eval": round(calls, 3)}


def _row(res):
    return {k: res[k] for k in ("a_removed", "b_removed", "privacy", "c_kept", "score")} | {
        "by_type": {k: res["by_type"].get(k) for k in TYPE_KEYS}}


def _span_stats(rows, p, keep):
    y, pp, lab = _labelled(rows, p) if p is not None else (None, None, None)
    if lab is None:
        lab = [r for r in rows if r["y"] is not None]
        y = np.array([r["y"] for r in lab])
    k = np.array([keep[r["key"]] for r in lab])
    out = {"span_cost": span_cost(y, k), "labelled_spans": len(lab)}
    if pp is not None:
        out["span_auc"] = round(float(roc_auc_score(y, pp)), 4)
    return out


def run(exp: str, llm_tag: str | None, thetas: list[float], delta: float, limit: int | None):
    tr, ev = load_json(OUT / "features_train.json"), load_json(OUT / "features_eval.json")
    t0 = time.time()
    # Pipeline state saved by the features step (built once here if missing); each theta only replays the tail.
    states = {s: load_states(s, limit) for s in ("train", "eval")}
    result = {"exp": exp, "desc": EXPERIMENTS[exp], "llm": llm_tag, "train_sweep": []}
    llm_tr = llm_ev = None
    if exp in NEEDS_LLM:
        if not llm_tag:
            raise SystemExit(f"{exp} needs --llm <tag> (run experiments.bc_llm.llm_score first)")
        L_tr, L_ev = (load_json(OUT / "llm" / f"{llm_tag}_{s}.json") for s in ("train", "eval"))
        llm_tr, llm_ev = L_tr["p"], L_ev["p"]
        result["llm_meta"] = {s: L["meta"] for s, L in (("train", L_tr), ("eval", L_ev))}

    if exp == "E0":
        keep = lambda rows: {r["key"]: r["baseline_keep"] for r in rows}
        p_tr = {r["key"]: r["f"]["score"] for r in tr}
        p_ev = {r["key"]: r["f"]["score"] for r in ev}
        for split, rows, p in (("train", tr, p_tr), ("eval", ev, p_ev)):
            res = metrics(split, finalize_split(states[split]))
            row = _row(res) | _span_stats(rows, p, keep(rows))
            result["train_sweep" if split == "train" else "eval"] = [row] if split == "train" else row
        result["theta"] = 0.5
    else:
        if exp == "E3":
            p_tr, p_ev = llm_tr, llm_ev
        else:
            p_tr, p_ev, names = classifier_probs(
                tr, ev, use_embeddings=exp in ("E1", "E4", "E6"),
                llm_tr=llm_tr if exp in ("E4", "E5") else None, llm_ev=llm_ev if exp in ("E4", "E5") else None)
            result["features"] = names + (["llm_logit"] if exp in ("E4", "E5") else [])
        best = None
        for theta in thetas:
            extra = {}
            if exp == "E6":
                k_tr, k_ev, extra = cascade_decisions(tr, ev, p_tr, p_ev, llm_tr, llm_ev, theta, delta)
            else:
                k_tr = {r["key"]: bool(p_tr[r["key"]] >= theta) for r in tr}
                k_ev = {r["key"]: bool(p_ev[r["key"]] >= theta) for r in ev}
            res = metrics("train", finalize_split(states["train"], k_tr))
            row = {"theta": theta} | _row(res) | _span_stats(tr, p_tr, k_tr) | extra
            result["train_sweep"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "by_type"}))
            if best is None or row["score"] > best[0]["score"]:
                best = (row, k_ev, extra)
        row, k_ev, extra = best
        result["theta"] = row["theta"]
        res = metrics("eval", finalize_split(states["eval"], k_ev))
        result["eval"] = {"theta": row["theta"]} | _row(res) | _span_stats(ev, p_ev, k_ev) | extra
    result["llm_calls_eval"] = (result["eval"].get("llm_calls_eval") if exp == "E6" else
                                1.0 if exp in NEEDS_LLM else 0.0)
    result["seconds"] = round(time.time() - t0, 1)
    name = exp + (f"_{llm_tag}" if exp in NEEDS_LLM else "")
    save_json(result, OUT / "results" / f"{name}.json")
    print(json.dumps({"exp": exp, "llm": llm_tag, "theta": result["theta"],
                      "train": {k: v for k, v in max(result["train_sweep"], key=lambda r: r["score"]).items()
                                if k != "by_type"},
                      "eval": {k: v for k, v in result["eval"].items() if k != "by_type"}}, indent=1))
    print(f"-> {OUT / 'results' / f'{name}.json'}")


def summary():
    rows = []
    for p in sorted((OUT / "results").glob("*.json")):
        r = load_json(p)
        tr = max(r["train_sweep"], key=lambda x: x["score"])
        ev = r["eval"]
        sec = (r.get("llm_meta") or {}).get("eval", {}).get("sec_per_span")
        rows.append((tr["score"], r, tr, ev, sec))
    rows.sort(key=lambda x: -x[0])
    print("| exp | decider | LLM | θ | train a | train b | train priv | train c | **train score** | eval a | eval b | "
          "eval priv | eval c | **eval score** | eval span AUC | LLM calls | LLM s/span |")
    print("|" + "---|" * 17)
    f = lambda v: "" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    for _, r, tr, ev, sec in rows:
        print(f"| {r['exp']} | {r['desc']} | {r['llm'] or ''} | {r['theta']} | {f(tr['a_removed'])} | "
              f"{f(tr['b_removed'])} | {f(tr['privacy'])} | {f(tr['c_kept'])} | **{f(tr['score'])}** | "
              f"{f(ev['a_removed'])} | {f(ev['b_removed'])} | {f(ev['privacy'])} | {f(ev['c_kept'])} | "
              f"**{f(ev['score'])}** | {f(ev.get('span_auc'))} | {r['llm_calls_eval']:.0%} | {f(sec)} |")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", choices=sorted(EXPERIMENTS))
    p.add_argument("--llm", default=None, help="LLM tag used in llm_score (e.g. qwen35_4b)")
    p.add_argument("--thetas", default=None, help="comma list of keep thresholds swept on train")
    p.add_argument("--delta", type=float, default=0.1, help="E6: half-width of the uncertain band")
    p.add_argument("--limit", type=int, default=None, help="first N encounters (smoke test only)")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args(argv)
    if a.summary:
        return summary()
    if not a.exp:
        p.error("--exp or --summary required")
    thetas = [float(x) for x in (a.thetas or DEFAULT_THETAS.get(a.exp, "0.5,0.6,0.7,0.8,0.9")).split(",")]
    run(a.exp, a.llm, thetas, a.delta, a.limit)


if __name__ == "__main__":
    main()
