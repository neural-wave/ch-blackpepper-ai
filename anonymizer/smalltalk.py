"""Small-talk filter: dialogue units (turn pieces) that carry personal context but no clinical content.

A TF-IDF + logistic-regression classifier trained on the train pseudo ground truth: a unit is positive when it
contains a personal quasi-identifier (a: QUASI / OCCUPATION / LOCATION / DATE) and no code evidence (c).
Train encounters are scored by the fold model that did not see them, so train metrics stay honest.

python -m anonymizer.smalltalk --train      # fit, cross-validate, save to models/smalltalk.joblib
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache

from .config import MODELS_DIR, ROOT

MODEL_PATH = MODELS_DIR / "smalltalk.joblib"
SPEAKER = re.compile(r"\[[a-z_ ]+\]", re.IGNORECASE)
MAX_WORDS = 30
POS_TYPES = {"QUASI", "OCCUPATION", "LOCATION", "DATE"}
N_FOLDS = 5


def units(text: str) -> list[tuple[int, int]]:
    """Speaker turns, split at sentence ends; long pieces cut into ~MAX_WORDS-word chunks."""
    bounds = [m.end() for m in SPEAKER.finditer(text)]
    starts = sorted(set([0] + bounds))
    turns = []
    for i, s in enumerate(starts):
        nxt = SPEAKER.search(text, s)
        e = nxt.start() if nxt else len(text)
        if i + 1 < len(starts):
            e = min(e, starts[i + 1])
        turns.append((s, e))
    out = []
    for s, e in turns:
        for m in re.finditer(r"[^.?!]+[.?!]?", text[s:e]):
            ps, pe = s + m.start(), s + m.end()
            words = list(re.finditer(r"\S+", text[ps:pe]))
            for k in range(0, len(words), MAX_WORDS):
                chunk = words[k:k + MAX_WORDS]
                if chunk:
                    out.append((ps + chunk[0].start(), ps + chunk[-1].end()))
    return out


def _occurrences(needle: str, hay: str):
    return [m.span() for m in re.finditer(rf"(?<![\w\[]){re.escape(needle)}(?!\w)", hay, re.IGNORECASE)]


def labelled_units(enc: dict, spans: list[dict]) -> list[tuple[str, int]]:
    text = enc["dialogue"]
    pos, clin = [], []
    for s in spans:
        if s["field"] != "dialogue":
            continue
        occ = _occurrences(s["text"], text)
        if s["category"] == "a" and s["type"] in POS_TYPES:
            pos += occ
        elif s["category"] in "bc":
            clin += occ
    rows = []
    for a, b in units(text):
        inside = lambda o: a <= o[0] < b
        rows.append((text[a:b], int(any(map(inside, pos)) and not any(map(inside, clin)))))
    return rows


def _fit(texts, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(texts, labels)


def train() -> dict:
    import joblib
    from .data import load_encounters
    gt = {p.stem: json.loads(p.read_text())["spans"] for p in (ROOT / "pseudo_gt" / "train").glob("*.json")}
    encs = [e for e in load_encounters("train") if e["encounter_id"] in gt]
    rows = {e["encounter_id"]: labelled_units(e, gt[e["encounter_id"]]) for e in encs}
    ids = sorted(rows)
    folds = {eid: i % N_FOLDS for i, eid in enumerate(ids)}
    models, oof = [], []
    for k in range(N_FOLDS):
        tr = [r for eid in ids if folds[eid] != k for r in rows[eid]]
        m = _fit([t for t, _ in tr], [y for _, y in tr])
        models.append(m)
        te = [r for eid in ids if folds[eid] == k for r in rows[eid]]
        oof += zip(m.predict_proba([t for t, _ in te])[:, 1], [y for _, y in te])
    full = _fit([t for eid in ids for t, _ in rows[eid]], [y for eid in ids for _, y in rows[eid]])
    bundle = {"full": full, "folds": models, "fold_of": folds}
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    n_pos = sum(y for _, y in oof)
    print(f"units {len(oof)}, positive {n_pos}")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        tp = sum(1 for p, y in oof if p >= thr and y)
        fp = sum(1 for p, y in oof if p >= thr and not y)
        print(f"thr {thr}: precision {tp / max(tp + fp, 1):.2f} recall {tp / max(n_pos, 1):.2f} flagged {tp + fp}")
    return bundle


@lru_cache(maxsize=1)
def _bundle():
    import joblib
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train()


def predict(text: str, encounter_id: str | None = None) -> list[tuple[int, int, float]]:
    """(start, end, probability of small talk) for every dialogue unit."""
    b = _bundle()
    k = b["fold_of"].get(encounter_id)
    model = b["full"] if k is None else b["folds"][k]
    us = units(text)
    if not us:
        return []
    probs = model.predict_proba([text[s:e] for s, e in us])[:, 1]
    return [(s, e, float(p)) for (s, e), p in zip(us, probs)]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", action="store_true")
    if p.parse_args().train:
        train()
