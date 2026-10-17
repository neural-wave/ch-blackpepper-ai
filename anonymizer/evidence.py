"""Learned evidence (relevance signal 5): which terms the train pseudo GT marks as code evidence for which ICD
category / chapter. Catches clinical links that are not textual similarity (ibuprofen for knee pain, JVD for
heart failure). Built in memory from pseudo_gt/train at first use; train encounters see an out-of-fold lexicon.

python -m anonymizer.evidence ibuprofen M25.561      # inspect
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from functools import lru_cache

import numpy as np

from .config import ROOT
from .relevance import category, hierarchy

N_FOLDS = 5             # same fold assignment as smalltalk.py
CAT_SCORE = 0.92        # term was evidence for a code of the same ICD category
CHAPTER_SCORE = 0.80    # ... for codes of the same chapter, in >= CHAPTER_MIN encounters
CHAPTER_MIN = 2
NEIGHBOUR_SIM = 0.9     # unseen term inherits from its nearest lexicon term (SapBERT) above this
_DET = r"^(?:my|your|his|her|their|our|the|a|an|some|any|no|mild|severe|moderate|chronic|acute|left|right|bilateral)\s+"


def normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text.lower()).strip(" .,;:")
    while re.match(_DET, t):
        t = re.sub(_DET, "", t, count=1)
    if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    return t


def _chapter(code: str) -> str | None:
    return hierarchy(code)[1]


@lru_cache(maxsize=1)
def _training_data() -> tuple[dict, dict]:
    """encounter_id -> list of (term, 'c'|'b', code or None, encounter code list); encounter_id -> fold."""
    from .data import load_encounters
    encs = {e["encounter_id"]: e for e in load_encounters("train")}
    rows = {}
    for p in sorted((ROOT / "pseudo_gt" / "train").glob("*.json")):
        ann = json.loads(p.read_text())
        eid = ann["encounter_id"]
        codes = [c["code"] for c in encs[eid]["icd10_codes"]]
        rows[eid] = [(normalize(s["text"]), s["category"], s.get("code"), codes)
                     for s in ann["spans"] if s["category"] in "bc"]
    ids = sorted(rows)
    return rows, {eid: i % N_FOLDS for i, eid in enumerate(ids)}


class Lexicon:
    def __init__(self, exclude_fold: int | None):
        rows, folds = _training_data()
        pos = defaultdict(set)      # (term, level key) -> encounters where term is evidence
        neg = defaultdict(set)      # (term, level key) -> encounters where term is b despite such a code
        for eid, items in rows.items():
            if folds[eid] == exclude_fold:
                continue
            for term, cat, code, codes in items:
                if cat == "c" and code:
                    pos[(term, "cat", category(code))].add(eid)
                    pos[(term, "ch", _chapter(code))].add(eid)
                elif cat == "b":
                    for c in codes:
                        neg[(term, "cat", category(c))].add(eid)
                        neg[(term, "ch", _chapter(c))].add(eid)
        self.pos = {k: len(v) for k, v in pos.items()}
        self.neg = {k: len(v) for k, v in neg.items()}
        self.terms = sorted({k[0] for k in self.pos})
        self._vecs = None

    def _support(self, term: str, level: str, key: str | None) -> tuple[int, int]:
        return self.pos.get((term, level, key), 0), self.neg.get((term, level, key), 0)

    def lookup(self, term: str, codes: list[str]) -> tuple[float, str | None, str]:
        best = (0.0, None, "")
        for code in codes:
            p, n = self._support(term, "cat", category(code))
            if p and p > n and CAT_SCORE > best[0]:
                best = (CAT_SCORE, code, f"evidence for {category(code)} in {p} train encounter(s)")
            p, n = self._support(term, "ch", _chapter(code))
            if p >= CHAPTER_MIN and p > 2 * n and CHAPTER_SCORE > best[0]:
                best = (CHAPTER_SCORE, code, f"evidence for chapter {_chapter(code)} in {p} train encounter(s)")
        return best

    def neighbours(self, embedder, queries: list[str]) -> list[tuple[str, float]]:
        if self._vecs is None:
            self._vecs = embedder(self.terms)
        sims = embedder(queries) @ self._vecs.T
        j = sims.argmax(axis=1)
        return [(self.terms[k], float(sims[i, k])) for i, k in enumerate(j)]


@lru_cache(maxsize=N_FOLDS + 1)
def lexicon(exclude_fold: int | None) -> Lexicon:
    return Lexicon(exclude_fold)


def fold_of(encounter_id: str | None) -> int | None:
    return _training_data()[1].get(encounter_id)


def score(texts: list[str], codes: list[str], embedder, encounter_id: str | None = None) -> list[tuple]:
    """(score, code, reason) per text; 0.0 when the train pseudo GT has no evidence."""
    lex = lexicon(fold_of(encounter_id))
    terms = [normalize(t) for t in texts]
    out = [lex.lookup(t, codes) for t in terms]
    missing = [i for i, r in enumerate(out) if not r[0] and terms[i]]
    if missing and lex.terms:
        for i, (nb, sim) in zip(missing, lex.neighbours(embedder, [terms[i] for i in missing])):
            if sim >= NEIGHBOUR_SIM:
                s, c, why = lex.lookup(nb, codes)
                if s:
                    out[i] = (s, c, f"like '{nb}' ({sim:.2f}): {why}")
    return out


if __name__ == "__main__":
    lex = lexicon(None)
    term, codes = normalize(sys.argv[1]), sys.argv[2:]
    print(len(lex.terms), "terms;", term, "->", lex.lookup(term, codes))
