"""Stage 1: run the baseline pipeline once per split, record every detected clinical span (DISEASE / CHEMICAL /
ENTITY) with features and its pseudo-GT label, and save the pipeline state before the final step.

The state (`prepared_<split>.pkl`) holds everything that does not depend on the b/c decision: detections, scores,
personal-data spans, names, small-talk probabilities. `finalize` replays only the decision-dependent tail
(keep/remove, small-talk and social-history gating, overlap resolution, rewriting, name sweep), so every
experiment / theta costs seconds instead of a pipeline run. finalize(state, None) == Anonymizer.anonymize.

python -m experiments.bc_llm.features    # -> outputs/bc_llm/features_{split}.json, prepared_{split}.pkl
"""

from __future__ import annotations

import argparse
import copy
import pickle
import random
import re
from collections import defaultdict
from unittest import mock

from tqdm import tqdm

from anonymizer import evidence, smalltalk
from anonymizer.config import Config
from anonymizer.data import load_encounters
from anonymizer.pipeline import FIELDS, PRIORITY, Anonymizer
from anonymizer.relevance import RelevanceScorer, hierarchy
from anonymizer.spans import CLINICAL_CODE, CLINICAL_OTHER, KEEP, REMOVE

from .common import OUT, gt_occurrences, label, save_json

NEG = re.compile(r"\b(no|not|denies|denied|deny|without|negative for|never|nor|free of)\b", re.I)
SECTIONS = [("cc", r"CHIEF COMPLAINT"), ("hpi", r"HISTORY OF PRESENT ILLNESS|SUBJECTIVE"),
            ("ros", r"REVIEW OF SY"), ("pe", r"PHYSICAL EXAM|^EXAM"), ("results", r"RESULTS|EKG"),
            ("ap", r"ASSESSMENT|PLAN|IMPRESSION|ORDERS"), ("meds", r"MEDICATIONS|ALLERGIES"),
            ("hist", r"MEDICAL HISTORY|PAST HISTORY|SURGICAL HISTORY|PAST MEDICAL"), ("social", r"SOCIAL HISTORY"),
            ("family", r"FAMILY HISTORY"), ("instr", r"INSTRUCTIONS"), ("vitals", r"VITALS")]
SEC_NAMES = [s for s, _ in SECTIONS] + ["other"]
SIGS = ["s1", "s2", "s3", "s4b", "s4c", "s5", "none"]
LABELS = ["DISEASE", "CHEMICAL", "clinical detail", "test", "procedure", "exam finding"]
# Features that need SapBERT (similarity, nearest ICD code, or a signal that depends on them). E2 / E5 drop them.
EMBEDDING_FEATURES = {"score", "s1", "s5", "nearest_sim", "near_R", "near_letter_in_codes", "near_chapter_in_codes",
                      "sig_s1", "sig_s3", "sig_s4b", "sig_s4c", "sig_s5", "sig_s2", "sig_none"}


def _sections(note: str) -> list[tuple[int, str]]:
    out = []
    for m in re.finditer(r"^([A-Z][A-Z /&]+)\s*$", note, re.M):
        out.append((m.start(), next((n for n, pat in SECTIONS if re.search(pat, m.group(1))), "other")))
    return out


def _signal(reason: str) -> str:
    r = reason or ""
    for key, sig in (("learned:", "s5"), ("matches code concept", "s2"), ("same category", "s3"),
                     ("same ICD block", "s4b"), ("organ-system chapter", "s4c"), ("similar to", "s1")):
        if key in r:
            return sig
    return "none"


class DeciderAnonymizer(Anonymizer):
    """Baseline pipeline split into `prepare` (decision-independent, run once) and `finalize` (decision tail)."""

    def __init__(self, cfg: Config, record: bool = False):
        super().__init__(cfg)
        self.record, self.rows = record, []
        if record:
            self.lexical = RelevanceScorer(self.embedder, self.index, (2,))
            self.sapbert = RelevanceScorer(self.embedder, self.index, (1,))

    def _clinical_spans(self, texts, codes):
        spans = super()._clinical_spans(texts, codes)
        clin = [s for s in spans if s.label in ("DISEASE", "CHEMICAL", "ENTITY")]
        if self.record and clin:
            self._record(clin, texts, codes)
        return spans

    def prepare(self, enc: dict) -> dict:
        """Same steps as Anonymizer.anonymize up to the decision-dependent tail."""
        codes = enc["icd10_codes"]
        self._eid = enc.get("encounter_id")
        texts = {f: enc[f] for f in FIELDS}
        clinical = self._clinical_spans(texts, codes)
        personal = self._personal_spans(texts)
        names = self._name_spans(texts, personal, clinical)
        personal = [s for s in personal if s.label != "NAME"] + names
        self._guard_personal(personal, codes)
        personal += self._propagate(texts, personal)
        if 3 in self.cfg.a_ideas:
            personal += self._propagate_ext(texts, personal)
        preds = smalltalk.predict(texts["dialogue"], self._eid) if self.cfg.smalltalk else []
        for s in clinical + personal:
            s.extra.pop("match", None)      # re.Match: not picklable, never read when writing the output
        return {"enc": enc, "clinical": clinical, "personal": personal, "names": names,
                "registry": dict(self._registry), "smalltalk": preds}

    def finalize(self, state: dict, keep: dict[str, bool] | None = None) -> dict:
        """Decision tail of Anonymizer.anonymize; `keep` (span key -> bool) overrides clinical keep/remove."""
        enc, codes = state["enc"], state["enc"]["icd10_codes"]
        eid = enc.get("encounter_id")
        texts = {f: enc[f] for f in FIELDS}
        clinical, personal, names = copy.deepcopy((state["clinical"], state["personal"], state["names"]))
        if keep is not None:
            for s in clinical:
                k = keep.get(f"{eid}|{s.field}|{s.start}|{s.end}") if s.label in ("DISEASE", "CHEMICAL", "ENTITY") \
                    else None
                if k is True:
                    s.category, s.action, s.reason = CLINICAL_CODE, KEEP, "kept by b/c decider"
                    s.extra["prio"] = PRIORITY["code"]
                elif k is False:
                    s.category, s.action, s.reason = CLINICAL_OTHER, REMOVE, "removed by b/c decider"
                    s.extra["prio"] = PRIORITY["clinical"]
        spans = clinical + personal
        if 2 in self.cfg.a_ideas:
            spans += self._social_history_spans(texts["note"], spans, codes)
        if self.cfg.smalltalk:
            with mock.patch.object(smalltalk, "predict", lambda *_: state["smalltalk"]):
                spans += self._smalltalk_spans(texts["dialogue"], spans, eid)
        chosen = self._resolve(spans)
        rng = random.Random(f"{self.cfg.seed}:{eid}")
        out = dict(enc)
        for f in FIELDS:
            out[f] = self._apply(texts[f], [s for s in chosen if s.field == f], rng)
        self._registry = state["registry"]
        return self._sweep_names(out, names)[0]

    def _record(self, clin, texts, codes):
        names = [s.text for s in clin]
        s1 = self.sapbert.score(names, codes, self._eid)
        s2 = self.lexical.score(names, codes, self._eid)
        s5 = evidence.score(names, [c["code"] for c in codes], self.embedder, self._eid)
        lex = evidence.lexicon(evidence.fold_of(self._eid))
        secs = _sections(texts["note"])
        bounds = secs + [(len(texts["note"]), "end")]
        sec_text = defaultdict(str)
        for (st, n), (en, _) in zip(bounds, bounds[1:]):
            sec_text[n] += " " + texts["note"][st:en].lower()
        chapters = {hierarchy(c["code"])[1] for c in codes}
        letters = {c["code"][0] for c in codes}
        note_l, dia_l = texts["note"].lower(), texts["dialogue"].lower()
        for s, a, b, e in zip(clin, s1, s2, s5):
            low = s.text.lower()
            pat = rf"(?<!\w){re.escape(low)}(?!\w)"
            sent = re.split(r"[.!?\n]", texts[s.field][max(0, s.start - 80):s.start])[-1]
            near = s.extra.get("nearest", "")
            f = {
                "score": s.score or 0.0, "s1": a["score"], "s2_lex": float(b["score"] >= 1.0), "s5": e[0],
                "s5_exact": lex.lookup(evidence.normalize(s.text), [c["code"] for c in codes])[0],
                "nearest_sim": s.extra.get("nearest_sim", 0.0), "near_R": float(near.startswith("R")),
                "near_letter_in_codes": float(near[:1] in letters),
                "near_chapter_in_codes": float(hierarchy(near)[1] in chapters) if near else 0.0,
                "neg": float(bool(NEG.search(sent))), "field_note": float(s.field == "note"),
                "n_note": len(re.findall(pat, note_l)), "n_dia": len(re.findall(pat, dia_l)),
                "ntok": len(low.split()), "n_codes": len(codes),
            }
            for n in SEC_NAMES:
                f[f"in_{n}"] = float(bool(re.search(pat, sec_text.get(n, ""))))
            sec = "dialogue"
            if s.field == "note":
                sec = next((n for st, n in reversed(secs) if st <= s.start), "other")
            for n in SEC_NAMES + ["dialogue"]:
                f[f"at_{n}"] = float(sec == n)
            sig = _signal(s.reason)
            for g in SIGS:
                f[f"sig_{g}"] = float(sig == g)
            lab = s.label if s.label != "ENTITY" else s.extra.get("kind")
            for name in LABELS:
                f[f"lab_{name}"] = float(lab == name)
            self.rows.append({"key": f"{self._eid}|{s.field}|{s.start}|{s.end}", "eid": self._eid, "field": s.field,
                              "start": s.start, "end": s.end, "text": s.text, "f": f,
                              "baseline_keep": s.action == KEEP})


def prepare_split(split: str, record: bool = False, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    anon = DeciderAnonymizer(Config(), record=record)
    states = [anon.prepare(enc) for enc in tqdm(load_encounters(split)[:limit], desc=f"pipeline {split}")]
    return states, anon.rows


def load_states(split: str, limit: int | None = None) -> list[dict]:
    """Prepared states from disk; built once (one pipeline pass) if missing."""
    path = OUT / f"prepared_{split}.pkl"
    if limit is None and path.exists():
        return pickle.loads(path.read_bytes())
    states, _ = prepare_split(split, limit=limit)
    if limit is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(states))
    return states


_FINALIZER = None


def finalize_split(states: list[dict], keep: dict[str, bool] | None = None) -> list[dict]:
    global _FINALIZER
    if _FINALIZER is None:
        _FINALIZER = DeciderAnonymizer(Config())
    return [_FINALIZER.finalize(st, keep) for st in states]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="train,eval")
    p.add_argument("--limit", type=int, default=None, help="first N encounters (smoke test, nothing cached)")
    a = p.parse_args(argv)
    for split in a.splits.split(","):
        states, rows = prepare_split(split, record=True, limit=a.limit)
        occ = gt_occurrences(split)
        for r in rows:
            r["y"] = label(r, occ)
        save_json(rows, OUT / f"features_{split}.json")
        if a.limit is None:
            (OUT / f"prepared_{split}.pkl").write_bytes(pickle.dumps(states))
        print(f"{split}: {len(rows)} clinical spans, {sum(r['y'] is not None for r in rows)} labelled "
              f"-> {OUT / f'features_{split}.json'}, prepared_{split}.pkl")


if __name__ == "__main__":
    main()
