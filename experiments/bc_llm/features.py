"""Stage 1: run the baseline pipeline, record every detected clinical span (DISEASE / CHEMICAL / ENTITY) with
features and its pseudo-GT label. Also hosts `DeciderAnonymizer`, which replaces the t_keep cut-off with an
external keep/remove decision per span.

python -m experiments.bc_llm.features            # -> outputs/bc_llm/features_{train,eval}.json
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

from tqdm import tqdm

from anonymizer import evidence
from anonymizer.config import Config
from anonymizer.data import load_encounters
from anonymizer.pipeline import PRIORITY, Anonymizer
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
    """Baseline pipeline that (a) records span features and/or (b) takes keep/remove from `keep` (key -> bool)."""

    def __init__(self, cfg: Config, keep: dict[str, bool] | None = None, record: bool = False):
        super().__init__(cfg)
        self.keep, self.record, self.rows = keep, record, []
        if record:
            self.lexical = RelevanceScorer(self.embedder, self.index, (2,))
            self.sapbert = RelevanceScorer(self.embedder, self.index, (1,))

    def _clinical_spans(self, texts, codes):
        spans = super()._clinical_spans(texts, codes)
        clin = [s for s in spans if s.label in ("DISEASE", "CHEMICAL", "ENTITY")]
        if self.record and clin:
            self._record(clin, texts, codes)
        if self.keep is not None:
            for s in clin:
                k = self.keep.get(f"{self._eid}|{s.field}|{s.start}|{s.end}")
                if k is None:
                    continue
                if k:
                    s.category, s.action, s.reason = CLINICAL_CODE, KEEP, "kept by b/c decider"
                    s.extra["prio"] = PRIORITY["code"]
                else:
                    s.category, s.action, s.reason = CLINICAL_OTHER, REMOVE, "removed by b/c decider"
                    s.extra["prio"] = PRIORITY["clinical"]
        return spans

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


def run_pipeline(split: str, keep: dict[str, bool] | None = None, record: bool = False, limit: int | None = None):
    anon = DeciderAnonymizer(Config(), keep=keep, record=record)
    outputs = [anon.anonymize(enc)[0] for enc in tqdm(load_encounters(split)[:limit], desc=f"pipeline {split}")]
    return outputs, anon.rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="train,eval")
    p.add_argument("--limit", type=int, default=None, help="first N encounters (smoke test)")
    a = p.parse_args(argv)
    for split in a.splits.split(","):
        _, rows = run_pipeline(split, record=True, limit=a.limit)
        occ = gt_occurrences(split)
        for r in rows:
            r["y"] = label(r, occ)
        save_json(rows, OUT / f"features_{split}.json")
        print(f"{split}: {len(rows)} clinical spans, {sum(r['y'] is not None for r in rows)} labelled "
              f"-> {OUT / f'features_{split}.json'}")


if __name__ == "__main__":
    main()
