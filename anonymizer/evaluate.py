"""Local proxies for the organizers' metrics (the real pipeline is not public).

Privacy
  ref_phi_removed   share of reference personal-data mentions (independent detector: spaCy en_core_web_sm
                    + regex for ages/dates/contacts) no longer present in the anonymized text
  unrelated_removed share of medical mentions scored as unrelated to the codes that were removed (category b)
Utility
  code_concept_recall  share of code-relevant medical mentions still present verbatim
  desc_term_recall     share of ICD-description words present in the original that survive (scorer-independent)
  retrieval_f1_*       F1 at ICD category level of a simple retrieval coder, on original vs anonymized text

python -m anonymizer.evaluate --original eval --anonymized outputs/eval_anonymized.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from . import medical
from .config import Config
from .data import load_encounters
from .models import load_spacy
from .pipeline import FIELDS, Anonymizer, no_code_predictions
from .relevance import category

# ORG is left out: in clinical text en_core_web_sm tags section headers, tests and drugs as ORG.
REF_LABELS = {"PERSON", "GPE", "LOC", "FAC", "DATE", "NORP"}
REF_NOISE = set("kinda gonna wanna yeah okay alright uhm uh um mm-hmm hmm doc".split())
DATE_HINT = re.compile(r"\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|monday|tuesday|wednesday|thursday|"
                       r"friday|saturday|sunday|christmas|thanksgiving|easter", re.I)
REF_REGEX = {
    "AGE": r"\b\d{1,3}\s*-?\s*years?\s*-?\s*old\b",
    "DATE_NUM": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "YEAR": r"\b(?:19|20)\d{2}\b",
    "PHONE": r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}",
    "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
}
DESC_STOP = set("""unspecified without with other and or of the in on to due not elsewhere classified by for type
initial encounter subsequent site part than as a an at""".split())


def _present(needle: str, hay: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle.lower())}(?!\w)", hay.lower()) is not None


def reference_phi(text: str, field: str = "note") -> list[tuple[str, str]]:
    nlp = load_spacy("en_core_web_sm")
    # The general-domain model tags drugs/findings as PERSON/ORG; drop anything scispaCy calls medical.
    med = [(e.start, e.end) for e in medical.detect(text, field, Config.scispacy_model)]
    out = []
    for ent in nlp(text).ents:
        if ent.label_ not in REF_LABELS or len(ent.text) < 3 or re.search(r"[\n•:]", ent.text):
            continue
        if ent.label_ == "DATE":
            if not DATE_HINT.search(ent.text) or not re.search(
                    r"[a-z]{3,}day|\d{4}|/|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|christmas|easter",
                    ent.text, re.I):
                continue  # durations like "2 weeks ago" or "the 1st" are not identifiers
        elif re.search(r"\d", ent.text) or ent.text.lower() in REF_NOISE:
            continue
        if any(s < ent.end_char and ent.start_char < e for s, e in med):
            continue
        out.append((ent.label_, ent.text))
    for lab, pat in REF_REGEX.items():
        out += [(lab, m.group(0)) for m in re.finditer(pat, text, re.I)]
    return out


def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(original: list[dict], anonymized: list[dict], anon: Anonymizer) -> dict:
    by_id = {e["encounter_id"]: e for e in anonymized}
    agg = defaultdict(lambda: [0, 0])     # metric -> [hits, total]
    per_type = defaultdict(lambda: [0, 0])
    f1_orig, f1_anon = [], []
    cfg = anon.cfg
    for enc in original:
        out = by_id[enc["encounter_id"]]
        assert out["icd10_codes"] == enc["icd10_codes"], f"codes changed in {enc['encounter_id']}"
        codes = enc["icd10_codes"]
        for f in FIELDS:
            # Privacy: reference identifiers
            for lab, text in reference_phi(enc[f], f):
                gone = not _present(text, out[f])
                agg["ref_phi_removed"][0] += gone
                agg["ref_phi_removed"][1] += 1
                per_type[lab][0] += gone
                per_type[lab][1] += 1
            # Medical mentions split by relevance
            ents = medical.detect(enc[f], f, cfg.scispacy_model)
            for e, r in zip(ents, anon.scorer.score([e.text for e in ents], codes)):
                kept = _present(e.text, out[f])
                if r["score"] >= cfg.t_keep:
                    agg["code_concept_recall"][0] += kept
                    agg["code_concept_recall"][1] += 1
                elif r["score"] < cfg.t_remove:
                    agg["unrelated_removed"][0] += not kept
                    agg["unrelated_removed"][1] += 1
        # Utility: ICD description words (scorer-independent)
        orig_text = " ".join(enc[f] for f in FIELDS)
        anon_text = " ".join(out[f] for f in FIELDS)
        for c in codes:
            for w in set(re.findall(r"[a-z]{3,}", c["description"].lower())) - DESC_STOP:
                if _present(w, orig_text):
                    agg["desc_term_recall"][0] += _present(w, anon_text)
                    agg["desc_term_recall"][1] += 1
        gold = {category(c["code"]) for c in codes}
        f1_orig.append(f1({category(c["code"]) for c in no_code_predictions(anon, enc)}, gold))
        f1_anon.append(f1({category(c["code"]) for c in no_code_predictions(anon, out)}, gold))

    res = {k: round(h / t, 4) if t else None for k, (h, t) in agg.items()}
    res["counts"] = {k: t for k, (h, t) in agg.items()}
    res["retrieval_f1_original"] = round(sum(f1_orig) / len(f1_orig), 4)
    res["retrieval_f1_anonymized"] = round(sum(f1_anon) / len(f1_anon), 4)
    res["ref_phi_removed_by_type"] = {k: f"{h}/{t}" for k, (h, t) in sorted(per_type.items())}
    return res


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", default="eval")
    p.add_argument("--anonymized", required=True)
    p.add_argument("--out", default=None, help="write metrics JSON here")
    a = p.parse_args(argv)
    anonymized = load_encounters(a.anonymized)
    ids = {e["encounter_id"] for e in anonymized}
    original = [e for e in load_encounters(a.original) if e["encounter_id"] in ids]
    res = evaluate(original, anonymized, Anonymizer(Config()))
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
