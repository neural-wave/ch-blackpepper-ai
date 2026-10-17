"""End-to-end anonymization of one encounter (see docs/architecture.md for the stages)."""

from __future__ import annotations

import random
import re
from collections import defaultdict

from . import medical, phi_ner, phi_rules
from .config import Config
from .models import resolve_device
from .relevance import Embedder, ICDIndex, RelevanceScorer, block_description, category
from .resources.lexicons import AMBIGUOUS_NAMES, EPONYM_CONTEXT, EPONYMS, NOT_NAMES
from .spans import (CLINICAL_CODE, CLINICAL_OTHER, DIRECT_IDENTIFIERS, GENERALIZE, KEEP, PERSONAL, PERTURB,
                    PSEUDONYMIZE, REMOVE, Span)

FIELDS = ("dialogue", "note")
TAGS = {"DATE": "[DATE]", "DOB": "[DATE]", "YEAR": "[YEAR]", "PHONE": "[PHONE]", "EMAIL": "[EMAIL]", "URL": "[URL]",
        "ID": "[ID]", "ADDRESS": "[ADDRESS]", "LOC": "[LOCATION]", "HOSP": "[HOSPITAL]", "ORG": "[ORGANIZATION]",
        "OTHERPHI": "[REDACTED]"}
# Higher = wins overlaps.
PRIORITY = {"direct": 4, "code": 3, "personal": 2, "clinical": 1, "measure": 0}
# Personal spans that could themselves be code evidence (e.g. Y92 place, Y93 activity) get a relevance guard.
GUARDED = {"LOC", "HOSP", "ORG", "OTHERPHI", "OCCUPATION"}


class Anonymizer:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.device = resolve_device(self.cfg.device)
        self.embedder = Embedder(self.cfg.sapbert_model, self.device)
        self.index = ICDIndex(self.embedder)
        self.scorer = RelevanceScorer(self.embedder, self.index, self.cfg.signals)

    # ------------------------------------------------------------------ public
    def anonymize(self, enc: dict, codes: list[dict] | None = None) -> tuple[dict, dict]:
        """Return (anonymized encounter, decision log). `codes` overrides the encounter's codes (no-code mode)."""
        codes = enc["icd10_codes"] if codes is None else codes
        rng = random.Random(f"{self.cfg.seed}:{enc.get('encounter_id')}")
        texts = {f: enc[f] for f in FIELDS}

        clinical = self._clinical_spans(texts, codes)
        personal = self._personal_spans(texts)
        names = self._name_spans(texts, personal, clinical)
        personal = [s for s in personal if s.label != "NAME"] + names
        self._guard_personal(personal, codes)
        chosen = self._resolve(clinical + personal)

        out = dict(enc)
        for f in FIELDS:
            out[f] = self._apply(texts[f], [s for s in chosen if s.field == f], rng)
        out, leaks = self._sweep_names(out, names)

        evidence = defaultdict(int)
        for s in chosen:
            if s.category == CLINICAL_CODE and s.matched_code:
                evidence[s.matched_code] += 1
        log = {
            "encounter_id": enc.get("encounter_id"),
            "decisions": [s.to_log() for s in sorted(chosen, key=lambda s: (s.field, s.start))],
            "codes_without_evidence": [c["code"] for c in codes if not evidence.get(c["code"])],
            "name_leaks_swept": leaks,
        }
        return out, log

    # ------------------------------------------------------------ [1] keep mask
    def _clinical_spans(self, texts: dict, codes: list[dict]) -> list[Span]:
        spans = []
        for f, text in texts.items():
            spans += medical.detect(text, f, self.cfg.scispacy_model)
        scores = self.scorer.score([s.text for s in spans], codes)
        for s, r in zip(spans, scores):
            s.score, s.matched_code = r["score"], r["code"]
            s.extra["nearest"], s.extra["nearest_sim"] = r["nearest"], r["nearest_sim"]
            is_symptom = r["nearest"].startswith("R") and r["nearest_sim"] >= 0.6
            if r["score"] >= self.cfg.t_keep:
                s.category, s.action, s.reason = CLINICAL_CODE, KEEP, f"code evidence for {r['code']}: {r['reason']}"
            elif is_symptom and self.cfg.keep_symptoms:
                s.category, s.action = CLINICAL_CODE, KEEP
                s.reason = f"symptom/finding (nearest {r['nearest']}), kept as diagnostic context"
                s.matched_code = None
            elif r["score"] >= self.cfg.t_remove:
                s.category, s.action = CLINICAL_OTHER, GENERALIZE
                s.reason = f"weakly related to codes (score {r['score']:.2f}); generalized"
            else:
                s.category, s.action = CLINICAL_OTHER, REMOVE
                s.reason = f"unrelated to encounter codes (score {r['score']:.2f}); removed"
            s.extra["prio"] = PRIORITY["code"] if s.category == CLINICAL_CODE else PRIORITY["clinical"]
        for f, text in texts.items():
            for s in medical.detect_measurements(text, f):
                s.category, s.action, s.reason = CLINICAL_CODE, PERTURB, "measurement value slightly perturbed"
                s.extra["prio"] = PRIORITY["measure"]
                spans.append(s)
        return spans

    # ------------------------------------------------- [2] personal data detection
    def _personal_spans(self, texts: dict) -> list[Span]:
        spans = []
        for f, text in texts.items():
            spans += phi_rules.detect(text, f)
            if self.cfg.use_deid_ner:
                spans += phi_ner.detect(text, f, self.cfg.deid_model, self.device)
        for s in spans:
            s.category = PERSONAL
            direct = s.label in DIRECT_IDENTIFIERS
            s.extra["prio"] = PRIORITY["direct"] if direct else PRIORITY["personal"]
            if s.label == "AGE":
                s.action, s.reason = GENERALIZE, "age generalized to a 10-year band"
            elif s.label == "OCCUPATION":
                s.action, s.reason = GENERALIZE, "occupation is a quasi-identifier; generalized to sector"
            else:
                s.action, s.reason = REMOVE, f"{s.label.lower()} is personal data ({s.source})"
        return spans

    def _guard_personal(self, spans: list[Span], codes: list[dict]):
        guarded = [s for s in spans if s.label in GUARDED]
        for s, r in zip(guarded, self.scorer.score([s.text for s in guarded], codes)):
            if r["score"] >= self.cfg.t_keep:
                s.category, s.action, s.score, s.matched_code = CLINICAL_CODE, KEEP, r["score"], r["code"]
                s.reason = f"{s.label.lower()} kept: evidence for {r['code']} ({r['reason']})"
                s.extra["prio"] = PRIORITY["code"]

    # ---------------------------------------- names: registry, eponyms, propagation
    def _is_eponym_use(self, text: str, s: Span, clinical: list[Span]) -> bool:
        tok = re.sub(r"'s$", "", s.text.lower())
        if tok not in EPONYMS:
            return False
        before = text[max(0, s.start - 6):s.start].lower()
        if re.search(r"\b(mr|mrs|ms|dr|miss)\s*\.?\s*$", before):
            return False
        after = re.findall(r"[a-z']+", text[s.end:s.end + 30].lower())[:2]
        if any(w in EPONYM_CONTEXT for w in after) or text[s.end:s.end + 2] == "'s":
            return True
        return any(c.field == s.field and c.start <= s.start and s.end <= c.end and c.label == "DISEASE"
                   for c in clinical)

    def _name_spans(self, texts: dict, personal: list[Span], clinical: list[Span]) -> list[Span]:
        """Build a per-encounter name registry and mark every occurrence of a registered name."""
        registry: dict[str, str] = {}
        counters = defaultdict(int)
        seeds = [s for s in personal if s.label == "NAME"]
        # NER on the cased note is the most reliable source of roles; register those first.
        seeds.sort(key=lambda s: (s.source.startswith("rule"), s.field != "note"))
        for s in seeds:
            if self._is_eponym_use(texts[s.field], s, clinical):
                continue
            toks = [t for t in re.findall(r"[a-z][a-z'\-]+", s.text.lower()) if t not in NOT_NAMES and len(t) > 1]
            toks = [re.sub(r"'s$", "", t) for t in toks]
            if not toks:
                continue
            tag = next((registry[t] for t in toks if t in registry), None)
            if tag is None:
                role = s.extra.get("role")
                if role == "PATIENT" and not counters["PATIENT"]:
                    tag = "[PATIENT]"
                    counters["PATIENT"] += 1
                else:
                    kind = "CLINICIAN" if role == "STAFF" else "PERSON"
                    counters[kind] += 1
                    tag = f"[{kind}_{counters[kind]}]"
            for t in toks:
                registry.setdefault(t, tag)

        spans = []
        for f, text in texts.items():
            for tok, tag in registry.items():
                for m in re.finditer(rf"\b{re.escape(tok)}(?:'s)?\b", text, re.IGNORECASE):
                    sp = Span(f, m.start(), m.start() + len(tok), text[m.start():m.start() + len(tok)], "NAME",
                              "registry", category=PERSONAL, action=PSEUDONYMIZE, replacement=tag,
                              reason=f"name of a person, consistently pseudonymized as {tag}")
                    sp.extra["prio"] = PRIORITY["direct"]
                    if tok in AMBIGUOUS_NAMES and not m.group(0)[:1].isupper() and not any(
                            s.overlaps(sp) for s in seeds):
                        continue
                    if self._is_eponym_use(text, sp, clinical):
                        continue
                    spans.append(sp)
        self._registry = registry
        return spans

    # ------------------------------------------------------------- [3] resolve
    @staticmethod
    def _resolve(spans: list[Span]) -> list[Span]:
        """Greedy non-overlapping selection: priority first, then longer span."""
        chosen: list[Span] = []
        for s in sorted(spans, key=lambda s: (-s.extra["prio"], -s.length)):
            if not any(s.overlaps(c) for c in chosen):
                chosen.append(s)
        return chosen

    # ----------------------------------------------------------- [4] transform
    def _replacement(self, s: Span, rng: random.Random) -> str:
        if s.action == KEEP:
            return s.text
        if s.action == PSEUDONYMIZE:
            return s.replacement
        if s.label == "AGE":
            m, n_span = s.extra.get("match"), s.extra.get("n_span")
            age = phi_rules.parse_age(s.extra["n"]) if "n" in s.extra else (
                int(s.text) if s.text.isdigit() else None)
            if age is None:
                return "[AGE]"
            # "50s" rather than "50-59": a range would still contain the exact age as a substring.
            band = "90+" if age >= 90 else f"{age // 10 * 10}s"
            if n_span:
                a, b = n_span[0] - s.start, n_span[1] - s.start
                return s.text[:a] + band + s.text[b:]
            return band
        if s.label == "OCCUPATION":
            return f"[{s.extra['sector']} worker]"
        if s.label == "ETHNICITY":
            return "[ETHNICITY]"
        if s.label in ("DISEASE", "CHEMICAL"):
            kind = "condition" if s.label == "DISEASE" else "medication"
            if s.action == GENERALIZE:
                # A poor nearest-code match would generalize to a wrong group; fall back to a generic tag.
                reliable = s.label == "DISEASE" and s.extra.get("nearest_sim", 0) >= 0.75
                block = block_description(s.extra["nearest"]) if reliable else None
                return f"[{block.lower()}]" if block else f"[{kind}]"
            return f"[other {kind}]"
        if s.action == PERTURB:
            return _perturb(s.text, self.cfg.perturb_pct, rng)
        return TAGS.get(s.label, "[REDACTED]")

    def _apply(self, text: str, spans: list[Span], rng: random.Random) -> str:
        for s in sorted(spans, key=lambda s: s.start, reverse=True):
            s.replacement = self._replacement(s, rng)
            text = text[:s.start] + s.replacement + text[s.end:]
        for s in spans:
            s.extra.pop("match", None)
        return text

    # -------------------------------------------------------------- [5] verify
    def _sweep_names(self, out: dict, names: list[Span]) -> tuple[dict, int]:
        """Last-line defence: any registered name token still present is replaced."""
        leaks = 0
        for f in FIELDS:
            text = out[f]
            for tok, tag in self._registry.items():
                if tok in AMBIGUOUS_NAMES or tok in EPONYMS:
                    continue
                text, n = re.subn(rf"(?<!\[)\b{re.escape(tok)}\b", tag, text, flags=re.IGNORECASE)
                leaks += n
            out[f] = text
        return out, leaks


def _perturb(value: str, pct: float, rng: random.Random) -> str:
    decimals = len(value.split(".")[1]) if "." in value else 0
    v = float(value)
    step = 10 ** -decimals
    delta = max(step, round(abs(v) * pct, decimals)) * rng.choice([-1, 1])
    new = max(0.0, v + delta)
    return f"{new:.{decimals}f}"


def no_code_predictions(anon: Anonymizer, enc: dict, k: int = 5) -> list[dict]:
    """Bonus (no ICD codes at inference): retrieve likely codes from disease mentions via the ICD index."""
    ents = [s for f in FIELDS for s in medical.detect(enc[f], f, anon.cfg.scispacy_model) if s.label == "DISEASE"]
    if not ents:
        return []
    import simple_icd_10_cm as cm
    votes: dict[str, float] = defaultdict(float)
    for hits in anon.index.nearest(anon.embedder([e.text.lower() for e in ents]), k=1):
        code, sim = hits[0]
        if sim >= 0.7:
            votes[code] += sim
    top = sorted(votes, key=votes.get, reverse=True)[:k]
    return [{"code": c, "description": cm.get_description(c)} for c in top]


__all__ = ["Anonymizer", "no_code_predictions", "category"]
