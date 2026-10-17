"""Medical entity detection: scispaCy BC5CDR (DISEASE, CHEMICAL), en_core_sci_md (untyped ENTITY), measurements."""

from __future__ import annotations

import re

from .models import load_spacy
from .resources.lexicons import ENTITY_KINDS, GENERIC_PHRASES, GENERIC_TERMS, MEASUREMENT_UNITS
from .spans import Span

MEASUREMENT = rf"(?<![\w/])(?P<v>\d+(?:\.\d+)?)\s*(?P<u>{MEASUREMENT_UNITS})(?!\w)"
BP = r"(?<![\d/])(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})(?![\d/])"
# Pronoun-like / filler entities the NER sometimes emits.
_JUNK = {"pain", "it", "this", "that"}
_LEADING = re.compile(r"^(?:(?:the|a|an|your|my|his|her|their|our|any|some|no)\s+)+", re.IGNORECASE)


def detect(text: str, field: str, model_name: str) -> list[Span]:
    nlp = load_spacy(model_name)
    spans = []
    for ent in nlp(text).ents:
        if ent.text.lower() in _JUNK or sum(ch.isalpha() for ch in ent.text) < 3 or ent.text.endswith("-"):
            continue
        # Section headers ("Psychiatric: ...") are structure, not content.
        if text[ent.end_char:ent.end_char + 1] == ":":
            continue
        spans.append(Span(field, ent.start_char, ent.end_char, ent.text, ent.label_, "scispacy"))
    return spans


def detect_entities(text: str, field: str, model_name: str, typed: list[Span]) -> list[Span]:
    """Untyped biomedical entities (procedures, labs, exam findings) not already covered by a typed span."""
    spans = []
    for s in detect(text, field, model_name):
        if text[s.start - 1:s.start] == "[":     # speaker tag "[doctor]"
            continue
        lead = _LEADING.match(s.text)
        start = s.start + (lead.end() if lead else 0)
        raw = text[start:s.end]
        end = start + len(raw.rstrip())
        norm = raw.strip().lower()
        toks = re.findall(r"[a-z0-9'\-]+", norm)
        if not toks or norm in GENERIC_PHRASES or all(t in GENERIC_TERMS for t in toks):
            continue
        sp = Span(field, start, end, text[start:end], "ENTITY", "scispacy:sci_md")
        if any(sp.overlaps(t) for t in typed):
            continue
        sp.extra["kind"] = next((k for k, pat in ENTITY_KINDS if re.search(rf"\b(?:{pat})", norm)), "clinical detail")
        spans.append(sp)
    return spans


def detect_measurements(text: str, field: str) -> list[Span]:
    spans = []
    for m in re.finditer(MEASUREMENT, text, re.IGNORECASE):
        spans.append(Span(field, m.start("v"), m.end("v"), m.group("v"), "MEASUREMENT", "rule:measure",
                          extra={"unit": m.group("u")}))
    for m in re.finditer(BP, text):
        ctx = text[max(0, m.start() - 40):m.start()].lower()
        if "pressure" in ctx or "bp" in ctx:
            for g in ("sys", "dia"):
                spans.append(Span(field, m.start(g), m.end(g), m.group(g), "MEASUREMENT", "rule:bp"))
    return spans
