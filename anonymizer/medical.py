"""Medical entity detection (scispaCy BC5CDR: DISEASE, CHEMICAL) + measurement values."""

from __future__ import annotations

import re

from .models import load_spacy
from .resources.lexicons import MEASUREMENT_UNITS
from .spans import Span

MEASUREMENT = rf"(?<![\w/])(?P<v>\d+(?:\.\d+)?)\s*(?P<u>{MEASUREMENT_UNITS})(?!\w)"
BP = r"(?<![\d/])(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})(?![\d/])"
# Pronoun-like / filler entities the NER sometimes emits.
_JUNK = {"pain", "it", "this", "that"}


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
