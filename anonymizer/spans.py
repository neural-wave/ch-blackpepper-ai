"""Span = one detected piece of text plus the decision taken on it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# Categories from docs/strategy.md
PERSONAL = "a"          # sensitive personal
CLINICAL_OTHER = "b"    # sensitive clinical, not code-relevant
CLINICAL_CODE = "c"     # code-relevant clinical

KEEP, GENERALIZE, PERTURB, REMOVE, PSEUDONYMIZE = "keep", "generalize", "perturb", "remove", "pseudonymize"

# Direct identifiers are removed even when they overlap code evidence.
DIRECT_IDENTIFIERS = {"NAME", "PHONE", "EMAIL", "URL", "ID", "ADDRESS", "DOB"}


@dataclass
class Span:
    field: str          # "dialogue" | "note"
    start: int
    end: int
    text: str
    label: str          # NAME, DATE, AGE, LOC, DISEASE, CHEMICAL, MEASUREMENT, ...
    source: str         # detector that produced it
    category: str = PERSONAL
    action: str = REMOVE
    replacement: str = ""
    reason: str = ""
    score: float | None = None      # relevance to the encounter's codes (clinical spans)
    matched_code: str | None = None
    extra: dict = field(default_factory=dict)

    def overlaps(self, other: "Span") -> bool:
        return self.field == other.field and self.start < other.end and other.start < self.end

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_log(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        if d["score"] is not None:
            d["score"] = round(d["score"], 3)
        return d
