"""Rule-based detection of personal data (category a): dates, ages, contacts, IDs, addresses,
title/greeting-anchored names, occupations."""

from __future__ import annotations

import re

from .resources.lexicons import AMBIGUOUS_NAMES, FIRST_NAMES, NOT_NAMES, OCCUPATIONS
from .spans import Span

_I = re.IGNORECASE

MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_ORD = r"\d{1,2}(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d{2}"
_NUM_WORDS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
              "seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_WORD_NUM = rf"(?:(?:{'|'.join(_TENS)})(?:[\s-](?:{'|'.join(_NUM_WORDS[:9])}))?|{'|'.join(_NUM_WORDS)})"

DATE_PATTERNS = [
    rf"\b\d{{1,2}}[/-]\d{{1,2}}[/-](?:{_YEAR}|\d{{2}})\b",                       # 9/29/1957
    rf"\b{_YEAR}-\d{{2}}-\d{{2}}\b",                                            # 2021-03-04
    rf"\b{MONTHS}\.?\s+{_ORD}(?:\s*,?\s*{_YEAR})?\b",                           # march 3rd, 2021
    rf"\b{_ORD}\s+(?:of\s+)?{MONTHS}(?:\s*,?\s*{_YEAR})?\b",                    # 14 may 1958
    rf"\b{MONTHS}\s*,?\s+{_YEAR}\b",                                            # march 2020
    rf"\b\d{{1,2}}/{_YEAR}\b",                                                  # 9/2020
]
YEAR_PATTERN = rf"(?<![\$\d.])\b{_YEAR}\b(?!\s*(?:mg|milligrams?|units?|ml|cc|calories|steps|feet|ft|meters|dollars|pounds|lbs|miles))"
DOB_PATTERN = r"\b(?:date of birth|dob|d\.o\.b\.?|born on|birthday is)\b\s*[:,]?\s*"

AGE_PATTERNS = [
    rf"\b(?P<n>\d{{1,3}}|{_WORD_NUM})\s*-?\s*(?:years?|yrs?)\s*-?\s*old\b",
    r"\b(?:age|aged)\s+(?P<n>\d{1,3})\b",
    r"\b(?P<n>\d{1,3})\s*(?:yo|y/o|y\.o\.)(?!\w)",
    rf"\bi(?:'m| am)\s+(?P<n>\d{{2}}|{_WORD_NUM})(?=\s*(?:years|now|\.|,|$))",
]

PHONE = r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
EMAIL = r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"
URL = r"\b(?:https?://|www\.)\S+"
SSN = r"\b\d{3}-\d{2}-\d{4}\b"
ID_ANCHORED = (r"\b(?:mrn|medical record(?: number)?|account(?: number)?|ssn|social security(?: number)?|"
               r"insurance(?: id| number)?|policy(?: number)?|member id|patient id)\s*(?:number|#|no\.?)?\s*[:#]?\s*"
               r"(?P<id>[a-z0-9][a-z0-9-]{3,})")
LONG_DIGITS = r"(?<![\d.])\d{7,}(?![\d.])"
ADDRESS = (r"\b\d{1,5}\s+(?:[a-z]+\s){1,3}(?:street|st|avenue|ave|road|rd|drive|lane|ln|boulevard|blvd|way|"
           r"court|ct|place|pl|circle|terrace|highway|hwy)\b\.?")

OCCUPATION_CONTEXT = (r"\b(?:i'm|i am|i was|he's|he is|he was|she's|she is|she was|they're|you're|you are|you were|"
                      r"your job|works? as|worked as|"
                      r"working as|job as|career as|retired|former|used to be|employed as|occupation|profession)"
                      r"\b[\w\s,'-]{0,25}$")

_ETHNIC = ("caucasian|hispanic|latino|latina|latinx|african[- ]american|asian[- ]american|asian|mexican|"
           "puerto rican|cuban|dominican|guatemalan|salvadoran|brazilian|colombian|filipino|filipina|vietnamese|"
           "chinese|japanese|korean|indian|pakistani|bangladeshi|arab|persian|iranian|turkish|russian|ukrainian|"
           "polish|german|french|italian|irish|scottish|english|british|greek|portuguese|spanish|swedish|"
           "norwegian|jewish|native american|nigerian|ethiopian|somali|haitian|jamaican|canadian")
# Ethnicity/nationality as a person attribute only ("french fries" is not one).
ETHNICITY = (rf"\b(?P<eth>{_ETHNIC})(?=\s+(?:male|female|man|woman|gentleman|lady|boy|girl|patient|descent|"
             rf"heritage|background|origin|ancestry|decent|american|family|parents?|mother|father)\b)")

_NAME_TOK = r"[a-z][a-z'\-]+"
TITLE_NAME = rf"\b(?:mr|mrs|ms|miss|mister|dr)\s*\.?\s+(?P<n1>{_NAME_TOK})(?:\s+(?P<n2>{_NAME_TOK}))?"
DOCTOR_NAME = rf"\bdoctor\s+(?P<n1>{_NAME_TOK})(?:\s+(?P<n2>{_NAME_TOK}))?"
# Filler between greeting and name ("hey there mark") is skipped.
GREETING_NAME = (rf"\b(?:hi|hey|hello|morning|good morning|good afternoon|good evening|thanks|thank you|bye)\s*,?\s+"
                 rf"(?:(?:there|again|back)\s*,?\s+)?(?P<n1>{_NAME_TOK})\b")
INTRO_NAME = (rf"\b(?:my name is|my name's|name is|this is|i am seeing|i'm seeing|seeing|patient named|"
              rf"patient is|meet)\s+(?:patient\s+)?(?P<n1>{_NAME_TOK})(?:\s+(?P<n2>{_NAME_TOK}))?")


def _find(pattern: str, text: str, field: str, label: str, source: str, group: str | int = 0, flags=_I):
    for m in re.finditer(pattern, text, flags):
        s, e = m.span(group)
        if s < e:
            yield Span(field, s, e, text[s:e], label, source, extra={"match": m})


def _is_name_token(tok: str, original: str, strict: bool, anchored: bool = False) -> bool:
    t = tok.lower()
    if t in NOT_NAMES or len(t) < 2:
        return False
    if t in AMBIGUOUS_NAMES:
        return original[:1].isupper() or anchored
    if strict:
        return t in FIRST_NAMES or original[:1].isupper()
    return True


def detect_names(text: str, field: str) -> list[Span]:
    """Names anchored by titles, greetings, or self-introductions."""
    spans = []
    for pat, strict, src in [(TITLE_NAME, False, "rule:title"), (DOCTOR_NAME, True, "rule:doctor"),
                             (GREETING_NAME, True, "rule:greeting"), (INTRO_NAME, True, "rule:intro")]:
        for m in re.finditer(pat, text, _I):
            head = m.group(0)[:3].lower()
            role = "STAFF" if src == "rule:doctor" or head.startswith("dr") else None
            # "ms" without a period is often multiple sclerosis ("my ms has been ...").
            if src == "rule:title" and head.startswith("ms") and "." not in m.group(0)[:4]:
                strict = True
            elif src == "rule:title":
                strict = False
            first_ok = False
            for g in ("n1", "n2"):
                if g not in m.groupdict() or not m.group(g):
                    continue
                tok = m.group(g)
                # The second token must look like a name on its own (avoid "dr. smith said").
                if g == "n2" and not (first_ok and _is_name_token(tok, tok, strict=True)):
                    break
                # A greeting anchors an ambiguous name ("hey there mark") unless a verb follows ("hi, will you").
                anchored = src == "rule:greeting" and g == "n1" and not re.match(
                    r"\s+(?:i|you|we|he|she|is|are|be|have|has|do|does|was|need|go|get|see|take)\b", text[m.end(g):], _I)
                if not _is_name_token(tok, tok, strict, anchored):
                    break
                first_ok = True
                s, e = m.span(g)
                spans.append(Span(field, s, e, text[s:e], "NAME", src, extra={"role": role}))
    return spans


def detect(text: str, field: str) -> list[Span]:
    spans: list[Span] = []
    for p in DATE_PATTERNS:
        spans += _find(p, text, field, "DATE", "rule:date")
    # Bare years only when not already part of a date.
    spans += _find(YEAR_PATTERN, text, field, "YEAR", "rule:year")
    for m in re.finditer(DOB_PATTERN, text, _I):
        nxt = re.match(rf"\s*(\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|{MONTHS}\.?\s+{_ORD},?\s*{_YEAR}|{_ORD}\s+{MONTHS},?\s*{_YEAR})",
                       text[m.end():], _I)
        if nxt:
            s, e = m.end() + nxt.start(1), m.end() + nxt.end(1)
            spans.append(Span(field, s, e, text[s:e], "DOB", "rule:dob"))
    for p in AGE_PATTERNS:
        for m in re.finditer(p, text, _I):
            spans.append(Span(field, m.start(), m.end(), m.group(0), "AGE", "rule:age",
                              extra={"n": m.group("n"), "n_span": m.span("n")}))
    spans += _find(PHONE, text, field, "PHONE", "rule:phone")
    spans += _find(EMAIL, text, field, "EMAIL", "rule:email")
    spans += _find(URL, text, field, "URL", "rule:url")
    spans += _find(SSN, text, field, "ID", "rule:ssn")
    spans += _find(ID_ANCHORED, text, field, "ID", "rule:id", group="id")
    spans += _find(LONG_DIGITS, text, field, "ID", "rule:long_digits")
    spans += _find(ADDRESS, text, field, "ADDRESS", "rule:address")
    for occ, sector in OCCUPATIONS.items():
        for sp in _find(rf"\b{re.escape(occ)}s?\b", text, field, "OCCUPATION", "rule:occupation"):
            # Only the patient's (or a relative's) own job; "the nurse will be in" is not a quasi-identifier.
            if re.search(OCCUPATION_CONTEXT, text[max(0, sp.start - 40):sp.start], _I):
                sp.extra["sector"] = sector
                spans.append(sp)
    spans += _find(ETHNICITY, text, field, "ETHNICITY", "rule:ethnicity", group="eth")
    spans += detect_names(text, field)
    return spans


_WORD_TO_INT = {w: i + 1 for i, w in enumerate(_NUM_WORDS)} | {w: (i + 2) * 10 for i, w in enumerate(_TENS)}


def parse_age(n: str) -> int | None:
    n = n.lower().strip()
    if n.isdigit():
        return int(n)
    total = 0
    for part in re.split(r"[\s-]+", n):
        if part not in _WORD_TO_INT:
            return None
        total += _WORD_TO_INT[part]
    return total
