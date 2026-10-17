"""Quasi-identifiers the PHI rules and de-id model miss: holidays/named events, family members and pets,
stores/brands/teams, workplaces/schooling, and hobbies/activities (with an injury-cause cue for the gate)."""

from __future__ import annotations

import re

from .resources.lexicons import ACTIVITIES, ACTIVITY_GENERIC, BRANDS, HOLIDAYS, SPORTS_TEAMS, SPORTS_TEAMS_AMBIGUOUS
from .spans import Span

_I = re.IGNORECASE


def _alt(words) -> str:
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


HOLIDAY = rf"\b(?:{_alt(HOLIDAYS)})\b"

_POSS = r"(?:my|his|her|their|your|our|the patient's|patient's)"
_MOD = r"(?:(?:older|younger|little|big|baby|twin|oldest|youngest|eldest|step-?|ex-?|late|new|two|three|four)\s+)*"
_KIN = (r"wife|husband|spouse|partner|boyfriend|girlfriend|fianc[ée]e?|sons?|daughters?|kids?|child|children|"
        r"babies|baby|grandsons?|granddaughters?|grandkids?|grandchild(?:ren)?|grandma|grandmother|grandpa|"
        r"grandfather|grandparents?|mom|mother|dad|father|parents?|brothers?|sisters?|siblings?|twins?|aunts?|"
        r"uncles?|cousins?|nieces?|nephews?|(?:mother|father|brother|sister|son|daughter)s?-in-law|in-laws|"
        r"stepson|stepdaughter|stepmom|stepdad|roommates?|buddy|best friend|girlfriends|boyfriends")
_PET = r"dogs?|cats?|pupp(?:y|ies)|kittens?|horses?|pets?"
FAMILY = rf"\b{_POSS}\s+{_MOD}(?P<kin>{_KIN})\b"
FAMILY_BARE = r"\b(?:boyfriend|girlfriend|fianc[ée]e?|grandkids|grandchildren|twin (?:sister|brother)|buddy of mine)\b"
PET = rf"\b{_POSS}\s+{_MOD}(?:{_PET})\b"
# "family history" talk names relatives for medical reasons; the relation itself is generic, the name is not.
_FAMILY_SKIP = r"\b(?:my|his|her|your)\s+(?:mother|father|mom|dad|parents?)\s*(?:'s)?\s*(?:side)\b"

WORKPLACE = (r"\b(?:i|he|she|you|they)\s+(?:currently\s+)?(?:work|works|worked|working)\s+(?:at|for|in)\s+"
             r"(?:a|an|the)?\s*(?P<w>(?!home\b|the yard\b|yard\b|garden\b|house\b|office\b)[a-z'&]+(?:\s+(?!and\b|so\b|but\b)[a-z'&]+)?)")
SCHOOLING = (r"\b(?:going to school|in school|back to school|in (?:high school|college|middle school|grad school)|"
             r"(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|"
             r"twelfth) grade|(?:a )?(?:freshman|sophomore|junior|senior) (?:in|at) (?:high school|college))\b")
JOB_CHANGE = r"\b(?:job transition|changing jobs|new job|lost (?:my|his|her) job|got laid off|laid off|retired from \w+)\b"
# Trips and family events: dated, located life events.
LIFE_EVENT = (r"\b(?:(?:on |a |our |my |the |family |summer |spring )?(?:vacation|vacations|honeymoon|cruise|road trip)|"
              r"(?:my|his|her|our|your) (?:\w+ )?(?:wedding|graduation|anniversary|retirement party|baby shower)|"
              r"birthday part(?:y|ies)|house part(?:y|ies)|family reunion|moved here from|just moved|"
              r"bought a (?:new |small )?house|married for \w+(?: \w+)? years)\b")

ACTIVITY = rf"\b(?:{_alt(ACTIVITIES)})\b"
# Everyday movements count as hobbies only in a hobby frame ("i love to swim", "my favorite is running").
ACTIVITY_FRAME = (r"\b(?:i|we|he|she|they|you)\s+(?:(?:really|also|still|just|usually|like to|love to|used to|"
                  r"started|start|enjoy|enjoys|like|love|likes|loves|do|does|did|go|goes|went|was|am|is|been)\s+){1,3}"
                  rf"(?:a lot of\s+|some\s+|for a\s+)?(?P<act>{_alt(ACTIVITY_GENERIC)})\b|"
                  rf"\bmy favorite (?:\w+\s+){{0,3}}(?:is|was)\s+(?P<act2>{_alt(ACTIVITY_GENERIC)})\b")

BRAND = rf"\b(?:{_alt(BRANDS)})\b"
TEAM = rf"\b(?:{_alt(SPORTS_TEAMS)})\b"
TEAM_AMBIG = rf"\bthe\s+(?P<t>{_alt(SPORTS_TEAMS_AMBIGUOUS)})\b"
SPORTS_CONTEXT = r"\b(?:game|games|team|teams|fan|fans|season|playoffs?|play|played|playing|win|won|lose|lost|watch|watched|watching|score|scored|series|bowl|super bowl|championship|draft)\b"

# Mechanism of an injury: the activity is then code evidence (cause), not a hobby.
INJURY_CUE = (r"\b(?:hurt|hurts|hurting|injur\w*|fell|fall|falling|fallen|twist\w*|roll(?:ed|ing)|pop(?:ped)?|"
              r"sprain\w*|strain\w*|tore|torn|tear|landed|landing|slipp\w*|trip(?:ped)?|hit|struck|collid\w*|"
              r"tackl\w*|jamm\w*|pull(?:ed)?|felt|feel|pain|painful|sore|ache|aching|swell\w*|bruis\w*|broke|"
              r"broken|fracture\w*|started|since|after|ever since|flare\w*|exacerbat\w*|aggravat\w*|worse|"
              r"bothers?|bothering|overuse|repetitive|blister\w*|bite|bit|bitten|scratch\w*|allerg\w*|asthma|"
              r"dander|sneez\w*|wheez\w*|hives|tick)\b")
CUE_WINDOW = 12     # words on each side


def _cue_near(text: str, s: int, e: int) -> bool:
    left = " ".join(re.findall(r"\S+", text[max(0, s - 200):s])[-CUE_WINDOW:])
    right = " ".join(re.findall(r"\S+", text[e:e + 200])[:CUE_WINDOW])
    return bool(re.search(INJURY_CUE, left + " " + right, _I))


def detect(text: str, field: str) -> list[Span]:
    spans: list[Span] = []

    def add(s, e, label, source, **extra):
        if s < e:
            spans.append(Span(field, s, e, text[s:e], label, source, extra=extra))

    for m in re.finditer(HOLIDAY, text, _I):
        add(m.start(), m.end(), "DATE", "rule:holiday")
    skip = [m.span() for m in re.finditer(_FAMILY_SKIP, text, _I)]
    for m in re.finditer(FAMILY, text, _I):
        if not any(a <= m.start() < b for a, b in skip):
            add(m.start(), m.end(), "FAMILY", "rule:family")
    for m in re.finditer(FAMILY_BARE, text, _I):
        add(m.start(), m.end(), "FAMILY", "rule:family")
    for m in re.finditer(PET, text, _I):
        add(m.start(), m.end(), "PET", "rule:pet", cause_cue=_cue_near(text, *m.span()))
    for m in re.finditer(WORKPLACE, text, _I):
        add(*m.span("w"), "WORKPLACE", "rule:workplace")
    for m in re.finditer(SCHOOLING, text, _I):
        add(m.start(), m.end(), "OCCUPATION", "rule:schooling", sector="education")
    for m in re.finditer(JOB_CHANGE, text, _I):
        add(m.start(), m.end(), "QUASI", "rule:job_change")
    for m in re.finditer(LIFE_EVENT, text, _I):
        add(m.start(), m.end(), "QUASI", "rule:life_event")
    for m in re.finditer(ACTIVITY, text, _I):
        add(m.start(), m.end(), "ACTIVITY", "rule:activity", cause_cue=_cue_near(text, *m.span()))
    for m in re.finditer(ACTIVITY_FRAME, text, _I):
        g = "act" if m.group("act") else "act2"
        add(*m.span(g), "ACTIVITY", "rule:activity_frame", cause_cue=_cue_near(text, *m.span(g)))
    for m in re.finditer(BRAND, text, _I):
        add(m.start(), m.end(), "ORG", "rule:brand")
    for m in re.finditer(TEAM, text, _I):
        add(m.start(), m.end(), "ORG", "rule:team")
    for m in re.finditer(TEAM_AMBIG, text, _I):
        window = text[max(0, m.start() - 120):m.end() + 120]
        if re.search(SPORTS_CONTEXT, window, _I):
            add(*m.span("t"), "ORG", "rule:team")
    return spans
