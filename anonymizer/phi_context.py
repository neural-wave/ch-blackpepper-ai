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


# Idea 4: only mechanism words, not symptoms or time words ("pain", "since", "started" sit next to most hobbies).
STRICT_CUE = (r"\b(?:hurt|hurts|hurting|injur\w*|fell|fall|falling|fallen|twist\w*|roll(?:ed|ing)|pop(?:ped)?|"
              r"sprain\w*|strain\w*|tore|torn|landed|landing|slipp\w*|trip(?:ped)?|struck|collid\w*|tackl\w*|"
              r"jamm\w*|broke|broken|fracture\w*|overuse|repetitive|blister\w*|bite|bit|bitten|scratch\w*|"
              r"allerg\w*|asthma|dander|sneez\w*|wheez\w*|hives|tick|accident|crash\w*|ran into)\b")
# "while/when/during <activity>", "from <activity>": the activity is the circumstance of the problem.
CAUSAL_LEFT = r"\b(?:while|when|during|from|after|playing|doing)\s+(?:\w+\s+){0,2}$"
STRICT_WINDOW = 8


def _cue_strict(text: str, s: int, e: int) -> bool:
    left = " ".join(re.findall(r"\S+", text[max(0, s - 200):s])[-STRICT_WINDOW:])
    right = " ".join(re.findall(r"\S+", text[e:e + 200])[:STRICT_WINDOW])
    return bool(re.search(STRICT_CUE, left + " " + right, _I) or re.search(CAUSAL_LEFT, left + " ", _I))


def _cue_near(text: str, s: int, e: int) -> bool:
    left = " ".join(re.findall(r"\S+", text[max(0, s - 200):s])[-CUE_WINDOW:])
    right = " ".join(re.findall(r"\S+", text[e:e + 200])[:CUE_WINDOW])
    return bool(re.search(INJURY_CUE, left + " " + right, _I))


def detect(text: str, field: str, ideas: frozenset = frozenset()) -> list[Span]:
    """`ideas` = personal-data extensions of study G (Config.a_ideas): 1 extra rules, 4 strict injury cue."""
    spans: list[Span] = []
    cue = _cue_strict if 4 in ideas else _cue_near

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
        add(m.start(), m.end(), "PET", "rule:pet", cause_cue=cue(text, *m.span()))
    for m in re.finditer(WORKPLACE, text, _I):
        add(*m.span("w"), "WORKPLACE", "rule:workplace")
    for m in re.finditer(SCHOOLING, text, _I):
        add(m.start(), m.end(), "OCCUPATION", "rule:schooling", sector="education")
    for m in re.finditer(JOB_CHANGE, text, _I):
        add(m.start(), m.end(), "QUASI", "rule:job_change")
    for m in re.finditer(LIFE_EVENT, text, _I):
        add(m.start(), m.end(), "QUASI", "rule:life_event")
    for m in re.finditer(ACTIVITY, text, _I):
        add(m.start(), m.end(), "ACTIVITY", "rule:activity", cause_cue=cue(text, *m.span()))
    for m in re.finditer(ACTIVITY_FRAME, text, _I):
        g = "act" if m.group("act") else "act2"
        add(*m.span(g), "ACTIVITY", "rule:activity_frame", cause_cue=cue(text, *m.span(g)))
    for m in re.finditer(BRAND, text, _I):
        add(m.start(), m.end(), "ORG", "rule:brand")
    for m in re.finditer(TEAM, text, _I):
        add(m.start(), m.end(), "ORG", "rule:team")
    for m in re.finditer(TEAM_AMBIG, text, _I):
        window = text[max(0, m.start() - 120):m.end() + 120]
        if re.search(SPORTS_CONTEXT, window, _I):
            add(*m.span("t"), "ORG", "rule:team")
    if 1 in ideas:
        spans += _detect_extended(text, field, cue)
    return spans


# ---------------------------------------------------------------- idea 1: extended rules (study G)
# Kin without a possessive ("Older sister has ...", "Maternal grandmother"): notes drop the "my/his/her".
_KIN_BARE = (r"(?:grand)?(?:mother|father|ma|pa)|mom|dad|mama|papa|parents|sisters?|brothers?|siblings?|"
             r"grandparents?|great[- ]aunt|great[- ]uncle|aunts?|uncles?|cousins?|wife|husband|spouse|daughters?|"
             r"sons|twins?|kids|children|roommates?|boyfriend|girlfriend|fianc[ée]e?|friends?|significant other|"
             r"lady friend|partners?")
_KIN_MOD = r"(?:(?:maternal|paternal|older|younger|little|big|twin|other|oldest|youngest|eldest|step-?)\s+)*"
KIN_BARE = rf"\b(?:the\s+|a\s+)?{_KIN_MOD}(?:{_KIN_BARE})\b"
HOUSEHOLD = (r"\b(?:lives? (?:alone|with (?:\w+\s+){0,4}?(?:\w+))(?=[.,;!?]|\s+and\b|\s+in\b|$)|"
             r"married(?: with (?:\w+\s+)?children)?|(?:has|have) no siblings|do not have any siblings|"
             r"(?:been )?dating(?: for (?:\w+\s+){1,2}years?)?|(?:was |is )?adopted|"
             r"(?:expecting|about to have) a (?:newborn|new baby|baby)(?: baby)?|new baby|"
             r"(?:about to |going to )?be(?:come)? a (?:father|mother|dad|mom)|impending (?:father|mother)hood|"
             r"household (?:family )?members|family (?:get[- ]togethers|gatherings|support(?: system)?))\b")
_ACT_AGENT = r"runner|biker|cyclist|golfer|skier|swimmer|hiker|climber|gymnast|athlete|hunter|surfer|lifter"
_ACT_ANY = rf"{_alt(set(ACTIVITIES) | set(ACTIVITY_GENERIC))}|{_ACT_AGENT}"
HOBBY_3P = (rf"\b(?:enjoys|likes|loves|avid|frequent|keen|regular|big|passionate)\s+(?:to\s+)?(?:\w+\s+){{0,2}}?"
            rf"(?:{_ACT_ANY})\b|\b(?:{_ACT_AGENT})\b")
# Exercise routine: movement verb + amount/frequency ("bike 30 miles", "walks 30-minutes most days").
_MOVE = r"walk\w*|run|runs|running|jog\w*|ride|rides|riding|bike|bikes|biking|bicycle|cycl\w*|swim\w*|hik\w*|ski\w*"
HABIT = (rf"\b(?:power\s+)?(?:{_MOVE})\b[^.?!\n]{{0,40}}?\b(?:\d+(?:\s*(?:to|-)\s*\d+)?|one|two|three|four|five|six|"
         rf"ten|twenty|thirty|a few|a couple)\s*[- ]?(?:miles?|minutes?|hours?|times (?:a|per) (?:week|day)|"
         rf"days? (?:a|per|of the) week)\b|"
         rf"\b(?:power walking|long walks|morning walks|my walks|take walks|taking (?:lots of )?walks|go(?:es|ing)? for "
         rf"(?:a |long )?(?:run|walk|walks|bike ride)|ride (?:a|the|my) bike|rid(?:es|ing) (?:a|the|his|her|my|their) "
         rf"bike|hit(?:ting)? the gym|lift(?:s|ed)? weights|strength training|walking program|gym routine|"
         rf"gym membership|stretching (?:out )?every morning|outdoor stretching|very active|active lifestyle|"
         rf"sedentary lifestyle)\b")
WORK = (r"\b(?:works? (?:from|at) home|working (?:from|at) home|(?:sedentary|desk|office|physical|labor-intensive|"
        r"manual|construction|new|old) job|(?:at|for|from) work\b|stress(?:ed)? (?:at|with) work|"
        r"(?:work(?:ing)?|putting in) (?:more|longer|extra) hours|increased workload|co-?workers?|(?:my|your|his|her) "
        r"boss|(?:my|his|her|other) employe+s|short-staffed|onboarding|(?:my|his|her|your) job|(?:go|going|went) "
        r"(?:back )?to work|switching jobs|retiring|veteran|workaholic|typing all day|sit(?:s|ting)? (?:at|behind) "
        r"(?:a|her|his|my|the) desk|(?:works|working) at a desk|standing all day|in the office|computer screen all day|"
        r"employed in \w+|studying (?:to be )?(?:a |an )?\w+)\b")
LIFE_EXT = (r"\b(?:(?:renovat|remodel)\w*\s+(?:the |my |his |her |our |your |their )?\w+|man cave|"
            r"construction on (?:the |their |his |her |my |our )?\w+|home renovations?|"
            r"(?:recently )?travel(?:ed|led|ing)? (?:internationally|abroad|overseas|all over (?:the )?\w+)|"
            r"(?:a )?trip planned|planning a trip|(?:long|business|work) trip|due to travel|prior to traveling|"
            r"(?:bought|purchased) a (?:new )?(?:house|home)|moved (?:out )?(?:in|to) the city|surprise party|"
            r"having people over|weekend plans?|trade show|convention)\b")


def _detect_extended(text: str, field: str, cue) -> list[Span]:
    spans = []
    for pat, label, cued in ((KIN_BARE, "FAMILY", False), (HOUSEHOLD, "HOUSEHOLD", False), (HOBBY_3P, "ACTIVITY", True),
                             (HABIT, "ACTIVITY", True), (WORK, "WORK", False), (LIFE_EXT, "QUASI", False)):
        for m in re.finditer(pat, text, _I):
            extra = {"cause_cue": cue(text, *m.span())} if cued else {}
            spans.append(Span(field, m.start(), m.end(), m.group(0), label, f"rule:ext_{label.lower()}", extra=extra))
    return spans
