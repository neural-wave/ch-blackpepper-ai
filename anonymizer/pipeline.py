"""End-to-end anonymization of one encounter (see docs/architecture.md for the stages)."""

from __future__ import annotations

import copy
import random
import re
from collections import defaultdict

from . import cache, medical, phi_context, phi_ner, phi_rules, smalltalk
from .config import Config
from .models import resolve_device
from .relevance import Embedder, ICDIndex, RelevanceScorer, block_description, category
from .resources.lexicons import AMBIGUOUS_NAMES, EPONYM_CONTEXT, EPONYMS, GENERIC_TERMS, NOT_NAMES
from .spans import (CLINICAL_CODE, CLINICAL_OTHER, DIRECT_IDENTIFIERS, GENERALIZE, KEEP, PERSONAL, PERTURB,
                    PSEUDONYMIZE, REMOVE, Span)

FIELDS = ("dialogue", "note")
TAGS = {"DATE": "[DATE]", "DOB": "[DATE]", "YEAR": "[YEAR]", "PHONE": "[PHONE]", "EMAIL": "[EMAIL]", "URL": "[URL]",
        "ID": "[ID]", "ADDRESS": "[ADDRESS]", "LOC": "[LOCATION]", "HOSP": "[HOSPITAL]", "ORG": "[ORGANIZATION]",
        "OTHERPHI": "[REDACTED]", "SMALLTALK": "[small talk]"}
# Higher = wins overlaps.
PRIORITY = {"smalltalk": 5, "direct": 4, "code": 3, "personal": 2, "clinical": 1, "measure": 0}
# Personal spans that could themselves be code evidence (e.g. Y92 place, Y93 activity) get a relevance guard.
GUARDED = {"LOC", "HOSP", "ORG", "OTHERPHI", "OCCUPATION", "ACTIVITY", "PET", "WORKPLACE", "WORK"}
# Detected once in an encounter -> every other mention of the same string is treated the same way.
PROPAGATED = {"LOC", "HOSP", "ORG", "OCCUPATION", "WORKPLACE"}
# An activity/pet next to an injury or allergy cue is the mechanism of a coded problem when one of these is coded.
CAUSE_CODES = {"ACTIVITY": ("S", "T", "M", "W", "X", "V", "Y"),
               "PET": ("J30", "J45", "L23", "L50", "T78", "W53", "W54", "W55", "A69")}
# Clinical spans the learned b/c decider (decider.py) acts on.
DECIDED_LABELS = ("DISEASE", "CHEMICAL", "ENTITY")


def span_key(encounter_id: str | None, s: Span) -> str:
    return f"{encounter_id}|{s.field}|{s.start}|{s.end}"


GENERALIZED_TAGS = {"FAMILY": "[family member]", "PET": "[pet]", "ACTIVITY": "[activity]", "WORKPLACE": "[workplace]",
                    "QUASI": "[life event]", "WORK": "[work detail]", "HOUSEHOLD": "[household detail]",
                    "SOCIAL": "[social history detail]"}
# Idea 3: labels whose mentions are copied across fields (by stem for activities).
PROPAGATED_EXT = {"ACTIVITY", "QUASI", "WORK", "HOUSEHOLD"}
# Idea 2: a SOCIAL HISTORY sentence about substances stays when a substance-use code is present.
SUBSTANCE = r"\b(?:smok\w*|tobacco|cigar\w*|vap\w*|alcohol|drink\w*|beer|wine|liquor|drugs?|marijuana|cannabis)\b"
SUBSTANCE_CODES = ("F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "Z72", "Z87.891", "Z71.4",
                   "Z71.6", "T65.2")


class Anonymizer:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.device = resolve_device(self.cfg.device)
        self.embedder = Embedder(self.cfg.sapbert_model, self.device, persist=self.cfg.cache)
        self.index = ICDIndex(self.embedder)
        self.scorer = RelevanceScorer(self.embedder, self.index, self.cfg.signals)

    # ------------------------------------------------------------------ public
    def anonymize(self, enc: dict, codes: list[dict] | None = None) -> tuple[dict, dict]:
        """Return (anonymized encounter, decision log). `codes` overrides the encounter's codes (no-code mode).
        Uses the relevance threshold for b/c; anonymize_all also applies the learned decider."""
        return self.finalize(self.prepare(enc, codes))

    def anonymize_all(self, encounters: list[dict], codes: list[list[dict] | None] | None = None,
                      progress=None) -> list[tuple[dict, dict]]:
        """All encounters in phases so only one GPU model is resident at a time (<= 4 GB VRAM): detection with
        SapBERT + de-id NER, then (decider on) those released and the 4-bit LLM judge loaded alone."""
        codes = codes or [None] * len(encounters)
        it = progress(list(zip(encounters, codes))) if progress else zip(encounters, codes)
        states = [self.prepare(e, c) for e, c in it]
        keep = None
        if self.cfg.decider:
            from . import decider
            self.release_models()
            keep = decider.decide(states, self.cfg, self.device)
        return [self.finalize(st, keep) for st in states]

    def release_models(self):
        """Free SapBERT and the de-id model from the GPU (reloaded lazily if needed again)."""
        import gc
        import torch
        from .models import load_hf
        self.embedder.release()
        load_hf.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------- prepare (decision-independent) / finalize
    def prepare(self, enc: dict, codes: list[dict] | None = None) -> dict:
        """Everything that does not depend on the b/c decision: detections, scores, personal data, names,
        small-talk probabilities."""
        codes = enc["icd10_codes"] if codes is None else codes
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
        state = {"enc": enc, "codes": codes, "clinical": clinical, "personal": personal, "names": names,
                 "registry": dict(self._registry),
                 "smalltalk": smalltalk.predict(texts["dialogue"], self._eid) if self.cfg.smalltalk else []}
        if self.cfg.decider:
            from . import decider
            state["features"] = decider.span_features(self, state)
        for s in clinical + personal:
            s.extra.pop("match", None)      # re.Match: not picklable, never read when writing the output
        return state

    def finalize(self, state: dict, keep: dict[str, bool] | None = None) -> tuple[dict, dict]:
        """Decision-dependent tail. `keep` (span key "eid|field|start|end" -> bool) overrides the threshold
        decision for detected clinical spans. The state is not modified (replayable)."""
        enc, codes = state["enc"], state["codes"]
        p_keep = state.get("p_keep", {})
        eid = enc.get("encounter_id")
        texts = {f: enc[f] for f in FIELDS}
        clinical, personal, names = copy.deepcopy((state["clinical"], state["personal"], state["names"]))
        if keep is not None:
            for s in clinical:
                key = span_key(eid, s)
                k = keep.get(key) if s.label in DECIDED_LABELS else None
                why = f"decider p={p_keep[key]:.2f}" if key in p_keep else "b/c decider"
                if k is True:
                    s.category, s.action, s.reason = CLINICAL_CODE, KEEP, f"code evidence ({why})"
                    s.extra["prio"] = PRIORITY["code"]
                elif k is False:
                    s.category, s.action = CLINICAL_OTHER, REMOVE
                    s.reason = f"unrelated to encounter codes ({why}); removed"
                    s.extra["prio"] = PRIORITY["clinical"]
        spans = clinical + personal
        if 2 in self.cfg.a_ideas:
            spans += self._social_history_spans(texts["note"], spans, codes)
        if self.cfg.smalltalk:
            spans += self._smalltalk_spans(texts["dialogue"], spans, eid, preds=state["smalltalk"])
        chosen = self._resolve(spans)

        rng = random.Random(f"{self.cfg.seed}:{eid}")
        out = dict(enc)
        for f in FIELDS:
            out[f] = self._apply(texts[f], [s for s in chosen if s.field == f], rng)
        self._registry = state["registry"]
        out, leaks = self._sweep_names(out, names)

        evidence = defaultdict(int)
        for s in chosen:
            if s.category == CLINICAL_CODE and s.matched_code:
                evidence[s.matched_code] += 1
        log = {
            "encounter_id": eid,
            "decisions": [s.to_log() for s in sorted(chosen, key=lambda s: (s.field, s.start))],
            "codes_without_evidence": [c["code"] for c in codes if not evidence.get(c["code"])],
            "name_leaks_swept": leaks,
        }
        return out, log

    # ------------------------------------------------------------ [1] keep mask
    def _clinical_spans(self, texts: dict, codes: list[dict]) -> list[Span]:
        spans = []
        for f, text in texts.items():
            typed = cache.spans(medical.detect, text, f, self.cfg.scispacy_model, enabled=self.cfg.cache)
            spans += typed
            if self.cfg.entity_model:
                spans += cache.spans(medical.detect_entities, text, f, self.cfg.entity_model, typed,
                                     enabled=self.cfg.cache)
        scores = self.scorer.score([s.text for s in spans], codes, self._eid)
        for s, r in zip(spans, scores):
            s.score, s.matched_code = r["score"], r["code"]
            s.extra["nearest"], s.extra["nearest_sim"] = r["nearest"], r["nearest_sim"]
            gate_value = r["score"] if self.cfg.entity_gate_on == "relevance" else r["nearest_sim"]
            if (s.label == "ENTITY" and s.extra["kind"] in self.cfg.entity_sim_gate and r["score"] < self.cfg.t_keep
                    and gate_value < self.cfg.entity_min_sim):
                s.extra["drop"] = True
                continue
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
        spans = [s for s in spans if not s.extra.get("drop")]
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
            spans += phi_context.detect(text, f, frozenset(self.cfg.a_ideas))
            if self.cfg.use_deid_ner:
                spans += cache.spans(phi_ner.detect, text, f, self.cfg.deid_model, self.device, enabled=self.cfg.cache)
        for s in spans:
            s.category = PERSONAL
            direct = s.label in DIRECT_IDENTIFIERS
            s.extra["prio"] = PRIORITY["direct"] if direct else PRIORITY["personal"]
            if s.label == "AGE":
                s.action, s.reason = GENERALIZE, "age generalized to a 10-year band"
            elif s.label == "OCCUPATION":
                s.action, s.reason = GENERALIZE, "occupation is a quasi-identifier; generalized to sector"
            elif s.label in GENERALIZED_TAGS:
                s.action, s.reason = GENERALIZE, f"{s.label.lower()} is a quasi-identifier ({s.source}); generalized"
            else:
                s.action, s.reason = REMOVE, f"{s.label.lower()} is personal data ({s.source})"
        return spans

    def _guard_personal(self, spans: list[Span], codes: list[dict]):
        # A cue at any mention marks the activity as the mechanism for the whole encounter (the cue is often
        # several turns away from later mentions).
        cued = {_stem(s.text) for s in spans if s.extra.get("cause_cue")}
        for s in spans:
            prefixes = CAUSE_CODES.get(s.label)
            if prefixes and _stem(s.text) in cued:
                code = next((c["code"] for c in codes if c["code"].startswith(prefixes)), None)
                if code:
                    s.category, s.action, s.matched_code = CLINICAL_CODE, KEEP, code
                    s.reason = f"{s.label.lower()} next to an injury/exposure cue: mechanism for {code}; kept"
                    s.extra["prio"] = PRIORITY["code"]
        guarded = [s for s in spans if s.label in GUARDED and s.action != KEEP]
        for s, r in zip(guarded, self.scorer.score([s.text for s in guarded], codes, self._eid)):
            if r["score"] >= self.cfg.t_keep:
                s.category, s.action, s.score, s.matched_code = CLINICAL_CODE, KEEP, r["score"], r["code"]
                s.reason = f"{s.label.lower()} kept: evidence for {r['code']} ({r['reason']})"
                s.extra["prio"] = PRIORITY["code"]

    def _smalltalk_spans(self, text: str, spans: list[Span], encounter_id: str | None,
                         preds: list | None = None) -> list[Span]:
        """Whole dialogue units classified as small talk, unless they hold anything kept for the codes."""
        out = []
        for s, e, p in (smalltalk.predict(text, encounter_id) if preds is None else preds):
            if p < self.cfg.smalltalk_threshold:
                continue
            inside = [x for x in spans if x.field == "dialogue" and x.start < e and s < x.end]
            if any(x.category == CLINICAL_CODE or x.start < s or x.end > e for x in inside):
                continue
            sp = Span("dialogue", s, e, text[s:e], "SMALLTALK", "smalltalk", category=PERSONAL, action=REMOVE,
                      reason=f"small talk (p={p:.2f}): personal context without clinical content; removed")
            sp.extra["prio"] = PRIORITY["smalltalk"]
            out.append(sp)
        return out

    @staticmethod
    def _social_history_spans(text: str, spans: list[Span], codes: list[dict]) -> list[Span]:
        """Idea 2: SOCIAL HISTORY sentences in the note with no code evidence are generalized whole."""
        m = re.search(r"(?m)^SOCIAL HISTORY\s*$", text)
        if not m:
            return []
        end = re.search(r"(?m)^[A-Z][A-Z /&]{3,}\s*$", text[m.end():])
        sec_end = m.end() + end.start() if end else len(text)
        substance_coded = any(c["code"].startswith(SUBSTANCE_CODES) for c in codes)
        out = []
        for sm in re.finditer(r"[^.!?\n]+[.!?]?", text[m.end():sec_end]):
            s, e = m.end() + sm.start(), m.end() + sm.end()
            sent = text[s:e]
            if len(sent.strip()) < 3:
                continue
            if substance_coded and re.search(SUBSTANCE, sent, re.IGNORECASE):
                continue
            inside = [x for x in spans if x.field == "note" and x.start < e and s < x.end]
            if any(x.category == CLINICAL_CODE or x.start < s or x.end > e for x in inside):
                continue
            lead = len(sent) - len(sent.lstrip())
            sp = Span("note", s + lead, e, sent.strip(), "SOCIAL", "rule:social_history", category=PERSONAL,
                      action=GENERALIZE, reason="social history sentence without code evidence; generalized")
            sp.extra["prio"] = PRIORITY["smalltalk"]
            out.append(sp)
        return out

    @staticmethod
    def _propagate_ext(texts: dict, spans: list[Span]) -> list[Span]:
        """Idea 3: activities and life events said in the dialogue are restated in the note (and back)."""
        keys = {}
        for s in spans:
            if s.label not in PROPAGATED_EXT or s.action == KEEP or len(s.text) < 4 or len(s.text.split()) > 4:
                continue
            t = s.text.lower().strip()
            # Single-word lexicon hobbies match by stem ("golfing" -> golf, golfer); generic movements ("walking")
            # only verbatim, since they are also gait/exam findings.
            by_stem = s.label == "ACTIVITY" and " " not in t and s.source == "rule:activity"
            pat = rf"{re.escape(_stem(t))}(?:ing|er|ers|s|es|ed)?" if by_stem else re.escape(t)
            if s.label == "ACTIVITY" and not by_stem and " " not in t:
                continue
            if t in GENERIC_TERMS or len(_stem(t)) < 3:
                continue
            keys.setdefault(pat, s)
        out = []
        for pat, src in keys.items():
            for f, text in texts.items():
                for m in re.finditer(rf"(?<![\w\[]){pat}(?!\w)", text, re.IGNORECASE):
                    if any(o.field == f and o.start < m.end() and m.start() < o.end for o in spans + out):
                        continue
                    out.append(Span(f, m.start(), m.end(), m.group(0), src.label, "propagated_ext",
                                    category=src.category, action=src.action,
                                    reason=f"same {src.label.lower()} as a detected mention",
                                    extra={k: v for k, v in src.extra.items() if k != "match"}))
        return out

    @staticmethod
    def _propagate(texts: dict, spans: list[Span]) -> list[Span]:
        """Mentions the detectors missed ("you're a teacher" after "i work as a teacher") follow the first one."""
        seen = {}
        for s in spans:
            t = s.text.lower().strip()
            if (s.label in PROPAGATED and s.action != KEEP and len(t) >= 3 and t not in GENERIC_TERMS
                    and t not in NOT_NAMES):
                seen.setdefault(t, s)
        out = []
        for t, src in seen.items():
            for f, text in texts.items():
                for m in re.finditer(rf"(?<!\w){re.escape(t)}s?(?!\w)", text, re.IGNORECASE):
                    if any(o.field == f and o.start <= m.start() < o.end for o in spans + out):
                        continue
                    sp = Span(f, m.start(), m.end(), m.group(0), src.label, "propagated", category=src.category,
                              action=src.action, reason=f"same string as a detected {src.label.lower()}",
                              extra={k: v for k, v in src.extra.items() if k != "match"})
                    out.append(sp)
        return out

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
                    if (tok in AMBIGUOUS_NAMES and not m.group(0)[:1].isupper()
                            and not any(s.overlaps(sp) for s in seeds) and _common_word_use(tok, text, m)):
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
            return f"[{s.extra['sector']} worker]" if "sector" in s.extra else "[occupation]"
        if s.label in GENERALIZED_TAGS:
            return GENERALIZED_TAGS[s.label]
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
        if s.label == "ENTITY":
            return f"[{s.extra['kind']}]"
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


def _stem(text: str) -> str:
    """'hiking' / 'hikes' / 'hiked' -> 'hik': mentions of one activity share a key."""
    return re.sub(r"(?:ing|ed|es|s)$", "", text.lower().strip())


# Ambiguous names whose ordinary sense dominates ("may", "will"): never propagated in lowercase.
_COMMON_SENSE_NAMES = {"may", "will", "june", "april", "august", "art", "sky", "stone", "win", "rich", "max", "miles",
                       "joy", "hope", "faith", "grace", "summer", "dawn", "pat", "ray", "bill", "chase", "drew",
                       "sunny", "autumn"}


def _common_word_use(tok: str, text: str, m: re.Match) -> bool:
    """True when a lowercase ambiguous name is likely the ordinary word ("a mark on", "to be frank")."""
    if tok in _COMMON_SENSE_NAMES:
        return True
    before = re.findall(r"[a-z']+", text[max(0, m.start() - 20):m.start()].lower())[-1:]
    return bool(before) and before[0] in {"a", "the", "this", "that", "be", "any", "no", "birth", "check", "question"}


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
