"""Learned b/c decider (study H, docs/experiments.md): keep/remove for every detected clinical span
(DISEASE / CHEMICAL / ENTITY) = gradient-boosted classifier over the relevance signals + context features + the
zero-shot P(Yes) of a 4-bit LLM judge ("is this span evidence for one of the codes?"). Replaces t_keep.

The LLM runs alone on the GPU after SapBERT / de-id are released (Anonymizer.anonymize_all) -> peak VRAM ~3.5 GB.
Classifier bundle in models/decider.joblib; train encounters use out-of-fold models (same folds as smalltalk.py).

python -m anonymizer.decider --train                       # features + LLM on train, fit, save
python -m anonymizer.decider --train --features outputs/bc_llm/features_train.json \
    --llm-scores outputs/bc_llm/llm/qwen35_4b_q4_train.json  # reuse study outputs (seconds)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict

import numpy as np

from . import evidence
from .config import MODELS_DIR, ROOT, Config
from .relevance import RelevanceScorer, hierarchy

MODEL_PATH = MODELS_DIR / "decider.joblib"
N_FOLDS = 5
DECIDED = ("DISEASE", "CHEMICAL", "ENTITY")

# ------------------------------------------------------------------------------------------- features
NEG = re.compile(r"\b(no|not|denies|denied|deny|without|negative for|never|nor|free of)\b", re.I)
SECTIONS = [("cc", r"CHIEF COMPLAINT"), ("hpi", r"HISTORY OF PRESENT ILLNESS|SUBJECTIVE"),
            ("ros", r"REVIEW OF SY"), ("pe", r"PHYSICAL EXAM|^EXAM"), ("results", r"RESULTS|EKG"),
            ("ap", r"ASSESSMENT|PLAN|IMPRESSION|ORDERS"), ("meds", r"MEDICATIONS|ALLERGIES"),
            ("hist", r"MEDICAL HISTORY|PAST HISTORY|SURGICAL HISTORY|PAST MEDICAL"), ("social", r"SOCIAL HISTORY"),
            ("family", r"FAMILY HISTORY"), ("instr", r"INSTRUCTIONS"), ("vitals", r"VITALS")]
SEC_NAMES = [s for s, _ in SECTIONS] + ["other"]
SIGS = ["s1", "s2", "s3", "s4b", "s4c", "s5", "none"]
LABELS = ["DISEASE", "CHEMICAL", "clinical detail", "test", "procedure", "exam finding"]
# Need SapBERT (similarity, nearest ICD code, or a signal whose win depends on them); dropped in the no-embedding
# variants of the study.
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


def span_features(anon, state: dict) -> list[dict]:
    """One row per detected clinical span: key, position, text, feature dict, baseline decision."""
    enc, codes = state["enc"], state["codes"]
    eid = enc.get("encounter_id")
    clin = [s for s in state["clinical"] if s.label in DECIDED]
    if not clin:
        return []
    if not hasattr(anon, "_decider_scorers"):
        anon._decider_scorers = (RelevanceScorer(anon.embedder, anon.index, (1,)),
                                 RelevanceScorer(anon.embedder, anon.index, (2,)))
    sapbert, lexical = anon._decider_scorers
    texts = {f: enc[f] for f in ("dialogue", "note")}
    names = [s.text for s in clin]
    code_ids = [c["code"] for c in codes]
    s1 = sapbert.score(names, codes, eid)
    s2 = lexical.score(names, codes, eid)
    s5 = evidence.score(names, code_ids, anon.embedder, eid)
    lex = evidence.lexicon(evidence.fold_of(eid))
    secs = _sections(texts["note"])
    bounds = secs + [(len(texts["note"]), "end")]
    sec_text = defaultdict(str)
    for (st, n), (en, _) in zip(bounds, bounds[1:]):
        sec_text[n] += " " + texts["note"][st:en].lower()
    chapters = {hierarchy(c)[1] for c in code_ids}
    letters = {c[0] for c in code_ids}
    note_l, dia_l = texts["note"].lower(), texts["dialogue"].lower()
    rows = []
    for s, a, b, e in zip(clin, s1, s2, s5):
        low = s.text.lower()
        pat = rf"(?<!\w){re.escape(low)}(?!\w)"
        sent = re.split(r"[.!?\n]", texts[s.field][max(0, s.start - 80):s.start])[-1]
        near = s.extra.get("nearest", "")
        f = {
            "score": s.score or 0.0, "s1": a["score"], "s2_lex": float(b["score"] >= 1.0), "s5": e[0],
            "s5_exact": lex.lookup(evidence.normalize(s.text), code_ids)[0],
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
        rows.append({"key": f"{eid}|{s.field}|{s.start}|{s.end}", "eid": eid, "field": s.field, "start": s.start,
                     "end": s.end, "text": s.text, "f": f, "baseline_keep": s.action == "keep"})
    return rows


# ------------------------------------------------------------------------------------------ LLM judge
PROMPT = """ICD-10 codes assigned to this encounter:
{codes}

Excerpt from the {source}:
"{context}"

The clinical mention in question is marked with [[ ]]: "{span}".
Is this mention evidence for, a cause of, or a treatment of at least one of the ICD-10 codes above (keep it), \
or is it clinical information unrelated to all of these codes (remove it)?
Answer with exactly one word: Yes (evidence for a code) or No (unrelated)."""


def context(text: str, start: int, end: int, width: int = 220) -> str:
    """The span's sentence plus about one sentence either side, span marked with [[ ]]."""
    left, right = text[max(0, start - width):start], text[end:end + width]
    cuts = list(re.finditer(r"[.!?\n]\s", left))
    if len(cuts) >= 2:
        left = left[cuts[-2].end():]
    cuts = list(re.finditer(r"[.!?\n]\s", right))
    if len(cuts) >= 2:
        right = right[:cuts[1].start() + 1]
    return re.sub(r"\s+", " ", f"{left}[[{text[start:end]}]]{right}").strip()


class LLMJudge:
    """Zero-shot P(Yes) per span: one forward pass, softmax of the next-token logits {Yes, yes} vs {No, no}."""

    def __init__(self, model_id: str, device: str, quant: str | None = "4bit", dtype: str = "bf16"):
        import torch
        from transformers import AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"dtype": {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]}
        if quant and not device.startswith("cuda"):
            print(f"[decider] bitsandbytes {quant} needs CUDA; loading {model_id} in {dtype} on {device} "
                  "(fine for development, not VRAM-representative)", file=sys.stderr)
            quant = None
        if quant:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=kw["dtype"])
                                         if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
            kw["device_map"] = device
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        except Exception:
            # multimodal checkpoints (Qwen3.5, Gemma 3) without a causal-LM auto mapping
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        self.model = (model if quant else model.to(device)).eval()
        first = lambda ws: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws})
        self.yes, self.no = first(["Yes", "yes"]), first(["No", "no"])

    def prompt(self, enc: dict, codes: list[dict], row: dict) -> str:
        msg = PROMPT.format(codes="\n".join(f"- {c['code']}: {c['description']}" for c in codes),
                            source="doctor-patient conversation" if row["field"] == "dialogue" else "clinical note",
                            context=context(enc[row["field"]], row["start"], row["end"]), span=row["text"])
        # enable_thinking=False: Qwen3.x answers directly; other templates ignore the variable.
        return self.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def score(self, prompts: list[str], batch: int = 1, progress=None) -> tuple[np.ndarray, dict]:
        torch = self.torch
        p_yes, mass = np.zeros(len(prompts)), np.zeros(len(prompts))
        order = np.argsort([len(p) for p in prompts])       # similar lengths per batch -> little padding
        steps = range(0, len(order), batch)
        n_tok, t0 = 0, time.time()
        with torch.no_grad():
            for i in (progress(steps) if progress else steps):
                idx = order[i:i + batch]
                enc = self.tok([prompts[j] for j in idx], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.device)
                n_tok += int(enc["attention_mask"].sum())
                # logits_to_keep=1: full logits (batch x tokens x ~250k vocab) would dominate VRAM.
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
                logp = torch.log_softmax(logits.float(), dim=-1)
                y, n = torch.logsumexp(logp[:, self.yes], -1), torch.logsumexp(logp[:, self.no], -1)
                p_yes[idx] = torch.sigmoid(y - n).cpu().numpy()
                mass[idx] = (y.exp() + n.exp()).cpu().numpy()
        dt = time.time() - t0
        cuda = self.device.startswith("cuda")
        return p_yes, {"sec_per_span": round(dt / max(len(prompts), 1), 4), "tok_per_s": round(n_tok / max(dt, 1e-9)),
                       "seconds": round(dt, 1), "yes_no_mass_mean": round(float(mass.mean()), 4) if len(mass) else None,
                       "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None}

    def release(self):
        import gc
        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def llm_scores(states: list[dict], cfg: Config, device: str, progress=None) -> tuple[dict[str, float], dict]:
    judge = LLMJudge(cfg.llm_model, device, cfg.llm_quant)
    items = [(st, r) for st in states for r in st["features"]]
    prompts = [judge.prompt(st["enc"], st["codes"], r) for st, r in items]
    p, meta = judge.score(prompts, cfg.llm_batch, progress)
    judge.release()
    return {r["key"]: float(v) for (_, r), v in zip(items, p)}, meta


# ----------------------------------------------------------------------------------------- classifier
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _matrix(rows: list[dict], names: list[str], llm: dict[str, float]) -> np.ndarray:
    X = np.array([[r["f"][n] for n in names] for r in rows], dtype=float).reshape(len(rows), len(names))
    return np.c_[X, _logit([llm[r["key"]] for r in rows])]


def _new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    # early_stopping off: sklearn enables it only above 10k samples, i.e. for some fits and not others.
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def train_folds() -> dict[str, int]:
    """Same assignment as smalltalk.py / evidence.py: sorted train ids, i % 5."""
    from .data import load_encounters
    ids = sorted(e["encounter_id"] for e in load_encounters("train"))
    return {eid: i % N_FOLDS for i, eid in enumerate(ids)}


def gt_labels(split: str, rows: list[dict]) -> None:
    """Sets r["y"]: 1 = overlaps pseudo-GT code evidence (c); 0 = overlaps b or personal data (a); else None."""
    from .data import load_encounters
    encs = {e["encounter_id"]: e for e in load_encounters(split)}
    occ = defaultdict(list)
    for p in (ROOT / "pseudo_gt" / split).glob("*.json"):
        ann = json.loads(p.read_text())
        for s in ann["spans"]:
            text = encs[ann["encounter_id"]][s["field"]]
            for m in re.finditer(rf"(?<![\w\[]){re.escape(s['text'])}(?!\w)", text, re.IGNORECASE):
                occ[(ann["encounter_id"], s["field"])].append((m.start(), m.end(), s["category"]))
    for r in rows:
        cats = {c for s, e, c in occ.get((r["eid"], r["field"]), []) if s < r["end"] and r["start"] < e}
        r["y"] = 1 if "c" in cats else 0 if cats & {"a", "b"} else None


def fit(rows: list[dict], llm: dict[str, float], cfg: Config, llm_meta: dict | None = None) -> dict:
    names = sorted(rows[0]["f"])
    lab = [r for r in rows if r.get("y") is not None]
    X, y = _matrix(lab, names, llm), np.array([r["y"] for r in lab])
    fold_of = train_folds()
    f = np.array([fold_of[r["eid"]] for r in lab])
    bundle = {"names": names, "theta": cfg.decider_theta, "llm_model": cfg.llm_model, "llm_quant": cfg.llm_quant,
              "fold_of": fold_of, "folds": {k: _new_model().fit(X[f != k], y[f != k]) for k in range(N_FOLDS)},
              "full": _new_model().fit(X, y), "n_train": len(lab), "llm_meta": llm_meta}
    return bundle


def load_bundle(cfg: Config) -> dict:
    import joblib
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} missing: copy the trained bundle there, run `python -m anonymizer.decider --train`, or use --no-decider")
    bundle = joblib.load(MODEL_PATH)
    if bundle["llm_model"] != cfg.llm_model:
        raise SystemExit(f"decider was trained with {bundle['llm_model']}, config uses {cfg.llm_model}: retrain")
    return bundle


def decide(states: list[dict], cfg: Config, device: str, progress=None) -> dict[str, bool]:
    """keep/remove for every detected clinical span of every state; sets state["p_keep"] for the log."""
    bundle = load_bundle(cfg)
    if progress is None:
        from tqdm import tqdm
        progress = lambda it: tqdm(it, desc=f"llm judge (batch {cfg.llm_batch})", file=sys.stderr)
    llm, meta = llm_scores(states, cfg, device, progress)
    print(f"[decider] LLM {cfg.llm_model} ({cfg.llm_quant}): {meta}", file=sys.stderr)
    keep = {}
    for st in states:
        rows = st["features"]
        if not rows:
            continue
        eid = st["enc"].get("encounter_id")
        fold = bundle["fold_of"].get(eid)
        model = bundle["folds"][fold] if fold is not None else bundle["full"]
        p = model.predict_proba(_matrix(rows, bundle["names"], llm))[:, 1]
        st["p_keep"] = {r["key"]: float(v) for r, v in zip(rows, p)}
        keep.update({r["key"]: bool(v >= cfg.decider_theta) for r, v in zip(rows, p)})
    return keep


def main(argv=None):
    import joblib
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", action="store_true", required=True)
    p.add_argument("--features", default=None, help="reuse features_train.json from experiments/bc_llm")
    p.add_argument("--llm-scores", default=None, help="reuse llm/<tag>_train.json from experiments/bc_llm")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    cfg = Config(device=a.device)
    if a.features and a.llm_scores:
        rows = json.loads(open(a.features).read())
        L = json.loads(open(a.llm_scores).read())
        llm, meta = L["p"], L.get("meta")
        if meta and meta.get("model") != cfg.llm_model:
            raise SystemExit(f"{a.llm_scores} is from {meta.get('model')}, config uses {cfg.llm_model}")
    else:
        from tqdm import tqdm
        from .data import load_encounters
        from .pipeline import Anonymizer
        anon = Anonymizer(cfg)
        states = [anon.prepare(e) for e in tqdm(load_encounters("train"), desc="features train")]
        anon.release_models()
        llm, meta = llm_scores(states, cfg, anon.device, lambda it: tqdm(it, desc="llm"))
        rows = [r for st in states for r in st["features"]]
    if any("y" not in r for r in rows):
        gt_labels("train", rows)
    bundle = fit(rows, llm, cfg, meta)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"decider: {bundle['n_train']} labelled train spans, {len(bundle['names'])} features + LLM, "
          f"theta {bundle['theta']} -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
