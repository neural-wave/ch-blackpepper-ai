"""Code-conditioned relevance: how strongly does a text span support the encounter's ICD-10 codes?

Signals:
- lexical hit on the code's concept set (description, ICD ancestors, inclusion terms, curated expansions)
- SapBERT cosine similarity to that concept set
- same ICD-10 category as the span's nearest code in the full ICD-10-CM index
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import simple_icd_10_cm as cm
import torch

from .cache import EmbeddingStore
from .config import CACHE_DIR
from .models import load_hf

EXPANSIONS = json.loads((Path(__file__).parent / "resources" / "code_expansions.json").read_text())


class Embedder:
    def __init__(self, model_name: str, device: str, persist: bool = False):
        self.tok, self.model = load_hf(model_name, "base", device)
        self.model_name, self.device = model_name, device
        self._cache: dict[str, np.ndarray] = {}
        self._store = EmbeddingStore(model_name) if persist else None

    def release(self):
        """Drop the model (frees GPU memory for the LLM stage); reloaded on the next uncached call."""
        self.tok = self.model = None

    def __call__(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if self._store and missing:
            self._cache.update(self._store.get_many(missing))
            missing = [t for t in missing if t not in self._cache]
        fresh = {}
        if missing and self.model is None:
            self.tok, self.model = load_hf(self.model_name, "base", self.device)
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=32, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                cls = self.model(**enc).last_hidden_state[:, 0].float()  # SapBERT uses [CLS]
            cls = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
            self._cache.update(zip(batch, cls))
            fresh.update(zip(batch, cls))
        if self._store:
            self._store.put_many(fresh)
        dim = self.model.config.hidden_size if self.model is not None else 768
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, dim), dtype=np.float32)


class ICDIndex:
    """All ICD-10-CM categories/subcategories embedded once, cached on disk."""

    def __init__(self, embedder: Embedder):
        path = CACHE_DIR / "icd_index.npz"
        if path.exists():
            data = np.load(path, allow_pickle=False)
            self.codes, self.vecs = list(data["codes"]), data["vecs"]
        else:
            codes = [c for c in cm.get_all_codes(True) if cm.is_category_or_subcategory(c)]
            vecs = embedder([cm.get_description(c).lower() for c in codes])
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(path, codes=np.array(codes), vecs=vecs.astype(np.float16))
            self.codes, self.vecs = codes, vecs.astype(np.float16)
        self.vecs = self.vecs.astype(np.float32)

    def nearest(self, vecs: np.ndarray, k: int = 1) -> list[list[tuple[str, float]]]:
        sims = vecs @ self.vecs.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return [[(self.codes[j], float(sims[i, j])) for j in row] for i, row in enumerate(idx)]


def category(code: str) -> str:
    return code.replace(".", "")[:3].upper()


@lru_cache(maxsize=None)
def hierarchy(code: str) -> tuple[str | None, str | None]:
    """(block, chapter) of a code, or (None, None) if unknown."""
    dotted = code if "." in code or len(code) <= 3 else code[:3] + "." + code[3:]
    try:
        anc = cm.get_ancestors(dotted)
    except ValueError:
        try:
            anc = cm.get_ancestors(category(code))
        except ValueError:
            return None, None
    block = next((a for a in anc if cm.is_block(a)), None)
    chapter = next((a for a in anc if cm.is_chapter(a)), None)
    return block, chapter


def block_description(code: str) -> str | None:
    """Coarser ICD level (block, e.g. 'Diseases of appendix') used to generalize b-soft spans."""
    try:
        for anc in cm.get_ancestors(code):
            if cm.is_block(anc):
                desc = re.sub(r"\s*\([A-Z0-9-]+\)\s*$", "", cm.get_description(anc))
                return re.sub(r"^(?:other and unspecified|other|unspecified)\s+", "", desc, flags=re.I)
    except ValueError:
        return None
    return None


@lru_cache(maxsize=None)
def concept_terms(code: str, description: str) -> tuple[str, ...]:
    """Concept set for one code: description + ICD ancestors up to category + inclusion terms + expansions."""
    terms = [description]
    cat = category(code)
    dotted = code if "." in code or len(code) <= 3 else code[:3] + "." + code[3:]
    try:
        if cm.is_valid_item(dotted):
            for anc in [dotted] + cm.get_ancestors(dotted):
                if cm.is_category_or_subcategory(anc):
                    terms.append(cm.get_description(anc))
                    terms += cm.get_inclusion_term(anc)
    except ValueError:
        pass
    for key in (cat, cat[0]):
        terms += EXPANSIONS.get(key, [])
    seen, out = set(), []
    for t in terms:
        t = re.sub(r"\s+", " ", t.lower()).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return tuple(out)


class RelevanceScorer:
    def __init__(self, embedder: Embedder, index: ICDIndex, signals: tuple[int, ...] = (1, 2, 3, 4)):
        """signals: 1 = SapBERT similarity, 2 = lexical match, 3 = same ICD category, 4 = same block/chapter,
        5 = learned evidence from the train pseudo GT (evidence.py)."""
        self.embed = embedder
        self.index = index
        self.signals = set(signals)

    def code_concepts(self, codes: list[dict]) -> list[tuple[str, str]]:
        return [(c["code"], t) for c in codes for t in concept_terms(c["code"], c["description"])]

    def score(self, texts: list[str], codes: list[dict], encounter_id: str | None = None) -> list[dict]:
        """For each text: relevance score, best code, reason, nearest ICD code (any).
        encounter_id selects the out-of-fold evidence lexicon for train encounters (signal 5)."""
        if not texts:
            return []
        learned = [(0.0, None, "")] * len(texts)
        if 5 in self.signals and codes:
            from . import evidence
            learned = evidence.score(texts, [c["code"] for c in codes], self.embed, encounter_id)
        concepts = self.code_concepts(codes)
        span_vecs = self.embed([t.lower() for t in texts])
        nearest = self.index.nearest(span_vecs, k=1)
        concept_vecs = self.embed([t for _, t in concepts]) if concepts else None
        code_cats = {category(c["code"]) for c in codes}
        code_blocks = {hierarchy(c["code"])[0]: c["code"] for c in codes}
        code_chapters = {hierarchy(c["code"])[1]: c["code"] for c in codes}
        results = []
        for i, text in enumerate(texts):
            low = text.lower()
            near_code, near_sim = nearest[i][0]
            best = {"score": 0.0, "code": None, "reason": "no code-related signal", "nearest": near_code,
                    "nearest_sim": near_sim}
            if concepts and 1 in self.signals:
                sims = concept_vecs @ span_vecs[i]
                j = int(np.argmax(sims))
                best.update(score=float(sims[j]), code=concepts[j][0], reason=f"similar to '{concepts[j][1]}'")
            if concepts and 2 in self.signals:
                for code, term in concepts:
                    if len(term) >= 3 and re.search(rf"\b{re.escape(term)}\b", low) or (
                            len(low) >= 4 and re.search(rf"\b{re.escape(low)}\b", term)):
                        best.update(score=1.0, code=code, reason=f"matches code concept '{term}'")
                        break
            if 3 in self.signals and category(near_code) in code_cats and near_sim >= 0.6 and best["score"] < 1.0:
                best.update(score=max(best["score"], 0.95), code=next(c["code"] for c in codes
                                                                       if category(c["code"]) == category(near_code)),
                            reason=f"nearest ICD code {near_code} is in the same category")
            # Same ICD block (e.g. heart diseases) or chapter (organ system) as a coded diagnosis.
            near_block, near_chapter = hierarchy(near_code)
            if 4 in self.signals and near_sim >= 0.6 and best["score"] < 0.9:
                if near_block in code_blocks:
                    best.update(score=0.9, code=code_blocks[near_block],
                                reason=f"nearest ICD code {near_code} is in the same ICD block {near_block}")
                elif near_chapter in code_chapters and best["score"] < 0.6:
                    best.update(score=0.6, code=code_chapters[near_chapter],
                                reason=f"nearest ICD code {near_code} is in the same organ-system chapter")
            ev_score, ev_code, ev_reason = learned[i]
            if ev_score > best["score"]:
                best.update(score=ev_score, code=ev_code, reason=f"learned: {ev_reason}")
            results.append(best)
        return results
