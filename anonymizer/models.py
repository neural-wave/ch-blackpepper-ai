"""Lazy, cached model loading so each model is loaded once per process."""

from __future__ import annotations

from functools import lru_cache

import torch


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=None)
def load_spacy(name: str):
    import spacy
    return spacy.load(name, disable=["lemmatizer"])


@lru_cache(maxsize=None)
def load_hf(name: str, kind: str, device: str):
    from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    cls = AutoModelForTokenClassification if kind == "token" else AutoModel
    model = cls.from_pretrained(name)
    # fp16 halves VRAM; keeps de-id RoBERTa-large (~355M params) well under the 4 GB budget.
    if device == "cuda":
        model = model.half()
    return tok, model.to(device).eval()
