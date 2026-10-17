"""Clinical de-identification NER (obi/deid_roberta_i2b2, trained on i2b2 2014 notes)."""

from __future__ import annotations

import torch

from .models import load_hf
from .spans import Span

# i2b2 label -> our label
LABEL_MAP = {
    "PATIENT": "NAME", "STAFF": "NAME", "AGE": "AGE", "DATE": "DATE", "EMAIL": "EMAIL", "PHONE": "PHONE",
    "ID": "ID", "LOC": "LOC", "HOSP": "HOSP", "PATORG": "ORG", "OTHERPHI": "OTHERPHI",
}
MAX_TOKENS = 510
STRIDE = 128


def detect(text: str, field: str, model_name: str, device: str) -> list[Span]:
    tok, model = load_hf(model_name, "token", device)
    id2label = model.config.id2label
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    # Sliding windows; each token takes the label from the window where it is most central.
    labels = ["O"] * len(ids)
    best_margin = [-1] * len(ids)
    step = MAX_TOKENS - STRIDE
    for w0 in range(0, max(len(ids), 1), step):
        window = ids[w0:w0 + MAX_TOKENS]
        if not window:
            break
        inp = torch.tensor([[tok.cls_token_id] + window + [tok.sep_token_id]], device=model.device)
        with torch.no_grad():
            pred = model(input_ids=inp).logits[0, 1:-1].argmax(-1).tolist()
        for i, p in enumerate(pred):
            margin = min(i, len(window) - 1 - i)
            if margin > best_margin[w0 + i]:
                best_margin[w0 + i], labels[w0 + i] = margin, id2label[p]
        if w0 + MAX_TOKENS >= len(ids):
            break

    # BILOU -> spans: merge consecutive tokens of the same entity type.
    spans, cur = [], None
    for (s, e), lab in zip(offsets, labels):
        if lab == "O":
            cur = None
            continue
        prefix, typ = lab.split("-", 1)
        contiguous = cur is not None and cur["type"] == typ and s - cur["end"] <= 1
        if prefix in ("B", "U") or not contiguous:
            cur = {"type": typ, "start": s, "end": e}
            spans.append(cur)
        else:
            cur["end"] = e
    out = []
    for sp in spans:
        s, e = sp["start"], sp["end"]
        while s < e and not text[s].isalnum():
            s += 1
        # Labels sit on the first sub-word only ("and|rew"): snap to whole words.
        while s > 0 and text[s - 1].isalnum():
            s -= 1
        while e < len(text) and text[e].isalnum():
            e += 1
        if s >= e:
            continue
        role = {"PATIENT": "PATIENT", "STAFF": "STAFF"}.get(sp["type"])
        out.append(Span(field, s, e, text[s:e], LABEL_MAP.get(sp["type"], "OTHERPHI"), f"ner:{sp['type'].lower()}",
                        extra={"role": role}))
    return out
