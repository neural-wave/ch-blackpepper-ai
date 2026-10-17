"""Shared paths, folds, pseudo-GT labels and metrics for the b/c decider study (see README.md here)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from anonymizer.config import ROOT
from anonymizer.data import load_encounters
from anonymizer.evaluate_gt import evaluate, load_gt

OUT = ROOT / "outputs" / "bc_llm"
N_FOLDS = 5
# Score weight of one b/a occurrence relative to one c occurrence on train (privacy pools ~4.3k a+b occurrences,
# c_kept ~11.7k); a keep is worth it only above ~73 % precision. Used for span-level cost only.
W_B = 2.7

LLM_PRESETS = {
    "qwen35_2b": "Qwen/Qwen3.5-2B",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "medgemma15_4b": "google/medgemma-1.5-4b-it",
}


def folds() -> dict[str, int]:
    """Same assignment as smalltalk.py / evidence.py: sorted train ids, i % 5."""
    ids = sorted(e["encounter_id"] for e in load_encounters("train"))
    return {eid: i % N_FOLDS for i, eid in enumerate(ids)}


def gt_occurrences(split: str) -> dict:
    """(encounter_id, field) -> [(start, end, category)] for every occurrence of every GT span."""
    encs = {e["encounter_id"]: e for e in load_encounters(split)}
    occ = defaultdict(list)
    for p in (ROOT / "pseudo_gt" / split).glob("*.json"):
        ann = json.loads(p.read_text())
        for s in ann["spans"]:
            text = encs[ann["encounter_id"]][s["field"]]
            for m in re.finditer(rf"(?<![\w\[]){re.escape(s['text'])}(?!\w)", text, re.IGNORECASE):
                occ[(ann["encounter_id"], s["field"])].append((m.start(), m.end(), s["category"]))
    return occ


def label(row: dict, occ: dict) -> int | None:
    """1 = overlaps code evidence (c); 0 = overlaps b or personal data (a) — keeping those costs privacy."""
    cats = {c for s, e, c in occ.get((row["eid"], row["field"]), []) if s < row["end"] and row["start"] < e}
    return 1 if "c" in cats else 0 if cats & {"a", "b"} else None


def metrics(split: str, outputs: list[dict]) -> dict:
    orig = {e["encounter_id"]: e for e in load_encounters(split)}
    res, _ = evaluate(orig, {e["encounter_id"]: e for e in outputs}, load_gt(split))
    res["score"] = round((res["privacy"] + res["c_kept"]) / 2, 4)
    return res


def span_cost(y: np.ndarray, keep: np.ndarray) -> float:
    return float(W_B * np.sum(keep & (y == 0)) + np.sum(~keep & (y == 1)))


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
