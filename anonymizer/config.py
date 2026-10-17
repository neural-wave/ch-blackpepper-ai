"""Tunable knobs of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"


@dataclass
class Config:
    # Relevance thresholds = the privacy-utility knob.
    t_keep: float = 0.72        # >= keep verbatim (category c)
    t_remove: float = 0.50      # <  remove (b-hard); in between: generalize (b-soft)
    # Symptoms/findings (ICD chapter R) are rarely identifying and often the only evidence for codes.
    keep_symptoms: bool = True
    # Relevance signals: 1 SapBERT similarity, 2 lexical match, 3 same ICD category, 4 same ICD block/chapter.
    signals: tuple[int, ...] = (1, 2, 3, 4)
    perturb_pct: float = 0.03   # relative noise on measurements
    use_deid_ner: bool = True
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    deid_model: str = "obi/deid_roberta_i2b2"
    scispacy_model: str = "en_ner_bc5cdr_md"
    device: str = "auto"        # auto | cpu | mps | cuda
    seed: int = 13
