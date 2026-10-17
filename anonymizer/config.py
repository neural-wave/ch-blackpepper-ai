"""Tunable knobs of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"


@dataclass
class Config:
    # Defaults = current baseline (docs/leaderboard.md).
    # Relevance thresholds = the privacy-utility knob.
    t_keep: float = 0.50        # >= keep verbatim (category c)
    t_remove: float = 0.50      # <  remove (b-hard); in between: generalize (b-soft)
    # True keeps symptoms/findings (ICD chapter R) regardless of score; off trades ~0.05 c_kept for ~+0.13 privacy
    # (docs/experiments.md).
    keep_symptoms: bool = False
    # Relevance signals: 1 SapBERT similarity, 2 lexical match, 3 same ICD category, 4 same ICD block/chapter,
    # 5 learned evidence from the train pseudo GT (evidence.py).
    signals: tuple[int, ...] = (1, 2, 3, 4, 5)
    perturb_pct: float = 0.03   # relative noise on measurements
    use_deid_ner: bool = True
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    deid_model: str = "obi/deid_roberta_i2b2"
    scispacy_model: str = "en_ner_bc5cdr_md"
    # Untyped NER for what BC5CDR misses (procedures, labs, exam findings); None = BC5CDR only.
    entity_model: str | None = "en_core_sci_md"
    # Untyped entities of the kinds in entity_sim_gate ("clinical detail", "test", "procedure", "exam finding") are
    # acted on only if this close to some ICD concept; below it they stay as they are ("yard", "walking").
    entity_min_sim: float = 0.65
    entity_sim_gate: tuple[str, ...] = ("clinical detail", "test", "procedure", "exam finding")
    # What the gate compares with entity_min_sim: "nearest_sim" (closest of all ICD codes) or "relevance" (score
    # against this encounter's codes).
    entity_gate_on: str = "nearest_sim"
    # Small-talk filter (anonymizer/smalltalk.py): dialogue units above this probability are removed whole.
    smalltalk: bool = True
    smalltalk_threshold: float = 0.6
    # Reuse model outputs (spans, embeddings) across runs: .cache/spans.sqlite, .cache/embeddings.sqlite.
    cache: bool = True
    device: str = "auto"        # auto | cpu | mps | cuda
    seed: int = 13
