# Anonymization_Bosch_NeuralWave

Bosch @ Neural Wave hackathon: anonymize doctor–patient encounters (dialogue + note) while preserving ICD-10 code
utility. Must run on-prem, ≤4 GB VRAM at eval. Strategy in `docs/strategy.md`, pipeline in `docs/architecture.md`.

## Layout
- `task/` — challenge PDF
- `data/patient_doctor_encounters.jsonl` (train, 140) / `data/eval_patient_doctor_encounters.jsonl` (eval, 67)
  - fields: `encounter_id`, `dataset` (aci / virtassist / virtscribe), `dialogue`, `note`, `icd10_codes[{code, description}]`
  - ACI-Bench + ICD annotations (amazon-science/toward-clinical-coding-verification-adaptation, CC BY-NC 4.0)
- `anonymizer/`
  - `cli.py` — entry point; `pipeline.py` — `Anonymizer` (stages 1–5), `no_code_predictions`
  - `phi_rules.py` (regex/rules), `phi_ner.py` (obi/deid_roberta_i2b2), `medical.py` (scispaCy BC5CDR + en_core_sci_md untyped entities + measurements)
  - `phi_context.py` — quasi-identifiers (holidays, family, pets, brands/teams, workplaces, activities + injury cue)
  - `smalltalk.py` — small-talk unit classifier trained on train pseudo GT → `.cache/smalltalk.joblib`
  - `cache.py` — disk cache of model outputs (scispaCy/de-id spans, SapBERT embeddings) in `.cache/*.sqlite`;
    keyed by model + detector source hash, so studies reuse them (`--no-cache` to bypass)
  - `relevance.py` — SapBERT embedder, ICD-10-CM index, code-conditioned relevance scorer
  - `spans.py` (Span + categories/actions), `config.py` (thresholds), `models.py` (cached loaders), `data.py` (I/O)
  - `evaluate.py` — local proxy metrics (no labels)
  - `evaluate_gt.py` — scores output against pseudo ground truth (a_removed, b_removed, privacy, c_kept)
  - `sweep.py` — ablation grids (`--grid signals|thresholds|a_ideas`), writes `<out>/results.json`
  - `resources/lexicons.py` (names, eponyms, occupations, generic-term stoplist), `resources/code_expansions.json` (ICD category → terms)
- `pseudo_gt/` — LLM-annotated a/b/c spans: `guidelines.md`, `validate.py`, `train/<encounter_id>.json` (all 140),
  `eval/<encounter_id>.json` (all 67). Tune on train; use eval only for final scoring.
- `docs/` — strategy, architecture, `experiments.md` (log of studies/results — append new ones), `leaderboard.md` (all runs ranked; current baseline); `outputs/` — generated (gitignored); `.cache/` — ICD index (gitignored)

## Commands
- `source .venv/bin/activate` (Python 3.12)
- `python -m anonymizer.cli --input eval --output outputs/eval_anonymized.jsonl [--no-codes] [--limit N]`
- `python -m anonymizer.evaluate --original eval --anonymized outputs/eval_anonymized.jsonl`
- `python -m anonymizer.evaluate_gt --anonymized outputs/eval_anonymized.jsonl [--misses 20]`
- `python -m anonymizer.evaluate_gt --original train --gt train --anonymized outputs/train_anonymized.jsonl` (dev loop)
- `python -m anonymizer.smalltalk --train` — refit small-talk classifier after pseudo GT changes
- `python -m anonymizer.sweep --split train --out outputs/sweep_train` (eval → `outputs/sweep`) — ablation: relevance signals × keep_symptoms
- `python -m anonymizer.sweep --grid a_ideas --split train --out outputs/a_ideas_train` — study G personal-data extensions
- CLI knobs: `--signals 1,2,3,4,5` (1 SapBERT sim, 2 lexical, 3 same ICD category, 4 same block/chapter, 5 learned evidence), `--keep-symptoms` (default off), `--t-keep` (default 0.5), `--a-ideas 1,2,3,4` (study G personal-data extensions in `phi_context.py` / `pipeline.py`, default off). Defaults = baseline in `docs/leaderboard.md`
- `python pseudo_gt/validate.py pseudo_gt/{train,eval}/*.json`
- `python -m anonymizer.data --split eval --show 0`

## Conventions
- Only `dialogue` / `note` change; `icd10_codes` pass through (evaluate asserts this).
- Categories: a = personal, b = clinical not code-relevant, c = code evidence. Every decision gets a `reason`.
- Overlap priority: direct identifier > code evidence > personal > other clinical > measurement.
- Age bands as "50s" (never "50-59": would contain the exact age).
- Experiments: every new run adds one record per split (train + eval) to `docs/leaderboard_runs.json` (study, config,
  a_removed, b_removed, privacy, c_kept, source) and a row with all columns to `docs/leaderboard.md` (ranked by train
  score = mean of privacy and c_kept); findings go to `docs/experiments.md`. Tune on train, eval only for reporting.
