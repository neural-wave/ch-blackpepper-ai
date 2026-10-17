# Architecture — v1

Implements [strategy.md](strategy.md). Package: `anonymizer/`. Entry points: `anonymizer.cli`, `anonymizer.evaluate`.

```
 input: dialogue + note + icd10_codes          (no-code mode: codes predicted by ICD retrieval)
   ▼
 [1] KEEP MASK              medical.py, relevance.py
     scispaCy BC5CDR → DISEASE / CHEMICAL spans
     relevance per span vs code concept set:
       lexical hit (description, ICD ancestors, inclusion terms, curated expansions) → 1.0
       SapBERT cosine to concept set
       nearest ICD-10-CM code (98k index): same category → 0.95, same block → 0.9, same chapter → 0.6
     ≥ t_keep → c keep | ICD chapter R symptom → keep | ≥ t_remove → generalize | else remove
     measurements (%, mmHg, bpm, kg, …, BP) → perturb ±3 %
   ▼
 [2] PERSONAL DATA          phi_rules.py, phi_ner.py
     rules: dates, DOB, years, ages, phone, email, URL, IDs, addresses, occupations (in self-context),
            ethnicity (as person attribute), title / greeting / intro-anchored names
     NER:   obi/deid_roberta_i2b2 (PATIENT, STAFF, AGE, DATE, LOC, HOSP, …), sliding window, word-snapped
   ▼
 [3] NAMES + CONFLICTS      pipeline.py
     name registry per encounter: NER on the cased note seeds roles → tokens propagated to dialogue
     eponym check (Parkinson's disease, Neer's test, Clark level …) → not a name
     relevance guard on LOC/HOSP/ORG/OCCUPATION (Y92 place, Y93 activity codes) → keep if code evidence
     greedy overlap resolution by priority: direct identifier > code evidence > personal > other clinical > measurement
   ▼
 [4] TRANSFORM
     names → consistent pseudonyms [PATIENT], [CLINICIAN_n], [PERSON_n]
     dates → [DATE], years → [YEAR], age → decade ("50s"), occupation → "[sector worker]", ethnicity → [ETHNICITY]
     b-soft → ICD block label (e.g. "[influenza and pneumonia]") or [condition]/[medication]
     b-hard → [other condition] / [other medication]
   ▼
 [5] VERIFY
     sweep: any registered name token left → replaced
     log: codes_without_evidence per encounter
   ▼
 output .jsonl (icd10_codes untouched) + <output>.decisions.jsonl (span, category, action, score, reason)
```

## Models and budget
| Model | Role | Size | Device |
|---|---|---|---|
| `en_ner_bc5cdr_md` (scispaCy) | medical NER | ~120 MB | CPU |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | relevance embeddings + ICD index | 110M params, ~0.45 GB | GPU/MPS/CPU |
| `obi/deid_roberta_i2b2` | clinical de-id NER | 355M params, ~0.7 GB fp16 | GPU/MPS/CPU |

- Peak RAM ~3.1 GB on Apple M5 Pro (MPS), ~1.1 s / encounter after warm-up.
- ICD index (98k codes) built once on first run (~85 s), cached in `.cache/icd_index.npz`.
- Not yet measured on a CUDA box with a 4 GB VRAM cap; expected < 1.5 GB VRAM in fp16.

## Local evaluation (`anonymizer/evaluate.py`)
Proxies; the organizers' pipeline is private.
- `ref_phi_removed` — independent reference detector (en_core_web_sm PERSON/GPE/LOC/FAC/NORP/DATE minus medical
  overlaps + regex). Its PERSON/GPE labels are noisy on clinical text (drug names, anatomy) → a lower bound.
- `unrelated_removed`, `code_concept_recall` — use the same relevance scorer as the pipeline (circular; measures
  collateral damage, not scorer quality).
- `desc_term_recall` — ICD description words surviving; scorer-independent.
- `retrieval_f1_*` — weak retrieval coder on original vs anonymized; only the delta is meaningful.

## Pseudo ground truth (`pseudo_gt/`, `anonymizer/evaluate_gt.py`)
All 67 eval and 140 train encounters annotated by Claude agents with a/b/c spans per
[pseudo_gt/guidelines.md](../pseudo_gt/guidelines.md) (eval 729 a / 1192 b / 3411 c, train 1536 a / 1849 b / 7251 c distinct
strings). Tune on train, score on eval. Study results: [experiments.md](experiments.md). Not independent of the pipeline author — spot-check before relying on it.

v1 on eval (occurrence-weighted):
| | a_removed | b_removed | privacy (a+b) | c_kept |
|---|---|---|---|---|
| with codes | 0.64 | 0.31 | 0.44 | 0.86 |
| no codes | 0.63 | 0.21 | 0.37 | 0.87 |

- Strong: NAME 0.99, AGE 0.99, DATE 0.96.
- Weak: QUASI 0.06, OCCUPATION 0.09, LOCATION 0.63 (hobbies, teams, workplaces, pharmacies).
- b detection is the bottleneck: only DISEASE/CHEMICAL are detected → PROCEDURE 0.06, FINDING 0.14, LAB 0.26;
  symptoms are kept by `keep_symptoms`.
- c loss mostly MEDICATION 0.58 (analgesics etc. scored "unrelated" for pain/injury codes).

## Known gaps / next steps
1. Small LLM stage (≤3B, 4-bit) for quasi-identifiers rules miss: hobbies, events, teams, family details.
2. LLM attacker loop (bonus: adversarial) + silver PHI labels on train for a real privacy metric.
3. Better ICD-F1 proxy (LLM coder on original vs anonymized).
4. Relevance for signs/findings is weak when the curated expansion table has no entry for the code
   (e.g. "jugular venous distension" for heart failure) — LLM or co-occurrence knowledge needed.
5. Medication doses can remain after the drug name is removed ("[other medication] 10.").
6. Threshold sweep (t_keep, t_remove) → privacy–utility Pareto curve.
