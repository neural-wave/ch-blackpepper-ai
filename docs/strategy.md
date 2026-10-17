# Strategy — Patient Data Anonymization (Bosch @ Neural Wave)

## Goal
Anonymize `dialogue` + `note` so that private information is unidentifiable, while the
text still supports the encounter's ICD-10 codes. `icd10_codes` pass through unchanged.

## Data taxonomy
| Cat | What | Treatment | Scored by |
|---|---|---|---|
| **a** — sensitive personal | names (patient, family, staff), dates, DOB, age, locations, IDs, contacts, employer, quasi-identifiers (job, hobbies, events) | remove / pseudonymize / generalize aggressively | Metric A (privacy) |
| **b** — sensitive clinical, not code-relevant | unrelated history, meds, procedures, labs not supporting the code | generalize / perturb / remove depending on relevance | Metric A (privacy) |
| **c** — code-relevant clinical | evidence and cause of the coded diagnoses | keep verbatim (light numeric perturbation allowed) | Metric B (ICD F1 + concept recall) |

- Metric A = proportion of removed privacy-related info → covers **a + b**.
- Metric B = ICD-10 F1 + medical concept recall of **code-relevant** info → covers **c**.
- Splitting clinical data into b vs c removes the conflict between the metrics. The **b/c
  boundary is the core of the system**: over-keeping leaks (A↓), under-keeping loses evidence (B↓).

## Key decisions
1. **Predict what to keep, not what to remove.** Build a keep mask of code evidence first;
   everything medical outside it is a privacy target.
2. **Code-conditioned evidence retrieval, not a trained classifier.** Codes are available at
   inference. Train has 140 encounters / 146 unique codes; 79 of 112 eval codes are unseen in
   train → a classifier cannot generalize. Instead: expand each code into concepts (ICD-10-CM
   hierarchy, inclusion terms, curated synonyms) and score text spans against them.
3. **Graded relevance, not binary.** Each medical span gets a relevance score per code:
   - high → keep (c)
   - medium → generalize / perturb (b-soft)
   - low → remove (b-hard)
   The two thresholds form the **privacy–utility knob**; sweeping them yields a Pareto curve.
4. **Context decides category** (e.g. "Parkinson" disease vs surname, age as identifier vs
   code-relevant, sex-specific codes). Explicit conflict rules:
   - c always kept if it is the cause/evidence of a diagnosis (even if rare)
   - privacy for rare c is achieved by anonymizing the identifying *context around it*
   - direct identifiers (names, IDs, contacts) always removed
5. **High-recall personal-data detection**: union of rules + clinical de-id NER (+ small LLM
   later). Names found once are propagated to every occurrence in dialogue and note.
6. **Consistent pseudonyms** per encounter (`[PATIENT]`, `[DOCTOR_1]`, …), **date shifting /
   generalization**, **age bands**, **quasi-identifier generalization** (k-anonymity style).
7. **Verify loop**: after transformation, check code evidence is still present (roll back
   edits that removed it); later an LLM attacker tries to re-identify and the pipeline
   re-anonymizes what it recovers.
8. **Every decision is logged** (span, category, relevance, action, reason) → explainability
   bonus comes for free.

## Constraints
- Inference on-prem, ≤ 4 GB VRAM, no cloud models. Development may use more compute.
- Models run sequentially; target budget: SapBERT (~0.5 GB), de-id RoBERTa (~1.4 GB fp32 /
  0.7 GB fp16), scispaCy (CPU), optional 1.5–3B LLM at 4-bit (~2–2.5 GB).

## Bonus tasks — how the design covers them
- **Adversarial attacks**: LLM attacker in the verify loop; fixed red-team prompt set tracked
  as a regression metric.
- **Explainability**: decision log with reason + relevance score per span.
- **Efficiency**: CPU-first components, profiling of runtime / peak RAM / peak VRAM.
- **No ICD codes at inference**: predict candidate codes (embedding retrieval over ICD-10-CM
  descriptions / small LLM), then run the same evidence pipeline with a lower threshold.
- **Robustness**: synthetic hospital / discharge notes with different styles.

## Open questions for organizers
- Exact entity types counted by the privacy metric.
- Concept-recall matching: exact string or concept ID (CUI)?
- Does the ICD predictor read the note, the dialogue, or both? Full code or category match?

## References
- Pilán et al., *The Text Anonymization Benchmark (TAB)*, Computational Linguistics 2022.
- Norgeot et al., *Philter*, npj Digital Medicine 2020.
- Chambon et al., *Automated deidentification of radiology reports combining transformer and
  "hide in plain sight" rule-based methods*, JAMIA 2023.
- Carrell et al., *Hiding in plain sight*, JAMIA 2013.
- Sweeney, *k-anonymity*, 2002.
- Staab et al., *Beyond Memorization: Violating Privacy via Inference with LLMs*, ICLR 2024.
- Staab et al., *Large Language Models are Advanced Anonymizers*, 2024.
- Liu et al., *SapBERT: Self-Alignment Pretraining for Biomedical Entity Representations*, NAACL 2021.
- Neumann et al., *ScispaCy*, BioNLP 2019.
- Mullenbach et al., *CAML*, NAACL 2018; Cheng et al., *MDACE*, ACL 2023 (code evidence spans).
- DeYoung et al., *ERASER* (rationale sufficiency/comprehensiveness), ACL 2020.
- Jain & Wallace, *Attention is not Explanation*, NAACL 2019.
