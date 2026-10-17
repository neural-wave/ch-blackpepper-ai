# Experiments log

Protocol: tune on **train** (140 encounters), report on **eval** (67) only for final numbers. All scores are
occurrence-weighted against the pseudo ground truth (`anonymizer/evaluate_gt.py`). Privacy = removed share of a+b;
c_kept = kept share of c.

## 2026-10-17 — Pseudo GT for train
- 140 train encounters annotated by 7 parallel Claude agents (20 each), one manual decision per mention,
  per [pseudo_gt/guidelines.md](../pseudo_gt/guidelines.md). All files pass `pseudo_gt/validate.py`.
- Distinct strings: train 1536 a / 1849 b / 7251 c; eval 729 a / 1192 b / 3411 c.
- Per encounter: train 11.0 a / 13.2 b / 51.8 c; eval 10.9 a / 17.8 b / 50.9 c — consistent labelling density.
- Judgment calls that recur (keep consistent in future annotation):
  - cause of a coded injury (sport, fall, ladder, car crash) → c, not a;
  - holidays / "the pandemic" tied to an event → a DATE;
  - celebrities, brands, sports teams, clinician small talk → a;
  - uncoded conditions (murmur, anxiety, diabetes) → b, even when central to the visit.
- Caveat: same annotator family as the pipeline author — spot-check before trusting fine differences.

## 2026-10-17 — Ablation: relevance signals × keep_symptoms
Command: `python -m anonymizer.sweep --split {train,eval} --out outputs/sweep[_train]`.
Signals: 1 SapBERT similarity, 2 lexical match, 3 same ICD category, 4 same ICD block/chapter.
Other settings at defaults (`t_keep` 0.72, `t_remove` 0.50, de-id NER on, codes given).

### Train (140)
| run | a_removed | b_removed | privacy | c_kept | b:COND | c:COND | c:MED | c:FINDING |
|---|---|---|---|---|---|---|---|---|
| 1, sym on | 0.594 | 0.236 | 0.408 | 0.838 | 0.300 | 0.859 | 0.636 | 0.779 |
| 1, sym off | 0.594 | 0.546 | 0.569 | 0.780 | 0.721 | 0.767 | 0.633 | 0.673 |
| 1+2, sym on | 0.594 | 0.235 | 0.408 | 0.856 | 0.298 | 0.885 | 0.660 | 0.791 |
| 1+2, sym off | 0.594 | 0.542 | 0.567 | 0.799 | 0.714 | 0.794 | 0.658 | 0.691 |
| 1+2+3, sym on | 0.594 | 0.235 | 0.408 | 0.862 | 0.298 | 0.892 | 0.660 | 0.806 |
| 1+2+3, sym off | 0.594 | 0.542 | 0.567 | 0.805 | 0.714 | 0.802 | 0.658 | 0.708 |
| 1+2+3+4, sym on | 0.593 | 0.232 | 0.406 | 0.867 | 0.293 | 0.899 | 0.660 | 0.815 |
| 1+2+3+4, sym off | 0.593 | 0.533 | 0.562 | 0.816 | 0.700 | 0.818 | 0.658 | 0.718 |

### Eval (67)
| run | a_removed | b_removed | privacy | c_kept | b:COND | c:COND | c:MED | c:FINDING |
|---|---|---|---|---|---|---|---|---|
| 1, sym on | 0.636 | 0.325 | 0.448 | 0.831 | 0.386 | 0.840 | 0.544 | 0.870 |
| 1, sym off | 0.636 | 0.559 | 0.589 | 0.776 | 0.735 | 0.765 | 0.537 | 0.737 |
| 1+2, sym on | 0.636 | 0.324 | 0.447 | 0.847 | 0.384 | 0.865 | 0.572 | 0.877 |
| 1+2, sym off | 0.636 | 0.558 | 0.588 | 0.793 | 0.733 | 0.790 | 0.565 | 0.749 |
| 1+2+3, sym on | 0.636 | 0.324 | 0.447 | 0.851 | 0.384 | 0.873 | 0.572 | 0.883 |
| 1+2+3, sym off | 0.636 | 0.558 | 0.588 | 0.799 | 0.733 | 0.799 | 0.565 | 0.755 |
| 1+2+3+4, sym on | 0.636 | 0.313 | 0.440 | 0.858 | 0.368 | 0.884 | 0.576 | 0.888 |
| 1+2+3+4, sym off | 0.636 | 0.527 | 0.570 | 0.809 | 0.682 | 0.818 | 0.569 | 0.762 |

### Findings — eval
- `keep_symptoms` off: privacy +0.13 to +0.14 (b:COND 0.37–0.39 → 0.68–0.74), c_kept −0.05 (c:FINDING 0.87–0.89 →
  0.74–0.76; symptom evidence is the main casualty).
- Signal 2 (lexical): c_kept +0.016 / +0.017 (sym on / off), privacy −0.001 → cheapest win.
- Signal 3 (same ICD category): c_kept +0.004 / +0.006, privacy unchanged.
- Signal 4 (block/chapter): c_kept +0.007 / +0.010 but privacy −0.007 / −0.019 — the chapter rule keeps unrelated
  same-organ-system diseases (b:COND −0.016 / −0.051). Costs more privacy on eval than on train.
- c:MEDICATION is flat at 0.54–0.58 across all runs (train 0.63–0.66): no signal links a drug to the diagnosis it
  treats (analgesics for pain/injury codes, diuretics for HF) → needs drug→indication knowledge, not similarity.
- b:MEDICATION 0.71 in every run: unrelated drugs are removed by default (low similarity), signals never rescue them.
- a_removed 0.636 constant (NAME 0.99, AGE 0.99, DATE 0.96; QUASI 0.06, OCCUPATION 0.09, LOCATION 0.63).
- Eval operating point: 1+2+3 / sym off = best balance (0.588 privacy / 0.799 c_kept); adding 4 trades
  −0.018 privacy for +0.010 c_kept. Max utility: 1+2+3+4 / sym on (0.858 c_kept, 0.440 privacy).
- Metric caveat: string-level matching (see `evaluate_gt.py`) — comparable across pipeline versions, not a proof of
  information removal (reversible encodings or near-miss spellings would score perfectly).

### Findings (hold on both splits)
- `keep_symptoms` is the dominant privacy–utility knob: off → privacy +0.16, c_kept −0.05.
- Signals 2–4 add utility monotonically (train c_kept 0.838 → 0.867) at ≤0.01 privacy cost; signal 2 gives the
  largest step. Signal 4 costs most b_removed when symptoms are off (train −0.009, eval −0.031).
- a_removed is flat across all runs (signals only touch clinical spans) — personal-data recall (QUASI ~0.05,
  OCCUPATION 0.09–0.17) is the main open gap.
- Train is lower than eval on a/b removal (rules were developed looking at eval) but equal on c_kept → no
  overfitting on utility; mild optimism on eval privacy.
- Operating points: 1+2+3+4 / sym off = best balance (train 0.562 privacy / 0.816 c_kept);
  1+2+3+4 / sym on = max utility (0.867 c_kept).

## 2026-10-17 — Personal data (category a): quasi-identifier rules + small-talk filter
Why a_removed was ~0.6: 79 % of missed a-occurrences were QUASI (family, hobbies, events, brands), 11 % OCCUPATION;
almost none were detected at all. NAME / AGE / DATE were already ≥ 0.96.

Changes (`phi_context.py`, `smalltalk.py`, `phi_rules.py`, `pipeline.py`):
1. Occupation / location / org / workplace detected once → every other mention (`_propagate`); "you're a …" cue.
2. Holidays and named events ("labor day", "the pandemic", "super bowl") → [DATE].
3. Family members → [family member], pets → [pet], trips / weddings / job changes → [life event].
4. Names: greeting with filler ("hey there mark") anchors an ambiguous name; NER-found ambiguous names propagate
   unless used as the ordinary word.
5. Stores, brands, insurers, teams (ambiguous team names only after "the" + sports words) → [ORGANIZATION].
6. Activities (lexicon + hobby frames for walking/running/…) → [activity], **kept** when any mention has an
   injury/exposure cue and an S/T/M/W/X/V/Y code is present (pets: allergy/bite codes). Encounter-level cue
   (vs per-mention) cut c losses 76 → 29 occurrences on train for −12 a hits.
7. Small-talk classifier: TF-IDF + logistic regression on dialogue units, labels from train pseudo GT, out-of-fold
   on train; units ≥ 0.6 without any code-evidence span are removed whole. Neighbour-unit context did not help.

Pipeline at defaults otherwise (signals 1+2+3+4, keep_symptoms on). `ent` = untyped entity model
(`entity_model = en_core_sci_md`, added separately the same day) on / `noent` = off.

| split | run | a_removed | b_removed | privacy | c_kept | a:QUASI | a:OCC | a:LOC | c:QUASI |
|---|---|---|---|---|---|---|---|---|---|
| train | before (s1234 sym on) | 0.593 | 0.232 | 0.406 | 0.867 | 0.05 | 0.17 | 0.61 | 0.92 |
| train | noent, rules 1–6 | 0.760 | 0.232 | 0.486 | 0.865 | 0.46 | 0.35 | 0.75 | 0.86 |
| train | noent, rules + small talk | 0.816 | 0.241 | 0.518 | 0.857 | 0.58 | 0.55 | 0.82 | 0.79 |
| train | ent, before | 0.772 | 0.481 | 0.621 | 0.655 | 0.49 | 0.53 | 0.75 | 0.42 |
| train | ent, rules + small talk | 0.880 | 0.487 | 0.676 | 0.657 | 0.72 | 0.74 | 0.88 | 0.53 |
| eval | before (s1234 sym on) | 0.636 | 0.313 | 0.440 | 0.858 | 0.06 | 0.09 | 0.63 | – |
| eval | noent, rules 1–6 | 0.733 | 0.315 | 0.479 | 0.854 | 0.33 | 0.21 | 0.74 | 0.92 |
| eval | noent, rules + small talk | 0.806 | 0.321 | 0.512 | 0.839 | 0.51 | 0.37 | 0.89 | 0.78 |
| eval | ent, rules + small talk | 0.859 | 0.553 | 0.673 | 0.648 | 0.67 | 0.44 | 0.89 | 0.66 |

Findings:
- Rules 1–6 are nearly free in utility (train c_kept −0.002, eval −0.004) and add +0.10–0.17 a_removed.
- Small talk adds +0.06–0.07 a_removed for −0.008 (train) / −0.015 (eval) c_kept; the loss is mostly c:QUASI
  (injury causes in units with no detected clinical span). Threshold 0.7: train a 0.784 / c 0.861 (pre-lexicon run).
- Eval QUASI (0.33 → 0.51) lags train (0.46 → 0.58): lexicons were built from train misses; small talk transfers.
- Remaining a gap is long-tail QUASI (food, preferences, possessions, anecdotes) and occupation phrased without a
  job noun ("works outside", "rocking kids") → candidate for the small-LLM stage.
- Untyped entity model (`ent`): +0.25 b_removed but −0.21 c_kept on both splits — needs its own tuning before it
  is a default.

### Same study with keep_symptoms off (signals 1+2+3+4, entity model off)
| split | run | a_removed | b_removed | privacy | c_kept | a:QUASI | a:OCC | a:LOC | c:QUASI |
|---|---|---|---|---|---|---|---|---|---|
| train | baseline | 0.593 | 0.533 | 0.562 | 0.816 | 0.05 | 0.17 | 0.61 | – |
| train | + rules 1–6 | 0.760 | 0.533 | 0.642 | 0.813 | 0.46 | 0.35 | 0.75 | 0.86 |
| train | + rules 1–6 + small talk | 0.816 | 0.542 | 0.674 | 0.804 | 0.58 | 0.55 | 0.82 | 0.78 |
| eval | baseline | 0.636 | 0.527 | 0.570 | 0.809 | 0.06 | 0.09 | 0.63 | – |
| eval | + rules 1–6 | 0.733 | 0.529 | 0.609 | 0.805 | 0.33 | 0.21 | 0.74 | 0.92 |
| eval | + rules 1–6 + small talk | 0.809 | 0.535 | 0.643 | 0.790 | 0.52 | 0.37 | 0.89 | 0.78 |

- The personal-data gains are identical to keep_symptoms on (the switch only affects clinical spans); c costs are the
  same size (rules −0.003/−0.004, small talk −0.009/−0.015). Effects are additive with the symptom switch.
- Best balance so far: rules 1–7 + keep_symptoms off → eval privacy 0.643 / c_kept 0.790.
- Model outputs are now cached (`anonymizer/cache.py`); a warm run is ~3× faster with identical output.

## Current best configuration (as of 2026-10-17, entity-model study excluded)
- Personal rules 1–7 on (default), signals 1+2+3+4, **keep_symptoms off**.
- Eval: a_removed 0.809, b_removed 0.535, privacy 0.643, c_kept 0.790 (train: 0.816 / 0.542 / 0.674 / 0.804).
- Path from the original baseline (eval, signals 1–4, symptoms kept): privacy 0.440 / c_kept 0.858
  → symptoms off 0.570 / 0.809 → + rules 1–7 0.643 / 0.790.
- keep_symptoms off is a privacy-for-utility trade (≈ +0.13 privacy, −0.05 c_kept), the largest single lever.
- Signals 1–3 vs 1–4 (symptoms off) is close: signal 4 gives +0.010 c_kept for −0.005 (train) / −0.018 (eval)
  privacy. Rules 1–7 measured only with 1–4.
- Defaults now match this setup (`keep_symptoms = False`); `--keep-symptoms` restores the old behaviour.

## 2026-10-17 — Category a error analysis (current best config)
Defaults (rules 1–7, signals 1+2+3+4, symptoms off). Misses dumped to `outputs/a_study/{train,eval}_a_misses.json`.
Remaining missed a-occurrences: train 380 / 2067, eval 182 / 955. 92 % are never detected (no span at all);
~8 % are activities the injury-cue gate keeps as code evidence. NAME / AGE / DATE ≥ 0.98 — the gap is QUASI + OCC.

| theme (keyword buckets, approx.) | train | eval |
|---|---|---|
| exercise / hobby as verb phrase ("power walking every day", "ride the bike") | 21 % | 22 % |
| work without a job noun ("works from home", "sedentary job", "coworker", "RN") | 13 % | 19 % |
| family / household / social in bare form ("Maternal grandmother", "older sister", "my friend") | 12 % | 15 % |
| travel, home projects, life events ("renovating the basement", "expecting a newborn") | 15 % | 10 % |
| food / drink / products / brands ("pickle juice", "Ripple Pea Protein milk") | 8 % | 9 % |
| activity kept by injury-cue gate | 8 % | 9 % |
| sports teams / fandom, often ASR-garbled ("brooklyn net", "long horns") | 5 % | 4 % |
| other (handedness, Fitzpatrick type, parity, places like "the zoo") | 18 % | 12 % |

- ~36 % of train misses are in the `note` (third-person phrasing: "Lives with wife", "enjoys running"); rules are
  dialogue-shaped (possessive kin, first-person hobby frames).
- Bare kin words are safe to generalize: train GT kin spans are 129 a vs 6 c.
- Activity keep gate: 151 kept activities are GT c vs 56 GT a — gate precision ~73 %; not fixable by code group alone.
- Some GT a spans are debatable (e.g. "wheelchair", "G5P5", "epic", "my nurse", doctor's own hobby).
