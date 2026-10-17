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

## 2026-10-17 — Ablation: t_keep × t_remove
Settings: current defaults (signals 1+2+3+4, keep_symptoms off, personal rules 1–7 on, entity model off).
Command: `python -m anonymizer.sweep --grid thresholds --split train --out outputs/thresholds_train` (full grid,
results.json there); eval only for the candidates (t_remove 0.5). Score distribution: continuous SapBERT values
< ~0.52, then steps 0.6 (same chapter), 0.9 (same block), 0.95 (same category), 1.0 (lexical) — t_keep only matters
between steps.

### Train (140), t_remove = 0.5
| t_keep | a_removed | b_removed | privacy | c_kept | b:COND | b:FINDING | c:COND | c:FINDING | c:MED | c:PROC |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.814 | 0.448 | 0.624 | 0.851 | 0.59 | 0.24 | 0.87 | 0.82 | 0.67 | 0.96 |
| 0.60 | 0.814 | 0.468 | 0.635 | 0.842 | 0.61 | 0.25 | 0.86 | 0.80 | 0.66 | 0.95 |
| **0.72** | 0.816 | 0.542 | 0.674 | 0.804 | 0.71 | 0.31 | 0.81 | 0.72 | 0.65 | 0.94 |
| 0.90 | 0.816 | 0.547 | 0.676 | 0.796 | 0.71 | 0.33 | 0.80 | 0.69 | 0.65 | 0.94 |
| 0.93 | 0.817 | 0.558 | 0.683 | 0.785 | 0.73 | 0.33 | 0.78 | 0.68 | 0.65 | 0.93 |
| 0.97 | 0.817 | 0.558 | 0.683 | 0.770 | 0.73 | 0.33 | 0.76 | 0.65 | 0.65 | 0.93 |
| 1.00 | 0.817 | 0.558 | 0.683 | 0.768 | 0.73 | 0.33 | 0.76 | 0.65 | 0.65 | 0.93 |

t_remove 0.3 → 0.6 at any t_keep: b_removed +0.004–0.005, c_kept −0.001, a unchanged.

### Eval (67), t_remove = 0.5
| t_keep | a_removed | b_removed | privacy | c_kept | b:COND | b:FINDING | c:COND | c:FINDING | c:MED | c:PROC |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.804 | 0.439 | 0.583 | 0.848 | 0.55 | 0.24 | 0.88 | 0.87 | 0.61 | 0.95 |
| 0.60 | 0.804 | 0.455 | 0.592 | 0.830 | 0.58 | 0.24 | 0.86 | 0.84 | 0.58 | 0.95 |
| **0.72** | 0.809 | 0.535 | 0.643 | 0.790 | 0.69 | 0.30 | 0.80 | 0.76 | 0.57 | 0.94 |
| 0.90 | 0.809 | 0.536 | 0.644 | 0.783 | 0.69 | 0.30 | 0.80 | 0.74 | 0.57 | 0.94 |
| 0.93 | 0.812 | 0.567 | 0.663 | 0.773 | 0.74 | 0.31 | 0.78 | 0.73 | 0.56 | 0.94 |

### Findings
- **t_remove is nearly inert** for the metric: generalizing and removing both make the original string disappear.
  It only decides how informative the replacement is (block label vs "[other condition]") — choose it for
  readability / downstream coding, not for the score. Keep 0.5.
- **t_keep is the knob**, and it acts at the score steps:
  - 0.60 → 0.72 (stop treating "same ICD chapter" as evidence): privacy +0.04/+0.05, c_kept −0.04 (train/eval).
  - 0.72 → 0.90 (stop trusting "same block"): ≈ 0 privacy, −0.007 c_kept → dominated; 0.72 beats 0.9.
  - 0.90 → 0.93 (stop trusting "same category"): train +0.006 / −0.011, eval +0.019 / −0.010 → split-dependent.
  - ≥ 0.97 (lexical only): no privacy gain, −0.015 c_kept → dominated.
- **a is affected slightly** (train 0.814 → 0.817, eval 0.804 → 0.812 from t_keep 0.55 → 0.93): the relevance
  guard keeps personal spans (places, activities, pets, occupations) that score ≥ t_keep as code evidence, and the
  small-talk filter skips units holding a kept clinical span; a lower t_keep keeps more of both.
- Pareto points: t_keep 0.55, 0.60, 0.72, 0.93. Default 0.72 stays the balanced choice.
- Lowering t_keep is a cheaper way to buy utility than keeping symptoms: eval t_keep 0.55 + symptoms off
  (0.583 privacy / 0.848 c_kept) beats t_keep 0.72 + symptoms on (0.512 / 0.839) on both axes.

## 2026-10-17 — Learned evidence (signal 5), entity model, t_keep re-tune
Diagnosis first (train, defaults): of c occurrences 49 % are never detected (stay → count as kept), 32 % detected
and kept, **18 % lost because the relevance score is < t_keep** (1509 in the SapBERT range < 0.52, 604 at the
chapter step), 1.3 % lost to personal rules. Of b, 43 % are never detected, only 2.7 % wrongly kept. → c is limited by
scoring (clinical links that are not textual similarity: ibuprofen ↔ knee pain, JVD ↔ heart failure), b by detection.

**Signal 5** (`anonymizer/evidence.py`): per term, in how many train encounters the pseudo GT marks it as evidence for
a code of the same ICD category (→ score 0.92) or chapter (≥ 2 encounters → 0.80), against how often it is b with
such a code present; unseen terms inherit from the nearest lexicon term (SapBERT ≥ 0.9). Out-of-fold on train
(same 5 folds as small talk); full train lexicon on eval. Entity-model spans are now cached too.

All runs: keep_symptoms off, personal rules 1–7 on, t_remove = min(0.5, t_keep).

### Signal 5 at t_keep 0.72, no entity model
| split | signals | a_removed | b_removed | privacy | c_kept | c:MED | c:FINDING |
|---|---|---|---|---|---|---|---|
| train | 1–4 | 0.816 | 0.542 | 0.674 | 0.804 | 0.65 | 0.72 |
| train | 1–5 | 0.815 | 0.508 | 0.656 | 0.862 | 0.82 | 0.80 |
| train | 1–5, t_keep 0.85 (category level only) | 0.816 | 0.534 | 0.670 | 0.843 | 0.76 | – |
| eval | 1–4 | 0.809 | 0.535 | 0.643 | 0.790 | 0.57 | 0.76 |
| eval | 1–5 | 0.807 | 0.498 | 0.620 | 0.856 | 0.80 | 0.83 |
| eval | 1–5, t_keep 0.85 (category level only) | 0.809 | 0.521 | 0.635 | 0.839 | 0.77 | – |

### t_keep × entity model (signals 1–5)
| t_keep | train no-ent priv / c | train entity priv / c | eval no-ent priv / c | eval entity priv / c |
|---|---|---|---|---|
| 0.30 | 0.366 / 0.980 | 0.337 / 0.971 | | |
| 0.40 | 0.469 / 0.943 | 0.515 / 0.888 | | |
| 0.45 | 0.534 / 0.922 | 0.635 / 0.833 | | 0.609 / 0.841 |
| 0.50 | 0.585 / 0.904 | 0.722 / 0.794 | 0.537 / 0.904 | **0.681 / 0.804** |
| 0.55 | 0.614 / 0.892 | 0.762 / 0.761 | | |
| 0.60 | 0.624 / 0.884 | 0.775 / 0.746 | 0.581 / 0.882 | 0.741 / 0.762 |
| 0.72 | 0.656 / 0.862 | 0.827 / 0.701 | 0.620 / 0.856 | 0.806 / 0.705 |
| 0.85 | 0.670 / 0.843 | 0.844 / 0.665 | 0.635 / 0.839 | 0.823 / 0.664 |
Entity model at signals 1–4, t_keep 0.72: train 0.858 / 0.591, eval 0.835 / 0.583 (signal 5 recovers +0.11 c).

### Findings
- Signal 5 is the biggest utility gain so far: eval c_kept +0.066 (c:MEDICATION 0.57 → 0.80) for −0.023 privacy —
  better than any t_keep move on the old scorer (t_keep 0.55 gave 0.583 / 0.848). Its category level is nearly free
  (eval +0.049 c for −0.009 privacy); the chapter level is a ~1:1 trade.
- The entity model moves the curve into the high-privacy region (privacy 0.68–0.82) that the no-entity pipeline cannot
  reach (max ≈ 0.67). Below privacy ≈ 0.65 the no-entity pipeline keeps ~0.04 more c at equal privacy.
- **Entity model + signal 5 + t_keep 0.5 beats the previous best on both axes**: eval 0.681 / 0.804 vs 0.643 / 0.790
  (train 0.722 / 0.794 vs 0.674 / 0.804).
- Low t_keep also lowers a_removed (train entity: 0.88 at 0.72 → 0.59 at 0.30; eval entity 0.45: 0.761): more spans
  count as code evidence, which protects places/activities (relevance guard) and blocks small-talk removal.
- Remaining c loss with the entity model is procedures and exam findings (c:PROCEDURE 0.65, c:FINDING 0.50–0.56):
  they rarely resemble a diagnosis description and the evidence lexicon covers few of them.

Operating points (eval privacy / c_kept):
- utility-leaning: no entity, signals 1–5, t_keep 0.72 → 0.620 / 0.856 (or 0.85 → 0.635 / 0.839)
- balanced: entity, signals 1–5, t_keep 0.50 → 0.681 / 0.804
- privacy-leaning: entity, signals 1–5, t_keep 0.60 → 0.741 / 0.762

## 2026-10-17 — Entity gate: which entity kinds need an ICD-similarity check
Signal 5 in one line: a train-learned table "term → was it code evidence for this ICD category / chapter" (score 0.92 /
0.80), out-of-fold on train; it captures clinical links that are not text similarity (drug ↔ indication).

Before: entity spans of kind "clinical detail" are left untouched when score < t_keep and nearest-ICD similarity < 0.65;
marked kinds (test, procedure, exam finding) were always scored and removed below t_keep. Now configurable:
`Config.entity_sim_gate` = kinds that get the check. Base = balanced pipeline (entity model, signals 1–5,
t_keep 0.5, keep_symptoms off, rules 1–7).

| split | gate applies to | a_removed | b_removed | privacy | c_kept | c:FINDING | c:PROCEDURE |
|---|---|---|---|---|---|---|---|
| train | clinical detail (before) | 0.853 | 0.600 | 0.722 | 0.794 | 0.69 | 0.78 |
| train | nothing (change 1) | 0.924 | 0.684 | 0.799 | 0.703 | 0.51 | 0.64 |
| train | all kinds (change 2) | 0.851 | 0.574 | 0.707 | **0.822** | 0.73 | 0.88 |
| train | marked kinds only (1 + 2) | 0.922 | 0.660 | 0.786 | 0.729 | 0.54 | 0.73 |
| eval | clinical detail (before) | 0.823 | 0.588 | 0.681 | 0.804 | 0.74 | 0.73 |
| eval | nothing (change 1) | 0.906 | 0.652 | 0.752 | 0.709 | 0.53 | 0.64 |
| eval | all kinds (change 2) | 0.822 | 0.552 | 0.658 | **0.833** | 0.78 | 0.84 |
| eval | marked kinds only (1 + 2) | 0.905 | 0.617 | 0.730 | 0.738 | 0.57 | 0.74 |

Compared against the t_keep curve of the same pipeline (interpolated at equal privacy):
- Change 2 (gate tests/procedures/exam findings too) moves the curve outward: +0.021 c (train) / +0.017 (eval) at
  equal privacy; c:PROCEDURE 0.78 → 0.88. Procedures/tests are mostly code evidence in this data, and the ones that
  look like no ICD concept are now left in place instead of removed.
- Change 1 (no gate on clinical details) is dominated: −0.02 (train) / −0.04 (eval) c vs simply raising t_keep to the
  same privacy. It removes many everyday words: a_removed jumps 0.85 → 0.92 (small talk words are entities too), but
  c:FINDING falls 0.69 → 0.51.
- Both together is also below the curve.
→ Keep the gate on clinical details and extend it to all kinds: `entity_sim_gate` = all four kinds.

Follow-up: same "all four kinds" gate but on the **relevance score** instead of nearest-ICD similarity
(`entity_gate_on = "relevance"`, threshold 0.65). With t_keep 0.5 this reduces to "relevance < 0.5 → untouched",
so entity spans are either kept or left in place, never removed.

| split | gate | a_removed | b_removed | privacy | c_kept | c:FINDING | c:PROCEDURE |
|---|---|---|---|---|---|---|---|
| train | all four kinds, relevance only | 0.788 | 0.378 | 0.575 | 0.908 | 0.89 | 0.97 |
| eval | all four kinds, relevance only | 0.761 | 0.364 | 0.520 | 0.913 | 0.91 | 0.96 |

- Behaves like the pipeline without the entity model (eval no-entity, t_keep 0.5: privacy 0.537 / c 0.904) but slightly
  worse on privacy (−0.017) and a_removed (0.80 → 0.76): entity spans kept as evidence now protect nearby places /
  activities and block small-talk removal, while no entity span is ever removed.
- Nearest-ICD similarity is what lets the gate separate "medical but unrelated" (remove) from "not diagnosis-like"
  (leave); with relevance alone the entity model adds detection without any removal.

## Baseline set as defaults (2026-10-17)
`config.py` now defaults to the leaderboard baseline (entity model, signals 1–5, t_keep 0.5, nearest-ICD gate on all four
entity kinds, keep_symptoms off, rules 1–7). A plain `python -m anonymizer.cli` run reproduces it exactly: train
a 0.851 / b 0.574 / privacy 0.707 / c 0.822, eval 0.822 / 0.552 / 0.658 / 0.833. Supersedes "Current best
configuration" above. Ranking of all runs: [leaderboard.md](leaderboard.md).
