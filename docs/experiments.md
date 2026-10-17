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
