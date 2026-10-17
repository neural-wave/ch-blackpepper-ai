# Leaderboard

Score = mean of privacy (a + b removed) and c_kept, both from `anonymizer/evaluate_gt.py` against the pseudo GT.
**Ranked by train** (140 encounters, where we tune); eval (67) shown for reporting only. Signal-5 runs on train are
out-of-fold. Details of every study: [experiments.md](experiments.md).

## Baseline (from 2026-10-17)
**Entity model on, signals 1–5, t_keep 0.5, t_remove 0.5, nearest-ICD gate on all four entity kinds,
keep_symptoms off, personal rules 1–7 on.**

| split | a_removed | b_removed | privacy | c_kept | score |
|---|---|---|---|---|---|
| train | 0.851 | 0.574 | 0.707 | 0.822 | 0.764 |
| eval | 0.822 | 0.552 | 0.658 | 0.833 | 0.746 |

Config: `entity_model="en_core_sci_md"`, `signals=(1,2,3,4,5)`, `t_keep=0.5`, `t_remove=0.5`,
`entity_sim_gate=("clinical detail","test","procedure","exam finding")`, `entity_gate_on="nearest_sim"`,
`entity_min_sim=0.65`, `keep_symptoms=False`, `smalltalk=True` (threshold 0.6).

Notes:
- Rank 1 and 2 tie on train (0.7645 vs 0.7640); rank 2 (entity, t_keep 0.72) scores higher on eval (0.756) but keeps
  only 70 % of code evidence. The baseline is the more balanced of the two (c_kept 0.82–0.83).
- Ranks 1–5 are within 0.006 on train — treat them as one cluster; differences below ~0.01 are noise at this size.

## All runs
Studies: A = signals × keep_symptoms, B = personal rules 1–7, C = thresholds (old scorer), D = signal 5 (no entity
model), E = entity model × t_keep, F = entity check variants (all at t_keep 0.5). a = personal data removed,
b = unrelated clinical removed, priv = a + b pooled (occurrence-weighted), c = code evidence kept.
Raw numbers: [leaderboard_runs.json](leaderboard_runs.json) (one record per run and split, with its source).

| # | study | configuration | train a | train b | train priv | train c | **train score** | eval a | eval b | eval priv | eval c | eval score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | F | **entity, s1–5, t0.5, nearest-ICD check on all 4 kinds** | 0.851 | 0.574 | 0.707 | 0.822 | **0.764** | 0.822 | 0.552 | 0.658 | 0.833 | 0.746 |
| 2 | E | entity, s1–5, t_keep 0.72 | 0.880 | 0.777 | 0.827 | 0.701 | **0.764** | 0.864 | 0.768 | 0.805 | 0.705 | 0.755 |
| 3 | E | entity, s1–5, t_keep 0.60 | 0.873 | 0.684 | 0.775 | 0.746 | **0.761** | 0.842 | 0.676 | 0.741 | 0.762 | 0.752 |
| 4 | D | no entity, s1–5, t_keep 0.72 | 0.815 | 0.508 | 0.656 | 0.862 | **0.759** | 0.807 | 0.498 | 0.620 | 0.856 | 0.738 |
| 5 | E | entity, s1–5, t_keep 0.50 | 0.853 | 0.600 | 0.722 | 0.794 | **0.758** | 0.823 | 0.588 | 0.681 | 0.804 | 0.742 |
| 6 | F | entity, s1–5, t0.5, check on tests/procedures/exam findings | 0.922 | 0.660 | 0.786 | 0.729 | **0.758** | 0.905 | 0.617 | 0.730 | 0.738 | 0.734 |
| 7 | D | no entity, s1–5, t_keep 0.85 | 0.816 | 0.534 | 0.670 | 0.843 | **0.756** | 0.809 | 0.521 | 0.634 | 0.839 | 0.737 |
| 8 | E | entity, s1–5, t_keep 0.85 | 0.881 | 0.810 | 0.844 | 0.665 | **0.754** | 0.865 | 0.795 | 0.823 | 0.664 | 0.743 |
| 9 | D | no entity, s1–5, t_keep 0.60 | 0.814 | 0.447 | 0.624 | 0.884 | **0.754** | 0.802 | 0.438 | 0.581 | 0.882 | 0.732 |
| 10 | F | entity, s1–5, t0.5, no check (like BC5CDR) | 0.924 | 0.684 | 0.799 | 0.703 | **0.751** | 0.906 | 0.652 | 0.752 | 0.709 | 0.731 |
| 11 | D | no entity, s1–5, t_keep 0.50 | 0.806 | 0.380 | 0.585 | 0.904 | **0.744** | 0.799 | 0.367 | 0.537 | 0.904 | 0.720 |
| 12 | F | entity, s1–5, t0.5, relevance-only check on all 4 kinds | 0.788 | 0.378 | 0.575 | 0.908 | **0.742** | 0.761 | 0.363 | 0.520 | 0.913 | 0.717 |
| 13 | B | s1–4, symptoms off + rules 1–7 (previous best) | 0.816 | 0.542 | 0.674 | 0.804 | **0.739** | 0.809 | 0.535 | 0.643 | 0.790 | 0.716 |
| 14 | C | previous best, t_keep 0.60 | 0.814 | 0.468 | 0.635 | 0.842 | **0.738** | 0.804 | 0.455 | 0.592 | 0.830 | 0.711 |
| 15 | C | previous best, t_keep 0.55 | 0.814 | 0.448 | 0.624 | 0.851 | **0.737** | 0.804 | 0.439 | 0.583 | 0.848 | 0.715 |
| 16 | C | previous best, t_keep 0.90 | 0.816 | 0.546 | 0.676 | 0.796 | **0.736** | 0.809 | 0.536 | 0.644 | 0.783 | 0.713 |
| 17 | E | entity, s1–5, t_keep 0.45 | 0.790 | 0.492 | 0.635 | 0.833 | **0.734** | 0.761 | 0.510 | 0.609 | 0.841 | 0.725 |
| 18 | C | previous best, t_keep 0.93 | 0.817 | 0.558 | 0.683 | 0.785 | **0.734** | 0.811 | 0.567 | 0.663 | 0.773 | 0.718 |
| 19 | B | s1–4, symptoms off + rules 1–6 | 0.760 | 0.533 | 0.642 | 0.813 | **0.728** | 0.733 | 0.528 | 0.609 | 0.805 | 0.707 |
| 20 | E | entity, s1–4, t_keep 0.72 | 0.882 | 0.835 | 0.858 | 0.591 | **0.724** | 0.865 | 0.816 | 0.835 | 0.583 | 0.709 |
| 21 | A | s1–4, symptoms off | 0.593 | 0.533 | 0.561 | 0.816 | **0.689** | 0.636 | 0.527 | 0.570 | 0.809 | 0.689 |
| 22 | B | s1–4, symptoms kept + rules 1–7 | 0.816 | 0.241 | 0.518 | 0.857 | **0.687** | 0.806 | 0.321 | 0.512 | 0.839 | 0.675 |
| 23 | A | s1–3, symptoms off | 0.594 | 0.542 | 0.567 | 0.805 | **0.686** | 0.636 | 0.558 | 0.588 | 0.799 | 0.694 |
| 24 | A | s1–2, symptoms off | 0.594 | 0.542 | 0.567 | 0.799 | **0.683** | 0.636 | 0.558 | 0.588 | 0.793 | 0.691 |
| 25 | B | s1–4, symptoms kept + rules 1–6 | 0.760 | 0.232 | 0.486 | 0.865 | **0.676** | 0.733 | 0.315 | 0.479 | 0.854 | 0.666 |
| 26 | A | s1, symptoms off | 0.594 | 0.546 | 0.569 | 0.780 | **0.674** | 0.636 | 0.559 | 0.589 | 0.776 | 0.682 |
| 27 | A | s1–4, symptoms kept (original baseline) | 0.593 | 0.232 | 0.406 | 0.867 | **0.636** | 0.636 | 0.313 | 0.440 | 0.858 | 0.649 |
| 28 | A | s1–3, symptoms kept | 0.594 | 0.235 | 0.407 | 0.862 | **0.635** | 0.636 | 0.324 | 0.447 | 0.851 | 0.649 |
| 29 | A | s1–2, symptoms kept | 0.594 | 0.235 | 0.407 | 0.856 | **0.632** | 0.636 | 0.324 | 0.447 | 0.847 | 0.647 |
| 30 | A | s1, symptoms kept | 0.594 | 0.236 | 0.408 | 0.838 | **0.623** | 0.636 | 0.325 | 0.448 | 0.831 | 0.639 |
