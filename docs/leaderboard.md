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
- This config is still the default in `config.py`. Study G (personal-data extensions, `--a-ideas 1,2,3,4`) on top of it
  is now rank 1: train 0.774 / eval 0.754, +0.010 / +0.008 score, all from privacy (a +0.037 / +0.038) at
  c_kept −0.001 / +0.001. Not yet promoted to default.
- Among the pre-G runs, the baseline and entity t_keep 0.72 tie on train (0.7645 vs 0.7640); t_keep 0.72 scores
  higher on eval (0.756) but keeps only 70 % of code evidence. Differences below ~0.01 are noise at this size.

## Study G — personal-data extensions (2026-10-17)
On top of the baseline above; switch `--a-ideas` / `Config.a_ideas`, off by default. Ideas: 1 extended rules (bare
kin, third-person hobbies, exercise routines, work phrases, household, life events), 2 SOCIAL HISTORY sentences
without code evidence generalized, 3 activities / life events propagated across dialogue ↔ note, 4 strict injury cue
for keeping activities. Per-type columns = share removed (a) / kept (c).

| ideas | split | a | b | priv | c | score | a:QUASI | a:OCC | a:LOC | b:COND | b:PROC | c:COND | c:FIND | c:MED | c:PROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (baseline) | train | 0.851 | 0.574 | 0.707 | 0.822 | 0.764 | 0.662 | 0.631 | 0.842 | 0.621 | 0.454 | 0.826 | 0.729 | 0.827 | 0.878 |
| none (baseline) | eval | 0.822 | 0.552 | 0.658 | 0.833 | 0.746 | 0.577 | 0.419 | 0.833 | 0.597 | 0.374 | 0.869 | 0.779 | 0.809 | 0.835 |
| 1 | train | 0.883 | 0.575 | 0.723 | 0.821 | 0.772 | 0.729 | 0.811 | 0.842 | 0.623 | 0.454 | 0.825 | 0.729 | 0.827 | 0.878 |
| 1 | eval | 0.849 | 0.552 | 0.669 | 0.833 | 0.751 | 0.644 | 0.558 | 0.833 | 0.597 | 0.374 | 0.869 | 0.779 | 0.809 | 0.835 |
| 2 | train | 0.858 | 0.576 | 0.712 | 0.821 | 0.767 | 0.679 | 0.658 | 0.842 | 0.625 | 0.460 | 0.825 | 0.729 | 0.827 | 0.878 |
| 2 | eval | 0.833 | 0.554 | 0.663 | 0.833 | 0.748 | 0.597 | 0.512 | 0.833 | 0.600 | 0.374 | 0.869 | 0.779 | 0.809 | 0.835 |
| 3 | train | 0.852 | 0.574 | 0.707 | 0.822 | 0.765 | 0.663 | 0.631 | 0.842 | 0.621 | 0.454 | 0.826 | 0.729 | 0.827 | 0.878 |
| 3 | eval | 0.822 | 0.552 | 0.658 | 0.833 | 0.746 | 0.577 | 0.419 | 0.833 | 0.597 | 0.374 | 0.869 | 0.779 | 0.809 | 0.835 |
| 4 | train | 0.850 | 0.574 | 0.707 | 0.821 | 0.764 | 0.659 | 0.631 | 0.842 | 0.621 | 0.454 | 0.826 | 0.729 | 0.827 | 0.878 |
| 4 | eval | 0.825 | 0.552 | 0.660 | 0.834 | 0.747 | 0.587 | 0.419 | 0.833 | 0.597 | 0.374 | 0.869 | 0.779 | 0.809 | 0.835 |
| 1+3 | train | 0.884 | 0.575 | 0.724 | 0.821 | 0.773 | 0.731 | 0.811 | 0.842 | 0.623 | 0.454 | 0.825 | 0.729 | 0.827 | 0.878 |
| 1+3 | eval | 0.849 | 0.552 | 0.669 | 0.833 | 0.751 | 0.644 | 0.558 | 0.833 | 0.597 | 0.374 | 0.869 | 0.779 | 0.809 | 0.835 |
| 1+2+3 | train | 0.886 | 0.578 | 0.726 | 0.821 | 0.774 | 0.736 | 0.820 | 0.842 | 0.627 | 0.460 | 0.825 | 0.729 | 0.827 | 0.878 |
| 1+2+3 | eval | 0.855 | 0.554 | 0.672 | 0.833 | 0.753 | 0.654 | 0.605 | 0.833 | 0.600 | 0.374 | 0.869 | 0.779 | 0.809 | 0.835 |
| 1+2+3+4 | train | 0.888 | 0.578 | 0.727 | 0.821 | 0.774 | 0.742 | 0.820 | 0.842 | 0.627 | 0.460 | 0.825 | 0.729 | 0.827 | 0.878 |
| 1+2+3+4 | eval | 0.860 | 0.554 | 0.674 | 0.834 | 0.754 | 0.671 | 0.605 | 0.833 | 0.600 | 0.374 | 0.869 | 0.779 | 0.809 | 0.835 |

- Idea 1 carries almost all of the gain (occupation +0.18 train / +0.14 eval, QUASI +0.07 on both); no c loss.
- Idea 2 is small and free (+0.007 / +0.011 a, +0.001–0.003 b). Idea 3 alone does nothing and adds ≤ 0.001 on top
  of 1 — the note restates content in different words, not the same strings.
- Idea 4 is mixed: alone −0.001 a on train, +0.003 on eval; on top of 1+2+3 it adds +0.002 / +0.005 a. It drops
  some correct keeps (gymnastics, trampoline injuries) and wrong ones (hunting as a hobby) in equal measure.
- b and c are essentially untouched by all four ideas (|Δ| ≤ 0.004), as intended — the gain is pure category a.

## All runs
Studies: A = signals × keep_symptoms, B = personal rules 1–7, C = thresholds (old scorer), D = signal 5 (no entity
model), E = entity model × t_keep, F = entity check variants (all at t_keep 0.5), G = personal-data extensions on
the baseline (`Config.a_ideas`, below). a = personal data removed,
b = unrelated clinical removed, priv = a + b pooled (occurrence-weighted), c = code evidence kept.
Raw numbers: [leaderboard_runs.json](leaderboard_runs.json) (one record per run and split, with its source).

| # | study | configuration | train a | train b | train priv | train c | **train score** | eval a | eval b | eval priv | eval c | eval score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | G | **baseline + 1 extended rules + 2 social-history sentences + 3 cross-field propagation + 4 strict injury cue** | 0.888 | 0.578 | 0.727 | 0.821 | **0.774** | 0.860 | 0.554 | 0.674 | 0.834 | 0.754 |
| 2 | G | baseline + 1 extended rules + 2 social-history sentences + 3 cross-field propagation | 0.886 | 0.578 | 0.726 | 0.821 | **0.774** | 0.855 | 0.554 | 0.672 | 0.833 | 0.753 |
| 3 | G | baseline + 1 extended rules + 3 cross-field propagation | 0.884 | 0.575 | 0.724 | 0.821 | **0.773** | 0.849 | 0.552 | 0.669 | 0.833 | 0.751 |
| 4 | G | baseline + 1 extended rules | 0.883 | 0.575 | 0.723 | 0.821 | **0.772** | 0.849 | 0.552 | 0.669 | 0.833 | 0.751 |
| 5 | G | baseline + 2 social-history sentences | 0.858 | 0.576 | 0.712 | 0.821 | **0.767** | 0.833 | 0.554 | 0.663 | 0.833 | 0.748 |
| 6 | G | baseline + 3 cross-field propagation | 0.852 | 0.574 | 0.707 | 0.822 | **0.765** | 0.822 | 0.552 | 0.658 | 0.833 | 0.746 |
| 7 | F | entity, s1–5, t0.5, nearest-ICD check on all 4 kinds | 0.851 | 0.574 | 0.707 | 0.822 | **0.764** | 0.822 | 0.552 | 0.658 | 0.833 | 0.746 |
| 8 | E | entity, s1–5, t_keep 0.72 | 0.880 | 0.777 | 0.827 | 0.701 | **0.764** | 0.864 | 0.768 | 0.805 | 0.705 | 0.755 |
| 9 | G | baseline + 4 strict injury cue | 0.850 | 0.574 | 0.707 | 0.821 | **0.764** | 0.825 | 0.552 | 0.660 | 0.834 | 0.747 |
| 10 | E | entity, s1–5, t_keep 0.60 | 0.873 | 0.684 | 0.775 | 0.746 | **0.761** | 0.842 | 0.676 | 0.741 | 0.762 | 0.752 |
| 11 | D | no entity, s1–5, t_keep 0.72 | 0.815 | 0.508 | 0.656 | 0.862 | **0.759** | 0.807 | 0.498 | 0.620 | 0.856 | 0.738 |
| 12 | E | entity, s1–5, t_keep 0.50 | 0.853 | 0.600 | 0.722 | 0.794 | **0.758** | 0.823 | 0.588 | 0.681 | 0.804 | 0.742 |
| 13 | F | entity, s1–5, t0.5, check on tests/procedures/exam findings | 0.922 | 0.660 | 0.786 | 0.729 | **0.758** | 0.905 | 0.617 | 0.730 | 0.738 | 0.734 |
| 14 | D | no entity, s1–5, t_keep 0.85 | 0.816 | 0.534 | 0.670 | 0.843 | **0.756** | 0.809 | 0.521 | 0.634 | 0.839 | 0.737 |
| 15 | E | entity, s1–5, t_keep 0.85 | 0.881 | 0.810 | 0.844 | 0.665 | **0.754** | 0.865 | 0.795 | 0.823 | 0.664 | 0.743 |
| 16 | D | no entity, s1–5, t_keep 0.60 | 0.814 | 0.447 | 0.624 | 0.884 | **0.754** | 0.802 | 0.438 | 0.581 | 0.882 | 0.732 |
| 17 | F | entity, s1–5, t0.5, no check (like BC5CDR) | 0.924 | 0.684 | 0.799 | 0.703 | **0.751** | 0.906 | 0.652 | 0.752 | 0.709 | 0.731 |
| 18 | D | no entity, s1–5, t_keep 0.50 | 0.806 | 0.380 | 0.585 | 0.904 | **0.744** | 0.799 | 0.367 | 0.537 | 0.904 | 0.720 |
| 19 | F | entity, s1–5, t0.5, relevance-only check on all 4 kinds | 0.788 | 0.378 | 0.575 | 0.908 | **0.742** | 0.761 | 0.363 | 0.520 | 0.913 | 0.717 |
| 20 | B | s1–4, symptoms off + rules 1–7 (previous best) | 0.816 | 0.542 | 0.674 | 0.804 | **0.739** | 0.809 | 0.535 | 0.643 | 0.790 | 0.716 |
| 21 | C | previous best, t_keep 0.60 | 0.814 | 0.468 | 0.635 | 0.842 | **0.738** | 0.804 | 0.455 | 0.592 | 0.830 | 0.711 |
| 22 | C | previous best, t_keep 0.55 | 0.814 | 0.448 | 0.624 | 0.851 | **0.737** | 0.804 | 0.439 | 0.583 | 0.848 | 0.715 |
| 23 | C | previous best, t_keep 0.90 | 0.816 | 0.546 | 0.676 | 0.796 | **0.736** | 0.809 | 0.536 | 0.644 | 0.783 | 0.713 |
| 24 | E | entity, s1–5, t_keep 0.45 | 0.790 | 0.492 | 0.635 | 0.833 | **0.734** | 0.761 | 0.510 | 0.609 | 0.841 | 0.725 |
| 25 | C | previous best, t_keep 0.93 | 0.817 | 0.558 | 0.683 | 0.785 | **0.734** | 0.811 | 0.567 | 0.663 | 0.773 | 0.718 |
| 26 | B | s1–4, symptoms off + rules 1–6 | 0.760 | 0.533 | 0.642 | 0.813 | **0.728** | 0.733 | 0.528 | 0.609 | 0.805 | 0.707 |
| 27 | E | entity, s1–4, t_keep 0.72 | 0.882 | 0.835 | 0.858 | 0.591 | **0.724** | 0.865 | 0.816 | 0.835 | 0.583 | 0.709 |
| 28 | A | s1–4, symptoms off | 0.593 | 0.533 | 0.561 | 0.816 | **0.689** | 0.636 | 0.527 | 0.570 | 0.809 | 0.689 |
| 29 | B | s1–4, symptoms kept + rules 1–7 | 0.816 | 0.241 | 0.518 | 0.857 | **0.687** | 0.806 | 0.321 | 0.512 | 0.839 | 0.675 |
| 30 | A | s1–3, symptoms off | 0.594 | 0.542 | 0.567 | 0.805 | **0.686** | 0.636 | 0.558 | 0.588 | 0.799 | 0.694 |
| 31 | A | s1–2, symptoms off | 0.594 | 0.542 | 0.567 | 0.799 | **0.683** | 0.636 | 0.558 | 0.588 | 0.793 | 0.691 |
| 32 | B | s1–4, symptoms kept + rules 1–6 | 0.760 | 0.232 | 0.486 | 0.865 | **0.676** | 0.733 | 0.315 | 0.479 | 0.854 | 0.666 |
| 33 | A | s1, symptoms off | 0.594 | 0.546 | 0.569 | 0.780 | **0.674** | 0.636 | 0.559 | 0.589 | 0.776 | 0.682 |
| 34 | A | s1–4, symptoms kept (original baseline) | 0.593 | 0.232 | 0.406 | 0.867 | **0.636** | 0.636 | 0.313 | 0.440 | 0.858 | 0.649 |
| 35 | A | s1–3, symptoms kept | 0.594 | 0.235 | 0.407 | 0.862 | **0.635** | 0.636 | 0.324 | 0.447 | 0.851 | 0.649 |
| 36 | A | s1–2, symptoms kept | 0.594 | 0.235 | 0.407 | 0.856 | **0.632** | 0.636 | 0.324 | 0.447 | 0.847 | 0.647 |
| 37 | A | s1, symptoms kept | 0.594 | 0.236 | 0.408 | 0.838 | **0.623** | 0.636 | 0.325 | 0.448 | 0.831 | 0.639 |
