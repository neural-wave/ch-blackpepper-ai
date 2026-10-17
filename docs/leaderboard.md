# Leaderboard

Score = mean of privacy (a + b removed) and c_kept, both from `anonymizer/evaluate_gt.py` against the pseudo GT.
**Ranked by train** (140 encounters, where we tune); eval (67) shown for reporting only. Signal-5 runs on train are
out-of-fold. Details of every study: [experiments.md](experiments.md).

## Baseline (from 2026-10-17, study H: learned b/c decider)
**Study G baseline + learned b/c decider (E4): gradient-boosted classifier over relevance / context features plus the
zero-shot P(Yes) of Qwen3.5-4B (4-bit) replaces t_keep for detected clinical spans; keep if p ≥ 0.8.**

| split | a_removed | b_removed | privacy | c_kept | score |
|---|---|---|---|---|---|
| train | 0.895 | 0.761 | 0.826 | 0.867 | 0.846 |
| eval | 0.886 | 0.732 | 0.792 | 0.883 | 0.838 |

Config: study G config + `decider=True`, `decider_theta=0.8`, `llm_model="Qwen/Qwen3.5-4B"`, `llm_quant="4bit"`,
`llm_batch=1`; classifier in `.cache/decider.joblib` (`python -m anonymizer.decider --train`). Peak VRAM of the LLM
stage 3.47 GB (batch 1, hard 4 GB cap, RTX 4090); it runs alone after SapBERT / de-id are released.

Notes:
- Numbers from `experiments/bc_llm` (E4, RTX 4090; train out-of-fold). End-to-end `anonymizer.cli --vram-cap-gb 4`
  on eval reproduces them (a 0.887 / b 0.732 / privacy 0.793 / c 0.884 / score 0.838), peak VRAM 3.34 GB,
  7.9 s/encounter at `llm_batch=1` (RTX 4090). Train end-to-end pending.
- Previous baseline (study G, `--no-decider`): train 0.774 / eval 0.754 → +0.072 / +0.084.
- E5 (same without SapBERT features) ties: 0.844 / 0.840; E6 cascade (LLM on 18 % of spans) 0.838 / 0.824.

## Study G — personal-data extensions (2026-10-17)
On top of the previous baseline (`a_ideas=()`); switch `--a-ideas` / `Config.a_ideas`, all on by default since this study. Ideas: 1 extended rules (bare
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
the baseline (`Config.a_ideas`, below), H = learned b/c decider (classifier ± LLM judge, `anonymizer/decider.py`). a = personal data removed,
b = unrelated clinical removed, priv = a + b pooled (occurrence-weighted), c = code evidence kept.
Raw numbers: [leaderboard_runs.json](leaderboard_runs.json) (one record per run and split, with its source).

| # | study | configuration | train a | train b | train priv | train c | **train score** | eval a | eval b | eval priv | eval c | eval score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | H | **classifier, all features + Qwen3.5-4B 4-bit P(Yes), θ 0.8** | 0.895 | 0.761 | 0.826 | 0.867 | **0.846** | 0.886 | 0.732 | 0.792 | 0.883 | 0.838 |
| 2 | H | classifier, no embedding features + Qwen3.5-4B 4-bit P(Yes), θ 0.7 | 0.882 | 0.709 | 0.792 | 0.895 | **0.844** | 0.877 | 0.689 | 0.763 | 0.918 | 0.840 |
| 3 | H | cascade, E1 decides, uncertain band (±0.1) → Qwen3.5-4B 4-bit, θ 0.7 | 0.894 | 0.726 | 0.807 | 0.869 | **0.838** | 0.889 | 0.690 | 0.768 | 0.880 | 0.824 |
| 4 | H | learned classifier, all features (SapBERT), θ 0.8 | 0.903 | 0.750 | 0.824 | 0.824 | **0.824** | 0.893 | 0.711 | 0.783 | 0.820 | 0.801 |
| 5 | H | LLM judge only, Qwen3.5-4B 4-bit, θ 0.2 | 0.871 | 0.690 | 0.777 | 0.861 | **0.819** | 0.865 | 0.702 | 0.766 | 0.883 | 0.825 |
| 6 | H | learned classifier, no embedding features, θ 0.7 | 0.877 | 0.642 | 0.755 | 0.851 | **0.803** | 0.883 | 0.620 | 0.723 | 0.853 | 0.788 |
| 7 | H | LLM judge only, MedGemma 1.5 4B 4-bit, θ 0.3 | 0.848 | 0.563 | 0.700 | 0.856 | **0.778** | 0.834 | 0.601 | 0.693 | 0.846 | 0.769 |
| 8 | H | LLM judge only, Qwen3.5-2B 4-bit, θ 0.5 | 0.833 | 0.549 | 0.686 | 0.868 | **0.777** | 0.821 | 0.563 | 0.665 | 0.862 | 0.763 |
| 9 | G | baseline + 1 extended rules + 2 social-history sentences + 3 cross-field propagation + 4 strict injury cue | 0.888 | 0.578 | 0.727 | 0.821 | **0.774** | 0.860 | 0.554 | 0.674 | 0.834 | 0.754 |
| 10 | G | baseline + 1 extended rules + 2 social-history sentences + 3 cross-field propagation | 0.886 | 0.578 | 0.726 | 0.821 | **0.774** | 0.855 | 0.554 | 0.672 | 0.833 | 0.753 |
| 11 | H | baseline (relevance score >= t_keep 0.5), replayed in experiments/bc_llm, θ 0.5 | 0.888 | 0.578 | 0.727 | 0.821 | **0.774** | 0.860 | 0.554 | 0.674 | 0.834 | 0.754 |
| 12 | G | baseline + 1 extended rules + 3 cross-field propagation | 0.884 | 0.575 | 0.724 | 0.821 | **0.773** | 0.849 | 0.552 | 0.669 | 0.833 | 0.751 |
| 13 | G | baseline + 1 extended rules | 0.883 | 0.575 | 0.723 | 0.821 | **0.772** | 0.849 | 0.552 | 0.669 | 0.833 | 0.751 |
| 14 | G | baseline + 2 social-history sentences | 0.858 | 0.576 | 0.712 | 0.821 | **0.767** | 0.833 | 0.554 | 0.663 | 0.833 | 0.748 |
| 15 | G | baseline + 3 cross-field propagation | 0.852 | 0.574 | 0.707 | 0.822 | **0.765** | 0.822 | 0.552 | 0.658 | 0.833 | 0.746 |
| 16 | F | entity, s1–5, t0.5, nearest-ICD check on all 4 kinds | 0.851 | 0.574 | 0.707 | 0.822 | **0.764** | 0.822 | 0.552 | 0.658 | 0.833 | 0.746 |
| 17 | E | entity, s1–5, t_keep 0.72 | 0.880 | 0.777 | 0.827 | 0.701 | **0.764** | 0.864 | 0.768 | 0.805 | 0.705 | 0.755 |
| 18 | G | baseline + 4 strict injury cue | 0.850 | 0.574 | 0.707 | 0.821 | **0.764** | 0.825 | 0.552 | 0.660 | 0.834 | 0.747 |
| 19 | E | entity, s1–5, t_keep 0.60 | 0.873 | 0.684 | 0.775 | 0.746 | **0.761** | 0.842 | 0.676 | 0.741 | 0.762 | 0.752 |
| 20 | D | no entity, s1–5, t_keep 0.72 | 0.815 | 0.508 | 0.656 | 0.862 | **0.759** | 0.807 | 0.498 | 0.620 | 0.856 | 0.738 |
| 21 | E | entity, s1–5, t_keep 0.50 | 0.853 | 0.600 | 0.722 | 0.794 | **0.758** | 0.823 | 0.588 | 0.681 | 0.804 | 0.742 |
| 22 | F | entity, s1–5, t0.5, check on tests/procedures/exam findings | 0.922 | 0.660 | 0.786 | 0.729 | **0.758** | 0.905 | 0.617 | 0.730 | 0.738 | 0.734 |
| 23 | D | no entity, s1–5, t_keep 0.85 | 0.816 | 0.534 | 0.670 | 0.843 | **0.756** | 0.809 | 0.521 | 0.634 | 0.839 | 0.737 |
| 24 | E | entity, s1–5, t_keep 0.85 | 0.881 | 0.810 | 0.844 | 0.665 | **0.754** | 0.865 | 0.795 | 0.823 | 0.664 | 0.743 |
| 25 | D | no entity, s1–5, t_keep 0.60 | 0.814 | 0.447 | 0.624 | 0.884 | **0.754** | 0.802 | 0.438 | 0.581 | 0.882 | 0.732 |
| 26 | F | entity, s1–5, t0.5, no check (like BC5CDR) | 0.924 | 0.684 | 0.799 | 0.703 | **0.751** | 0.906 | 0.652 | 0.752 | 0.709 | 0.731 |
| 27 | D | no entity, s1–5, t_keep 0.50 | 0.806 | 0.380 | 0.585 | 0.904 | **0.744** | 0.799 | 0.367 | 0.537 | 0.904 | 0.720 |
| 28 | F | entity, s1–5, t0.5, relevance-only check on all 4 kinds | 0.788 | 0.378 | 0.575 | 0.908 | **0.742** | 0.761 | 0.363 | 0.520 | 0.913 | 0.717 |
| 29 | B | s1–4, symptoms off + rules 1–7 (previous best) | 0.816 | 0.542 | 0.674 | 0.804 | **0.739** | 0.809 | 0.535 | 0.643 | 0.790 | 0.716 |
| 30 | C | previous best, t_keep 0.60 | 0.814 | 0.468 | 0.635 | 0.842 | **0.738** | 0.804 | 0.455 | 0.592 | 0.830 | 0.711 |
| 31 | C | previous best, t_keep 0.55 | 0.814 | 0.448 | 0.624 | 0.851 | **0.737** | 0.804 | 0.439 | 0.583 | 0.848 | 0.715 |
| 32 | C | previous best, t_keep 0.90 | 0.816 | 0.546 | 0.676 | 0.796 | **0.736** | 0.809 | 0.536 | 0.644 | 0.783 | 0.713 |
| 33 | E | entity, s1–5, t_keep 0.45 | 0.790 | 0.492 | 0.635 | 0.833 | **0.734** | 0.761 | 0.510 | 0.609 | 0.841 | 0.725 |
| 34 | C | previous best, t_keep 0.93 | 0.817 | 0.558 | 0.683 | 0.785 | **0.734** | 0.811 | 0.567 | 0.663 | 0.773 | 0.718 |
| 35 | B | s1–4, symptoms off + rules 1–6 | 0.760 | 0.533 | 0.642 | 0.813 | **0.728** | 0.733 | 0.528 | 0.609 | 0.805 | 0.707 |
| 36 | E | entity, s1–4, t_keep 0.72 | 0.882 | 0.835 | 0.858 | 0.591 | **0.724** | 0.865 | 0.816 | 0.835 | 0.583 | 0.709 |
| 37 | A | s1–4, symptoms off | 0.593 | 0.533 | 0.561 | 0.816 | **0.689** | 0.636 | 0.527 | 0.570 | 0.809 | 0.689 |
| 38 | B | s1–4, symptoms kept + rules 1–7 | 0.816 | 0.241 | 0.518 | 0.857 | **0.687** | 0.806 | 0.321 | 0.512 | 0.839 | 0.675 |
| 39 | A | s1–3, symptoms off | 0.594 | 0.542 | 0.567 | 0.805 | **0.686** | 0.636 | 0.558 | 0.588 | 0.799 | 0.694 |
| 40 | A | s1–2, symptoms off | 0.594 | 0.542 | 0.567 | 0.799 | **0.683** | 0.636 | 0.558 | 0.588 | 0.793 | 0.691 |
| 41 | B | s1–4, symptoms kept + rules 1–6 | 0.760 | 0.232 | 0.486 | 0.865 | **0.676** | 0.733 | 0.315 | 0.479 | 0.854 | 0.666 |
| 42 | A | s1, symptoms off | 0.594 | 0.546 | 0.569 | 0.780 | **0.674** | 0.636 | 0.559 | 0.589 | 0.776 | 0.682 |
| 43 | A | s1–4, symptoms kept (original baseline) | 0.593 | 0.232 | 0.406 | 0.867 | **0.636** | 0.636 | 0.313 | 0.440 | 0.858 | 0.649 |
| 44 | A | s1–3, symptoms kept | 0.594 | 0.235 | 0.407 | 0.862 | **0.635** | 0.636 | 0.324 | 0.447 | 0.851 | 0.649 |
| 45 | A | s1–2, symptoms kept | 0.594 | 0.235 | 0.407 | 0.856 | **0.632** | 0.636 | 0.324 | 0.447 | 0.847 | 0.647 |
| 46 | A | s1, symptoms kept | 0.594 | 0.236 | 0.408 | 0.838 | **0.623** | 0.636 | 0.325 | 0.448 | 0.831 | 0.639 |
