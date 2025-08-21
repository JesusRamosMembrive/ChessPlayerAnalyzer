# Errors and fixes (focused on app/analysis)

This document enumerates the issues found in calculations/methodology and proposes concrete, implementation-ready fixes, prioritizing changes in `app/analysis/*`. Only mention `test/run_local_analysis.py` if there is an intrinsic defect in that script; otherwise keep all proposals centered on `app/analysis`.

Sources:
- Logs: test/result2Text2.txt
- Quality: app/analysis/quality.py
- Timing: app/analysis/timing.py

1) ACPL is defined as eval swing, not loss-to-best
- Evidence:
  - In quality.acpl(), ACPL is mean(abs(eval_cp_after - eval_cp_before)) adjusted by color. This measures “evaluation swing” and NOT the distance to the engine’s top move.
  - Consequence: It penalizes strong moves that drastically improve evaluation for the player and may under-penalize bad moves with small swings.
- Observed impact: The DataFrame contains an extreme delta_eval (99491 at move 17) and the resulting ACPL is ~5600.44, dragging IPR and quality_score to absurd values.
- Fix in app/analysis/quality.py:
  - Switch ACPL to “loss vs engine best” when the DataFrame provides delta_eval (intended to be the loss-to-best = distance to PV[0]).
  - Implement in quality.acpl() the following:
    - If delta_eval exists: use mean(abs(delta_eval)) as ACPL.
    - If delta_eval is missing: fallback to mean(abs(eval_cp_after - eval_cp_before)).
  - This ensures the canonical definition (loss vs best) lives in app/analysis, regardless of DF construction.
- Concrete changes:
  - quality.acpl():
    - Detect delta_eval and prefer it over raw eval swings.
    - Keep color adjustment only for the fallback path.
  - aggregate_quality_features(): no signature change; it benefits from corrected acpl().

2) Mate scores blow up ACPL/IPR
- Evidence:
  - Logs show delta_eval=99491 on move 17 (mate nearby), ACPL ≈ 5600.44. That yields IPR ≈ -666.89 and quality_score ≈ -2191.68.
- Root cause:
  - Mate evaluations get mapped to huge centipawn magnitudes (e.g., 100000) contaminating aggregates.
- Fix in app/analysis/quality.py:
  - Add robustness in ACPL computation:
    1) Cap losses for ACPL, e.g., clip delta_eval to [-1500, 1500] before averaging.
    2) Optionally exclude moves whose PV contains a mate from ACPL if such a flag exists in DF; if not, apply the cap (solution 1).
    3) Alternatively, use robust aggregators (median or 10–20% trimmed mean).
  - Minimal safe recommendation: apply a symmetric cap (e.g., 1500 cp) to delta_eval in quality.acpl() whenever delta_eval is present. This protects aggregates from single terminal outliers.
- Concrete changes:
  - quality.acpl():
    - If using delta_eval, apply optional cap parameter cap_cp=1500 by default.
    - Document the cap and rationale.
  - aggregate_quality_features(): unchanged; it picks up the robust ACPL.

3) Keep ACPL coherent with best_rank/is_engine_best
- Evidence:
  - best_rank and is_engine_best derive from MultiPV; ACPL should measure loss to PV[0] to align conceptually.
- Fix in app/analysis/quality.py:
  - Ensure acpl() uses delta_eval when available (see 1). If missing, log a warning that fallback (eval swing) is used.

4) Phase-aware ACPL and blunder stability
- Evidence:
  - DF already has phase (opening/middlegame/endgame). Terminal/endgame swings bias overall ACPL.
- Fix in app/analysis/quality.py:
  - Add per-phase ACPL outputs inside aggregate_quality_features() when phase is present, reusing a helper that computes ACPL per phase using robust delta_eval.
  - Alternatively expose a new helper phase_acpl(game_df) and let higher layers include it. quality.compute_phase_quality() already exists for lists of DFs and can be mirrored.

5) quality_score robustness
- Evidence:
  - quality_score mixes ACPL, match_rate, weighted_match_rate; if ACPL explodes, the score collapses.
- Fix in app/analysis/quality.py:
  - Once ACPL is robust (cap/median), quality_score improves automatically. Additionally:
    - Normalize ACPL to a [0,1] range for scoring (e.g., scaled_acpl = 1 - min(acpl, 100) / 100).
    - Document expected ranges and caps.

6) Reproducibility of depth/time (interpretation/logging)
- Evidence:
  - Logs show go depth 12 movetime 5000 repeatedly. Fixed time with a target depth may produce variability.
- Fix (non-engine change) in app/analysis/quality.py:
  - Improve logging to include effective depth if it arrives in DF and warn when below target. This improves interpretability without changing engine use.

7) Engine failures: exclude from aggregates
- Evidence:
  - In this run there were none. Generally, rows with missing/invalid evals should be excluded rather than imputing zeros.
- Fix in app/analysis/quality.py:
  - If DF includes flags/NaNs for unavailable evals, exclude those rows from acpl(), match_rate, etc., and log counts.

8) Typing/robustness warnings
- Evidence:
  - Static diagnostics on ACPLModel predict/fit inputs and .values on NDArray.
- Fix in app/analysis/quality.py:
  - Use .to_numpy(dtype=float) and concrete types for sklearn calls; minimize reliance on .values over NDArray.

Summary of proposed changes (all in app/analysis/quality.py)
- acpl(game_df, player_color='white', cap_cp: int | None = 1500):
  - Use delta_eval if present (with cap); fallback to eval swing adjusted by color otherwise.
  - Exclude invalid/NaN rows and log usage counts.
- aggregate_quality_features(game_df, elo=None, player_color='white'):
  - Inherits robust acpl(); optionally include phase ACPL if phase present.
  - Consider normalizing ACPL inside quality_score.
- Optional: phase_acpl(game_df, cap_cp=1500) helper.
- Minor typing/logging fixes across ACPLModel and helpers.

References to code and log anchors
- app/analysis/quality.py: acpl(), aggregate_quality_features(), compute_phase_quality() lines ~23–51, 261–312, 163–195.
- app/analysis/timing.py: aggregate_time_features() lines ~173–227 showing timing uses same DF.
- test/result2Text2.txt: shows delta_eval=99491 at move 17; ACPL ≈ 5600.44; IPR ≈ -666.89; quality_score ≈ -2191.68; MultiPV and go depth interactions at lines ~28–42 and ~480–492.
