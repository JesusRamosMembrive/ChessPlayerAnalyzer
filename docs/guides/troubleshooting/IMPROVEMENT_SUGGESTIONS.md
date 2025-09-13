[UPDATE — 2025-08-21]
Suggestion status after recent changes:
- Already implemented (not restated here): robust, capped ACPL in quality.py; timing robustness/traceability with meta fields; longitudinal slope guards to avoid polyfit issues on single-game inputs.
- The remaining suggestions below are still valid as future, optional enhancements.

# Additional improvement suggestions (non-critical, positive for the project)

These are enhancements to strengthen interpretability, stability, and auditability of metrics in `app/analysis/*`. They are not direct bug fixes.

1) Robust loss metric alongside ACPL
- Provide a “robust loss” metric (median or trimmed mean at 10–20%) on capped delta_eval to complement ACPL and reduce outlier sensitivity.

2) WDL-scaled loss
- Map evaluations to WDL probabilities and compute a [0,1] bounded loss vs best WDL move. This avoids mate blowups and focuses on result impact.

3) Standardized phase report
- Return a consistent phase_quality block including:
  - opening_acpl, middlegame_acpl, endgame_acpl (robust),
  - opening_blunder_rate, middlegame_blunder_rate, endgame_blunder_rate.
- Helps avoid overinterpretation of endgame outliers.

4) Normalization in quality_score
- Normalize ACPL to [0,1] before mixing with match rates (cap at 100cp for scoring). Document expected ranges and saturation behavior.

5) Second-choice behavior
- Expose a per-move “second_choice_rate” and by-phase variant: frequency of choosing PV[2] when PV[0] and PV[1] are close. Potential signal of “playing around” the best line.

6) Joint quality+timing investigative signals
- Add a lightweight helper in app/analysis to combine clutch_accuracy with robust ACPL windows to detect “pause then perfect” sequences.

7) Pre/post sanity logs
- In aggregate_quality_features() and aggregate_time_features():
  - Log % of usable rows for acpl/match_rate/correlation,
  - Log distribution quantiles (p10–p90) for delta_eval and move_time,
  - Log count of suspected mate-driven extremes (above cap).

8) Configurability
- Make caps, trims, and blunder thresholds configurable via kwargs with sensible defaults to facilitate calibration experiments.

9) Reproducibility/audit metadata
- Include a small meta section in quality outputs, e.g. {acpl_cap_used: 1500, used_delta_eval: true, rows_used: n, rows_total: N}.

10) Typing and consistency
- Prefer .to_numpy(dtype=float) to avoid NDArray .values pitfalls.
- Strengthen type annotations for sklearn calls and numpy arrays to reduce static analysis warnings.

References
- app/analysis/quality.py: aggregate_quality_features(), compute_phase_quality(), acpl().
- app/analysis/timing.py: aggregate_time_features().
- test/result2Text2.txt: engine depth/time pattern; presence of extreme delta_eval near mate that motivates robust handling.
