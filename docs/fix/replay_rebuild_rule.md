# Rebuild rule for S2-F in the step 3 replay (frozen with review_step2.md)

Sources: fix_prereg.md section 4 item 5, final_plan.md amendments 3 and 4, shadow_decision.md, and review_step1.md (addendum conditions). The replay must follow this text exactly. Any change before the day-40 report voids the freeze and must be logged as a deviation.

## Code
- All code runs from commit 3b92475e76cdbf3f9ccd43508b6ead5d2e79c029 (src tree 47670a8747ad04e7b652f8e1cb05b88d1e3768c1).
- HOUR_LABELS is "start_of_hour", set at runtime as in fix/fix_stage.py and fix/lear_stage_aligned.py.

## Inputs
- **History:** data/forecast/inputs_aligned.csv, produced by `thermo-fr align-inputs` at that commit from data/forecast/inputs.csv (sha256 c06e285f…).
- **Live days:** each laptop window saved under data/shadow/fresh/ for a counted arm A day is converted with `forecast.align.align_window`, using the corrected weights below. The converted windows are overlaid on the corrected history in run order, later windows winning, as shadow.py does for arm A.
- Windows are never overlaid on the as-logged inputs.csv, and no stored proxy column is reused.
- Nothing is re-downloaded. The non-blocking re-download diagnostic of final_plan.md amendment 4 is separate.

## honest_v2-F model files
One file for each published honest_v2 file that arm A used. Each is rebuilt to match it, and the match is checked against that file's honest_v2.json:
- same feature set (honest_v2);
- REFIT_PARAMS (300 trees, 31 leaves) over GBM_PARAMS, with the same seed;
- the same train_from and train_to;
- built with `forecast.refit.refit_sets` on the corrected table, truncated to train_to.

For the refit of 2 November, the history after inputs.csv ends is the same as-logged history that refit used. It is converted with `align-inputs` at the same commit.

## Proxy weights
- The solar and wind weights of each honest_v2-F file are `gen_proxy.latest_weights` on the trailing 365 days of the corrected table, ending where that refit's weights end.
- `align_window` applies them to the days that the file serves.

## LEAR-F
- Recalibrated every counted day with `forecast.lear` as in the shadow: windows of 364, 728, 1092 and 1456 days, AIC penalty, asinh scaling.
- Fitted on the corrected table with the converted windows overlaid.

## S2-F weights
- The same rule as arm A: inverse 14-day exponentially weighted MAE, settled Tuesday-to-Saturday days up to D-1, and equal weights until 14 days exist.
- The weight state comes from the F members' walk-forward forecasts over arm A's weight-state window (shadow.py STATE_START to STATE_END). It is used as weight state only and is never reported.

## Comparison
- honest_v2-F is compared with arm A's honest_v2, and S2-F with arm A's S2, on the same counted days.
- **Pass:** the mean daily difference is at most zero, and the upper end of the 90% moving-block interval is at most +1.0.
- A day with no saved window or a lost run is not replayed, and is reported.

## Amendment, 2026-10-18 (designer), made before any shadow score was read
1. Fallback days. On a counted day where arm A's S2 fell back to honest_v2, S2-F is scored as honest_v2-F that day. If only the replay cannot issue S2-F, it is scored as honest_v2-F and counted as a replay fallback.
2. Weight-state settings. The weight-state walk-forward forecasts of the F members use arm A's weight-state settings (GBM 800 trees, 63 leaves; LEAR as in selection).
3. The release question about the fallback model's label setting is out of scope of this replay.
