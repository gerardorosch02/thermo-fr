# Independent review: the yardstick and LEAR to the published recipe

Reviewer's files are in this folder. Reviewed on 2026-11-17 against branch lear-recipe at eb5ea6b (working tree clean before
and after), pre-registration docs/prereg/lear_recipe_prereg.md, outputs in the builder's working folder.

## Verdict

**Confirmed with corrections.**

Every figure in results.json was recomputed with my own code from inputs.csv, the raw chunk files and the pre-registration
text, and every one agrees to at least ten significant digits. The rule was applied as written. No candidate passes, so
"LEAR stays as it is" is the correct outcome under the pre-registration.

Three corrections are needed before anything is reported.

1. There is look-ahead on one delivery day, 2024-02-17, in every LEAR-type forecast (L0, A, B, C and therefore S1). It predates
   this experiment and sits in the logged 15.36 and 14.36 as well. The builder's scrambling test passes on its 20 sampled
   days, but the sample does not contain this day, and the builder's own scramble applied to it changes the forecast by up to
   43 EUR/MWh. No verdict changes when the day is removed.
2. The summary "C improves the average by about 0.52 but is worse on the most expensive hours" is true and incomplete. All
   three candidates are worse than L0 overall from 2025-10-01, C is worse than L0 in every band of actual price above 50
   EUR/MWh (55% of hours), and C's gain comes from 2024 H1 and 2025 H1 while 2026 H1 goes the other way.
3. C's fail on condition (ii) is correct as registered, but the margin is not statistically secure and three days decide
   it. It should be reported as a fail by the rule, not as a decisive finding about C.

## Numbers recomputed

Strict selection rows built from inputs.csv by my own code: delivery days 2024-02-17 to 2026-06-30, weather issued two days
ahead present for all three variables, price present. 20,752 hours and 865 days, the same as the builder. L0, A, B, S1 and
LEAR-AR were rebuilt from the raw chunk files. C was rebuilt with my own rolling slope choice (plain loops over D-56 to
D-1); 0 of 20,752 strict cells differ from the builder's choice. honest_v2 was not refitted, see the last line of the table.

| Quantity | Builder | Reviewer | Match |
|---|---|---|---|
| Pre-registration sha256 | dc032197...b39d | dc0321976f7bcc1ed1fa9f69fe8bd52cf55a4ffd8527164b0f49e12b7474b39d | yes |
| Strict hours, days | 20,752, 865 | 20,752, 865 | yes |
| MAE honest_v2 | 15.7040 | 15.7040 | yes (builder's stored forecasts) |
| MAE L0 | 15.3646 | 15.3646 | yes |
| MAE S1 | 14.3608 | 14.3608 | yes |
| MAE A | 15.3724 | 15.3724 | yes |
| MAE B | 15.1098 | 15.1098 | yes |
| MAE C | 14.8420 | 14.8420 | yes |
| MAE LEAR-AR (M6) | 17.5514 | 17.5514 | yes |
| A: gain, worst slice, one-sided upper | -0.0078, top 5% +2.924, +0.4283 | -0.0078, +2.924, +0.4283 | yes |
| B: gain, worst slice, one-sided upper | 0.2548, top 5% +2.044, +0.0364 | 0.2548, +2.044, +0.0364 | yes |
| C: gain, worst slice, one-sided upper | 0.5226, top 5% +1.366, -0.1794 | 0.5226, +1.366, -0.1794 | yes |
| C paired daily difference, 95% | -0.5223 [-0.853, -0.207] | -0.5223 [-0.853, -0.207] | yes |
| L0 slices (weekends, holidays, windiest, sunniest, top 5%) | 15.36, 19.40, 17.56, 15.57, 20.91 | 15.362, 19.396, 17.557, 15.569, 20.909 | yes |
| C minus L0 by slice | -1.12, -0.98, -0.09, +0.28, +1.37 | -1.121, -0.980, -0.088, +0.283, +1.366 | yes |
| B minus L0 by slice | -0.67, -1.07, +0.59, +0.54, +2.04 | -0.669, -1.062, +0.589, +0.538, +2.044 | yes |
| A minus L0 by slice | -0.53, -1.14, +1.56, +1.14, +2.92 | -0.535, -1.139, +1.559, +1.142, +2.924 | yes |
| Verdicts (i), (ii), (iii) | A fail fail fail; B pass fail fail; C pass fail pass | the same | yes |
| M1 naive MAE (D-1, D-7, Lago) | 21.8330, 31.3902, 24.4914 | 21.8330, 31.3902, 24.4914 | yes |
| M1 rMAE against Lago: honest_v2, L0, S1, LEAR-AR | 0.6412, 0.6273, 0.5864, 0.7166 | 0.6412, 0.6273, 0.5864, 0.7166 | yes |
| M1 rMAE against D-1: honest_v2, L0, S1 | 0.7193, 0.7037, 0.6578 | 0.7193, 0.7037, 0.6578 | yes |
| M2 hourly slope: honest_v2, L0, S1, LEAR-AR | 0.9054, 0.8893, 0.9820, 0.8544 | 0.9054, 0.8893, 0.9820, 0.8544 | yes |
| M2 daily slope: honest_v2, L0, S1, LEAR-AR | 0.8598, 0.9156, 0.9777, 0.8596 | 0.8598, 0.9156, 0.9777, 0.8596 | yes |
| M3 honest_v2 minus S1 | +1.3427 [1.024, 1.687] | +1.3427 [1.024, 1.687] | yes |
| M3 L0 minus S1 | +1.0043 [0.711, 1.289] | +1.0043 [0.711, 1.289] | yes |
| M4 mean Spearman: honest_v2, L0, S1, D-1 profile | 0.8685, 0.8778, 0.8871, 0.7572 | 0.8685, 0.8778, 0.8871, 0.7572 | yes |
| M4 spread capture: honest_v2, L0, S1, D-1 profile | 0.9075, 0.9188, 0.9263, 0.7935 | 0.9075, 0.9188, 0.9263, 0.7935 | yes |
| M5 split sizes | 592 days 14,200 h; 273 days 6,552 h | the same | yes |
| M5 MAE before, from: honest_v2 | 15.466, 16.220 | 15.466, 16.220 | yes |
| M5 MAE before, from: L0 | 15.029, 16.093 | 15.029, 16.093 | yes |
| M5 MAE before, from: S1 | 14.015, 15.110 | 14.015, 15.110 | yes |
| M5 MAE before, from: A | 14.650, 16.939 | 14.650, 16.939 | yes |
| M5 MAE before, from: B | 14.481, 16.472 | 14.481, 16.472 | yes |
| M5 MAE before, from: C | 14.133, 16.378 | 14.133, 16.378 | yes |
| M6 L0 minus LEAR-AR | -2.188 [-2.603, -1.759] | -2.188 [-2.603, -1.759] | yes |
| M7 complete days, tercile cuts | 787 of 791; 3.088 and 8.703 GW | 787 of 791; 3.088 and 8.703 GW | yes |
| M7 daily share explained: pooled, terciles 1 to 3 | 0.0309, 0.0156, 0.0408, 0.0615 | 0.0309, 0.0156, 0.0408, 0.0615 | yes |
| M7 hourly share explained: pooled, tercile 3 | 0.0194, 0.0412 | 0.0194, 0.0412 | yes |
| M7 daily wind coefficient: pooled, tercile 3 | -1.788, -2.632 | -1.788, -2.632 | yes |
| Negative-price hours (1,264): L0, A, B, C | 21.51, 17.84, 18.24, 17.46 | 21.508, 17.843, 18.240, 17.458 | yes |
| Volatility terciles, C | 13.178, 15.248, 16.095 | 13.178, 15.248, 16.095 | yes |
| Per-window MAE at slope 1 (56 to 1456) | 18.124, 17.876, 16.196, 15.751, 15.749, 15.820 | the same | yes |
| Fixed-slope MAE (0.1 to 1) | 14.962, 14.887, 14.982, 15.091, 15.110 | the same | yes |
| Slope share of strict hours (0.1 to 1) | 41.4%, 16.3%, 6.9%, 7.3%, 28.1% | the same | yes |
| L0 minus honest_v2, against the figure logged on 2026-10-16 | logged -0.338 [-0.920, +0.211] | -0.3383 [-0.9201, +0.2109] | yes |
| honest_v2 forecasts themselves | gbm_honest_v2.csv | not refitted | not checked independently |

No number differs beyond rounding. The moving-block bootstrap is my own implementation (blocks of 7 consecutive days, starts
uniform, ceil(n/7) blocks cut to n, 5000 resamples, seed 0, percentile interval, 98.33rd percentile for the one-sided
upper end); it draws the same random numbers as the project's function and reproduces its intervals exactly.

## Departures from the pre-registration

None of these can change a verdict. Each is listed with the reason.

1. scikit-learn version (already disclosed). The pre-registration says the noise-variance change came in 1.1. Checked
   against the changelog: the corrected AIC and the error for more regressors than rows came in 1.0.2; the `noise_variance`
   argument was exposed in 1.1. The short-window rule coded is the old criterion either way (a unit test rebuilds the old
   formula along the LARS path and gets the same penalty). No numerical effect. Note that the old default also normalised
   the columns, which is not reproduced; the pre-registration did not ask for it.
2. `MIN_SHORT_ROWS = 14` in lear.py is not in the pre-registration. It never applied: every 56-day and 84-day window had
   exactly 56 and 84 complete rows on all 921 days, and no chunk holds a missing value.
3. L0 was not produced by calling the present code path. The runner fits all six windows with the short-window option
   switched on and L0 is the average of four of them. This is the same model only if no long window ever falls to 303 rows
   or fewer. Checked: the minimum is 364, 728, 1078 and 1078 rows against a threshold of 303, so the option never applies
   to a long window. Confirmed directly as well, see "L0 reproduction" below.
4. Condition (ii) is computed on slice MAEs rounded to 0.01, as the sprint did. The pre-registration is silent. Unrounded
   differences give the same three verdicts (C: +1.366).
5. Rolling slope choice, a reading the text leaves open. The code takes the mean over hourly rows, so the repeated hour 2
   of the autumn clock-change day counts twice. A mean over days would change the chosen slope in 7 of 20,752 strict
   cells (seven days in November and December 2024). C's MAE moves from 14.84199 to 14.84198 and the top 5% difference
   does not move.
6. Windows, slope grid, transform, tie rule, start date, bootstrap settings, gain bar, slice limit and the slice set for
   (ii): all as written. Grid {0.1, 0.3, 0.5, 0.7, 1.0}; transform sign(x)[asinh(|x| + a) - asinh(a)], a = sqrt(1/c^2 - 1),
   on the target, the four French lag blocks, CH and DE-LU, with the project's median and MAD x 1.4826 scaling; selection
   over delivery days D-56 to D-1 per clock hour; ties to the larger c (no exact tie occurred); every slope from
   2023-12-23; (ii) on weekends, holidays, windiest 10%, sunniest 10% and top 5% hours only (negative-price hours are
   reported and not in the rule).
7. The longest window is shorter than its name for part of the period. Data start on 2021-01-01 and the first complete
   row is 2021-01-09, so the "1456-day" window is not full until 2025-01-04, which covers 322 of the 865 strict days. This
   is how L0 has always run and the pre-registration does not mention it. It weakens the claim that A is "the published
   window set" for 2024; it does not affect the comparison.
8. Reproduction of L0 and S1 is asserted at two decimals only (15.36, 14.36), because that is all that was logged. That
   is what the pre-registration asks. Stronger evidence is given below.
9. Order of work. results.json and the headline were produced at 00:36 while the scrambling test, a stop condition, was
   still running. It finished at 01:08 and passes on its sample. Separately, the first chunks (12:44 to 13:03 on 16
   November) were written before the code was committed (24db6dc at 12:52, eb5ea6b at 13:07) and before lear_recipe.py
   reached its final state (13:03). lear.py, which holds the fit, was final at 12:39. My refit of 2024-03-05, a day from
   that first batch, reproduces the stored chunk exactly with the committed code.
10. Choices the pre-registration leaves to the builder and that I would have made the same way: sample standard deviation
    for the volatility terciles, "complete" days in M7 (every hour strict, with forecast, price, load forecast, both
    proxies, onshore wind, solar and the early nuclear mean), offshore wind missing counted as zero, tercile upper bounds
    inclusive. My M7 was written from the text alone and gives the same 787 days and the same coefficients.

## Look-ahead checks

**Reading the code.** For delivery day D each window trains on design rows D-w to D-1 (`window_days`, `_fit_predict_window`).
The scalers come from the training rows only. The row for D holds French prices of D-1, D-2, D-3 and D-7, CH and DE-LU
prices of D-1, the load forecast for D, D-1 and D-7, the weather forecast for D, the mean nuclear output of the D-1 hours
before 08:00, and calendar dummies. All of these are public by 10:05 Paris on D-1 under timing.py, and `assert_gate` is
called for every chunk at both deadlines. The rolling choice for D sums absolute errors over rows D-56 to D-1
(`sliding_window_view(...)[:-1]` assigned from row 56 onward, which I checked index by index and against my own loop). The
price of D-1 is published at about 13:00 on D-2, so every error used is settled before the forecast for D is issued. The
row's own price is never read for its own choice. I found no path from day D or later into the forecast for D, with one
exception.

**The exception: 2024-02-17.** `assert_gate` checks the declared timing rule of each column, not where the value came from.
`build_features` fills the three weather columns from the not-point-in-time proxy archive wherever the issued forecast is
missing, hour by hour, and the strict flag is set hour by hour. LEAR uses the whole day's 24-hour weather vectors for every
hour's model. On 2024-02-17 the issued wind forecast is missing for hours 0 to 6, so those seven hours are not strict and
carry proxy weather (the latest model run before each hour, published after the gate). The other 17 hours are strict and
are scored, and their LEAR forecasts read the proxy weather of hours 0 to 6. It is the only such day.

- My truncation of the inputs makes the forecast for this day impossible to compute (the design row is incomplete).
- With the proxy weather of that day put back, the stored L0 is reproduced to 7e-15, which proves the dependence.
- The builder's own `scramble_from` applied to this day changes the stored 364-day-window forecast by up to 43 EUR/MWh,
  12.5 on average over hours 7 to 23. The same scramble on 2025-10-26 changes nothing (7e-15).
- Replacing the early-hour weather with the same hours of D-1 (public in time) moves the L0 forecast of the 17 strict hours
  by 12.1 EUR/MWh on average, and their MAE from 6.7 to 12.9. This is a sensitivity, not a corrected forecast.
- Without this day (20,735 hours, 864 days): gains A -0.0087, B 0.2542, C 0.5222; top 5% differences +2.93, +2.05, +1.37;
  one-sided upper ends +0.435, +0.031, -0.185. Every verdict is unchanged. L0 is 15.3717, S1 14.3687, honest_v2 15.7124.

By the letter of the stop condition ("any look-ahead found by the scrambling test") this is the kind of finding that
stops a run. It is confined to 17 of 20,752 hours, it is in the baseline and the candidates alike, it was already in the
sprint's LEAR and S1, and it does not move any verdict. honest_v2 and LEAR-AR do not have it (per-hour features; no
weather blocks). It should be logged and the strict set for day-vector models should start on 2024-02-18.

**The builder's scrambling test.** Design: for a sampled day D, every column of every later day is replaced by noise, every
column on D except the load forecast, the weather issued two days ahead and the generation proxies, and nuclear output of
D-1 from 08:00. All 30 (slope, window) forecasts for D are refitted and compared with the stored chunks at 1e-9. Then the
rolling choice is rerun with prices from D onward and forecasts after D replaced by noise. This design would catch a leak
of the kinds that matter for LEAR: later rows in a training window, scalers fitted beyond D-1, prices or neighbour prices
of D, late nuclear hours, the proxy weather of D, and any use of D's price in the choice. It covers both short and long
windows and every slope, whatever slope C chose. Its limits: it samples 20 random days of 865 and forces no edge days,
which is why it missed 2024-02-17; it does not cover honest_v2 or the LEAR-AR chunks; and no scrambling test can detect a
vintage problem (a value in inputs.csv that was revised after its publication time, for example the load forecast, of
which the API keeps only the latest version).

Result (scramble_check.json, finished 01:08): 20 days, largest change 5.7e-14 EUR/MWh over all paths (short and long
windows, slope 1 and below), rolling choice identical on all 20. It passes on its sample.

**My truncation test.** Instead of scrambling I deleted: every row after D dropped; on D only the load forecast, issued
weather and generation proxies kept and every other column set to missing; nuclear and actual wind and solar generation
of D-1 from 08:00 set to missing. The slope choice for D was recomputed from the stored forecasts of days before D only.
Days: four drawn with my own seed (2024-06-11, 2024-07-30, 2024-08-16, 2025-02-19) and six chosen on purpose (the first
strict day 2024-02-17, 2024-03-05 from the first batch of chunks, 2025-03-31 after the spring clock change, 2025-10-26
the autumn clock-change day, and 2026-03-09 and 2026-06-24, the two days that weigh most in C's top 5% loss).

| Day | Slopes chosen (hours at 0.1, 0.3, 0.5, 0.7, 1) | L0 by the code of 0459b68 | Refit against chunk | B | C |
|---|---|---|---|---|---|
| 2024-02-17 | 0, 0, 0, 1, 23 | cannot be computed | cannot be computed | cannot be computed | cannot be computed |
| 2024-03-05 | 0, 0, 0, 0, 24 | not run | 1.4e-14 | 2.8e-14 | 2.8e-14 |
| 2024-06-11 | 12, 7, 0, 0, 5 | 7.1e-15 | 7.1e-15 | 7.1e-15 | 7.1e-15 |
| 2024-07-30 | 9, 7, 7, 0, 1 | 1.4e-14 | 1.4e-14 | 1.4e-14 | 1.4e-14 |
| 2024-08-16 | 13, 3, 5, 1, 2 | 1.4e-14 | 1.4e-14 | 2.8e-14 | 2.8e-14 |
| 2025-02-19 | 11, 5, 4, 1, 3 | 2.8e-14 | 2.8e-14 | 4.3e-14 | 2.8e-14 |
| 2025-03-31 | 12, 2, 3, 2, 5 | 7.1e-15 | 1.4e-14 | 1.4e-14 | 2.8e-14 |
| 2025-10-26 | 15, 4, 0, 1, 4 | 3.6e-15 | 7.1e-15 | 3.6e-15 | 7.1e-15 |
| 2026-03-09 | 3, 4, 0, 3, 14 | 2.8e-14 | 2.8e-14 | 1.4e-14 | 1.4e-14 |
| 2026-06-24 | 15, 5, 0, 0, 4 | 2.8e-14 | 5.7e-14 | 2.8e-14 | 2.8e-14 |

Figures are the largest absolute difference in EUR/MWh from the stored forecast. Nine of ten days reproduce L0, B and C
from inputs that hold nothing unknown on the morning of D-1, and my slope choice equals the builder's on all ten. Eight of
them include hours where C chose a slope below 1. LEAR-AR was refitted the same way for three days (2024-07-30, 2025-03-31,
2026-03-09) and matches its chunks to 1.4e-14.

**Holdout.** `selection_inputs` cuts the inputs before 2026-07-01 at the start of every stage (gram check, gradient
boosting, LEAR chunks, assembly, report, M7, scramble) and refuses a later end. No output file holds a date after
2026-06-30 (frame.csv, slopes.csv, gbm_honest_v2.csv, 26 prediction files, 1,050 chunks, results.json). The slope choice
reads only the frame from 2023-12-23 to 2026-06-30. Two of the builder's timing scripts (scripts/probe0.py, probe1.py)
loaded the whole inputs file, holdout rows included, to print the design shape and time a fit for 2025-03-12; they compute
no forecast or score on the holdout. I computed nothing on the holdout: my scripts cut the inputs first.

**L0 reproduction.** The diff 0459b68..eb5ea6b of lear.py adds `short_rule`, `slope`, `share_gram` and `noise_variance`
with defaults None, 1.0, False and None. With the defaults a short window returns None as before, slope 1 goes through
`asinh_forward` itself, and `LassoLarsIC(..., noise_variance=None)` is scikit-learn's default. transforms.py only gains
functions. I extracted the source tree of 0459b68 and ran its `lear_forecast_days` with defaults on the truncated inputs:
the stored L0 is reproduced on all eight days where it can be computed, to 2.8e-14, window by window. L0 minus honest_v2
gives -0.3383 [-0.9201, +0.2109], the figure logged for "LEAR alone" on 2026-10-16 (-0.338 [-0.920, +0.211]), and the S1
slices equal the logged ones (14.08, 18.04, 16.19, 14.53, 20.06). The existing tests cover the default path
(tests/test_sprint.py, and a bit-for-bit test against the old code written out in tests/test_lear_recipe.py). I ran
tests/test_lear_recipe.py, tests/test_yardstick.py and tests/test_sprint.py only: 56 passed. The full suite was not run.

## Robustness of the C verdict

**Definition.** The slice is the one the protocol has always used (`backtest.evaluate` through `sprint.metrics_for`,
docs/experiments.md "top 5% price hours by actual price"): strict hours whose actual price is at or above the 95th
percentile of the actual price over the strict rows. Cut 133.14 EUR/MWh, 1,038 hours on 216 days. It is the same set as in
the sprint, where S1 scored 20.06 on it, which I reproduce. I built it independently from inputs.csv and get L0 20.909 and
C 22.276, a difference of +1.366 against the limit of 1.0. The pre-registered definition is the one applied.

**Other ways of drawing the slice.**

| Slice | Hours | A minus L0 | B minus L0 | C minus L0 |
|---|---|---|---|---|
| Actual price, top 5% of the strict window (registered) | 1,038 | +2.92 | +2.04 | +1.37 |
| Top 5% within each calendar year | 1,039 | +2.60 | +1.81 | +1.32 |
| Top 5% within each side of 2025-10-01 | 1,038 | +3.06 | +2.11 | +1.38 |
| Top 5% within each calendar month | 1,053 | +4.93 | +3.10 | +2.46 |
| Top 1% | 208 | +6.85 | +4.54 | +4.06 |
| Top 2.5% | 519 | +4.37 | +2.87 | +2.08 |
| Top 10% | 2,076 | +1.82 | +1.27 | +0.75 |
| Top 20% | 4,152 | +1.71 | +1.18 | +0.95 |
| Top 5% of hours by L0's own forecast (known in advance) | 1,038 | +0.65 | +0.09 | +0.47 |

The fail holds for every definition by actual price at 5% or tighter, and it grows as the slice narrows. It would not
hold at 10% or 20%, or if the hours were picked by the forecast instead of the outcome. Neither of those is the
registered slice.

**Concentration.** C is worse than L0 on 122 of the 216 days in the slice and better on 94; 58% of the slice hours are
worse and the median hourly difference is +1.37. The total excess is 1,418 EUR/MWh-hours against the 1,038 that the limit
allows. The three worst days (2026-06-24, 2026-03-09, 2026-06-23) carry 477 of it. Without the worst day the difference
is 1.16, without two 1.03, without three 0.93. So three days of 216 decide the margin.

**Time.** Before 2025-10-01 (732 slice hours) C minus L0 on the slice is -0.04. From 2025-10-01 (306 hours) it is +4.73.
By year: 2024 -0.26 (222 hours), 2025 +0.26 (539), 2026 +4.82 (277). January, March, May and June 2026 hold 95% of the net
excess. B shows the same pattern (+0.88 before, +4.82 from).

**Interval.** A moving-block bootstrap over days of the slice difference (ratio of day sums, blocks of 7, 5000 resamples)
gives +1.37 with a 95% interval of [+0.39, +2.48]. 27% of resamples are at or below 1.0 (38% with blocks of 14, 45% with
28) and 0.1% at or below zero. For comparison B: [+1.28, +2.96], 0.4% at or below 1.0. A: [+1.71, +4.43].

**Reading.** C is worse than L0 on the most expensive hours; that much is established. Whether it is worse by more than
1.0 is not established: the rule compares a point estimate with a limit and the point estimate is over it, so C fails as
registered, but the data are compatible with a true difference under the limit. B's and A's fails on this slice are
secure. Conditions (i) and (iii) for C are not sensitive: the one-sided upper end stays between -0.19 and -0.16 over
twenty other seeds and is negative for blocks of 1 to 28 days.

## Other findings ranked by severity

1. **Look-ahead on 2024-02-17 in all LEAR-type forecasts (medium, pre-existing).** Described above. It does not move a
   verdict. It does mean that "no look-ahead" cannot be stated without qualification, for this experiment or for the logged
   LEAR and S1 figures, and that the scrambling test should always include the first strict day and the clock-change days
   by name.

2. **The gain of every candidate reverses from 2025-10-01 (medium, interpretation).** The builder's figures are right:
   C 16.378 against L0 16.093 from that day, 14.133 against 15.029 before. Paired daily differences against L0, mean and
   two-sided 95%:

   | | Before 2025-10-01 (592 days) | From 2025-10-01 (273 days) |
   |---|---|---|
   | A | -0.379 [-0.895, +0.115] | +0.846 [+0.267, +1.445] |
   | B | -0.547 [-0.882, -0.211] | +0.379 [-0.030, +0.801] |
   | C | -0.895 [-1.296, -0.494] | +0.285 [-0.213, +0.794] |

   A is significantly worse in the later part; B and C are worse without the interval excluding zero. By half-year, L0
   against C: 2024 H1 15.45 and 13.46, 2024 H2 14.83 and 14.62, 2025 H1 15.35 and 14.23, 2025 H2 13.44 and 13.22, 2026 H1
   17.82 and 18.36. Of C's total gain 59% comes from 2024 H1 and 45% from 2025 H1; 2026 H1 takes 21% away. C beats L0 in
   15 of 20 months before October 2025 and in 4 of 9 from then. The loss sits in January to April 2026, not in the last
   quarter of 2025, so the quarter-hour auction date is a convenient split, not a demonstrated cause. A significant paired
   difference over the whole window is therefore an average of a large early gain and a recent loss, and it is no guide to
   what C would do next.

3. **C is worse than L0 on more than the top 5% (medium, interpretation).** C minus L0 by band of the actual price: below
   zero -4.05 (1,264 hours), 0 to 20 -2.52 (3,955), 20 to 50 -1.47 (4,114), 50 to 80 +0.68 (4,167), 80 to 133 +0.97 (6,214),
   above 133 +1.37 (1,038). C is worse in every band above 50 EUR/MWh, which is 55% of the strict hours. The mean signed
   error (actual minus forecast) is -1.85 for L0, -0.86 for B and -0.67 for C. So most of what the candidates do is remove
   part of L0's tendency to forecast too high: they win where the price turns out low and lose where it turns out high.
   Sorting by the outcome exaggerates such a pattern, so this is a description, not a new test, but "worse on the most
   expensive hours" understates it.

4. **The short windows cause the loss at the top, the slope does not (low).** Top 5% MAE by window at slope 1: 31.2 (56),
   30.3 (84), 24.2 (364), 22.3 (728), 20.1 (1092), 19.7 (1456). The short windows fit very small models (2 to 14 non-zero
   coefficients of 302, median about 6, on the three days I looked at) and under-forecast the spikes (mean signed error on
   the slice +15.9 for L0, +20.0 for B, +18.5 for C). Every fixed slope below 1 lowers the six-window top 5% MAE a little
   (22.95 at slope 1, 21.66 at 0.1). An exploratory figure, computed after seeing the results and therefore not evidence:
   the four long windows at C's chosen slopes give 15.06 overall and are 0.46 better than L0 on the top 5% hours. That is a
   hypothesis for a new pre-registration scored on forward data, nothing more. The selection window has now been used for
   one more decision.

5. **The rolling choice adds little over a fixed slope (low).** C minus the six-window average at a fixed slope of 0.3 is
   -0.046 [-0.121, +0.031]. C minus B is -0.268 [-0.406, -0.137] overall, -0.348 before 2025-10-01 and -0.094
   [-0.328, +0.129] from it. The fixed-slope figures are hindsight; the point is only that the per-hour 56-day selection
   is not what earns the gain.

6. **B sits on two edges (low).** Its gain is 0.2548 against a bar of 0.25 (0.2542 without 2024-02-17). Its one-sided
   upper end is +0.036; over twenty other seeds it stays between +0.024 and +0.044, but with blocks of 1 or 3 days it would
   be below zero. B fails (ii) by a wide and secure margin, so nothing turns on this.

7. **Numerical fragility of the 364-day window (low, pre-existing).** The rejected Gram speed-up moved forecasts by 5 to 6
   EUR/MWh on some days from rounding-level changes, all in the 364-day window (364 rows for 302 regressors). The lasso
   refit also stops at its 2,500-iteration limit for some hours of that window: 2 of 24 at slope 1 on 2024-07-30, and 7 of
   24 at slope 0.1 on two of the three days I checked. The stored forecasts are deterministic and I reproduce them bit for
   bit on this machine, but an exact match on another machine or library build is not guaranteed. The 364-day window is
   also the weakest long window (16.20).

8. **Process (low).** See departure 9: headline before the stop-condition test finished; first chunks before the commit.
   Both are closed by later evidence.

9. **A is not quite the published window set in 2024 (low).** See departure 7.

10. **Not checked.** honest_v2 was not refitted; its forecasts are the builder's, and they reproduce the logged 15.7040
    and the logged slice figures. LEAR forecasts were refitted for 10 days (9 computable) of 921 and LEAR-AR for 3; the
    builder's scramble covers 20 more. Whether inputs.csv holds values as they stood at their publication time cannot be
    tested from the file. The full test suite was not run.

## What should be said to the owner

- The experiment was run as pre-registered and every number checks out. No candidate passes the rule. LEAR stays as it
  is and the result is logged as a null.
- A, the published window set alone, is no better than L0 (15.37 against 15.36) and fails all three conditions. B gains
  0.25 and fails on the top 5% hours (+2.04) and on the interval. C gains 0.52 with an interval that excludes zero and fails
  on the top 5% hours (+1.37 against a limit of 1.0).
- C's fail is a fail by the rule, and the rule should be honoured. It is not a strong statement about C: three days decide
  the margin and about a quarter of bootstrap resamples fall under the limit. Equally, C's pass on the average is not a
  strong statement in its favour: the gain was made in 2024 H1 and 2025 H1, and from October 2025, and in 2026 above all,
  C is worse than L0 overall and by nearly 5 EUR/MWh on the expensive hours. The honest summary is that the short windows
  and the flatter transform pull LEAR's forecasts down, which helped while L0 forecast too high on cheap hours and hurt
  once prices rose in 2026.
- One real look-ahead was found, on a single day. On 2024-02-17 the LEAR forecasts of 17 scored hours read weather for
  hours 0 to 6 that was not available at the gate. It is in L0, in the candidates, and in the LEAR 15.36 and S1 14.36
  logged in October. Removing the day changes no verdict (L0 15.3717, S1 14.3687, C's gain 0.5222, top 5% +1.37). It should
  be recorded in docs/experiments.md, the strict set for LEAR-type models should start on 2024-02-18, and the scrambling
  test should name the first strict day and the clock-change days instead of relying on a random draw. The builder's
  scrambling test itself passes on its 20 sampled days.
- The pre-registration's stop condition speaks of look-ahead "found by the scrambling test". The builder's test did not
  find this; mine did, and the builder's own scramble pointed at that day confirms it. The owner should decide whether to
  treat it as a stop followed by a rerun on 864 days, or as a logged correction. The arithmetic on 864 days is already in
  this folder (exclude_first_day.json) and the outcome is the same.
- Nothing from the holdout was used.
- The four-long-windows-with-slope figure in finding 4 must not be quoted as a result. If it is to be tested, it needs its
  own pre-registration and forward data.

## Files in this folder

- recompute.py, recompute.json: every figure of the table above, from inputs.csv, the chunks and the prediction files.
- robustness.py, robustness.json: the top 5% slice under other definitions, its concentration, bootstrap intervals, the
  split by period.
- truncate_refit.py, truncate_old.json, truncate_new.json, truncate_new_extra.json and logs: the truncation test.
- extra_checks.py, extra_checks.json: LEAR-AR refits, the 2024-02-17 sensitivity, L0 minus honest_v2.
- scramble_first_day.py, scramble_first_day.json: the builder's scramble applied to 2024-02-17 and to a control day.
- exclude_first_day.py, exclude_first_day.json: the rule on 864 days.
- alt_choice.py, alt_choice.json: C under the other reading of the 56-day mean.
- rowcounts.py, rowcounts.json: complete training rows per window.
- short_window_check.py, short_window_check.json: size and convergence of the short-window fits.
- pytest_targeted.log: the three test files run.
- recompute.py writes an intermediate frame (my_frame.pkl) that the other scripts read. It holds hourly prices and
  generation taken from inputs.csv, so I deleted it after the run; run recompute.py first to recreate it.
