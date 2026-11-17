# Independent review: neighbours' load and residual load in LEAR

Reviewed on 2026-11-17 against `docs/prereg/neighbour_rl_prereg.md` (commit 68c9f1c, sha256 ae9898ed...fe1e, unchanged since), the code of commit 8b56891 and the outputs in `/home/claude/work/neighbour-rl/`. All figures below come from the reviewer's own scripts in this folder, reading the chunk files and the baseline frame. The builder's report functions were not called.

## Verdict

**Confirmed.** Every figure in `results.json` that I recomputed (96 of them) agrees with my own computation to better than 1e-13 EUR/MWh, the pre-registered rule was applied as written, and I obtain the same verdicts. H fails (gain 0.0095, conditions (i) and (iii) not met). I meets the ceiling (gain 0.8046, one-sided 97.5% upper end -0.568) and by construction cannot pass or be adopted. I found no look-ahead and no use of the holdout. No number and no verdict needs correcting. There are three minor departures from the text, none of which changes a verdict, and one caution that matters for how the result is worded. An exploratory control (not pre-registered, 116 sampled days, one window) suggests that most of I's gain is also obtained from the French operator's own post-gate wind and solar forecast, so the result supports "a wind and solar forecast issued after the gate carries about 0.8 EUR/MWh" more firmly than it supports "the neighbours' wind and solar matter".

## Figures recomputed

Scored rows as I count them from the baseline frame are 864 delivery days, 2024-02-18 to 2026-06-30, 20,735 strict hours (859 days of 24 hours, 3 of 23, 2 of 25), no non-strict row in the period. Each of the eight chunk tables has 864 days and no missing cell. The window average is the plain mean of the four windows. My hourly H and I equal the builder's `frame.csv` to 6e-14 on every row.

| Figure | Builder | Reviewer | Match |
|---|---|---|---|
| MAE honest_v2 (reference) | 15.7124 | 15.7124 | yes |
| MAE L0 (reference) | 15.3717 | 15.3717 | yes |
| MAE S1 (reference) | 14.3687 | 14.3687 | yes |
| MAE H | 15.3622 | 15.3622 | yes |
| Gain H against L0 | 0.0095 | 0.0095 | yes |
| H paired daily difference | -0.0096 | -0.0096 | yes |
| H interval, 95% two-sided | [-0.1090, 0.0899] | [-0.1090, 0.0899] | yes |
| H one-sided 97.5% upper end | 0.0899 | 0.0899 | yes |
| H share of days better | 0.4734 | 0.4734 | yes |
| MAE I | 14.5671 | 14.5671 | yes |
| Gain I against L0 | 0.8046 | 0.8046 | yes |
| I paired daily difference | -0.8048 | -0.8048 | yes |
| I interval, 95% two-sided | [-1.0575, -0.5680] | [-1.0575, -0.5680] | yes |
| I one-sided 97.5% upper end | -0.5680 | -0.5680 | yes |
| I share of days better | 0.5972 | 0.5972 | yes |
| Slice differences against L0, H (weekends, holidays, windiest, sunniest, top 5%) | 0.016, -0.400, -0.274, -0.135, 0.125 | same | yes |
| Slice differences against L0, I | -1.222, -0.896, -1.611, -0.573, -1.915 | same | yes |
| Slice MAE of L0 (rounded) | 15.39, 19.40, 17.56, 15.57, 20.93 | same | yes |
| Negative-price hours (1,264), L0 / H / I | 21.508 / 21.403 / 20.432 | same | yes |
| MAE before 2025-10-01 (14,183 h), L0 / H / I | 15.0386 / 15.0244 / 14.1743 | same | yes |
| MAE from 2025-10-01 (6,552 h), L0 / H / I | 16.0928 / 16.0933 / 15.4173 | same | yes |
| Price below 0, L0 / H / I | 21.508 / 21.403 / 20.432 | same | yes |
| Price 0 to 20 | 18.672 / 18.742 / 17.385 | same | yes |
| Price 20 to 50 | 16.485 / 16.568 / 15.789 | same | yes |
| Price 50 to 80 | 11.309 / 11.283 / 11.089 | same | yes |
| Price 80 to 133 | 13.079 / 12.966 / 12.357 | same | yes |
| Price above 133 | 20.818 / 20.941 / 18.928 | same | yes |
| Per-window MAE, L0 (364, 728, 1092, 1456) | 16.2019, 15.7565, 15.7574, 15.8285 | same | yes |
| Per-window MAE, H | 16.1547, 15.7572, 15.7675, 15.8357 | same | yes |
| Per-window MAE, I | 15.3845, 14.9561, 15.1186, 15.0787 | same | yes |
| MAE S1-H | 14.3720 | 14.3720 | yes |
| S1-H paired difference against S1 | 0.0032 [-0.0489, 0.0593] | same | yes |
| MAE S1-I | 13.7860 | 13.7860 | yes |
| S1-I paired difference against S1 | -0.5827 [-0.7103, -0.4627] | same | yes |
| Under-reaction slope hourly, L0 / H / I | 0.8893 / 0.8863 / 0.8999 | same | yes |
| Mean rank correlation, L0 / H / I | 0.8778 / 0.8752 / 0.8831 | same | yes |
| Spread capture, L0 / H / I | 0.9189 / 0.9170 / 0.9227 | same | yes |
| Kept shares, all scored days and 60-day sample (12 + 12 values) | see results.json | same | yes |
| Filled values, H (DE-LU) | 50 | 50 | yes |
| Filled values, I (DE-LU, BE, NL, ES, IT-North, CH) | 56, 0, 72, 528, 504, 6 | same | yes |
| Fallbacks to L0 on scored hours, H / I | 0 / 0 | 0 / 0 | yes |

The full list of 96 compared figures is in `r9_table.md`. The largest difference is 7e-15.

About the bootstrap. My implementation is written separately (a loop over resamples, overlapping blocks of 7, ceil(864/7) blocks cut to 864 days) but with seed 0 it happens to draw the same block starts as the project's function, so the interval ends agree to 1e-15 and that agreement is not independent evidence. With seeds 1 to 5 the ends move by about 0.01. For I the low end lies between -1.071 and -1.057 and the high end between -0.570 and -0.562. For H the high end lies between 0.0887 and 0.0914. With other block lengths the I interval is [-1.037, -0.584] (block 1), [-1.085, -0.561] (14), [-1.146, -0.554] (28), [-1.199, -0.602] (56), and a stationary bootstrap with mean block 7 gives [-1.056, -0.542]. The lag-1 autocorrelation of the daily difference is 0.12. No reasonable variant brings I's upper end near zero or H's below zero.

The paired daily mean (-0.8048) differs from the hourly gain (0.8046) only because days of 23 and 25 hours weigh the same as others in the daily mean.

## Rule check

- Bonferroni. Two candidates, one-sided level 97.5%. The code takes the upper end of the two-sided 95% percentile interval, which is the 97.5% quantile of the resampled means. Correct.
- H, condition (i). Gain 0.0095 against a threshold of 0.25. Not met.
- H, condition (ii). The worst slice is the top 5% of hours, +0.12 (rounded as the project does, +0.125 unrounded). No slice is worse by more than 1.0. Met.
- H, condition (iii). Upper end +0.090. Not met.
- H verdict. Fails. I obtain the same.
- I, condition (i). Gain 0.8046. Met.
- I, condition (iii). Upper end -0.568. Met.
- I verdict. "Ceiling met", the first of the three pre-registered outcomes. Its slices are all better than L0 (-0.57 to -1.92) and do not bind, as written.
- I cannot pass. `apply_rule` gives an idealised candidate no `passes` field, only one of three ceiling verdicts, and a unit test covers it. `results.json` has `"idealised": true` and no `passes` for I.
- "If neither moves" does not apply.
- The reproduction stop condition is coded and was met. I confirm 15.7124, 15.3717 and 14.3687 from the baseline frame, that S1 is the equal average of honest_v2 and L0 (6e-14) and that L0 is the mean of its four stored windows (6e-14).

## Look-ahead and timing

(a) Fitting path. `fit_predict_day` trains on design rows from D-364 (or the window length) to D-1 and predicts from the design row of D and the neighbours' row of D. The training targets are prices of days up to D-1. The neighbours' scaling (median and MAD) uses training rows only. The design row of D holds price lags of D-1 and earlier, the load forecast and as-issued weather of D, neighbour prices of D-1, and the early nuclear output of D-1. I see no path by which a value dated after D, or any price of D, reaches the forecast.

(b) Fill rule. `_filled` is a forward fill down the day axis with a limit of 7, column by clock hour. It can only copy from an earlier day. My own explicit look-back implementation gives identical tables for all six zones and both candidates over all 2,007 days (largest difference 0.0, same missing cells) and the same counts. The longest look-back actually used is 5 days (Spain). The hour-2 interpolation of the day layout fired only on the six spring clock-change days.

(c) Independent scramble. Five delivery days the builder did not use (2024-02-19, 2024-04-01, 2025-01-01, 2025-10-27, 2026-06-29), window 364, my own noise. Replacing every neighbour value dated after D changed neither H nor I by more than 2.8e-14. Replacing in addition every input after D, every input of D not known at the gate (prices of D included) and the D-1 nuclear output from 08:00 gave the same result, at most 2.8e-14. The test has power. On two days, scrambling the neighbours' load of D itself moved H by 50 to 110 EUR/MWh, and scrambling the neighbours' wind and solar of D itself moved I by 226 to 505 and left H unchanged (4e-15). Refits from scratch of five day-window pairs (H 728 on 2024-08-15, I 1456 on 2025-12-25, H 1092 on 2026-05-10, I 364 on 2024-02-18, I 728 on 2025-05-02) reproduce the stored chunks to 7e-15. The fit function with no added block reproduces the stored L0 window on two further days (2024-06-15 window 364, 2025-08-20 window 1456) to 1.4e-14. The builder's 11 unit tests pass.

(d) Whole-day vectors. The first day on which all 24 hours have as-issued weather is 2024-02-18. The only partly covered day is 2024-02-17 (17 of 24 hours, the missing ones are Paris hours 0 to 6). None of the 864 scored days has an hour without as-issued weather, so no whole-day weather vector of a scored day contains proxy values. The weather of D-1 does not enter the design row of D. The 2024-02-17 fault is therefore not repeated on 2024-02-18 or elsewhere. Proxy weather remains in training rows before 2024-02-18, as in L0, and for the first scored day the seven proxy hours of 2024-02-17 are early morning hours known before the 10:05 issue time.

(e) Timing claims. `assert_gate` checks rules, not data. It assigns the French load forecast's rule (10:00 Paris on D-1) to the six neighbours' load columns and verifies that this passes the gate and the 10:05 issue time, and for I it verifies that the wind and solar rule (18:00 on D-1) fails the gate. I called it on the scored index and it behaves as described. It cannot verify when any value was actually published because the archive carries no publication time. H's "honest" label is fair in the same sense as the French load forecast already used by L0, with the same caveat that the archive holds the latest version. That caveat can only flatter H, and H shows no gain, so the null for H is safe. For I the caveat is part of the declared idealisation.

## Data handling

- Filled values and fallbacks match `results.json` exactly (table above). Inside the scored period the days with any gap are ES 20, IT-North 5, DE-LU 2, CH 2, BE 0, NL 0, as the pre-registration states. Only one scored hour has a filled load value (DE-LU). 27 scored days have a filled value in some zone, and I's gain on them (0.93) is close to its gain on the other days (0.80).
- Days lost to training. One day, 2021-01-01, in every zone, because the file starts at 00:00 UTC and so lacks the first Paris hour. L0 cannot use that day either.
- Offshore. ES, IT-North and CH have an empty offshore column and count zero. DE-LU, BE and NL have full series from 2021. Handled as written.
- Level breaks. Confirmed from yearly means. The Swiss solar forecast averages 30 to 37 MW in 2021 to 2023 and 650 to 700 MW from 2024. The Dutch solar forecast falls from 1,342 MW (2023) to 810 MW (2024). Not mentioned in the pre-registration, the Dutch wind forecasts also fall by about a quarter from 2024 (onshore 1,900 to 1,427 MW, offshore 828 to 640 MW).
- Could the breaks manufacture I's gain? I find no sign of it. With window 364, every scored day from 2025-01-03 trains only on days after the breaks, and I's gain there is 0.91 against 0.65 before. The half-year in which every window straddles the breaks most recently, 2024 H1, is the only one where I loses. The two affected zones are kept less often by the lasso (NL 71%, CH 70%) than BE, DE-LU and ES (92% to 100%). If anything the breaks cost I something early on.

Exploratory, not a result. I's gain against L0 by half-year and window, in EUR/MWh:

| Period | Hours | Average | w364 | w728 | w1092 | w1456 | H average |
|---|---|---|---|---|---|---|---|
| 2024 H1 (from 02-18) | 3,215 | -0.18 | -0.48 | -0.25 | -0.18 | 0.02 | -0.05 |
| 2024 H2 | 4,417 | 1.33 | 1.38 | 1.32 | 1.06 | 0.97 | 0.03 |
| 2025 H1 | 4,343 | 1.27 | 1.24 | 1.26 | 1.12 | 1.28 | 0.10 |
| 2025 H2 | 4,417 | 0.67 | 0.87 | 0.65 | 0.44 | 0.75 | -0.17 |
| 2026 H1 | 4,343 | 0.68 | 0.74 | 0.76 | 0.53 | 0.54 | 0.13 |

By year the gain is 0.70 (2024), 0.96 (2025) and 0.68 (2026). It is positive in 23 of 29 months. Over the whole period it is present in each window on its own (0.82, 0.80, 0.64, 0.75). H is within 0.05 of L0 in every window.

## Departures from the pre-registration

None changes a verdict.

1. "The download stops on 2026-06-30." The file stops at 23:00 UTC on 2026-06-30, so its last two rows are Paris hours 0 and 1 of delivery day 2026-07-01, the first holdout day (42 non-missing forecast values, no prices). `load_neighbours` removes them before any use, the design ends on 2026-06-30, and no chunk, frame row or scored row is later than 2026-06-30. The holdout was not used. The download should be cut two hours earlier next time so the statement is literally true.
2. Look-ahead test for H. The text says every input of D not known at the gate is replaced by noise. The builder's `scramble_neighbours` leaves all the neighbours' columns of D alone for both candidates, so for H the wind and solar of D were not scrambled. H never reads them, and my own test confirms that scrambling them leaves H unchanged, so this is a gap in the test and not in the model.
3. Level breaks. The text lists two. The Dutch wind forecasts show a third of similar size (see above).

Points checked and found in line with the text. Six added regressors on 302 for 308 in all, each window keeping L0's penalty rule. Same-hour values only. Each zone scaled on the training window over all 24 hours, then plain asinh. Components filled one by one before the subtraction, which the text allows. Slices computed on the scored rows with the project's definitions. Kept shares on 60 days drawn with seed 3. The 20 scramble days (6 named, 14 drawn with seed 2) and the 5 same-path days. The inputs file itself runs to 2026-10-11, and every entry point of the module cuts it at 2026-07-01 before anything is built.

Order of events from commit and file times. Neighbour file written 11:09, pre-registration committed 11:11:10, same-path check 11:16:06, first chunk 11:17:25, code committed 11:19:23, last chunk 15:45, scramble check 15:57:34, report 15:57:54. The run started two minutes before the code was committed, and the committed code reproduces the stored chunks, including the first day of the first chunk. The two stop-condition checks precede the report.

## Interpretation and cautions

Breadth of I's gain. I is better than L0 on 516 of 864 days (59.7%). The median daily difference is -0.56 and the mean -0.80. The largest daily improvements are -19.2, -14.3, -13.4, -12.3 and -11.1, the largest deteriorations +14.0, +8.5, +8.4, +8.3 and +8.1. The ten best days carry 17.6% of the net gain. Without them the paired mean is -0.671 with interval [-0.905, -0.446], and the hourly gain is 0.671, still above 0.25. Without the best 5% of days (43) the mean is -0.365. So the gain is broad with a heavy favourable tail, not the work of a few days. By hour it is about zero from midnight to 03:00, 0.8 to 1.3 from 08:00 to 15:00 and 1.2 to 1.5 from 18:00 to 23:00. It is larger at weekends (1.22) and on the windiest days (1.61). It is uneven in time (table above), negative in 2024 H1.

Is the gain more than fresher weather information? This is the caution that matters. L0 sees French weather only as issued two days ahead and has no wind or solar generation estimate. I adds operators' day-ahead wind and solar forecasts that may be published up to 18:00 on D-1, and weather systems are shared across the region. As a control I gave L0 a single extra same-hour regressor, the French residual load built from the French operator's own day-ahead wind and solar forecast (the same idealisation applied to France), on 116 scored days (4 per month, my seed 11), window 364 only, with the builder's fit function. This is exploratory, not pre-registered, on a small sample, and must not be used for selection.

| Window 364, 116 sampled days | MAE | Paired daily difference [95%, plain bootstrap over days] |
|---|---|---|
| L0 | 15.18 | |
| I (six neighbours, stored) | 14.32 | -0.86 [-1.41, -0.30] against L0 |
| F (France only, one regressor) | 13.90 | -1.27 [-1.84, -0.71] against L0 |
| F and I together | 13.69 | -1.49 [-2.21, -0.77] against L0 |
| Neighbours added to F | | -0.21 [-0.67, +0.23] |
| France added to I | | -0.63 [-1.02, -0.25] |

One French regressor does at least as well as six neighbours' regressors, and once it is in, the neighbours add about 0.2 with an interval that includes zero. On this evidence roughly three quarters of I's gain is information that the French post-gate forecast also carries. A second, weaker sign points the same way. I's gain is 0.47, 0.77 and 1.46 across terciles of the day's gap between the French operator's wind forecast and the project's honest wind proxy.

Zones. In I the lasso keeps BE in 99.6% of hourly fits, DE-LU 93%, ES 92%, NL 71%, CH 70% and IT-North 61%. Kept shares are a weak guide. In H the zones are kept 58% to 73% of the time and the forecast does not improve at all, and the six series are collinear. I did not run leave-one-zone-out refits, so I cannot say which zones carry the gain.

What should temper "the neighbours' wind and solar matter, their load does not".
- The pre-registered finding is narrower. With idealised timing, the neighbours' residual load lowers LEAR's error by 0.80 (5.2%), and their load forecast alone adds nothing to what L0 already has.
- H and I differ in two ways at once, the wind and solar content and the timing. The experiment cannot separate them, and the control above suggests timing and shared weather explain most of the difference.
- "Their load does not matter" overstates it. L0 already holds the French load forecast, calendar terms and yesterday's Swiss and German prices, so the neighbours' load forecast has little left to add. That is a statement about incremental information.
- I is a ceiling. An honest version built from weather forecasts issued before the gate will recover only part of 0.80, and the part specific to the neighbours may be near 0.2. Before building estimates for six zones, the cheaper test is the French one, since the project already has honest French wind and solar proxies and L0 does not use them. I would pre-register a control of that kind alongside the next step.
- The selection window has been used many times, as the pre-registration says. The S1-I diagnostic (-0.58 against S1) is not a decision figure.

## Not checked

- The download itself (ENTSO-E document and process types, the averaging to hourly means) and the actual publication time of any forecast in any zone.
- The source paper's figures (14.67 to 13.59).
- Whether any model was fitted before the pre-registration commit. File times are consistent with the text (data written two minutes before the commit, first output five minutes after) but cannot prove it.
- honest_v2 and L0 themselves beyond the reference figures, the builder's five same-path days and my two.
- The stored chunks beyond 15 refitted day-window forecasts. My scramble used window 364 only, the builder's covers all four windows on his 20 days.
- The relative MAE against naive forecasts and the volatility terciles in `results.json` (not pre-registered items).
- Which zones carry I's gain (no leave-one-out refit).
- The French control beyond one window and 116 days.

## Files in this folder

`r1_figures.py` (headline figures), `r2_data.py` (file range, coverage, own fill, level breaks), `r3_lookahead.py` (refits, scrambles, power), `r4_interpret.py` (half-years, concentration, kept shares, weather flags), `r5_french_control.py` and `r5_summary.py` (exploratory control), `r6_gaps.py`, `r7_extras.py`, `r8_l0_path.py`, `r9_compare.py`, their `.json` outputs and `r9_table.md`. Files starting with `_` are logs and the sampled control forecasts. Two intermediate tables that held copies of the hourly frame and of the neighbours' day tables (`_frame.pkl`, `_own_tables.pkl`) were deleted after use so that no raw data sits in this folder. `r1_figures.py` and `r2_data.py` write them again, and the later scripts need them.
