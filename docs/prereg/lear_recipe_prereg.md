# Pre-registration: the yardstick (rank 0) and LEAR to the published recipe (rank 1)

Written on 2026-11-16 by the designer, before any number in this file's scope was computed. Source of the ideas: the review
"French power price forecasting methods" (2026-11-16). Nothing here changes the public model.

## Status of the evidence
The selection window (strict rows, delivery days 2024-02-17 to 2026-06-30, 865 days, walk-forward from 2024-01-01) has now
been used for many decisions, so a pass here is a reason to start a forward live log of at least 90 days, not a proof. The
holdout (2026-07-01 to 2026-10-10) is worn and is not used. The live shadow of S2 was stopped on 2026-11-16 without a verdict.

## Part 1. The yardstick (measurement only, no decision)
Models: honest_v2 (monthly walk-forward, backtest settings, must reproduce 15.7040), LEAR as it is (L0, daily, windows 364,
728, 1092, 1456, must reproduce 15.36), S1 (their equal average, must reproduce 14.36). S2 is retired: the log shows 14.32
against 14.36 and a LEAR weight between 0.45 and 0.58.

- M1 Relative error. MAE of each model divided by the MAE of three naive forecasts on the same strict rows: the same hour of
  D-1; the same hour of D-7; and the benchmark naive of Lago et al. (D-1 for delivery days Tuesday to Friday, D-7 for Saturday,
  Sunday and Monday).
- M2 Under-reaction slope. Ordinary least squares, with an intercept, of (actual minus the same hour of D-1) on (forecast
  minus the same hour of D-1): once on hourly rows and once on daily means. A slope above one means the forecast moves less
  than the price does. Reported with the correlation.
- M3 Members against S1: paired daily MAE difference with the project's moving-block bootstrap (block 7, 5000 resamples,
  seed 0), two-sided 95%.
- M4 Shape. Per day: the Spearman rank correlation between the forecast and the actual 24 hourly prices; and the spread
  capture, (mean actual price in the forecast's four highest hours minus mean actual price in its four lowest hours) summed
  over days, divided by the same quantity with the hours chosen from the actual prices. The same two numbers for the D-1
  profile used as a forecast.
- M5 The split at 2025-10-01 (quarter-hour auction): MAE and M1 before and from that day.
- M6 A no-feature baseline, LEAR-AR: the L0 machinery with only the four French price-lag blocks and the calendar dummies.
- M7 Weather effect by tightness. Days from 2024-05-01 with complete data. Tightness of day D is the maximum over its hours
  of (load forecast minus wind proxy minus solar proxy) minus the mean D-1 nuclear output over the hours ending by 08:00
  Paris, in GW; terciles over those days. Within each tercile and pooled: the regression of S1's daily mean error (actual
  minus forecast) on the daily mean wind surprise and solar surprise (actual minus proxy, GW), with an intercept; reported
  are the coefficients in EUR/MWh per GW and the share of variance explained. The same at hourly resolution with the
  hourly error. The question is whether the slope and the share are larger on tight days. No decision follows from M7 here;
  the review's rule is that the weather-ensemble route stays parked unless the share in the tightest tercile is well above
  the pooled figure.

## Part 2. LEAR to the published recipe (decision)
Baseline: L0. Three candidates, all with daily recalibration and the L0 regressors, scaling and penalty rule unless stated.

- A, the published window set: 56, 84, 1092 and 1456 days, averaged.
- B, six windows: 56, 84, 364, 728, 1092 and 1456 days, averaged.
- C, B with a rolling choice of the asinh slope. After the median and MAD scaling the project already uses (MAD x 1.4826;
  the source paper uses the raw MAD, a stated difference), the transform of a scaled value x is
  sign(x) [asinh(|x| + a) - asinh(a)] with a = sqrt(1/c^2 - 1); c = 1 is the present transform. It applies to the target and
  to every price block (the four French lag blocks, CH and DE-LU); the other blocks keep c = 1. Grid c in {0.1, 0.3, 0.5,
  0.7, 1.0} (the paper's grid thinned for computing time, fixed before any run). For each c the six-window average is
  computed for every day from 2023-12-23. For delivery day D and hour h, C uses the c with the lowest mean absolute error
  for hour h over the delivery days D-56 to D-1 (all settled before the gate); ties go to the larger c.
  Source: Uniejewski, arXiv 2511.13603, equation 2 and the rolling selection over 56 days per hour.

Short windows. With about 300 regressors the 56 and 84 day windows have fewer rows than regressors, where L0's penalty rule
(AIC with the noise variance estimated from a full least-squares fit) is undefined. For windows with rows <= regressors + 1
the AIC is computed with the noise variance set to the variance of the transformed target over the training rows, which is
what scikit-learn's LassoLarsIC did before version 1.1 and therefore what the published benchmark ran on (to be confirmed
by the builder from the scikit-learn changelog; if that reading is wrong the fallback is reported as this project's own
choice, unchanged). Long windows keep L0's rule, so L0 is reproduced exactly.

Speed. The builder may share one precomputed Gram matrix across the 24 hourly fits of a window if the forecasts of L0 on 20
sampled days agree with the present code to 1e-6 EUR/MWh.

Rule, written before any run. A candidate passes if, on the strict selection rows, (i) its MAE is at least 0.25 EUR/MWh
below L0's, (ii) no protocol slice (weekends, holidays, windiest 10%, sunniest 10%, top 5% hours) is worse than L0 by more
than 1.0, and (iii) the upper end of the one-sided 98.33% moving-block bootstrap interval (Bonferroni for three candidates;
block 7, 5000 resamples, seed 0) of the paired daily MAE difference against L0 is below zero. The passing candidate with
the lowest MAE becomes LEAR-R. If none passes, LEAR stays as it is and the result is logged as a null.

Reported for every candidate whatever the outcome: MAE by tercile of recent volatility (the standard deviation of the seven
daily mean prices D-7 to D-1), before and from 2025-10-01, on negative-price hours, the per-window MAE, the share of days
each c is chosen (C), and M1, M2 and M4 of Part 1.

Consequence of a pass. S1-R, the equal average of honest_v2 and LEAR-R, is reported against S1 and honest_v2 as a
diagnostic. No public change follows from this file: the next step would be a forward live log of at least 90 days with a
significance test, pre-registered separately.

Stop conditions: any look-ahead found by the scrambling test (inputs from D onward scrambled must not change the forecast
for D), or a failure to reproduce 15.7040, 15.36 or 14.36.
