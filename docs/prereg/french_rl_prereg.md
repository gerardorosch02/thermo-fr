# Pre-registration: French residual load in LEAR, honest and idealised, and what the neighbours add to it

Written on 2026-11-17, before any model of this test has been fitted. Branch `french-rl`, cut from `neighbour-rl`.

## Why
The neighbours test (docs/prereg/neighbour_rl_results.md) found that the neighbours' residual load with idealised timing
lowers LEAR's error by 0.80, and that their load forecast alone adds nothing. By its rule the next step was an honest
version for the neighbours. The reviewer's exploratory control suggested that most of that gain is fresher wind and solar
information that a French regressor also carries. LEAR holds French wind speed and radiation but no estimate of wind and
solar output, and the project already has honest ones (the generation proxies that honest_v2 uses). The owner decided on
2026-11-17 to test France first. This file therefore asks two questions:

1. Does LEAR improve with the honest French residual load? This candidate could be adopted.
2. With idealised timing, do the six neighbours add anything once the French residual load is in? This decides whether
   building wind and solar estimates for six zones is worth it.

## What has been seen before writing this
- The results of the neighbours test and its review, including the reviewer's exploratory control on 116 days and window
  364 (French idealised regressor -1.27, six neighbours -0.86, neighbours added to France -0.21 [-0.67, +0.23]).
- The yearly coverage of the French columns. The operator's wind and solar forecasts are present from 2021 (offshore from
  2023). The honest proxies begin in 2024: the solar proxy on 2024-04-01 and the wind proxy on 2024-05-01, with no gap
  afterwards inside the scored period.
- No LEAR model with any French residual load has been fitted by the builder.

## Series
All are same-hour values of delivery day D in Paris clock hours, laid out as the other LEAR blocks, each scaled by its own
median and MAD x 1.4826 over the training window (all 24 hours together) and then the plain asinh, as in the neighbours
test. A missing value takes the same clock hour of the most recent earlier day that has one, at most seven days back,
component by component (counts reported).

- Honest French residual load: `load_fc_mw - wind_proxy_mw - solar_proxy_mw`, the columns of inputs.csv as honest_v2 uses
  them. Exists from 2024-05-01.
- Idealised French residual load: `load_fc_mw - solar_fc_mw - wind_onshore_fc_mw - wind_offshore_fc_mw`, the operator's
  day-ahead forecasts. A missing offshore value counts as zero, as in the project's `extended` feature set.
- Neighbours' idealised residual load: the six series of the neighbours test, unchanged.

## Timing
- The load forecast and the two generation proxies pass the gate and the 10:05 issue time under the project's rules
  (`load_forecast`, `wind_proxy`, `solar_proxy`). The code asserts it.
- The operators' wind and solar forecasts fail the gate (18:00 deadline). Every candidate that reads them on day D is
  idealised, labelled so in the code, and can never be adopted or published.
- In training rows of days before D the operator's forecast for those days is long public. Candidate Q uses it there only,
  as a training stand-in, in the same way that LEAR's weather vectors use archive weather before issued forecasts exist.

## Models
Baseline L0: the present LEAR (windows 364, 728, 1092, 1456, daily recalibration, slope 1). Every candidate is L0 with
added same-hour regressors in each hourly model.

Honest candidates, all four windows, averaged as L0 is:
- **P**: one regressor, the honest French residual load. On training days before it exists the scaled regressor is set
  to zero (its training median), so the same training days are used as in L0. If fewer than 28 training days in the window
  have it, or D itself lacks it, the regressor is left out and the forecast is L0's.
- **Q**: one regressor, the honest French residual load from 2024-05-01 and the idealised French residual load on
  training days before that. It never reads the operator's forecast for D or for any day from 2024-05-01. If D lacks the
  honest value the forecast is L0's.

Idealised references, window 728 only (the best single window of the neighbours test, chosen to limit computing):
- **F**: one regressor, the idealised French residual load.
- **FI**: seven regressors, F plus the six neighbours.
- **M**: F restricted to the days on which the honest series exists and handled exactly as P (zero before, 28-day
  minimum). It separates the two handicaps of P: M against P is the quality of the information, F against M is the short
  history.

honest_v2 is not changed. S1-P and S1-Q, the equal averages of honest_v2 with each honest candidate, are diagnostics.
honest_v2 already reads the proxies, so a gain in LEAR need not carry through to S1.

## Scored rows
- Honest candidates: strict rows of delivery days 2024-05-29 to 2026-06-30 (763 days expected), the days on which both P
  and Q can have a forecast of their own. The figure on all 864 days of the project's reference set (candidate equal to L0
  before 2024-05-29) is reported beside it.
- Idealised references: strict rows of delivery days 2024-02-18 to 2026-06-30 (864 days), against the stored window-728
  forecast of L0 and of I.
- The holdout (from 2026-07-01) is not used. Reference figures on the 864 days: honest_v2 15.7124, L0 15.3717, S1 14.3687.
  The window has been used for selection many times; a pass is a reason to test forward and not evidence of a live gain.

## Rules, written before any run
Paired daily MAE differences use the project's moving-block bootstrap (block 7, 5000 resamples, seed 0). There are three
decisions (P, Q, and the neighbours), so one-sided intervals are at 98.33% (Bonferroni). Protocol slices: weekends,
holidays, windiest 10%, sunniest 10%, top 5% hours, computed on the rows of each comparison.

- **P and Q.** A candidate passes if (i) its MAE is at least 0.25 EUR/MWh below L0's, (ii) no protocol slice is worse than
  L0 by more than 1.0, and (iii) the upper end of the one-sided 98.33% interval of the paired daily difference against L0
  is below zero. If both pass, the one with the lower MAE is taken. A pass makes it the LEAR for a forward live log of at
  least 90 days, pre-registered separately. No public change follows from this file.
- **Neighbours.** FI against F on window 728. The neighbours are worth building only if FI's MAE is at least 0.25 below
  F's and the upper end of the one-sided 98.33% interval of the paired daily difference is below zero, and only if P or Q
  passes (an honest route has to work for France, where the data are best, before six more zones are built). In every other
  case the neighbours' residual load is parked for LEAR at this gate.
- If neither P nor Q passes, the result is logged as a null and the next item of the build list is the probabilistic
  layer.
- F, FI and M are never adopted. They are reported against the stored window-728 forecasts of L0 and I, and P's
  window-728 forecast against M and F, as diagnostics.

## Reported whatever the outcome
For L0, P and Q on the 763 days: MAE, the protocol slices, negative-price hours, MAE before and from 2025-10-01, MAE by
half-year, MAE by band of the actual price, the under-reaction slope, rank correlation and spread capture, per-window MAE,
and the share of hourly fits in which the lasso kept the regressor. For window 728 on the 864 days: MAE of L0, I, F, FI
and M and the paired differences F against L0, FI against F, FI against I. On the 763 days: P, M and F at window 728 and
the share of F's gain that P captures. S1-P and S1-Q against S1. Counts of replaced values and of forecasts that fell back
to L0.

## Checks and stop conditions
- Reproduction. On the 864 days the stored forecasts must give honest_v2 15.7124, L0 15.3717 and S1 14.3687.
- Same code path. The fitting code with no added regressor, and with a regressor that is left out by the 28-day rule, must
  reproduce the stored L0 window forecasts to 1e-9 EUR/MWh on 2024-02-18, 2024-05-10, 2024-10-27, 2025-03-30 and
  2026-06-24.
- Look-ahead. For 20 delivery days (the first day of the honest set 2024-05-29, the first scored day 2024-02-18, the
  clock-change days 2024-10-27, 2025-03-30, 2025-10-26 and 2026-03-29, and 14 drawn with seed 4), every input dated D or
  later that is not known at the gate for D is replaced by noise, the neighbours' values of later days included; every
  stored forecast for D must not change by more than 1e-9. For P and Q the operator's wind and solar forecast of D is
  scrambled too. For F, FI and M it is treated as known, which is the declared idealisation.
- A failure of any of the three stops the run before a result is read.
- The code is committed before the first fit. An independent reviewer recomputes the figures and checks the rule before
  anything is reported.
