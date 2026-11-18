# Pre-registration: the risk band, an hourly predictive interval for the day-ahead price

Written on 2026-11-18, before any fit or score of this work. Branch `forward-q`. Rank 4 of the build list. It has its own
version history and does not touch the frozen point candidate of `forward_q_prereg.md`.

## 1. What the band is

For each hour of delivery day D, an interval that should contain the auction price with a stated probability, issued at
the same time as the point forecast. Two nominal levels, 80% (the 10th to the 90th percentile) and 95% (the 2.5th to the
97.5th). It is a statement about the next price. It is not a confidence interval around a historical error.

Every method below uses only forecast errors that were settled when the forecast was issued. At 10:05 Paris on D-1 the
last settled delivery day is D-1, whose auction cleared on D-2. So for D a method may use the errors of days up to D-1
and nothing later. Update schedule: every day, by the same rule, with no tuning during the run.

The point forecast is S1 as stored (`work/lear-recipe/frame.csv`), with honest_v2 as stored run through the same two
methods as a secondary table, because honest_v2 is the deployed model. No point forecast is refitted.

## 2. Step A, reproduce what exists

The quantile models of `forecast/probabilistic.py` (q10, q50, q90 by LightGBM, with a conformal margin on a trailing 90
days) are rerun on the selection window by `walk_forward_prob` and must reproduce the logged row of 2026-10-11 on its own
set (2024-02-17 to 2026-06-30, 20,752 strict hours): mean pinball loss 4.881, coverage of the 10 to 90 interval 80.2%,
mean width 49.75, and for its benchmark (a) 5.105, 78.7% and 48.35. If they do not reproduce to the printed digits, the
difference is reported and the step stops there.

The holdout figure on record (73.8%, with 56% on windy days and on holidays) is quoted from the log and not recomputed.
The holdout is not used.

The live record is described as well: the q10 to q90 band the laptop has stored each morning since 2026-10-13, against
the auction results in the laptop's store, coverage and width only. These are few days and they are described, not
tested.

## 3. Step B, the simple baseline

**B, rolling empirical error quantiles.** For day D and clock hour h, take the errors (price minus point forecast) of the
same clock hour on the previous 182 delivery days. The interval is the point forecast plus their 2.5%, 10%, 90% and 97.5%
empirical quantiles (linear interpolation). The median for the pinball score is the point forecast plus their 50%
quantile. At least 120 past days are needed.

## 4. Step C, one challenger, built on the thesis tail code

**T, scaled errors with generalised Pareto tails.** The code is the spliced distribution of the MSc thesis (normal body,
generalised Pareto tails, free tail masses) and its maximum-likelihood tail fit, copied from `de-power-caps` into
`forecast/tail_model.py` with its tests.

1. Scale. Hours are grouped in six blocks of four clock hours. For day D and block b, the location m is the median and
   the scale s is 1.4826 times the median absolute deviation of the errors of that block over the previous 28 days.
2. Standardise. Each past error is divided by the scale of its own day and block, after subtracting that day's location.
   Those scales were themselves computed from earlier days only.
3. Pool. The standardised errors of all hours over the previous 364 days are pooled, at least 120 days.
4. Fit. Splice points at the empirical 10% and 90% quantiles of the pool, tail masses 0.10 each, a generalised Pareto
   distribution fitted to the exceedances beyond each, body mean and standard deviation from the values between.
5. Issue. The quantile of the fitted distribution at 2.5%, 10%, 50%, 90% and 97.5%, multiplied by today's scale and
   shifted by today's location, added to the point forecast.

What T adds to B is a scale that follows the last four weeks, which is aimed at the failure on record (a calibration
window that lagged a change of regime), and a fitted tail for the 95% band, where 182 values per hour give B only four
or five observations beyond each end.

Not in this registration: conditioning the tail on system tightness, quantile regression averaging, a joint distribution
across hours. Daily baseload, spreads and battery values need dependence across hours, which an hourly band does not
give. Averaging hourly band ends does not give a band for the daily mean. That is a later, separate piece of work.

## 5. Scoring

Scored set: strict hours of delivery days 2024-08-18 to 2026-06-30, the days on which B and T both have their minimum
history, the same hours for every method. The existing quantile models are scored on the same hours at the 80% level.

For each method and level: empirical coverage, mean width, and the interval score (width plus 2/alpha times the distance
by which the price falls outside), which punishes both missing and being wide. Also the pinball loss at each quantile,
the mean over 10%, 50% and 90% (comparable with the logged 4.881), and the count of prices above and below the band.

Reported by slice: the five protocol slices, negative-price hours, half-years, the six hour blocks, and the tenth of days
with the widest price range. Each with its number of hours.

Paired inference for T against B: the daily mean interval score at 80%, T minus B, moving-block bootstrap of whole days,
block 7, 5000 resamples, seed 0, one-sided 95%.

## 6. Rule

T replaces B as the working band if all three hold.

1. The interval score at 80% is lower for T and the upper end of the one-sided 95% interval of the paired daily
   difference is below zero.
2. Coverage at 80% lies between 77% and 83% overall, and on the windiest tenth of days and on holidays T covers no more
   than three points less than B.
3. Coverage at 95% lies between 93% and 97%.

Otherwise B stays. Either way nothing is published. The working band then needs a forward calibration record of its own
before it is shown to anyone as calibrated, registered separately once the method is fixed.

## 7. Limits stated in advance

The scored days are the selection window, used many times already. Coverage is measured, not guaranteed. Errors are
dependent in time and regimes change, so no formal coverage guarantee is claimed for any method. The stored point
forecasts were made with the archive load forecast, which is fresher than the one available at the gate, so the errors
behind these bands are a little smaller than live errors will be.
