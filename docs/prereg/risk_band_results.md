# Results: the risk band, baseline and one challenger

Pre-registration: `docs/prereg/risk_band_prereg.md` (sha256 468d0224a98cb8f6643c179d351f3a38a1ae0700528c0efff3fa4de91263437a,
commit f67d7c5, written before any fit or score). Code committed at e34647d before the first run. Run on 2026-11-18 on
branch `forward-q`. Independent review: `docs/prereg/forward_q_review.md`. Machine-readable figures:
`docs/prereg/risk_band_results.json`.

The band is an hourly predictive interval for the auction price at 80% and 95%. Every figure below is on the selection
window. The holdout was not used. No point forecast was refitted.

## Outcome

- **The baseline stays.** By the rule, the challenger built on the thesis tail code does not replace the rolling
  empirical band, because its interval score is not lower. Nothing is published.
- **The honest reading is closer than the rule makes it sound.** The baseline is the sharpest of the three bands and
  wins on the score that weighs width against misses. It also covers too little, 78.0% for a nominal 80% and 93.3% for a
  nominal 95%, and it fell to 72.7% in the first half of 2026 when prices moved. The challenger held 78.7% there and is
  within a point of nominal at both levels, at the price of a band about a tenth wider.
- **No band is calibrated where it matters most.** On holidays, the windiest days, the highest prices and the days with
  the widest price range, all three cover roughly 67% to 74% for a nominal 80%.

## Step A. What exists reproduces

| | Mean pinball loss (10, 50, 90%) | Coverage of the 80% band | Mean width |
|---|---|---|---|
| Logged on 2026-10-11, quantile models with the conformal step | 4.881 | 80.2% | 49.75 |
| Recomputed from the stored predictions of that run (20,752 strict hours) | 4.881 | 80.18% | 49.75 |
| Logged, its benchmark (point forecast plus trailing-year error quantiles) | 5.105 | 78.7% | 48.35 |
| Recomputed from the stored predictions | 5.105 | 78.72% | 48.35 |

The three quantile models were also refitted month by month. Their forecasts match the stored ones on all 20,759 rows,
largest difference 8.5e-14. Without the conformal step the raw 10 to 90 band covers 58.75% at a width of 32.79, as
logged. The weak slices are as logged too (holidays 72.2%, windiest tenth of days 71.3%, top 5% of hours 70.7%).

The holdout figure on record (73.8% overall, about 56% on the windiest days and on holidays) is quoted from the log and
was not recomputed.

The live record, described only. Over 25 delivery days from 2026-10-13 to 2026-11-19 (600 hours), the band the laptop
stored each morning covered 88.2% of auction prices at a mean width of 108.9 EUR/MWh, with 7.2% of prices below it and
4.7% above. Prices in those weeks averaged 196 EUR/MWh, more than three times the selection window, and the band is more
than twice as wide as in the backtest. Twenty-five days cannot settle anything, but live it is erring on the wide side,
the opposite of the holdout.

## Steps B and C. The two bands around S1

Scored set: strict hours of delivery days 2024-08-18 to 2026-06-30, 16,368 hours on 682 days, the same hours for every
method.

| Band around S1 | Coverage | Mean width | Interval score | Prices below / above |
|---|---|---|---|---|
| **80% level** | | | | |
| B, rolling empirical quantiles (182 days, same clock hour) | 78.0% | 43.5 | 68.0 | 1,694 / 1,905 |
| T, scaled errors with generalised Pareto tails | 79.9% | 47.4 | 69.7 | 1,603 / 1,682 |
| Existing quantile models with the conformal step | 79.9% | 48.7 | 70.3 | |
| **95% level** | | | | |
| B | 93.3% | 69.7 | 103.7 | 507 / 583 |
| T | 94.5% | 79.1 | 106.4 | 437 / 459 |

The interval score is the width plus a penalty for every miss, so lower is better. Mean pinball loss over the 10%, 50%
and 90% quantiles: B 4.665, T 4.748, existing models 4.839.

Paired daily interval score at 80%, T minus B: +1.78, one-sided 95% interval from +0.55 to +2.96, T better on 42% of
days. At 95%: +2.62, from -1.03 to +5.80.

The rule, condition by condition.

| Condition | Result |
|---|---|
| 1. Interval score at 80% lower for T, upper end of the one-sided interval below zero | Fails. T is higher by 1.78 and the whole interval is above zero |
| 2. Coverage at 80% between 77% and 83%, and no more than three points below B on windy days and holidays | Holds (79.9%; windy days 71.7% against 70.7%; holidays 73.3% against 67.9%) |
| 3. Coverage at 95% between 93% and 97% | Holds (94.5%) |

B stays as the working band.

## Where each band is weak

Coverage of the 80% band by slice, with the number of hours.

| Slice | Hours | B | T | Existing |
|---|---|---|---|---|
| Holidays | 480 | 67.9% | 73.3% | 70.0% |
| Windiest tenth of days | 1,656 | 70.7% | 71.7% | 71.9% |
| Sunniest tenth of days | 1,656 | 77.0% | 83.4% | 78.8% |
| Top 5% of prices | 819 | 67.4% | 70.3% | 68.6% |
| Negative-price hours | 975 | 79.6% | 80.1% | 82.8% |
| Widest price range, tenth of days | 1,656 | 66.5% | 72.3% | 74.4% |
| 2024 H2 | 3,265 | 80.1% | 81.5% | 77.7% |
| 2025 H1 | 4,343 | 78.3% | 78.3% | 81.9% |
| 2025 H2 | 4,417 | 81.4% | 81.5% | 80.9% |
| 2026 H1 | 4,343 | 72.7% | 78.7% | 78.7% |

By block of four clock hours all three stay between 77.6% and 80.9%. At 95%, B covers 87.5% on holidays, 83.6% of the
top 5% of prices and 90.4% in 2026 H1, against 92.5%, 86.9% and 93.0% for T.

## The same two bands around honest_v2, the deployed model

| | Coverage 80% | Width | Interval score | Coverage 95% | Width | Interval score |
|---|---|---|---|---|---|---|
| B | 78.6% | 47.7 | 74.6 | 93.2% | 76.5 | 114.6 |
| T | 80.5% | 51.3 | 76.0 | 94.7% | 86.0 | 116.8 |

T minus B at 80%: +1.40, from -0.05 to +2.89. The same picture, with every band about a tenth wider than around S1
because the point forecast is worse.

## What this does and does not show

- The simple baseline is hard to beat on sharpness. A band from the last half-year of the model's own errors, hour by
  hour, is narrower than both alternatives and loses little coverage in ordinary conditions.
- Its weakness is the one on record. It adapts slowly, and it under-covers when the regime moves (2026 H1) and in the
  tails. The challenger's four-week scale repairs most of that but widens the band everywhere, and the rule, written
  in advance, counts width against it.
- The fitted tail earns little. At 95% the challenger is better calibrated but not better scored.
- Nothing here is a coverage guarantee. The days are the selection window, used many times. The errors were made with the
  archive load forecast, which is fresher than the one available at the gate, so live errors will be somewhat larger
  and a band built from backtest errors will start too narrow.
- An hourly band says nothing about the daily base price, a spread or a battery schedule. Those need the hours
  together.

## Departures from the pre-registration

- Step A said the quantile models would be rerun and must reproduce the logged row, and that the step stops if they do not.
  The rerun's aggregate does not match to the printed digits and I did not stop, for the reason that follows. The
  reviewer lists this as a caution. The logged row reproduces exactly from
  the stored predictions of the original run. The rerun reproduces every quantile forecast to 8.5e-14 but started on
  2024-01-01, without the earlier forecasts the original run had behind it, so its conformal margin differs in the first
  quarter of 2024 and its trailing-year benchmark through 2024. On its own aggregate the rerun gives 4.880, 80.04% and
  49.60. From 2025-02-01 the two are identical. All of this is before the scored set of steps B and C or does not
  enter it: the existing models' band is the same in both from 2024 Q2.
- The 28-day scale of T needs at least 20 of its 28 days. The text gave no minimum.
- T's body mean and standard deviation are the plain mean and standard deviation of the values between the splice
  points. They affect only the median, never the band ends, which sit on the empirical deciles and in the fitted tails.

## Not done

Conditioning the tail on system tightness. Quantile regression averaging. Any joint distribution across hours. A forward
calibration record, which the working band needs before anyone is told it is calibrated.
