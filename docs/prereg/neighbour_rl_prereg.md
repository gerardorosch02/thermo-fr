# Pre-registration: neighbours' day-ahead load and residual load in LEAR

Written on 2026-11-17, before any model has been fitted on the new data. Branch `neighbour-rl`, cut from `lear-recipe`.

## Why
The review of 2026-11-16 ranked the neighbours' day-ahead residual load third. France clears in one coupled auction with
its neighbours, so the price of an hour depends on how much demand is left for thermal plant across the coupled area, not
only in France. The present models see the neighbours only through yesterday's prices. Source for the size of the effect:
Chomon and Ziel, working paper of June 2026 (arXiv 2606.07014, not peer reviewed), where adding the day-ahead residual load
of nearby zones to a per-hour linear model cut the French MAE from 14.67 to 13.59.

## What has been seen before writing this
Only the coverage and the yearly means of the downloaded forecasts. No price was read against them and no model was run.
- Source: ENTSO-E Transparency Platform, day-ahead total load forecast (A65, A01) and day-ahead wind and solar forecast
  (A69, A01) for DE-LU, BE, NL, ES, IT-North and CH, 2021-01-01 to 2026-06-30, hourly means.
- Coverage: load forecast 99.9% to 100% of hours in every zone; solar and onshore wind 99.25% to 100%. No offshore series
  for ES, IT-North and CH. Days with any gap inside the scored period: ES 20, IT-North 5, DE-LU 2, CH 2, BE 0, NL 0.
- Level breaks visible in the yearly means, left as published: the Swiss solar forecast is near zero before 2024, and the
  Dutch solar forecast drops by about 40% from 2024.

## Data rules
- Residual load of a zone = load forecast minus solar forecast minus onshore wind forecast minus offshore wind forecast.
  A missing offshore series counts as zero.
- A missing value is replaced by the value of the same clock hour on the most recent earlier day that has one, at most
  seven days back. The count of replaced values is reported. A day still incomplete after that is left out of training,
  and a candidate that cannot forecast a scored hour takes L0's forecast for it (count reported).
- Hours are Paris clock hours, laid out day by day exactly as the other LEAR blocks (the repeated autumn hour averaged, the
  missing spring hour filled from its neighbours).
- The holdout (delivery days from 2026-07-01) is not used. The download stops on 2026-06-30.

## Timing
- Neighbours' load forecast. Same rule as the French load forecast: Regulation (EU) 543/2013, Article 6(1)(b), published no
  later than two hours before the day-ahead gate closure, taken as 10:00 Paris on D-1. It passes the project's gate and the
  10:05 issue time. As for France, the archive holds the latest version, so a later update cannot be ruled out.
- Neighbours' wind and solar forecast. Article 14(1)(d) sets the deadline at 18:00 Brussels time on D-1, after the gate, and
  the archive gives no publication time. By the project's rule it fails the gate. Candidate I below uses it as if it were
  known at the gate. That is an idealised test, labelled as such in the code, and I can never be adopted or published.

## Models
Baseline L0: the present LEAR (windows 364, 728, 1092 and 1456 days, daily recalibration, slope 1), unchanged.

Both candidates are L0 with one added regressor per zone in each hourly model: the model for hour h receives the zone's
value for hour h of delivery day D, for the six zones (six added regressors, about 308 in all, so every window keeps L0's
penalty rule). Each zone's series is scaled by its own median and MAD x 1.4826 over the training window (all 24 hours
together) and then takes the plain asinh, as every non-price block does. Same-hour values follow the source paper; the
project's 24-hour vectors are not used for the new blocks because 144 added regressors would exceed the rows of the
364-day window.

- H, honest timing: the neighbours' load forecast.
- I, idealised timing: the neighbours' residual load.

honest_v2 is not changed. S1-H and S1-I, the equal averages of honest_v2 with each candidate, are reported against S1 as
diagnostics only.

## Scored rows
Strict rows of delivery days 2024-02-18 to 2026-06-30 (864 days, 20,735 hours). The first strict day of earlier tests,
2024-02-17, is excluded for every model after the correction logged on 2026-11-17. Reference figures on these rows:
honest_v2 15.7124, L0 15.3717, S1 14.3687. The window has been used for selection several times; a pass here is a reason
to test forward and not evidence of a live gain.

## Rules, written before any run
Paired daily MAE differences use the project's moving-block bootstrap (block 7, 5000 resamples, seed 0). There are two
candidates, so one-sided intervals are at 97.5% (Bonferroni). Protocol slices are computed on the scored rows: weekends,
holidays, windiest 10%, sunniest 10%, top 5% hours.

- H passes if (i) its MAE is at least 0.25 EUR/MWh below L0's, (ii) no protocol slice is worse than L0 by more than 1.0,
  and (iii) the upper end of the one-sided 97.5% interval of the paired daily difference against L0 is below zero. A pass
  makes H the LEAR for a forward live log of at least 90 days, pre-registered separately. No public change follows from
  this file.
- I is a ceiling test and cannot be adopted. If it meets (i) and (iii), the next step is to build the project's own wind and
  solar estimates for the neighbours from weather forecasts issued before the gate, and to test that honest version under
  its own pre-registration. If it does not meet (i), the neighbours' residual load is parked for LEAR at this gate. If it
  meets (i) and not (iii), it is logged as inconclusive and parked unless the owner decides otherwise. Its slices are
  reported and do not bind.
- If neither moves, the result is logged as a null.

## Reported whatever the outcome
For L0, H and I: MAE, the protocol slices, negative-price hours, MAE before and from 2025-10-01, MAE by band of the actual
price (below 0, 0 to 20, 20 to 50, 50 to 80, 80 to 133, above 133), the under-reaction slope, the rank correlation and the
spread capture of the yardstick, the per-window MAE, and for each zone the share of hourly fits in which the lasso kept its
regressor (on a fixed sample of 60 days, seed 3). For S1-H and S1-I: MAE and the paired daily difference against S1.

## Checks and stop conditions
- Reproduction. On the scored rows the stored forecasts must give honest_v2 15.7124, L0 15.3717 and S1 14.3687.
- Same code path. The new fitting code with no added block must reproduce the stored L0 forecast to 1e-9 EUR/MWh on five
  days (2024-02-18, 2024-10-27, 2025-03-30, 2025-10-26, 2026-06-24).
- Look-ahead. For 20 delivery days (the first scored day 2024-02-18, the clock-change days 2024-03-31, 2024-10-27,
  2025-03-30, 2025-10-26 and 2026-03-29, and 14 drawn with seed 2), every input dated D or later that is not known at the
  gate for D is replaced by noise, the neighbours' columns of later days included; the forecasts for D must not change by
  more than 1e-9. For I the neighbours' wind and solar forecast of D itself is treated as known, which is the declared
  idealisation.
- A failure of any of the three stops the run before a result is read.
- An independent reviewer recomputes the figures and checks the rule before anything is reported.
