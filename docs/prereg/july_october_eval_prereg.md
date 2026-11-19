# One development evaluation on 1 July to 10 October 2026: frozen plan

Written on 2026-11-19 at about 11:40 London, before any model was fitted on a day after 2026-06-30 by this work and before
any error on such a day was computed for the LEAR, the ensemble or Q. **Nothing in this plan runs until the owner approves it
in his own words.** His approval, with its time, is logged in `docs/experiments.md` before the first fit.

## What this is, and what it is not

The project's rule says the holdout from 2026-07-01 is never used to choose. This plan asks the owner to use the period
**once**, as development evidence. It is not a holdout test and will not be described as one, for three reasons.

- The prices of the period are known to the project.
- The tree model has been scored on it (honest_v2 25.81, honest 26.23, row of 2026-10-10), and several candidates were each
  evaluated on it once.
- The question about the LEAR at high prices was formed in November 2026 from 18 live days, after the period.

After this run the period is spent for every model named here. A variant designed after reading the results gets a new name
and is judged only on days it has not seen.

Amendment 3 of the forward test is unchanged: Q and P stay unscored on every live day.

## Period and inputs

- Delivery days 2026-07-01 to 2026-10-10 inclusive, strict hours only. Expected 2,448 hours on 102 days.
- Inputs: `data/forecast/inputs.csv`, sha256 `c06e285f2e757dd1de04b2f9ca7f6b982621db249f160ea850b793a1d7b0dd9f`, the frozen research file. It is **not** corrected: the German
  price lag is the EXAA blend and radiation is on the legacy labels. The corrected rebuild is separate work.
- The file holds no wind or solar proxy from 2026-08-20 to 2026-10-09. Every model reads one history, with the proxies
  filled by the rule of amendment 1 (`forward_q.fill_proxy_gaps`: values in the file are never replaced).
- Every forecast of day D uses only what `forward_q.as_known_q` and the project's timing rules allow at 10:05 Paris on D-1.

## Models

| Name | What it is |
|---|---|
| T300 | Tree model, honest_v2 inputs, 300 trees of 31 leaves (`GBM_PARAMS` with `REFIT_PARAMS`), refitted at the start of each calendar month on all strict rows from 2021-01-01 before that month |
| T800 | The same at 800 trees of 63 leaves (`GBM_PARAMS`) |
| L0 | The present LEAR, four windows of 364, 728, 1092 and 1456 days, refitted every day, through `forward_q.q_live` |
| Q | L0 with the French residual load regressor, through `forward_q.q_live` |
| S1 | (T300 + L0) / 2 |
| S1-Q | (T300 + Q) / 2 |

L0's four per-window forecasts are kept. P is produced by the same call and is not scored.

## Three comparisons

1. **T800 against T300.** Is the larger tree better on these days?
2. **S1 against T300.** Does the ensemble beat the tree model it would replace?
3. **S1-Q against S1.** Does the residual load step add anything here?

For each: the two errors, the paired daily difference with a two-sided 95% interval (moving-block bootstrap, block 7, 5,000
resamples, seed 0), the one-sided upper end at level 1 - 0.05/3, the share of days better, the five protocol slices, and the
result with and without 1 to 10 October (the crash).

Reading rule, written now. A comparison **supports** the candidate if the one-sided upper end is below zero, **contradicts**
it if the mirrored lower end is above zero, and is **undecided** otherwise. No release follows from this run alone.

## The distance diagnostic

**Definition, frozen.** For delivery day D and each of L0's four windows w, take the 24 prices of D-1 and put them on the
scale that window was fitted on: asinh((p - m_w) / s_w), where (m_w, s_w) is the target scaler of the fit of window w for
day D (`lear.fit_scalers(...)["target"]`, from the training days ending D-1). The **distance** of day D is the mean of those
values over the 24 hours and the four windows. It is known at 10:05 Paris on D-1.

**Bands, frozen from the live look.** Low: below 1.0. Middle: 1.0 to below 1.6. High: 1.6 or more. For comparison with the
live days the plain split is also reported: yesterday's mean price at 200 EUR/MWh or more.

**A spell** is a maximal run of consecutive delivery days in the high band.

**Reported per band.** Days and spells. The realised change of the daily mean from D-1. L0's predicted change and T300's.
L0's error minus T300's, paired, with its interval. S1's error minus T300's. Then everything again without the longest
high spell.

**What would count as the live pattern recurring**, written now. In the high band L0 predicts a change more than 5 EUR/MWh
below the realised one **and** L0's error is above T300's, and both still hold without the longest spell. The verdict is one
of: recurs, does not recur, or too few days (fewer than 5 high days, or a single spell).

## Stop conditions, checked before any new figure is read

1. T800 on this history gives the logged 25.81 on the period within 0.10. That figure is already known, so reading it
   reveals nothing. If it does not reproduce, the run stops and the reason is reported first.
2. The code of this run reproduces stored forecasts on the selection window to 1e-9: T300 and T800 for June 2026, L0 for
   five sampled June days.
3. On three days of the period, every input not known at the gate is replaced by noise and the forecasts do not move.
   Forecasts are compared with forecasts. No price is read.
4. The period has 2,448 strict hours on 102 days.

## Review

An independent reviewer that has not seen the build recomputes every reported figure from the forecast files before
anything is reported.
