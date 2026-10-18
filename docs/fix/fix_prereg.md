# Pre-registration: the radiation label fix (F), step 2 of panel/final_plan.md

2026-10-17, quant designer. Written before any price result for F exists.

## Amendment 1, 2026-10-17, written before any price result for F had been computed or seen

**Builder's step 1 finding.** The builder found that ENTSO-E actual wind and nuclear generation are about one hour late over the same period as solar (builder_notes.md section 11). The audit compared monthly lags inside September 2023 to March 2024 with those outside:

| Series | Reference | Lag inside minus outside (outside sd) |
|---|---|---|
| Wind | ENTSO-E wind forecast | +1.08 h (0.57) |
| Wind | Open-Meteo wind speed cubed | +0.71 h (0.84) |
| Nuclear | day-ahead price | +0.79 h (0.18) |
| Nuclear | load forecast | +0.93 h (0.46) |
| Solar | Open-Meteo radiation | +0.96 h (0.08) |

The reference series against each other stay flat: -0.18, +0.04 and +0.03 h. All the generation series come from the same ENTSO-E A75 documents.

**Decision.** F relabels every ENTSO-E actual generation series the project uses from those documents (solar, wind onshore, wind offshore, nuclear) on one flagged-day set.
- The set is the days the solar physical rule of amendment 2 flags: 215 days, namely 2023-07-20 to 07-24, 2023-08-28, 2023-09-03, and 2023-09-06 to 2024-03-31.
- Nuclear has no daily shape to test, so it cannot be flagged by its own physical rule. Shared documents, period and direction justify a single set.
- Correcting solar and leaving its sibling series wrong would knowingly keep a label error inside a labelling correction.

The cross-checks above are recorded in the log row and are not used to choose days.

**Relabel rule.** On flagged days, the value labelled h takes the value originally labelled h+1.
- The last hour of each flagged block has no same-period value and is set to missing. For solar it is night and set to 0.
- The original first value of each block duplicates the previous hour and is dropped.
- Missing hours are counted. A daily or early-hours mean is taken over the hours present, as the code does today.

**What changes through it.**
- The nuclear lag features (`nuclear_d2_mw`, `nuclear_d2_mean_mw`, `nuclear_d1_early_mw`) and `residual_v2_mw`, in training rows and on strict days up to 2024-04-02.
- The wind proxy, recalibrated by its unchanged monthly rule on the relabelled target. Its as-issued points start on 2024-02-17, so calibrations from May 2024 to March 2025 contain the six relabelled weeks.

Slice day sets stay copied from the logged run.

**Timing gate, confirmed safe.** A late label means the value labelled h describes h-1. As logged, the early D-1 nuclear mean (labels 00:00 to 07:00 Paris) therefore held older data, ending at 07:00.
- After the relabel, each label holds its own period. The mean covers 00:00 to 08:00, exactly what the rule `nuclear_d1_issue` already assumes: public by 10:00 Paris, two hours after the period ends, against the regulation's one hour.
- The D-2 features are public by 02:00 on D-1.
- No value enters a feature later than its documented timing rule allows, and every rule still passes the 10:05 pre-market deadline.
- The builder adds a perturbation test: altering any generation value whose true period ends after 08:00 Paris on D-1 leaves the features for D unchanged.

**Hash.** The superseded file is not kept; its sha256 was not recorded before this amendment. The hash recorded for step 2 is this amended file's.

## 0. What has already been seen

**Proxy-level and labelling evidence only:**
- The hefty reviewer's in-sample NNLS: solar proxy MAE 1,014 to 646 MW with radiation labelled T+1.
- The independent reviewer's walk-forward proxy, using the repository's `rolling_proxy`, reproduced to 7e-12:
  - 989 to 637 MW overall, and 853 to 456 MW from 15:00 to 19:00 UTC;
  - after the shift, a remaining daytime bias of 0.8 to 1.1 GW;
  - ENTSO-E actual solar about one hour late in early 2024.
- The designer's and scout's labelling checks:
  - the daily radiation centroid sits +0.51 h after solar noon, and the actuals -0.42 h before it;
  - correlation peaks at radiation labelled T+1;
  - the reviewer's clear-sky integrals agree.

**No price result of F, or of any model with shifted inputs, has been computed or seen.**

## 1. The two decisions

1. **honest_v2-F against honest_v2.** The public model. Logged at 15.7040.
2. **S2-F against S2.** The shadow candidate. Logged at 14.3192.

These are separate decisions on different models, not a choice between candidates, so there is no multiplicity correction. Either can pass without the other.

## 2. What F changes, and nothing else

**1. Radiation inputs relabelled to start-of-hour.**
- **What moves:** the 21 `solar_pt_*` columns, `radiation_fc_wm2` and `radiation_proxy_wm2` are Open-Meteo preceding-hour means. In each of them, row T takes the value labelled T+1.
  - The last Paris hour of each delivery day (23:00 to 24:00) is set to 0. It is always dark in France, so this is exact, and it needs only hours already in the day's window.
  - `radiation_proxy_wm2` is shifted only if the clear-sky labelling test (section 4) shows it is end-labelled; otherwise it is left as is and reported.
- **Missing values:** where a shifted value is missing on a strict row, the unshifted value of the same column at T is used (the only point-in-time value). These rows are listed and counted.
- **Timing:** the timing rule of a shifted radiation column becomes `weather_issued_issue(T + 1 h)`. Its latest run is 18Z on D-2, which passes the gate and the pre-market deadline. A test covers it.

**2. ENTSO-E actual solar relabelled by the physical rule** (final_plan.md amendment 2), applied before any calibration.
- **Per-day offset:**
  - the centroid of actual solar minus the centroid of clear-sky output, both on start-of-hour labels;
  - averaged over a centred 7-day window;
  - taken relative to the median offset over the whole history.
- **Relabel:** days more than 0.5 h above that median are moved back one hour.
- **Reported, without changing the rule:** the flagged days, the cross-check against ENTSO-E's day-ahead solar forecast, and RTE eCO2mix if fetched.

**3. Solar proxy recalibrated** on the shifted points and the relabelled actuals, by the unchanged monthly rule:
- non-negative least squares;
- trailing 365 days ending two days before the month;
- at least 60 days.

`residual_v2_mw` is recomputed from the new proxy with its formula unchanged.

**4. Downstream refits only.**
- LEAR-F is LEAR refitted daily with the shifted radiation vector.
- S2-F's online weights are recomputed from the F members' errors by the unchanged rule.

**2a. Wind and nuclear actuals relabelled** on the same flagged days (amendment 1), with the wind proxy and the nuclear lag features recomputed by their unchanged rules.

**Unchanged.** Every other input, including the wind and temperature forecasts, load, prices and neighbour prices, and every feature list, setting and window:
- **GBM:** 800 trees, 63 leaves, `random_state` 7, trained from 2021-01-01, refitted monthly.
- **LEAR:** windows of 364, 728, 1092 and 1456 days, the AIC penalty and the asinh scaling.
- **S2:** a 14-day half-life, with Tuesday to Saturday updates.
- **Bootstrap:** seed 0.

The wind and nuclear label checks of step 1 are reported. They do not change F.

## 3. Window, rows and rule

- **Window.** Strict rows 2024-02-17 to 2026-06-30, `--test-start 2024-01-01`, `--test-end 2026-07-01`. The holdout is not used.
- **Same rows.** The strict-row flag is taken from the unshifted columns, exactly as in the logged runs. The builder asserts the set equals the logged 20,752 hours (865 days).
- **Same slices.** The day sets for the five slices are copied from the logged honest_v2 run, not recomputed.
- **Bar, for each decision** (final_plan.md amendment 1):
  - the one-sided 95% moving-block upper bound of the paired daily MAE difference (F minus base; blocks of 7 days, 5,000 resamples) is at most +0.10 EUR/MWh;
  - no protocol slice is worse by more than 1.0;
  - the sunniest 10% of days and the 15:00 to 19:00 UTC hours are each not worse by more than 0.5 EUR/MWh, on the point estimate of the MAE difference.
- **Passing makes F eligible, nothing more.** Adoption follows only through the replay and the burn-in of step 3.

## 4. Order of operations

1. **Step 1 (builder), signed off by the reviewer before any run.**
   - The code on a branch, including the morning-run path.
   - The clear-sky labelling test on every shifted column, and on `solar_mw` before and after the relabel.
   - The relabel rule's flagged-day list.
   - The wind and nuclear label checks.
   - The timing test, and a perturbation test showing no value after τ enters.
2. **Hash.** The sha256 of this file is recorded.
3. **Reproduce the bases.** honest_v2 must give 15.7040 and S2 14.3192 on the branch, or the run stops.
4. **Run.** honest_v2-F, LEAR-F and S2-F, then the report.
5. **Freeze after review.** The reviewer reviews the run. This file, the code commit and the model-file rebuild rule (honest_v2-F refitted as the 2 November refit would have been; LEAR-F daily) are then frozen by sha256 in the log. That happens before anyone reads arm A's daily scores.
   - No agent queries the `shadow_*` score tables or reads `shadow-settle` output before day 40.
   - The owner is asked not to share them.

## 5. Reported, not decisive

For each base, its F and LEAR-F:
- MAE and RMSE overall; peak and off-peak; the five slices; by year; 15:00 to 19:00 UTC; the sunniest 10%; negative-price hours.
- The lag diagnostic: signed, slope and absolute.
- The paired daily difference: mean, two-sided 95% and one-sided 95% intervals, and the share of days won.
- Solar proxy MAE and hourly bias, before and after, walk-forward, overall and from 15:00 to 19:00 UTC.
- Counts:
  - relabelled actual days, and hours left missing per series;
  - rows that fell back to unshifted radiation;
  - whether `radiation_proxy_wm2` was shifted.
- Wind proxy MAE before and after, walk-forward.
- MAE on strict days 2024-02-17 to 2024-04-02, the only strict days whose nuclear lags change.

## 6. Log row (docs/experiments.md)

| 2026-10-?? | Pre-registration, branch radiation-fix (step 2 of the panel plan): a labelling correction, not a new input. Open-Meteo radiation (the 21 solar points, radiation_fc_wm2 and, if the clear-sky test shows it is end-labelled, radiation_proxy_wm2) is a preceding-hour mean labelled at the end of the hour, while ENTSO-E actuals and prices are labelled at the start, and nothing shifted either. F relabels the radiation to start-of-hour (row T takes the value labelled T+1; the last Paris hour of D set to 0), relabels ENTSO-E actual solar, wind and nuclear (same A75 documents) by one hour on the 215 days the pre-registered clear-sky centroid rule flags on solar (mainly 2023-09-06 to 2024-03-31; amendment of 2026-10-17, written before any price result; cross-checks inside minus outside: wind +1.08 h against the ENTSO-E wind forecast, nuclear +0.79 h against the day-ahead price and +0.93 h against the load forecast, reference series flat against each other), recalibrates the solar and wind proxies and recomputes the nuclear lag features on that basis by their unchanged rules, and changes nothing else; the timing gate is unchanged, since after the relabel each label holds its own period. Already seen: proxy-level evidence only (walk-forward proxy MAE 989 to 637 MW, 15 to 19 UTC 853 to 456 MW; centroid and clear-sky labelling checks by three panel members); no price result | radiation inputs relabelled; solar proxy and residual_v2 recomputed; LEAR and S2 weights refitted on them; nothing else | selection 2024-02-17 to 2026-06-30, the logged 20,752 strict hours and slice day sets, --test-start 2024-01-01, same training starts and seeds | rule written before any run | | | | | | not used (worn) | not applicable | two separate decisions, honest_v2-F against honest_v2 (15.7040) and S2-F against S2 (14.3192): each passes if the one-sided 95% moving-block upper bound of the paired daily MAE difference is at most +0.10 EUR/MWh, no slice is worse by more than 1.0, and the sunniest 10% of days and 15 to 19 UTC hours are each not worse by more than 0.5; a pass makes F eligible only, with adoption through the replay over arm A's 40 days from the saved morning windows and a 10-day live burn-in | written before running; the spec, code and model-file rule are frozen by hash after review and before anyone reads the shadow's daily scores, which no agent reads before day 40 |
