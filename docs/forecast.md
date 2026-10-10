# Day-ahead price forecast: method, information timing and backtest

Phase 1 of the hourly forecast of French day-ahead prices. Everything here uses only information available before the single day-ahead coupling (SDAC) closes at 12:00 Paris time on the day before delivery, which this document calls the gate for delivery day D. Where an input cannot be shown to meet the gate it is kept in a separate feature set and labelled.

## Information timing

The ENTSO-E API returns no publication time for past data. A historical query produces a fresh document whose `createdDateTime` is the moment the query ran (checked on 2026-10-05: every document carried that day's time), and `revisionNumber` only counts resubmissions (3 for the wind and solar forecast, 1 for the load forecast and prices). The timing rules below therefore come from the publication rules in Regulation (EU) 543/2013 and from the market calendar, not from the API. They are encoded in `forecast/timing.py` and a test fails if any feature for D is dated after the gate.

| Input | Rule | Issue time used | Relative to the 12:00 gate | Verified how |
|---|---|---|---|---|
| Day-ahead price of D-1 (and D-2, D-7) | SDAC results published shortly after the auction | 13:00 Paris on the auction day (D-2 for the D-1 price) | 23 hours before | Market calendar; `timing-probe` saw the price for D present at 16:43 Paris on D-1 |
| Day-ahead total load forecast (6.1.B) | Article 6(1)(b): no later than two hours before gate closure, "updated when significant changes occur" | 10:00 Paris on D-1 | 2 hours before | Regulation deadline; the API holds the latest version, so a later update cannot be excluded |
| Day-ahead wind and solar forecast (14.1.D) | Article 14(1)(d): no later than 18:00 Brussels time on D-1 | 18:00 Paris on D-1 | 6 hours after | Regulation deadline; fails the gate, so it is only in the extended set |
| Open-Meteo weather, previous-runs archive, lead day 2 | run initialised 48 to 53 hours before the valid hour, plus 6 hours for dissemination | at the latest 00:00 UTC on D-1 | at least 10 hours before | Previous-runs values matched the Single Runs API run by run (2026-09-15 test) |
| Open-Meteo historical-forecast archive | stitched from the latest run before each hour | one hour before valid time | after | Documentation; used only as a training proxy before the previous-runs archive begins |
| Calendar | known in advance | | | |

Two things to keep in mind:

- The honest set is conservative on weather. A desk at 11:30 on D-1 has the 00 UTC run of D-1 (lead 22 to 46 hours); the archive's lead day 1 would instead take the run issued 24 hours before each hour, which for the afternoon and evening of D is the 12 or 18 UTC run of D-1, after the gate. Lead day 2 is the clean choice available from the archive.
- The ENTSO-E load forecast may have been revised after 10:00 on D-1. The only way to measure actual first-appearance times is to ask the API during the morning of D-1. `thermo-fr timing-probe` does this and appends to `data/timing_probe.csv`; run it a few times between 08:00 and 19:00 Paris to pin down the real times for the load forecast, the wind and solar forecast and the prices. The one run so far (16:43 Paris on 2026-10-05) found all three items present for 2026-10-06 and none for 2026-10-07.

## Data

All series are hourly UTC, 2021-01-01 to 2026-09-30, cached under `data/cache/` and assembled into `data/forecast/inputs.csv` by `thermo-fr forecast-fetch`.

| Series | Source | Notes |
|---|---|---|
| Day-ahead price | ENTSO-E A44, FR | hourly to 2025-09-30, 15-minute from 2025-10-01 (averaged to hourly); no missing hours |
| Load forecast | ENTSO-E A65 / A01 | hourly to 2023, mixed from 2024, 15-minute from 2025; no missing hours |
| Solar, wind onshore, wind offshore forecasts | ENTSO-E A69 / A01, psrType B16, B19, B18 | 158 and 167 hours missing (mostly 2021); offshore starts 2023-08-06; the repeated autumn hour is absent |
| Temperature, 100 m wind, radiation as issued | Open-Meteo previous-runs API, best_match, lead day 2 | temperature from 2021-03-25, radiation from 2024-01-20, wind from 2024-02-17 |
| The same, proxy | Open-Meteo historical-forecast API, best_match | complete from 2021; training proxy only |
| 100 m wind at 17 wind-region points, as issued | Open-Meteo previous-runs API, best_match, lead day 2 | from 2024-02-17; input of the wind proxy |
| Actual wind onshore and offshore generation | ENTSO-E A75 / A16, psrType B19, B18 | 15-minute, averaged to hourly; calibration target of the wind proxy and definition of the windy-day slice, never a feature |
| Shortwave radiation at 21 solar-region points, as issued | Open-Meteo previous-runs API, best_match, lead day 2 | from 2024-01-20; input of the solar proxy |
| Actual solar generation | ENTSO-E A75 / A16, psrType B16 | 15-minute, averaged to hourly; 29 hours missing since 2024; calibration target of the solar proxy and definition of the sunny-day slice, never a feature |

Temperature is population weighted over the eight cities, as in the thermosensitivity model; wind and radiation are plain means, since they stand for renewable output rather than heating demand.

ENTSO-E API prices against the CSV exports in `data/csv` (2021 to 2025, 43,824 overlapping hours): identical, no hour differs by more than 0.005 EUR/MWh, and the CSV side has no hour the API lacks. The 2026 export is empty, so 2026 comes from the API only.

ENTSO-E curve type A03 omits a point when its value repeats the previous one; each period is expanded to its full length and forward filled. The gateway in front of the API intermittently answers 599 or 527 on year-long queries; these are retried with backoff.

## Method

Target: the hourly French day-ahead price in Paris delivery hours (23 rows on the spring day, 25 on the autumn day).

Features (honest set): hour, weekday, month, day of year, public holiday, and the calendar structure of `daytypes.py` (day type: working day, Saturday, Sunday or holiday; bridge day; eve of a holiday; day after a holiday; the 24 December to 2 January break; the number of days back to the most recent day of the same type); ENTSO-E load forecast; temperature, 100 m wind and radiation as forecast two days ahead; the wind and solar generation proxies (see below); price lags for the same local hour on D-1, D-2 and D-7, the mean, minimum and maximum of D-1 and the mean of D-7, and the same hour and the daily mean of the most recent earlier day of the same type (the previous Friday for a Monday, the previous Saturday for a Saturday, the previous Sunday or holiday for a holiday). The lagged prices stand in for gas and carbon, which are not inputs here. Extended set: honest plus the ENTSO-E solar and wind forecasts and the residual load (load forecast minus solar minus wind).

Weather for a row is the as-issued forecast when the archive has all three variables for that hour, else the proxy. Rows with proxy weather are kept for training but excluded from the strict metrics; in the 2024 to 2025 test window this affects 2024-01-01 to 2024-02-16.

Reduced sets exist only to measure what each addition is worth: `honest_base` has neither proxy and only the plain calendar (the original honest set of 2026-10-05), `honest_wind` adds the wind proxy (the honest set of 2026-10-09), `honest_solar` and `honest_calendar` add to `honest_wind` only the solar proxy or only the calendar structure with the same-type lag.

Models: same hour on D-1 and same hour on D-7 as benchmarks; LightGBM (800 trees, learning rate 0.03, 63 leaves, bagging and feature subsampling); ridge regression on the same information with one-hot hour, weekday and month.

Walk-forward: for every month from 2024-01 to 2025-12 the models are fitted on every delivery day before the first of the month and forecast that month. Peak hours are 08:00 to 20:00 on weekdays. Top price hours are the top 5 percent of actual prices in the test window; negative hours are those with a negative actual price.

## Results

Run on 2026-10-05 with inputs 2021-01-01 to 2026-09-30 (`thermo-fr forecast-backtest`, test window 2024-01 to 2025-12, 24 monthly refits). Numbers are copied from `reports/forecast/summary.json`. MAE / RMSE in EUR/MWh; the improvement columns are the MAE reduction against the benchmark.

Strict rows, 2024-02-17 to 2025-12-31, 16,409 hours (weather as issued for every row):

| Feature set | Same hour D-1 | Same hour D-7 | Gradient boosting | Linear | GBM vs D-1 | GBM vs D-7 | Linear vs D-1 |
|---|---|---|---|---|---|---|---|
| honest | 20.68 / 29.15 | 29.59 / 39.98 | 16.98 / 23.34 | 18.70 / 23.94 | 17.9% | 42.6% | 9.6% |
| extended (may use late information) | 20.68 / 29.15 | 29.59 / 39.98 | 15.89 / 22.69 | 17.87 / 22.85 | 23.2% | 46.3% | 13.6% |

Full window, 2024-01-01 to 2025-12-31, 17,544 hours (the first seven weeks use the weather proxy):

| Feature set | Same hour D-1 | Same hour D-7 | Gradient boosting | Linear | GBM vs D-1 | GBM vs D-7 | Linear vs D-1 |
|---|---|---|---|---|---|---|---|
| honest | 20.21 / 28.58 | 29.43 / 39.73 | 16.72 / 22.99 | 18.73 / 23.96 | 17.3% | 43.2% | 7.3% |
| extended (may use late information) | 20.21 / 28.58 | 29.43 / 39.73 | 15.70 / 22.36 | 17.98 / 23.06 | 22.3% | 46.7% | 11.0% |

By slice, strict rows, MAE:

| Hours | Count | D-1 | D-7 | GBM honest | Linear honest | GBM extended | Linear extended |
|---|---|---|---|---|---|---|---|
| all | 16,409 | 20.68 | 29.59 | 16.98 | 18.70 | 15.89 | 17.87 |
| peak (08 to 20, weekdays) | 5,856 | 22.03 | 30.11 | 18.80 | 19.71 | 17.31 | 18.15 |
| off peak | 10,553 | 19.94 | 29.30 | 15.97 | 18.14 | 15.10 | 17.72 |
| top 5% price hours (at or above 131.31) | 821 | 25.94 | 40.50 | 22.55 | 21.11 | 22.94 | 20.10 |
| negative price hours | 857 | 15.61 | 21.80 | 17.17 | 20.27 | 14.56 | 20.33 |

What the tables say:

- The honest gradient boosting model cuts the error of the same-hour-yesterday benchmark by 18% and of the same-weekday-last-week benchmark by 43%. The linear model on the same features gets about half of that gain, so the boosting adds roughly 1.7 EUR/MWh of MAE on top of a linear fit.
- The ENTSO-E wind and solar forecasts are worth another 1.1 EUR/MWh (16.98 to 15.89), most of it in off-peak and negative-price hours, which is where renewable output sets the price. Whether that gain is available at 12:00 depends on when RTE actually publishes the day-ahead renewables forecast, which the probe has not yet measured.
- By hour, the models help most in the morning ramp (07:00 to 09:00), where the D-1 benchmark is worst, and least in the midday hours. The evening peak (18:00 to 20:00) remains the hardest for every method.
- On the top 5% price hours the linear model has the lower MAE (21.1 against 22.6 for the boosting), and in negative-price hours the honest boosting is worse than the naive benchmark (17.2 against 15.6). Both are symptoms of the same problem described under the worst days.
- Month by month, the honest boosting has a lower MAE than the D-1 benchmark in 21 of 23 strict months; the exceptions are April 2024 (21.6 against 19.2) and January 2025 (32.6 against 29.4).

Charts: `reports/forecast/mae_by_hour_{honest,extended}.png`, `mae_by_month_{honest,extended}.png`, `sample_week.png` (week of 2025-01-27) and `day_2025-01-15.png` from `thermo-fr forecast --date 2025-01-15`.

Sample day, 2025-01-15 (a cold Wednesday, forecast temperature around 0 to 4 C, load forecast up to 84 GW), honest set, made from the stored history plus a live refresh of the surrounding days:

| Hour | Forecast | Same hour D-1 | Actual |
|---|---|---|---|
| 00 | 136.6 | 122.2 | 119.3 |
| 04 | 116.5 | 112.2 | 111.8 |
| 07 | 178.1 | 161.0 | 192.0 |
| 08 | 194.4 | 186.5 | 275.0 |
| 09 | 172.0 | 152.5 | 250.0 |
| 12 | 130.3 | 112.4 | 166.3 |
| 16 | 156.7 | 145.1 | 135.0 |
| 18 | 190.1 | 188.3 | 221.9 |
| 19 | 188.7 | 189.5 | 215.0 |
| 23 | 133.2 | 122.3 | 123.9 |

The model lifts the whole curve above the previous day, in the right direction, but catches only part of the morning spike: MAE 20.4 against 25.4 for the D-1 benchmark over the 24 hours.

## Worst forecast days

The twenty days with the largest daily MAE of the honest gradient boosting model are listed in `reports/forecast/report.md` (and `worst_days_honest.csv`). What they have in common:

- **Regime changes, not weather surprises.** The mean absolute jump of the daily mean price against the day before is 34.7 EUR/MWh on these days against 15.7 on an average test day. The temperature anomaly is small on average (+1.0 C) and only two of the twenty are cold anomalies below -3 C. Weekends and holidays are not over-represented (3 of 20, two of them 1 January 2025 and 1 May 2024, where the price collapses on the holiday).
- **January 2025 dominates: 7 of the 20 days.** A cold, low-wind spell (20 to 23 January, prices to 473 EUR/MWh at 18:00 on the 20th) followed by a sharp drop with strong wind (27 and 28 January, 11 m/s, daily means around 30). The single worst day, 21 January, has a daily MAE of 194.6 against 40.5 for the naive benchmark: the model forecast 270 to 480 EUR/MWh for every hour while the actual price was 115 to 245. The cause is visible in the features: the D-1 maximum of 473 and the D-1 evening lags push the trees into the leaves learnt on the 2022 crisis, where such lag levels went with prices of 300 to 700. The linear model, which cannot jump between regimes, stayed at 150 to 290 that day.
- **Days after a price collapse or a holiday.** 1 January 2025, 1 May 2024, 19 April 2024, 14 December 2024 (a Saturday after a 109 EUR/MWh Friday) and 30 September 2024 all have the naive benchmark failing by 40 to 75 EUR/MWh too; the model inherits the previous day's level through its lag features. On 17 of the 20 days the model is worse than the naive benchmark, so these are days where the lagged prices mislead rather than days the model simply missed.
- **Wind swings.** On the worst "collapse" days (6, 27, 28 January 2025; 2 January 2024) the city-mean 100 m wind forecast is 11 to 12 m/s against 6.8 on the worst days as a whole, and the ENTSO-E renewables forecast is 19 to 20 GW. The honest set sees wind only through city wind speed; the extended set, with the actual renewables forecast, does better on 6 and 27 January but not enough to leave the list.
- **Negative prices** are not the issue: 25% of the worst days have negative hours against 22% of all test days.
- **Outages** cannot be assessed with the current inputs; there is no nuclear availability series. The January 2025 spike days coincide with low wind and high load, and plant availability is the obvious missing input for them.

The practical conclusion for phase 2: forecast the price relative to a recent level (for example the D-1 daily mean) rather than its raw level, so that lag features carry shape and not regime, and add a fuel-cost proxy (gas and carbon) and nuclear availability so the model does not have to infer the regime from lagged prices alone.

## Wind generation proxy (2026-10-09)

The gas-trader review asked for a pre-gate wind input. ENTSO-E's day-ahead wind forecast may be published after the auction (the morning runs on 2026-10-08 and 10-09 saw it absent at every poll before 12:00 Paris and present after 18:00), so the honest set replaced it with a proxy built only from inputs that pass the gate:

- 100 m wind speed as forecast two days ahead (Open-Meteo previous runs, lead day 2, the same timing rule as the other weather inputs) at 17 points: Somme, Aisne, Champagne, Lorraine, Beauce, Indre, Finistere, Vendee, Eure, Poitou, Aude, Lauragais, Bourgogne, Rhone valley, and the Saint-Nazaire, Fecamp and Saint-Brieuc offshore farms. 100 m is the closest archived level to the hub heights of the French fleet (the API also has 80 and 120 m, all from 2024-02-17).
- A generic turbine power curve (cut-in 3 m/s, cubic ramp to rated output at 12 m/s, cut-out 25 m/s) turns each point's speed into a capacity factor.
- Non-negative least squares fits one weight in MW per point against ENTSO-E actual wind generation (A75, onshore plus offshore). The weights are refitted at the start of every month on the trailing 365 days ending two days before the month, and applied to that month. Actual generation per type is published within an hour of the operating period, so the calibration data of a month is public before the gate of its first day; `timing.py` has the rule (`wind_proxy`) and the look-ahead test covers it. The first calibration needs 60 days of overlap, so the proxy exists from 2024-05.
- The monthly refit saves the latest weights to `published/model/wind_proxy.json` (fitted 2025-10-01 to 2026-09-30: 21,011 MW of effective capacity, R2 0.79) and the morning run applies them to the fresh point forecasts.

How good is the proxy itself, against actual generation, 2024-05 to 2026-09 (21,141 hours, mean 5,172 MW, standard deviation 3,808 MW):

| Series | MAE | Bias | R2 | Correlation |
|---|---|---|---|---|
| Wind proxy (pre-gate, this work) | 1,387 MW | -815 MW | 0.777 | 0.918 |
| ENTSO-E day-ahead wind forecast (post-gate, for reference) | 650 MW | +232 MW | 0.935 | 0.971 |

The proxy under-reads by about 800 MW, mostly because a trailing-year fit lags the growth of the fleet; the boosting model learns the offset, so it was left as is. It carries about twice the error of the TSO's own forecast, which uses the actual fleet, shorter lead times and more weather models.

Backtest with and without the proxy, same walk-forward as above (2026-10-09, inputs to 2026-09-30, `thermo-fr forecast-backtest`, test window 2024-01 to 2025-12). `honest_base` is the honest set without the proxy, which is what the tables above describe. Strict rows, 2024-02-17 to 2025-12-31, MAE in EUR/MWh:

| Hours | Count | D-1 baseline | GBM without proxy | GBM with proxy | Change | Linear without | Linear with |
|---|---|---|---|---|---|---|---|
| all | 16,409 | 20.68 | 16.98 | 15.90 | -1.08 (-6.4%) | 18.70 | 18.59 |
| peak (08 to 20, weekdays) | 5,856 | 22.03 | 18.80 | 17.37 | -1.43 (-7.6%) | 19.71 | 19.27 |
| off peak | 10,553 | 19.94 | 15.97 | 15.09 | -0.88 (-5.5%) | 18.14 | 18.20 |
| top 5% price hours | 821 | 25.94 | 22.55 | 24.11 | +1.56 (+6.9%) | 21.11 | 21.16 |
| negative price hours | 857 | 15.61 | 17.17 | 15.79 | -1.38 (-8.0%) | 20.27 | 20.73 |
| windiest 10% of days (actual generation at or above 10,210 MW, 69 days) | 1,656 | 27.46 | 23.63 | 18.98 | -4.65 (-19.7%) | 21.77 | 20.48 |

The windiest days are the 10% of strict test days with the highest mean actual wind generation. On them the model without the proxy was barely better than the linear fit; with it the error falls by a fifth and 47 of the 69 days improve. On the calmest 10% of days nothing changes (14.10 against 14.41). The proxy only enters the training data from 2024-06, so the first months of the window are identical for both sets; from 2024-06 to 2025-12 the monthly MAE is lower with the proxy in 19 of 19 months, by 0.15 to 2.41 EUR/MWh. Over the strict rows from 2024-05, the comparison where the feature exists, the figures are 17.15 without and 15.95 with the proxy (baseline 21.26). The extended set, which uses the post-gate ENTSO-E forecasts, is at 15.36 on all strict rows and 16.39 on the windy days, so the proxy recovers about half of the gap between the honest and the extended set overall and most of it on windy days.

The one slice that gets worse is the top 5% price hours (+1.56 EUR/MWh). Those are mostly cold, calm winter evenings where the proxy adds little information and a few more trees spent on wind cost some sharpness at the top; the linear model is unchanged there. The regime-jump problem of January 2025 is untouched (32.63 to 30.59).

The published model was refitted with the proxy (`thermo-fr refit-model --from-file`): holdout September 2026 MAE 27.83 against 36.80 for the baseline, where the previous model had 29.33. The error band in the dashboards now comes from this backtest's predictions.

## Solar generation proxy and calendar structure (2026-10-09, later)

Two more additions to the honest set, measured the same way. The reissue check for Saturday 2026-10-10 (below) had shown every variant far above an auction that went to zero at midday, so a pre-gate solar input and a better reading of the working calendar were the obvious next steps.

**Solar proxy** (`forecast/solar_proxy.py`, sharing the method of the wind proxy in `forecast/gen_proxy.py`): shortwave radiation as forecast two days ahead (Open-Meteo previous runs, lead day 2, archived from 2024-01-20) at 21 points in the solar regions (Landes, Gironde, Lot-et-Garonne, Charente, Toulouse, Herault, Gard, Roussillon, Provence, Var, Alpes-de-Haute-Provence, Drome, Lyon, Nantes, Orleans, Champagne, Alsace, Bourgogne, Paris, Lille, Rennes). Each point's irradiance divided by 1,000 W/m2 is its capacity factor; non-negative least squares against ENTSO-E actual solar generation (A75, psrType B16) gives one weight in MW per point, refitted at the start of each month on the trailing 365 days ending two days before the month (`timing.py` rule `solar_proxy`, the same as the wind rule). Tilted panels and module temperature make the ratio of generation to horizontal irradiance drift through the year, and the fleet grows by about a fifth a year, so shorter windows were tried: on the proxy's own error against actual generation (2024-05 to 2026-09, 21,163 hours) windows of 60, 90, 120, 180, 240 and 365 days were all within 3 percent of each other in MAE (1,039 to 1,075 MW), the 365-day window lowest, so it keeps the wind proxy's window. The weights fitted 2025-10-01 to 2026-09-30 sum to 22,403 MW with R2 0.90 and are saved to `published/model/solar_proxy.json`.

| Series | MAE | Bias | R2 |
|---|---|---|---|
| Solar proxy (pre-gate, this work) | 1,039 MW | -471 MW | 0.878 |
| ENTSO-E day-ahead solar forecast (post-gate, for reference) | 652 MW | +49 MW | 0.932 |

**Calendar structure** (`forecast/daytypes.py`): weekday and public holiday were already features, and the D-7 lag is already the same weekday of the previous week; what was missing is how the French calendar behaves around holidays. Added: the day type (working day, Saturday, Sunday or holiday), bridge days (a working day between two non-working days, such as the Friday after Ascension), the eve and the day after a public holiday, the 24 December to 2 January break, the number of days back to the most recent day of the same type, and the same hour and the daily mean of that comparable day as price lags (the previous Friday for a Monday, the previous Saturday for a Saturday, the previous Sunday or holiday for a holiday). The comparable day is D-1 or earlier, so its price was published at the latest at 13:00 Paris on D-2 (`timing.py` rule `price_lag_same_type`).

Backtest, same walk-forward (inputs to 2026-09-30, test window 2024-01 to 2025-12, LightGBM 800 trees). `honest_wind` is the honest set of the previous section; `honest_solar` and `honest_calendar` add one of the two to it; `honest` adds both. Strict rows, 2024-02-17 to 2025-12-31, GBM MAE in EUR/MWh:

| Hours | Count | D-1 baseline | honest_wind | + solar proxy | + calendar | honest (both) | Change vs honest_wind | extended |
|---|---|---|---|---|---|---|---|---|
| all | 16,409 | 20.68 | 15.90 | 15.65 | 15.61 | 15.51 | -0.39 (-2.5%) | 14.93 |
| peak (08 to 20, weekdays) | 5,856 | 22.03 | 17.37 | 16.94 | 16.89 | 16.77 | -0.60 (-3.5%) | 15.97 |
| off peak | 10,553 | 19.94 | 15.09 | 14.93 | 14.90 | 14.82 | -0.27 (-1.8%) | 14.36 |
| weekends | 4,697 | 19.85 | 14.20 | 14.12 | 14.12 | 13.92 | -0.28 (-2.0%) | 13.76 |
| public holidays (21 days) | 504 | 24.70 | 18.89 | 18.39 | 18.14 | 16.91 | -1.98 (-10.5%) | 15.25 |
| sunniest 10% of days (actual solar at or above 5,184 MW daily mean, 69 days) | 1,656 | 19.41 | 15.94 | 15.59 | 15.64 | 15.71 | -0.23 (-1.4%) | 15.75 |
| windiest 10% of days | 1,656 | 27.46 | 18.98 | 18.11 | 18.54 | 17.87 | -1.11 (-5.8%) | 15.87 |
| top 5% price hours | 821 | 25.94 | 24.11 | 24.30 | 22.07 | 23.21 | -0.90 (-3.7%) | 22.09 |
| negative price hours | 857 | 15.61 | 15.79 | 15.07 | 15.15 | 14.39 | -1.40 (-8.9%) | 13.16 |

The two additions are worth about the same on their own (0.25 and 0.29 EUR/MWh overall) and add up almost fully. Holidays gain most (16.91 against 18.89, 13 of 21 days improved) and the midday hours of the sunniest days go from 13.71 to 12.95 (414 hours, mean price 15.1); the sunniest days as a whole move little because their error sits in the evening ramp, not at noon. The top 5% price hours recover part of what the wind proxy had cost (23.21 against 24.11, still above the 22.55 of the set with neither proxy), so that known weakness stays in the Limitations text. Month by month the new set beats `honest_wind` in 16 of 23 months; the largest gain is January 2025 (27.27 against 30.59) and the largest loss December 2024 (21.49 against 20.64). On the strict rows from 2024-05, where both proxies exist, the figures are 15.95 (`honest_wind`), 15.46 (`honest`) and 14.91 (`extended`, post-gate ENTSO-E forecasts), so the honest set now closes about half of the remaining gap to the extended one.

The published model (`published/model/honest.txt`, 300 trees, 31 leaves) was refitted with the new set: holdout September 2026 MAE 27.73 against 36.80 for the baseline (27.83 with `honest_wind`). Because the reissue check below showed the 300-tree model far from the 800-tree refit on one day, the published settings were walked forward over 2025 as well: honest 15.47 against 15.19 for the backtest settings, `honest_wind` 15.73 against 15.76, so the smaller file costs at most 0.3 EUR/MWh on average and the single-day gap was not systematic.

**Reissue check, delivery 2026-10-10** (a Saturday; inputs as stored by the 11:00 Paris run of 2026-10-09, prices of the day blanked, proxies from point forecasts of 10-08 and weights fitted to 2026-09-30). The auction's base was 51.26 EUR/MWh with 0 from 13:00 to 16:00 and 124 to 173 in the evening; the same-hour-previous-day baseline was 75.56. Base (24-hour mean) by variant, live refits with 800 trees: 74.08 without proxies (the version actually issued), 69.13 with the wind proxy, 63.14 with wind and solar proxies, 72.71 with the wind proxy and the calendar structure, 72.83 with everything. Stored 300-tree models: 83.05 without proxies, 74.01 with the wind proxy (the model published on 10-09), 88.28 with the new set. Every variant stayed well above the auction; the new published model's evening (135 to 148 against 137 to 173) was the closest of any run, its night and midday were too high. One day decides nothing, which is why the backtest above is the evidence.

## Versus the market (2026-10-09, later)

A gas and power trader's review drew the line that this document had blurred: "market data" means traded prices, not ENTSO-E. For French day-ahead power the traded price is the EEX French Day-Ahead Base (and Peak) future, traded on the morning before the auction, liquid from about 08:00 to 12:30 Paris and tradeable until about 12:00; the EPEX auction result follows around 12:50 CET. The README glossary now keeps fundamentals (ENTSO-E, RTE, weather), spot (the auction result) and market (the traded price) apart.

The right test of a forecast is therefore not its error against yesterday's spot price, and not the share of days it beats a naive baseline, but whether, issued before the trading window, it points the right way relative to where the market already trades, and what that would have earned. `forecast/market.py` does this for every day with a traded price in `data/market/eex_fr_da.csv` (hand-entered, git-ignored, never published): the latest honest forecast issued before the window opened is averaged to the product (base: every hour of the Paris day, 23 or 25 on the clock-change days; peak: 08:00 to 20:00), the entry is the window VWAP (the close is reported as an alternative), the direction is long above the entry and short below, and P&L per MWh is (auction result - entry) x direction. A day whose only forecasts came after the window is skipped, so each day is scored with the model that was live at the time. No-trade bands (trade only when the forecast is more than X EUR/MWh from the entry) are tried for a few X and labelled in sample.

The sample so far is one day. Delivery 2026-10-10: the EEX FR DA Base traded 11:15 to 12:00 Paris on 10-09 (the prices are in the local file, not here). The forecast live before 11:15 was the 11:00 version at a base of 74.08, above the traded VWAP, so the signal was long; the auction settled at 51.26, below the entry, so the trade lost a little while the forecast error was 22.82 (the traded price's error, and so the P&L, is not quoted here because together with the auction result it would give the VWAP away). The trader quoted "about 80" for the forecast, which is the version the public dashboard showed later that day (issued after the window and therefore not scored). Every no-trade band from 0 to 10 EUR/MWh keeps this one trade. One day decides nothing; the record grows one hand-entered day at a time. The public app shows aggregates only once 20 days are scored (`MIN_PUBLIC_DAYS`), since with a handful of days the mean P&L and the public auction result would let a reader back out the traded prices; until then it says "collecting data, n of 20 days".

Two consequences for the pipeline. The headline forecast of a delivery day, in both dashboards and in `published/tomorrow.csv`, is now the pre-market version: the last one issued before 11:15 Paris on the day before, the one a trader could have acted on; later versions are listed underneath as issued after the market window and not tradeable, and the daily scores use the pre-market version of each day. And the GitHub Actions schedule was moved so that a run completes inside the pre-market window all year: the workflow's old crons (09:30 and 10:30 UTC) landed at 11:30 and 12:30 Paris in summer, after the window had opened, so no cloud forecast could have been scored against the market. Four pre-market crons now fire (08:10, 08:35, 09:10, 09:35 UTC) and `forecast/schedule.py` keeps the ones that start between 10:05 and 11:00 Paris in the current season, skipping a run that is before the load forecast deadline or could finish after the window opens; one later run (10:30 UTC) is kept for information. The local Task Scheduler times were aligned (09:15 and 09:45 London are the pre-market runs).

## Collecting the traded prices (2026-10-10)

The traded prices were first logged by hand from a trader's message. They are now collected: EEX's public Market Data Hub shows, with a 15 minute delay and for about 45 days back, the trade-by-trade tape of each day future, so the 11:15 to 12:00 Paris window can be rebuilt exactly (open, high, low, close, volume-weighted average, trade count and volume) rather than approximated from a last price. A check against the trader's five figures for the 2026-10-10 delivery matched to the cent. Two things about that page shaped the design. Its data is served to the page's own widget and refused to a bare request, so the collector sends one extra header the page expects, with a plain user agent naming the project and a contact address, no proxies and no rotation, a handful of requests a day ten seconds apart; and EEX's notice says the data is for information purposes, with systematic republication needing its permission. The collector therefore lives in `local/`, a git-ignored folder, and nothing it reads, including its raw responses, is committed, published or shown in the public app; the repository carries only the generic contract the module meets (`run_fetcher` in `forecast/market.py`), the paste parser, the evaluation and the dashboard panel. It runs from the laptop's Task Scheduler on weekdays at 12:20 Paris with one retry at 12:35, never from GitHub Actions; on Fridays it also collects the Saturday, Sunday and Monday deliveries, whose futures trade on Friday (the market file now carries a trade_date column and the evaluation uses it for the window; Sunday and Monday deliveries have no pre-market forecast because the morning runs are weekday jobs, so those rows are kept for the record and reported as skipped). A refusal or a changed page is logged and shown in the data status panel and the day is left to the paste fallback; nothing is guessed. Peak day futures exist for weekday deliveries only, and a thin contract can have trades on the day but none inside the window, which is logged as skipped too.

Backfilled record, delivery days 2026-08-29 to 2026-10-10, scored as a walk-forward so that nothing is in sample: the September and late-August days with a model fitted on data through August, the October days with the published model (fitted on data through September) or, for the three rows where a live version existed before the window, with that live version. 38 base and 4 peak windows (the peak contract rarely trades inside the window; Sunday and Monday contracts, traded on Friday, had trades on the day but none inside it). Hit rate 29%, mean P&L -2.5 EUR/MWh per window (total -105 over 42), model error against the auction 27.0 EUR/MWh against 5.1 for the traded VWAP, model error below the market's on 12% of windows, long on 69% of them. Base alone: 32% hit rate, -2.1 per window, 24.8 against 4.9. October alone, the published model and the live runs: 15% hit rate, -4.5 per window, 34.9 against 6.3, a period in which the base price fell from about 200 to about 50 EUR/MWh within a week. The main reason is that the model reacts to large moves about two days late, because its price lags carry yesterday's level into tomorrow's forecast, while the market reprices the same morning. So the two measurements give two different answers: the model beats the naive baseline on forecast error and loses against the traded price. The per-day table, which carries the traded prices, stays in the local dashboard.

## Structural drivers, fundamentals-v2 (2026-10-10)

The branch adds three groups of inputs to the honest set, each with a timing rule in `forecast/timing.py` and a look-ahead test, and checks them against two deadlines: the 12:00 Paris gate and the pre-market issue time, 10:05 Paris on D-1, when the earliest scheduled run starts (`check_point_in_time(..., deadline="premarket")`). The protocol, the pre-registered acceptance rule and every result are in `docs/experiments.md`.

**Nuclear.** ENTSO-E actual generation per type (A75, psrType B14) for France. Checked on 2026-10-10 at 08:51 UTC: the latest quarter-hour on the platform ended 08:00 UTC, so the data arrives within about an hour, as Regulation 543/2013 Article 16(1)(a) requires; the rules allow two hours. Features: the same hour of D-2, the D-2 daily mean, and the mean of the D-1 hours ending by 08:00 Paris (known by 10:00 Paris on D-1). Planned available nuclear capacity for D as of the issue time was the intended second feature and was not built. The unavailability documents (A77 production units, A80 generation units) were probed the same day: the API returns only the latest revision of each notice (one revision per mRID in the answer, `createdDateTime` being that revision's time), earlier revisions are not retrievable, the `periodStartUpdate` filter selects documents by their latest update only, and an answer is capped at 200 documents (a week of French nuclear A80 notices exceeded it). A notice first published before the issue time and revised later therefore cannot be seen as it stood at the time, and dropping every notice whose latest revision came later would under-count outages known at the time, with a bias that grows with the length of the outage. A strict as-of reconstruction is only possible going forward, by archiving a daily snapshot of the notices. The nuclear-archive branch (2026-10-10) starts that: `thermo-fr outage-snapshot` saves the raw A80 and A77 answers for France every morning at 09:00 London on weekdays (before the pre-market runs) and once a day at weekends, paging with the endpoint's `offset` parameter past the 200-document cap (one week of French notices was 288 documents in two pages), the retrieval time in every file name, under `data/entsoe/outage_snapshots/` (git-ignored). `thermo_fr.data.outages.planned_nuclear(delivery_day, as_of)` rebuilds hourly planned unavailability and availability from the latest snapshot taken at or before `as_of`, keeping only A80 notices of nuclear units created by then, dropping cancelled and withdrawn ones and the retired Fessenheim units; a test checks that no later snapshot is ever read. The psrType filter is not honoured by that endpoint, so the nuclear filter is applied when reading. The archive is of no use to the backtest until it has covered a selection window, which is why no model change comes with it. The residual load therefore uses the latest known actual nuclear generation in place of planned availability.

**Neighbour prices.** Day-ahead prices for delivery D-1 of DE-LU, BE, NL, ES, IT-North and CH (ENTSO-E A44, one query per zone, averaged to hourly), known about 13:00 Paris on D-2, as same-hour lags and as the French minus neighbour spread. Great Britain is not on the platform after 2021 and is skipped. The Swiss auction for D itself clears before the French gate but is not reliably published by the pre-market issue time, so it is not used. Cross-border capacity is out of scope.

**Residual load.** Load forecast minus the wind proxy minus the solar proxy minus the latest known nuclear generation, known with its latest component, the load forecast, at 10:00 Paris on D-1.

**Results (docs/experiments.md has the full rows).** Selection window, strict rows 2024-02-17 to 2026-06-30, against the current set's 16.39 EUR/MWh: nuclear alone 16.36, neighbours alone 16.26, both 16.16, both plus the residual load 15.70, every slice better or within 0.4 of the baseline, so all four pass the pre-registered rule and the full set is the accepted variant. Its single holdout evaluation (2026-07-01 to 2026-10-10, walk-forward with monthly retraining) gives 25.81 against 26.23 for the current set and 29.37 for the naive baseline: a gain of 0.4, concentrated in peak hours (29.0 against 31.0), with the windiest days (43.1 against 40.5) and the sunniest days (22.3 against 19.1) worse. In the October crash, 2026-10-01 to 10-10, it is worse than the current set (36.0 against 33.8) and no better than the naive baseline (36.3). The lag diagnostic improves in sample (correlation of the absolute daily error with the absolute two-day price move 0.149 to 0.101) and worsens out of sample (0.202 to 0.265; signed correlation -0.067 to -0.138): the forecast still follows large moves late. The EEX record on the same 42 windows moves from a 33% hit rate, -2.8 EUR/MWh per window and a model MAE of 29.5 against the market's 5.1 to 33%, -2.4 and 28.5. The gradient boosting model spends about a fifth of its split gain on the Swiss D-1 price next to the French D-1 and same-type lags; the nuclear and residual features take about 2% between them. The v2 model is saved as `published/model/honest_v2.txt` with its metadata; the morning run keeps `honest.txt` as its default and falls back to it when a v2 input is missing (recorded in data_status under the item "model" with status "fallback").

## Things to double-check

1. **Open-Meteo lead time.** The honest set uses forecasts issued 48 to 53 hours before each hour (lead day 2), which is more conservative than what a desk has at 11:30 on D-1 (the 00 UTC run of D-1). Lead day 1 would be closer but uses the 12 and 18 UTC runs of D-1 for the afternoon and evening of D, after the gate. If you prefer the realistic mix (day 1 for hours before 12:00 UTC on D, day 2 after), say so and it is a small change in `weather_forecast.py` and `timing.py`.
2. **Weather archive start.** As-issued wind and radiation forecasts exist only from 2024-02-17 (temperature from 2021-03-25). Training rows before that use the historical-forecast proxy, which is close to actual weather; the model therefore learns on cleaner weather than it is tested on. The strict metrics exclude 2024-01-01 to 2024-02-16 from the test window.
3. **Load forecast revisions.** The ENTSO-E API serves the latest version of the day-ahead load forecast. The regulation requires publication two hours before gate closure, but updates "when significant changes occur" are allowed and the API does not say when the stored value was last changed. The honest set treats it as a 10:00 D-1 input.
4. **Publication times are from the rules, partly measured.** The scheduled morning runs (timing log in the dashboard) have so far seen the load forecast and the weather present at the 08:00 Paris poll and the ENTSO-E wind and solar forecast absent at every poll before the gate, so the extended set does use late information and the wind proxy above is the pre-gate substitute. Keep watching the log; if the wind and solar forecast ever appears before 12:00, the extended set becomes legitimate.
5. **Regime jumps.** The gradient boosting model extrapolates badly after extreme days (21 January 2025); see the worst days. The headline MAE is still better than the benchmarks, but a desk would not accept a 480 EUR/MWh forecast for a 120 EUR/MWh hour. This is the first thing to fix in phase 2.
6. **Price comparison.** API prices and the CSV exports are identical on 2021 to 2025, so either source can be used for the target; 2026 exists only through the API.
7. **Nothing from the thermosensitivity pipeline changed** except the `entsoe` source, which now uses the direct REST client instead of entsoe-py (the `entsoe` extra is gone, `forecast` is new). The cached RTE load in `data/cache/rte` was not touched.
