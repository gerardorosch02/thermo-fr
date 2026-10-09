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

Temperature is population weighted over the eight cities, as in the thermosensitivity model; wind and radiation are plain means, since they stand for renewable output rather than heating demand.

ENTSO-E API prices against the CSV exports in `data/csv` (2021 to 2025, 43,824 overlapping hours): identical, no hour differs by more than 0.005 EUR/MWh, and the CSV side has no hour the API lacks. The 2026 export is empty, so 2026 comes from the API only.

ENTSO-E curve type A03 omits a point when its value repeats the previous one; each period is expanded to its full length and forward filled. The gateway in front of the API intermittently answers 599 or 527 on year-long queries; these are retried with backoff.

## Method

Target: the hourly French day-ahead price in Paris delivery hours (23 rows on the spring day, 25 on the autumn day).

Features (honest set): hour, weekday, month, day of year, public holiday; ENTSO-E load forecast; temperature, 100 m wind and radiation as forecast two days ahead; the wind generation proxy (see below); price lags for the same local hour on D-1, D-2 and D-7, and the mean, minimum and maximum of D-1 and the mean of D-7. The lagged prices stand in for gas and carbon, which are not inputs here. Extended set: honest plus the ENTSO-E solar and wind forecasts and the residual load (load forecast minus solar minus wind).

Weather for a row is the as-issued forecast when the archive has all three variables for that hour, else the proxy. Rows with proxy weather are kept for training but excluded from the strict metrics; in the 2024 to 2025 test window this affects 2024-01-01 to 2024-02-16.

A third set, `honest_base`, is the honest set without the wind proxy; it exists only to measure what the proxy adds.

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

## Things to double-check

1. **Open-Meteo lead time.** The honest set uses forecasts issued 48 to 53 hours before each hour (lead day 2), which is more conservative than what a desk has at 11:30 on D-1 (the 00 UTC run of D-1). Lead day 1 would be closer but uses the 12 and 18 UTC runs of D-1 for the afternoon and evening of D, after the gate. If you prefer the realistic mix (day 1 for hours before 12:00 UTC on D, day 2 after), say so and it is a small change in `weather_forecast.py` and `timing.py`.
2. **Weather archive start.** As-issued wind and radiation forecasts exist only from 2024-02-17 (temperature from 2021-03-25). Training rows before that use the historical-forecast proxy, which is close to actual weather; the model therefore learns on cleaner weather than it is tested on. The strict metrics exclude 2024-01-01 to 2024-02-16 from the test window.
3. **Load forecast revisions.** The ENTSO-E API serves the latest version of the day-ahead load forecast. The regulation requires publication two hours before gate closure, but updates "when significant changes occur" are allowed and the API does not say when the stored value was last changed. The honest set treats it as a 10:00 D-1 input.
4. **Publication times are from the rules, not measured.** Run `thermo-fr timing-probe` a few times during a weekday morning (08:00, 10:30, 11:55, 13:15, 18:30 Paris). The log in `data/timing_probe.csv` will show when the load forecast, the wind and solar forecast and the prices for the next day first appear. If the wind and solar forecast turns out to be there before 12:00, the extended set becomes legitimate.
5. **Regime jumps.** The gradient boosting model extrapolates badly after extreme days (21 January 2025); see the worst days. The headline MAE is still better than the benchmarks, but a desk would not accept a 480 EUR/MWh forecast for a 120 EUR/MWh hour. This is the first thing to fix in phase 2.
6. **Price comparison.** API prices and the CSV exports are identical on 2021 to 2025, so either source can be used for the target; 2026 exists only through the API.
7. **Nothing from the thermosensitivity pipeline changed** except the `entsoe` source, which now uses the direct REST client instead of entsoe-py (the `entsoe` extra is gone, `forecast` is new). The cached RTE load in `data/cache/rte` was not touched.
