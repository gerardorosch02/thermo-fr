# One out-of-sample test of the France to Germany spread forecast: frozen plan

Written on 2026-11-19 from 16:10 London and committed with its code the same afternoon, before any German price or German
forecast dated after 2026-06-30 was brought into the project's files and before the spread of any day after 2026-06-30 was
computed by anyone. The time of the commit is the time this plan was frozen.

**Owner's approval.** On 2026-11-19 the owner was asked, in these words, for "your approval to use the data after 30 June
for this one test, since your earlier exception covered only the French replay" and for a refetch of German prices and
forecasts on his laptop. He answered "yes" at 16:08 London. The approval covers this one test and nothing else.

## What is tested

A forecast of the hourly day-ahead price spread, France minus Germany, issued at 10:05 Paris on the day before delivery.
The exploration of 2026-11-19 (`scripts/explore_fr_de_spread_points.py`, row of that date in `docs/experiments.md`) found on
1 May 2024 to 30 June 2026 that German wind and solar forecasts built from weather points lower the hourly error from 22.16
to 18.47 EUR/MWh. That window was also used to design the model, so the figure is a backtest. This plan tests the same
model, unchanged, on days it has never seen.

## What has and has not been seen

- No German price and no German ENTSO-E forecast dated after 2026-06-30 is in the project's stored files
  (`neighbour_fundamentals.csv` and the per-sequence price extract both end on 2026-06-30).
- The spread of a day after 2026-06-30 has never been computed.
- On 2026-11-18 raw German prices to 2026-09-30 were read once on the owner's laptop to identify which of the two published
  series is the coupled auction, by the share of hours equal to the French price. No forecast was involved. This is logged in
  `docs/prereg/price_seq_results.md`.
- The French prices and French inputs of the window are known to the project. The July to October evaluation of 2026-11-19
  scored the French price models on them. No spread was involved.
- The German weather points were fetched on 2026-11-19 to 2026-11-18. Summary statistics of the whole file were printed once
  (minimum, mean, maximum and missing hours per point). No proxy has been computed for a day after 2026-06-30.
- The model was designed on the development window with one leak found by the reviewer and fixed (lags taken 24 hours back in
  UTC reached into the delivery day on the two 25-hour days). The fix is part of the frozen code.

## Window and data

- Delivery days 2026-07-01 to 2026-10-10 inclusive, strict hours. Expected 2,448 hours on 102 days. The window ends on
  10 October because the French inputs file ends there.
- French inputs: `data/forecast/inputs.csv`, sha256 `c06e285f2e757dd1de04b2f9ca7f6b982621db249f160ea850b793a1d7b0dd9f`, with
  the wind and solar proxies filled by `forward_q.fill_proxy_gaps`. This is the history of the July to October evaluation.
  The fill touches only days from 2026-08-20 (asserted in the code).
- German data to 2026-06-30: `data/forecast/neighbour_fundamentals.csv`, sha256
  `c170f29f35ded68f8b6d4d340c833c67d051171c982f42ab77bdc83bf4889c5a`, and the per-sequence extract
  `neighbour_sequences.csv.gz`, sha256 `7420fc6f0b5ca17cc4c7cec67d6e50f3217064876556abcc16cd61c589c1d8c3`.
- German data after 2026-06-30: raw ENTSO-E XML fetched by the owner with `scripts/fetch_spread_test_raw.py` (day-ahead prices,
  day-ahead load forecast, day-ahead wind and solar forecast, DE-LU, 2026-05-31 to 2026-10-11 UTC). The file hashes are
  written to the stop-check record when the files arrive. June is fetched so the files can be compared with the stored ones.
- German weather points: `german_wind_points.csv.gz`, sha256
  `80652bef984d40f42d0a79eb0f44a92b32245bc7767a1d2edaf0243bdd26b933`, and `german_solar_points.csv.gz`, sha256
  `063ec8b851c1ad5cbbad32ab8209076c419022a00283fb174aad2ebe9e43fcf2` (Open-Meteo, forecasts as issued two days ahead).
- Stored values are never replaced by refetched ones. The refetched values are used only for days after 2026-06-30.

## Models, frozen

Code: `src/thermo_fr/forecast/spread_eval.py`. Each model is a gradient-boosted regression of the hourly spread at 300 trees
of 31 leaves (`GBM_PARAMS` with `REFIT_PARAMS`), refitted at the start of every calendar month on every earlier delivery day
from 2021. The monthly refits continue through the window: the model for August is fitted on days to 31 July, and so on.

| Name | Inputs | Available at 10:05 |
|---|---|---|
| H | The spread's own past, the calendar, and the French inputs of the public model (honest_v2) | yes |
| G | H plus the German block | yes |
| C | G plus ENTSO-E's German load, wind and solar forecasts of the delivery day | no, diagnostic only |

The German block, for delivery day D: wind and solar proxies of D, ENTSO-E's load forecast of D-1 and D-7 at the same Paris
hour, ENTSO-E's wind and solar forecasts of D-1, the proxies of D-1, a German national holiday flag, and load of D-7 minus
the proxies. The proxies are built by `gen_proxy.rolling_proxy` with the French curves from 14 onshore wind points, 3
offshore points and 13 radiation points. Radiation is moved one hour earlier to period-start labels. The weights are refitted
each month by non-negative least squares on the trailing 365 days of ENTSO-E's day-ahead forecasts of past days, ending two
days before the month. On training days before May 2024, when no proxy exists, ENTSO-E's forecast of that day stands in.

H, G and C are the sets `honest`, `points_hist` and `ceiling` of the exploration. The frozen module reproduces the
exploration's stored forecasts with a largest difference of 0.0 on all 18,981 hours.

Reference figures on the development window, 2024-05-01 to 2026-06-30, 18,982 hours on 791 days (the frozen module, stored
forecasts sha256 `0b13c70fea1d001c3f5e771ec3d99d93c220706d0f1722a26e0d976649a2c151`):

| | Hourly error, EUR/MWh |
|---|---|
| Yesterday's spread at the same hour | 25.069 |
| H | 22.163 |
| G | 18.473 |
| C | 17.388 |

On the same calendar window in earlier years, 1 July to 10 October, G minus H was -2.85 in 2024 and -4.07 in 2025.

## One decision

**G against H**, by the project's gain rule, on all scored hours of the window. G passes if all three hold.

1. The hourly error of G is at least 0.25 EUR/MWh below that of H.
2. On none of the five slices is G worse than H by more than 1.0 EUR/MWh. The slices are weekends, French or German public
   holidays, the windiest tenth of days by the German wind proxy, the sunniest tenth of days by the German solar proxy, and
   the top 5% of hours by absolute realised spread.
3. The one-sided 95% upper end of the mean paired daily difference (G minus H, daily mean absolute errors) is below zero.
   Moving-block bootstrap, block 7, 5,000 resamples, seed 0. There is one decision, so no Bonferroni division.

Reading rule, written now.

- **Passes.** The German block improved the spread forecast on days it had never seen. This is a reason to log the forecast
  forward, never proof that it will keep working.
- **Fails.** At least one condition is not met. Which one is reported. The backtest gain is then not confirmed out of sample.
- **Contradicts.** The mirrored lower end of the interval is above zero, so G was reliably worse than H.

Nothing public changes on this result alone, and no trading value is claimed from it. There is no market price of the spread
in the project, so the test says whether German inputs improve the forecast, not whether the forecast beats the market.

## Reported with the decision, not part of it

- The errors of yesterday's spread, H, G and C, overall and by month.
- G against yesterday's spread and C against G, with intervals. The share of C's gain over H that G recovers.
- The error of the daily base spread, and the share of days with a day-on-day move above 5 EUR/MWh on which each model has
  the direction right.
- The same errors before 2026-08-20 and from 2026-08-20, because from that day the French wind and solar proxies are the
  reconstructed ones.

## Stop conditions, in this order, before any error of the window is computed

The stage `stops` runs them and writes `stop_checks.json`. The stage `summary` refuses to run unless all passed and the
forecasts on disk are the ones the checks were run on.

1. **The refetched files are the same series as the stored ones.** On the delivery days of June 2026 the refetched German
   coupled price is within one cent of the stored one on at least 99.9% of hours, and each of the four ENTSO-E forecasts is
   within 1 MW on at least 99% of hours, with no hour missing.
2. **Coverage.** No German input, no price and no spread lag is missing on a whole day of the window. No French input is
   missing on a whole day unless it was missing in the same way on 1 July to 10 October 2025. The window has 102 days.
3. **The stored forecasts reproduce.** With the refetched data joined on, the forecasts of H, G and C on the development
   window equal the stored ones to 1e-9 on every stored hour.
4. **The reference figures reproduce.** The four development errors above, to 0.005.
5. **Scramble.** On 2026-07-01, 2026-08-15, 2026-08-26, 2026-09-30 and 2026-10-10, every value not known at 10:05 Paris on
   the eve is replaced by noise: German prices and ENTSO-E forecasts of D and later, the French inputs by the project's rule
   (`lear_recipe.scramble_from`, which includes the French price of D), German wind points after the last hour of D and
   radiation points after the first label of D+1. The inputs of H and G for D must be identical and their forecasts must
   differ by at most 1e-9. C reads forecasts of D, so it must move, which shows the scramble bites. The same check passed on
   the development days 2025-03-30 (23 hours) and 2025-10-26 (25 hours) with a difference of 0.0.
6. **Hours.** The scored hours cover 102 days and at least 2,424 hours.

If a condition fails, nothing is scored. The failure and its cause are written into an amendment before anything else is
read, as in the July to October evaluation.

## Review

An independent reviewer who has not seen the build recomputes the decision and the reported figures from the saved forecasts
before anything is reported to the owner as a result.

## Limits known in advance

- 102 days of one summer and early autumn. No winter.
- From 2026-08-20 the French wind and solar proxies are reconstructed. Whether those values were available on each of those
  mornings has not been independently verified.
- The forecast vintage of the weather points is taken on Open-Meteo's word.
- Whether ENTSO-E revised its day-ahead forecasts after first publication is unknown. This affects the lagged German inputs
  and the calibration target of the proxies.
- The weather archive starts in 2024, so the model learned the proxy columns partly from a stand-in.

## After the run

The window is spent for H, G and C. A variant designed after reading the result gets a new name and is judged only on later
days. Raw rows of the inputs, the German tables and the prices are never printed or published; aggregates only. Amendment 3
of the forward test is unchanged: Q and P stay unscored on every live day.
