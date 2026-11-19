# W, the redesigned spread forecast: design record and frozen plan for its test

Written on 2026-11-19 in the evening, London, and committed with the code the same night. The time of the commit is the
time this plan was frozen. No German price or German forecast of a delivery day after 2026-10-10 has been used by anyone.

**Owner's approvals.** After the spread test failed its rule (`spread_eval_results.md`), the owner was asked on 2026-11-19
"Do you approve opening 1 July to 10 October 2026 for diagnosis? Anything I change afterwards would be a new model, judged
only on days after 10 October." He answered "i approve any new changes for a new model" at 18:35 London and again at 22:35.
That is read as approval to open the window and to build a new model. It does not cover any change to a scheduled task on
his laptop or to anything public.

## Why the first model failed, in short

The diagnosis is logged in `docs/experiments.md` (rows of 2026-11-19) and its working scripts are in
`scripts/spread_opened_window/`.

- In July 2026 the French discount to Germany closed. The monthly mean spread was -45 and -43 EUR/MWh in May and June, then
  -10, -4, -2 and -2 from July to October. French prices rose from 66 to 153 while German prices rose from 110 to 155.
- Nothing in the model's inputs announces it. French nuclear output in July was at its 2025 level, and no input left its
  training range except German solar on about 5% of hours.
- The German wind signal kept working (windiest tenth of days: 7.0 better than without German inputs, as in the backtest).
- On calm, sunny days the model kept forecasting the old pattern, France far below Germany at night, by 17 to 20 EUR/MWh
  more than realised. Those are the days on which it lost to the model without German inputs.
- The models were refitted once a month on all history with equal weight, so they adapted slowly.

## W

Code: `src/thermo_fr/forecast/spread_w.py` (sha256 b616ecc971b5a2737462a1caa9d65480a284e0469d9002d2c6a8ac1b5781ae2e) on
top of `spread_eval.py` (sha256 01507a3c97a3a7aa5c86b6d08eb51c37058ca4d5c640bb01aec1f220e2fbe5d6, unchanged).

W reads exactly the inputs of G. Two things change.

1. It is refitted every seventh day, counted from 2024-05-01, instead of once a month.
2. Each training day is weighted 0.5 ** (age in days / 180).

Everything else is G: 300 trees of 31 leaves, the level of the spread as the target, every earlier day from 2021.

### What was tried on the way, all on design data

Monthly refits unless said. Hourly error in EUR/MWh, backtest (2024-05-01 to 2026-06-30) then opened window.

| Variant | Backtest | Opened window |
|---|---|---|
| G as frozen | 18.47 | 16.72 |
| Target measured from the 7-day mean of the spread | 19.12 | 16.70 |
| Target measured from yesterday's spread | 18.96 | 17.33 |
| Weights with half-life 730 days | 17.97 | 15.85 |
| Half-life 365 days | 17.89 | 15.70 |
| Half-life 180 days | 17.78 | 15.52 |
| Half-life 90 days | 17.77 | 16.10 |
| Weekly refits, no weights | 18.00 | 16.18 |
| **Weekly refits, half-life 180 days (W)** | **17.34** | **14.84** |
| W plus a correction from its errors of the last 14 days | 17.34 | 14.94 |
| Half-and-half blend of H and G | 19.46 | 16.18 |

The half-life was chosen on the backtest among 90, 180, 365 and 730 days with monthly refits. Weekly refits were tried only
without weights and at 180 days. Weights with a half-life of 365 days on H do little (22.02 and 18.70).

### Design figures of W, not test results

| | Backtest, 791 days | Opened window, 102 days |
|---|---|---|
| H | 22.16 | 18.53 |
| G | 18.47 | 16.72 |
| W | 17.34 | 14.84 |
| C (not attainable) | 17.39 | 15.79 |
| W minus H, paired, 95% interval | -4.83 [-5.55, -4.14] | -3.69 [-5.49, -2.03] |
| W minus G, paired, 95% interval | -1.13 [-1.49, -0.80] | -1.88 [-2.70, -1.03] |
| Sunniest tenth of days, W minus H | -4.06 | +0.22 |

W was designed with both periods in view, so neither column is evidence that it works on new days. The frozen module
reproduces the explored variant to 5.7e-14 on all 21,430 hours (stored forecasts sha256
ab885a4f3e5fb882a5de62ff4053d58dcbea03faead17d13d64b6b17a6b367cb). The scramble check passes on 2025-10-26, 2026-07-01 and
2026-08-26 with a difference of 0.0 and identical inputs.

## The test

- **Window.** Delivery days 2026-10-11 to 2027-01-18 inclusive, strict hours: 100 days, 2,401 hours expected (25 October
  has 25). One read, on or after 2027-01-19. The owner may shorten or lengthen the window by an amendment written before any
  German price or forecast after 2026-10-10 is fetched, and not after.
- **Models.** W as frozen here. H and G as frozen in `spread_eval.py`, their monthly refits continuing. C as a diagnostic.
- **Two decisions**, each by the project's gain rule on all scored hours, with Bonferroni for two:
  1. **W against H.** Is the forecast with German inputs, as redesigned, better than the forecast without them?
  2. **W against G.** Did the redesign help?

  A comparison passes if the hourly error is at least 0.25 EUR/MWh lower, no slice is worse by more than 1.0, and the
  one-sided upper end of the mean paired daily difference at level 1 - 0.05/2 is below zero (moving-block bootstrap, block 7,
  5,000 resamples, seed 0). The slices are those of the first test, defined inside the new window: weekends, French or
  German public holidays, the windiest and the sunniest tenth of days by the German proxies, the top 5% of hours by absolute
  realised spread.
- **Reading.** Both pass: W held up on days it had never seen. Only the first: German inputs help, the redesign is not
  confirmed. Only the second: the redesign helps a model whose advantage over H is not confirmed. Neither: not confirmed.
  A pass is a reason to keep logging the forecast, never proof.
- **Reported, not part of the decisions.** The errors of yesterday's spread, H, G, W and C overall and by month; the daily
  base spread error and the direction figure; the same two comparisons with slices of fewer than 10 days set aside. The
  backtest showed that a holiday slice of about three days fails by chance in one 102-day window in six
  (`scripts/explore_spread_sunny_days.py`), so that second reading is given beside the rule. The rule itself is unchanged.

## Stop conditions, before any error of the new window is computed

1. The refetched German prices and ENTSO-E forecasts equal the stored raw files of the first test on the delivery days
   2026-09-11 to 2026-10-10: prices within one cent on at least 99.9% of hours, forecasts within 1 MW on at least 99%.
2. The refetched French inputs give the same H inputs as the stored file on every hour to 2026-10-10 where the stored file
   has them, and the refetched weather points equal the hashed files of 2026-11-19 on the hours they share. Where they do
   not, the stored values are used to 2026-10-10 and the difference is reported.
3. Coverage: no input missing on a whole day of the window unless it was missing in the same way a year earlier.
4. With the new data joined on, the stored forecasts of H, G, C and W on 2024-05-01 to 2026-10-10 reproduce to 1e-9.
5. Scramble on 2026-10-11, 2026-10-25, 2026-11-11, 2026-12-25 and 2027-01-18: inputs identical and forecasts of H, G and W
   unchanged to 1e-9, while C moves.
6. 100 days and at least 2,377 hours scored.

The code that joins the new data does not exist yet, because the French refetch is not yet specified. It will be written
before the data is fetched where possible, and in any case it is judged by conditions 1, 2 and 4: if the stored forecasts do
not reproduce, nothing is scored. If a condition fails, the cause goes into an amendment before anything else is read.

## Known limits

- Two hours of delivery day 2026-10-11 are present in the stored files (the French inputs file and the raw German files end
  at midnight UTC). They are cut on load and have not been used.
- From 2026-08-20 the French wind and solar proxies are reconstructed, as in the first test.
- The window is autumn and winter. W was designed on a backtest of all seasons and on a summer window.
- Recency weights and weekly refits make W follow the latest regime. If the old regime returns abruptly, W will lag in the
  same way G lagged in July, for a shorter time.
- There is no market price of the spread in the project, so no trading value is claimed.

## Review and what does not change

An independent reviewer who has not seen the build recomputes the decisions before anything is reported. Raw rows of the
inputs and prices are never printed or published. Nothing public changes on design figures. Logging W each morning would
change the scheduled task on the owner's laptop and needs his go-ahead; it is not part of this plan. Amendment 3 of the
forward test is unchanged: Q and P stay unscored on every live day.
