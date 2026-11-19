# One out-of-sample test of the France to Germany spread forecast: result

Plan: `docs/prereg/spread_eval_prereg.md` (sha256 dd6cc46256aaa64b398ce5aab40b7ff194192436f6d8b8aa7bb2c7b080ba257d), frozen with
its code in commit b4111ab at 2026-11-19 16:30:10 UTC. The owner fetched the raw ENTSO-E files at 16:34:33 UTC. The stop
conditions ran at 16:41 UTC and the result was read once at 18:13 UTC, after a break in the session. Figures:
`spread_eval_results.json`. Stop record: `spread_eval_stop_checks.json`.

## Verdict by the frozen rule: fails

G is the spread forecast with the German block, H the same forecast without it. Window 2026-07-01 to 2026-10-10, 2,448 hours
on 102 days.

| Condition | Result | Met |
|---|---|---|
| Hourly error of G at least 0.25 below H | 16.72 against 18.53, a gain of 1.81 | yes |
| No slice worse by more than 1.0 | holidays +1.81, sunniest tenth of days +1.70 | no |
| One-sided 95% upper end of the paired daily difference below zero | -0.32 (mean -1.81, two-sided 95% interval -3.82 to -0.11) | yes |

The plan's reading for a fail: the backtest gain is not confirmed out of sample.

| Slice | Days | Hours | H | G | G minus H |
|---|---|---|---|---|---|
| Weekends | 29 | 696 | 16.05 | 15.37 | -0.69 |
| Holidays, French or German | 3 | 72 | 8.30 | 10.11 | +1.81 |
| Windiest tenth of days | 11 | 264 | 29.96 | 22.96 | -6.99 |
| Sunniest tenth of days | 11 | 264 | 13.34 | 15.04 | +1.70 |
| Top 5% of hours by absolute spread | 33 | 123 | 83.14 | 71.54 | -11.60 |

## Reported with the decision

| Hourly error, EUR/MWh | Yesterday | H | G | C (not attainable) |
|---|---|---|---|---|
| All 102 days | 20.47 | 18.53 | 16.72 | 15.79 |
| July, 31 days | 15.97 | 17.87 | 15.96 | 15.27 |
| August, 31 days | 15.85 | 14.97 | 14.58 | 13.61 |
| September, 30 days | 29.28 | 23.93 | 18.81 | 17.85 |
| October, 10 days | 22.29 | 15.41 | 19.44 | 17.93 |
| Before 20 August, 50 days | 13.58 | 15.27 | 14.18 | 13.40 |
| From 20 August, 52 days, French proxies reconstructed | 27.10 | 21.67 | 19.16 | 18.08 |
| Daily base spread | 13.52 | 13.20 | 10.06 | 9.24 |

- Direction of the day-on-day move of the base spread, on the 71 days with a move above 5: H 69%, G 80%, C 86%.
- G against yesterday's spread: -3.75, two-sided 95% interval -7.43 to +0.11.
- C against G: -0.93, interval -1.20 to -0.64. G recovers 66% of C's gain over H.
- The realised spread of the window averaged -5.0 with a mean absolute value of 14.5. On the development window those were
  -29.1 and 32.5, and on the same calendar window of 2024 and 2025 the mean was -24.7 and -32.5.

## How firm the outcome is

- The gain is about half the backtest's 3.69, and half or less of the same months of 2024 (2.85) and 2025 (4.07).
- G beat H on 55% of days. The mean daily difference is -1.81 and the median -0.77. Without the five best days the interval
  no longer excludes zero.
- By month the difference is -1.92, -0.40, -5.12 and, on the ten days of October, +4.03.
- Before 20 August neither H nor G beat yesterday's spread.
- The holiday slice rests on three days (14 July, 15 August, 3 October) and fails on 3 October alone. The sunny slice rests on
  eleven days and stays above 1.0 with any one of them removed. The slice condition uses point estimates with no allowance
  for the number of days. That is how the rule was frozen, and it is applied as frozen.
- H is biased by -5.24 on the window and G by -0.46, so part of G's gain is a correction of the level.

## Stop conditions, all passed before any error was computed

1. Refetched June equals stored June: German coupled price and the four forecasts equal on 100% of 720 hours, largest
   difference 0.0.
2. Coverage: no input missing on a whole day, 102 days.
3. Stored development forecasts reproduce with the new data joined on: largest difference 5.7e-14 on 18,982 hours.
4. Reference figures reproduce: 25.069, 22.163, 18.473, 17.388.
5. Scramble on five days of the window: inputs of H and G identical, forecasts differ by 0.0; C moved by 37 to 183.
6. 2,448 hours on 102 days.

## Review

An independent reviewer who had not seen the build or the result recomputed everything from the saved forecasts with its own
code: 99 figures, no mismatch, the same verdict. It rebuilt the realised spread from the raw XML with its own parser and
matched it on 100% of hours, and confirmed that the plan and code hashes match the log and predate the fetch by four
minutes (both time stamps are self-reported). Review verdict: confirmed with cautions. The cautions are the ones under
"How firm the outcome is".

## What follows

- The window is spent for H, G and C.
- Nothing public changes. No trading value is claimed in either direction: there is no market price of the spread here.
- Any variant gets a new name and is judged on later days only.

Correction to the registration row in `docs/experiments.md`: it says 13 unit tests passed. Ten of them belong to this module
and three to the coverage check.
