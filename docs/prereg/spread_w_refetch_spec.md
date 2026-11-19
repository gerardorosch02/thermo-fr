# Test of W: where each input comes from after 10 October 2026, and what can be recovered as of 10:05 Paris

Written on 2026-11-19 at the owner's request, before any French or German value of a delivery day after 2026-10-10 is
fetched for the test. It belongs to `spread_w_prereg.md` and its amendment 1. The timing rules are the project's own
(`src/thermo_fr/forecast/timing.py`).

## The problem

The forecast of delivery day D is issued at 10:05 Paris on D-1. For a day forecast live, the inputs are whatever the
sources showed that morning. For a day rebuilt afterwards, the inputs are whatever the sources show on the day of the
refetch. Some sources keep only the latest version of a value, so a rebuilt day can contain a number that was revised after
10:05. This document says, input by input, which case applies and what is done about it.

## Three classes of delivery day

Every delivery day of the window gets exactly one class, recorded with the result.

- **Saved.** The forecasts of every comparator were written by the spread logger before the auction closed (12:00 Paris on
  D-1), with a time stamp, from inputs fetched that morning. The scored forecast is the saved one. It is never recomputed.
- **Rebuilt, French inputs saved.** No forecast was saved, but the existing forward logger saved the French input window
  before 10:05 Paris on D-1. The French inputs are taken from that window. The German inputs come from the refetch.
- **Rebuilt.** Every input comes from the refetch.

As of 2026-11-18 the copy of the forward logger's files in the workspace holds a French window saved by 10:05 Paris for 22
delivery days between 2026-10-20 and 2026-11-19. None exists for 11 to 19 October, for Sundays and Mondays (the logger runs on
weekday mornings), or for 2026-11-17 and 2026-11-20 (the laptop woke late). No German input and no spread forecast has been
saved on any morning so far. The first Saved day is the first morning the spread logger runs.

The full 100-day result uses every day. The numbers of days in each class are reported, and the errors are reported
separately for Saved days and for rebuilt days.

## Input by input

| Input of the forecast for D | Source | Public by | Does the source keep the 10:05 version? | Handling on a rebuilt day |
|---|---|---|---|---|
| French price of D-1 and earlier, and its lags | ENTSO-E day-ahead prices | about 13:00 Paris on D-2 | Final values, the same later | Refetched |
| Neighbours' prices of D-1 (DE-LU coupled, BE, NL, ES, IT North, CH) | ENTSO-E day-ahead prices | about 13:00 Paris on D-2 | Final values | Refetched |
| French load forecast of D | ENTSO-E day-ahead load forecast | 10:00 Paris on D-1, "updated when significant changes occur" | No. The API returns the latest version only | Saved window where one exists, otherwise refetched and counted as possibly revised |
| French nuclear output of D-2 and of the early hours of D-1 | ENTSO-E actual generation per type | within about an hour of each period | No. Later corrections replace the value | Saved window where one exists, otherwise refetched and counted as possibly revised |
| French weather at D: temperature, wind, radiation, and the wind and solar points | Open-Meteo previous-runs archive, lead day 2 | run of 18 UTC on D-2 at the latest | Yes, on the provider's word | Refetched from the archive. Compared with the saved window where one exists |
| French wind and solar proxies of D | The points above with weights refitted each month on actual generation to two days before the month | as the points | Points yes. The generation behind the weights is as now published | `forward_q.fill_proxy_gaps`, as in both earlier evaluations |
| Calendar | computed | always | n/a | computed |
| German coupled price of D-1 and the spread's lags | ENTSO-E day-ahead prices, sequence 1 | about 13:00 Paris on D-2 | Final values | Refetched |
| German load forecast of D-1 and D-7 | ENTSO-E day-ahead load forecast | 10:00 Paris on D-2 and D-8 | No, latest version only | Refetched and counted as possibly revised |
| German wind and solar forecasts of D-1 | ENTSO-E day-ahead generation forecast | 18:00 Paris on D-2 | No, latest version only | Refetched and counted as possibly revised |
| German wind and solar points of D, and of D-1 | Open-Meteo previous-runs archive, lead day 2 | run of 18 UTC on D-2 at the latest | Yes, on the provider's word | Refetched from the archive |
| German proxy weights | fitted each month on ENTSO-E's forecasts of past days, to two days before the month | n/a | Forecasts as now published | Refitted from the refetched forecasts |
| C only: ENTSO-E's German forecasts of D | ENTSO-E | after the gate | n/a | Refetched. C is not attainable and is never a Saved forecast |

The operator's French wind and solar forecast of D is not an input. It is published after the gate.

## Rules

1. **Stored values are never replaced.** Everything to 2026-10-10 stays as it is in the files of the first test. The stop
   conditions compare the refetch with those files and with the saved windows before any score.
2. **A saved value beats a refetched one.** Where an input of a day was saved before 10:05 Paris on the eve, the saved value
   is used for the forecast of that day. As a training row for later days the refetched value is used, the same for every
   model, because training rows have always been built from the latest published values.
3. **Missing values stay missing.** No fill, no interpolation, no carry-forward beyond what the frozen feature code already
   does. The models were trained with missing values and can forecast with them. An hour is left out of scoring only by the
   rule of the first test: no realised spread, no spread of the previous day at that hour, or no German proxy. Every hour
   left out is counted and reported with its reason.
4. **A whole day without an input stops the run**, unless the same input was missing in the same way a year earlier. The
   cause goes into an amendment before anything is scored.
5. **Revisions are measured, not guessed.** For every day that has both a saved French window and a refetched value, the
   difference between the two is reported for the load forecast, the nuclear output and the weather points: share of hours
   that differ and the size of the difference. This involves no price and is done before scoring. If more than 5% of hours
   of the load forecast or the nuclear output differ by more than 1% of the value, the test still runs, and the result for
   rebuilt days carries that figure beside it.
6. **Rebuilt forecasts are checked against saved ones.** For every Saved day the forecast is also rebuilt from the refetch,
   and the largest difference between saved and rebuilt forecasts is reported per comparator. The saved forecast is the one
   scored.

## The refetch itself

It runs once, after the window is complete, on the owner's laptop with his key, by commands written and committed
beforehand:

1. French inputs: the project's own `forecast-fetch` for 2021-01-01 to 2027-01-19, written to a new file. `inputs.csv` is
   not overwritten.
2. German prices and ENTSO-E forecasts: `scripts/fetch_spread_test_raw.py` with windows from 2026-09-11 to 2027-01-19.
3. German weather points: `scripts/fetch_german_points.py` to 2027-01-18.
4. The forward logger's saved French windows and the spread logger's records, copied as they are.

## What cannot be recovered at all

- The 10:05 version of ENTSO-E's French load forecast and nuclear output for any day without a saved window.
- The 10:05 version of ENTSO-E's German forecasts for any day before the spread logger starts.
- Proof that the Open-Meteo archive holds exactly what was served two days ahead. It is the provider's statement.

The French limits apply to H, HW, G and W alike. The German ones touch only the models that read the German block, G
and W, so on rebuilt days a revised German forecast could flatter those two against H and HW. Saved days are free of both,
which is why they are reported on their own.
