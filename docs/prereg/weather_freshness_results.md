# Fresher weather for the German wind proxy: results on development days

Read 2026-11-21 under `weather_freshness_prereg.md`. Development days 2024-05-01 to 2026-06-30, 18,982 scored hours, 791
days. Not a test of a model. Every figure below was recomputed by an independent reviewer with its own code, from the saved
forecasts and the fetched weather file, and agrees. Full figures in `weather_freshness_results.json`.

## Stop conditions

All held. Arm B reproduces the stored forecasts of G and W to 3e-14 and the stored proxy to 4e-12 MW. Both GFS tables are
unchanged for every day up to D when all later runs are replaced by noise. Every scored hour has every forecast in every
arm. The reviewer rebuilt both tables from the weather file and found them identical, with the permitted run behind every
value, on all five clock-change days.

## The registered reading

**Fresh against stale, same weather model, curve as built, for W: -0.232 EUR/MWh [-0.391, -0.079].**

The registration asked for a gain of at least 0.25 with the interval below zero. The interval is below zero. The gain is
0.232, under the bar and at the size of the numerical noise described in amendment 3 of the test of W. By the
registration's own wording the result is not distinguishable. Freshness alone did not earn a forward challenger.

## In MW, against the operator's day-ahead forecast

Mean operator forecast 14,693 MW.

| Arm | Error | Bias | Correlation | Hours 00-05 | 06-11 | 12-17 | 18-23 |
|---|---|---|---|---|---|---|---|
| B, stored blend | 3,886 | -2,798 | 0.946 | 4,043 | 3,575 | 3,923 | 4,003 |
| S, stale GFS | 4,740 | -3,175 | 0.911 | 5,579 | 4,367 | 4,012 | 5,004 |
| F, fresh GFS | 4,360 | -2,967 | 0.933 | 4,984 | 3,928 | 3,661 | 4,868 |
| S, fleet curve | 2,796 | -285 | 0.938 | 2,807 | 2,786 | 2,880 | 2,714 |
| F, fleet curve | 2,393 | -144 | 0.956 | 2,118 | 2,296 | 2,628 | 2,532 |

- The fresher run cuts the error by about 380 to 400 MW, 8% with the curve as built and 14% with the fleet curve.
- The gain is largest in the first hours of the day, where the fresher run is about a day newer, and smallest in the last,
  where it is about six hours newer. With the fleet curve it is 689, 490, 252 and 182 MW by block.
- GFS is a worse input than the stored blend. Stale for stale, its error is 4,740 against 3,886.

## In EUR/MWh, the hourly spread error

| Arm | G | W |
|---|---|---|
| B, stored blend | 18.473 | 17.339 |
| S, stale GFS | 18.718 | 17.545 |
| F, fresh GFS | 18.498 | 17.313 |
| S, fleet curve | 18.585 | 17.317 |
| F, fleet curve | 18.369 | 16.998 |

Paired daily differences, candidate minus base, with 95% intervals.

| Contrast | W | G |
|---|---|---|
| Fresh against stale, curve as built (registered primary for W) | -0.232 [-0.391, -0.079] | -0.220 [-0.387, -0.057] |
| Fresh against stale, fleet curve | -0.319 [-0.505, -0.146] | -0.217 [-0.395, -0.046] |
| Stale GFS against the stored blend | +0.205 [-0.048, +0.477] | +0.244 [-0.025, +0.541] |
| Fresh GFS against the stored blend | -0.027 [-0.298, +0.246] | +0.024 [-0.222, +0.283] |
| Fresh GFS with the fleet curve against the stored blend | -0.341 [-0.605, -0.086] | -0.105 [-0.360, +0.145] |
| Fleet curve against the curve as built, fresh GFS | -0.315 [-0.534, -0.122] | -0.129 [-0.322, +0.054] |

All but the first are descriptive.

## What it says

- **Freshness is real and small.** Within one weather model it lowers the spread error by 0.2 to 0.3 for both G and W,
  under either curve, with every interval below zero. That is consistent, and it is at the edge of what these models can
  resolve.
- **Against today's inputs, fresh GFS gains nothing.** Its freshness only makes up for being a coarser model than the
  blend: -0.03 for W, with an interval that spans zero.
- **The fleet-style curve again helps W by about 0.3**, here on a second weather source. On the stored blend the day
  before it gave -0.29. Its width was tuned on these days, so this stays descriptive.
- **The combination not tested is the one that matters:** the better weather model, fresh. Its past runs are not in this
  study.

## Departures from the registration

- **The arms' first month.** The registration said the GFS tables start on 2024-01-01 as the stored table does. The stored
  table has no value before 17 February 2024, so with complete GFS tables the GFS proxies began in April 2024, a month
  before the stored arm's. This was seen in the log of the first fits, before any forecast file was opened. The GFS tables
  were then blanked wherever the stored table has no value, 1,140 hours, none scored, and all GFS arms were fitted again.
  The reviewer could not verify that no figure was seen in between. No figure was computed.
- **Leads of the fresh arm** are 22 to 46 hours, not 22 to 47 as written.
- **The rule was applied by hand.** The code computed the contrast and did not store a verdict.
- **Two of the 18,982 hours** have no operator forecast, so the MW figures rest on 18,980.

## Limits

As registered: the proxy is judged against the operator's forecast and not against generation; these are days W's design
had in view; whether the midnight run was on hand on every past morning cannot be shown; the ensemble is not in this
study.
