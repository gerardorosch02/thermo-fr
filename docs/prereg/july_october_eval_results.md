# One development evaluation on 1 July to 10 October 2026: results

Run on 2026-11-19 under `july_october_eval_prereg.md` (sha256
cf3f4b448c83d65ec5bded959909cc2603b34058a3807e5a358820abba5e26c6) and `july_october_eval_amendment_1.md`, approved by the
owner at 11:31 London the same day. Figures in `july_october_eval_results.json`.

**This is development evidence, not a holdout test.** The period's prices were known to the project, the tree model had
been scored on it before, and the question about the LEAR at high prices was formed afterwards. The period is now spent
for every model named here. Anything changed because of these results is to be judged on later days.

All prespecified results are reported together below, whatever they show.

## In one paragraph

On 102 days and 2,448 hours, at a mean price of 123 EUR/MWh, **the ensemble beat its tree member** (22.66 against 24.14,
supports). **The larger tree is not shown to be better** (23.56 against 24.14, undecided), and **the residual load step is
not shown to add anything** (22.35 against 22.66, undecided). **The LEAR diagnostic could not be tested**: no day of the
period reached the high band, so the pattern seen on the live days in November neither recurred nor failed to. A separate
finding came out of the stop conditions: the project's logged holdout figure for the tree model, 25.81, was measured with
the model missing three of its inputs for half the period.

## Stop conditions

| | Result |
|---|---|
| 1, as amended. T800 on the inputs as they are gives the logged 25.81 | 25.8101. Passes |
| 2. The code reproduces stored forecasts on the selection window | T800 June 2026 to 8.5e-14, T300 June 2026 exactly, L0 and Q on five June days to 5.7e-14. Passes |
| 3. Inputs not known at the gate replaced by noise on three days | 127,914, 66,426 and 6,402 values scrambled. Every forecast of L0, Q, T300 and T800 unchanged (0.0). Passes |
| 4. Hours and days | 2,448 strict hours on 102 days, every model present on every hour. Passes |

Condition 1 failed as first written (T800 on the filled history scores 23.56, not 25.81). The run stopped, the reason was
found, and amendment 1 was written before anything else was fitted or read. See "The logged holdout figure" below.

## The models on the period

Hourly mean absolute error in EUR/MWh, with the mean error (forecast minus price) in brackets.

| Model | Whole period | Without 1 to 10 October | 1 to 10 October |
|---|---|---|---|
| T300, tree at 300 trees of 31 leaves | 24.14 (-3.2) | 23.34 (-4.5) | 31.51 (+8.3) |
| T800, tree at 800 trees of 63 leaves | 23.56 (-4.2) | 23.10 (-5.3) | 27.80 (+6.4) |
| L0, the LEAR | 23.79 (-0.4) | 23.32 (-0.6) | 28.11 (+1.2) |
| Q, the LEAR with French residual load | 23.35 (-1.0) | 23.04 (-1.4) | 26.25 (+2.5) |
| S1 = (T300 + L0) / 2 | 22.66 (-1.8) | 22.13 (-2.5) | 27.52 (+4.7) |
| S1-Q = (T300 + Q) / 2 | 22.35 (-2.1) | 21.96 (-2.9) | 25.90 (+5.4) |
| (T800 + L0) / 2, not a registered comparison | 22.34 (-2.3) | 21.96 (-2.9) | 25.81 (+3.8) |

Mean price 123.4 on the whole period, 120.2 without the first ten days of October, 153.2 in them.

By window, the LEAR scores 25.41 (364 days), 23.74 (728), 24.39 (1092) and 23.79 (1456).

## The three comparisons

Candidate minus the model it is compared with, paired by delivery day. Two-sided 95% interval from the moving-block
bootstrap, then the one-sided upper end at level 1 - 0.05/3 that the reading rule uses.

| Comparison | Days | Difference | 95% interval | One-sided upper end | Days better | Reading |
|---|---|---|---|---|---|---|
| T800 against T300 | 102 | -0.58 | [-1.11, +0.43] | +0.48 | 54% | undecided |
| S1 against T300 | 102 | -1.48 | [-2.28, -0.43] | -0.38 | 63% | **supports** |
| S1-Q against S1 | 102 | -0.32 | [-0.65, +0.19] | +0.23 | 59% | undecided |

Without 1 to 10 October (92 days): T800 against T300 -0.24 [-1.06, +0.47], undecided. S1 against T300 -1.21
[-2.25, -0.30], supports. S1-Q against S1 -0.18 [-0.49, +0.30], undecided.

On 1 to 10 October alone (10 days) all three differences favour the candidate (-3.72, -3.99, -1.62) and the rule, applied
mechanically, prints "supports" for each. **Those three readings are set aside as too few days.** Ten days give the
bootstrap four possible blocks, the lower end of two intervals equals the point estimate, and the reviewer found that S1
against T300 there rests on one day (10 October contributes -29.6; without it the mean is -1.15). An exact sign-flip test
gives p = 0.22 for S1 against T300, 0.11 for S1-Q against S1 and 0.04 for T800 against T300.

Slices, difference in error on the hours of the slice:

| Slice | Hours | T800 minus T300 | S1 minus T300 | S1-Q minus S1 |
|---|---|---|---|---|
| Weekends | 696 | -0.67 | -2.91 | +0.30 |
| Holidays | 48 | +1.47 | -4.01 | +1.18 |
| Windiest tenth of days | 264 | -2.69 | -3.75 | -0.83 |
| Sunniest tenth of days | 264 | +0.31 | -1.42 | -0.01 |
| Top 5% of hours by price | 123 | -1.80 | -4.08 | -1.08 |

The windiest and sunniest days and the top hours are picked on realised outcomes.

Not registered, for information: (T800 + L0) / 2 against T800 -1.22 [-2.11, -0.37]. L0 against T300 -0.36
[-2.07, +1.61]. Q against L0 -0.43 [-0.91, +0.36].

## The LEAR diagnostic

**Verdict by the rule written in advance: too few days.** The distance ran from -0.47 to 1.58 over the period, with a
median of 0.68. No day reached the high band (1.6 or more), so there is no high-price spell in the period at all. The live
days of November, where the pattern was seen, had ten days above 1.6.

| Band | Days | Spells | Yesterday's mean price | Realised change | L0 predicted | T300 predicted | L0 minus T300 | S1 minus T300 |
|---|---|---|---|---|---|---|---|---|
| Low, below 1.0 | 76 | 12 | 107 | +5.7 | +5.9 | +0.5 | -1.26 [-3.32, +1.03] | -1.89 [-2.84, -0.72] |
| Middle, 1.0 to below 1.6 | 26 | 11 | 173 | -19.6 | -21.5 | -16.8 | +2.29 [-1.98, +5.76] | -0.30 [-2.98, +1.90] |
| High, 1.6 or more | 0 | 0 | | | | | | |

In the middle band the LEAR predicts a slightly larger fall than happened (by 1.9) and loses a little to the tree model,
with an interval that includes zero. That is the same direction as the live pattern and far weaker. The ensemble's
advantage over the tree model is in the low band and is about nil in the middle band.

The plain split: only 4 days follow a day averaging 200 or more, each on its own (15 and 29 September, 2 and 7 October).
On them the LEAR predicted a fall of 31.2 against a realised 25.8 and still beat the tree model by 5.2. Four isolated days
say nothing either way: the bootstrap returns a single value for them, one day (15 September) drives the row, and "without
the longest spell" is arbitrary when every spell is one day long.

So the period cannot confirm or refute the November finding. It does show that the LEAR and the ensemble worked as in the
backtest up to a previous-day level of about 200 and a distance of about 1.6.

## The logged holdout figure

The log's holdout figure for the tree model, 25.81 (row of 2026-10-10), was computed on an inputs file that holds no wind
or solar proxy from 2026-08-20 to 2026-10-10. For those 52 days the model was scored without the two proxies and the
residual load built from them.

| T800 | Inputs as they are | Proxies filled by the project's rule |
|---|---|---|
| Whole period | 25.81 | 23.56 |
| 1 July to 19 August | 21.25 | 21.25 |
| 20 August to 9 October | 29.46 | 25.35 |
| 1 to 10 October | 36.00 | 27.80 |

Every earlier decision that compared a candidate with 25.81 on the holdout used a tree model without its inputs. Which of
those decisions it could change has not been examined. The saved morning data of the live run does carry the proxies, with
one exception seen (the first saved window, 19 October, has no wind proxy).

## What this does and does not support

- The ensemble's advantage over its tree member held on 102 days outside the selection window: 1.48 EUR/MWh, 6.1%. The
  specification was fixed before these days were scored, but the period is not unseen by the project.
- It did not hold on the 18 live days of 20 October to 19 November, when prices averaged 207 and the distance was above 1.6
  on ten days. The July to October period contains no such day, so it cannot say whether that was the LEAR's weakness at
  high prices or 18 unlucky days.
- The larger tree was better by 0.58 with an interval that includes zero. Its backtest gain of 0.38 is neither confirmed
  nor contradicted.
- The residual load step was better by 0.32 with an interval that includes zero, the same size as its backtest gain of
  0.28. Neither confirmed nor contradicted.
- The inputs are uncorrected (EXAA blend for the German price, legacy radiation labels).

## Independent review

A reviewer that had not seen the code recomputed every figure from the forecast files: **confirmed with cautions**. 354
values compared, none different; all 88 interval ends agree. The three whole-period readings hold across 20 seeds and block
lengths of 1, 3, 7 and 14 days. It could not recompute the tree model on the inputs as they are (that needs a refit). Its
cautions, beyond those already written into the sections above:

- **The gains of Q and of the larger tree sit in the stretch where the proxies were filled.** This is the reviewer's split,
  not a registered one. Before 20 August (50 days): Q against L0 +0.26, S1-Q against S1 +0.26, T800 against T300 -0.06.
  From 20 August (52 days): -1.10 [-1.71, -0.44], -0.87 [-1.21, -0.51] and -1.09. S1 against T300 is -1.48 in both halves.
  The fill is the project's own point-in-time rule (weights refitted each month on generation up to two days before the
  month), which reproduces every stored proxy to 7e-12 MW, but its timing was not re-tested by this run's scramble check.
- **The ensemble's gain is uneven by month:** July -2.49, August +0.10, September -1.24, October -3.99.
- **Multiplicity.** The level 1 - 0.05/3 covers the three whole-period comparisons. The labels printed on sub-periods,
  extras and bands at the same level are descriptions, not tests.
- **Slices.** Actual wind and solar are missing for the last 14 hours of 10 October, so that day is ranked on 10 hours;
  without it S1 minus T300 on the windiest days goes from -3.75 to -1.61. The holiday slice is two days.
- **No ranking among the ensembles is supported.** (T800 + L0) / 2 against S1 is -0.32 [-0.64, +0.22].
- **Records.** The plan and the amendment give their times as "about 11:40" and "about 12:05" London; their commits are at
  11:27 and 11:37. The order of events is as stated: plan, approval, trees, amendment, then the LEAR. The hole in the
  proxies is 1,246 hours inside the period (20 August to 10 October).

## Departures from the plan

- Stop condition 1 was replaced by amendment 1 after it failed as written. Explained above.
- P was not fitted at all. The plan said it would be produced and not scored.
- The sub-period figures of T800 on the two versions of the inputs were read to find the reason for the failed stop
  condition, before the other models were fitted.
