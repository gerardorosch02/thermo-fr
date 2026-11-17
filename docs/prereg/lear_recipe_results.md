# Results: the yardstick and LEAR to the published recipe

Pre-registration: `docs/prereg/lear_recipe_prereg.md` (sha256 dc0321976f7bcc1ed1fa9f69fe8bd52cf55a4ffd8527164b0f49e12b7474b39d,
commit 0459b68, written before any run). Run on 2026-11-16 and 2026-11-17 on branch `lear-recipe` at eb5ea6b. Independent
review: `docs/prereg/lear_recipe_review.md`. Machine-readable figures: `docs/prereg/lear_recipe_results.json`.

All errors are hourly mean absolute errors in EUR/MWh on the strict rows of the selection window, delivery days 2024-02-17
to 2026-06-30 (20,752 hours, 865 days). The holdout was not used.

## Outcome

No candidate passes the pre-registered rule. LEAR stays as it is and the public model is unchanged.

| | MAE | Gain over L0 | (i) gain at least 0.25 | (ii) no slice worse by more than 1.0 | (iii) one-sided 98.33% upper end below zero | Verdict |
|---|---|---|---|---|---|---|
| L0, the present LEAR | 15.3646 | | | | | baseline |
| A, windows 56, 84, 1092, 1456 | 15.3724 | -0.0078 | no | no (top 5% hours +2.92, windiest +1.56, sunniest +1.14) | no (+0.428) | fails |
| B, windows 56, 84, 364, 728, 1092, 1456 | 15.1098 | 0.2548 | yes | no (top 5% hours +2.04) | no (+0.036) | fails |
| C, B with the rolling asinh slope | 14.8420 | 0.5226 | yes | no (top 5% hours +1.37) | yes (-0.179) | fails |

Because no candidate passes there is no LEAR-R and no S1-R diagnostic.

Protocol slices:

| | Weekends | Holidays | Windiest 10% | Sunniest 10% | Top 5% hours | Negative-price hours (not in the rule) |
|---|---|---|---|---|---|---|
| L0 | 15.36 | 19.40 | 17.56 | 15.57 | 20.91 | 21.51 |
| A | 14.83 | 18.26 | 19.12 | 16.71 | 23.83 | 17.84 |
| B | 14.69 | 18.33 | 18.15 | 16.11 | 22.95 | 18.24 |
| C | 14.24 | 18.42 | 17.47 | 15.85 | 22.28 | 17.46 |

## Reproduction

honest_v2 15.7040 (logged 15.7040), L0 15.3646 (logged 15.36), S1 14.3608 (logged 14.36). All three reproduce.

## What the candidates do

- The short windows and the smaller slopes help where prices are low and hurt where they are high. C minus L0 by band of
  the actual price: below zero -4.05, 0 to 20 -2.52, 20 to 50 -1.47, 50 to 80 +0.68, 80 to 133 +0.97, above 133 +1.37.
  C is worse than L0 on every band above 50 EUR/MWh, which is 55% of the hours.
- The gain is early and the loss is recent. Paired daily difference against L0 (negative is better), two-sided 95%:

  | | Before 2025-10-01 (592 days) | From 2025-10-01 (273 days) |
  |---|---|---|
  | A | -0.379 [-0.895, +0.115] | +0.846 [+0.267, +1.445] |
  | B | -0.547 [-0.882, -0.211] | +0.379 [-0.030, +0.801] |
  | C | -0.895 [-1.296, -0.494] | +0.285 [-0.213, +0.794] |

  The loss sits in January to April 2026, so the quarter-hour auction date is a convenient split and not a shown cause.
- C's fail on condition (ii) is a fail by the rule, not a secure finding about its size. The difference on the top 5% of
  hours is +1.37 with a 95% interval of [+0.39, +2.48]; three days of 216 decide whether it is above 1.0. That C is worse
  than L0 on those hours is established. B's and A's fails on the slice are secure.
- Per-window MAE at slope 1: 56 days 18.12, 84 days 17.88, 364 days 16.20, 728 days 15.75, 1092 days 15.75, 1456 days 15.82.
  With six windows and one fixed slope: 0.1 14.96, 0.3 14.89, 0.5 14.98, 0.7 15.09, 1.0 15.11. The rolling choice picked 0.1
  on 41% of hours and 1.0 on 28%.

## Part 1, the yardstick (measurement only)

| | honest_v2 | L0 | S1 | LEAR-AR (prices and calendar only) |
|---|---|---|---|---|
| MAE | 15.70 | 15.36 | 14.36 | 17.55 |
| M1 rMAE against D-1 same hour (naive MAE 21.83) | 0.719 | 0.704 | 0.658 | 0.804 |
| M1 rMAE against D-7 same hour (naive MAE 31.39) | 0.500 | 0.489 | 0.457 | 0.559 |
| M1 rMAE against the Lago naive (naive MAE 24.49) | 0.641 | 0.627 | 0.586 | 0.717 |
| M2 slope, hourly | 0.905 | 0.889 | 0.982 | 0.854 |
| M2 slope, daily means | 0.860 | 0.916 | 0.978 | 0.860 |
| M4 mean Spearman rank correlation | 0.869 | 0.878 | 0.887 | 0.858 |
| M4 spread capture | 0.907 | 0.919 | 0.926 | 0.894 |
| M5 MAE before 2025-10-01 | 15.47 | 15.03 | 14.02 | 16.97 |
| M5 MAE from 2025-10-01 | 16.22 | 16.09 | 15.11 | 18.82 |
| M5 rMAE (Lago) before, from | 0.667, 0.593 | 0.648, 0.589 | 0.605, 0.553 | 0.732, 0.688 |

- M2. A slope above one would mean the forecast moves less than the price. Every slope is below one, so on the selection
  window none of the models under-reacts to the move away from yesterday's auction; S1 is close to one.
- M3. Members against S1, paired daily difference: honest_v2 +1.34 [+1.02, +1.69], L0 +1.00 [+0.71, +1.29].
- M4. Yesterday's profile used as a forecast gives a rank correlation of 0.757 and a spread capture of 0.793.
- M6. L0 against LEAR-AR: -2.19 [-2.60, -1.76]. The weather, load, neighbour and nuclear inputs are worth about 2.2.
- M7. Regression of S1's daily error on the wind and solar surprise, 787 complete days from 2024-05-01. Share of variance
  explained: pooled 3.1%, loosest tercile 1.6%, middle 4.1%, tightest 6.1%. Wind coefficient in EUR/MWh per GW: pooled
  -1.79, loosest -0.90, middle -2.85, tightest -2.63. The effect of a wind surprise roughly triples from loose to tight
  days, and the share explained doubles, but it stays small. No decision was registered to follow from M7.

## Checks

- Look-ahead scrambling test, 20 sampled days, every slope and window and the rolling choice: largest change 5.7e-14
  EUR/MWh. It passes on its sample.
- Independent review (2026-11-17): every figure recomputed with the reviewer's own code and matched; verdict "confirmed
  with corrections". The corrections are the three interpretation points above and the finding below.
- **Look-ahead on one delivery day, 2024-02-17, found by the reviewer.** The strict flag is set per hour, but LEAR reads
  the whole day's weather vectors. On that day the issued wind forecast is missing for hours 0 to 6, those hours carry
  weather that was not available at the gate, and the 17 scored hours of every LEAR-type forecast depend on it (L0, A, B,
  C and so S1; honest_v2 and LEAR-AR do not). The builder's scrambling sample did not contain the day. The same fault is
  in the 15.36 and 14.36 logged in October. Without the day (864 days, 20,735 hours): L0 15.3717, S1 14.3687, honest_v2
  15.7124; gains A -0.0087, B 0.2542, C 0.5222; top 5% differences +2.93, +2.05, +1.37; one-sided upper ends +0.435,
  +0.031, -0.185. Every verdict is unchanged. By the letter of the stop condition this is a stop-class finding; it is
  recorded here with both sets of figures and the owner decides whether to treat it as a stop with a formal rerun on 864
  days or as a logged correction. From now on the strict set for models that read day vectors starts on 2024-02-18 and the
  scrambling test must include the first strict day and the clock-change days by name.

## Departures from the pre-registration

None changes a verdict. Full list in the review.

- The scikit-learn change to the AIC noise variance came in version 1.0.2, not 1.1 as the text says. The rule coded is the
  old criterion either way.
- The shared Gram matrix speed-up changed L0 on 5 of 20 sampled days by 0.5 to 6.2 EUR/MWh and was not used. Every fit
  went through the original path.
- A minimum of 14 rows for a short window is in the code and not in the text. It never applied.
- Condition (ii) uses slice errors rounded to 0.01, as the sprint did. Unrounded differences give the same verdicts.
- The 1456-day window is not full until 2025-01-04 because the data start in 2021.
- The headline report was produced before the scrambling test had finished.

## Not checked

honest_v2 was not refitted by the reviewer. LEAR was refitted independently on 10 of 921 days. Whether the inputs hold
values as they stood at publication time cannot be tested from the file. A figure the reviewer computed after seeing the
results (long windows only with C's slopes) is exploratory and is not a result.
