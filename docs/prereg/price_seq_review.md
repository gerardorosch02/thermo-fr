# Independent review of the price-series measurement (branch price-seq)

Reviewed on 2026-11-18. All figures below were recomputed with my own pandas and numpy code in this folder
(`a_ident.py`, `b_inputs.py`, `cd_scores.py`, `f_refit.py`, `f2_g2.py`). I read the project code for definitions only.
The two refits in part F call the project's live entry point, as asked. LightGBM was not run.

**Verdict: confirmed with cautions.**

Every claimed number reproduces to the precision given. The verdict "small" under the pre-registered rule is correct, and
so is "not worth its own registration". There are two literal disagreements, both immaterial, and several cautions about
what the measurement can and cannot support.

## Holdout

No forecast was scored after 2026-06-30. The frame and the forecast files end on 2026-06-30 and my scripts assert it.

One disclosure. My first pass of part A grouped the price comparisons by calendar year and its 2026 row ran to
2026-10-01, so it printed aggregate shares of hours in which neighbour prices equal the French price using three months
of holdout-period prices. No forecast was involved and nothing was tuned on it, but it is holdout-period data and I should
have cut it first. The script now cuts at 2026-06-30 22:00 UTC and every 2026 figure in this review and in `review.json`
is from the cut version. Part B compares the two input files over their full length, which is needed to confirm that no
other column changed and involves no price comparison with France.

## A. Identification

Years are UTC calendar years. "Equal" means within 1e-9 unless a tolerance is given.

### A1. Stored German price against the two sequences

| Year | Stored equals sequence 2 (claimed 100%) | Stored equals sequence 1 hourly (claimed never) | Mean absolute difference, stored against sequence 1 (claimed) | Mine | Agree |
|---|---|---|---|---|---|
| 2021 | 100% of 8,760 h | 6 hours | 7.9 | 7.937 | yes |
| 2022 | 100% of 8,760 h | 3 hours | 19.6 | 19.560 | yes |
| 2023 | 100% of 8,760 h | 3 hours | 8.8 | 8.834 | yes |
| 2024 | 100% of 8,784 h | 0 hours | 8.4 | 8.381 | yes |

The stored price first differs from sequence 2 at 2025-09-30 22:00 UTC, the first hour of delivery day 2025-10-01. From
then to 2026-06-30 the stored price equals sequence 1 in 10% of hours, sequence 2 in 12%, and lies between the two in 78%,
which fits a blend. The correlation of the two sequences is 0.953 to 0.979 by year, as the pre-registration says.

Small disagreement: "never equals sequence 1" is not literally true. The stored price equals hourly sequence 1 exactly in
12 of 35,064 hours of 2021 to 2024 (0.03%), which is chance agreement of two prices quoted to the cent. It changes nothing.

### A2. Share of hours equal to the French price within 0.011

| Year | Sequence 1 claimed | Sequence 1 mine | Sequence 2 claimed | Sequence 2 mine | Agree |
|---|---|---|---|---|---|
| 2021 | 49.2% | 49.25% | 0.0 to 0.2% | 0.23% | yes |
| 2022 | 34.3% | 34.29% | 0.0 to 0.2% | 0.03% | yes |
| 2023 | 29.4% | 29.39% | 0.0 to 0.2% | 0.05% | yes |
| 2024 | 14.2% | 14.21% | 0.0 to 0.2% | 0.10% | yes |
| 2025 (hourly part, to 30 Sep) | 12.8% | 12.78% | 0.0 to 0.2% | 0.10% | yes |
| 2025-10-01 to 2026-06-30 (quarter-hourly sequence 1) | not claimed | 5.17% | not claimed | 0.05% | n/a |

Is sequence 1 the coupled day-ahead auction? I agree, and I see no other plausible reading. Two prices can only agree to
the cent in 13% to 49% of hours if the same algorithm set both, and a falling share over the years is what a coupled pair
shows as the border binds more often. Sequence 1 also changes from hourly to quarter-hourly exactly at delivery day
2025-10-01 and has no overlap between its two resolutions, which is the behaviour of one auction changing its time unit.
The evidence is thinner after October 2025 (5.2% against 49% in 2021), but sequence 1 still matches France about a hundred
times as often as sequence 2 does.

What sequence 2 is remains open, as the pre-registration says. It is some other quarter-hourly German price. That matters
for interpretation in part G.

### A3. Spain

| Claim | Mine | Agree |
|---|---|---|
| Stored equals the unnumbered series on 100% of hours, 2021 to 2024 | 100% in each year | yes |
| First difference at 2025-09-04 22:00 UTC | 2025-09-04 22:00 UTC | yes |
| Unnumbered series equals the French price in 27% to 37% of hours | 34.8%, 26.7%, 32.8%, 32.7%, 36.4% for 2021 to 2025; 32.5% from 2025-10-01 to 2026-06-30 | yes |
| Numbered series about 1% | 1.55%, 0.73%, 0.76% (to 2026-06-30) | yes |

The third numbered Spanish series only has values for the second half of each delivery day (UTC hours 10 to 22, half the
hour count of the others). That pattern suggests the numbered series are intraday auctions rather than day-ahead prices.
This is my inference from the file, not something I could establish.

## B. Corrected inputs

| Check | Claimed | Mine | Agree |
|---|---|---|---|
| Same index and columns (50,616 rows, 61 columns) | yes | yes | yes |
| Hours changed, price_de_lu_eur_mwh | 49,571 | 49,571 by exact comparison; 49,506 differ by more than 1e-9 | yes |
| Hours changed, price_es_eur_mwh | 9,024 | 9,024 by exact comparison; 9,003 differ by more than 1e-9 | yes |
| New DE-LU value equals sequence 1 (hourly, then quarter-hourly) wherever it exists | yes | maximum difference 0.0 over 50,398 covered hours | yes |
| New Spanish value equals the unnumbered series wherever it exists | yes | maximum difference 0.0 over 50,398 covered hours | yes |
| Hours not covered by the sequences file are left as stored | implied | yes, 218 hours, all from 2026-10-01 22:00 UTC, none before the holdout | yes |
| No other column changed | yes | not literally, see below | no (immaterial) |

Disagreement: when both files are read from disk with the project's loader, a third column differs. `wind_proxy_mw` is
different in 11 hours by at most 5.7e-14 (relative size 3e-16). This is the text round trip of the file, not an edit. The
author's stop check compared the two tables in memory before the corrected file was written, so it could not see what the
refits later read from disk. The effect on any forecast is nil for practical purposes, but the stop check does not test
the file that was actually used and should compare the reloaded file in future.

Of the changed hours, 47,532 (DE-LU) and 6,888 (Spain) fall before the holdout. The remainder never enter a fit because
the selection inputs stop at 2026-06-30.

## C. Tree model on all days

Strict rows of the frame joined to the two folders of tree-model files. All 20,735 frame rows are strict and matched.

| Quantity | Claimed | Mine | Agree |
|---|---|---|---|
| Research, 864 days, stored | 15.7124 | 15.71237 | yes |
| Research, 864 days, corrected | 15.8397 | 15.83965 | yes |
| Research, 763 days, stored | 15.5424 | 15.54243 | yes |
| Research, 763 days, corrected | 15.6616 | 15.66156 | yes |
| Research, paired daily difference | +0.119 | +0.11919 | yes |
| Research, 95% interval | [+0.012, +0.229] | [+0.016, +0.229] my code and seed; [+0.0120, +0.2286] with the registered seed | yes, materially the same |
| Production, 763 days, stored | 15.9199 | 15.91989 | yes |
| Production, 763 days, corrected | 15.8912 | 15.89118 | yes |
| Production, paired daily difference | -0.029 | -0.02864 | yes |
| Production, 95% interval | [-0.116, +0.054] | [-0.117, +0.052] my code and seed; [-0.1159, +0.0542] with the registered seed | yes, materially the same |

The reference figures also reproduce (honest_v2 15.7124, L0 15.3717, S1 14.3687 on 864 days).

## D. The 191 sampled days

The sample is every fourth of the 763 consecutive development days starting 2024-05-29 and ending 2026-06-28. There are
191 files, each for the right delivery day, 24 strict hours each, 4,584 hours in total. Weekdays are balanced (27 or 28 of
each).

| Quantity | Claimed | Mine | Agree |
|---|---|---|---|
| Stored: L0 | 15.8124 | 15.81243 | yes |
| Stored: Q | 14.9456 | 14.94559 | yes |
| Stored: tree model, production | 15.7791 | 15.77912 | yes |
| Stored: S1 | 14.5946 | 14.59464 | yes |
| Stored: S1_Q | 14.2798 | 14.27978 | yes |
| Stored: gain of S1_Q over S1 | 0.3149 | 0.31487 | yes |
| Corrected: L0 | 15.7118 | 15.71179 | yes |
| Corrected: Q | 14.8787 | 14.87872 | yes |
| Corrected: tree model | 15.7215 | 15.72147 | yes |
| Corrected: S1 | 14.5424 | 14.54235 | yes |
| Corrected: S1_Q | 14.2385 | 14.23846 | yes |
| Corrected: gain | 0.3039 | 0.30389 | yes |
| Change of gain | -0.0110 | -0.01097 | yes |
| Paired S1_Q minus S1, corrected | -0.304 | -0.30389 | yes |
| Its 95% interval | [-0.535, -0.137] | [-0.539, -0.140] mine; [-0.5353, -0.1368] registered seed | yes, materially the same |
| Its one-sided 95% upper end | -0.164 | -0.171 mine; -0.1644 registered seed | yes, materially the same |
| Corrected minus stored, L0 | -0.101 [-0.239, +0.043] | -0.1006 [-0.233, +0.043] | yes |
| Corrected minus stored, Q | -0.067 | -0.0669 [-0.184, +0.089] | yes |
| Corrected minus stored, S1 | -0.052 [-0.144, +0.045] | -0.0523 [-0.143, +0.047] | yes |
| Corrected minus stored, S1_Q | -0.041 | -0.0413 [-0.124, +0.060] | yes |
| Mean absolute change of forecast, L0 | 1.63 | 1.629 | yes |
| Mean absolute change of forecast, Q | 1.61 | 1.609 | yes |
| Mean absolute change of forecast, tree model | 1.68 | 1.678 | yes |

"Mine" intervals use my own bootstrap code with a different seed. With the registered seed and the documented algorithm
written out again in my own code, the author's intervals reproduce to every printed digit.

## E. The rule

| Condition | Value | Holds |
|---|---|---|
| 1. On corrected inputs S1_Q beats S1 at production settings, and the one-sided 95% upper end is below zero | gain 0.3039, upper end -0.164 (mine -0.171) | yes |
| 2. The gain differs from the stored-input gain by less than 0.10 | change -0.0110 | yes |

"Small" is the correct verdict. Condition 1 holds for every block length I tried from 1 to 14 (upper ends from -0.159 to
-0.189). Condition 2 holds with a wide margin. My interval for the change of gain is [-0.047, +0.016].

"Not worth its own registration" is correct. S1 falls by 0.052 on the sample, the tree model at production settings falls
by 0.029 on 763 days (0.058 on the sample), and the tree model at research settings rises by 0.119. Nothing falls by 0.25.

The verdict also holds at research settings on the sample, which was not asked for. The gain is 0.286 on stored and 0.255
on corrected inputs, with a one-sided upper end of -0.12.

## F. One refit

2025-06-14 is not a sampled day (it is day 381 of the development period, and the sample takes multiples of 4). The
nearest sampled day is 2025-06-13.

| Check on 2025-06-13 | Largest absolute difference |
|---|---|
| Four-window L0 on corrected inputs against `L0_corrected` in the lear file | 1.4e-14 |
| 364-day window run alone against the same window inside the four-window run | 0.0 |
| Four-window L0 on stored inputs against L0 in the frame | 1.4e-14 |
| 364-day window on stored inputs against `lear_w364` in the frame | 1.4e-14 |

For information, the single 364-day window differs from the four-window average by up to 7.65 EUR/MWh (3.03 on average)
that day, so the comparison is sensitive enough to tell the two apart. The refit took 4 seconds for one window and 13 for
four.

I added a second day in the blend period, 2026-02-12, with both L0 and Q. Corrected inputs against the lear file and
stored inputs against the frame all agree to 7.1e-15.

## G. Could the design mislead?

**Is the sample representative?** Close enough for the rule, less so for the size of the correction's effect.

| Quantity, stored inputs, production settings | Sample (191) | All 763 days | The other three every-fourth-day samples |
|---|---|---|---|
| Gain of S1_Q over S1 | 0.315 | 0.285 | 0.323, 0.218, 0.283 |
| L0 | 15.81 | 15.47 | 15.31, 15.24, 15.51 |
| Gain of Q over L0 | 0.867 | 0.656 | 0.620, 0.523, 0.614 |
| Tree model, corrected minus stored | -0.058 | -0.029 | +0.038, 0.000, -0.096 |

The claimed 0.285 for all days reproduces (0.2848). The sample's S1 gain is inside the spread of the four possible
samples. The sample does hold harder days for L0 than average, and Q's gain over L0 alone is clearly higher on it than on
any of the other three. For the tree model, where all four samples can be scored, the effect of the correction moves by
about 0.07 either way between samples. So the "corrected minus stored" figures for L0, Q, S1 and S1_Q (all between -0.04
and -0.10) cannot be told from zero on one sample, which the intervals already say. The change of gain, the quantity the
rule uses, is far more stable than either gain.

**Blocks of 7 on every fourth day.** Sound, and as registered. A block spans 28 calendar days. The sampled daily
differences are almost uncorrelated (-0.05, 0.09, 0.16 at lags 1 to 3; the full daily series has 0.11 at lag 1 and 0.02
at lag 4), so the block length hardly matters and blocks of 7 are mildly conservative. One thing to know about the main
result: the gain is concentrated. The mean daily difference is -0.304, the median is -0.171, 58% of days are better, and
without the 10 best days of 191 the mean is -0.120. No single day decides it (leaving one day out gives -0.277 to -0.321).

**Look-ahead.** I found none. L0 and Q read the German price as a block shifted by one delivery day, the tree model reads
the same hour of D-1, and the live entry point blanks everything published after 10:05 Paris on D-1. The coupled price of
D-1 is fixed about 13:00 on D-2. Whatever sequence 2 is, its D-1 value was also public by then, so the stored backtest had
no look-ahead from it either. The alignment of the sequences file is confirmed by the exact match of its sequence 2 and
its unnumbered Spanish series with the stored columns in 2021 to 2024.

**Fetch-to-fetch instability.** The measurement says nothing about it, as the pre-registration states. The stored blend
is one draw. The live forecasts will read other draws, and a corrected history was not what the live path reads. The only
indirect evidence is that Q's advantage is about the same whichever German series is used, which makes it likely to
survive a blend too. That is an inference, not a measurement.

**Further cautions.**

1. The period that resembles the forward test is the weakest part of the sample. From 2025-10-01 there are 68 sampled
   days. The gain is 0.443 stored and 0.434 corrected, and the one-sided upper end on corrected inputs is -0.118, so
   condition 1 would still hold there, but the two-sided interval is wide, [-0.97, -0.06]. In that period the correction
   lowers every error by about 0.1 to 0.2 (L0 -0.18 [-0.31, -0.03], S1 -0.12 [-0.24, -0.04], S1_Q -0.11 [-0.19, -0.04],
   tree model -0.25 [-0.66, +0.02] on the sample and -0.09 [-0.28, +0.07] on all 273 days). Before October 2025 it does
   almost nothing for S1 (-0.01). So "under 0.25 overall" hides a cost of the blend of roughly 0.1 EUR/MWh that is visible
   after October 2025. This split is mine, made after seeing the results, on few days. Treat it as a pointer.
2. The corrected series is not plainly better for the tree model. At research settings the error rises by 0.119 with an
   interval that excludes zero (+0.145 before October 2025), while at production settings it does not move. Sequence 2
   seems to carry information of its own. A later correction that swaps the series may lose something that adding the
   coupled price beside it would keep. The sign depending on the settings is also a warning about how firm any tree-model
   conclusion is.
3. The tree-model comparison assumes a LightGBM refit is repeatable. The stored-input fits at production settings come
   from an earlier run (forward-q) and the research ones from the frame. If refits are not bit-identical from run to run,
   part of the 1.6 to 2.3 mean absolute change is refit noise. I could not test this.
4. Condition 1 is mostly a second look at an advantage already known on stored inputs, so "small" was the likely outcome
   in advance. It is still legitimately met.
5. Timing of the pre-registration. Git shows the pre-registration committed at 17:00:25, the code at 17:01:34, the
   corrected inputs written at 17:01:38 and the first fit output at 17:02:21. That order is consistent with "before any
   fit". The margins are a minute or so and I can only go by the recorded times.

## What I could not check

- The LightGBM fits. The four tree-model columns are taken as given.
- How `neighbour_sequences.csv.gz` was built from the raw ENTSO-E files, which are not on this machine. The match with
  the stored columns supports the method for sequence 2 and the Spanish unnumbered series only.
- Which markets DE-LU sequence 2 and the numbered Spanish series are.
- Whether any series, sequence 1 included, is stable from one fetch to the next.
- 189 of the 191 LEAR refits. Two were refitted and agree to 1e-14.

## Disagreements, in full

1. A1. "Never equals sequence 1" is 12 hours of 35,064 in fact. Immaterial.
2. B. "No other column changed" does not hold for the file on disk. `wind_proxy_mw` differs in 11 hours by at most
   5.7e-14. Immaterial, but the stop check ran before the file was written and so did not test the file the fits read.

No claimed score, difference, interval or count disagrees.

**Verdict: confirmed with cautions.**
