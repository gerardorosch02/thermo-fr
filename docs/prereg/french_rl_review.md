# Independent review: French residual load in LEAR (candidates P, Q, F, FI, M)

Reviewed on 2026-11-17 against `docs/prereg/french_rl_prereg.md` (commit 16eaebc, sha256 e1e34af2...3101, unchanged since), the code of commit 3af161b (working tree clean) and the outputs in `/home/claude/work/french-rl/`. Every figure below comes from my own scripts in this folder (`r0_common.py` to `r9_last.py`), which read the chunk files and the baseline frame. I did not call `full_report`, `apply_rule`, `add_candidates` or the yardstick functions. The project's fitting functions were called only for the look-ahead refits, where they are the thing under test, and the project's bootstrap function once, to check exact agreement. No holdout row (delivery days from 2026-07-01) was loaded past the first filter, scored or fitted.

## Verdict

**Confirmed with cautions.** All 492 values in `results.json` reproduce (largest difference 3.6e-15), the rule was applied as written, P and Q pass, Q is taken, and the neighbours' condition is met. I found no look-ahead in P or Q in 76 refits, and the stored proxies are reproduced exactly by a walk-forward calibration that stops two days before each month. The cautions are about meaning, not arithmetic: Q's lead over P is a bridging effect of the stand-in days that has already vanished in the last 63 days (Q minus P -0.046 [-0.508, +0.297]), P's own pass is thin (gain 0.2384 without its ten best days), the gain in the public average S1 is 0.2614, negative-price hours get worse, and the neighbours' "worth building" rests on an idealised 0.4320 of which an honest build can expect only a part.

## Figures recomputed

Rows as I count them: 864 scored days (2024-02-18 to 2026-06-30), 20,735 strict hours, no non-strict row. Rule days: 763 (2024-05-29 to 2026-06-30), 18,312 hours. P, Q and M chunks hold 791 days each (from 2024-05-01), F and FI 864, no missing cell. My hourly columns equal the builder's `frame.csv` to 5.7e-14 on every row.

**Match count: 492 of 492 leaves of `results.json` (455 numeric, 37 flags and labels) agree with my computation, tolerance 1e-9, largest numeric difference 3.6e-15. There is no mismatch.** The leaf-by-leaf list is in `r2_compare.md`.

| Figure | Builder | Reviewer | Match |
|---|---|---|---|
| Reference, 864 days: honest_v2 / L0 / S1 | 15.7124 / 15.3717 / 14.3687 | same | yes |
| Rule days: MAE L0 / P / Q | 15.4681 / 15.1621 / 14.8121 | same | yes |
| Gain P / Q against L0 | 0.3060 / 0.6560 | same | yes |
| All 864 days: MAE P / Q (as the builder built them) | 15.1014 / 14.8153 | same | yes (see departure 1) |
| P paired daily difference, 95% interval | -0.3060 [-0.4508, -0.1611] | same | yes |
| P one-sided 98.33% upper end, share of days better | -0.1486, 0.5976 | same | yes |
| Q paired daily difference, 95% interval | -0.6557 [-0.8684, -0.4691] | same | yes |
| Q one-sided 98.33% upper end, share of days better | -0.4510, 0.5898 | same | yes |
| Slices L0 (weekends, holidays, windiest, sunniest, top 5%, negative) | 15.28, 19.84, 17.91, 15.86, 21.36, 20.94 | same | yes |
| Slices P | 15.00, 19.65, 17.31, 15.56, 20.01, 21.42 | same | yes |
| Slices Q | 14.81, 19.20, 16.56, 15.44, 20.48, 21.75 | same | yes |
| Slice differences P, unrounded (five protocol slices) | -0.2800, -0.1853, -0.5905, -0.3072, -1.3538 | same | yes |
| Slice differences Q, unrounded | -0.4650, -0.6424, -1.3437, -0.4192, -0.8786 | same | yes |
| Negative-price hours (1,119): L0 / P / Q | 20.9413 / 21.4221 / 21.7469 | same | yes |
| Per-window MAE L0 (364, 728, 1092, 1456) | 16.2155, 15.7930, 15.8644, 15.9984 | same | yes |
| Per-window MAE P | 15.6781, 15.5060, 15.6786, 15.8800 | same | yes |
| Per-window MAE Q | 15.4504, 15.2100, 15.2657, 15.3555 | same | yes |
| Before / from 2025-10-01: L0, P, Q | 15.1201 / 16.0928, 15.0127 / 15.4301, 14.6465 / 15.1093 | same | yes |
| Half-years, price bands, volatility terciles, M1, M2, M4 for L0, P, Q | see results.json | same | yes |
| Window 728, 864 days: L0 / I / F / FI | 15.7565 / 14.9561 / 14.7882 / 14.3561 | same | yes |
| F against L0 | -0.9681 [-1.2287, -0.7470] | same | yes |
| FI against F (the neighbours' decision), upper end, share better | -0.4324 [-0.6390, -0.2279], -0.2138, 0.5625 | same | yes |
| FI against I; I against L0 | -0.5998 [-0.7932, -0.4394]; -0.8008 [-1.0906, -0.5355] | same | yes |
| Window 728, rule days: L0 / P / Q / M / F / FI | 15.7930 / 15.5060 / 15.2100 / 15.3728 / 14.7086 / 14.2030 | same | yes |
| P, M, F against L0 at 728 | -0.2872, -0.4201, -1.0841 | same | yes |
| P against M; M against F | +0.1330 [0.0186, 0.2516]; +0.6639 [0.4384, 0.9005] | same | yes |
| Share of F's gain captured by P | 0.2647 | same | yes |
| S1 on rule days: honest_v2 / S1 / S1-P / S1-Q | 15.5424 / 14.3953 / 14.2635 / 14.1338 | same | yes |
| S1-P and S1-Q against S1 | -0.1317 [-0.2043, -0.0642]; -0.2613 [-0.3652, -0.1700] | same | yes |
| Kept shares P (four windows) | 0.9084, 0.8858, 0.8395, 0.8529 | same | yes |
| Kept shares Q | 0.9763, 0.9976, 1.0000, 1.0000 | same | yes |
| Kept shares F, M, FI (France and six zones) | 1.0000; 0.9195; 0.9991, 0.9670, 1.0000, 0.7348, 0.9263, 0.6567, 0.7292 | same | yes |
| Forecasts that fell back to L0 from each candidate's first day | 0 for all nine candidate-windows | 0 | yes |
| Hours equal to L0 by construction before 2024-05-01 (P, Q, M) | 1,751 per window | 1,751 | yes |
| P and M: first own day, own days, rule days without own forecast | 2024-05-29, 763, 0 | same | yes |
| Replaced values, France (load, wind proxy, solar proxy, solar forecast, onshore forecast) | 0, 0, 0, 157, 167 | same | yes |
| Replaced values, neighbours | 56, 0, 72, 528, 504, 6 | same | yes |

Figures the pre-registration or the task asks for that `results.json` does not hold, from my code only:

- M at window 728 on the 864 days: 15.3854.
- Q on the 864 days with L0 before 2024-05-29, as the text words it: 14.7923 (the file has 14.8153, see departure 1).
- On the 864 days, L0 / P / Q: weekends 15.3868 / 15.1397 / 14.9719, holidays 19.3962 / 19.2452 / 19.0229, windiest 17.5570 / 17.0520 / 16.3204, sunniest 15.5685 / 15.2640 / 15.1608, top 5% 20.9262 / 19.7658 / 19.9982, negative-price hours 21.5082 / 21.9339 / 22.3486.
- P against F at window 728, rule days: +0.7969 [0.5610, 1.0386].

About the bootstrap. My implementation is a separate loop, but with seed 0 it draws the same block starts as the project's function, so the interval ends agree to 0.0 for P, Q, FI against F and Q against P. That agreement shows the two codes compute the same thing and is not independent evidence. With my seeds 1 to 5 the one-sided upper ends lie in [-0.1523, -0.1460] for P, [-0.4588, -0.4469] for Q and [-0.2176, -0.2017] for FI against F. With blocks of 1, 14, 28 and 56 days (my seed 7): P -0.1808, -0.1280, -0.1112, -0.0907; Q -0.4626, -0.4524, -0.4714, -0.4790; FI against F -0.2262, -0.1972, -0.1946, -0.2317. The lag-1 autocorrelations of the daily differences are 0.14, 0.08 and 0.13. No variant brings an upper end to zero. P's is the one that moves towards it as the block grows.

The paired daily mean and the hourly gain differ in the fourth decimal (P 0.30600 and 0.30605, Q 0.6557 and 0.6560) because days of 23 and 25 hours weigh the same as others in the daily mean.

## Rule

- Days. 763 strict days from 2024-05-29, the first day on which P has 28 training days with the proxies. The kept files confirm it: P's regressor is left out on 2024-05-01 to 2024-05-28 in all four windows and used on every day after.
- Level. Three decisions, one-sided 98.33%. The code takes the upper end of the two-sided 96.67% percentile interval, which is the 98.33% quantile. Correct.
- P. (i) gain 0.3060 against 0.25, met. (ii) worst slice holidays, -0.19, met. (iii) upper end -0.1486, met. Passes.
- Q. (i) gain 0.6560, met. (ii) worst slice sunniest 10%, -0.42, met. (iii) upper end -0.4510, met. Passes.
- Taken. Both pass, Q has the lower MAE (14.8121 against 15.1621). Q is taken. Correct.
- Neighbours. FI against F at window 728 on the 864 days: gain 0.4320 against 0.25, upper end -0.2138, and an honest candidate passes. "Worth building" follows from the text. On the 763 rule days the same comparison gives -0.5062 with upper end -0.2867.
- Negative-price hours are not one of the five protocol slices, so their worsening does not enter condition (ii). That is as written.
- The reproduction stop condition is coded and met. The other two stop conditions are not enforced by the report stage. File times show both checks passed before the report was written.
- I obtain the same verdicts from my own code.

## Look-ahead

Reading. `fit_predict_day` takes training rows from D minus the window length to D-1 and predicts from the design row of D and the regressor row of D. `french_tables` builds the honest series from `load_fc_mw`, `wind_proxy_mw` and `solar_proxy_mw` only, and Q's series by overwriting the idealised series with the honest one on every day from 2024-05-01. The fill is a forward fill down the day axis, so it can copy only from earlier days, and for the three honest components it never fires (0 values). The scale of the regressor is `robust_scale(values[has])` over the training rows that have it, and the zero is written into training rows only. `selection_inputs` and `load_neighbours` cut the holdout before anything is built. I see no path from a value dated after D, from a price or realised generation of D, or from the operator's forecast from 2024-05-01, into P or Q.

Experiment. 76 refits of day-window pairs, each compared with the stored chunk value (six of them are controls that are meant to move), and one further fit opened for its coefficients. Unperturbed refits reproduce the chunks to 7.1e-15.

| Test | What was replaced by noise | Fits | Largest change |
|---|---|---|---|
| A, for (a) and (b) | Operator's solar, onshore and offshore forecast on every delivery day from 2024-05-01, D and all days before D included | 16 Q, 16 P, six days, all four windows on two of them | 1.4e-14 |
| B, for (b) | The same three columns on every day of the file from 2021 | 8 P | 1.4e-14 |
| B controls | Same | Q at 1456 on 2024-09-07 and at 364 on 2024-06-01 must move: 12.9 and 24.1 EUR/MWh. Q at 364 on 2026-01-25 must not: 1.4e-14 | as expected |
| C, for (c) | Every column on every day after D. On D every column except the load forecast, issued weather and the two proxies, so price, realised generation, nuclear, neighbours' prices, operator's forecast. Nuclear of D-1 from 08:00. Neighbours' file after D | 16 (P and Q, windows 728 and 1092, four days of my own draw) | 2.8e-14 |
| D, for (c) | Realised wind and solar generation on every day of the file | 6 (P and Q) | 2.8e-14 |
| Power | Wind proxy of D raised 20%: P moves 1.57, Q 4.78. Wind proxy of D-10 raised 50%: P moves 0.42 | 3 | as expected |
| E, for (e) | Nothing. My own day-by-hour table, my own median and MAD over training days only, my own zero for days without the series, my own lasso call | 3 P, 2 Q, including P on 2024-06-01 at window 728 where 31 of 728 training days have the series | 1.4e-14 |

So: (a) Q does not read the operator's forecast for D or for any day from 2024-05-01. (b) P does not read it at all. (c) Neither reads anything dated after D, nor realised generation or prices of D. (e) Scaling and zero-imputation use training days only, and my independent version gives the stored forecasts.

(d) The proxies in `inputs.csv`. They are built by `gen_proxy.rolling_proxy`: non-negative least squares of realised generation on point forecasts, refitted each month on the 365 days ending two days before the month. I recomputed both with my own code from the point columns and the realised generation in the file. Under the documented rule the stored values are reproduced to 1.1e-11 MW (wind, 18,984 hours) and 7.3e-12 MW (solar, 19,704 hours). With a lag of one day instead of two, 18,855 wind hours differ (mean 9.0 MW). With a fit on the month itself the mean difference is 90 MW. The stored proxies are therefore the two-day-lag walk-forward and not an in-sample fit. The first wind month, 2024-05, was fitted on 73 days ending 2024-04-29, the first solar month on 71 days ending 2024-03-30. The 26 and 27 monthly fits recorded in `sources.json` up to 2026-06 all end on month start minus two. The timing rule (realised generation public within an hour, weather run of D-2 at 18 UTC counted available six hours later) passes the gate and the 10:05 issue time.

What I cannot establish from the repository, and say so plainly:

- That the point columns (`wind_pt_*`, `solar_pt_*`) really are forecasts issued two days ahead. The code requests Open-Meteo's `previous_day2` variables and its docstring says the meaning was checked against single runs. I could not use the network, and the archive carries no issue time.
- That the realised generation used for calibration is the vintage that was public at the time. The file was fetched on 2026-10-10, so any later revision by the operator is in it. The effect on 365-day least-squares weights is likely small.
- That the load forecast held by the archive is the version of 10:00 on D-1. L0 shares this, so it is not differential.

The builder's checks.

- `same_path_check.json` tests what the text says: on the five named days the code with no regressor, and P with its regressor left out, both give the stored L0 windows (largest 2.8e-14). The leaving-out is forced by an unreachable minimum, not by the real 28-day rule. I closed that from the chunks: P's stored forecasts on 2024-05-01 to 2024-05-28, where the real rule left the regressor out, equal the stored L0 windows to 3.6e-15 in all four windows, and M's likewise.
- `scramble_check.json` tests what the text says: 20 days (six named, 14 drawn with seed 4), largest change 2.8e-14, the operator's forecast of D scrambled for P and Q and kept for F, FI and M. Two limits. Two of the 20 days (2024-02-18, 2024-04-27) precede 2024-05-01, so P, Q and M were tested on 18. And it scrambles D and later only, so it does not test the sentence "Q never reads the operator's forecast for any day from 2024-05-01" for the days between 2024-05-01 and D-1. Only a synthetic unit test covered that. My test A covers it on real data. The check also treats the proxies of D as known, so it says nothing on (d).

## Consistency check of P and Q at window 364

It holds exactly. The first delivery day whose 364 training days all lie on or after 2024-05-01 is 2025-04-30. On the 427 days from then the stored P and Q forecasts at window 364 are identical in every hour (largest difference 0.0) and the kept counts are equal. On the 336 rule days before, they differ on every day (median largest hourly difference 14.8 EUR/MWh), and the last differing day is 2025-04-29. The same holds at window 728 from 2026-04-29 (63 days, 0.0), and M equals F at window 728 from 2026-04-29 (0.0). Windows 1092 and 1456 never reach this state inside the test (2027-04-28 and 2028-04-26).

## Departures

None changes a verdict.

1. Q on all 864 days. The text says "candidate equal to L0 before 2024-05-29". The file's 14.8153 uses Q's own forecasts on 2024-05-01 to 2024-05-28. As written it is 14.7923. P is unaffected (15.1014) because the 28-day rule makes it L0 there. The rule uses the 763 days only.
2. M at window 728 on the 864 days is listed among the reported items and is missing from `results.json`. It is 15.3854.
3. P against F at window 728 is named as a diagnostic and is reported only through the share of gain (0.2647). The paired difference is +0.7969 [0.5610, 1.0386].
4. Kept shares are asked "on the 763 days". Q's are over the 791 days from 2024-05-01. On the 763 days they are 0.9754, 0.9975, 1.0000, 1.0000 against 0.9763, 0.9976, 1.0000, 1.0000. F's and FI's are over 864 days.
5. Condition (ii) is evaluated on slice MAEs rounded to 0.01 and the difference rounded again, the project's habit. It could matter only within 0.01 of the limit of 1.0. The worst slices here are -0.19 and -0.42.
6. Counts of replaced values. The offshore forecast is set to zero where missing, as the text allows, but no count is given: 22,750 hours, all before its first value on 2023-08-07, none on a scored day. The reported counts cover the whole file from 2021. On scored days the operator's solar and onshore forecasts are missing in 26 hours each, which touches F, FI and M only.
7. The look-ahead check covers P, Q and M on 18 of the 20 days (see above). This follows from the wording "every stored forecast" and is a limit, not a breach.
8. The stop conditions for the same-path and look-ahead checks are procedural. The report stage would have run after a failed check. Both passed first.
9. A case the text does not cover: a training day from 2024-05-01 without the honest value would be dropped from Q's fit. It never occurs (791 complete days).

In line with the text: the series and their components, the fill rule (my own tables give the same counts), the scaling over all 24 hours of the training window, P's zero and 28-day minimum, Q's switch at 2024-05-01, the four windows averaged for P and Q, window 728 for F, FI and M, the stored L0 and I as comparators, the block bootstrap settings, the gate assertions for the load forecast and both proxies at both deadlines, the five same-path days, the 20 look-ahead days.

Order of events, from commits, file times and logs: pre-registration committed 16:26:58, code committed 16:30:25, same-path check written 16:33:19, first chunk written 16:36:02 (its fits began about 16:33:28), last chunk 21:34:47, look-ahead check 21:48:42, report 21:49:09. The code commit precedes the first fit. The logs show 292 chunks, each written once, 8,847 day-window fits. The eight unit tests of `tests/test_french_rl.py` pass.

## Exploratory robustness

Nothing in this section was pre-registered. Intervals are 95% moving-block bootstrap, block 7, unless stated.

Without the ten best days for the candidate, and breadth.

| Comparison | All days | Without ten best days: paired mean, interval | Upper end 98.33% | Hourly gain | Share of gain in ten days | Share of days better | Quarters better |
|---|---|---|---|---|---|---|---|
| P against L0 | -0.3060 | -0.2383 [-0.3704, -0.1127] | -0.0978 | 0.2384 | 23.1% | 0.5976 | 7 of 9 |
| Q against L0 | -0.6557 | -0.5528 [-0.7440, -0.3842] | -0.3673 | 0.5532 | 16.8% | 0.5898 | 9 of 9 |
| FI against F | -0.4324 | -0.3255 [-0.5183, -0.1373] | -0.1173 | 0.3251 | 25.6% | 0.5625 | 9 of 10 |

P without its ten best days no longer meets condition (i) (0.2384 against 0.25). Seven of those ten days are in 2026 H1. Q and the neighbours' comparison keep both conditions.

By half-year, paired daily difference.

| Half-year | Days | P against L0 | Q against L0 | Q against P | FI against F (864 days) |
|---|---|---|---|---|---|
| 2024 H1 | 33 (134 for FI) | -0.309 [-0.578, +0.017] | -0.138 [-0.774, +0.502] | +0.171 [-0.553, +0.891] | +0.260 [-0.212, +0.829] |
| 2024 H2 | 184 | +0.138 [-0.237, +0.489] | -0.332 [-0.681, +0.005] | -0.470 [-0.970, +0.025] | -0.878 [-1.315, -0.510] |
| 2025 H1 | 181 | -0.220 [-0.374, -0.061] | -0.662 [-0.934, -0.336] | -0.443 [-0.723, -0.120] | -0.741 [-1.093, -0.246] |
| 2025 H2 | 184 | -0.365 [-0.588, -0.192] | -0.761 [-1.149, -0.349] | -0.396 [-0.632, -0.098] | -0.230 [-0.594, +0.154] |
| 2026 H1 | 181 | -0.783 [-1.141, -0.503] | -0.965 [-1.551, -0.493] | -0.182 [-0.469, +0.085] | -0.388 [-0.823, +0.030] |

P is worse than L0 in 2024 H2 and its gain grows as its history grows. Q is better in every half-year. The neighbours' gain sits in 2024 H2 and 2025 H1, and in the last two half-years its interval includes zero.

By window, rule days, each against L0's same window.

| Window | L0 | P | Q | P against L0 | Q against L0 | Q against P |
|---|---|---|---|---|---|---|
| 364 | 16.2155 | 15.6781 | 15.4504 | -0.537 [-0.831, -0.249] | -0.765 [-1.035, -0.528] | -0.228 [-0.526, +0.005] |
| 728 | 15.7930 | 15.5060 | 15.2100 | -0.287 [-0.470, -0.111] | -0.583 [-0.818, -0.384] | -0.296 [-0.497, -0.118] |
| 1092 | 15.8644 | 15.6786 | 15.2657 | -0.186 [-0.288, -0.081] | -0.598 [-0.791, -0.427] | -0.412 [-0.601, -0.246] |
| 1456 | 15.9984 | 15.8800 | 15.3555 | -0.119 [-0.205, -0.027] | -0.643 [-0.850, -0.452] | -0.524 [-0.728, -0.332] |

Does Q's lead over P depend on the stand-in days? Yes, entirely, and by construction: where a window holds no stand-in day, P and Q are the same model. The stand-in days leave window 364 on 2025-04-30 and window 728 on 2026-04-29, which gives three periods.

| Period | Days | L0 | P | Q | P against L0 | Q against L0 | Q against P |
|---|---|---|---|---|---|---|---|
| A, 2024-05-29 to 2025-04-29, stand-in days in all windows | 336 | 15.1657 | 15.1498 | 14.7073 | -0.016 [-0.212, +0.220] | -0.458 [-0.713, -0.221] | -0.442 [-0.798, -0.141] |
| B, 2025-04-30 to 2026-04-28, window 364 all honest | 364 | 15.2143 | 14.6928 | 14.3757 | -0.522 [-0.708, -0.334] | -0.839 [-1.152, -0.525] | -0.317 [-0.507, -0.125] |
| C, 2026-04-29 to 2026-06-30, windows 364 and 728 all honest | 63 | 18.5475 | 17.9386 | 17.8928 | -0.609 [-1.203, -0.276] | -0.655 [-1.660, -0.041] | -0.046 [-0.508, +0.297] |

Q minus P in hourly MAE by window (364, 728, 1092, 1456): period A -0.517, -0.189, -0.359, -0.364; period B 0, -0.446, -0.494, -0.680; period C 0, 0, -0.232, -0.484. Q's lead is exactly zero wherever the stand-in days have gone and is carried by the two longest windows at the end.

Pooled over day-window pairs by the share of stand-in days in the training window, difference from L0's same window:

| Stand-in share | Pairs | Q | P |
|---|---|---|---|
| 0 (same model) | 490 | -0.853 | -0.853 |
| up to 25% | 273 | -1.035 | -0.552 |
| 25 to 50% | 581 | -0.869 | -0.194 |
| 50 to 75% | 910 | -0.617 | -0.159 |
| 75 to 100% | 798 | -0.261 | -0.044 |

What this implies for a forward test in late 2026 and 2027.

- Q and P will be the same model in windows 364 and 728, that is in half of the average. The stand-in share of window 1092 will be 13.6% on 2026-12-01 and zero from 2027-04-28. That of window 1456 will be 35.2% on 2026-12-01, 20.7% on 2027-06-30, 8.0% on 2027-12-31 and zero from 2028-04-26.
- The forward test is therefore in effect a test of P with a long honest history. The nearest thing to it inside the window is period C, where Q against P is -0.046 [-0.508, +0.297]. A forward log of 90 days will not be able to tell Q from P, and the choice between them matters less each month.
- I find no sign that the gain over L0 falls when the stand-in days go. Where the window is all honest the gain is the largest in the table: at window 364 from 2025-04-30 it is -0.913 [-1.263, -0.600] over 427 days (upper end -0.569), and at window 728 in period C it is -0.441 on 63 days. In period C the four-window gains are P -0.609 (upper end -0.242) and Q -0.655 (upper end +0.027, 63 days).
- What has not been seen is windows 1092 and 1456 with an all-honest history. That evidence cannot exist before 2027 and 2028.
- Q is weakest where the window is nearly all stand-in. At window 364 with a stand-in share above 75% (63 days) it is 0.299 worse than L0, and on its 28 own days before the rule start (2024-05-01 to 2024-05-28) it scores 15.2590 against L0's 14.5505. A likely reason is a level break in Q's regressor: on the days both exist the honest series is on average 1,556 MW above the operator-based one (mean absolute difference 2,081 MW, correlation 0.981), because the wind proxy runs 1,083 MW (19%) and the solar proxy 474 MW (14%) below the operator's forecasts. A model trained on the operator's series is then fed the proxy. Scored from Q's first own day (791 days) Q still gains 0.6077 with upper end -0.4167, so the rule start did not make the pass.

Negative-price hours. They get worse under both candidates. On the 1,119 hours (205 days) of the rule days, by a day-block bootstrap of the ratio:

| Against | Candidate | Difference | 95% interval | 2025 H2 (150 h) | 2026 H1 (407 h) |
|---|---|---|---|---|---|
| L0 (20.9413) | P | +0.4808 | [-0.1516, +1.1945] | +0.586 | +1.484 |
| L0 (20.9413) | Q | +0.8056 | [-0.2790, +1.9757] | +2.015 | +2.338 |
| S1 (16.89) | S1-P | +0.2023 | [-0.1000, +0.5373] | +0.168 | +0.518 |
| S1 (16.89) | S1-Q | +0.3198 | [-0.1941, +0.8497] | +0.874 | +0.765 |

The intervals include zero, but the direction is the same in the last three half-years for P and the last two for Q, and it grows. In these hours Q's mean forecast is 3.61 below the actual price where L0's is 0.42 above, so Q overshoots downwards. On the 864 days the figures are L0 21.5082, P 21.9339, Q 22.3486.

## Other points

- Multiple comparisons and re-use of the window. The level of 98.33% corrects for the three decisions of this file. The same 864 days have now carried twelve pre-registered decisions (four, three, two and three) besides exploratory work, and the idealised French regressor had been seen on 116 of these days before this file was written, which the text declares. At a one-sided level of 99.58% (Bonferroni over twelve) the upper ends are P -0.1158, Q -0.4044, FI against F -0.1663. At 99.9% they are -0.0709, -0.3593 and -0.1482. All stay below zero. The text's own sentence stands: a pass is a reason to test forward and not evidence of a live gain.
- P's pass is thin. Its gain is 0.3060 against a threshold of 0.25, 0.2384 without ten days, absent in period A (-0.016) and negative in 2024 H2. P passing does not matter to the verdicts, because Q passes and the neighbours' condition needs only one.
- The public average. S1-Q gains 0.2614 on S1 (14.3953 to 14.1338), which is 40% of the gain in LEAR, with interval [-0.3652, -0.1700] and a one-sided 98.33% upper end of -0.1612. Without its ten best days it is 0.2074. S1-P gains 0.1317. By half-year S1-Q is -0.170, -0.134, -0.211, -0.297, -0.421, and in period C -0.243. So the gain survives in S1 with the right sign in every half-year, but at about the size of the project's usual threshold and not above it with any margin. honest_v2 already reads the proxies, as the text warns.
- Kept shares. Q at windows 1092 and 1456, and F, keep the regressor in every one of their hourly fits. This is not suspicious and not informative either. In one fit I opened (Q, 2025-08-22, window 1092) the lasso keeps on average 73 of 303 regressors per hour (26 to 102), the residual load has the largest absolute coefficient in the median hour and is positive in all 24. P's shares rise with its history (window 364: 0.336 in 2024 H1, 0.850 in 2024 H2, 0.989 in 2025 H1). One oddity: at window 364, where P and Q are by then the same all-honest model, 434 hourly fits dropped the regressor between July and December 2025 (31 to 93 a month) against 16 in the other eight months from May 2025. The gain at window 364 is 0.645 on the 177 days with a dropped hour and 1.103 on the 250 days without. I did not find the cause.
- The neighbours' decision is an idealised one. Three things should temper "worth building". First, the honest French series captures 26% of the idealised French gain at window 728 (P) or 54% (Q, with stand-in days). The same fractions of 0.4320 are 0.11 and 0.23, both below 0.25. Second, the short history costs more than the proxy quality: M against F is +0.6639 and P against M +0.1330, and the neighbours' own estimates would start with the same short history unless a stand-in is used as in Q. Third, the gain has thinned in the last year (table above). The earlier review's control on 116 days found -0.21 [-0.67, +0.23] for the neighbours added to France. The full run finds twice that with a secure interval, so that control understated it.
- Q's design mixes two series in one regressor, scaled together. It works as a bridge and is honest, but any reader should know that the coefficient is learned largely on a better forecast than the one fed on day D, and that this is where its early weakness comes from.
- The first proxy months rest on little data (73 and 71 days). Honest, but P's and Q's regressor of mid-2024 is the weakest version of itself.

## Not checked

- Whether the weather point forecasts behind the proxies, and the issued weather in L0, are truly as issued two days ahead, and whether they would be in hand at 10:05 in live running. Needs the network and the provider's archive.
- The vintage of the realised generation used in the proxy calibration, and of the load forecast.
- L0, honest_v2, S1 and I were taken from the stored frames, not refitted. L0 is supported by the same-path check and by the 112 day-windows of P that equal it on 2024-05-01 to 2024-05-28.
- 76 of the 8,847 day-window fits were refitted. The rest are trusted on the strength of those and of the exact P and Q agreement at window 364.
- Look-ahead of F, FI and M beyond the builder's own check. They are idealised by declaration and never adopted. I checked only that M equals F where it should.
- The neighbours' series themselves (publication times, level breaks). See the earlier review.
- The cause of the dropped fits at window 364 in the second half of 2025.
- The hourly profile of the gain, and any effect by weekday.
- Whether Q can be produced by the live pipeline as it stands. A forward log needs the proxy of D as computed at 10:05 on D-1, not the archive value.
- Nothing was computed on, or fitted for, any delivery day from 2026-07-01.
