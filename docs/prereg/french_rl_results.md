# Results: French residual load in LEAR, honest and idealised, and what the neighbours add to it

Pre-registration: `docs/prereg/french_rl_prereg.md` (sha256 e1e34af24ce6ea213bae18af1c0c57959bc27cb149af8256ddc5c39bffd23101,
commit 16eaebc, written before any fit). Run on 2026-11-17 on branch `french-rl` at 3af161b, code committed before the
first fit. Independent review: `docs/prereg/french_rl_review.md`. Machine-readable figures:
`docs/prereg/french_rl_results.json`.

All errors are hourly mean absolute errors in EUR/MWh on strict rows. The honest candidates are scored on delivery days
2024-05-29 to 2026-06-30 (763 days, 18,312 hours), the idealised references on 2024-02-18 to 2026-06-30 (864 days,
20,735 hours). The holdout was not used.

## Outcome

- **P passes.** LEAR with the honest French residual load (load forecast minus the wind and solar proxies) is 0.31 better
  than the present LEAR. The pass is thin.
- **Q passes and is taken.** The same regressor with the operator's forecast as a training stand-in before the proxies
  exist is 0.66 better than the present LEAR. By the rule Q becomes the LEAR for a forward live log of at least 90 days,
  to be pre-registered separately. Nothing public changes.
- **The neighbours are worth building, by the rule.** With idealised timing the six neighbours lower the error by a
  further 0.43 once the French residual load is in. The review tempers this (see the cautions).

| | MAE | Gain over L0 | (i) gain at least 0.25 | (ii) no slice worse by more than 1.0 | (iii) one-sided 98.33% upper end below zero | Verdict |
|---|---|---|---|---|---|---|
| L0, the present LEAR | 15.4681 | | | | | baseline |
| P, honest, zero before the proxies exist | 15.1621 | 0.3060 | yes | yes (every slice better, least -0.19 on holidays) | yes (-0.149) | passes |
| Q, honest, stand-in before 2024-05-01 | 14.8121 | 0.6560 | yes | yes (every slice better, least -0.42 on the sunniest 10%) | yes (-0.451) | passes, taken |

Paired daily difference against L0 (negative is better), two-sided 95%: P -0.306 [-0.451, -0.161], better on 59.8% of
days; Q -0.656 [-0.868, -0.469], better on 59.0% of days.

Neighbours' decision, window 728, 864 days:

| | MAE | Against | Gain | Paired daily difference, 95% | One-sided 98.33% upper end |
|---|---|---|---|---|---|
| L0, window 728 | 15.7565 | | | | |
| I, six neighbours, idealised (earlier test) | 14.9561 | L0 | 0.8004 | -0.801 [-1.091, -0.536] | |
| F, French residual load, idealised | 14.7882 | L0 | 0.9683 | -0.968 [-1.229, -0.747] | |
| FI, F plus the six neighbours | 14.3561 | F | 0.4320 | -0.432 [-0.639, -0.228] | -0.214 |

FI is also 0.60 [0.44, 0.79] better than I. France and the neighbours each carry information the other lacks.

Protocol slices on the 763 days:

| | Weekends | Holidays | Windiest 10% | Sunniest 10% | Top 5% hours | Negative-price hours (not in the rule) |
|---|---|---|---|---|---|---|
| L0 | 15.28 | 19.84 | 17.91 | 15.86 | 21.36 | 20.94 |
| P | 15.00 | 19.65 | 17.31 | 15.56 | 20.01 | 21.42 |
| Q | 14.81 | 19.20 | 16.56 | 15.44 | 20.48 | 21.75 |

On all 864 days of the reference set, with each candidate equal to L0 before it has a forecast of its own under the
rule: L0 15.3717, P 15.1014, Q 14.7923.

Diagnostics with honest_v2 on the 763 days (not decision figures): S1 14.3953; S1-P 14.2635, -0.132 [-0.204, -0.064]
against S1; S1-Q 14.1338, -0.261 [-0.365, -0.170] against S1. About 40% of the gain in LEAR reaches the public average,
because honest_v2 already reads the proxies.

## Reproduction and checks

- Reference figures reproduce on the 864 days: honest_v2 15.7124, L0 15.3717, S1 14.3687.
- Same code path: with no added regressor, and with P's regressor left out, the fitting code matches the stored L0
  forecasts on the five registered days (largest difference 2.8e-14).
- Look-ahead: on the 20 registered days scrambling every input not known at the gate changes the forecasts by at most
  2.8e-14. For P and Q the operator's wind and solar forecast of D was scrambled too. It passes.
- Independent review (2026-11-17): all 492 values of the results file recomputed with the reviewer's own code, largest
  difference 3.6e-15. No look-ahead in 76 refits with perturbed inputs, including the operator's forecast on every day
  from 2024-05-01 for Q and on every day for P. The stored proxies are reproduced to 1e-11 MW by a walk-forward fit that
  ends two days before each month. Verdict "confirmed with cautions".
- P and Q are the same model where the training window holds no stand-in day. Their stored window-364 forecasts are
  identical on all 427 days from 2025-04-30, and their window-728 forecasts on the 63 days from 2026-04-29.

## What the candidates do

| | L0 | P | Q |
|---|---|---|---|
| MAE before 2025-10-01 | 15.12 | 15.01 | 14.65 |
| MAE from 2025-10-01 | 16.09 | 15.43 | 15.11 |
| 2024 H1 (33 days) | 18.09 | 17.78 | 17.95 |
| 2024 H2 | 14.83 | 14.97 | 14.50 |
| 2025 H1 | 15.35 | 15.13 | 14.69 |
| 2025 H2 | 13.44 | 13.08 | 12.68 |
| 2026 H1 | 17.82 | 17.04 | 16.86 |
| Price below 0 | 20.94 | 21.42 | 21.75 |
| Price 0 to 20 | 18.89 | 18.22 | 17.89 |
| Price 20 to 50 | 16.84 | 16.28 | 16.14 |
| Price 50 to 80 | 11.63 | 11.55 | 11.34 |
| Price 80 to 133 | 13.04 | 12.94 | 12.16 |
| Price above 133 | 20.75 | 19.62 | 19.82 |
| Under-reaction slope, hourly | 0.915 | 0.922 | 0.920 |
| Mean rank correlation | 0.886 | 0.891 | 0.891 |
| Spread capture | 0.925 | 0.927 | 0.928 |
| Window 364 | 16.22 | 15.68 | 15.45 |
| Window 728 | 15.79 | 15.51 | 15.21 |
| Window 1092 | 15.86 | 15.68 | 15.27 |
| Window 1456 | 16.00 | 15.88 | 15.36 |

- Share of hourly fits in which the lasso kept the regressor. P: 91%, 89%, 84% and 85% for windows 364, 728, 1092 and
  1456. Q on the 763 days: 97.5%, 99.8%, 100% and 100%. F 100%, M 92%. FI: France 99.9%, DE-LU 97%, BE 100%, NL 73%,
  ES 93%, IT-North 66%, CH 73%.
- Where the gain comes from at window 728 on the 763 days. L0 15.7930, P 15.5060, Q 15.2100, M 15.3728, F 14.7086,
  FI 14.2030. P captures 26% of F's gain and Q 54%. M against F is +0.664 [0.438, 0.901] and P against M +0.133
  [0.019, 0.252], so the short history of the honest series costs about five times as much as its lower quality. P
  against F is +0.797 [0.561, 1.039]. M at window 728 on the 864 days is 15.3854.
- Replaced values over the whole file. France: load forecast 0, wind proxy 0, solar proxy 0, operator's solar forecast
  157, operator's onshore forecast 167. Neighbours: 56, 0, 72, 528, 504 and 6 for DE-LU, BE, NL, ES, IT-North and CH. The
  operator's offshore forecast counts as zero in 22,750 hours, all before its first value on 2023-08-07 and none on a
  scored day. No forecast fell back to L0 from a candidate's first own day.

## Cautions from the review

All of these are exploratory and were not pre-registered.

- **Q's lead over P is a bridge that is already closing.** Q beats P only through the stand-in days, by construction.
  Those days left window 364 on 2025-04-30 and window 728 on 2026-04-29. In the last 63 days Q against P is -0.046
  [-0.508, +0.297]. From late 2026 the two are the same model in half of the average, and a forward log of 90 days will
  not tell them apart. The forward test is in effect a test of P with a long honest history.
- **The gain does not fade when the stand-in days go.** Where the window is all honest the gain is the largest seen. At
  window 364 from 2025-04-30 it is -0.913 [-1.263, -0.600] over 427 days. What has not been seen is windows 1092 and
  1456 with an all-honest history, which cannot exist before 2027 and 2028.
- **P's pass is thin.** Its gain is 0.2384 without its ten best days, below the bar of 0.25. It is absent in the first
  eleven months (-0.016 [-0.212, +0.220]) and negative in 2024 H2 (+0.138). It grows as the history grows, to -0.783
  in 2026 H1. Q gains 0.5532 without its ten best days and is better in nine of nine quarters.
- **Negative-price hours get worse.** On the 1,119 hours, P is +0.481 [-0.152, +1.195] against L0 and Q +0.806
  [-0.279, +1.976], and Q is +2.34 in 2026 H1. The intervals include zero, but the direction has held for the last
  two to three half-years and grows. In these hours Q's mean forecast is 3.61 below the actual price where L0's is 0.42
  above. In the public average the effect is smaller (S1-Q +0.320 [-0.194, +0.850]).
- **Q is weak where the window is nearly all stand-in.** A model trained on the operator's series is then fed the proxy,
  which sits on average 1,556 MW higher. At window 364 with more than 75% stand-in days Q is 0.299 worse than L0. This
  state does not recur.
- **The gain in the public average is at the bar, not above it.** S1-Q gains 0.2614 on S1, 0.2074 without ten days,
  with the right sign in every half-year.
- **"Worth building" for the neighbours is an idealised 0.43.** The honest French series captures 26% (P) to 54% (Q) of
  its idealised gain. The same fractions of 0.43 are 0.11 and 0.23, both below 0.25. FI against F is 0.3251 without
  ten days, and its interval includes zero in each of the last two half-years. The neighbours' estimates would start
  with the same short history unless a stand-in is used as in Q.
- **Re-use of the window.** The same days have now carried twelve pre-registered decisions. At a one-sided level of
  99.58% (Bonferroni over twelve) the upper ends are P -0.116, Q -0.404 and FI against F -0.166, all still below zero.
  A pass is a reason to test forward and not evidence of a live gain.
- **Unexplained.** At window 364, 434 hourly fits dropped the regressor between July and December 2025 against 16 in
  the other eight months from May 2025. The gain is smaller on those days (0.645 against 1.103). The cause was not
  found.

## Departures from the pre-registration

None changes a verdict.

- The results file gives Q on the 864 days as 14.8153, which uses Q's own forecasts on 2024-05-01 to 2024-05-28. As the
  text words it (equal to L0 before 2024-05-29) the figure is 14.7923, the one given above.
- Two listed items were missing from the results file and are given above from the reviewer's code: M at window 728 on
  the 864 days, and P against F as a paired difference.
- Q's kept shares in the results file cover the 791 days from 2024-05-01. The figures above are on the 763 days.
- Condition (ii) is evaluated on slice errors rounded to 0.01, the project's habit. The worst slices are -0.19 and
  -0.42 against a limit of +1.0.
- The count of offshore values set to zero was not in the results file.
- The look-ahead check covers P, Q and M on 18 of its 20 days, since two fall before 2024-05-01. It scrambles D and
  later, so it does not itself test that Q never reads the operator's forecast between 2024-05-01 and D-1. The
  reviewer's own test on real data covers that.
- The same-path and look-ahead stop conditions were enforced by hand and not by the report code. Both passed before the
  report was written.
- The text does not say what Q does with a training day from 2024-05-01 that lacks the honest value. It never occurs.

## Not checked

Whether the weather point forecasts behind the proxies are truly as issued two days ahead, and whether they would be in
hand at 10:05 in live running. The vintage of the realised generation used to calibrate the proxies, and of the load
forecast. L0, honest_v2, S1 and I were taken from stored frames and not refitted. 76 of the 8,847 day-window fits were
refitted by the reviewer. The cause of the dropped fits at window 364 in the second half of 2025. Whether the live
pipeline can produce Q as it stands, which a forward log needs.
