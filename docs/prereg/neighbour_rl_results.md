# Results: neighbours' day-ahead load and residual load in LEAR

Pre-registration: `docs/prereg/neighbour_rl_prereg.md` (sha256 ae9898edf4e9c196d62ec6649aa7d8f7fafcb5cc00c3a8eae728e08fe820fe1e,
commit 68c9f1c, written before any run). Run on 2026-11-17 on branch `neighbour-rl` at 8b56891. Independent review:
`docs/prereg/neighbour_rl_review.md`. Machine-readable figures: `docs/prereg/neighbour_rl_results.json`.

All errors are hourly mean absolute errors in EUR/MWh on the strict rows of delivery days 2024-02-18 to 2026-06-30
(864 days, 20,735 hours). The holdout was not used.

## Outcome

- **H, the neighbours' load forecast (honest timing), fails.** It changes nothing. LEAR stays as it is and the public model
  is unchanged.
- **I, the neighbours' residual load (idealised timing), meets the ceiling.** It cannot be adopted, because the operators'
  wind and solar forecasts are not available at the gate. By the pre-registered rule the route stays open and the next step
  is an honest version under its own pre-registration.

| | MAE | Gain over L0 | (i) gain at least 0.25 | (ii) no slice worse by more than 1.0 | (iii) one-sided 97.5% upper end below zero | Verdict |
|---|---|---|---|---|---|---|
| L0, the present LEAR | 15.3717 | | | | | baseline |
| H, neighbours' load forecast | 15.3622 | 0.0095 | no | yes (worst +0.12, top 5% hours) | no (+0.090) | fails |
| I, neighbours' residual load, idealised | 14.5671 | 0.8046 | yes | does not bind (every slice better) | yes (-0.568) | ceiling met |

Paired daily difference against L0 (negative is better), two-sided 95%: H -0.010 [-0.109, +0.090]; I -0.805 [-1.058, -0.568].

Protocol slices:

| | Weekends | Holidays | Windiest 10% | Sunniest 10% | Top 5% hours | Negative-price hours (not in the rule) |
|---|---|---|---|---|---|---|
| L0 | 15.39 | 19.40 | 17.56 | 15.57 | 20.93 | 21.51 |
| H | 15.40 | 19.00 | 17.28 | 15.43 | 21.05 | 21.40 |
| I | 14.17 | 18.50 | 15.95 | 15.00 | 19.01 | 20.43 |

Diagnostics with honest_v2 (not decision figures): S1 14.3687; S1-H 14.3720, +0.003 [-0.049, +0.059] against S1; S1-I
13.7860, -0.583 [-0.710, -0.463] against S1.

## Reproduction and checks

- Reference figures reproduce: honest_v2 15.7124, L0 15.3717, S1 14.3687.
- Same code path: with no added block the new fitting code matches the stored L0 forecasts on the five registered days
  (largest difference 2.8e-14).
- Look-ahead: on the 20 registered days, all windows and both candidates, scrambling every input not known at the gate
  changes the forecasts by at most 2.8e-14. It passes.
- Independent review (2026-11-17): 96 figures recomputed with the reviewer's own code, all matching to better than 1e-13;
  five further scramble days and seven refits from scratch agree. Verdict "confirmed", with the cautions below.

## What the candidates do

| | L0 | H | I |
|---|---|---|---|
| MAE before 2025-10-01 | 15.04 | 15.02 | 14.17 |
| MAE from 2025-10-01 | 16.09 | 16.09 | 15.42 |
| Price below 0 | 21.51 | 21.40 | 20.43 |
| Price 0 to 20 | 18.67 | 18.74 | 17.39 |
| Price 20 to 50 | 16.48 | 16.57 | 15.79 |
| Price 50 to 80 | 11.31 | 11.28 | 11.09 |
| Price 80 to 133 | 13.08 | 12.97 | 12.36 |
| Price above 133 | 20.82 | 20.94 | 18.93 |
| Under-reaction slope, hourly | 0.889 | 0.886 | 0.900 |
| Mean rank correlation | 0.878 | 0.875 | 0.883 |
| Spread capture | 0.919 | 0.917 | 0.923 |
| Window 364 | 16.20 | 16.15 | 15.38 |
| Window 728 | 15.76 | 15.76 | 14.96 |
| Window 1092 | 15.76 | 15.77 | 15.12 |
| Window 1456 | 15.83 | 15.84 | 15.08 |

- I is better than L0 in both periods, in every price band and in every window. It is better on 59.7% of days. The ten
  best days carry 17.6% of the gain, and without them the difference is -0.671 [-0.905, -0.446].
- Share of hourly fits in which the lasso kept a zone's regressor, all scored days. H: DE-LU 69%, BE 58%, NL 66%, ES 63%,
  IT-North 66%, CH 73%. I: DE-LU 93%, BE 99.6%, NL 71%, ES 92%, IT-North 61%, CH 70%. H's regressors are kept often and
  change nothing, so kept shares are a weak guide to which zones matter.
- Replaced values: H 50 (DE-LU); I 56, 0, 72, 528, 504 and 6 for DE-LU, BE, NL, ES, IT-North and CH. No scored hour fell
  back to L0.

## Cautions from the review

- **The idealised gain may be mostly timing, not neighbours.** H and I differ in two ways at once: I adds wind and solar,
  and it adds information published after the gate. In an exploratory control by the reviewer (not pre-registered, 116
  sampled days, window 364 only, not to be used for selection) a single French regressor built from the French operator's
  own post-gate wind and solar forecast lowered the error by 1.27 against 0.86 for the six neighbours on the same days.
  With the French regressor in, the neighbours added 0.21 [-0.67, +0.23]. On that evidence about three quarters of I's
  gain is fresher wind and solar information that is not specific to the neighbours.
- **The finding is about incremental information.** With idealised timing the neighbours' residual load is worth 0.80
  (5.2%). Their load forecast adds nothing to what L0 already holds (the French load forecast, the calendar and
  yesterday's Swiss and German prices).
- **Uneven in time (exploratory).** I's gain by half-year is -0.18 (2024 H1), 1.33, 1.27, 0.67 and 0.68, positive in 23 of
  29 months.
- The selection window has been used many times. Neither figure is evidence of a live gain.

## Departures from the pre-registration

None changes a verdict.

- The downloaded file ends at 23:00 UTC on 2026-06-30, so its last two rows are Paris hours 0 and 1 of holdout day
  2026-07-01 (forecast values only, no prices). The code drops them before use.
- The registered look-ahead test did not scramble the neighbours' wind and solar of D for H. H never reads them, and the
  reviewer's own test confirms it.
- The Dutch wind forecasts also drop by about a quarter from 2024, a level break the text does not list. The Swiss and
  Dutch solar breaks do not produce I's gain: with window 364 the gain is 0.91 on days trained only on data after the
  breaks, against 0.65 before.
- The run started two minutes before the code was committed. The committed code reproduces the stored forecasts.

## Not checked

The download itself and the actual publication time of any forecast. The source paper's figures. Whether a model was
fitted before the pre-registration commit, which file times support but cannot prove. Which zones carry I's gain (no
leave-one-zone-out refit). The French control beyond one window and 116 days.
