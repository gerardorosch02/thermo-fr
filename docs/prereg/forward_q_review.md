# Independent review: checks before the forward test of Q, and the risk band

Reviewed on 2026-11-18. Repository `/home/claude/thermo-fr`, branch `forward-q`. Every figure below was recomputed with
my own pandas, numpy and scipy code in `/home/claude/work/review` (scripts `ab.py`, `c_existing.py`, `c_bands.py`,
`c_scores.py`, `c_lookahead.py`, `d_vintage.py`, `d_extra.py`, `e_stop.py`, `f_replay.py`, `g_revision.py`,
`x_power.py`). The project's scoring functions were not used for any number checked. Machine-readable figures are in
`review.json`.

**Verdict: confirmed with cautions.**

Rules kept. No data after 2026-06-30 was used for scoring. The saved windows and the store were used only for the
vintage checks of part D and for forecast-against-forecast comparisons. The tables `actuals`, `scores`, `shadow_scores`
and `shadow_errors` were never opened. LightGBM was not run. Nothing was written outside `/home/claude/work/review`.

The repository changed while I worked. Commit 8ffd403 (amendment 1, 15:41 UTC) landed after I had started, then
`forward_q_checks.py` gained an uncommitted `replay-filled` stage. I reviewed the code as it stood at the end, with
`forward_q_checks.py` still modified and uncommitted.

## A. Reference figures

| Figure | Claimed | Recomputed | Agree | Note |
|---|---|---|---|---|
| 864 days, honest_v2 | 15.7124 | 15.7124 | yes | 20,735 strict hours, 2024-02-18 to 2026-06-30 |
| 864 days, L0 | 15.3717 | 15.3717 | yes | |
| 864 days, S1 | 14.3687 | 14.3687 | yes | S1 equals (honest_v2 + L0)/2 to 1e-13 |
| 763 days, L0 | 15.4681 | 15.4681 | yes | 18,312 hours |
| 763 days, P | 15.1621 | 15.1621 | yes | |
| 763 days, Q | 14.8121 | 14.8121 | yes | Q equals the mean of its four windows to 1e-13 |
| 763 days, S1 | 14.3953 | 14.3953 | yes | |
| 763 days, S1_Q | 14.1338 | 14.1338 | yes | |
| Reduction S1_Q against S1 | 1.82% | 1.82% | yes | |

The two frames (lear-recipe and french-rl) hold the same actual, honest_v2, L0 and S1 on every common row.

## B. Production settings

| Figure | Claimed | Recomputed | Agree | Note |
|---|---|---|---|---|
| 864 days, honest_v2_prod | 16.0914 | 16.0914 | yes | |
| 864 days, S1_prod | 14.5693 | 14.5693 | yes | |
| 763 days, honest_v2_prod | 15.9199 | 15.9199 | yes | |
| 763 days, S1_prod | 14.6234 | 14.6234 | yes | |
| 763 days, S1_Q_prod | 14.3386 | 14.3386 | yes | |
| Gain | 0.2848 (1.95%) | 0.2848 (1.95%) | yes | |
| Daily paired, S1_Q_prod minus S1_prod | -0.2846 [-0.389, -0.191] | -0.2846 [-0.391, -0.194] | yes | own bootstrap, other seed. Materially the same |
| Daily paired, S1_Q_prod minus honest_v2_prod | -1.581 [-1.947, -1.294] | -1.581 [-1.939, -1.277] | yes | materially the same |
| Daily paired, honest_v2_prod minus honest_v2 | +0.378 [+0.139, +0.619] | +0.378 [+0.129, +0.625] | yes | materially the same |

The main interval excludes zero for blocks of 1, 3, 7, 14 and 28 days. I could not check the honest_v2_prod forecasts
themselves, because that needs LightGBM.

## C. Risk band

| Figure | Claimed | Recomputed | Agree | Note |
|---|---|---|---|---|
| Scored set | 16,368 hours, 682 days | 16,368 hours, 682 days | yes | no row dropped |
| Band B rebuilt from the frame against bands.csv | same band | largest difference 5.7e-14 | yes | all 17,855 rows with a band, five quantiles, S1 and honest_v2 |
| Band T rebuilt with scipy genpareto against bands.csv | same band | largest difference 4.7e-12 | yes | S1 on all 724 days with a band (asked for 5). honest_v2 on 7 days, 1.7e-13 |
| Tail parameters of S1 | tail_params_S1.csv | equal to 1e-16 | yes | my own optimiser reaches the same likelihood on 7 days |
| B 80% coverage, width, score | 0.7801, 43.50, 67.97 | 0.7801, 43.50, 67.97 | yes | |
| B 95% | 0.9334, 69.75, 103.73 | 0.9334, 69.75, 103.73 | yes | |
| T 80% | 0.7993, 47.42, 69.75 | 0.7993, 47.42, 69.75 | yes | |
| T 95% | 0.9453, 79.05, 106.36 | 0.9453, 79.05, 106.36 | yes | |
| Paired daily score at 80%, T minus B | +1.776, ends about +0.55 and +2.96 | +1.776, ends +0.53 and +3.04 | yes | other seeds give +0.55 and +3.02. Materially the same |
| Existing quantile models, same hours | 0.7992, 48.67, 70.29 | 0.7992, 48.67, 70.29 | yes | identical from existing_ready.csv and from the parquet |
| Holidays, 80% | B 0.679, T 0.733 | B 0.679, T 0.733 | yes | 480 hours, 20 days |
| First half of 2026, 80% | B 0.727, T 0.787 | B 0.727, T 0.787 | yes | 4,343 hours |
| Decision rule | T does not replace B, condition 1 fails | condition 1 fails, 2 and 3 hold, B stays | yes | windiest tenth of days B 0.7065, T 0.7168. Same outcome for honest_v2 |
| Look-ahead, B and T | none | none | yes | see below |
| Original quantile run, parquet | 4.881, 0.8018, 49.75 | 4.8812, 0.8018, 49.755 | yes | 20,752 strict hours |
| Its benchmark | 5.105, 0.7872, 48.35 | 5.1054, 0.7872, 48.351 | yes | |
| q10, q50, q90 of existing_ready.csv against the parquet | below 1e-12 | 8.5e-14 | yes | all 20,759 common rows |

Look-ahead test. By construction both bands read only rows before D. By perturbation, through the project's own band
functions, I added noise to the errors of D and of every later day for D = 2024-10-27, 2025-06-15 and 2026-03-29. The
bands of every day up to and including D did not move at all (0.0, no change in missing values). The band of D+1 moved
by 3 to 12 EUR/MWh, so the test has power. Changing D-1 alone moves the band of D, which the pre-registration allows.

## D. Data vintage, from the saved windows

| Figure | Claimed | Recomputed | Agree | Note |
|---|---|---|---|---|
| Load forecast, days | 21, all differ | 21, all differ | yes | 504 of 504 hours differ |
| Mean absolute difference | 718 MW, 1.37% | 717.7 MW, 1.371% | yes | |
| Worst day, worst hour | about 2,150 and 4,860 MW | 2,153 MW (2026-11-11) and 4,863 MW | yes | |
| Weather of D, 41 series | 0 of 21,648 changed | 0 of 21,648 | yes | |
| Weather of D+1 seen on D-1 | 54% changed | 53.9% (11,683 of 21,689) | yes | |
| DE-LU price | 87% of hours, 5.7 | 87.4% of 935 hours, 5.70 | yes | |
| Spanish price | 98%, 6.7 | 98.0% of 911 hours, 6.68 | yes | |
| French price, Swiss price | 0%, 0% | 0%, 0% | yes | |
| Nuclear | 1.2% of hours, about 4 MW | 1.22% of 899 hours, 4.33 MW | yes | |
| revision_vectors.csv | 21 vectors | rebuilt, equal to 2e-16 | yes | |

## E. Stop conditions

| Figure | Claimed | Recomputed | Agree | Note |
|---|---|---|---|---|
| same_path.json | 5.7e-14, passes | file says 5.68e-14, passes | yes | five registered days |
| scramble.json | 0.0 | file says 0.0, passes | yes | |
| Own refit, 2025-03-30, window 364, L0 | equals lear_w364 | largest difference 7.1e-15 | yes | 23 clock hours |
| Own refit, 2025-03-30, window 364, Q | equals Q_w364 | largest difference 0.0 | yes | regressor kept in 24 of 24 hourly models. Q differs from L0 by up to 31.7, so the comparison is not trivial |

Review of `as_known_q` and `fit_known`. I found no way for the forecast of D to read anything published after 10:05
Paris on D-1.

- The operator's wind and solar forecast survives in the known table only on delivery days before 2024-05-01 (64,523
  values). From 2024-05-01 it holds zero such values. Q's regressor on the row of D is the load forecast minus the two
  proxies. A day before 2024-05-01 is never forecast with Q, it falls back to L0.
- My own wider scramble, without any fit. On four days I replaced by noise every price of D and later, every operator
  forecast from 2024-05-01 on, every actual and stitched weather value not yet an hour old at the issue time, and
  everything after D. The design matrix, the price table up to D-1 and the two residual-load tables came out identical
  (0.0, same missing values). So nothing scrambled can reach a fit.
- What remains is by rule, not by measurement. The backtest takes the load forecast of D as known at 10:00 although the
  archive holds a later version (part D shows all 21 days differ). That is what diagnostic (f) is for.

## F. Replay

| Figure | Claimed | Recomputed | Agree | Note |
|---|---|---|---|---|
| Days | not stated | 18 (2026-10-20 to 11-11, and 11-19) | | equals the 18 days in the store with a pre-market live LEAR |
| Largest L0 difference | below 1e-9 so far | 1.20e-9 (2026-11-10) | **no** | above 1e-9, far below the registered tolerance of 0.01. The author's later write-up says 1.2e-9 |
| L0, P, Q complete on every day | yes | yes, 18 of 18, no missing value | yes | |
| Errors | none | none | yes | |
| Replay never reads a price table | yes | yes | yes | its only SQL reads `shadow_versions` and `shadow_values`. Prices enter only as model inputs through the saved windows |
| No price error after 2026-06-30 under work/forward-q | none | none found | yes | every file scanned. CSVs end 2026-06-30, JSONs hold forecasts and forecast differences only |

Second pass with amendment 1 (`replay_filled`, finished during the review, 18 days, same days). L0 is unchanged (largest
difference from the live LEAR again 1.20e-9). P and Q are complete on every day. P and Q are identical in windows 364
and 728 and differ in the two long windows. 1,224 hours of each proxy are filled, 2026-08-20 to 2026-10-10.

## G. Load-revision sensitivity

59 days, 1,416 strict hours, all 59 files present. The plan is every 13th of the 763 rule days starting with
2024-05-29, and the draws equal `numpy.random.default_rng(0).integers(0, 21, size=59)`. Both confirmed.

| On the 59 days | Archive load | Revised load | Change |
|---|---|---|---|
| L0 | 16.5044 | 16.6114 | +0.1070 |
| Q | 15.7003 | 15.8718 | +0.1715 |
| honest_v2_prod | 15.8079 | 15.8259 | +0.0179 |
| S1_prod | 15.2824 | 15.3261 | +0.0437 |
| S1_Q_prod | 14.8929 | 14.9626 | +0.0697 |
| Gain of Q over L0 | 0.8041 | 0.7396 | -0.0645 |
| Gain of S1_Q over S1 | 0.3895 | 0.3635 | -0.0260 |

Mean absolute change of each forecast: L0 1.384, Q 1.489, honest_v2_prod 0.797, S1_prod 1.071, S1_Q_prod 1.134
EUR/MWh. The author's checks.json, written after I had my figures, agrees to every digit.

The change in the gain is small and not distinguishable from zero on 59 days (S1_Q over S1: 0.026, interval -0.015 to
+0.074. Q over L0: 0.064, interval -0.038 to +0.176, plain bootstrap of days). I could not refit L0_revised and
Q_revised. As a consistency check, the size of the forecast change follows the size of the drawn revision (rank
correlation 0.81 for L0, 0.84 for Q).

## Disagreements

1. Part F. The largest difference between the replayed L0 and the live LEAR is 1.20e-9 EUR/MWh on 2026-11-10, not
   below 1e-9. It passes the registered tolerance of 0.01 by seven orders of magnitude. No other figure disagrees.

## Weaknesses and cautions

1. **The first replay did not test the Q that will be issued.** It ran before amendment 1 and without the proxy fill.
   With the fill, Q moves by 6.2 EUR/MWh on average and P by 4.2 (mean absolute change over the 18 days, prices near
   197), L0 by nothing. Only `replay_filled` describes the forward Q. The fill sits outside `q_live`, so checks (b) and
   (c) do not cover it. I read `rolling_proxy` and it is point in time by construction (weights from generation up to
   two days before each month), but I did not test it by perturbation, and I did not check the 7e-12 reproduction
   claim of the amendment.
2. **The record's commit hash does not prove which code ran.** The dry-run record made with the fill carries commit
   e34647d, which does not contain the fill. The working tree is modified and uncommitted again now. The
   pre-registration ends the record on any change to `q_live`, and the record as written cannot show that. A flag for
   a modified tree, or a hash of the source files, would close this. At the end of the review `forward_q.py` had a
   further uncommitted change to `issue`, a new flag "g file differs from the published file" that is not in the
   pre-registered list of flags. It does not touch `q_live`, and the day stays scored.
3. **The DE-LU price is unstable between fetches, and LEAR reads it.** Block `de_lu_d1` is the DE-LU price of D-1. In
   the saved windows an hour takes 7 different values on average (up to 13), and it does not settle with age. The value
   seen before the market for D-1 differs from the last value seen in 94% of hours, by 5.3 EUR/MWh on average, 18.7 at
   the 95th percentile, 71 at most (20 days, 480 hours). French and Swiss prices never change. Diagnostic (f) covers
   only the load forecast. L0 and Q both read this block, so the main comparison is partly protected, but the size of
   the effect is untested and the cause is unknown.
4. **The revision vectors understate a little.** All 21 come from one autumn month (stated by the author). One of
   them, 2026-11-19, is almost the identity (0.12% against 1.37% on average) because its "last value seen" was saved
   only about two hours after the pre-market window. It was drawn for 3 of the 59 days. Without those days the gain of
   S1_Q over S1 goes from 0.387 to 0.360, so the conclusion holds.
5. **The 59 days are not a tight sample.** Their archive gain of S1_Q over S1 is 0.39 against 0.28 on all 763 days.
   Part G shows that the gain barely moves under revision. It does not re-estimate the gain.
6. **Risk band, step A.** The rerun did not reproduce the logged row to the printed digits (4.880, 80.04%, 49.60
   against 4.881, 80.2%, 49.75, and the benchmark 5.102, 77.71%, 47.14 against 5.105, 78.7%, 48.35). The
   pre-registration says the step stops there. The author went on from the stored predictions, which do reproduce, and
   says so. I confirm the cause is warm-up only. The quantiles are identical, the conformal band differs on 1,032 rows
   in February and March 2024, all before the scored set, and the existing band on the scored set is the same from
   both sources.
7. **The working band B under-covers.** 78.0% for 80% and 93.3% for 95%, 72.7% in the first half of 2026, 67.9% on
   holidays, 70.7% on the windiest days, 67.4% in the top 5% of prices. The rule was applied correctly and T lost on
   the interval score, but B should not be described as calibrated.
8. **T's fitted tail does nothing at 80%.** Its 80% ends are the empirical deciles of the pooled scaled errors. The
   generalised Pareto fit only enters at 95%. The minimum of 20 days for the 28-day scale is not in the
   pre-registration (the author states this).
9. **The live running may lose too many days.** Of 23 expected Tuesday to Saturday delivery days from 2026-10-20 to
   2026-11-19, two have no pre-market window at all (17 and 18 November), one was issued late, and the shadow lost two
   more to a model-file mismatch. The forward test needs 90 scored days of 100. At this rate "inconclusive" by the day
   count alone is a real possibility.
10. **The live band figures use prices after 2026-06-30.** `work/risk-band/live_band.json` (25 days, coverage 88.2%)
    compares the stored band with auction prices from October and November 2026. The pre-registration announces it as
    a description. It is outside work/forward-q and I did not recompute it.

One further check, not asked for. The pre-registration says windows of 100 Tuesday to Saturday development days came
out "supported" in about four of five cases. I get 79% at research settings and 78% at production settings (90
windows), never "contradicted".

## Not checked, and why

- The honest_v2_prod and honest_v2_prod_revised forecasts, and the refit of the quantile models. LightGBM was not to
  be run.
- L0_revised and Q_revised. Only two window fits were allowed, and I used them for the same-path check.
- The other four same-path days and the other three windows, and the scramble by fitting. I replaced the latter with
  the comparison of everything the fit reads.
- The live band, the holdout coverage quoted from the log, and anything else that needs a price after 2026-06-30.
- Whether the laptop's own library versions reproduce these numbers each morning.
- The cause of the DE-LU instability.

**Verdict: confirmed with cautions.**
