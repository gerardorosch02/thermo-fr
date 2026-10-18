# Pre-registration: the smoothed wind curve W on top of S2-F

2026-10-18, quant designer. Written before any price result with W.

## 0. What has already been seen

**Proxy level only, all on corrected labels:**
- The hefty agent's in-sample NNLS on the logged labels: 1,352, 1,329 and 1,207 MW (repos/wind_shift_check.py). W's form was copied from it, so the form was chosen after an in-sample figure had been seen.
- The step 4 walk-forward proxy (step4_results.md):
  - W 1,258.2 against 1,400.0 MW;
  - smoothing alone 1,280.5, and the T/T+1 mean alone 1,377.3;
  - 15:00 to 19:00 UTC 1,245.0 against 1,420.3;
  - windiest 10% of days 1,824.0 against 1,928.8;
  - bias +705 against +814 MW.
- The step 4 shape pre-check failed (R2 0.032; wind alone 0.013).
- Step 2: honest_v2-F 15.7308, S2-F 14.3086, LEAR-F 15.3688.

**No price result with W exists.**

## 1. The decision

**One decision: S2-F-W against S2-F.**
- S2-F is the only model eligible to go live.
- honest_v2-F failed its bar, and the public model stays on the logged labels.

**honest_v2-F-W against honest_v2-F is reported as a diagnostic only.** It is not a decision, because honest_v2-F is not eligible.

## 2. What W changes, and nothing else

- **Wind proxy.** It is rebuilt with W:
  - each point's capacity factor is the mean of c(v_T) and c(v_T+1);
  - c is the cubic curve (3, 12 and 25 m/s) averaged over v(1 + 0.15z), with 31 Gaussian nodes on [-3, 3];
  - the monthly NNLS rule is unchanged (trailing 365 days ending two days before the month, at least 60 days);
  - it is built on inputs_aligned.csv (sha256 b7ba09b2...), with the corrected labels and actuals of F.
- **Residual.** `residual_v2_mw` is recomputed from it, with its formula unchanged.
- **The last hour of D.** Before any price number, the builder checks whether the laptop's saved morning windows contain the wind value one hour after D's last hour.
  - If they do: the step 4 definition stands.
  - If they do not: the last Paris hour of each day uses c(v_T) alone, in the backtest as well, so that backtest, replay and live agree. The proxy MAE is then recomputed and reported.
  - The check's result is recorded.
- **The gradient boosting member.** honest_v2-F-W is refitted walk-forward with the honest_v2-F settings: 800 trees, 63 leaves, seed 7, trained from 2021-01-01, refitted monthly.
- **LEAR-F.** Its predictions are reused unchanged from scratchpad/fix/. LEAR uses neither the wind proxy nor the residual; the builder asserts this from its feature list.
- **S2-F-W weights.** Recomputed from the errors of honest_v2-F-W and LEAR-F by the unchanged rule: a 14-day half-life, settled Tuesday to Saturday days, equal weights until 14 days exist.

## 3. Bar

**W is a new modelling choice, not a correction.** Most of its proxy gain comes from the smoothing (1,280.5 MW alone), and its form was picked after an in-sample figure.
- A not-worse bar would admit a post-hoc choice on no evidence of benefit.
- Non-inferiority is reserved for documented errors. The T/T+1 mean could count as one, but alone it moves the proxy only from 1,400 to 1,377 MW, and splitting it out would add a decision.

**So the sprint's gain rule applies, with one candidate (no multiplicity correction).** S2-F-W passes only if:
1. its overall strict MAE is at least 0.25 lower than S2-F's, so at most 14.0586;
2. no protocol slice is worse than S2-F's by more than 1.0;
3. the upper end of the one-sided 95% moving-block interval of the paired daily MAE difference (blocks of 7 days, 5,000 resamples, seed 0) is below zero.

The expectation is stated now. Wind surprise explains only 1.3 to 2.4% of the error variance, so a pass is unlikely.

## 4. Rows and order

1. **Same rows.** The logged 20,752 strict hours (865 days, 2024-02-17 to 2026-06-30, `--test-start 2024-01-01`), identical targets and the logged slice day sets, all asserted.
2. **Bases first.** honest_v2-F (15.7308) and S2-F (14.3086) are reproduced from the step 2 files before W runs. A difference above 1e-9 stops the run.
3. **Gates.** Timing gates asserted at both deadlines, including the T+1 wind value.
4. **Hash.** The sha256 of this file is recorded and checked before the run.

## 5. Reported, not decisive

For S2-F-W, S2-F, honest_v2-F-W and honest_v2-F:
- MAE and RMSE overall, peak and off-peak;
- the five slices, with the windiest 10% shown separately;
- negative-price hours and by year;
- 15:00 to 19:00 UTC;
- the lag diagnostic: signed, slope and absolute;
- the paired daily difference: mean, two-sided 95% interval, one-sided 95% upper bound and days won;
- S2-F-W's mean LEAR weight;
- the wind proxy MAE as built for the run.

## 6. What follows

**On a pass:**
- W joins S2-F in the post-day-40 replay. The rebuild rule is extended with W's transform and its weights from each monthly refit, and re-hashed, before anyone reads a shadow score.
- The replay's decision object becomes S2-F-W against arm A's S2, with S2-F reported alongside.

**On a fail:**
- Logged and closed for this cycle.
- The replay stays S2-F under the current frozen rebuild rule.
