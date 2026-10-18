# Smoothed wind curve W on top of S2-F: result

2026-10-18, builder. The run followed wind_prereg.md.

- **Pre-registration:** sha256 9c47b5bbc9049b33f81fb7937bcb4ec206f1e0e55e92020b6c81bb8da183f501, checked first.
- **Code:** solar-align 61d6185, with a clean tree under src.
- **Inputs:** inputs_aligned.csv, sha256 b7ba09b2..., checked.
- **Runs:** in scratchpad/wind/ (run_wind.sh; logs and .exit per step). The full report is wind/wind_report.json.
- **Excluded:** the holdout was not used and no shadow table was read.

## Verdict: S2-F-W fails the gain rule. W is logged and closed for this cycle, and the replay stays S2-F under the frozen rebuild rule.

| Condition | Result | Passed |
|---|---|---|
| 1. MAE at least 0.25 below S2-F's (at most 14.0586) | 14.2533 against 14.3086, a gain of 0.0553 | No |
| 2. No protocol slice worse by more than 1.0 | worst is holidays at +0.08 | Yes |
| 3. One-sided 95% upper bound of the paired daily difference below zero | -0.0223 (mean -0.0553, two-sided 95% [-0.0952, -0.0165]; F-W wins 49.1% of days) | Yes |

So W gives a small improvement that the interval can separate from zero, but about a fifth of the size the rule requires.

## Pre-run checks

**Saved morning windows.**
- No saved morning window exists yet. `morning-run --save-shadow-window` starts with the shadow on 19 October.
- The copy of the laptop's data in this session has no `data/shadow/fresh/`, and the store keeps only the delivery day's 24 hours of 16 series.
- A live Open-Meteo call from the container is refused by the proxy.
- The fetch range (D-10 to D+2) does request the hour after D, but whether the archive returns it at the pre-market fetch cannot be shown here.
- So the check fails, and the prescribed fallback applies: the last Paris hour of each day uses c(v_T) alone (`W_LAST_HOUR = "own"`), in the backtest as well. Recorded in wind/windows_check.json.

**Wind proxy as built for the run** (fallback, walk-forward, 18,957 hours from 2024-04-30 22:00 UTC):
- MAE 1,258.9 MW against 1,400.0 MW for the existing curve; 15:00 to 19:00 UTC 1,245.2 against 1,420.3; bias +706 against +814 MW.
- Step 4's definition, with T+1 taken for the last hour too, gives 1,258.2 MW, the reviewer's figure, reproduced by a test.

**LEAR** uses neither the wind proxy nor residual_v2. This was asserted from LEAR_COLUMNS and LEAR_BLOCK_TIMINGS. LEAR-F's predictions from scratchpad/fix/F were reused unchanged (asserted).

**Bases first.**
- honest_v2-F was rerun on the branch from inputs_aligned.csv. It is identical to the step 2 file (largest difference 0.0).
- honest_v2-F gives 15.7308 and S2-F 14.3086.

**Same rows and gate.**
- Same rows: 20,752 strict hours on 865 days, identical targets, and the logged slice day sets.
- Gate: asserted at both deadlines with the W timing rule, which takes the run of the T+1 speed (still the 18Z run of D-2).

## Diagnostics (strict rows, EUR/MWh)

| | S2-F-W | S2-F | honest_v2-F-W | honest_v2-F |
|---|---|---|---|---|
| MAE | 14.2533 | 14.3086 | 15.6385 | 15.7308 |
| RMSE | 19.591 | 19.660 | 21.462 | 21.597 |
| Peak / off-peak | 15.03 / 13.82 | 15.08 / 13.88 | 16.05 / 15.41 | 16.11 / 15.52 |
| Weekends | 14.05 | 14.08 | 14.98 | 15.00 |
| Holidays | 17.91 | 17.83 | 18.82 | 18.66 |
| Windiest 10% | 15.95 | 16.11 | 17.30 | 17.44 |
| Sunniest 10% | 14.39 | 14.41 | 15.13 | 15.18 |
| Top 5% hours | 20.08 | 20.18 | 22.29 | 22.63 |
| Negative-price hours | 17.08 | 17.07 | 16.32 | 16.32 |
| 15 to 19 UTC | 15.942 | 16.048 | 16.912 | 17.089 |
| 2024 / 2025 / 2026 | 14.026 / 13.225 / 16.727 | 13.998 / 13.279 / 16.933 | 15.914 / 14.400 / 17.651 | 15.778 / 14.505 / 18.121 |
| Lag: signed / slope / absolute | -0.055 / -0.023 / 0.129 | -0.053 / -0.022 / 0.131 | -0.059 / -0.029 / 0.100 | -0.054 / -0.027 / 0.101 |

**honest_v2-F-W against honest_v2-F** (diagnostic only):
- Mean paired difference -0.092, two-sided 95% [-0.195, +0.006], one-sided 95% upper bound -0.009; F-W wins 47.7% of days.
- Slices: holidays +0.16, top 5% hours -0.34.

S2-F-W's mean LEAR weight is 0.509, against 0.511 for S2-F.

**Note.** The proxy gain (10.1%) carries into a small price gain: about 0.06 for S2-F-W and 0.09 for its gradient boosting member. That is consistent with the step 4 shape pre-check (wind surprise R2 0.013), and it falls well short of the 0.25 bar.
