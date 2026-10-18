# Step 4 results: shape pre-check and wind-curve pre-check

2026-10-18, reviewer. These runs follow step4_spec.md, whose sha256 e03b449e415002b7ecc618526997bb3e32e261be8d03c340c8375da1d019a456 was verified inside the script before anything was computed.
- **Inputs:** /home/claude/thermo-fr-solar/data/forecast/inputs_aligned.csv (sha256 b7ba09b2…, checked) and S2-F's selection predictions (scratchpad/fix/F, built by fix/common.py).
- **Script and output:** scratchpad/step4/run_step4.py; figures in step4/step4_results.json.
- **Excluded:** no price model was fitted, no holdout row was read, and no shadow table was read.

## Verdicts

**A. Hourly-shape pre-check: FAILS (stop).** R2 = 0.0321 against the bar of 0.10. Under the spec, the fresher-weather shape routes stop: IFS 00Z D-1, WeatherNext and pinning Open-Meteo models.
- 18,888 hours on 787 complete strict days, 2024-05-01 to 2026-06-30. The shape error SD is 14.90 EUR/MWh.
- Chance level, p/n, is 0.0006, so overfitting is not a concern here.
- Wind alone gives 0.0129, solar alone 0.0212, and the 48-term hourly version 0.0343.
- Every block coefficient is positive, which is the expected sign: a surprise of more wind or sun than the proxy means the forecast was too high. Values in EUR/MWh per GW: wind 0.6 to 2.0, solar 1.4 to 3.2.
- **Diagnostic:** S2 on the logged labels gives 0.0312. The label fix barely changes the share.
- This is an upper bound. The surprises are realised errors, and the solar surprise also contains the proxy's fleet-growth bias, as the designer noted.

**C. IFS 00Z timing check: NOT RUN.** The spec makes it conditional on A, which failed. IFS 00Z D-1 stays parked, and no listing script was written.

**B. Wind-curve pre-check: PASSES.** The candidate W is the smoothed cubic curve with sigma 0.15 and 31 nodes, averaged over T and T+1. Its walk-forward MAE is 1,258.2 MW against 1,400.0 MW for the existing proxy, a gain of 10.1%. The bar was at most 1,330.0 MW.
- **Same basis for both:**
  - corrected labels;
  - 18,957 hours from 2024-04-30 22:00 UTC to 2026-06-30, the same hours as step 2's 1,400.0 MW;
  - the same monthly 365-day NNLS rule;
  - the recomputed existing proxy equals the stored one to 4e-12.
- **Timing:** for every day, the last hour's T+1 value comes from the 18Z D-2 run, so the gate is unchanged (asserted).
- **Decomposition, not decisive:**
  - smoothing alone gives 1,280.5 MW;
  - the T/T+1 mean alone gives 1,377.3 MW.
- **Other diagnostics for W:**
  - 15:00 to 19:00 UTC: 1,245.0 against 1,420.3 MW;
  - windiest 10% of days: 1,824.0 against 1,928.8 MW;
  - bias: +705 against +814 MW. The proxy still under-forecasts on average, which is the fleet-growth lag.
- **What a pass means:** W becomes a candidate only. It needs its own pre-registration under the gain rule, and it was chosen after the hefty agent's in-sample figure had been seen.

## Reading
- Weather surprises explain about 3% of the hourly-shape error, against 4% of the daily level. The route of fresher or better weather forecasts is closed for this model on both counts.
- A better wind proxy is still a better input. But wind surprise accounts for only 1.3% of the shape variance and 2.4% of the daily level, so any price gain from W is likely small. A gain bar of 0.25 is unlikely to be met.
- I would pre-register W only as a non-inferiority correction, or not at all. That is the designer's call.
