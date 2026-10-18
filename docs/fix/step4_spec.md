# Step 4 specification: shape pre-check, wind-curve pre-check, IFS timing check

2026-10-18, reviewer (owner of step 4 of panel/final_plan.md). Every choice below is written before any step 4 number is computed. The holdout is not used. No shadow data is read.

## Already seen
- **Daily-level pre-check** (weather_precheck.md): weather surprises explain R2 0.042 of honest_v2's daily mean error (wind 0.024, solar 0.014). It did not measure the hourly shape.
- **Wind curve (hefty agent, repos/wind_shift_check.py):** an in-sample NNLS on the logged labels, 2024-05 to 2026-06.
  - Cubic curve 3/12/25 m/s, as used today: 1,352 MW.
  - Mean of T and T+1: 1,329 MW.
  - That mean with the smoothed curve (sigma 0.15): 1,207 MW.
  - So the functional form below was chosen after an in-sample figure was seen. That figure is not walk-forward.
- **Wind proxy walk-forward MAE (step 2):** 1,399.0 MW on the logged labels and 1,400.0 MW on the corrected labels.
- **Step 2 results for S2-F:** MAE and slices, including 15:00 to 19:00 UTC. No hourly regression on surprises has been run.
- **IFS 00Z arrival:**
  - hefty measured about 08:35 UTC before Oct 2025 and about 07:35 UTC since;
  - the scout reports about 07:34 UTC.
  - No per-day listing has been seen.

## A. Hourly-shape pre-check
- **Model:** S2-F's selection predictions, from scratchpad/fix/F, combined by fix/common.py `members`.
  - Why S2-F: it is the model eligible to go live, and its inputs are the corrected proxies.
  - Its errors and the surprises below are on the same labels. On the logged labels, solar surprise would contain the one-hour bug.
  - honest_v2-F failed its bar, and S2 is built on the logged labels.
- **Rows:** strict selection rows on delivery days from 2024-05-01 to 2026-06-30 on which every hour is strict and both corrected proxies and both actuals are present. Rows before 2024-05-01 are excluded because the corrected wind proxy starts on 2024-04-30 at 22:00 UTC.
- **Shape error:** e(d,h) = forecast minus actual price. s(d,h) = e(d,h) minus the mean of e over the delivery day's own hours (23 or 25 on clock-change days).
- **Surprises,** each demeaned within the day in the same way:
  - wind: actual onshore plus offshore minus `wind_proxy_mw`;
  - solar: `solar_mw` minus `solar_proxy_mw`.
  - Both are taken from data/forecast/inputs_aligned.csv (sha256 b7ba09b2…).
  - Load surprise is not measurable, because inputs.csv has no actual load.
- **Statistic:**
  - pooled OLS without an intercept, of s on the wind and solar surprises interacted with six Paris clock-hour blocks (00 to 03, 04 to 07, 08 to 11, 12 to 15, 16 to 19, 20 to 23), giving 12 coefficients;
  - variance share R2 = 1 minus SSR divided by the sum of s squared;
  - fitted in-sample, so it is an upper bound on what a fresher forecast could recover.
- **Pass:** R2 at least 0.10. Otherwise the fresher-weather shape routes stop: IFS 00Z D-1, WeatherNext and pinning Open-Meteo models.
- **Reported, not decisive:** the 48-term version by hour, the wind-only and solar-only shares, the same statistic for S2 on the logged labels, and coefficients by block.

## B. Wind-curve pre-check
- **Candidate W:** each point's capacity factor is the mean of c(v_T) and c(v_T+1).
  - c is the cubic curve (cut-in 3, rated 12, cut-out 25 m/s), averaged over a Gaussian spread of speeds, v(1 + 0.15z).
  - z runs over 31 nodes evenly spaced on [-3, 3], with normalised Gaussian weights.
  - All of these are fixed now, exactly as in the hefty script.
  - Averaging T and T+1 turns instant speeds into a mean over [T, T+1). For the last hour of D, T+1 is the end of D, which comes from the 18Z D-2 run like every other hour, so the timing gate is unchanged. A test asserts this.
- **Fit:** `gen_proxy.rolling_proxy` with W's transform: NNLS weights refitted each month on the trailing 365 days ending two days before the month, with a 60-day minimum. Same points and corrected actuals (inputs_aligned.csv).
- **Comparison:** against the existing proxy on the same corrected basis, 1,400.0 MW, on the same hours (from 2024-04-30 22:00 UTC to 2026-06-30), against corrected actuals.
- **Pass:** W's walk-forward MAE is at most 1,330.0 MW, a gain of at least 5%. A pass makes W a candidate, which then needs its own pre-registration and the gain rule.
- **Reported, not decisive:**
  - smoothing alone and T/T+1 averaging alone;
  - 15:00 to 19:00 UTC;
  - bias by hour;
  - the windiest 10% of days.

## C. IFS 00Z timing check (only if A passes)
- **Measured by:** the owner's laptop, unsigned S3 list_objects_v2 on the bucket ecmwf-forecasts, prefix YYYYMMDD/00z/ifs/0p25/oper/, recording each object's LastModified. The container cannot reach AWS.
- **Files:** the oper-fc .grib2 files of the D-1 00Z run, for steps 21 to 48 h, which cover delivery day D (in summer D starts at 22:00 UTC on D-1 and IFS steps are 3-hourly, so step 21 is needed).
- **A day's upload time:** the latest LastModified of those files. A missing file, or a missing day, counts as late. A re-upload makes the time later, which is the conservative direction.
- **Days:** selection delivery days D whose D-1 falls on Paris summer time in 2025, 2025-03-30 to 2025-10-25.
- **Pass:** at least 95% of those days have an upload time before 07:50 UTC on D-1. IFS 00Z D-1 reopens only if both A and C pass.


Amendment 2026-10-18 (designer, before any computation): step range changed from 24-48 h to 21-48 h. Previous hash 2eda823d... void.
