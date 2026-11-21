# Fresher weather for the German wind proxy: a comparison on development days

Written 2026-11-21, before any fit. This is a study on development data, 2024-05-01 to 2026-06-30. It is not a test of a
model on unseen days and nothing it finds changes the test of W, which stays on its frozen inputs and is read on or after
2027-01-19. No day after 2026-06-30 is read.

## The question

The spread models read German wind through a proxy built from weather forecasts issued about two days ahead. A forecast
made at 10:05 Paris could use the weather run started at midnight UTC that morning. Does the fresher run make the proxy
better in MW, and does any of that reach the forecast of the France minus Germany spread in EUR/MWh?

## Data

- **NOAA GFS**, 0.25 degrees, hourly, 100 m wind, every run of 00, 06, 12 and 18 UTC from 2023-01-01 to 2026-06-30, at the
  17 German points of the stored table. Fetched on the owner's laptop on 20 to 21 November 2026 from dynamical.org's point
  archive (CC BY 4.0). 5,108 runs, none missing, no missing value. Packed file sha256
  b6bc63103f7a480df1b615ff359216384644e0e70dda5b4386aef8d0a1c91057.
- **Everything else** is the stored development data of the spread work, unchanged.

Arrival. On the two mornings looked at so far, 20 and 21 November 2026, the midnight GFS files to lead 54 had arrived at
NOAA's public store by 04:04 and 04:14 UTC. The relayed review of 20 November reports the same on 18 sampled mornings of
2025 and 2026. The gate is 09:05 UTC in winter and 08:05 UTC in summer. Whether the run was on hand on every past morning
cannot be shown.

## Arms

All arms use the same 17 points, the same monthly non-negative fit against the operator's day-ahead forecast, the same
solar table and the same frozen spread code. Only the wind table and, in two arms, the wind curve differ.

- **B, the stored table.** Open-Meteo's blended forecast as issued two days ahead. This is what G and W use now.
- **S, stale GFS.** Each hour from the latest GFS run started at least 48 hours before it, leads 48 to 53. This is the
  present convention applied to one named model.
- **F, fresh GFS.** Every hour of delivery day D from the 00 UTC run of D-1, leads 22 to 47.

Each of S and F is run with the wind curve as built and with the fleet-style curve of the exploration of 20 November (the
curve averaged over a normal spread of 3.5 m/s). That width was the best of three tried on these same days against the
stored table, so the fleet arms are descriptive.

The GFS tables start on 2024-01-01, as the stored table does, so the proxies begin in the same month in every arm and the
stand-in used before then is the same.

## What is fitted

For each of B, S and F with the curve as built, and S and F with the fleet curve: the proxies, then G (monthly refits) and
W (weekly refits, 180-day half-life), walk-forward over 2024-05-01 to 2026-06-30 exactly as in the design runs of W.

## Stop conditions, checked before any figure is read

1. Arm B reproduces the stored forecasts of G and W to 1e-9 and the stored wind proxy to 1e-6 MW.
2. The tables do not read ahead. With every GFS run started after 00 UTC of D-1 replaced by noise, both tables are
   unchanged for every delivery day up to D, on four dates including both clock changes.
3. Every scored hour has a proxy and both forecasts in every arm.

If any fails, no figure is reported and the failure is.

## Figures

On the 18,982 scored hours of the design runs.

- **In MW**, the wind proxy (onshore plus offshore) against the operator's day-ahead forecast: mean absolute error, bias,
  correlation, rank correlation, hourly ramp error, and the error by block of six delivery hours. The freshness gain in
  lead is about 25 hours for the first delivery hour and about 6 for the last, so the blocks show whether the gain follows
  the lead.
- **In EUR/MWh**, the hourly spread error of G and of W, and paired daily differences with a moving-block interval (blocks
  of 7 days, 5000 resamples, seed 0).

## Primary contrast and how it will be read

**F against S with the curve as built, for W.** Same weather model, same points, same curve, same fit. The only difference
is the age of the run.

- Freshness is worth a forward challenger if W's error falls by at least 0.25 EUR/MWh and the two-sided 95% interval lies
  below zero.
- By amendment 3 of the test of W, a difference between two models carries numerical noise of up to about 0.25. A result
  near that size is reported as not distinguishable.
- There is no slice rule. These are development days and nothing here passes or fails a model.

Everything else is descriptive: the same contrast for G, the contrast with the fleet curve, S against B (what changing
the weather model alone does), F against B (what the whole change does against today's inputs), and the fleet curve
against the curve as built.

## What this cannot show

- **Operational gain.** B is a finer regional blend than GFS. If S is worse than B, a fresh GFS may still not beat today's
  inputs. F against B answers that, on days W's design had in view.
- **Truth.** The proxy is judged against the operator's forecast, not against German generation, which is not in the
  project's files.
- **Out-of-sample value.** A gain here is a reason to build a challenger and test it forward, under its own registration.
- **Uncertainty.** The ensemble is not in this study. Its history is not fetched yet.

## Review

An independent reviewer that has not seen the build recomputes the figures from the saved forecasts before they are
reported.
