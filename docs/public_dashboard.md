# Public dashboard: how it is kept up to date and how to deploy it

The public dashboard (`streamlit_app.py`) reads only the files under `published/`. A GitHub Actions workflow (`.github/workflows/forecast.yml`) rewrites that folder on weekday mornings and after the auction results each afternoon, so neither the dashboard nor the data depends on a local machine being on.

## What is published

| File | Content |
|---|---|
| `published/forecasts.csv` | every forecast version of the last 90 days of the default set honest_v2 and the fallback set honest (column feature_set), each with the model that produced it (column model: gbm:honest_v2.txt, gbm:honest.txt:fallback, or gbm for a live fit): delivery day, issue time, hour, forecast, same-hour-previous-day baseline, and `premarket` (1 when issued before the 11:15 Paris market window of the day before delivery) |
| `published/actuals.csv` | actual day-ahead prices for the same window |
| `published/scores.csv` | daily MAE and RMSE of each version against actuals and the naive same-hour-previous-day baseline, with `mae_below_baseline` per day |
| `published/tomorrow.csv` | the headline forecast for the next delivery day: the last version issued before the market window, else the latest |
| `published/error_band.csv` | backtest error percentiles by hour (the shaded band) |
| `published/model/honest.txt` | the fallback LightGBM model (the honest set), refitted on the first weekday of each month (300 trees, 31 leaves, about 0.9 MB; chosen because its holdout MAE is within 0.2 EUR/MWh of the backtest settings) |
| `published/model/honest.json` | its training period, fit date, feature list and holdout metrics |
| `published/model/wind_proxy.json` | weights of the wind generation proxy (MW per forecast point), refitted with the model on the trailing year of actual wind generation; the morning run applies them to the fresh point forecasts |
| `published/probabilistic.csv`, `tomorrow_probabilistic.csv` | per version and hour: the 10th, 50th and 90th percentile forecasts, the conformal 10-90 band (lo, hi), the probabilities of a negative price and of a spike, and the spike threshold; the headline version's rows for tomorrow in the second file |
| `published/probabilistic_backtest.json` | aggregates of the probabilistic backtest on the selection window and the holdout (pinball loss, coverage, width, Brier skill, reliability tables), written by `thermo-fr prob-backtest` |
| `published/model/honest_v2_q10.txt`, `_q50`, `_q90`, `_negative`, `_spike`, `honest_v2_probabilistic.json` | the quantile and event models refitted monthly with the point model, and their metadata (features, initial conformal margin, holdout coverage); when any file is missing the forecast carries no band |
| `published/shape_battery.json` | shape and battery value: the backtest aggregates written by `thermo-fr shape-backtest` (shape MAE, spread error, extreme-hour hit rates, battery EUR/day and share of perfect foresight for the shape model, the D-1 and same-type benchmarks) and the live record of the pre-market version of each settled day with its daily battery values; no traded prices |
| `published/model/honest_v2.txt`, `honest_v2.json` | the default model since 2026-10-12: the honest set plus lagged nuclear generation, neighbour prices and a residual load (docs/experiments.md); its metadata lists the features |
| `published/model/solar_proxy.json` | the same for the solar generation proxy (radiation forecasts at 21 points against actual solar generation) |
| `published/status.json` | when the dataset was written, the last run, the model summary, the attributions, and under `market` the aggregates of the forecast against EEX traded prices (days scored, hit rate, mean P&L per MWh, model and market error); never a traded price, and only from `MIN_PUBLIC_DAYS` (20) scored days, before which the app says "collecting data, n of 20 days" |

Only the honest_v2 and honest feature sets are published; `status.json` names the default model file, the fallback model file and both models' metadata. The extended set, the data-status panel and the timing log remain in the local database and the local dashboard (`thermo-fr dashboard`).

The workflow is stateless: each run imports `published/` into a fresh SQLite store (`thermo-fr import-published`), runs one step, exports again (`thermo-fr publish`) and commits only `published/`. No inputs history is kept in the repository. The daily `morning-run` fetches only the ten days around the next delivery day and predicts with the stored model file. On the first weekday of each month the `refit` step fetches the full history since 2021 with the secret, refits, and commits only `published/model/` (the model file plus metadata with the training period, fit date and holdout metrics). The public dataset keeps at most 90 days of actual prices.

## Schedule

GitHub cron is UTC. The workflow runs `morning-run` at 09:30 and 10:30 UTC on weekdays (10:30 and 11:30 London in summer, 09:30 and 10:30 in winter), `settle` at 13:15 UTC every day (14:15 London in summer, 13:15 in winter), and `refit` at 02:00 UTC on the first weekday of the month (the cron fires on the 1st to the 3rd and the job checks the weekday). GitHub may start scheduled runs several minutes late. Each step can also be started by hand from the Actions tab (Run workflow, choose the step). The very first run must be `refit`, so that `morning-run` has a model file.

## One-time setup

### (a) Add the ENTSO-E token as a repository secret

1. On GitHub open the repository, then Settings, then Secrets and variables, then Actions.
2. Click New repository secret.
3. Name: `ENTSOE_API_KEY`. Value: your token (the same value as your local environment variable). Click Add secret.
4. The workflow reads it as `${{ secrets.ENTSOE_API_KEY }}` into the job's environment only. GitHub masks it in logs, and the code never prints or writes it.

Then run the workflow by hand twice: Actions, Day-ahead forecast, Run workflow, step `refit` (fetches five years of inputs and fits; allow 20 to 40 minutes, it commits `published/model/`), then step `morning-run` (two to four minutes, commits `published/`).

### (b) Deploy on Streamlit Community Cloud

1. Make the repository public (after the history check below) or keep it private and grant Streamlit access to it.
2. Go to https://share.streamlit.io, sign in with GitHub, click Create app (or New app).
3. Choose the repository, the branch (`main` once this branch is merged), and set Main file path to `streamlit_app.py`. The Python dependencies come from `requirements.txt` at the repository root (streamlit, pandas, plotly only).
4. Click Deploy. The first build takes a minute or two. No secrets are needed: the public app never calls an API.
5. Optionally set a custom subdomain in the app settings (for example `thermo-fr`), which gives a URL like `https://thermo-fr.streamlit.app`.

Streamlit Community Cloud redeploys automatically when the branch changes, so every workflow commit to `published/` refreshes the app within a minute or two. Apps that receive no visits for a while are put to sleep and wake on the next visit.

### (c) Put the live link at the top of the README

Replace the placeholder line near the top of `README.md`:

```
**Live dashboard:** https://YOUR-APP.streamlit.app (updated on weekday mornings and after each auction)
```

with the URL Streamlit gave you, commit, and push.

## Before making the repository public

Run the history check once more from a clean clone:

```bash
git log --all --name-only --format="" | sort -u | grep -E "^(data|reports|logs)/"   # must print nothing
git log --all -p | grep -i -E "securityToken=[0-9a-f]{8}" | grep -v -E "\.\.\.|SECRET-TOKEN"   # must print nothing
```

What the check found on 2026-10-09: no data, report or log files were ever committed; no token-like strings; no local paths; the only CSV files in history are the small test fixtures. The commit author e-mail is the one visible in `git log`, which GitHub will show publicly.

## Data terms and what stays private

- ENTSO-E Transparency Platform (General Terms and Conditions, section II.2, clause 5): data may be used "for any purpose whatsoever" provided the user acts in good faith, mentions the ENTSO-E Transparency Platform as the source of publication, does not imply ENTSO-E endorsement, and does not prejudice any copyright held by the primary owner of the data. Since 2019 ENTSO-E also publishes a list of data items open for re-use without the primary owner's prior agreement. The public app and `status.json` cite the platform as the source and carry a no-endorsement line.
- No price history is republished. The repository holds at most 90 days of actual prices (`published/actuals.csv`) next to the forecasts, plus the fitted model file; the training history is fetched by the monthly refit and discarded. Day-ahead prices originate with the power exchanges, whose own market-data terms are stricter than ENTSO-E's, which is why the full history stays out of git.
- RTE eCO2mix data is under the Licence Ouverte v2.0 (Etalab), which allows re-use with attribution; it is not part of the published dataset but is credited because the project uses it.
- Open-Meteo forecasts are CC BY 4.0 and are credited.
