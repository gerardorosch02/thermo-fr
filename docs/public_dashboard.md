# Public dashboard: how it is kept up to date and how to deploy it

The public dashboard (`streamlit_app.py`) reads only the files under `published/`. A GitHub Actions workflow (`.github/workflows/forecast.yml`) rewrites that folder on weekday mornings and after the auction results each afternoon, so neither the dashboard nor the data depends on a local machine being on.

## What is published

| File | Content |
|---|---|
| `published/forecasts.csv` | every honest forecast version of the last 90 days: delivery day, issue time, hour, forecast, same-hour-previous-day benchmark |
| `published/actuals.csv` | actual day-ahead prices for the same window |
| `published/scores.csv` | daily MAE and RMSE of each version against actuals and benchmark |
| `published/tomorrow.csv` | the latest forecast for the next delivery day |
| `published/error_band.csv` | backtest error percentiles by hour (the shaded band) |
| `published/history.csv` | the hourly inputs table the model is refitted on (prices, load and renewables forecasts, weather, 2021 to date) |
| `published/status.json` | when the dataset was written, the last run, the attributions |

Only the honest feature set is published. The extended set, the data-status panel and the timing log remain in the local database and the local dashboard (`thermo-fr dashboard`).

The workflow is stateless: each run imports `published/` into a fresh SQLite store (`thermo-fr import-published`), runs `morning-run` or `settle`, exports again (`thermo-fr publish`) and commits only `published/`. `history.csv` is the training set, so it has to travel with the repository; it is plain CSV so that each daily append is a small git delta.

## Schedule

GitHub cron is UTC. The workflow runs `morning-run` at 09:30 and 10:30 UTC on weekdays (10:30 and 11:30 London in summer, 09:30 and 10:30 in winter) and `settle` at 13:15 UTC every day (14:15 London in summer, 13:15 in winter). GitHub may start scheduled runs several minutes late. Each step can also be started by hand from the Actions tab (Run workflow, choose the step).

## One-time setup

### (a) Add the ENTSO-E token as a repository secret

1. On GitHub open the repository, then Settings, then Secrets and variables, then Actions.
2. Click New repository secret.
3. Name: `ENTSOE_API_KEY`. Value: your token (the same value as your local environment variable). Click Add secret.
4. The workflow reads it as `${{ secrets.ENTSOE_API_KEY }}` into the job's environment only. GitHub masks it in logs, and the code never prints or writes it.

Then run the workflow once by hand: Actions, Day-ahead forecast, Run workflow, step `morning-run`. It should finish in two to four minutes and push a commit that touches only `published/`.

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
- The item to think about is `published/history.csv`: it republishes the full hourly day-ahead price history since 2021. Day-ahead prices originate with the power exchanges (the primary owners), whose own market-data terms are more restrictive than ENTSO-E's. Forecasts, errors and a rolling 90-day window of actual prices are low risk; a complete multi-year price history is the part that could be challenged. Options: keep it (with attribution, as now), limit it to the last two or three years, or keep the training history out of git and ship a monthly refitted model file instead so the workflow only needs the last ten days of inputs.
- RTE eCO2mix data is under the Licence Ouverte v2.0 (Etalab), which allows re-use with attribution; it is not part of the published dataset but is credited because the project uses it.
- Open-Meteo forecasts are CC BY 4.0 and are credited.
