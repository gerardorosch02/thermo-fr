# thermo-fr

**Live dashboard:** [thermo-fr-cy6smxhzz5tj77jmvyqfng](https://thermo-fr-cy6smxhzz5tj77jmvyqfng.streamlit.app/) (updated on weekday mornings and after each auction; see [docs/public_dashboard.md](docs/public_dashboard.md) for how it is produced)

How much does French electricity demand rise when it gets colder, and what does that do to the day-ahead price?

France heats a large share of its homes with electricity, so its demand is unusually sensitive to temperature. RTE usually puts the winter figure at roughly 2,400 MW for each degree colder. This project estimates that number from public data, along with the matching day-ahead price effect, and checks the model out of sample.

## What it does

1. Downloads hourly French actual load and day-ahead prices from one of several pluggable sources (see below), and hourly temperatures for eight French cities from the Open-Meteo archive.
2. Builds a population-weighted national temperature and averages everything into local calendar days.
3. Fits a piecewise-linear model. Above a threshold temperature, weather barely moves demand. Below it, each degree colder adds a fixed amount of load. The threshold is chosen by grid search.
4. Fits the same shape to the day-ahead price, using the load model's threshold.
5. Trains on all but the last 12 months and forecasts the last 12 months of load to check the model out of sample.
6. Writes a summary and two charts to `reports/`.

## Results

Run on 2021-01-01 to 2025-12-31 (1,826 days) with RTE load, ENTSO-E day-ahead prices and Open-Meteo temperatures. All numbers below are copied from `reports/summary.json` and `reports/yearly.csv` as produced by `thermo-fr fit`.

- Heating threshold: 13.00 °C
- Load: +1,605 MW per degree colder below the threshold (s.e. 22), R² 0.962
- Price: +9.67 EUR/MWh per degree colder (s.e. 0.45), R² 0.874; excluding 2022: +6.58 (s.e. 0.30)
- Out of sample, last 365 days trained to 2024-12-31: MAE 1,593 MW, MAPE 3.12%

Year by year, with the threshold held at 13.00 °C:

| Year | Load MW per °C (s.e.) | Price EUR/MWh per °C (s.e.) | Price R² | Mean price EUR/MWh | Price gradient, % of mean |
|---|---|---|---|---|---|
| 2021 | 1,649 (50) | 6.92 (0.74) | 0.863 | 109.2 | 6.33 |
| 2022 | 1,763 (50) | 22.40 (1.69) | 0.754 | 275.9 | 8.12 |
| 2023 | 1,694 (48) | 8.27 (0.53) | 0.714 | 96.9 | 8.54 |
| 2024 | 1,456 (53) | 4.62 (0.56) | 0.647 | 58.0 | 7.96 |
| 2025 | 1,469 (46) | 6.30 (0.50) | 0.722 | 61.1 | 10.31 |

Load sensitivity has fallen: the 2024 and 2025 estimates are about 14% below the 2021 to 2023 average, in line with the demand reductions that followed the 2022 energy crisis. The price effect in EUR/MWh moves with the price level, 22.4 in 2022 against 4.6 to 8.3 in the other years, but as a share of each year's mean price it stays within 6 to 10%. A cold day therefore raises the day-ahead price by a fairly constant fraction, whatever gas costs that year.

![Daily load against temperature](docs/img/load_vs_temperature.png)

![Daily day-ahead price against temperature](docs/img/price_vs_temperature.png)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ",forecast" inside the brackets for the price forecast (LightGBM, scikit-learn)
```

Open-Meteo, Energy-Charts and RTE need no key. Only the `entsoe` source does.

## Usage

```bash
thermo-fr demo                                      # offline, synthetic data with known answers
thermo-fr fetch --start 2021-01-01 --end 2026-01-01 # real data, saved to data/hourly.csv
thermo-fr fit                                       # results in reports/
pytest                                              # run the tests, all offline
```

`fetch` takes `--load-source` and `--price-source`, each one of `rte`, `energy-charts`, `entsoe` or `csv`. The defaults are `rte` for load and `energy-charts` for prices, which need no key:

```bash
thermo-fr fetch --start 2021-01-01 --end 2026-01-01 --load-source rte --price-source energy-charts
thermo-fr fetch --start 2021-01-01 --end 2026-01-01 --load-source csv --price-source csv --csv-dir data/csv
```

Next to `data/hourly.csv`, `fetch` writes `sources.json` (which source produced which series, with licence attribution and the dataset details) and `quality.json`. A short quality summary is printed: the share of missing hours per series, negative load, load outside 20,000 to 100,000 MW, and prices outside -500 to 4,000 EUR/MWh. Gaps are reported, never filled. `fit` copies the source information into `summary.json` and `summary.md`.

## Day-ahead price forecast

A second pipeline forecasts the hourly French day-ahead price for a delivery day using only information available when the auction closes, at 12:00 Paris time the day before. Method, timing findings and backtest results are in [docs/forecast.md](docs/forecast.md).

```bash
pip install -e ".[dev,forecast]"                  # adds LightGBM and scikit-learn
export ENTSOE_API_KEY="your-token"
thermo-fr forecast-fetch                           # inputs 2021-01-01 to the last complete month, into data/forecast/
thermo-fr forecast-backtest                        # walk-forward 2024 and 2025, report in reports/forecast/
thermo-fr forecast --date 2026-10-06               # one day's curve and chart, as of 12:00 the day before
thermo-fr timing-probe                             # log which ENTSO-E items already exist for tomorrow
thermo-fr market add --date 2026-10-10 --product base --window 11:15-12:00 --open 100.00 --high 102.00 --low 99.00 --close 101.00 --vwap 100.50 --source "EEX via trader"   # illustrative prices
thermo-fr market paste --date 2026-10-10 --text "FR DA Base EEX Trades 11:15-12:00 O: 100.00 H: 102.00 L: 99.00 C: 101.00 VWAP: 100.50"
thermo-fr market fetch                             # collect tomorrow's window with the local collector module (laptop scheduler only)
thermo-fr market fetch --backfill 40               # one-off: the available history, one request every 10 seconds
thermo-fr market evaluate                          # every recorded traded price against the forecast that was live before its window
```

Traded prices live in `data/market/eex_fr_da.csv` (columns: delivery_date, product, trade_date, window_start, window_end in Paris time, open, high, low, close, vwap, trades, volume_mwh, source, note; the trade date is the day before delivery, or the Friday before a Saturday, Sunday or Monday delivery, since the day futures for those three trade on Friday). The whole `data/` tree is git-ignored and a test checks that this file is; raw EEX prices are never committed, published or shown in the public app.

Three ways in. `market fetch` collects the window itself: it runs a collector module that lives in `local/`, a git-ignored folder outside the package, because the module depends on the undocumented structure of EEX's public Market Data Hub page and on a request header that page expects, and EEX's notice says the data shown there is for information purposes. The public repository therefore carries only the contract the module must meet (`run_fetcher` in `forecast/market.py`: a `fetch(delivery_dates, window, raw_dir, spacing_s, log)` function returning one outcome per delivery day and product) and never the endpoint, the header or the contract codes. The module rebuilds the 11:15 to 12:00 Paris open, high, low, close and volume-weighted average from the trade-by-trade tape, with the trade count and the volume of the window, and marks the row "exact, from EEX tape"; it sends a plain user agent naming the project and a contact address, at most a handful of requests a day spaced ten seconds apart, no proxies and no rotation, and keeps the raw responses under `data/market/raw/` (git-ignored). It runs from the laptop's Task Scheduler only (weekdays at 12:20 Paris with one retry at 12:35, after the 15 minute display delay has passed the end of the window; on a Friday it also collects the Saturday, Sunday and Monday deliveries), never from GitHub Actions. A refusal, a changed page or a missing tape is logged to `data/market/fetch_log.csv`, shown in the dashboard's data status panel, and the day is skipped: nothing is guessed. `market paste` is the fallback for such a day: it parses the free text a trader sends (product, window, O, H, L, C, VWAP; decimal commas accepted; the delivery date from `--date` or an ISO date in the text) and refuses anything incomplete. `market add` takes the numbers as flags. `market evaluate` scores each day with the latest honest forecast issued before the window opened, so a day whose only forecast came later is skipped, not scored with hindsight. It reports per day and in total: forecast, entry (the window VWAP, with the close as an alternative), auction result, direction, P&L per MWh, model error and market error against the auction, hit rate, cumulative P&L, how often the model's error was below the market's, and a few no-trade bands (trade only when the forecast is more than X from the entry), labelled in sample.

Two measurements, kept apart (see the [glossary](#glossary)), and two different answers. On forecast error the model beats the naive baseline, the spot price of the same hour on the previous day: MAE 15.5 EUR/MWh against 20.7 in the 2024 to 2025 backtest. Against EEX traded prices before the auction it loses. Over 42 windows (38 base, 4 peak) for delivery days 2026-08-29 to 2026-10-10, the direction implied by the forecast against the window's VWAP (long above, short below), settled at the auction result, had a hit rate of 29% and a mean P&L of -2.5 EUR/MWh per window; the forecast's error against the auction was 27.0 EUR/MWh where the traded VWAP's was 5.1, and the forecast's error was the smaller one on 12% of windows. The main reason is that the model reacts to large moves about two days late, because its price lags carry yesterday's level into tomorrow's forecast, while the market reprices the same morning. October alone, when the base price fell from about 200 to about 50 EUR/MWh within a week: 13 windows, 15% hit rate, -4.5 per window, 34.9 against 6.3. The record was scored as a walk-forward so that nothing is in sample: late August and September with a model fitted on data through August, October with the published model (fitted on data through September) or, for the three windows where a live version existed before the window, with that live version. The peak contract rarely trades inside the window, which is why there are four peak rows; Sunday and Monday contracts, traded on Friday, had trades on the day but none inside it. The sample grows by one collected day per weekday (`thermo-fr market fetch`, `thermo-fr market evaluate`), and the public app shows aggregates once 20 days have a live pre-market forecast. The error band on the dashboards is the model's past error, not a probability forecast. A known weakness is the top 5% price hours, typically cold, calm winter evenings when gas sets the price, where the generation proxies make the backtest error slightly worse (23.2 against 22.6 EUR/MWh without them) while improving every other slice. Backtest, strict rows 2024-02-17 to 2025-12-31: honest set MAE 15.5 EUR/MWh against 20.7 for the baseline; the wind proxy, the solar proxy and the calendar structure contributed 1.1, 0.25 and 0.3 of that.

Fundamentals v2 (October 2026). Three groups of inputs were added under a pre-registered protocol (docs/experiments.md): lagged actual nuclear generation, the day-ahead prices of six neighbouring zones for the previous day, and a residual load built from the load forecast, the generation proxies and the latest nuclear output. On the selection window every group passed the acceptance rule and the full set lowered the strict backtest MAE from 16.4 to 15.7 EUR/MWh. On the frozen holdout (July to October 2026) the gain was 0.4 EUR/MWh, the October crash was forecast no better than by the naive baseline, and the lag diagnostic did not improve out of sample. The model still follows large moves about two days late because three price lags carry most of its weight. Against EEX traded prices the record barely moved (hit rate 33%, mean P&L -2.4 against -2.8 EUR/MWh per window, model error 28.5 against the market's 5.1). Planned nuclear availability could not be reconstructed as of the issue time from ENTSO-E, whose outage API serves only the latest revision of each notice; since 2026-10-10 the laptop saves a raw snapshot of the notices every morning (`thermo-fr outage-snapshot`, `data/entsoe/outage_snapshots/`, git-ignored, paged past the platform's 200-document cap) and `thermo-fr nuclear-availability` rebuilds planned availability for a delivery day as of any time from the snapshots taken before it, so a future branch can test the feature without look-ahead. The v2 model, `published/model/honest_v2.txt`, is the default since 2026-10-12; `honest.txt` is kept as the fallback for days on which a v2 input is missing, and every forecast version names the model that produced it.

Probabilistic forecasts (October 2026). Next to the point forecast the default model computes, every day, three price quantiles (10th, 50th, 90th, LightGBM quantile objectives on the same honest_v2 inputs) with the 10-90 interval widened by a conformal margin from the last settled days, and two event probabilities, a negative price and a spike above the 95th percentile of the trailing year's hourly prices known at the issue time. The rules were pre-registered in `docs/experiments.md` and the outcomes are these. The negative-price probability was accepted on the selection window and confirmed on the holdout: 158 events, Brier skill +0.37 against climatology and +0.41 against the last 7 days. It is published: the public app shows it per hour with a calibration panel (Brier scores, skill scores, reliability tables) and its selection and holdout results. The quantile interval was accepted on the selection window (pinball 4.88 against 5.11 and 7.53 for the two benchmarks, 80.2% coverage) but under-covered on the holdout: 73.8% overall, 56% on the windiest days and 56% on holidays, because the trailing 90-day calibration window lagged the move to a high-price regime. The spike probability was accepted on the selection window but not confirmed on the holdout, where the trailing-year threshold labelled 41% of hours as spikes and the 7-day frequency beat the classifier, so the definition itself needs revisiting. Both are computed and stored locally every day so a live record builds, shown in the local dashboard labelled "under evaluation", and not published; the shaded band on the public chart remains the backtest error band. The five model files are refitted monthly with the point model in `published/model/`; a version issued without them carries no band and no probabilities.

Shape and battery value (branch shape-and-battery, October 2026). The level forecast is scored on its hourly error; a storage asset trades the shape of the day, each hour's price minus the day's mean. `thermo-fr shape-backtest` runs a shape model (gradient boosting on the honest_v2 inputs with the shape as target, retrained monthly) against two benchmarks, yesterday's shape and the shape of the most recent earlier day of the same type, and values each with a battery: 1 MW / 2 MWh, 88% round-trip efficiency, one cycle a day, a two-hour charge block before a two-hour discharge block chosen on the forecast before the gate and settled at the auction result, skipped when the forecast spread does not cover the efficiency loss. On the selection window (865 strict days, 2024-02-17 to 2026-06-30) the shape model beats both benchmarks: hourly shape MAE 11.3 EUR/MWh against 15.9 for yesterday's shape and 15.1 for the same-type day, battery 125.7 EUR/day (91% of perfect foresight) against 115.7 (83%) and 117.5 (85%), so it met the pre-registered rule. On the frozen holdout (102 days, 2026-07-01 to 2026-10-10, evaluated once) it does not confirm: shape MAE 22.4 against 23.1 and 22.0, battery 257.1 EUR/day (92% of perfect) against 265.4 (95%) and 266.5 (96%); in a period of very high and volatile prices the day-before shape was the better guide, and the model's spread error (30.6 against 25.4 and 23.6) says why. A paired comparison with a moving-block bootstrap (blocks of 7 days, 5,000 resamples) says the holdout battery differences are distinguishable from zero and against the model (about -8 EUR/day against yesterday's shape and -9 against the same-type day), while the holdout shape MAE differences are not distinguishable from zero. Both dashboards carry a "Shape and battery value" panel with these aggregates and the live record of the pre-market version of each settled day; `published/shape_battery.json` holds the aggregates and the live daily values, nothing from EEX.

Inputs: ENTSO-E day-ahead prices, day-ahead total load forecast and day-ahead wind and solar forecasts (RESTful API, cached under `data/cache/entsoe/`), Open-Meteo weather forecasts for the eight cities as they were issued two days ahead (previous-runs archive, cached under `data/cache/open-meteo/`), and two pre-gate generation proxies built the same way (`forecast/gen_proxy.py`): 100 m wind forecasts issued two days ahead at 17 points covering the French wind regions, passed through a turbine power curve (`forecast/wind_proxy.py`), and radiation forecasts issued two days ahead at 21 points covering the solar regions, as a ratio to 1,000 W/m2 (`forecast/solar_proxy.py`), each weighted in MW by non-negative least squares against ENTSO-E actual generation of that type, with the weights refitted at the start of each month on the trailing year. The calendar goes beyond hour, weekday and month: public holidays, the day type (working day, Saturday, Sunday or holiday), bridge days, the eve and the day after a holiday, the Christmas to New Year break, and the price of the most recent earlier day of the same type as an extra lag (`forecast/daytypes.py`). The honest feature set uses only inputs published before the gate; the `honest_wind` and `honest_base` variants drop the later additions so the backtest can measure them, and an extended set adds the ENTSO-E wind and solar forecasts, which the platform allows until 18:00 on D-1. A test fails if any feature for delivery day D is timestamped after 12:00 Paris on D-1.

## Scheduled jobs and dashboard

Two jobs keep a local SQLite database (`data/forecast.db`) up to date, and a Streamlit dashboard reads it. The jobs are the only code that calls the APIs.

```bash
pip install -e ".[dev,forecast,dashboard]"       # adds streamlit and plotly
thermo-fr morning-run                             # tomorrow's inputs, timing log, both forecasts (new version each run); --refit fits live
thermo-fr settle                                  # actual prices, then every unscored forecast version is scored
thermo-fr dashboard                               # http://localhost:8501
```

`morning-run` fetches the ENTSO-E day-ahead load forecast, wind and solar forecasts and prices around the next delivery day, and the Open-Meteo weather forecasts issued two days ahead. For each input it records whether it is present for the delivery day, its hour count, revision number and a hash of its values, so the timing log (table `timing_log`) shows when each input first appeared relative to the 12:00 Paris gate and whether it changed between runs. It then stores the hourly inputs and a forecast for the honest feature set, predicted with the stored model `published/model/honest_v2.txt` (the same file the GitHub Actions workflow predicts with, so a local run and the workflow give the same curve for the same inputs; when one of its inputs is missing for the day the run predicts with the fallback `published/model/honest.txt` and records the fallback in the data status; `--refit` fits the model live on the inputs history instead, and a missing model file is recorded and falls back to a live fit), and, when the wind and solar forecasts exist, for the extended set, which has no stored model and is always fitted live. Every forecast is a new version stamped with its issue time; nothing is overwritten. A source that is down is logged in `data_status` and the run finishes as `partial` or `failed` instead of crashing. `settle` fetches the actual prices for today, tomorrow and every forecast day, stores them, and scores each version against them and against the same-hour-previous-day baseline. Both commands log to `logs/<command>_<date>.log`.

Windows Task Scheduler entries (the jobs run as the current user and read `ENTSOE_API_KEY` from the user environment, so set it with `setx ENTSOE_API_KEY ...` once):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 -Install   # morning-run weekdays 07:00, 08:00, 09:15, 09:45, 10:30, 11:30; settle daily 14:00; market fetch weekdays 11:20 and 11:35; outage snapshot weekdays 09:00 and weekends 10:00; fuel snapshot daily 22:15 (private collector, only if local\fuel_fetch.py exists) (local time, London intended; 09:15 and 09:45 London are pre-market runs at 10:15 and 10:45 Paris, 11:20 and 11:35 London are 12:20 and 12:35 Paris)
powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 -Show
powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 -Remove
```

All five tasks have "run task as soon as possible after a scheduled start is missed" and "wake the computer to run this task" turned on. Waking from sleep or hibernation also needs Windows to allow wake timers: Power Options, Sleep, Allow wake timers set to Enable for both plugged in and on battery (`powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1`, the same with `/setdcvalueindex`, then `powercfg /setactive SCHEME_CURRENT`). A machine that is shut down cannot be woken by a task.

The dashboard shows tomorrow's pre-market forecast (the last version issued before the EEX market window opens at 11:15 Paris on the day before; versions issued later are listed underneath as "issued after the market window, not tradeable") with today's actual prices, the same-hour-previous-day baseline and a shaded band built from the backtest's error distribution at each hour (historical error, not a probability forecast); tomorrow's forecast load, wind, solar, residual load and temperature with the change against today's inputs; the last 30 settled days against actual prices with rolling MAE and the share of days whose MAE was below the naive baseline's; a "Versus the market" panel with the daily table and cumulative P&L of the forecast against the recorded EEX traded prices; the data status for tomorrow (arrival time of each input, anything missing or late), the market data collection log with any day the collector skipped or was refused, and the timing-probe summary across all logged days. A sidebar toggle switches between the honest and extended feature sets, with a note that the extended set may use information published after the gate until the timing log shows otherwise. Database reads are cached; the "Refresh now" button runs `morning-run` once.

![Dashboard](docs/img/dashboard.png)

## Public dashboard and automated updates

`streamlit_app.py` is a public version of the dashboard that reads only `published/` and runs on Streamlit Community Cloud. A GitHub Actions workflow keeps that folder current without any local machine: on weekday mornings it fetches the days around the next delivery day and publishes the forecast made with the stored model, every afternoon it fetches the auction results and scores the stored forecasts, and on the first weekday of each month it fetches the full history, refits the honest model and commits the model file with its metadata. The morning runs are timed for the market: GitHub cron is UTC, so four pre-market crons fire (08:10, 08:35, 09:10, 09:35 UTC) and `thermo-fr schedule-step` keeps only the ones that start between 10:05 and 11:00 Paris in the current season, after the 10:00 load forecast deadline and before the 11:15 market window; a later run at 10:30 UTC is kept for information only. The public app's headline for each day is the pre-market version. The token lives in a repository secret and is never printed or committed. `published/` holds the last 90 days of forecasts (with issue times), actual prices, the baseline, daily errors, tomorrow's latest forecast, the backtest error band and the model; no inputs history is kept in the repository. Of the market comparison the public app shows aggregates only (days scored, hit rate, mean P&L per MWh, model against market error), and only once 20 days are scored (`MIN_PUBLIC_DAYS` in `forecast/market.py`), because with a handful of days the mean P&L and the public auction result would let a reader back out the traded prices; until then it says "Versus the market: collecting data, n of 20 days". The prices themselves stay local.

```bash
thermo-fr publish                   # export the public dataset from the local database
thermo-fr import-published          # the reverse, used by the workflow to restore its state
thermo-fr refit-model               # fetch the full history and save published/model/honest_v2.txt and honest.txt, their metadata, wind_proxy.json and solar_proxy.json
thermo-fr morning-run --feature-sets honest_v2   # what the workflow runs: predict with the stored default model, honest.txt as fallback (the default locally too)
streamlit run streamlit_app.py      # the public app, locally
```

Setup steps (secret, Streamlit deployment, live link) and the data-terms notes are in [docs/public_dashboard.md](docs/public_dashboard.md).

## Glossary

Three kinds of price appear in this project and the words are kept apart throughout the code, the dashboards and these documents:

| Term | Meaning | Source here |
|---|---|---|
| **Fundamentals** | the inputs the forecast is built from: load forecast, actual and forecast wind and solar generation, prices of earlier days, weather | ENTSO-E Transparency Platform, RTE eCO2mix, Open-Meteo |
| **Spot**, or the **auction result** | the EPEX day-ahead clearing prices for France, published around 12:50 CET on the day before delivery; what the forecast tries to predict and what every error is measured against | ENTSO-E A44 (the same figures the exchange publishes) |
| **Market**, or the **traded price** | the EEX French Day-Ahead Base and Peak futures traded on the morning before the auction, liquid from about 08:00 to 12:30 Paris and tradeable until about 12:00; where the market already stands when the forecast is issued | EEX, entered by hand into a local, git-ignored file |

ENTSO-E data is never called market data here. The **baseline** is the naive forecast used for error comparisons: the spot price of the same hour on the previous day. **Trading value** is measured against the traded price, not against the baseline.

## Data sources

| Name | Series | Key | Where the data comes from |
|---|---|---|---|
| `rte` | load | none | RTE eCO2mix national consumption on ODRE, `eco2mix-national-cons-def` (definitive and consolidated) plus `eco2mix-national-tr` (real time) for the most recent weeks |
| `energy-charts` | load, prices | none | Energy-Charts API by Fraunhofer ISE: `/price?bzn=FR` and the `Load` series of `/public_power?country=fr` |
| `entsoe` | load, prices, day-ahead forecasts | `ENTSOE_API_KEY` | ENTSO-E Transparency Platform RESTful API, raw XML cached under `data/cache/entsoe/` |
| `csv` | load, prices | none | CSV files exported by hand from the ENTSO-E Transparency Platform website, placed in `--csv-dir` |

Raw responses from `rte` and `energy-charts` are cached under `data/cache/`, so a rerun downloads nothing. Delete that directory to refresh. Requests are spaced out to respect the published rate limits (about 2 per minute on the Energy-Charts price endpoint), and retried with exponential backoff on 429 and 503. If a service stays down, the fetch stops with a clear error instead of writing partial data.

### Attribution and licences

- Energy-Charts data is published by Fraunhofer ISE at https://energy-charts.info under the CC BY 4.0 licence. French day-ahead prices on Energy-Charts originate from Bundesnetzagentur | SMARD.de. Whenever this source is used the attribution is written into `sources.json`, `summary.json` and `summary.md`.
- RTE eCO2mix data is published on https://opendata.reseaux-energies.fr under the Licence Ouverte v2.0 (Etalab).
- Weather data by Open-Meteo.com, https://open-meteo.com, CC BY 4.0.
- ENTSO-E Transparency Platform data is subject to the platform's terms of use.

### ENTSO-E API key (only for the `entsoe` source)

1. Register for a free account at https://transparency.entsoe.eu.
2. Email transparency@entsoe.eu with the subject "Restful API access" and the email address of your account in the body.
3. Once access is granted, generate the token in your account settings, then:

```bash
export ENTSOE_API_KEY="your-token"   # Windows: set ENTSOE_API_KEY=your-token
```

### CSV exports from the ENTSO-E website (the `csv` source)

Export "Actual Total Load" and "Day-ahead Prices" for the France bidding zone as CSV and put the files in one directory. Files are recognised by their header, so names do not matter and one file per year is fine; empty files are skipped and timestamps repeated across files are dropped. The current export writes "MTU (UTC)" intervals such as "01/01/2025 00:00:00 - 01/01/2025 00:15:00", which are taken as UTC. Older exports with "MTU (CET/CEST)" or "Time (CET/CEST)" are parsed as Paris local time, and the repeated 02:00 hour on the autumn daylight-saving day is resolved by file order. Both 15-minute and hourly files work. When a Sequence column lists more than one auction, only the main day-ahead coupling result ("Without Sequence", else the lowest sequence number) is kept.

## Design choices

- **UTC everywhere, local time only for grouping into days.** The spring daylight-saving day has 23 hours and the autumn one has 25. Converting to Paris time only at the grouping step keeps both days intact, and the tests check this for every source.
- **Sub-hourly data is averaged to hourly.** The day-ahead market moved to 15-minute products in 2025, so prices and load are resampled to hourly means before anything else.
- **Fixed effects for each calendar month.** The slope is learned from day-to-day weather swings within a month, not from the gap between summer and winter. This stops changes in gas prices, demand trends or policy from leaking into the temperature estimate.
- **Population weighting.** Paris carries more weight than Strasbourg because that is where the heating load is. The weights are approximate metropolitan populations.
- **Weekday and public-holiday controls.** French holidays come from the `holidays` package.
- **Sources are interchangeable.** Every source implements the same two methods, `load(start, end)` and `day_ahead_prices(start, end)`, returning hourly tz-aware UTC series named `load_mw` and `price_eur_mwh`. A source that lacks a series says so instead of guessing.

## Limitations

- Standard errors are conditional on the chosen threshold, so they understate the true uncertainty a little.
- The model is daily. Hourly shape, especially the evening peak, would need a separate model.
- The price effect mixes the demand response with anything else that tends to happen on cold days (low wind, plant outages). Adding nuclear availability and renewable output would separate them.
- The out-of-sample check uses actual temperatures, so it measures the model, not the weather forecast.
- RTE's consolidated data omits the repeated hour on the autumn daylight-saving day, so that day has 24 of its 25 hours. It is kept, since the daily table only requires 22.

## Layout

```
src/thermo_fr/
  config.py              cities, weights, settings, plausibility ranges
  data/sources.py        the Source interface and get_source() factory
  data/http.py           rate-limited GET with retries, plus the raw-response cache
  data/weather.py        Open-Meteo temperatures and population weighting
  data/entsoe_rest.py    ENTSO-E RESTful API client: prices, load, day-ahead forecasts (needs a key)
  data/entsoe_client.py  the entsoe source built on it
  data/weather_forecast.py Open-Meteo forecasts as issued (previous runs) and the historical-forecast proxy
  data/wind_points.py    100 m wind forecasts as issued at 17 points in the wind regions
  data/solar_points.py   radiation forecasts as issued at 21 points in the solar regions
  forecast/timing.py     the 12:00 Paris gate, issue-time rules, look-ahead check
  forecast/inputs.py     one hourly table of every forecast input, plus the API-versus-CSV price comparison
  forecast/daytypes.py   day types, bridge days, holiday neighbours, the comparable day of the same type
  forecast/features.py   honest and extended feature sets in Paris delivery hours
  forecast/gen_proxy.py  shared method of the generation proxies: NNLS weights per point, monthly point-in-time refits
  forecast/wind_proxy.py pre-gate wind generation proxy: turbine power curve on 100 m wind forecasts
  forecast/solar_proxy.py pre-gate solar generation proxy: irradiance ratio on radiation forecasts
  forecast/models.py     benchmarks, LightGBM, ridge
  forecast/backtest.py   monthly walk-forward, metrics by slice, worst days
  forecast/report.py     reports/forecast/ tables and charts
  forecast/day.py        one delivery day as of 12:00 the day before
  forecast/probe.py      timing probe for tomorrow's ENTSO-E items (superseded by morning-run)
  forecast/store.py      SQLite store: runs, data status, timing log, inputs, forecast versions, actuals, scores, error band
  forecast/jobs.py       morning-run and settle
  data/outages.py        daily raw snapshots of the ENTSO-E unavailability notices of French units, and planned nuclear availability as of a time
  forecast/probabilistic.py  quantile forecasts with a conformal 10-90 band, negative-price and spike probabilities, their benchmarks, metrics and model files
  forecast/shape.py      the shape of the day (price minus the day's mean): shape model, D-1 and same-type benchmarks, metrics, battery backtest
  forecast/market.py     EEX traded prices (local file: collected, pasted or typed), the paste parser, the collector contract and log, the forecast scored against them, the pre-market headline rule
  forecast/schedule.py   which step a scheduled GitHub Actions run performs, from the Paris clock (pre-market window 10:05 to 11:00)
  dashboard/data.py      read-only queries for the dashboard
  dashboard/app.py       the Streamlit app
scripts/schedule_tasks.ps1   install, show or remove the Task Scheduler entries
scripts/install_hooks.py     install the local git pre-commit hook that runs the test suite and blocks a commit on any failure (python scripts/install_hooks.py; local only, not part of the workflow)
local/                 git-ignored: the EEX collector module (`market fetch` loads it from here) and the private collector and loader for gas and carbon prices; nothing in it is committed. The public timing rules fuel_index_lag1 (end-of-day indices known from 22:00 CET on D-2) and fuel_morning_trades (D-1 trades up to the issue time) in forecast/timing.py are what the private loader checks itself against
scripts/screenshot_dashboard.py  full-page screenshot of the running dashboard
forecast/publish.py      export and import of the public dataset under published/
forecast/refit.py        monthly refit: model file and metadata under published/model/
streamlit_app.py         public dashboard reading published/ only (Streamlit Community Cloud entry point)
.github/workflows/forecast.yml  scheduled morning-run and settle, committing published/
  data/energy_charts.py  Energy-Charts load and prices (no key)
  data/rte_eco2mix.py    RTE eCO2mix load from ODRE (no key)
  data/csv_source.py     ENTSO-E CSV exports made by hand
  data/quality.py        missing-hour and plausibility checks
  data/dataset.py        hourly to daily, DST-safe, calendar features
  model.py               threshold search and regression
  report.py              summary, charts, out-of-sample check
  synthetic.py           synthetic data with known answers
  cli.py                 command line
tests/                   DST handling, weighting, parameter recovery, every source offline
tests/fixtures/          small ENTSO-E style CSV samples covering both DST days
```

## Licence

The code is released under the MIT licence (see `LICENSE`). The data keep their own licences, listed under Attribution and licences above.
