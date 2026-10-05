# thermo-fr

How much does French electricity demand rise when it gets colder, and what does that do to the day-ahead price?

France heats a large share of its homes with electricity, so its demand is unusually sensitive to temperature. RTE usually puts the winter figure at roughly 2,400 MW for each degree colder. This project estimates that number from public data, along with the matching day-ahead price effect, and checks the model out of sample.

## What it does

1. Downloads hourly French actual load and day-ahead prices from one of several pluggable sources (see below), and hourly temperatures for eight French cities from the Open-Meteo archive.
2. Builds a population-weighted national temperature and averages everything into local calendar days.
3. Fits a piecewise-linear model. Above a threshold temperature, weather barely moves demand. Below it, each degree colder adds a fixed amount of load. The threshold is chosen by grid search.
4. Fits the same shape to the day-ahead price, using the load model's threshold.
5. Trains on all but the last 12 months and forecasts the last 12 months of load to check the model out of sample.
6. Writes a summary and two charts to `reports/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ",entsoe" inside the brackets for the ENTSO-E client
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

## Data sources

| Name | Series | Key | Where the data comes from |
|---|---|---|---|
| `rte` | load | none | RTE eCO2mix national consumption on ODRE, `eco2mix-national-cons-def` (definitive and consolidated) plus `eco2mix-national-tr` (real time) for the most recent weeks |
| `energy-charts` | load, prices | none | Energy-Charts API by Fraunhofer ISE: `/price?bzn=FR` and the `Load` series of `/public_power?country=fr` |
| `entsoe` | load, prices | `ENTSOE_API_KEY` | ENTSO-E Transparency Platform through entsoe-py |
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

Export "Actual Total Load" and "Day-ahead Prices" for the France bidding zone as CSV and put the files in one directory. Files are recognised by their header, so names do not matter and one file per year is fine. The interval column ("MTU (CET/CEST)" or "Time (CET/CEST)") is parsed as Paris local time; the repeated 02:00 hour on the autumn daylight-saving day is resolved by file order. Both 15-minute and hourly files work.

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
  data/entsoe_client.py  ENTSO-E load and prices (needs a key)
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
