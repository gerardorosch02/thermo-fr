# Data: sources, terms, files and glossary

Moved from the README on 2026-10-15. The files of the public dataset under `published/` are described in [public_dashboard.md](public_dashboard.md#what-is-published).

## Glossary

Three kinds of price appear in this project and the words are kept apart throughout the code, the dashboards and these documents:

| Term | Meaning | Source here |
|---|---|---|
| **Fundamentals** | the inputs the forecast is built from: load forecast, actual and forecast wind and solar generation, prices of earlier days, weather | ENTSO-E Transparency Platform, RTE eCO2mix, Open-Meteo |
| **Spot**, or the **auction result** | the EPEX day-ahead clearing prices for France, published around 12:50 CET on the day before delivery; what the forecast tries to predict and what every error is measured against | ENTSO-E A44 (the same figures the exchange publishes) |
| **Market**, or the **traded price** | the EEX French Day-Ahead Base and Peak futures traded on the morning before the auction, liquid from about 08:00 to 12:30 Paris and tradeable until about 12:00; where the market already stands when the forecast is issued | EEX, entered by hand into a local, git-ignored file |

ENTSO-E data is never called market data here. The **benchmark** of the level forecast is the traded price: the window VWAP of the EEX future, scored at the auction result (hit rate, mean P&L per MWh, model error against the auction next to the market's, window count). **Forecast error** (MAE against the auction result) is the model's own error and is shown with no other comparison; until 2026-10-12 it was compared with the spot price of the same hour on the previous day, a figure that is no longer published and survives only in the historical rows of docs/experiments.md and docs/forecast.md.

## Forecast inputs

Inputs: ENTSO-E day-ahead prices, day-ahead total load forecast and day-ahead wind and solar forecasts (RESTful API, cached under `data/cache/entsoe/`), Open-Meteo weather forecasts for the eight cities as they were issued two days ahead (previous-runs archive, cached under `data/cache/open-meteo/`), and two pre-gate generation proxies built the same way (`forecast/gen_proxy.py`): 100 m wind forecasts issued two days ahead at 17 points covering the French wind regions, passed through a turbine power curve (`forecast/wind_proxy.py`), and radiation forecasts issued two days ahead at 21 points covering the solar regions, as a ratio to 1,000 W/m2 (`forecast/solar_proxy.py`), each weighted in MW by non-negative least squares against ENTSO-E actual generation of that type, with the weights refitted at the start of each month on the trailing year. The calendar goes beyond hour, weekday and month: public holidays, the day type (working day, Saturday, Sunday or holiday), bridge days, the eve and the day after a holiday, the Christmas to New Year break, and the price of the most recent earlier day of the same type as an extra lag (`forecast/daytypes.py`). The honest feature set uses only inputs published before the gate; the `honest_wind` and `honest_base` variants drop the later additions so the backtest can measure them, and an extended set adds the ENTSO-E wind and solar forecasts, which the platform allows until 18:00 on D-1. A test fails if any feature for delivery day D is timestamped after 12:00 Paris on D-1.

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

## Traded prices (local only)

Traded prices live in `data/market/eex_fr_da.csv` (columns: delivery_date, product, trade_date, window_start, window_end in Paris time, open, high, low, close, vwap, trades, volume_mwh, source, note; the trade date is the day before delivery, or the Friday before a Saturday, Sunday or Monday delivery, since the day futures for those three trade on Friday). The whole `data/` tree is git-ignored and a test checks that this file is; raw EEX prices are never committed, published or shown in the public app.
