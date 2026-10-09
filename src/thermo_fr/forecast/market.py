"""The forecast against the traded price: EEX French day-ahead futures before the auction.

Three kinds of price appear in this project and the words are kept apart:

- fundamentals: the ENTSO-E, RTE and weather inputs the forecast is built from;
- spot, or the auction result: the EPEX day-ahead clearing prices, published
  around 12:50 CET on the day before delivery;
- market, or the traded price: the EEX French Day-Ahead Base and Peak futures,
  traded on the morning before the auction (liquid from about 08:00 to 12:30
  Paris, tradeable until about 12:00).

A forecast error against yesterday's price says nothing about trading value.
The test that does is whether a forecast issued before the trading window
points the right way relative to where the market already trades, and what
that would have earned: direction = long if forecast > entry, short if below;
P&L per MWh = (auction result - entry) x direction. Entry is the VWAP of the
window, with the close reported as an alternative.

Traded prices are entered by hand into data/market/eex_fr_da.csv, which is
git-ignored (the whole data/ tree is). They are never committed, published or
shown in the public app; the public dataset carries aggregates only (days
scored, hit rate, mean P&L per MWh, model and market error). MIN_PUBLIC_DAYS
is the number of scored days required before those aggregates appear: with
one or two days the mean P&L and the public auction result let a reader back
out the traded price, so raise it if that matters.

The window lies on the day before delivery, in Paris time. For each market
row the forecast used is the latest honest version issued before the window
opened, so every day is scored with the model that was live at the time; a
day whose only forecasts came later is reported as skipped, never scored
with hindsight.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .store import Store

DEFAULT_PATH = Path("data/market/eex_fr_da.csv")
COLUMNS = ["delivery_date", "product", "window_start", "window_end", "open", "high", "low", "close", "vwap", "source", "note"]
PRODUCTS = ("base", "peak")
PEAK_HOURS = range(8, 20)  # 08:00 to 20:00 Paris, the EEX / EPEX peak block
DEFAULT_BANDS = (0.0, 1.0, 2.0, 5.0, 10.0)  # no-trade bands in EUR/MWh, tried in sample
MIN_PUBLIC_DAYS = 1


class MarketDataError(ValueError):
    """A market row is malformed or duplicates a stored one."""


class NoForecastBeforeWindowError(LookupError):
    """No forecast of the requested set was issued before the trading window opened."""


@dataclass(frozen=True)
class MarketRow:
    delivery_date: str
    product: str
    window_start: str
    window_end: str
    open: float
    high: float
    low: float
    close: float
    vwap: float
    source: str
    note: str = ""

    def validate(self) -> "MarketRow":
        try:
            pd.Timestamp(self.delivery_date).strftime("%Y-%m-%d")
        except (ValueError, TypeError) as exc:
            raise MarketDataError(f"delivery_date {self.delivery_date!r} is not a date") from exc
        if self.product not in PRODUCTS:
            raise MarketDataError(f"product must be one of {PRODUCTS}, not {self.product!r}")
        for name in ("window_start", "window_end"):
            value = getattr(self, name)
            if not (len(value) == 5 and value[2] == ":" and value[:2].isdigit() and value[3:].isdigit() and int(value[:2]) < 24 and int(value[3:]) < 60):
                raise MarketDataError(f"{name} must be HH:MM Paris time, not {value!r}")
        if self.window_end <= self.window_start:
            raise MarketDataError("window_end must be after window_start")
        prices = {k: float(getattr(self, k)) for k in ("open", "high", "low", "close", "vwap")}
        if any(np.isnan(v) for v in prices.values()):
            raise MarketDataError("open, high, low, close and vwap are all required")
        if prices["low"] > prices["high"]:
            raise MarketDataError("low is above high")
        for k in ("open", "close", "vwap"):
            if not prices["low"] <= prices[k] <= prices["high"]:
                raise MarketDataError(f"{k} {prices[k]} lies outside the low to high range")
        if not self.source:
            raise MarketDataError("source is required (where the traded prices came from)")
        return self


def parse_window(window: str) -> tuple[str, str]:
    """'11:15-12:00' -> ('11:15', '12:00')."""
    parts = window.replace(" ", "").split("-")
    if len(parts) != 2:
        raise MarketDataError(f"window must look like 11:15-12:00, not {window!r}")
    return parts[0], parts[1]


def load_market(path=DEFAULT_PATH) -> pd.DataFrame:
    """The stored rows, or an empty frame with the right columns."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path, dtype={"delivery_date": str, "product": str, "window_start": str, "window_end": str, "source": str, "note": str})
    frame["note"] = frame["note"].fillna("")
    return frame.reindex(columns=COLUMNS)


def add_row(row: MarketRow, path=DEFAULT_PATH) -> pd.DataFrame:
    """Validate and append one row; the same delivery date, product and window may only be stored once."""
    row = row.validate()
    path = Path(path)
    existing = load_market(path)
    same = existing[(existing["delivery_date"] == row.delivery_date) & (existing["product"] == row.product)
                    & (existing["window_start"] == row.window_start) & (existing["window_end"] == row.window_end)]
    if not same.empty:
        raise MarketDataError(f"a {row.product} row for {row.delivery_date} with window {row.window_start}-{row.window_end} is already stored")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: getattr(row, c) for c in COLUMNS})
    return load_market(path)


def window_bounds(delivery_date: str, window_start: str, window_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The trading window, on the day before delivery in Paris time, as UTC timestamps."""
    day_before = pd.Timestamp(delivery_date) - pd.Timedelta(days=1)
    start = pd.Timestamp(f"{day_before:%Y-%m-%d} {window_start}", tz=LOCAL_TZ)
    end = pd.Timestamp(f"{day_before:%Y-%m-%d} {window_end}", tz=LOCAL_TZ)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def product_mask(index: pd.DatetimeIndex, product: str) -> np.ndarray:
    """Which hours of a delivery day belong to the product: all of them for base, 08:00 to 20:00 Paris for peak."""
    if product == "base":
        return np.ones(len(index), dtype=bool)
    if product == "peak":
        local = pd.DatetimeIndex(index).tz_convert(LOCAL_TZ)
        return np.asarray(local.hour.isin(list(PEAK_HOURS)))
    raise MarketDataError(f"unknown product {product!r}")


def product_average(series: pd.Series, product: str) -> float:
    """Mean over the product's hours; the autumn day has 25 base hours and the spring day 23, peak always 12."""
    if series.empty:
        return float("nan")
    values = series[product_mask(series.index, product)].dropna()
    return float(values.mean()) if len(values) else float("nan")


def forecast_before(store: Store, delivery_date: str, before: pd.Timestamp, feature_set: str = "honest") -> tuple[dict, pd.DataFrame]:
    """The latest forecast version of the day issued strictly before `before` (UTC)."""
    versions = store.forecast_versions(delivery_date, feature_set)
    if versions.empty:
        raise NoForecastBeforeWindowError(f"no {feature_set} forecast stored for {delivery_date}")
    issued = pd.to_datetime(versions["issued_at_utc"], utc=True)
    earlier = versions[issued < before]
    if earlier.empty:
        first = issued.min().tz_convert(LOCAL_TZ)
        raise NoForecastBeforeWindowError(
            f"no {feature_set} forecast for {delivery_date} issued before the window opened at {before.tz_convert(LOCAL_TZ):%H:%M} Paris "
            f"(earliest version {first:%Y-%m-%d %H:%M} Paris)")
    meta = earlier.iloc[-1].to_dict()
    return meta, store.forecast_curve(int(meta["forecast_id"]))


def direction_of(forecast: float, entry: float) -> int:
    """1 for long (forecast above the entry), -1 for short, 0 when equal or unknown."""
    if np.isnan(forecast) or np.isnan(entry) or forecast == entry:
        return 0
    return 1 if forecast > entry else -1


def evaluate_day(store: Store, row: pd.Series | MarketRow, feature_set: str = "honest") -> dict:
    """Score one market row with the forecast that was live before its window. Raises NoForecastBeforeWindowError."""
    r = row if isinstance(row, MarketRow) else MarketRow(**{c: row[c] for c in COLUMNS})
    start, end = window_bounds(r.delivery_date, r.window_start, r.window_end)
    meta, curve = forecast_before(store, r.delivery_date, start, feature_set)
    forecast = product_average(curve["forecast"], r.product)
    auction = product_average(store.actuals_for(r.delivery_date), r.product)
    entry, close = float(r.vwap), float(r.close)
    direction = direction_of(forecast, entry)
    direction_close = direction_of(forecast, close)
    out = {
        "delivery_date": r.delivery_date,
        "product": r.product,
        "window_paris": f"{r.window_start}-{r.window_end}",
        "issued_at_utc": meta["issued_at_utc"],
        "issued_paris": pd.Timestamp(meta["issued_at_utc"]).tz_convert(LOCAL_TZ).strftime("%H:%M"),
        "model": meta["model"],
        "forecast": round(forecast, 2),
        "entry_vwap": entry,
        "entry_close": close,
        "auction": round(auction, 2) if not np.isnan(auction) else np.nan,
        "direction": {1: "long", -1: "short", 0: "none"}[direction],
        "signal": round(forecast - entry, 2),
        "pnl_per_mwh": round((auction - entry) * direction, 2) if direction and not np.isnan(auction) else np.nan,
        "pnl_per_mwh_close": round((auction - close) * direction_close, 2) if direction_close and not np.isnan(auction) else np.nan,
        "model_error": round(abs(forecast - auction), 2) if not np.isnan(auction) else np.nan,
        "market_error": round(abs(entry - auction), 2) if not np.isnan(auction) else np.nan,
    }
    out["model_beats_market"] = (out["model_error"] < out["market_error"]) if not np.isnan(auction) else np.nan
    return out


def evaluate(store: Store, market: pd.DataFrame, feature_set: str = "honest", bands=DEFAULT_BANDS) -> dict:
    """Every stored market row against the live forecast. Returns table, summary, bands and skipped days."""
    rows, skipped = [], []
    for _, r in market.sort_values(["delivery_date", "product"]).iterrows():
        try:
            rows.append(evaluate_day(store, r, feature_set))
        except NoForecastBeforeWindowError as exc:
            skipped.append({"delivery_date": r["delivery_date"], "product": r["product"], "reason": str(exc)})
    table = pd.DataFrame(rows)
    summary = {"feature_set": feature_set, "days": int(len(table)), "scored_days": 0, "skipped": skipped}
    band_table = pd.DataFrame(columns=["band_eur_mwh", "trades", "hit_rate", "total_pnl_per_mwh", "mean_pnl_per_mwh"])
    if not table.empty:
        table["cumulative_pnl_per_mwh"] = table["pnl_per_mwh"].fillna(0).cumsum().round(2)
        scored = table.dropna(subset=["pnl_per_mwh"])
        summary["scored_days"] = int(len(scored))
        if len(scored):
            summary.update({
                "hit_rate": round(float((scored["pnl_per_mwh"] > 0).mean()), 3),
                "total_pnl_per_mwh": round(float(scored["pnl_per_mwh"].sum()), 2),
                "mean_pnl_per_mwh": round(float(scored["pnl_per_mwh"].mean()), 2),
                "mean_pnl_per_mwh_close": round(float(scored["pnl_per_mwh_close"].mean()), 2),
                "model_mae": round(float(scored["model_error"].mean()), 2),
                "market_mae": round(float(scored["market_error"].mean()), 2),
                "share_model_beats_market": round(float(scored["model_beats_market"].astype(float).mean()), 3),
            })
            band_rows = []
            for band in bands:
                traded = scored[scored["signal"].abs() > band]
                band_rows.append({
                    "band_eur_mwh": band, "trades": int(len(traded)),
                    "hit_rate": round(float((traded["pnl_per_mwh"] > 0).mean()), 3) if len(traded) else np.nan,
                    "total_pnl_per_mwh": round(float(traded["pnl_per_mwh"].sum()), 2) if len(traded) else 0.0,
                    "mean_pnl_per_mwh": round(float(traded["pnl_per_mwh"].mean()), 2) if len(traded) else np.nan,
                })
            band_table = pd.DataFrame(band_rows)
    return {"table": table, "summary": summary, "bands": band_table, "bands_note": "No-trade bands are chosen in sample on the same days; "
            "they describe the sample, not a rule that was tested out of sample."}


def public_summary(result: dict, min_days: int = MIN_PUBLIC_DAYS) -> dict:
    """What the public dataset may carry: aggregates only, and only from min_days scored days; never a traded price."""
    s = result["summary"]
    out = {"source": "EEX French Day-Ahead Base and Peak futures traded before the auction, entered by hand; "
                     "traded prices are not republished.", "scored_days": s["scored_days"], "min_days_to_show": min_days}
    if s["scored_days"] >= min_days:
        out.update({k: s[k] for k in ("hit_rate", "mean_pnl_per_mwh", "model_mae", "market_mae", "share_model_beats_market")})
    else:
        out["note"] = f"Aggregates are shown once at least {min_days} days are scored; with fewer, they would reveal the traded prices."
    return out


def format_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "no market rows scored"
    columns = ["delivery_date", "product", "window_paris", "issued_paris", "model", "forecast", "entry_vwap", "auction", "direction",
               "pnl_per_mwh", "cumulative_pnl_per_mwh", "model_error", "market_error", "model_beats_market", "pnl_per_mwh_close"]
    return table[columns].to_string(index=False, na_rep="")
