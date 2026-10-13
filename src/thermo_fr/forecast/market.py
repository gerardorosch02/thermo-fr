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

Traded prices live in data/market/eex_fr_da.csv, which is git-ignored (the
whole data/ tree is). They get there in one of three ways: `thermo-fr market
fetch`, which runs a local collector module kept outside the repository (see
run_fetcher: the module reads EEX's public market data page, whose data is
for information purposes and personal use, so neither the collector nor its
raw responses are ever committed); `thermo-fr market paste`, which parses the
free text a trader sends; or `thermo-fr market add` with the numbers typed
out. They are never committed, published or shown in the public app; the public dataset carries aggregates only (days
scored, hit rate, mean P&L per MWh, model and market error), and only once
MIN_PUBLIC_DAYS days are scored: with a handful of days the mean P&L and the
public auction result would let a reader back out the traded prices. Until
then the public app says "collecting data, n of 20 days".

The market window also defines the headline forecast of every delivery day:
the last version issued before MARKET_WINDOW_START Paris on the day before
delivery is the "pre-market forecast", the one a trader could have acted on.
Versions issued later are shown underneath as "issued after the market
window, not tradeable". headline_version() applies the rule to any table of
versions.

The window lies on the trade date, in Paris time: the day before delivery
for a weekday, the Friday before for Saturday, Sunday and Monday deliveries
(the day futures for the three trade on Friday). For each market row the
forecast used is the latest honest version issued before the window opened, so every day is scored with the model that was live at the time; a
day whose only forecasts came later is reported as skipped, never scored
with hindsight.

A second, smaller file, data/market/eex_fr_da_early.csv (git-ignored too),
holds the laptop's snapshot of the base tape taken before the window opens:
the latest trade before MARKET_WINDOW_START Paris, with its time. The local
dashboard's brief shows the gap between the fair value and that trade; the
window close, which comes after the forecast is issued, is shown only as a
check when no snapshot exists. See run_snapshot. Nothing of either file is
exported.
"""

import datetime as dt
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .store import Store

DEFAULT_PATH = Path("data/market/eex_fr_da.csv")
FETCH_LOG_PATH = Path("data/market/fetch_log.csv")
RAW_DIR = Path("data/market/raw")
DEFAULT_PLUGIN = Path("local/eex_fetch.py")  # git-ignored; see run_fetcher
COLUMNS = ["delivery_date", "product", "trade_date", "window_start", "window_end", "open", "high", "low", "close", "vwap",
           "trades", "volume_mwh", "source", "note"]
LOG_COLUMNS = ["logged_at_utc", "delivery_date", "product", "status", "message"]
EARLY_PATH = Path("data/market/eex_fr_da_early.csv")  # git-ignored: the latest base trade before the window, from the morning snapshot
EARLY_COLUMNS = ["delivery_date", "product", "trade_date", "snapshot_at_utc", "trade_time_utc", "trade_time_paris", "price", "volume_mwh",
                 "trades_before_window", "volume_before_window_mwh", "source", "note"]
SNAPSHOT_PRODUCTS = ("base",)
FETCH_STATUSES = ("stored", "exists", "skipped", "error")
PRODUCTS = ("base", "peak")
PEAK_HOURS = range(8, 20)  # 08:00 to 20:00 Paris, the EEX / EPEX peak block
DEFAULT_BANDS = (0.0, 1.0, 2.0, 5.0, 10.0)  # no-trade bands in EUR/MWh, tried in sample
MIN_PUBLIC_DAYS = 20
MARKET_WINDOW_START = "11:15"  # Paris, on the day before delivery: the EEX day-ahead future is liquid from about then
MARKET_WINDOW_END = "12:00"


class MarketDataError(ValueError):
    """A market row is malformed or duplicates a stored one."""


class NoForecastBeforeWindowError(LookupError):
    """No forecast of the requested set was issued before the trading window opened."""


def _blank(value) -> bool:
    """None, pandas NA, NaN or an empty string: the ways a missing optional field arrives from a CSV or a command line."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


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
    trade_date: str = ""  # the day the window was traded; empty means the default, see trade_date_for()
    trades: int | None = None  # trades in the window, when the row was rebuilt from a tape
    volume_mwh: float | None = None  # volume traded in the window, MWh

    def __post_init__(self):
        for name in ("trade_date", "note"):
            if _blank(getattr(self, name)):
                object.__setattr__(self, name, "")
        for name in ("trades", "volume_mwh"):
            if _blank(getattr(self, name)):
                object.__setattr__(self, name, None)
        if self.trades is not None:
            object.__setattr__(self, "trades", int(self.trades))
        if self.volume_mwh is not None:
            object.__setattr__(self, "volume_mwh", float(self.volume_mwh))

    @property
    def traded_on(self) -> str:
        return self.trade_date or trade_date_for(self.delivery_date)

    def validate(self) -> "MarketRow":
        try:
            pd.Timestamp(self.delivery_date).strftime("%Y-%m-%d")
        except (ValueError, TypeError) as exc:
            raise MarketDataError(f"delivery_date {self.delivery_date!r} is not a date") from exc
        if self.trade_date:
            try:
                traded = pd.Timestamp(self.trade_date)
            except (ValueError, TypeError) as exc:
                raise MarketDataError(f"trade_date {self.trade_date!r} is not a date") from exc
            if traded >= pd.Timestamp(self.delivery_date):
                raise MarketDataError(f"trade_date {self.trade_date} must be before delivery_date {self.delivery_date}")
        if self.trades is not None and self.trades < 0:
            raise MarketDataError("trades cannot be negative")
        if self.volume_mwh is not None and self.volume_mwh < 0:
            raise MarketDataError("volume_mwh cannot be negative")
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


def trade_date_for(delivery_date) -> str:
    """The day a delivery day's future is traded: the day before, or the Friday before a Saturday, Sunday or Monday delivery."""
    day = pd.Timestamp(delivery_date).normalize() - pd.Timedelta(days=1)
    while day.weekday() > 4:
        day -= pd.Timedelta(days=1)
    return day.strftime("%Y-%m-%d")


def delivery_days_to_collect(today=None) -> list[str]:
    """The delivery days whose futures trade today: tomorrow, and on a Friday also Sunday and Monday."""
    now = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).normalize()
    today_text = now.strftime("%Y-%m-%d")
    days = [(now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
    for ahead in (2, 3):
        candidate = (now + pd.Timedelta(days=ahead)).strftime("%Y-%m-%d")
        if trade_date_for(candidate) == today_text:
            days.append(candidate)
    return days


def load_market(path=DEFAULT_PATH) -> pd.DataFrame:
    """The stored rows, or an empty frame with the right columns. Files written before trade_date, trades and volume_mwh existed are read too."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path, dtype={"delivery_date": str, "product": str, "trade_date": str, "window_start": str, "window_end": str,
                                     "source": str, "note": str})
    frame = frame.reindex(columns=COLUMNS)
    frame["note"] = frame["note"].fillna("")
    frame["trade_date"] = frame["trade_date"].fillna("")
    blank = frame["trade_date"] == ""
    frame.loc[blank, "trade_date"] = [trade_date_for(d) for d in frame.loc[blank, "delivery_date"]]
    frame["trades"] = pd.to_numeric(frame["trades"], errors="coerce").astype("Int64")
    frame["volume_mwh"] = pd.to_numeric(frame["volume_mwh"], errors="coerce")
    return frame


def write_market(frame: pd.DataFrame, path=DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=COLUMNS).to_csv(path, index=False, lineterminator="\n")


def find_same(existing: pd.DataFrame, row: MarketRow) -> pd.DataFrame:
    return existing[(existing["delivery_date"] == row.delivery_date) & (existing["product"] == row.product)
                    & (existing["window_start"] == row.window_start) & (existing["window_end"] == row.window_end)]


def add_row(row: MarketRow, path=DEFAULT_PATH) -> pd.DataFrame:
    """Validate and append one row; the same delivery date, product and window may only be stored once."""
    row = row.validate()
    existing = load_market(path)
    if not find_same(existing, row).empty:
        raise MarketDataError(f"a {row.product} row for {row.delivery_date} with window {row.window_start}-{row.window_end} is already stored")
    values = {c: getattr(row, c) for c in COLUMNS}
    values["trade_date"] = row.traded_on
    frame = pd.concat([existing, pd.DataFrame([values])], ignore_index=True) if len(existing) else pd.DataFrame([values])
    write_market(frame, path)
    return load_market(path)


def row_from_series(row: pd.Series) -> MarketRow:
    return MarketRow(**{c: row[c] for c in COLUMNS if c in row.index})


def window_bounds(delivery_date: str, window_start: str, window_end: str, trade_date: str = "") -> tuple[pd.Timestamp, pd.Timestamp]:
    """The trading window on the trade date (default: trade_date_for the delivery day) in Paris time, as UTC timestamps."""
    traded = pd.Timestamp(trade_date) if trade_date else pd.Timestamp(trade_date_for(delivery_date))
    start = pd.Timestamp(f"{traded:%Y-%m-%d} {window_start}", tz=LOCAL_TZ)
    end = pd.Timestamp(f"{traded:%Y-%m-%d} {window_end}", tz=LOCAL_TZ)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def premarket_cutoff(delivery_date) -> pd.Timestamp:
    """The moment the market window opens for a delivery day: MARKET_WINDOW_START Paris on the day before, in UTC."""
    day_before = pd.Timestamp(delivery_date) - pd.Timedelta(days=1)
    return pd.Timestamp(f"{day_before:%Y-%m-%d} {MARKET_WINDOW_START}", tz=LOCAL_TZ).tz_convert("UTC")


def premarket_flag(delivery_days, issued_at_utc) -> np.ndarray:
    """True where a version was issued before its delivery day's market window opened."""
    days = pd.DatetimeIndex(pd.to_datetime(list(delivery_days)))
    issued = pd.DatetimeIndex(pd.to_datetime(list(issued_at_utc), utc=True))
    if len(days) == 0:
        return np.zeros(0, dtype=bool)
    cutoffs = pd.DatetimeIndex([premarket_cutoff(d) for d in days])
    return np.asarray(issued < cutoffs)


def headline_version(versions: pd.DataFrame, day_column: str = "delivery_day", issued_column: str = "issued_at_utc") -> pd.DataFrame:
    """One row per delivery day: the last version issued before the market window, else the latest version.

    The returned frame carries a boolean `premarket` column saying which rule
    applied, so a dashboard can label the row "pre-market forecast" or
    "issued after the market window, not tradeable".
    """
    if versions.empty:
        out = versions.copy()
        out["premarket"] = pd.Series(dtype=bool)
        return out
    frame = versions.copy()
    frame["premarket"] = premarket_flag(frame[day_column], frame[issued_column])
    frame = frame.sort_values([day_column, issued_column])
    picked = []
    for _, group in frame.groupby(day_column, sort=True):
        before = group[group["premarket"]]
        picked.append(before.iloc[-1] if len(before) else group.iloc[-1])
    return pd.DataFrame(picked).reset_index(drop=True)


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


class PublishedVersions:
    """The versions the public has seen, with this machine's auction results: what the public market record is scored against.

    Store-like for forecast_before and evaluate_day. The versions come from published/forecasts.csv on origin/main, archived in the
    store by sync_published_versions so that days older than the 90 the file keeps stay scorable. A delivery day whose published
    versions were all issued after the window is not counted, even when this machine holds an earlier local version.
    """

    LABEL = "published pre-market versions (the public site's)"

    def __init__(self, store: Store):
        self.store = store

    def forecast_versions(self, delivery_day: str, feature_set: str | None = None) -> pd.DataFrame:
        return self.store.published_versions(delivery_day, feature_set)

    def forecast_curve(self, forecast_id: int) -> pd.DataFrame:
        return self.store.published_curve(forecast_id)

    def actuals_for(self, delivery_day: str) -> pd.Series:
        return self.store.actuals_for(delivery_day)


LOCAL_LABEL = "local versions (this machine's morning runs)"


def sync_published_versions(store: Store, forecasts_csv, now=None) -> int:
    """Archive the versions of a published forecasts.csv (a path or its text); returns how many were new."""
    import io

    source = io.StringIO(forecasts_csv) if isinstance(forecasts_csv, str) and "\n" in forecasts_csv else forecasts_csv
    frame = pd.read_csv(source)
    needed = {"delivery_day", "issued_at_utc", "feature_set", "timestamp_utc", "hour", "forecast"}
    if not needed <= set(frame.columns):
        raise MarketDataError(f"published forecasts file lacks {sorted(needed - set(frame.columns))}")
    return store.save_published_versions(frame, now=now)


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
    r = row if isinstance(row, MarketRow) else row_from_series(row)
    start, end = window_bounds(r.delivery_date, r.window_start, r.window_end, r.trade_date)
    meta, curve = forecast_before(store, r.delivery_date, start, feature_set)
    forecast = product_average(curve["forecast"], r.product)
    auction = product_average(store.actuals_for(r.delivery_date), r.product)
    entry, close = float(r.vwap), float(r.close)
    direction = direction_of(forecast, entry)
    direction_close = direction_of(forecast, close)
    out = {
        "delivery_date": r.delivery_date,
        "product": r.product,
        "trade_date": r.traded_on,
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
        "window_trades": r.trades,
        "window_volume_mwh": r.volume_mwh,
        "source": r.source,
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
    summary = {"feature_set": feature_set, "versions": getattr(store, "LABEL", LOCAL_LABEL), "days": int(len(table)), "scored_days": 0,
               "skipped": skipped}
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
    out = {"source": "EEX French Day-Ahead Base and Peak futures traded before the auction, collected locally; "
                     "traded prices are not republished.", "scored_days": s["scored_days"], "scored_windows": s["scored_days"],
           "min_days_to_show": min_days,
           "benchmark": "the only benchmark of the level forecast: the window VWAP of the traded future, scored at the auction result",
           "counts": "windows (one per delivery day and product, base or peak) of settled delivery days whose published pre-market version "
                     "of the default set (the version shown on the public site, "
                     "read from published/forecasts.csv on main) was issued before the market window (11:15 Paris on the trade date), scored "
                     "against the auction result; days whose published versions were all issued later, days with only a local version on "
                     "the collecting machine, and days without a traded price are not counted; the backfilled walk-forward record in the "
                     "README (42 windows) is a separate, out-of-sample reissue, not part of this count",
           "versions": s.get("versions", LOCAL_LABEL),
           "computed_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
    if s["scored_days"] >= min_days:
        out.update({k: s[k] for k in ("hit_rate", "mean_pnl_per_mwh", "model_mae", "market_mae", "share_model_beats_market")})
    else:
        out["note"] = f"Versus the market: collecting data, {s['scored_days']} of {min_days} days."
    return out


def format_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "no market rows scored"
    columns = ["delivery_date", "product", "window_paris", "issued_paris", "model", "forecast", "entry_vwap", "auction", "direction",
               "pnl_per_mwh", "cumulative_pnl_per_mwh", "model_error", "market_error", "model_beats_market", "pnl_per_mwh_close"]
    return table[columns].to_string(index=False, na_rep="")


# ---------------------------------------------------------------------------
# Getting the prices into the file: paste parser, fetch log, local collector
# ---------------------------------------------------------------------------

PASTE_FIELDS = {
    "open": r"\b(?:O|Open)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)",
    "high": r"\b(?:H|High)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)",
    "low": r"\b(?:L|Low)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)",
    "close": r"\b(?:C|Close|Last)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)",
    "vwap": r"\b(?:VWAP|Vwap|vwap)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)",
}
PASTE_WINDOW = r"(\d{1,2}:\d{2})\s*(?:-|to|–)\s*(\d{1,2}:\d{2})"
PASTE_DATE = r"(\d{4}-\d{2}-\d{2})"
PASTE_TRADES = r"(?<![\d-])(\d+)\s*(?:trades|trd)"
PASTE_VOLUME = r"(?:vol(?:ume)?|MWh)\s*[:=]?\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*MWh"


def parse_paste(text: str, delivery_date: str = "", source: str = "EEX via trader") -> MarketRow:
    """A traded-price row from free text such as
    'FR DA Base EEX Trades 11:15-12:00 O: 100.00 H: 102.00 L: 99.00 C: 101.00 VWAP: 100.50'.

    The product (base or peak), the window and the five prices must all be
    present; the delivery date comes from the argument or from an ISO date in
    the text (the argument wins). Decimal commas are accepted. Nothing is
    guessed: a missing field raises MarketDataError naming it.
    """
    flat = " ".join(text.split())
    lowered = flat.lower()
    product = "base" if re.search(r"\bbase\b", lowered) else "peak" if re.search(r"\bpeak\b", lowered) else None
    if product is None:
        raise MarketDataError("paste: neither 'base' nor 'peak' found in the text")
    window = re.search(PASTE_WINDOW, flat)
    if window is None:
        raise MarketDataError("paste: no trading window like 11:15-12:00 found in the text")
    start, end = (f"{int(w.split(':')[0]):02d}:{w.split(':')[1]}" for w in window.groups())
    prices = {}
    for name, pattern in PASTE_FIELDS.items():
        found = re.search(pattern, flat)
        if found is None:
            raise MarketDataError(f"paste: no {name} price found (expected something like '{name[0].upper()}: 100.50')")
        prices[name] = float(found.group(1).replace(",", "."))
    date_in_text = re.search(PASTE_DATE, flat)
    day = delivery_date or (date_in_text.group(1) if date_in_text else "")
    if not day:
        raise MarketDataError("paste: give the delivery date with --date; the text carries none")
    trades = re.search(PASTE_TRADES, lowered)
    volume = re.search(PASTE_VOLUME, flat, flags=re.IGNORECASE)
    volume_value = None
    if volume:
        volume_value = float((volume.group(1) or volume.group(2)).replace(",", "."))
    return MarketRow(day, product, start, end, prices["open"], prices["high"], prices["low"], prices["close"], prices["vwap"], source,
                     "pasted text", trades=int(trades.group(1)) if trades else None, volume_mwh=volume_value).validate()


def log_fetch(status: str, message: str, delivery_date: str = "", product: str = "", path=FETCH_LOG_PATH, now=None) -> None:
    """Append one line to the git-ignored collection log that the dashboard's data status panel shows."""
    if status not in FETCH_STATUSES:
        raise ValueError(f"status must be one of {FETCH_STATUSES}, not {status!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = (pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = pd.DataFrame([{"logged_at_utc": stamp, "delivery_date": delivery_date, "product": product, "status": status,
                          "message": " ".join(str(message).split())[:300]}])
    line.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False, lineterminator="\n")


def read_fetch_log(path=FETCH_LOG_PATH, limit: int = 40) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    return frame.reindex(columns=LOG_COLUMNS).tail(limit).iloc[::-1].reset_index(drop=True)


class FetchBlocked(RuntimeError):
    """The collector was refused (for example HTTP 403); nothing more is tried and the day is left to the paste fallback."""


@dataclass
class FetchOutcome:
    """What a collector module reports for one delivery day and product."""

    delivery_date: str
    product: str
    status: str  # stored (row attached), skipped (no contract or no trades, with the reason), error
    message: str = ""
    row: MarketRow | None = None


def load_plugin(path=DEFAULT_PLUGIN, required: str = "fetch"):
    """Import the local collector module from its file. It must define fetch(delivery_dates, window, raw_dir, spacing_s, log) -> list[FetchOutcome]
    (or, for the morning snapshot, snapshot(delivery_dates, products, raw_dir, spacing_s, log) -> list[TapeOutcome]; `required` names which).

    The module lives outside the package, in a git-ignored folder, because it
    depends on the undocumented page structure of a third party's website
    and on a request header that website expects; the public repository
    carries only this generic contract.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no collector module at {path}; the fetch step needs the local, git-ignored module (see README, market fetch)")
    spec = importlib.util.spec_from_file_location("thermo_fr_market_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, required):
        raise AttributeError(f"{path} defines no {required}() function")
    return module


def run_fetcher(delivery_dates: list[str], plugin=DEFAULT_PLUGIN, market_path=DEFAULT_PATH, log_path=FETCH_LOG_PATH, raw_dir=RAW_DIR,
                window=(MARKET_WINDOW_START, MARKET_WINDOW_END), spacing_s: float = 10.0, log=print, now=None) -> list[FetchOutcome]:
    """Collect the window for each delivery day with the local module, store the rows, log every outcome.

    Days and products already in the file are not requested again, so a second
    scheduled run (the single retry) only fills what the first one missed.
    A row the module returns for a day that was not asked for, or that fails
    validation, is refused and logged as an error: the data must be for the
    right delivery day. If the module raises FetchBlocked the remaining days
    are logged as errors and nothing else is requested.
    """
    existing = load_market(market_path)
    outcomes: list[FetchOutcome] = []
    wanted = []
    for day in delivery_dates:
        for product in PRODUCTS:
            probe = MarketRow(day, product, window[0], window[1], 0, 0, 0, 0, 0, "probe")
            if not find_same(existing, probe).empty:
                outcomes.append(FetchOutcome(day, product, "exists", f"{product} row for {day} already stored"))
            else:
                wanted.append((day, product))
    if not wanted:
        for o in outcomes:
            log_fetch(o.status, o.message, o.delivery_date, o.product, log_path, now=now)
            log(f"{o.delivery_date} {o.product}: {o.status}, {o.message}")
        return outcomes
    try:
        module = load_plugin(plugin)
        reported = module.fetch(sorted({d for d, _ in wanted}), window, Path(raw_dir), spacing_s, log)
    except FetchBlocked as exc:
        reported = [FetchOutcome(day, product, "error", f"blocked: {exc}") for day, product in wanted]
    except Exception as exc:  # noqa: BLE001
        reported = [FetchOutcome(day, product, "error", f"{exc.__class__.__name__}: {exc}") for day, product in wanted]
    by_key = {(o.delivery_date, o.product): o for o in reported}
    for day, product in wanted:
        o = by_key.get((day, product)) or FetchOutcome(day, product, "error", "the collector reported nothing for this day and product")
        if o.status == "stored" and o.row is not None:
            try:
                row = o.row.validate()
                if row.delivery_date != day or row.product != product:
                    raise MarketDataError(f"the collector returned a {row.product} row for {row.delivery_date} when {product} {day} was asked for")
                if row.trade_date and row.trade_date != trade_date_for(day):
                    raise MarketDataError(f"trade_date {row.trade_date} is not the trade date of {day} ({trade_date_for(day)})")
                add_row(row, market_path)
                o.message = o.message or f"{row.product} {row.delivery_date} stored, VWAP from {row.trades} trades, {row.volume_mwh} MWh"
            except MarketDataError as exc:
                o = FetchOutcome(day, product, "error", f"refused: {exc}")
        elif o.status == "stored":
            o = FetchOutcome(day, product, "error", "the collector reported stored without a row")
        elif o.status not in FETCH_STATUSES:
            o = FetchOutcome(day, product, "error", f"unknown status {o.status!r}: {o.message}")
        outcomes.append(o)
    for o in outcomes:
        log_fetch(o.status, o.message, o.delivery_date, o.product, log_path, now=now)
        log(f"{o.delivery_date} {o.product}: {o.status}, {o.message}")
    return outcomes


# ---------------------------------------------------------------------------
# The morning snapshot: the latest base trade before the window opens (laptop only, never published)
# ---------------------------------------------------------------------------


@dataclass
class TapeOutcome:
    """What a collector module reports for one delivery day and product when asked for the tape so far rather than a window."""

    delivery_date: str
    product: str
    status: str  # tape (frame attached), skipped (with the reason), error
    message: str = ""
    tape: pd.DataFrame | None = None  # one row per trade: time_utc (tz-aware UTC), price, volume_mwh
    source: str = ""


def load_early(path=EARLY_PATH) -> pd.DataFrame:
    """The stored snapshot rows, or an empty frame with the right columns."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=EARLY_COLUMNS)
    text_columns = ("delivery_date", "product", "trade_date", "snapshot_at_utc", "trade_time_utc", "trade_time_paris", "source", "note")
    frame = pd.read_csv(path, dtype={c: str for c in text_columns}).reindex(columns=EARLY_COLUMNS)
    frame["note"] = frame["note"].fillna("")
    return frame


def latest_before_window(tape: pd.DataFrame | None, delivery_date: str, trade_date: str = "", window_start: str = MARKET_WINDOW_START) -> dict | None:
    """The last trade of the tape before the window opened on the trade date, with the count and volume traded before it; None without one."""
    if tape is None or tape.empty:
        return None
    traded = pd.Timestamp(trade_date) if trade_date else pd.Timestamp(trade_date_for(delivery_date))
    opens = pd.Timestamp(f"{traded:%Y-%m-%d} {window_start}", tz=LOCAL_TZ).tz_convert("UTC")
    times = pd.to_datetime(tape["time_utc"], utc=True)
    before = tape.assign(time_utc=times)[times < opens].sort_values("time_utc")
    if before.empty:
        return None
    last = before.iloc[-1]
    volumes = pd.to_numeric(before["volume_mwh"], errors="coerce") if "volume_mwh" in before else pd.Series(dtype=float)
    return {"trade_time_utc": last["time_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"), "trade_time_paris": last["time_utc"].tz_convert(LOCAL_TZ).strftime("%H:%M"),
            "price": float(last["price"]), "volume_mwh": float(volumes.iloc[-1]) if len(volumes) else float("nan"),
            "trades_before_window": int(len(before)), "volume_before_window_mwh": float(volumes.sum()) if len(volumes) else float("nan")}


def find_same_early(existing: pd.DataFrame, values: dict) -> pd.DataFrame:
    return existing[(existing["delivery_date"] == values["delivery_date"]) & (existing["product"] == values["product"])
                    & (existing["trade_time_utc"] == values["trade_time_utc"])]


def add_early(values: dict, path=EARLY_PATH) -> pd.DataFrame:
    """Append one snapshot row (keys of EARLY_COLUMNS); the same delivery day, product and trade is stored once."""
    if values["product"] not in PRODUCTS:
        raise MarketDataError(f"product must be one of {PRODUCTS}, not {values['product']!r}")
    if _blank(values.get("price")) or np.isnan(float(values["price"])):
        raise MarketDataError("the snapshot row needs a price")
    traded = values.get("trade_date") or trade_date_for(values["delivery_date"])
    opens = pd.Timestamp(f"{traded} {MARKET_WINDOW_START}", tz=LOCAL_TZ).tz_convert("UTC")
    if pd.Timestamp(values["trade_time_utc"]).tz_convert("UTC") >= opens:
        raise MarketDataError(f"trade at {values['trade_time_utc']} is not before the window opened at {MARKET_WINDOW_START} Paris on {traded}")
    existing = load_early(path)
    if not find_same_early(existing, values).empty:
        raise MarketDataError(f"the {values['product']} trade of {values['trade_time_paris']} Paris for {values['delivery_date']} is already stored")
    row = {c: values.get(c, "") for c in EARLY_COLUMNS}
    row["trade_date"] = traded
    frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True) if len(existing) else pd.DataFrame([row])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=EARLY_COLUMNS).to_csv(path, index=False, lineterminator="\n")
    return load_early(path)


def run_snapshot(delivery_dates: list[str], plugin=DEFAULT_PLUGIN, early_path=EARLY_PATH, log_path=FETCH_LOG_PATH, raw_dir=RAW_DIR,
                 products=SNAPSHOT_PRODUCTS, spacing_s: float = 10.0, log=print, now=None) -> list[FetchOutcome]:
    """Before the window opens: read the base tape so far and store the latest trade before MARKET_WINDOW_START Paris for each delivery day.

    Scheduled on the laptop at 10:20 London (11:20 Paris): with the page's display delay the tape then reaches about 11:05 Paris,
    so the stored trade is one a trader could have seen before the window, after the pre-market forecast was issued. The local
    module must define snapshot(delivery_dates, products, raw_dir, spacing_s, log) -> list[TapeOutcome] and keep its raw responses
    apart from the window fetch's cache, so the 12:20 fetch still asks for the full day. A trade at or after the window open is
    never stored from here; a snapshot that finds the trade an earlier one stored is logged as exists. Every outcome goes to the
    same collection log as the window fetch, prefixed "snapshot:". The file is git-ignored and nothing of it is exported.
    """
    stamp = pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")
    wanted = [(day, product) for day in delivery_dates for product in products]
    try:
        module = load_plugin(plugin, required="snapshot")
        reported = module.snapshot(sorted({d for d, _ in wanted}), tuple(products), Path(raw_dir), spacing_s, log)
    except FetchBlocked as exc:
        reported = [TapeOutcome(day, product, "error", f"blocked: {exc}") for day, product in wanted]
    except Exception as exc:  # noqa: BLE001
        reported = [TapeOutcome(day, product, "error", f"{exc.__class__.__name__}: {exc}") for day, product in wanted]
    by_key = {(o.delivery_date, o.product): o for o in reported}
    outcomes: list[FetchOutcome] = []
    for day, product in wanted:
        o = by_key.get((day, product)) or TapeOutcome(day, product, "error", "the collector reported nothing for this day and product")
        if o.status == "tape":
            try:
                picked = latest_before_window(o.tape, day)
                if picked is None:
                    count = 0 if o.tape is None else len(o.tape)
                    out = FetchOutcome(day, product, "skipped", f"snapshot: {count} trades on the tape, none before {MARKET_WINDOW_START} Paris")
                else:
                    values = {"delivery_date": day, "product": product, "trade_date": trade_date_for(day),
                              "snapshot_at_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"), **picked, "source": o.source or "collector",
                              "note": o.message or "snapshot before the window"}
                    if not find_same_early(load_early(early_path), values).empty:
                        out = FetchOutcome(day, product, "exists", f"snapshot: the latest {product} trade before the window "
                                                                   f"({picked['trade_time_paris']} Paris) is already stored")
                    else:
                        add_early(values, early_path)
                        out = FetchOutcome(day, product, "stored", f"snapshot: latest {product} trade before the window at {picked['trade_time_paris']} "
                                                                   f"Paris, {picked['trades_before_window']} trades before the window")
            except Exception as exc:  # noqa: BLE001
                out = FetchOutcome(day, product, "error", f"snapshot: {exc.__class__.__name__}: {exc}")
        elif o.status in ("skipped", "error"):
            out = FetchOutcome(day, product, o.status, f"snapshot: {o.message}")
        else:
            out = FetchOutcome(day, product, "error", f"snapshot: unknown status {o.status!r}: {o.message}")
        outcomes.append(out)
    for o in outcomes:
        log_fetch(o.status, o.message, o.delivery_date, o.product, log_path, now=now)
        log(f"{o.delivery_date} {o.product}: {o.status}, {o.message}")
    return outcomes
