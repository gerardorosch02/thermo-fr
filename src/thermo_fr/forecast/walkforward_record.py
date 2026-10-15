"""The walk-forward trading record against EEX traded prices, as aggregates only.

The holdout walk-forward predictions (monthly retraining, backtest settings,
`reports/forecast/predictions_honest_v2_holdout.csv`) are scored against the
private EEX window file (`data/market/eex_fr_da.csv`) with the same rules as
`market.evaluate`: the product average (base: every hour of the delivery day,
peak: 08:00 to 20:00 Paris, `market.product_mask`) of the forecast against the
window VWAP, long when above, short when below, settled at the product average
of the auction result. One window per delivery day and product.

What is written is aggregates only: counts, rates, means, test statistics and
the selection-against-record comparison. No per-day or per-window values and no
traded-price levels leave this module, so the file can be published.

Statistics:
- Exact two-sided binomial test of the hit count against 0.5 (the sum of the
  probabilities of all outcomes no more likely than the observed one, as
  scipy.stats.binomtest does).
- Moving-block bootstrap over delivery days (both windows of a day stay
  together): blocks of BOOTSTRAP_BLOCK_DAYS consecutive scored days,
  BOOTSTRAP_RESAMPLES resamples, numpy default_rng(BOOTSTRAP_SEED); 95%
  percentile intervals for the hit rate and the mean P&L per window, and a
  two-sided p-value from the centred bootstrap distribution.
- Repricing share: least-squares slope of (forecast minus the D-1 auction
  product average) on (traded VWAP minus the D-1 auction product average),
  with its standard error: the share of the market's same-morning move away
  from yesterday's price that the forecast made.
- Persistence: the direction implied by the D-1 auction product average
  against the VWAP; its hit count and how often the model made the same call.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .market import direction_of, load_market, product_average

BOOTSTRAP_BLOCK_DAYS = 3
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 42
SOURCE = ("walk-forward predictions with monthly retraining (backtest settings) in reports/forecast/predictions_honest_v2_holdout.csv, "
          "scored against the EEX French Day-Ahead Base and Peak window VWAP (11:15 to 12:00 Paris on the trade date) kept privately in "
          "data/market/eex_fr_da.csv; product averages as in forecast/market.py; written by forecast/walkforward_record.py "
          "(thermo-fr market walkforward-record); aggregates only, traded prices are not republished")


def load_predictions(path) -> pd.DataFrame:
    """A walk-forward predictions file of forecast/backtest.py, indexed by UTC hour."""
    frame = pd.read_csv(path, parse_dates=["delivery_day"])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.set_index("timestamp_utc").sort_index()


def window_table(predictions: pd.DataFrame, market: pd.DataFrame, model: str = "gbm") -> pd.DataFrame:
    """One scored row per market window whose delivery day is in the predictions (internal: never written out)."""
    rows = []
    by_day = {day: part for day, part in predictions.groupby("delivery_day")}
    for _, r in market.sort_values(["delivery_date", "product"]).iterrows():
        day = pd.Timestamp(r["delivery_date"])
        part = by_day.get(day)
        if part is None or part.empty:
            continue
        forecast = product_average(part[model], r["product"])
        auction = product_average(part["actual"], r["product"])
        d1 = product_average(part["naive_day"], r["product"])
        entry = float(r["vwap"])
        if any(np.isnan(v) for v in (forecast, auction, entry)):
            continue
        direction = direction_of(forecast, entry)
        if direction == 0:
            continue
        rows.append({"day": day, "product": r["product"], "forecast": forecast, "auction": auction, "d1": d1, "entry": entry,
                     "direction": direction, "pnl": (auction - entry) * direction, "model_error": abs(forecast - auction),
                     "market_error": abs(entry - auction)})
    return pd.DataFrame(rows)


def binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (outcomes no more likely than k, relative tolerance as in scipy)."""
    if n == 0:
        return float("nan")
    probs = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    cut = probs[k] * (1 + 1e-7)
    return float(min(1.0, sum(q for q in probs if q <= cut)))


def block_bootstrap(table: pd.DataFrame, block: int = BOOTSTRAP_BLOCK_DAYS, resamples: int = BOOTSTRAP_RESAMPLES,
                    seed: int = BOOTSTRAP_SEED) -> dict:
    """Moving-block bootstrap over delivery days of the hit rate and the mean P&L per window."""
    daily = table.assign(hit=(table["pnl"] > 0).astype(float)).groupby("day").agg(
        pnl_sum=("pnl", "sum"), hit_sum=("hit", "sum"), n=("pnl", "size")).sort_index()
    nd = len(daily)
    block = max(1, min(block, nd))
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(nd / block))
    starts = rng.integers(0, nd - block + 1, size=(resamples, blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(resamples, -1)[:, :nd]
    pnl_sum, hit_sum, n = (daily[c].to_numpy()[idx].sum(axis=1) for c in ("pnl_sum", "hit_sum", "n"))
    pnl, hit = pnl_sum / n, hit_sum / n
    observed_pnl = float(table["pnl"].mean())
    observed_hit = float((table["pnl"] > 0).mean())

    def summary(draws, observed, null):
        return {"ci95": [round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)],
                "p_two_sided": round(float((np.abs(draws - draws.mean()) >= abs(observed - null)).mean()), 4)}

    return {"method": f"moving-block bootstrap over delivery days, blocks of {block} days, {resamples} resamples, numpy default_rng seed "
                      f"{seed}; percentile intervals; p-value from the centred bootstrap distribution",
            "days": int(nd), "hit_rate": summary(hit, observed_hit, 0.5), "mean_pnl_per_mwh": summary(pnl, observed_pnl, 0.0)}


def period_stats(predictions: pd.DataFrame, model: str = "gbm") -> dict:
    """Hourly MAE of the model, that MAE as a share of the mean absolute price, mean auction price and mean absolute day-to-day change
    of the daily mean. The traded price is the only public benchmark, so no naive comparison is published."""
    frame = predictions.dropna(subset=["actual", model])
    daily = frame.groupby("delivery_day")["actual"].mean().sort_index()
    return {"days": int(frame["delivery_day"].nunique()), "hours": int(len(frame)),
            "first_day": str(frame["delivery_day"].min().date()), "last_day": str(frame["delivery_day"].max().date()),
            "hourly_mae_model": round(float((frame[model] - frame["actual"]).abs().mean()), 2),
            "hourly_mae_share_of_price": round(float((frame[model] - frame["actual"]).abs().mean() / frame["actual"].abs().mean()), 2),
            "mean_price_eur_mwh": round(float(frame["actual"].mean()), 1),
            "mean_abs_daily_change_eur_mwh": round(float(daily.diff().abs().mean()), 1)}


def _record(table: pd.DataFrame) -> dict:
    hits = int((table["pnl"] > 0).sum())
    n = int(len(table))
    return {"windows": n, "hits": hits, "hit_rate": round(hits / n, 3) if n else None,
            "mean_pnl_per_mwh": round(float(table["pnl"].mean()), 2) if n else None,
            "binomial_p_two_sided": round(binomial_two_sided(hits, n), 4) if n else None}


def build_record(predictions: pd.DataFrame, market: pd.DataFrame, selection: pd.DataFrame | None = None, model: str = "gbm",
                 now=None) -> dict:
    """The aggregates-only walk-forward record (see the module docstring)."""
    table = window_table(predictions, market, model)
    if table.empty:
        raise ValueError("no market window falls on a delivery day of the predictions")
    base = table[table["product"] == "base"]
    peak = table[table["product"] == "peak"]
    hits = int((table["pnl"] > 0).sum())
    x = (table["entry"] - table["d1"]).to_numpy()
    y = (table["forecast"] - table["d1"]).to_numpy()
    ok = ~(np.isnan(x) | np.isnan(y))
    slope = se = None
    if ok.sum() >= 3:
        xc, yc = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        slope = float((xc * yc).sum() / (xc * xc).sum())
        resid = yc - slope * xc
        se = float(np.sqrt((resid**2).sum() / (ok.sum() - 2) / (xc * xc).sum()))
    persistence = np.sign(table["d1"] - table["entry"])
    persistence_pnl = (table["auction"] - table["entry"]) * persistence
    record_days = predictions[predictions["delivery_day"].isin(table["day"].unique())]
    comparison = {"record_days": period_stats(record_days, model), "holdout": period_stats(predictions, model)}
    if selection is not None:
        strict = selection[selection["strict"].astype(bool)] if "strict" in selection else selection
        comparison = {"selection": period_stats(strict, model), **comparison}
    written = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    return {
        "model": "honest_v2",
        "kind": "walk-forward record, not live",
        "period": {"first_delivery_day": str(table["day"].min().date()), "last_delivery_day": str(table["day"].max().date())},
        "windows": int(len(table)), "base_windows": int(len(base)), "peak_windows": int(len(peak)),
        "days": int(table["day"].nunique()),
        "hits": hits, "hit_rate": round(hits / len(table), 3),
        "mean_pnl_per_mwh": round(float(table["pnl"].mean()), 2),
        "model_mae": round(float(table["model_error"].mean()), 2),
        "market_mae": round(float(table["market_error"].mean()), 2),
        "share_model_beats_market": round(float((table["model_error"] < table["market_error"]).mean()), 3),
        "metric_note": "model_mae and market_mae are errors of the product average (base or peak) against the auction's product average, "
                       "not hourly errors",
        "base_only": _record(base),
        "binomial_p_two_sided": round(binomial_two_sided(hits, len(table)), 4),
        "bootstrap": block_bootstrap(table),
        "repricing_share": {"slope": round(slope, 3) if slope is not None else None, "se": round(se, 3) if se is not None else None,
                            "definition": "least-squares slope of (forecast minus the D-1 auction product average) on (traded VWAP minus "
                                          "the D-1 auction product average): the share of the market's same-morning move the model made"},
        "persistence": {"hits": int((persistence_pnl > 0).sum()),
                        "model_calls_equal_persistence": int((persistence == table["direction"]).sum()),
                        "definition": "long when the D-1 auction product average is above the traded VWAP, short when below"},
        "period_mean_price_eur_mwh": comparison["record_days"]["mean_price_eur_mwh"],
        "comparison": comparison,
        "comparison_note": "hourly MAE of the model, that MAE divided by the mean absolute hourly price, mean hourly auction price and mean absolute change "
                           "of the daily mean price, on the strict selection window, the frozen holdout and the delivery days of the record",
        "source": SOURCE,
        "written_at_utc": written.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if written.tzinfo else written.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_record(record: dict, out) -> Path:
    import json

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out


def run(predictions_path, market_path, out, selection_path=None, now=None) -> dict:
    record = build_record(load_predictions(predictions_path), load_market(market_path),
                          load_predictions(selection_path) if selection_path else None, now=now)
    write_record(record, out)
    return record
