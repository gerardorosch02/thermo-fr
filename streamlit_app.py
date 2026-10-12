"""Public dashboard for the French day-ahead price forecast, deployable on Streamlit Community Cloud.

It reads only the files under published/, which a GitHub Actions workflow
rewrites after every forecast and settlement run. No API is called here and no
key is needed. It depends on streamlit, pandas and plotly only (requirements.txt),
not on the thermo_fr package.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PUBLISHED = Path(__file__).parent / "published"
COLORS = {"forecast": "#2a78d6", "actual": "#eb6834", "benchmark": "#52514e", "grid": "#e6e5e1", "ink": "#0b0b0b"}
BAND = ("rgba(42,120,214,0.14)", "rgba(42,120,214,0.30)")
PARIS = "Europe/Paris"
MARKET_WINDOW_START = "11:15"  # Paris, on the day before delivery: when the EEX day-ahead future starts trading


def premarket_flag(frame: pd.DataFrame) -> pd.Series:
    """True where a version was issued before the market window of its delivery day opened (11:15 Paris on D-1)."""
    if frame.empty:
        return pd.Series(dtype=bool)
    day_before = pd.to_datetime(frame["delivery_day"]) - pd.Timedelta(days=1)
    cutoff = pd.to_datetime(day_before.dt.strftime("%Y-%m-%d") + " " + MARKET_WINDOW_START).dt.tz_localize(PARIS).dt.tz_convert("UTC")
    issued = pd.to_datetime(frame["issued_at_utc"], utc=True)
    return (issued < cutoff).rename("premarket")


def headline_versions(frame: pd.DataFrame) -> pd.DataFrame:
    """One (delivery_day, issued_at_utc, premarket) row per day: the last pre-market version, else the latest version."""
    if frame.empty:
        return pd.DataFrame(columns=["delivery_day", "issued_at_utc", "premarket"])
    versions = frame[["delivery_day", "issued_at_utc"]].drop_duplicates().copy()
    versions["premarket"] = premarket_flag(versions).to_numpy()
    versions = versions.sort_values(["delivery_day", "issued_at_utc"])
    picked = []
    for _, group in versions.groupby("delivery_day", sort=True):
        before = group[group["premarket"]]
        picked.append(before.iloc[-1] if len(before) else group.iloc[-1])
    return pd.DataFrame(picked).reset_index(drop=True)


def _premarket(frame: pd.DataFrame):
    return premarket_flag(frame).to_numpy()

LIVE_RECORD_RULE = ("Each delivery day counts once, scored against the auction result on the last version issued before the market window "
                    "opened (11:15 Paris on the day before); a day whose versions were all issued later is listed as late only and not scored. "
                    "The record is the current default model from its first pre-market day; the earlier days of the previous set are a "
                    "separate row. It states the model's own error only: the EEX traded price, in the Versus the market panel, is the one "
                    "benchmark of the level forecast.")


def _record_for(rows: pd.DataFrame, feature_set: str, days: int) -> tuple[dict, pd.DataFrame]:
    """One row per delivery day (its last pre-market version) with the aggregates of the last `days` of them; late-only days listed."""
    picked, late_only = [], []
    for day, group in rows.sort_values(["delivery_day", "issued_at_utc"]).groupby("delivery_day", sort=True):
        pre = group[group["premarket"].astype(bool)]
        if len(pre):
            picked.append(pre.iloc[-1])
        else:
            late_only.append({"delivery_day": day, "versions": int(len(group)), "first_issued_utc": str(group["issued_at_utc"].iloc[0])})
    table = pd.DataFrame(picked).reset_index(drop=True) if picked else rows.iloc[0:0]
    out = {"feature_set": feature_set, "days": int(len(table)), "late_only_days": late_only, "window_days": days}
    if len(table):
        recent = table.tail(days)
        out.update({"first_premarket_day": str(table["delivery_day"].iloc[0]), "last_premarket_day": str(table["delivery_day"].iloc[-1]),
                    "days_in_window": int(len(recent)), "model_mae": round(float(recent["mae"].mean()), 2),
                    "model_rmse": round(float(recent["rmse"].mean()), 2) if "rmse" in recent else None,
                    "scored_against": "the auction result; the EEX traded price is the only benchmark (Versus the market)"})
    return out, table


def live_record(scores: pd.DataFrame, default_set: str = "honest_v2", previous_set: str = "honest", days: int = 30) -> tuple[dict, dict]:
    """The public live record: (record, tables). record["default"] is the headline, record["previous"] the earlier set's own row or None."""
    needed = {"delivery_day", "issued_at_utc", "feature_set", "mae"}
    empty = {"feature_set": default_set, "days": 0, "late_only_days": [], "window_days": days, "status": "collecting data"}
    if scores is None or scores.empty or not needed <= set(scores.columns):
        return {"default": empty, "previous": None, "min_days_to_show": 1, "rule": LIVE_RECORD_RULE}, {"default": pd.DataFrame(), "previous": pd.DataFrame()}
    frame = scores.copy()
    frame["premarket"] = _premarket(frame)
    default, default_table = _record_for(frame[frame["feature_set"] == default_set], default_set, days)
    default["status"] = "ok" if default["days"] >= 1 else "collecting data"
    previous_rows = frame[frame["feature_set"] == previous_set]
    if default.get("first_premarket_day"):
        previous_rows = previous_rows[previous_rows["delivery_day"] < default["first_premarket_day"]]
    previous, previous_table = (_record_for(previous_rows, previous_set, days) if len(previous_rows) else (None, pd.DataFrame()))
    return ({"default": default, "previous": previous, "min_days_to_show": 1, "rule": LIVE_RECORD_RULE},
            {"default": default_table, "previous": previous_table})


# ---------------------------------------------------------------------------- display of event probabilities (the published values keep full precision)
BELOW_1PCT = "below 1%"
SUMMARY_THRESHOLD = 0.05  # hours above this count in the one-line summary
VERY_UNLIKELY = "Negative prices very unlikely tomorrow (every hour below 1%)."


def format_probability(p) -> str:
    """Display form of a probability: 'below 1%' under 1% (the lowest reliability bin is overconfident), one decimal under 10%, else whole."""
    if p is None or pd.isna(p):
        return "n/a"
    p = float(p)
    if p < 0.01:
        return BELOW_1PCT
    return f"{100 * p:.1f}%" if p < 0.10 else f"{100 * p:.0f}%"


def negative_summary(hourly: pd.DataFrame, column: str = "p_negative") -> dict:
    """One line about tomorrow: the highest hourly probability and its hour, and the hours above 5%; or the very-unlikely sentence."""
    if hourly is None or hourly.empty or column not in hourly or hourly[column].isna().all():
        return {"text": "", "max_p": None, "max_hour": None, "hours_above_5pct": 0, "all_below_1pct": False}
    values = hourly[[c for c in ("hour", column) if c in hourly]].dropna(subset=[column])
    top = values.loc[values[column].idxmax()]
    max_p, max_hour = float(top[column]), int(top["hour"]) if "hour" in values else None
    above = int((values[column] > SUMMARY_THRESHOLD).sum())
    all_below = bool((values[column] < 0.01).all())
    if all_below:
        text = VERY_UNLIKELY
    else:
        where = f" at {max_hour:02d}:00 Paris" if max_hour is not None else ""
        text = f"Highest hourly probability of a negative price tomorrow: {format_probability(max_p)}{where}; {above} hour(s) above 5%."
    return {"text": text, "max_p": max_p, "max_hour": max_hour, "hours_above_5pct": above, "all_below_1pct": all_below}


def probability_axis_top(max_p) -> float:
    """Top of the probability axis: the larger of 10% and 1.2 times the highest probability, never above 100%."""
    value = 0.0 if max_p is None or pd.isna(max_p) else float(max_p)
    return min(1.0, max(0.10, 1.2 * value))


def show_half_line(axis_top: float) -> bool:
    """The light reference line at 50% is drawn only when the axis reaches it."""
    return axis_top >= 0.5


def lowest_bin_sentence(backtest: dict) -> str:
    """Why probabilities below 1% are shown as 'below 1%': the lowest reliability bin of the negative-price backtest is overconfident."""
    parts = []
    for label, record in (backtest or {}).items():
        table = (record.get("events", {}).get("negative", {}) or {}).get("reliability") or []
        if table:
            first = table[0]
            parts.append(f"{label}: forecast about {100 * float(first['mean_forecast']):.1f}%, observed {100 * float(first['observed_frequency']):.1f}%")
    if not parts:
        return ""
    return ("The lowest reliability bin is overconfident: where the model gave well under 1%, negative prices still occurred about 2% of the "
            "time (" + "; ".join(parts) + "). Probabilities below 1% are therefore shown as \"below 1%\", never as 0.0%; the published CSV keeps "
            "the full values.")


def headline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of `frame` (forecasts or scores, with delivery_day and issued_at_utc) that belong to each day's headline version."""
    heads = headline_versions(frame)
    if heads.empty:
        return frame.iloc[0:0]
    return frame.merge(heads, on=["delivery_day", "issued_at_utc"], how="inner")


@st.cache_data(ttl=300, show_spinner=False)
def load_published(version: float) -> dict:
    def read(name, **kw):
        path = PUBLISHED / name
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, **kw)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    status = json.loads((PUBLISHED / "status.json").read_text()) if (PUBLISHED / "status.json").exists() else {}
    out = {
        "status": status,
        "tomorrow": read("tomorrow.csv"),
        "forecasts": read("forecasts.csv"),
        "actuals": read("actuals.csv"),
        "scores": read("scores.csv"),
        "band": read("error_band.csv"),
        "shape": json.loads((PUBLISHED / "shape_battery.json").read_text()) if (PUBLISHED / "shape_battery.json").exists() else {},
        "prob_tomorrow": read("tomorrow_probabilistic.csv"),
        "prob_backtest": json.loads((PUBLISHED / "probabilistic_backtest.json").read_text()) if (PUBLISHED / "probabilistic_backtest.json").exists() else {},
    }
    # the dataset may carry more than one feature set (the default model's and the fallback's); the app shows the default.
    # scores_all keeps every set: the live record splits them itself (default headline, previous model row)
    out["scores_all"] = out["scores"]
    for key in ("forecasts", "scores"):
        out[key] = default_set_rows(out[key], status)
    return out


def paris_clock(stamp) -> str:
    return pd.Timestamp(stamp).tz_convert(PARIS).strftime("%Y-%m-%d %H:%M")


def late_only_text(record: dict) -> str:
    items = record.get("late_only_days") or []
    if not items:
        return ""
    return "Late only, not scored: " + "; ".join(f"{d['delivery_day']} ({d['versions']} version(s), earliest issued {paris_clock(d['first_issued_utc'])} Paris)"
                                                   for d in items) + "."


def default_set_rows(frame: pd.DataFrame, status: dict) -> pd.DataFrame:
    wanted = status.get("feature_set")
    if frame.empty or not wanted or "feature_set" not in frame:
        return frame
    kept = frame[frame["feature_set"] == wanted]
    return kept.reset_index(drop=True) if len(kept) else frame


def published_version() -> float:
    path = PUBLISHED / "status.json"
    return path.stat().st_mtime if path.exists() else 0.0


def layout(fig: go.Figure, ytitle: str, xtitle: str) -> go.Figure:
    fig.update_layout(template="plotly_white", height=400, margin=dict(l=40, r=20, t=30, b=40), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      xaxis=dict(title=xtitle, gridcolor=COLORS["grid"]), yaxis=dict(title=ytitle, gridcolor=COLORS["grid"]))
    return fig


def model_label(model: str) -> str:
    """'gbm:honest_v2.txt' -> 'honest_v2.txt', 'gbm:honest.txt:fallback' -> 'honest.txt (fallback)', 'gbm' -> 'fitted live'."""
    parts = str(model).split(":")
    if len(parts) == 1:
        return "fitted live"
    return parts[1] + (" (fallback)" if len(parts) > 2 and parts[2] == "fallback" else "")


def tomorrow_versions(data: dict) -> dict:
    """The next delivery day's versions: the headline (pre-market when one exists) and the later, not tradeable ones."""
    forecasts = data["forecasts"]
    if forecasts.empty:
        tomorrow = data["tomorrow"]
        if tomorrow.empty:
            return {}
        flag = bool(premarket_flag(tomorrow.iloc[:1]).iloc[0])
        return {"day": tomorrow["delivery_day"].iloc[0], "headline": tomorrow, "premarket": flag, "later": pd.DataFrame()}
    day = forecasts["delivery_day"].max()
    rows = forecasts[forecasts["delivery_day"] == day].copy()
    head = headline_versions(rows).iloc[0]
    headline = rows[rows["issued_at_utc"] == head["issued_at_utc"]]
    flags = premarket_flag(rows)
    later_rows = rows[~flags.to_numpy()]
    if "model" not in rows:
        rows["model"] = "gbm"
        later_rows = later_rows.assign(model="gbm")
    later = (later_rows.groupby("issued_at_utc").agg(daily_mean=("forecast", "mean"), kind=("kind", "first"), model=("model", "first")).reset_index()
             if not later_rows.empty else pd.DataFrame(columns=["issued_at_utc", "daily_mean", "kind", "model"]))
    later["issued_paris"] = [pd.Timestamp(t).tz_convert(PARIS).strftime("%Y-%m-%d %H:%M") for t in later["issued_at_utc"]]
    model = str(headline["model"].iloc[0]) if "model" in headline and len(headline) else "gbm"
    return {"day": day, "headline": headline, "premarket": bool(head["premarket"]), "later": later, "model": model}


def tomorrow_table(data: dict) -> pd.DataFrame:
    versions = tomorrow_versions(data)
    if not versions:
        return pd.DataFrame()
    tomorrow = versions["headline"]
    by_hour = tomorrow.groupby("hour").agg(forecast=("forecast", "mean")).reset_index()
    day = versions["day"]
    previous = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    actuals = data["actuals"]
    if not actuals.empty:
        today = actuals[actuals["delivery_day"] == previous]
        if not today.empty:
            by_hour = by_hour.merge(today.groupby("hour")["price"].mean().rename("actual_today").reset_index(), on="hour", how="left")
    if "actual_today" not in by_hour:
        by_hour["actual_today"] = float("nan")
    band = data["band"]
    if not band.empty:
        by_hour = by_hour.merge(band[["hour", "p10", "p25", "p75", "p90", "n"]], on="hour", how="left")
        for q in ("p10", "p25", "p75", "p90"):
            by_hour[f"band_{q}"] = by_hour["forecast"] + by_hour[q]
        by_hour.attrs["band_source"] = "backtest"
    return by_hour


def tomorrow_chart(by_hour: pd.DataFrame, day: str, previous: str) -> go.Figure:
    fig = go.Figure()
    x = by_hour["hour"]
    if "band_p10" in by_hour:
        for lo, hi, color, name in (("band_p10", "band_p90", BAND[0], "Historical error, 10th to 90th pct"),
                                    ("band_p25", "band_p75", BAND[1], "Historical error, 25th to 75th pct")):
            fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([by_hour[hi], by_hour[lo][::-1]]), fill="toself",
                                     fillcolor=color, line=dict(width=0), name=name, hoverinfo="skip"))
    if by_hour["actual_today"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=by_hour["actual_today"], name=f"Actual price on {previous}", mode="lines",
                                 line=dict(color=COLORS["actual"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=by_hour["forecast"], name=f"Forecast for {day}", mode="lines+markers",
                             line=dict(color=COLORS["forecast"], width=3), marker=dict(size=6)))
    fig = layout(fig, "EUR/MWh", "Delivery hour (Paris time)")
    fig.update_xaxes(dtick=2)
    return fig


def daily_scores(data: dict, days: int = 30) -> pd.DataFrame:
    scores = data["scores"]
    if scores.empty:
        return scores
    latest = headline_rows(scores).sort_values("delivery_day").tail(days).copy()  # each day's pre-market version where one exists
    latest["rolling_mae"] = latest["mae"].rolling(7, min_periods=1).mean()
    return latest


def performance_chart(scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scores["delivery_day"], y=scores["mae"], name="Model daily MAE against the auction result", marker_color=COLORS["forecast"]))
    fig.add_trace(go.Scatter(x=scores["delivery_day"], y=scores["rolling_mae"], name="Model, 7-day rolling", mode="lines",
                             line=dict(color=COLORS["ink"], width=2)))
    fig = layout(fig, "MAE, EUR/MWh", "Delivery day")
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def history_chart(data: dict, days: int = 30) -> go.Figure | None:
    forecasts, actuals = data["forecasts"], data["actuals"]
    if forecasts.empty or actuals.empty:
        return None
    latest = headline_rows(forecasts)  # each day's pre-market version where one exists
    joined = latest.merge(actuals[["timestamp_utc", "price"]], on="timestamp_utc", how="inner")
    if joined.empty:
        return None
    joined["t"] = pd.to_datetime(joined["timestamp_utc"], utc=True).dt.tz_convert(PARIS)
    joined = joined.sort_values("t")
    last_days = sorted(joined["delivery_day"].unique())[-days:]
    joined = joined[joined["delivery_day"].isin(last_days)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=joined["t"], y=joined["price"], name="Actual price", mode="lines", line=dict(color=COLORS["actual"], width=2)))
    fig.add_trace(go.Scatter(x=joined["t"], y=joined["forecast"], name="Forecast (pre-market version)", mode="lines", line=dict(color=COLORS["forecast"], width=2)))
    return layout(fig, "EUR/MWh", "Delivery hour (Paris time)")


# The walk-forward trading record of the default model honest_v2, reissued over the 42 windows of the market-fetch record (docs/experiments.md,
# holdout row of 2026-10-10; docs/forecast.md); aggregates only, the traded prices themselves are not republished. Updated by hand when the
# record is re-scored. It is labelled walk-forward, not live: the live record is status.json's "market" section.
# The shaded band on tomorrow's chart is the backtest error band. The quantile band exists in the local store, under evaluation, and is
# not published; this constant stays "backtest" (the public dataset carries no quantile columns, so no other value would work).
BAND_SOURCE = "backtest"

MARKET_RECORD = {
    "model": "honest_v2", "period": "delivery days 2026-08-29 to 2026-10-10",
    "windows": 42, "base_windows": 38, "peak_windows": 4,
    "hit_rate": 0.33, "mean_pnl_per_mwh": -2.43, "model_mae": 28.5, "market_mae": 5.1, "share_model_beats_market": 0.07,
    "base_only": "38 windows, 37% hit rate, -1.99 EUR/MWh per window",
    "source": "docs/experiments.md, holdout row of 2026-10-10",
    "reason": "The model reacts to large moves about two days late, because its price lags carry yesterday's level into "
              "tomorrow's forecast, while the market reprices the same morning.",
}

HOW_IT_WORKS = """
**Target.** The hourly French day-ahead electricity price (EPEX / single day-ahead coupling), in Paris delivery hours.

**Information gate.** The auction closes at 12:00 Paris time on the day before delivery. Every input is dated by when it is
published and the forecast uses only inputs available before that moment: the ENTSO-E day-ahead load forecast (due two hours
before gate closure), Open-Meteo weather forecasts issued two days ahead (temperature, 100 m wind, solar radiation for eight
cities), wind and solar generation proxies (100 m wind forecasts at 17 points in the French wind regions and radiation
forecasts at 21 points in the solar regions, both issued two days ahead and turned into MW with weights fitted monthly on past
actual generation), the calendar (weekday, public holidays, bridge days, the days around holidays and the Christmas break),
and the prices of the previous days (D-1, D-2, D-7 and the most recent day of the same type), which also stand in for gas and
carbon costs. ENTSO-E's own wind and solar forecasts are not used, because the platform allows them until 18:00 on D-1.

**Model.** Gradient boosting (LightGBM) fitted on all history before the delivery day. The level forecast has one benchmark:
the EEX French day-ahead future traded before the auction, scored at the auction result (the "Versus the market" panel at the
top). Its own error is reported as the mean absolute error (MAE, in EUR/MWh) against the auction result, with no other
comparison. Model variants are still chosen by their strict-backtest MAE on 2024 to mid-2026 (about 15.7 EUR/MWh for the
current set), because the EEX history is too short to select on; that figure is a selection criterion, not a benchmark.

**Error band.** The shaded band around tomorrow's curve is the forecast plus the 10th to 90th (and 25th to 75th) percentile
of the model's signed error at the same hour in that backtest. It describes how wrong the model has been at that hour in
the past; it is not a probability forecast for tomorrow.

**Updates.** A GitHub Actions workflow publishes the forecast on weekday mornings and scores it against the
published prices every afternoon. Source code, method and backtest: the repository linked above.

**Limitations.** Against EEX French day-ahead futures traded before the auction the model loses (the "Versus the market"
panel): over 42 walk-forward windows the direction implied by the forecast had a 33% hit rate and a mean P&L of -2.43 EUR/MWh
per window, and the forecast's error against the auction was 28.5 EUR/MWh where the traded price's was 5.1. The model reacts
to large moves about two days late, because its price lags carry yesterday's level into tomorrow's forecast, while the market
reprices the same morning. The error band is the model's
past error distribution, not a probability forecast. Errors are largest on days with regime changes (cold snaps,
price collapses, days after holidays), which is also where a forecast matters most. A known weakness is the top 5% price
hours, typically cold, calm winter evenings when gas sets the price: the generation proxies improve the error elsewhere but
made those hours slightly worse in the backtest (23.2 against 22.6 EUR/MWh without them). Weekend and holiday middays with
very low prices are still forecast too high more often than not.
"""


def main() -> None:
    st.set_page_config(page_title="French day-ahead price forecast", layout="wide")
    st.title("French day-ahead price forecast")
    st.caption("Hourly forecast made with information available at 12:00 Paris time the day before delivery. "
               "Public, automatically updated, for research and discussion, not trading advice.")
    data = load_published(published_version())
    status = data["status"]
    if not status:
        st.error("No published dataset found. The workflow has not run yet.")
        st.stop()
    written = pd.Timestamp(status["written_at_utc"]).tz_convert(PARIS)
    default_model = status.get("default_model_file", "model/honest.txt").split("/")[-1]
    fallback_model = status.get("fallback_model_file", "").split("/")[-1]
    st.caption(f"Dataset written {written:%Y-%m-%d %H:%M} Paris. Window: last {status['window_days']} days. Default model {default_model}"
               + (f", fallback {fallback_model} when one of its inputs is missing" if fallback_model else "")
               + "; every version below names the model that produced it.")

    # 1. The headline: the forecast against the EEX traded price, the only benchmark of the level forecast
    st.subheader("Versus the market: the forecast against the EEX French day-ahead future traded before the auction")
    r = MARKET_RECORD
    market = status.get("market") or {}
    needed = int(market.get("min_days_to_show", 20))
    live_windows = int(market.get("scored_windows", market.get("scored_days", 0)) or 0)
    if live_windows >= needed and "hit_rate" in market:
        st.markdown(f"## Live record, {live_windows} windows")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Windows scored", live_windows)
        c2.metric("Hit rate", f"{100 * market['hit_rate']:.0f}%")
        c3.metric("Mean P&L per MWh", f"{market['mean_pnl_per_mwh']:+.2f} EUR")
        c4.metric("Model vs market error", f"{market['model_mae']:.1f} vs {market['market_mae']:.1f} EUR/MWh")
        st.caption("Live: the pre-market version published on this site against the traded VWAP of the EEX French day-ahead future "
                   "(11:15 to 12:00 Paris on the trade date), long when above, short when below, settled at the auction result. Market prices "
                   "come from EEX and are not republished here; only these aggregates are.")
    else:
        st.markdown(f"## Versus the market: collecting data, {live_windows} of {needed}")
        st.caption("What this counts: windows (one per delivery day and product) of settled delivery days whose pre-market version of the default "
                   "model was published on this site before the EEX trading window (11:15 Paris on the trade date) and for which a traded price "
                   "was recorded; versions issued later are not counted. The traded prices live on the author's machine, which computes these "
                   "aggregates; the automated updates carry them through unchanged"
                   + (f" (last computed {market['computed_at_utc'][:10]})" if market.get("computed_at_utc") else "") + ".")
    st.markdown(f"**Walk-forward record, not live.** {r['model']}, the default model, reissued over {r['windows']} windows ({r['base_windows']} base, "
                f"{r['peak_windows']} peak; {r['period']}): hit rate {100 * r['hit_rate']:.0f}%, mean P&L {r['mean_pnl_per_mwh']:+.2f} EUR/MWh per "
                f"window, model error against the auction {r['model_mae']:.1f} EUR/MWh where the traded VWAP's was {r['market_mae']:.1f}; the "
                f"model's error was the smaller one on {100 * r['share_model_beats_market']:.0f}% of windows. Base contract alone: {r['base_only']}. "
                f"{r['reason']}")
    st.caption("How a window is scored: the forecast issued before the window against the window's VWAP of the EEX French Day-Ahead Base or "
               "Peak future, 11:15 to 12:00 Paris on the trade date; long when the forecast is above, short when below; P&L per MWh is the "
               "auction result minus the entry, times the direction. The walk-forward reissues each day with a model fitted only on earlier "
               f"data, or with the version actually issued before the window where one existed ({r['source']}). Traded prices come from EEX and "
               "are not republished; only aggregates appear here. Since 2026-10-12 the EEX traded price is the only benchmark of the level "
               "forecast: no same-hour-previous-day comparison is shown or published.")
    default_set = status.get("feature_set", "honest_v2")
    previous_set = status.get("fallback_model_file", "model/honest.txt").split("/")[-1].replace(".txt", "")
    record, tables = live_record(data.get("scores_all", data["scores"]), default_set, previous_set)
    head = record["default"]
    if head["status"] == "ok":
        st.markdown(f"Model error against the auction result, live: MAE {head['model_mae']:.2f} EUR/MWh over {head['days_in_window']} settled "
                    f"day(s) with a pre-market {default_set} version (since {head['first_premarket_day']}). The model's own error, with no other "
                    "comparison; the panel below has the daily values.")
    else:
        st.caption(f"Model error against the auction result, live: collecting data, no settled delivery day with a pre-market {default_set} "
                   "version yet.")

    # 2. Tomorrow's forecast
    st.subheader("Tomorrow's forecast")
    by_hour = tomorrow_table(data)
    versions = tomorrow_versions(data)
    if by_hour.empty:
        st.info("No forecast for the next delivery day yet.")
    else:
        tomorrow = versions["headline"]
        day = versions["day"]
        previous = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        issued = pd.Timestamp(tomorrow["issued_at_utc"].iloc[0]).tz_convert(PARIS)
        if versions["premarket"]:
            st.markdown(f"**Pre-market forecast**: the last version issued before the market window opened at {MARKET_WINDOW_START} Paris "
                        f"on {previous} (issued {issued:%H:%M}).")
        else:
            st.warning(f"No version was issued before the {MARKET_WINDOW_START} Paris market window on {previous}. This one was issued after "
                       "the market window and is not tradeable; it is shown for information.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delivery day", day)
        c2.metric("Issued (Paris)", issued.strftime("%Y-%m-%d %H:%M"))
        c3.metric("Daily mean forecast", f"{by_hour['forecast'].mean():.1f} EUR/MWh")
        c4.metric("Model", model_label(versions.get("model", "gbm")), help="gbm:<file> is a stored model; :fallback means the default model's "
                  "inputs were missing and the fallback model was used; gbm alone is a model fitted live")
        if pd.Timestamp(day) < pd.Timestamp.now(tz=PARIS).normalize().tz_localize(None):
            st.warning(f"The latest published forecast is for {day}, which is in the past. The workflow may not have run today.")
        st.plotly_chart(tomorrow_chart(by_hour, day, previous), use_container_width=True)
        if "band_p10" in by_hour:
            st.caption("Shaded band: historical error of this model at each hour in the 2024 to 2025 backtest, not a probability forecast.")
        with st.expander("Hourly values"):
            st.dataframe(by_hour[["hour", "forecast", "actual_today"]].round(2).rename(columns={"actual_today": f"actual {previous}"}),
                         hide_index=True, use_container_width=True)
        later = versions["later"]
        if versions["premarket"] and not later.empty:
            st.markdown("**Issued after the market window, not tradeable**")
            st.dataframe(later[["issued_paris", "kind", "model", "daily_mean"]].round(2).rename(
                columns={"issued_paris": "Issued (Paris)", "kind": "Run kind", "model": "Model", "daily_mean": "Daily mean forecast (EUR/MWh)"}),
                hide_index=True, use_container_width=True)

    # 3. The model's own error against the auction result (no other comparison)
    st.subheader(f"Forecast error against the auction result, last 30 days: default model {default_set}, pre-market version of each day")
    st.caption(record["rule"])
    if head["status"] != "ok":
        st.info(f"Collecting data: no settled delivery day with a pre-market {default_set} version yet. The first scheduled pre-market run "
                "settles the day after it is issued.")
        if late_only_text(head):
            st.warning(late_only_text(head))
    else:
        scores = tables["default"].tail(30).copy()
        scores["rolling_mae"] = scores["mae"].rolling(7, min_periods=1).mean()
        c1, c2 = st.columns(2)
        c1.metric("Days settled", head["days_in_window"], help=f"pre-market {default_set} days since {head['first_premarket_day']}, last 30")
        c2.metric("Model MAE against the auction", f"{head['model_mae']:.2f} EUR/MWh")
        if late_only_text(head):
            st.warning(late_only_text(head))
        st.plotly_chart(performance_chart(scores), use_container_width=True)
        hist = history_chart(data)
        if hist is not None:
            st.plotly_chart(hist, use_container_width=True)
        with st.expander("Daily scores"):
            st.dataframe(scores[[c for c in ("delivery_day", "issued_at_utc", "model", "hours", "mae", "rmse") if c in scores]].round(2),
                         hide_index=True, use_container_width=True)
    previous = record.get("previous")
    if previous:
        if previous["days"]:
            st.markdown(f"**Previous model ({previous_set} set, {previous['first_premarket_day']} to {previous['last_premarket_day']}, before the "
                        f"default model's first pre-market day)**: {previous['days']} day(s) settled with a pre-market version, MAE against the "
                        f"auction {previous['model_mae']:.2f} EUR/MWh. Its versions include models fitted live before the stored model file "
                        f"existed (model column \"gbm\"). {late_only_text(previous)}")
        else:
            st.markdown(f"**Previous model ({previous_set} set)**: no day with a pre-market version. {late_only_text(previous)}")
        with st.expander(f"Daily scores, previous model ({previous_set})"):
            prev_table = tables["previous"]
            st.dataframe(prev_table[[c for c in ("delivery_day", "issued_at_utc", "model", "hours", "mae", "rmse") if c in prev_table]].round(2),
                         hide_index=True, use_container_width=True)

    st.subheader("Negative-price probability and its calibration")
    st.caption("This panel has no EEX equivalent: a probability of a negative price is compared with its own reference frequencies "
               "(climatology and the last 7 days), shown here only, below the market headline above.")
    prob = data.get("prob_tomorrow", pd.DataFrame())
    prob_status = status.get("probabilistic") or {}
    if not prob.empty and "p_negative" in prob:
        hourly = prob.groupby("hour").agg(p_negative=("p_negative", "mean")).reset_index()
        st.markdown("**Tomorrow's per-hour probability of a negative price** (headline version), from a classifier on the same inputs as the "
                    "point forecast.")
        summary = negative_summary(hourly)
        st.markdown(f"**{summary['text']}**")
        shown = hourly["p_negative"].map(format_probability)
        fig = go.Figure(go.Bar(x=hourly["hour"], y=hourly["p_negative"], name="P(price < 0)", marker_color=COLORS["forecast"],
                               customdata=shown, hovertemplate="%{x}:00 Paris: %{customdata}<extra></extra>"))
        top = probability_axis_top(summary["max_p"])
        fig.update_layout(yaxis=dict(range=[0, top], tickformat=".0%"))
        if show_half_line(top):
            fig.add_hline(y=0.5, line=dict(color=COLORS["grid"], width=1, dash="dot"))
        st.plotly_chart(layout(fig, "Probability", "Delivery hour (Paris time)"), use_container_width=True)
        with st.expander("Hourly probabilities"):
            st.dataframe(pd.DataFrame({"hour": hourly["hour"], "P(price < 0)": shown}), hide_index=True, use_container_width=True)
    else:
        st.info("No negative-price probability for tomorrow's headline version yet (a version issued without the probabilistic model files "
                "carries none).")
    backtest = data.get("prob_backtest") or {}
    live = prob_status.get("live") or {}
    rows = []
    for label, record in backtest.items():
        r = record.get("events", {}).get("negative", {})
        rows.append({"window": f"backtest {label} ({record.get('first_day')} to {record.get('last_day')})", "hours": r.get("hours"), "events": r.get("events"),
                     "Brier (model)": r.get("brier", {}).get("model"), "Brier (climatology)": r.get("brier", {}).get("clim"),
                     "Brier (last 7 days)": r.get("brier", {}).get("last7"), "BSS vs climatology": r.get("bss_vs_clim"), "BSS vs last 7 days": r.get("bss_vs_last7"),
                     "log loss": r.get("log_loss", {}).get("model")})
    if live.get("days") and live.get("negative"):
        n = live["negative"]
        rows.append({"window": f"live, {live['days']} settled days", "hours": n.get("hours"), "events": n.get("events"), "Brier (model)": n.get("brier"),
                     "Brier (climatology)": n.get("brier_base_rate"), "log loss": n.get("log_loss")})
    if rows:
        st.markdown("**Calibration** (aggregates only). Benchmarks: climatology, the frequency for that hour of day and calendar month over the "
                    "trailing year; and the frequency at that hour over the last 7 days. BSS is the Brier skill score; positive means better than "
                    "the benchmark. On the live row the benchmark is the base rate of the settled days.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for label, record in backtest.items():
            table = pd.DataFrame(record.get("events", {}).get("negative", {}).get("reliability", []))
            if not table.empty:
                with st.expander(f"Reliability table, backtest {label} (10 bins)"):
                    st.dataframe(table, hide_index=True, use_container_width=True)
        if live.get("negative", {}).get("reliability"):
            with st.expander("Reliability table, live"):
                st.dataframe(pd.DataFrame(live["negative"]["reliability"]), hide_index=True, use_container_width=True)
    if lowest_bin_sentence(backtest):
        st.caption(lowest_bin_sentence(backtest))
    st.caption("Accepted on the selection window and confirmed on the holdout (158 events, Brier skill +0.37 and +0.41); see docs/experiments.md. "
               "A quantile band and a spike probability are computed and stored every day but are under evaluation and not shown here.")

    st.subheader("Shape and battery value")
    st.caption("This panel has no EEX equivalent: the shape of the day is compared with its own references (yesterday's shape and the "
               "same-type day's shape) and valued with a battery, shown here only, below the market headline above.")
    shape = data.get("shape") or {}
    backtest = shape.get("backtest") or {}
    if backtest:
        st.markdown("The level forecast is one thing; the shape of the day, each hour's price minus the day's mean, is what a storage asset "
                    "trades on. A shape model (gradient boosting on the same inputs, the shape as target) is compared with two benchmarks: "
                    "yesterday's shape and the shape of the most recent day of the same type. The battery is 1 MW / 2 MWh with 88% "
                    "round-trip efficiency, one cycle a day: a two-hour charge block before a two-hour discharge block chosen on the forecast "
                    "before the gate and settled at the auction result, skipped when the forecast spread does not cover the efficiency loss.")
        for label, record in backtest.items():
            st.markdown(f"**{label.capitalize()} window, {record['first_day']} to {record['last_day']}, {record['shape']['days']} days**")
            rows = []
            for method, name in (("model", "shape model"), ("d1", "D-1 shape"), ("same_type", "same-type day shape")):
                s, b = record["shape"][method], record["battery"][method]
                rows.append({"method": name, "shape MAE (EUR/MWh)": s["shape_mae"], "spread error (EUR/MWh)": s["spread_error"],
                             "cheapest-2 hit rate": f"{100 * s['cheapest2_hit_rate']:.0f}%", "dearest-2 hit rate": f"{100 * s['dearest2_hit_rate']:.0f}%",
                             "battery EUR/day": b["eur_per_day"], "share of perfect foresight": f"{100 * (b['share_of_perfect'] or 0):.0f}%"})
            rows.append({"method": "perfect foresight", "battery EUR/day": record["battery"]["perfect_eur_per_day"], "share of perfect foresight": "100%"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            bars = pd.DataFrame({"method": ["shape model", "D-1 shape", "same-type day shape", "perfect foresight"],
                                 "EUR/day": [record["battery"]["model"]["eur_per_day"], record["battery"]["d1"]["eur_per_day"],
                                             record["battery"]["same_type"]["eur_per_day"], record["battery"]["perfect_eur_per_day"]]})
            fig = go.Figure(go.Bar(x=bars["method"], y=bars["EUR/day"], marker_color=[COLORS["forecast"], COLORS["benchmark"], COLORS["benchmark"],
                                                                                   COLORS["actual"]]))
            st.plotly_chart(layout(fig, "Battery value (EUR/day)", ""), use_container_width=True)
    live = shape.get("live") or {}
    if live.get("days"):
        st.markdown(f"**Live record, {live['days']} settled days** (pre-market version of each day): shape MAE "
                    f"{live['shape']['model']['shape_mae']:.2f} against {live['shape']['d1']['shape_mae']:.2f} for yesterday's shape; battery "
                    f"{live['battery']['model']['eur_per_day']:.1f} EUR/day, {100 * (live['battery']['model']['share_of_perfect'] or 0):.0f}% of "
                    f"perfect foresight ({live['battery']['perfect_eur_per_day']:.1f} EUR/day).")
        daily = pd.DataFrame(live["daily"])
        if not daily.empty:
            fig = go.Figure()
            for column, name, color in (("value_model", "shape model", COLORS["forecast"]), ("value_d1", "D-1 shape", COLORS["benchmark"]),
                                        ("value_perfect", "perfect foresight", COLORS["actual"])):
                fig.add_trace(go.Scatter(x=daily["delivery_day"], y=daily[column].cumsum(), name=name, mode="lines", line=dict(color=color)))
            st.plotly_chart(layout(fig, "Cumulative battery value (EUR)", "Delivery day"), use_container_width=True)
    elif backtest:
        st.caption("The live record starts with the first settled day forecast by the default model (honest_v2); none has settled yet.")
    else:
        st.info("No shape and battery results published yet.")
    st.caption("Aggregates and the live daily values only; no traded prices appear here.")

    st.subheader("How it works")
    st.markdown(HOW_IT_WORKS)

    st.subheader("Data sources and attribution")
    for line in status.get("attributions", []):
        st.markdown(f"- {line}")
    st.caption("Forecasts and derived numbers are the author's own and carry no endorsement by the data providers. "
               "Fundamentals come from ENTSO-E, RTE and Open-Meteo; spot prices are the EPEX auction results as published by ENTSO-E; "
               "traded prices are EEX day-ahead futures, collected locally and not republished. Nothing here is trading advice.")


if __name__ == "__main__":
    main()
