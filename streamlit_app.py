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
    return {
        "status": status,
        "tomorrow": read("tomorrow.csv"),
        "forecasts": read("forecasts.csv"),
        "actuals": read("actuals.csv"),
        "scores": read("scores.csv"),
        "band": read("error_band.csv"),
    }


def published_version() -> float:
    path = PUBLISHED / "status.json"
    return path.stat().st_mtime if path.exists() else 0.0


def layout(fig: go.Figure, ytitle: str, xtitle: str) -> go.Figure:
    fig.update_layout(template="plotly_white", height=400, margin=dict(l=40, r=20, t=30, b=40), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      xaxis=dict(title=xtitle, gridcolor=COLORS["grid"]), yaxis=dict(title=ytitle, gridcolor=COLORS["grid"]))
    return fig


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
    later = (later_rows.groupby("issued_at_utc").agg(daily_mean=("forecast", "mean"), kind=("kind", "first")).reset_index()
             if not later_rows.empty else pd.DataFrame(columns=["issued_at_utc", "daily_mean", "kind"]))
    later["issued_paris"] = [pd.Timestamp(t).tz_convert(PARIS).strftime("%Y-%m-%d %H:%M") for t in later["issued_at_utc"]]
    return {"day": day, "headline": headline, "premarket": bool(head["premarket"]), "later": later}


def tomorrow_table(data: dict) -> pd.DataFrame:
    versions = tomorrow_versions(data)
    if not versions:
        return pd.DataFrame()
    tomorrow = versions["headline"]
    by_hour = tomorrow.groupby("hour").agg(forecast=("forecast", "mean"), naive_day=("naive_day", "mean")).reset_index()
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
    return by_hour


def tomorrow_chart(by_hour: pd.DataFrame, day: str, previous: str) -> go.Figure:
    fig = go.Figure()
    x = by_hour["hour"]
    if "band_p10" in by_hour:
        for lo, hi, color, name in (("band_p10", "band_p90", BAND[0], "Historical error, 10th to 90th pct"),
                                    ("band_p25", "band_p75", BAND[1], "Historical error, 25th to 75th pct")):
            fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([by_hour[hi], by_hour[lo][::-1]]), fill="toself",
                                     fillcolor=color, line=dict(width=0), name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=by_hour["naive_day"], name=f"Same hour on {previous} (baseline)", mode="lines",
                             line=dict(color=COLORS["benchmark"], width=2, dash="dash")))
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
    latest["rolling_naive_mae"] = latest["naive_mae"].rolling(7, min_periods=1).mean()
    return latest


def performance_chart(scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scores["delivery_day"], y=scores["naive_mae"], name="Baseline daily MAE", marker_color=COLORS["benchmark"], opacity=0.5))
    fig.add_trace(go.Bar(x=scores["delivery_day"], y=scores["mae"], name="Model daily MAE", marker_color=COLORS["forecast"]))
    fig.add_trace(go.Scatter(x=scores["delivery_day"], y=scores["rolling_naive_mae"], name="Baseline, 7-day rolling", mode="lines",
                             line=dict(color=COLORS["benchmark"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=scores["delivery_day"], y=scores["rolling_mae"], name="Model, 7-day rolling", mode="lines",
                             line=dict(color=COLORS["ink"], width=2)))
    fig.update_layout(barmode="group")
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
    fig.add_trace(go.Scatter(x=joined["t"], y=joined["forecast"], name="Forecast (latest version)", mode="lines", line=dict(color=COLORS["forecast"], width=2)))
    fig.add_trace(go.Scatter(x=joined["t"], y=joined["naive_day"], name="Baseline", mode="lines", line=dict(color=COLORS["benchmark"], width=1, dash="dash")))
    return layout(fig, "EUR/MWh", "Delivery hour (Paris time)")


# The backfilled trading record, scored as a walk-forward (see the repository's docs/forecast.md); aggregates only,
# the traded prices themselves are not republished. Updated by hand when the record is re-scored.
MARKET_RECORD = {
    "period": "delivery days 2026-08-29 to 2026-10-10",
    "windows": 42, "base_windows": 38, "peak_windows": 4,
    "hit_rate": 0.29, "mean_pnl_per_mwh": -2.5, "model_mae": 27.0, "market_mae": 5.1, "share_model_beats_market": 0.12,
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

**Model.** Gradient boosting (LightGBM) fitted on all history before the delivery day. Results are reported as forecast
error (mean absolute error, MAE, in EUR/MWh) against a naive baseline: the price of the same hour on the previous day. In a
walk-forward backtest over 2024 and 2025 the model's MAE was about 15.5 EUR/MWh and the baseline's about 21.

**Error band.** The shaded band around tomorrow's curve is the forecast plus the 10th to 90th (and 25th to 75th) percentile
of the model's signed error at the same hour in that backtest. It describes how wrong the model has been at that hour in
the past; it is not a probability forecast for tomorrow.

**Updates.** A GitHub Actions workflow publishes the forecast on weekday mornings and scores it against the
published prices every afternoon. Source code, method and backtest: the repository linked above.

**Limitations.** Two measurements give two different answers. On forecast error the model beats the naive baseline, the spot
auction result of the same hour on the previous day. Against EEX French day-ahead futures traded before the auction it loses
(the "Versus the market" panel): over 42 windows the direction implied by the forecast had a 29% hit rate and a mean P&L of
-2.5 EUR/MWh per window, and the forecast's error against the auction was 27.0 EUR/MWh where the traded price's was 5.1. The
model reacts to large moves about two days late, because its price lags carry yesterday's level into tomorrow's forecast,
while the market reprices the same morning. The error band is the model's
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
    st.caption(f"Dataset written {written:%Y-%m-%d %H:%M} Paris. Window: last {status['window_days']} days.")

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
        c1, c2, c3 = st.columns(3)
        c1.metric("Delivery day", day)
        c2.metric("Issued (Paris)", issued.strftime("%Y-%m-%d %H:%M"))
        c3.metric("Daily mean forecast", f"{by_hour['forecast'].mean():.1f} EUR/MWh")
        if pd.Timestamp(day) < pd.Timestamp.now(tz=PARIS).normalize().tz_localize(None):
            st.warning(f"The latest published forecast is for {day}, which is in the past. The workflow may not have run today.")
        st.plotly_chart(tomorrow_chart(by_hour, day, previous), use_container_width=True)
        if "band_p10" in by_hour:
            st.caption("Shaded band: historical error of this model at each hour in the 2024 to 2025 backtest, not a probability forecast.")
        with st.expander("Hourly values"):
            st.dataframe(by_hour[["hour", "forecast", "naive_day", "actual_today"]].round(2).rename(
                columns={"naive_day": "baseline", "actual_today": f"actual {previous}"}), hide_index=True, use_container_width=True)
        later = versions["later"]
        if versions["premarket"] and not later.empty:
            st.markdown("**Issued after the market window, not tradeable**")
            st.dataframe(later[["issued_paris", "kind", "daily_mean"]].round(2).rename(
                columns={"issued_paris": "Issued (Paris)", "kind": "Run kind", "daily_mean": "Daily mean forecast (EUR/MWh)"}),
                hide_index=True, use_container_width=True)

    st.subheader("Forecast error, last 30 days (pre-market version of each day)")
    scores = daily_scores(data)
    if scores.empty:
        st.info("No settled days yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days settled", int(len(scores)))
        c2.metric("Model MAE", f"{scores['mae'].mean():.2f} EUR/MWh")
        c3.metric("Baseline MAE (same hour D-1)", f"{scores['naive_mae'].mean():.2f} EUR/MWh")
        c4.metric("Days with lower MAE than the baseline", f"{100 * scores['mae_below_baseline'].fillna(0).mean():.0f}%")
        st.plotly_chart(performance_chart(scores), use_container_width=True)
        hist = history_chart(data)
        if hist is not None:
            st.plotly_chart(hist, use_container_width=True)
        with st.expander("Daily scores"):
            st.dataframe(scores[["delivery_day", "issued_at_utc", "hours", "mae", "rmse", "naive_mae", "naive_rmse", "mae_below_baseline"]].round(2),
                         hide_index=True, use_container_width=True)

    st.subheader("Versus the market")
    r = MARKET_RECORD
    st.markdown(f"**The model beats the naive baseline on forecast error and loses against the traded price.** Over {r['windows']} windows "
                f"({r['base_windows']} base, {r['peak_windows']} peak; {r['period']}) the direction implied by the forecast against the EEX "
                f"French day-ahead future's 11:15 to 12:00 Paris VWAP, settled at the auction result, had a hit rate of {100 * r['hit_rate']:.0f}% "
                f"and a mean P&L of {r['mean_pnl_per_mwh']:+.1f} EUR/MWh per window. The forecast's error against the auction was "
                f"{r['model_mae']:.1f} EUR/MWh where the traded price's was {r['market_mae']:.1f}; the forecast's error was the smaller one on "
                f"{100 * r['share_model_beats_market']:.0f}% of windows. {r['reason']}")
    st.caption("Backfilled record scored as a walk-forward: each day with a model fitted only on data before it, or with the version "
               "actually issued before the window where one existed. Traded prices come from EEX and are not republished; the live "
               "record below grows one weekday at a time.")
    market = status.get("market") or {}
    if market.get("scored_days", 0) and "hit_rate" in market:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days scored", int(market["scored_days"]))
        c2.metric("Hit rate", f"{100 * market['hit_rate']:.0f}%")
        c3.metric("Mean P&L per MWh", f"{market['mean_pnl_per_mwh']:+.2f} EUR")
        c4.metric("Model vs market error", f"{market['model_mae']:.1f} vs {market['market_mae']:.1f} EUR/MWh")
        st.caption("Trading value: the forecast issued before the EEX trading window against the traded VWAP of the French day-ahead "
                   "future, long when above, short when below, settled at the auction result. Market prices come from EEX and are not "
                   "republished here; only these aggregates are.")
    else:
        needed = int(market.get("min_days_to_show", 20))
        st.markdown(f"**Live record: collecting data, {int(market.get('scored_days', 0))} of {needed} days.**")
        st.caption("The live record counts only days whose forecast was published before the trading window; its aggregates appear "
                   "once enough days are scored.")

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
