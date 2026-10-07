"""Streamlit dashboard for the French day-ahead price forecast.

Run with `thermo-fr dashboard` (which calls `streamlit run` on this file). It
reads only the SQLite database written by the scheduled jobs; the only way it
touches the network is the "Refresh now" button, which runs morning-run once
in a subprocess.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from thermo_fr.dashboard import data as q

COLORS = {"honest": "#2a78d6", "extended": "#eb6834", "actual": "#0b0b0b", "benchmark": "#52514e", "grid": "#e6e5e1"}
BAND_RGBA = {"honest": ("rgba(42,120,214,0.14)", "rgba(42,120,214,0.30)"), "extended": ("rgba(235,104,52,0.14)", "rgba(235,104,52,0.30)")}
EXTENDED_NOTE = (
    "The extended feature set adds the ENTSO-E day-ahead wind and solar forecasts, which the platform allows to be "
    "published until 18:00 on the day before delivery, after the 12:00 gate. Until the timing log shows them arriving "
    "before 12:00 on every logged day, treat extended forecasts as possibly using late information."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/forecast.db")
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


@st.cache_data(ttl=60, show_spinner=False)
def forecast_panel(path, version, day, feature_set, actual_day):
    return q.forecast_panel(path, day, feature_set, actual_day)


@st.cache_data(ttl=60, show_spinner=False)
def inputs_panel(path, version, day, previous):
    return q.inputs_panel(path, day, previous)


@st.cache_data(ttl=60, show_spinner=False)
def performance_panel(path, version, feature_set):
    return q.performance_panel(path, feature_set)


@st.cache_data(ttl=60, show_spinner=False)
def status_panel(path, version, day):
    return q.status_panel(path, day)


def base_layout(fig: go.Figure, ytitle: str, xtitle: str = "Delivery hour (Paris time)") -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=380, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=xtitle, gridcolor=COLORS["grid"], dtick=2), yaxis=dict(title=ytitle, gridcolor=COLORS["grid"]),
        hovermode="x unified",
    )
    return fig


def forecast_chart(by_hour: pd.DataFrame, feature_set: str, today: str, has_band: bool) -> go.Figure:
    fig = go.Figure()
    x = by_hour["hour"]
    if has_band:
        light, dark = BAND_RGBA[feature_set]
        for lo, hi, color, name in (("band_p10", "band_p90", light, "Historical error, 10th to 90th pct"),
                                    ("band_p25", "band_p75", dark, "Historical error, 25th to 75th pct")):
            fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([by_hour[hi], by_hour[lo][::-1]]), fill="toself",
                                     fillcolor=color, line=dict(width=0), name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=by_hour["naive_day"], name="Same hour previous day (benchmark)", mode="lines",
                             line=dict(color=COLORS["benchmark"], width=2, dash="dash")))
    if by_hour["actual_other"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=by_hour["actual_other"], name=f"Actual price today ({today})", mode="lines",
                                 line=dict(color=COLORS["actual"], width=2)))
    if by_hour["actual_own"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=by_hour["actual_own"], name="Actual price for this delivery day", mode="lines",
                                 line=dict(color=COLORS["actual"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=by_hour["forecast"], name=f"Forecast ({feature_set})", mode="lines+markers",
                             line=dict(color=COLORS[feature_set], width=3), marker=dict(size=6)))
    return base_layout(fig, "EUR/MWh")


def inputs_chart(hourly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = hourly.index.tz_convert("Europe/Paris").hour
    for column, color in (("load_fc_mw", "#2a78d6"), ("residual_fc_mw", "#eb6834"), ("wind_fc_mw", "#1baf7a"), ("solar_fc_mw", "#eda100")):
        fig.add_trace(go.Scatter(x=x, y=hourly[column], name=q.INPUT_LABELS[column], mode="lines", line=dict(color=color, width=2)))
    return base_layout(fig, "MW")


def temperature_chart(hourly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = hourly.index.tz_convert("Europe/Paris").hour
    fig.add_trace(go.Scatter(x=x, y=hourly["temp_fc_c"], name="Temperature forecast, population weighted", mode="lines",
                             line=dict(color="#4a3aa7", width=2)))
    return base_layout(fig, "C")


def performance_chart(scores: pd.DataFrame, feature_set: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scores["delivery_day"], y=scores["naive_mae"], name="Benchmark daily MAE", marker_color=COLORS["benchmark"], opacity=0.5))
    fig.add_trace(go.Bar(x=scores["delivery_day"], y=scores["mae"], name=f"Model daily MAE ({feature_set})", marker_color=COLORS[feature_set]))
    fig.add_trace(go.Scatter(x=scores["delivery_day"], y=scores["rolling_naive_mae"], name="Benchmark, 7-day rolling", mode="lines",
                             line=dict(color=COLORS["benchmark"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=scores["delivery_day"], y=scores["rolling_mae"], name="Model, 7-day rolling", mode="lines",
                             line=dict(color=COLORS["actual"], width=2)))
    fig.update_layout(barmode="group")
    return base_layout(fig, "MAE, EUR/MWh", "Delivery day")


def hourly_history_chart(hourly: pd.DataFrame, feature_set: str) -> go.Figure:
    fig = go.Figure()
    x = hourly.index.tz_convert("Europe/Paris")
    fig.add_trace(go.Scatter(x=x, y=hourly["actual"], name="Actual price", mode="lines", line=dict(color=COLORS["actual"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=hourly["forecast"], name=f"Forecast ({feature_set})", mode="lines", line=dict(color=COLORS[feature_set], width=2)))
    fig.add_trace(go.Scatter(x=x, y=hourly["naive_day"], name="Benchmark", mode="lines", line=dict(color=COLORS["benchmark"], width=1, dash="dash")))
    fig = base_layout(fig, "EUR/MWh", "Delivery hour (Paris time)")
    fig.update_xaxes(dtick=None)
    return fig


def run_morning(db: str) -> str:
    result = subprocess.run([sys.executable, "-m", "thermo_fr", "morning-run", "--kind", "manual", "--db", db],
                            capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()[-2000:]


def main() -> None:
    args = parse_args()
    db = args.db
    st.set_page_config(page_title="thermo-fr day-ahead forecast", layout="wide")
    st.title("French day-ahead price forecast")
    today, tomorrow = q.today_and_tomorrow()
    version = q.db_version(db)

    with st.sidebar:
        st.header("Settings")
        feature_set = st.radio("Feature set", ("honest", "extended"), index=0,
                               help="Honest: every input is published before 12:00 Paris on the day before delivery.")
        st.caption("Honest set: calendar, ENTSO-E load forecast, Open-Meteo weather issued two days ahead, lagged prices.")
        if feature_set == "extended":
            st.warning(EXTENDED_NOTE)
        st.divider()
        if st.button("Refresh now (runs morning-run once)"):
            with st.spinner("Running morning-run (fetching inputs, fitting the models) ..."):
                output = run_morning(db)
            st.code(output or "no output", language="text")
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Database: {db}")
        st.caption("Data is read from the database only; the scheduled jobs call the APIs.")
        if not Path(db).exists():
            st.error("No database yet. Run `thermo-fr morning-run` once.")
            st.stop()

    # 1. Tomorrow's forecast
    st.subheader(f"Tomorrow's forecast: delivery day {tomorrow}")
    panel = forecast_panel(db, version, tomorrow, feature_set, today)
    if panel["meta"] is None:
        st.info(f"No {feature_set} forecast stored for {tomorrow} yet. The morning jobs run from 07:00 London time; use Refresh now to run one.")
    else:
        meta = panel["meta"]
        issued = pd.Timestamp(meta["issued_at_utc"]).tz_convert("Europe/Paris")
        gate_badge = "passes the 12:00 gate" if meta["passes_gate"] else "may use information published after the 12:00 gate"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Issued at (Paris)", issued.strftime("%H:%M"), help=issued.strftime("%Y-%m-%d %H:%M Paris"))
        c2.metric("Versions today", int(len(panel["versions"])))
        c3.metric("Daily mean forecast", f"{panel['by_hour']['forecast'].mean():.1f} EUR/MWh")
        c4.metric("Run kind", str(meta["kind"]))
        st.caption(f"Issued {issued:%Y-%m-%d %H:%M} Paris. Model: {meta['model']} fitted on {int(meta['train_hours']):,} hours; "
                   f"this feature set {gate_badge}.")
        has_band = "band_p10" in panel["by_hour"]
        st.plotly_chart(forecast_chart(panel["by_hour"], feature_set, today, has_band), use_container_width=True)
        if has_band:
            n = int(panel["by_hour"]["n"].min())
            st.caption(f"Shaded band: the forecast plus the 10th to 90th (and 25th to 75th) percentile of the signed error "
                       f"of this model at the same hour in the 2024 to 2025 walk-forward backtest ({n:,}+ hours per hour of day). "
                       "It describes historical error, not a probability forecast for this day.")
        else:
            st.caption("No error band: run the backtest (thermo-fr forecast-backtest) so morning-run can derive it.")
        with st.expander("Hourly table"):
            show = panel["by_hour"][["hour", "forecast", "naive_day", "actual_other"]].rename(
                columns={"naive_day": "benchmark (D-1)", "actual_other": f"actual {today}"})
            st.dataframe(show.round(2), hide_index=True, use_container_width=True)
        if len(panel["versions"]) > 1:
            with st.expander("Earlier versions issued for this day"):
                st.dataframe(panel["versions"][["issued_at_utc", "kind", "train_hours", "passes_gate"]], hide_index=True)

    # 2. Inputs
    st.subheader(f"Inputs for {tomorrow}")
    inputs = inputs_panel(db, version, tomorrow, today)
    if not inputs["tomorrow"]:
        st.info("No inputs stored for tomorrow yet.")
    else:
        means = inputs["tomorrow"]["means"]
        previous = inputs["today"]["means"] if inputs["today"] else {}
        cols = st.columns(5)
        for col, key in zip(cols, ("load_fc_mw", "wind_fc_mw", "solar_fc_mw", "residual_fc_mw", "temp_fc_c")):
            value = means.get(key)
            if value is None or pd.isna(value):
                col.metric(q.INPUT_LABELS[key], "n/a")
                continue
            delta = None
            if key in previous and not pd.isna(previous[key]):
                delta = value - previous[key]
            unit = "C" if key.endswith("_c") else "MW"
            col.metric(q.INPUT_LABELS[key].replace(" (MW)", "").replace(" (C)", ""),
                       f"{value:,.1f} {unit}" if unit == "C" else f"{value:,.0f} {unit}",
                       None if delta is None else (f"{delta:+.1f} {unit}" if unit == "C" else f"{delta:+,.0f} {unit}"))
        st.caption(f"Daily means for {tomorrow}; the change is against the inputs stored for today ({today}) by yesterday's run."
                   if previous else f"Daily means for {tomorrow}; no stored inputs for today to compare with.")
        left, right = st.columns([2, 1])
        left.plotly_chart(inputs_chart(inputs["tomorrow"]["hourly"]), use_container_width=True)
        right.plotly_chart(temperature_chart(inputs["tomorrow"]["hourly"]), use_container_width=True)

    # 3. Recent performance
    st.subheader(f"Recent performance, last 30 settled days ({feature_set})")
    perf = performance_panel(db, version, feature_set)
    if perf["scores"].empty:
        st.info("No settled forecasts yet. The settle job runs at 14:00 London time after the auction results are out.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days settled", int(len(perf["scores"])))
        c2.metric("Model MAE", f"{perf['mae']:.2f} EUR/MWh")
        c3.metric("Benchmark MAE", f"{perf['naive_mae']:.2f} EUR/MWh")
        c4.metric("Share of days the model won", f"{100 * perf['share_won']:.0f}%")
        st.plotly_chart(performance_chart(perf["scores"], feature_set), use_container_width=True)
        if not perf["hourly"].empty:
            st.plotly_chart(hourly_history_chart(perf["hourly"], feature_set), use_container_width=True)
        with st.expander("Daily scores"):
            st.dataframe(perf["scores"][["delivery_day", "issued_at_utc", "hours", "mae", "rmse", "naive_mae", "naive_rmse", "model_won"]].round(2),
                         hide_index=True, use_container_width=True)

    # 4. Data status
    st.subheader(f"Data status for {tomorrow}")
    status = status_panel(db, version, tomorrow)
    if "table" not in status:
        st.info("No checks logged for tomorrow yet.")
    else:
        table = status["table"]
        problems = table[table["flag"] != "ok"]
        if problems.empty:
            st.success("Every input is present and arrived before the gate.")
        else:
            for item, row in problems.iterrows():
                st.warning(f"{item}: {row['flag']}" + (f" ({row['message'][:120]})" if row["message"] else ""))
        st.dataframe(table.rename(columns={"first_seen_paris": "first seen (Paris)", "minutes_before_gate": "minutes before gate",
                                           "checked_at_utc": "last checked (UTC)"}), use_container_width=True)
    st.markdown("**Timing probe so far: first appearance by input, across all logged delivery days**")
    if status["summary"].empty:
        st.info("The timing log is empty.")
    else:
        st.dataframe(status["summary"].round(0), hide_index=True, use_container_width=True)
        st.caption("Minutes before the 12:00 Paris gate at which the input was first seen (negative means after the gate). "
                   "Only the runs' polling times are known, so a first-appearance time is an upper bound on the true publication time, "
                   "and only days whose first poll was before the gate (days_polled_before_gate) say anything about the gate.")
        with st.expander("Timing log by day"):
            st.dataframe(status["timing"], hide_index=True, use_container_width=True)
    with st.expander("Recent job runs"):
        st.dataframe(status["runs"], hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
