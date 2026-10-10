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

COLORS = {"honest_v2": "#2a78d6", "honest": "#6f5fc6", "extended": "#1baf7a", "actual": "#eb6834", "benchmark": "#52514e", "grid": "#e6e5e1"}
BAND_RGBA = {"honest_v2": ("rgba(42,120,214,0.14)", "rgba(42,120,214,0.30)"), "honest": ("rgba(111,95,198,0.14)", "rgba(111,95,198,0.30)"),
             "extended": ("rgba(27,175,122,0.14)", "rgba(27,175,122,0.30)")}
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
def revisions_panel(path, version, day, feature_set):
    return q.revisions_panel(path, day, feature_set)


@st.cache_data(ttl=60, show_spinner=False)
def performance_panel(path, version, feature_set):
    return q.performance_panel(path, feature_set)


@st.cache_data(ttl=60, show_spinner=False)
def status_panel(path, version, day):
    return q.status_panel(path, day)


@st.cache_data(ttl=60, show_spinner=False)
def shape_panel(path, version, feature_set):
    return q.shape_panel(path, feature_set)


@st.cache_data(show_spinner=False)
def fetch_log_panel(log_version):
    return q.fetch_log_panel()


@st.cache_data(show_spinner=False)
def market_panel(path, version, market_version, feature_set):
    return q.market_panel(path, feature_set)


def market_chart(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    labels = table["delivery_date"] + " " + table["product"]
    colors = ["#1baf7a" if v > 0 else "#eb6834" for v in table["pnl_per_mwh"].fillna(0)]
    fig.add_trace(go.Bar(x=labels, y=table["pnl_per_mwh"], name="P&L per MWh (VWAP entry)", marker_color=colors))
    fig.add_trace(go.Scatter(x=labels, y=table["cumulative_pnl_per_mwh"], name="Cumulative P&L per MWh", mode="lines+markers",
                             line=dict(color="#0b0b0b", width=2)))
    fig = base_layout(fig, "EUR/MWh", "Delivery day and product")
    fig.update_xaxes(type="category", dtick=None, tickangle=-45)
    return fig


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
    fig.add_trace(go.Scatter(x=x, y=by_hour["naive_day"], name="Same hour previous day (baseline)", mode="lines",
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
    fig.add_trace(go.Bar(x=scores["delivery_day"], y=scores["naive_mae"], name="Baseline daily MAE", marker_color=COLORS["benchmark"], opacity=0.5))
    fig.add_trace(go.Bar(x=scores["delivery_day"], y=scores["mae"], name=f"Model daily MAE ({feature_set})", marker_color=COLORS[feature_set]))
    fig.add_trace(go.Scatter(x=scores["delivery_day"], y=scores["rolling_naive_mae"], name="Baseline, 7-day rolling", mode="lines",
                             line=dict(color=COLORS["benchmark"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=scores["delivery_day"], y=scores["rolling_mae"], name="Model, 7-day rolling", mode="lines",
                             line=dict(color="#0b0b0b", width=2)))
    fig.update_layout(barmode="group")
    fig = base_layout(fig, "MAE, EUR/MWh", "Delivery day")
    fig.update_xaxes(type="category", dtick=None, tickangle=-45)  # delivery days are labels, not a time axis
    return fig


def hourly_history_chart(hourly: pd.DataFrame, feature_set: str) -> go.Figure:
    fig = go.Figure()
    x = hourly.index.tz_convert("Europe/Paris")
    fig.add_trace(go.Scatter(x=x, y=hourly["actual"], name="Actual price", mode="lines", line=dict(color=COLORS["actual"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=hourly["forecast"], name=f"Forecast ({feature_set})", mode="lines", line=dict(color=COLORS[feature_set], width=2)))
    fig.add_trace(go.Scatter(x=x, y=hourly["naive_day"], name="Baseline", mode="lines", line=dict(color=COLORS["benchmark"], width=1, dash="dash")))
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
        feature_set = st.radio("Feature set", ("honest_v2", "honest", "extended"), index=0,
                               help="honest_v2 is the default model (published/model/honest_v2.txt); honest is the fallback model's set; "
                                    "both use only inputs published before 12:00 Paris on the day before delivery.")
        st.caption("honest_v2: the honest set (calendar, load forecast, weather issued two days ahead, wind and solar proxies, price lags) "
                   "plus lagged nuclear generation, the neighbours' D-1 prices and a residual load. Each version names the model that "
                   "produced it; ':fallback' means the default model's inputs were missing and honest.txt was used.")
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
        if meta.get("premarket"):
            st.markdown(f"**Pre-market forecast**: the last version issued before the market window opened at 11:15 Paris "
                        f"(issued {issued:%H:%M}).")
        else:
            st.warning("No version was issued before the 11:15 Paris market window. This one was issued after the market window and is "
                       "not tradeable; it is shown for information.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Issued at (Paris)", issued.strftime("%H:%M"), help=issued.strftime("%Y-%m-%d %H:%M Paris"))
        c2.metric("Versions today", int(len(panel["versions"])))
        c3.metric("Daily mean forecast", f"{panel['by_hour']['forecast'].mean():.1f} EUR/MWh")
        c4.metric("Run kind", str(meta["kind"]))
        fitted = (f"the stored model {str(meta['model']).split(':', 1)[-1]}"
                  + (" (used because the default model's inputs were missing)" if str(meta["model"]).endswith(":fallback") else "")
                  if int(meta["train_hours"]) == 0 else f"{meta['model']} fitted live on {int(meta['train_hours']):,} hours")
        st.caption(f"Issued {issued:%Y-%m-%d %H:%M} Paris. Model: {fitted}; this feature set {gate_badge}.")
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
                columns={"naive_day": "baseline (D-1)", "actual_other": f"actual {today}"})
            st.dataframe(show.round(2), hide_index=True, use_container_width=True)
        later = panel.get("later")
        if later is not None and not later.empty and meta.get("premarket"):
            st.markdown("**Issued after the market window, not tradeable**")
            shown = later.copy()
            shown["issued_paris"] = [pd.Timestamp(t).tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M") for t in shown["issued_at_utc"]]
            shown["daily_mean"] = shown["daily_mean"].round(2)
            st.dataframe(shown[["issued_paris", "kind", "model", "daily_mean"]].rename(
                columns={"issued_paris": "Issued (Paris)", "kind": "Run kind", "model": "Model", "daily_mean": "Daily mean forecast (EUR/MWh)"}),
                hide_index=True, use_container_width=True)
        if len(panel["versions"]) > 1:
            with st.expander("Every version issued for this day"):
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

    st.markdown("**Revisions this morning**")
    revisions = revisions_panel(db, version, tomorrow, feature_set)
    if revisions["table"].empty:
        st.caption("No morning-run has stored inputs for tomorrow yet.")
    else:
        parts = []
        if "run_count" in revisions:
            parts.append(f"inputs from {revisions['run_count']} run(s), first at {revisions['first_run_paris']}, "
                         f"latest at {revisions['latest_run_paris']} Paris")
        if "version_count" in revisions:
            parts.append(f"{revisions['version_count']} {feature_set} forecast version(s), issued "
                         f"{revisions['first_issue_paris']} to {revisions['latest_issue_paris']}")
        st.caption("Daily means for tomorrow in the first and the latest morning-run of today; " + "; ".join(parts) + ".")
        shown = revisions["table"].copy()
        for column in ("first", "latest", "change"):
            fmt = "{:+,.1f}" if column == "change" else "{:,.1f}"
            shown[column] = shown[column].map(lambda v, fmt=fmt: "n/a" if v is None or pd.isna(v) else fmt.format(v))
        st.dataframe(shown.rename(columns={"quantity": "Quantity", "first": "First run", "latest": "Latest run", "change": "Change"}),
                     hide_index=True, use_container_width=True)

    # 3. Recent performance
    st.subheader(f"Forecast error, last 30 settled days ({feature_set}, pre-market version of each day)")
    perf = performance_panel(db, version, feature_set)
    if perf["scores"].empty:
        st.info("No settled forecasts yet. The settle job runs at 14:00 London time after the auction results are out.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days settled", int(len(perf["scores"])))
        c2.metric("Model MAE", f"{perf['mae']:.2f} EUR/MWh")
        c3.metric("Baseline MAE (same hour D-1)", f"{perf['naive_mae']:.2f} EUR/MWh")
        c4.metric("Days with lower MAE than the baseline", f"{100 * perf['share_below_baseline']:.0f}%")
        st.plotly_chart(performance_chart(perf["scores"], feature_set), use_container_width=True)
        if not perf["hourly"].empty:
            st.plotly_chart(hourly_history_chart(perf["hourly"], feature_set), use_container_width=True)
        with st.expander("Daily scores"):
            st.dataframe(perf["scores"][["delivery_day", "issued_at_utc", "hours", "mae", "rmse", "naive_mae", "naive_rmse", "mae_below_baseline"]].round(2),
                         hide_index=True, use_container_width=True)

    # 4. Versus the market
    st.subheader(f"Versus the market: EEX day-ahead futures traded before the auction ({feature_set})")
    mkt = market_panel(db, version, q.db_version(q.MARKET_PATH), feature_set)
    if not mkt["rows"]:
        st.info(f"No traded prices recorded. The scheduled `thermo-fr market fetch` collects them, or add a day with `thermo-fr market paste` "
                f"or `thermo-fr market add`; rows are kept in {mkt['path']}, which is git-ignored and never published.")
    else:
        s = mkt["summary"]
        if s["scored_days"]:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Days scored", s["scored_days"])
            c2.metric("Hit rate", f"{100 * s['hit_rate']:.0f}%")
            c3.metric("Mean P&L per MWh", f"{s['mean_pnl_per_mwh']:+.2f} EUR")
            c4.metric("Model MAE vs auction", f"{s['model_mae']:.2f} EUR/MWh")
            c5.metric("Market MAE vs auction", f"{s['market_mae']:.2f} EUR/MWh")
            st.plotly_chart(market_chart(mkt["table"]), use_container_width=True)
            st.caption("Direction: long when the forecast is above the traded VWAP of the window, short when below. P&L per MWh = "
                       "(auction result - entry) x direction. Each day uses the latest forecast version issued before the window opened. "
                       "Traded prices come from EEX (collected on this machine from its public market data page, pasted or typed) and "
                       f"stay here. Rows by source: {mkt.get('sources', {})}.")
            with st.expander("No-trade bands (in sample)"):
                st.dataframe(mkt["bands"], hide_index=True, use_container_width=True)
                st.caption(mkt["bands_note"])
        with st.expander("Daily table", expanded=True):
            shown = mkt["table"][["delivery_date", "product", "trade_date", "window_paris", "issued_paris", "model", "forecast", "entry_vwap",
                                  "entry_close", "auction", "direction", "pnl_per_mwh", "cumulative_pnl_per_mwh", "model_error", "market_error",
                                  "model_beats_market", "window_trades", "window_volume_mwh", "source"]] if not mkt["table"].empty else mkt["table"]
            st.dataframe(shown, hide_index=True, use_container_width=True)
        for skipped in s["skipped"]:
            st.warning(f"{skipped['delivery_date']} {skipped['product']}: {skipped['reason']}")

    # 5. Shape and battery value
    st.subheader(f"Shape and battery value ({feature_set})")
    shape = shape_panel(db, version, feature_set)
    backtest = shape["backtest"]
    if backtest:
        for label, record in backtest.items():
            st.markdown(f"**Backtest, {label} window {record['first_day']} to {record['last_day']}, {record['shape']['days']} days** "
                        f"(shape = price minus the day's mean; battery 1 MW / 2 MWh, 88% round-trip, one cycle a day, blocks chosen on the "
                        f"forecast before the gate)")
            rows = []
            for method, name in (("model", "shape model"), ("d1", "D-1 shape"), ("same_type", "same-type day shape")):
                s, b = record["shape"][method], record["battery"][method]
                rows.append({"method": name, "shape MAE (EUR/MWh)": s["shape_mae"], "spread error": s["spread_error"],
                             "cheapest-2 hit rate": s["cheapest2_hit_rate"], "dearest-2 hit rate": s["dearest2_hit_rate"],
                             "battery EUR/day": b["eur_per_day"], "share of perfect": b["share_of_perfect"], "days traded": b["days_traded"],
                             "losing days": b["losing_days"]})
            rows.append({"method": "perfect foresight", "battery EUR/day": record["battery"]["perfect_eur_per_day"], "share of perfect": 1.0})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No shape backtest yet: run `thermo-fr shape-backtest`.")
    live = shape["live"]
    if live.get("days"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Settled days", live["days"])
        c2.metric("Shape MAE, model vs D-1", f"{live['shape']['model']['shape_mae']:.2f} vs {live['shape']['d1']['shape_mae']:.2f}")
        c3.metric("Battery EUR/day, model", f"{live['battery']['model']['eur_per_day']:.1f}",
                  help=f"perfect foresight {live['battery']['perfect_eur_per_day']:.1f} EUR/day")
        c4.metric("Share of perfect, model vs D-1", f"{100 * (live['battery']['model']['share_of_perfect'] or 0):.0f}% vs "
                  f"{100 * (live['battery']['d1']['share_of_perfect'] or 0):.0f}%")
        daily = live["daily"]
        fig = go.Figure()
        for column, name, color in (("value_model", "model", COLORS[feature_set]), ("value_d1", "D-1 shape", COLORS["benchmark"]),
                                    ("value_perfect", "perfect foresight", COLORS["actual"])):
            fig.add_trace(go.Scatter(x=daily["delivery_day"], y=daily[column].cumsum(), name=name, mode="lines", line=dict(color=color)))
        st.plotly_chart(base_layout(fig, "Cumulative battery value (EUR)", "Delivery day"), use_container_width=True)
        with st.expander("Daily table"):
            st.dataframe(daily.round(2), hide_index=True, use_container_width=True)
    else:
        st.info("No settled day with a pre-market version yet for the live shape record.")

    # 6. Data status
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
    st.markdown("**Market data collection** (the scheduled `thermo-fr market fetch`; raw responses and the log stay on this machine)")
    fetch_log = fetch_log_panel(q.db_version(q.FETCH_LOG_PATH))
    if fetch_log["entries"].empty:
        st.info("No collection run logged yet. Days without a collected row can be entered with `thermo-fr market paste`.")
    else:
        if fetch_log["problems"].empty:
            st.success(f"Last collection run {fetch_log['last_run_utc']}: {fetch_log.get('stored_in_last_run', 0)} row(s) stored, no problems.")
        else:
            for _, row in fetch_log["problems"].iterrows():
                (st.error if row["status"] == "error" else st.warning)(
                    f"{row['delivery_date']} {row['product']}: {row['status']}, {row['message']}"
                    + (" Enter the day with `thermo-fr market paste` if you have the prices." if row["status"] == "error" else ""))
        with st.expander("Collection log"):
            st.dataframe(fetch_log["entries"], hide_index=True, use_container_width=True)
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
