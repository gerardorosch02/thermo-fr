"""Public dashboard for the French day-ahead price forecast, deployed on Streamlit Community Cloud.

It reads only the files under published/, which a GitHub Actions workflow rewrites after every forecast and
settlement run. No API is called and no key is needed. It depends on streamlit, pandas and plotly only
(requirements.txt), not on the thermo_fr package.

Three pages: Overview (the three findings, tomorrow's forecast, what changed since yesterday's forecast),
Track record (market comparison, the model's own error, negative-price calibration, shape and battery) and
Method. Several helpers mirror thermo_fr/forecast/brief.py, thermo_fr/forecast/publish.py (live_record) and
thermo_fr/dashboard/probability_text.py; tests keep the copies equal. All user-facing text is in the COPY
section below.
"""

import json
import math
import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PUBLISHED = Path(os.environ.get("THERMO_FR_PUBLISHED", Path(__file__).parent / "published"))  # override for local checks only
PARIS = "Europe/Paris"
MARKET_WINDOW_START = "11:15"  # Paris, on the day before delivery: when the EEX day-ahead future starts trading

# Minimum samples before a live statistic is shown as a result; below them the page says "Collecting, n = x of y".
MIN_MARKET_WINDOWS = 30  # live statistics derived from EEX traded prices
MIN_LIVE_DAYS = 20  # the model's own live error and the live battery value

# ============================================================================ COPY
# Plain, direct, British spelling. Numbers are never typed here: they are read from published/ at run time,
# the market record included (published/market_walkforward.json).
PENDING = "[pending]"
# QUANT_ITEM3 and QUANT_ITEM4 are filled from published/market_walkforward.json (quant_item3, quant_item4); a text that cannot be
# filled stays PENDING and is not shown.
QUANT_ITEM3 = ("The backtest's {sel_mae:.1f} EUR/MWh is an hourly error over {sel_period}, {sel_share:.0f}% of the average price of "
               "{sel_price:.1f}. On the record's days the hourly error was {rec_mae:.1f} EUR/MWh, {rec_share:.0f}% of the average price of "
               "{rec_price:.1f}, and the daily price moved {rec_move:.1f} EUR/MWh on average against {sel_move:.1f}. The record's "
               "{record_mae:.1f} scores the daily base or peak average rather than single hours. A look-ahead check found no leakage.")
QUANT_ITEM4 = ("Borderline: binomial p = {p:.2f} over {windows} windows ({p_base:.2f} for the {base_windows} base windows alone). "
               "{base_windows} of the {windows} windows are base contracts, so this is mostly a base record. The average loss is clearer: a "
               "day-block bootstrap puts the 95% interval for the P&L at {lo:.1f} to {hi:.1f} EUR/MWh per window.")
EEX_ATTRIBUTION = ("Benchmark: EEX French day-ahead base and peak futures, trades between 11:15 and 12:00 Paris on the day before "
                   "delivery, from EEX's public website data (15-minute delay). Statistics are derived by the author; no EEX data are "
                   "redistributed. This project is not affiliated with or endorsed by EEX AG.")

TITLE = "French day-ahead power forecast"
TAGLINE = ("Hourly forecast of the French day-ahead auction, issued before the 12:00 Paris gate and scored against the auction and the "
           "EEX future traded before it.")
AUTHOR_LINE = ("Built by [Gerardo Rosales](https://github.com/gerardorosch02). Code and method: "
               "[github.com/gerardorosch02/thermo-fr](https://github.com/gerardorosch02/thermo-fr). Research, not trading advice.")
NO_DATA = "No published data yet."
NO_FORECAST = "No forecast for the next delivery day yet."
COLLECTING = "Collecting, n = {n} of {needed} {unit}"

FINDINGS_HEADER = "Key findings"
F1_TITLE = "1. The shape model beat simple shapes, then lost on the holdout."
F1_BODY = ("A 1 MW / 2 MWh battery earned {model:.1f} EUR/day on the model's shape against {d1:.1f} on yesterday's ({days} selection "
           "days), but {hold_model:.1f} against {hold_d1:.1f} on the {hold_days}-day holdout{interval}.")
F1_INTERVAL = " (difference {diff:+.1f}, 95% interval {lo:+.1f} to {hi:+.1f})"
F2_TITLE = "2. Negative-price probabilities have skill."
F2_BODY = ("Brier skill {bss:+.2f} against climatology on the holdout ({events} negative hours of {hours:,}); close to calibrated "
           "where most hours sit, under-confident above 40% ({high} hours).")
F3_TITLE = "3. The level forecast does not beat the traded market."
F3_BODY = ("Over {windows} walk-forward windows it was on the right side of the EEX future {hit:.0f}% of the time and {pnl_phrase} per "
           "window; by 11:15 the market has repriced and the model makes about {share:.2f} of that move.")
DETAILS_LINK = "Details in Track record"
NOT_AVAILABLE = "The walk-forward record against the market is not available."

TOMORROW_HEADER = "Delivery day {weekday} {day}"
ISSUED_PREMARKET = "Issued {time} Paris on {date}, before the EEX window opened at 11:15."
ISSUED_LATE = "No forecast was issued before the 11:15 Paris window. This one, issued at {time}, is not tradeable."
STALE = "The latest forecast is for {day}, which has passed. Today's run may not have happened yet."
LATER_VERSIONS = "Issued after the market window, not tradeable: {times} Paris."
NEGATIVE_NONE = "No hour has a 5% or higher chance of a negative price."
PEAK_BELOW_BASE = "Peak (08:00 to 20:00) is below base: the midday dip falls inside peak hours."
NEGATIVE_SOME = "Likely negative: {hours} Paris."
RECORD_STRIP = "Live record against the EEX future: {text}."
RECORD_STRIP_READY = ("Live record, {n} windows: right on direction {hit:.0f}% of the time, {pnl:+.2f} EUR/MWh per window, error "
                      "{model:.1f} against the market's {market:.1f} EUR/MWh.")
FORECAST_HEADER = "Tomorrow's forecast with its error band"
FORECAST_CAPTION = ("Model forecast in EUR/MWh, with today's auction prices for reference. The band is the model's past error at each "
                    "hour in the backtest, not a probability forecast.")
LEGEND_FORECAST, LEGEND_TODAY = "Model forecast", "Today's auction price"
LEGEND_BAND = ("Past error, 10th to 90th percentile", "Past error, 25th to 75th percentile")
HOURLY_EXPANDER = "Hourly values"
PROBABILITY_EXPANDER = "Tomorrow's chance of a negative price, by hour"
UNCALIBRATED = "This forecast shows the uncalibrated probability."

CHANGED_HEADER = "What changed since yesterday's forecast"
CHANGED_NOTE = ("Change in each input group's contribution to the base forecast, from the model's own attributions; the steps add "
                "up to the change.")
CHANGED_MISSING = "Fills in once two consecutive pre-market forecasts have their drivers."
INPUTS_EXPANDER = "Inputs"
INPUTS_CAPTION = "Model inputs at issue time, from ENTSO-E and Open-Meteo. Change against yesterday's forecast."
NEIGHBOUR_NAMES = {"DE": "Germany", "BE": "Belgium", "ES": "Spain", "IT-North": "Italy North", "CH": "Switzerland"}
NEIGHBOURS_LINE = "Previous-day base, EUR/MWh (France minus neighbour): {values}."

TRACK_INTRO = ("The headline market record is a walk-forward reissue: each delivery day from {period} forecast again with a model "
               "fitted only on earlier data. The live records of the forecasts published here before 11:15 Paris are still collecting.")
UNIT_EEX = "EEX windows (one per day and contract, base or peak; pre-market forecasts only)"
UNIT_MAE = "days with a pre-market forecast scored against the auction"
UNIT_PROB = "days with probabilities scored (the pre-market version, or the latest where none)"
UNIT_BATTERY = "days valued (the pre-market version, or the latest where none)"
MARKET_HEADER = "Versus the market"
MARKET_TAKEAWAY = ("The model does not beat the traded price: over {windows} walk-forward windows it was right on direction {hit:.0f}% "
                   "of the time and {pnl_phrase} per window.")
MARKET_ERROR_AXIS = "Average error of the daily base/peak forecast vs the auction"
MARKET_WALKFORWARD = ("Walk-forward, not live: {windows} windows ({base} base, {peak} peak), {period}. Error {model_mae:.1f} EUR/MWh "
                      "for the model, {market_mae:.1f} for the traded price; the model was closer on {closer:.0f}% of windows. Base only: "
                      "{base_hit:.0f}% right, {base_pnl:+.2f} EUR/MWh per window.")
MARKET_CAUSE = ("By 11:15 the market has moved away from the previous day's auction price; the model makes about {share:.2f} of that "
                "move, and its long or short call matched simply repeating yesterday's auction price in {same} of {windows} windows.")
MARKET_RULE = ("Long when the forecast is above the 11:15 to 12:00 Paris VWAP of the EEX French day-ahead future, short when below, "
               "settled at the auction. Walk-forward days use a model fitted on earlier data only.")
MARKET_WHY = "Why the backtest error is lower"
MARKET_SIGNIFICANT = "Is {hit:.0f}% just bad luck?"
MARKET_LIVE = "Live"

ERROR_HEADER = "Model error against the auction"
ERROR_HOLDOUT = "On the frozen holdout ({window}) the model's hourly error against the auction was {mae:.1f} EUR/MWh."
ERROR_LIVE = "The model's own error, no other comparison: MAE {mae:.1f} EUR/MWh over {n} days."
ERROR_NOT_SCORED = "Not scored, issued after the market window: {dates}."
ERROR_CHART = "Pre-market forecast and auction price, last {days} days"
ERROR_DAILY_CAPTION = "Daily MAE against the auction, with a 7-day rolling mean."

CALIB_HEADER = "Negative-price calibration"
CALIB_TAKEAWAY = "On the holdout the probability scores a Brier skill of {bss:+.2f} against climatology and {bss7:+.2f} against the last 7 days."
CALIB_DETAIL = ("Close to the diagonal below 40%, where {low:,} of {total:,} holdout hours sit (gaps of at most {gap:.0f} points). "
                "Under-confident above 40%: in those {high} hours prices turned negative more often than forecast, for example "
                "{fc:.0f}% forecast and {obs:.0f}% observed over {n} hours.")
CALIB_DETAIL_MIXED = ("Close to the diagonal below 40%, where {low:,} of {total:,} holdout hours sit (gaps of at most {gap:.0f} points); "
                      "the {high} hours above 40% are too few to read closely.")
CALIB_NOTE = ("Benchmarks: the frequency at that hour and month over the past year, and over the last 7 days. Positive skill beats "
              "the benchmark. Calibrated monthly on {pairs:,} out-of-sample hours. Marker size grows with the number of hours in a bin.")

SHAPE_HEADER = "Shape and battery"
SHAPE_TAKEAWAY = ("A shape model beat yesterday's shape on the selection window ({sel_model:.1f} against {sel_d1:.1f} EUR/day) but not "
                  "on the holdout ({hold_model:.1f} against {hold_d1:.1f} EUR/day).")
SHAPE_NOTE = ("The shape is each hour's price minus the day's mean. The battery is 1 MW / 2 MWh at {rt} round-trip efficiency, one "
              "cycle a day on day-ahead prices only: two hours charging, then two discharging, chosen on the forecast and settled at "
              "the auction. No degradation, no ancillary services.")
SHAPE_WHY = ("On the holdout the model's average shape error was lower ({m:.2f} against {d1:.2f} EUR/MWh) yet the battery earned less: "
             "battery value depends on the cheapest and most expensive hours and the day's spread, where yesterday's shape did better (spread "
             "error {ms:.1f} against {ds:.1f} EUR/MWh).")
SHAPE_METHODS = (("model", "Shape model"), ("d1", "Yesterday's shape"), ("same_type", "Last similar day (same day type)"),
                 ("perfect", "Perfect foresight"))
SHAPE_LIVE_READY = "Live, {n} days: {value:.1f} EUR/day, {share:.0f}% of perfect foresight; yesterday's shape {d1:.1f} EUR/day."
SHAPE_LIVE_NOTE = "The live record uses the shape of the level forecast; the shape model runs in the backtest only."
SHAPE_LIVE_SERIES = "Level forecast's shape"

METHOD_MD = """
#### Information gate
- The auction closes at 12:00 Paris the day before delivery.
- Inputs are dated by publication time; a test fails if any is stamped after the gate.
- The forecast that counts is issued 10:05 to 11:00 Paris, before the EEX window at 11:15.
- Since 1 Oct 2025 the auction clears 15-minute products; the model forecasts hourly prices and is scored against the hourly average
  of the four quarter-hours.

#### Model
- LightGBM on hourly data since 2021, refitted monthly.
- Inputs: load forecast, recent French and neighbour prices, recent nuclear output, wind and solar estimated from weather forecasts
  issued two days ahead, load minus wind, solar and nuclear, calendar.
- No gas or carbon price: the price lags stand in for fuel costs.
- A second model gives the chance of a negative price, recalibrated on out-of-sample results.

#### Testing protocol
- Each change is tested under a rule written before the run.
- Variants are chosen on 17 Feb 2024 to 30 Jun 2026, retrained monthly; the winner is scored once on a frozen holdout,
  1 Jul to 10 Oct 2026.
- The EEX history is too short to select on, so selection uses the model's own backtest error.

#### Limitations
- Loses to the traded market: by 11:15 the market has repriced for the next day and the model makes less than half of that move.
- Tried in October 2026: inputs describing the expected change from the previous day, and a target relative to yesterday's
  average. They removed the systematic lag on the selection window but not the error on the holdout (26.19 against 25.81 EUR/MWh),
  so the model is unchanged.
- Weakest in regime changes, the top 5% price hours and low-price weekend and holiday middays.
- Quantiles and a spike probability are computed but not shown until they pass their tests.
- Why the backtest error is lower than the trading record's, and whether the hit rate is significant: see Track record.

#### Next steps (not yet tested)
- Fresher weather: the D-1 00Z run instead of runs issued two days ahead (later D-1 runs arrive after the 10:05 issue time).
- Nuclear availability from outage notices: snapshots saved daily since 10 Oct 2026; ENTSO-E serves only the latest revision, so the
  history has to be built forward.
- Gas price: the free history did not match the exchange index closely enough under the pre-registered check.
- Carbon (EUA price) was tested and rejected on the selection window.
"""
METHOD_SOURCES = "Data sources and attribution"
SOURCES = ["ENTSO-E Transparency Platform: prices, load forecast, generation. Cited as the source under its terms.",
           "Open-Meteo.com: weather forecasts, CC BY 4.0.",
           "RTE eCO2mix via ODRE: load history, Licence Ouverte v2.0."]
SOURCES_NOTE = "Figures are the author's own; no endorsement by the data providers is implied."
# ============================================================================ end of COPY

# ============================================================================ the walk-forward record against EEX (aggregates only)
def market_record(walkforward: dict) -> dict | None:
    """The record from published/market_walkforward.json; None when the file or a figure is missing."""
    w = walkforward or {}
    needed = ("windows", "hits", "hit_rate", "mean_pnl_per_mwh", "model_mae", "market_mae", "period")
    if not all(w.get(k) is not None for k in needed):
        return None
    first, last = pd.Timestamp(w["period"]["first_delivery_day"]), pd.Timestamp(w["period"]["last_delivery_day"])
    return {"windows": int(w["windows"]), "base_windows": w.get("base_windows"), "peak_windows": w.get("peak_windows"), "hits": int(w["hits"]),
            "hit_rate": float(w["hit_rate"]), "mean_pnl_per_mwh": float(w["mean_pnl_per_mwh"]), "model_mae": float(w["model_mae"]),
            "market_mae": float(w["market_mae"]), "period": f"{first.day} {first:%b} to {last.day} {last:%b %Y}",
            "share": (w.get("repricing_share") or {}).get("slope"), "same": (w.get("persistence") or {}).get("model_calls_equal_persistence")}


def _period(stats: dict) -> str:
    first, last = pd.Timestamp(stats["first_day"]), pd.Timestamp(stats["last_day"])
    return f"{first:%B %Y} to {last:%B %Y}" if (last - first).days > 120 else f"{first.day} {first:%B} to {last.day} {last:%B %Y}"


def quant_item3(walkforward: dict) -> str:
    """QUANT_ITEM3 filled from the record's period comparison; PENDING when a figure is missing."""
    try:
        c = walkforward["comparison"]
        sel, rec = c["selection"], c["record_days"]
        return QUANT_ITEM3.format(sel_mae=sel["hourly_mae_model"], sel_period=_period(sel), sel_price=sel["mean_price_eur_mwh"],
                                  sel_share=100 * sel["hourly_mae_model"] / sel["mean_price_eur_mwh"], rec_mae=rec["hourly_mae_model"],
                                  rec_price=rec["mean_price_eur_mwh"], rec_share=100 * rec["hourly_mae_model"] / rec["mean_price_eur_mwh"],
                                  rec_move=rec["mean_abs_daily_change_eur_mwh"], sel_move=sel["mean_abs_daily_change_eur_mwh"],
                                  record_mae=walkforward["model_mae"])
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        return PENDING


def quant_item4(walkforward: dict) -> str:
    """QUANT_ITEM4 filled from the record's tests; PENDING when a figure is missing."""
    try:
        w = walkforward
        lo, hi = w["bootstrap"]["mean_pnl_per_mwh"]["ci95"]
        return QUANT_ITEM4.format(windows=w["windows"], base_windows=w["base_only"]["windows"], p=w["binomial_p_two_sided"],
                                  p_base=w["base_only"]["binomial_p_two_sided"], lo=lo, hi=hi)
    except (KeyError, TypeError, ValueError):
        return PENDING


# ============================================================================ chart style (one template for every chart)
BLUE, ORANGE, GREY, AQUA = "#3987e5", "#eb6834", "#898781", "#1baf7a"
BAND = ("rgba(57,135,229,0.22)", "rgba(57,135,229,0.50)")
GRID = "rgba(137,135,129,0.25)"
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def style(fig: go.Figure, ytitle: str = "", xtitle: str = "", height: int = 320, title: str = "") -> go.Figure:
    """Transparent background (works on light and dark themes), plain axis titles, legend below, no mode bar."""
    legend = len(fig.data) > 1  # a single series needs no legend: the title names it
    fig.update_layout(height=height + (44 if legend else 0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=6, r=6, t=44 if title else 12, b=56 if legend else 6),
                      title=dict(text=title, x=0, xanchor="left", font=dict(size=14)), hovermode="x unified", showlegend=legend,
                      legend=dict(orientation="h", yref="container", y=0, yanchor="bottom", xanchor="left", x=0, title_text=""),
                      font=dict(size=12))
    fig.update_xaxes(title_text=xtitle, showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, config=CHART_CONFIG)


def pending(text: str) -> bool:
    return not text or text.strip() == PENDING


def live_gate(n, needed: int, unit: str) -> dict:
    """One rule for every live statistic: a result only from `needed` observations, else the collecting line."""
    n = int(n or 0)
    return {"n": n, "needed": needed, "ready": n >= needed, "text": COLLECTING.format(n=n, needed=needed, unit=unit)}


def tile(label: str, value: str, note: str | None = None, help: str | None = None, width: int = 165) -> None:
    """A stat tile; the note sits under the value as plain grey text, without a delta arrow."""
    try:
        st.metric(label, value, note, delta_color="off", delta_arrow="off", help=help, width=width)
    except TypeError:  # older Streamlit without delta_arrow or width
        st.metric(label, value, note, delta_color="off", help=help)


def tiles():
    """A row of tiles that wraps (two per row on a phone) instead of stacking column by column."""
    try:
        return st.container(horizontal=True, gap="medium")
    except TypeError:
        return st.container()


PAGES: dict = {}  # filled by main(): the st.Page objects, for in-page links (the top navigation collapses on a phone)


def link(key: str, label: str) -> None:
    if key in PAGES:
        st.page_link(PAGES[key], label=label, icon=":material/arrow_forward:")


def day_text(day) -> str:
    stamp = pd.Timestamp(day)
    return f"{stamp.day} {stamp:%b %Y}"


def eur(value, digits: int = 1) -> str:
    return "n/a" if _nan(value) else f"{value:.{digits}f}"


# ============================================================================ version picking (mirrors thermo_fr/forecast/publish.py)
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


def paris_clock(stamp) -> str:
    return pd.Timestamp(stamp).tz_convert(PARIS).strftime("%Y-%m-%d %H:%M")


def late_only_text(record: dict) -> str:
    items = record.get("late_only_days") or []
    if not items:
        return ""
    return "Late only, not scored: " + "; ".join(f"{d['delivery_day']} ({d['versions']} version(s), earliest issued {paris_clock(d['first_issued_utc'])} Paris)"
                                                   for d in items) + "."


# ============================================================================ display of event probabilities (mirrors thermo_fr/dashboard/probability_text.py)
SUMMARY_THRESHOLD = 0.05  # hours above this count in the one-line summary


def format_probability(p) -> str:
    """The probability itself: one decimal below 10%, whole percents from there, and '<0.1%' under 0.05%."""
    if p is None or pd.isna(p):
        return "n/a"
    p = float(p)
    if p < 0.0005:
        return "<0.1%"
    return f"{100 * p:.1f}%" if p < 0.10 else f"{100 * p:.0f}%"


def negative_summary(hourly: pd.DataFrame, column: str = "p_negative") -> dict:
    """One line about tomorrow: the highest hour with its probability, and the hours above 5% when there are any."""
    if hourly is None or hourly.empty or column not in hourly or hourly[column].isna().all():
        return {"text": "", "max_p": None, "max_hour": None, "hours_above_5pct": 0, "label": ""}
    values = hourly[[c for c in ("hour", column) if c in hourly]].dropna(subset=[column])
    top = values.loc[values[column].idxmax()]
    max_p, max_hour = float(top[column]), int(top["hour"]) if "hour" in values else None
    above = int((values[column] > SUMMARY_THRESHOLD).sum())
    label = format_probability(max_p)
    where = f"Highest hour {max_hour:02d}:00 Paris" if max_hour is not None else "Highest hour"
    text = f"{where}, {label}" + (f"; {above} hour(s) above 5%" if above else "") + "."
    return {"text": text, "max_p": max_p, "max_hour": max_hour, "hours_above_5pct": above, "label": label}


def probability_axis_top(max_p) -> float:
    """Top of the probability axis: the larger of 10% and 1.2 times the highest probability, never above 100%."""
    value = 0.0 if max_p is None or pd.isna(max_p) else float(max_p)
    return min(1.0, max(0.10, 1.2 * value))


def show_half_line(axis_top: float) -> bool:
    """The light reference line at 50% is drawn only when the axis reaches it."""
    return axis_top >= 0.5


# ============================================================================ loading
def headline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of `frame` (forecasts or scores, with delivery_day and issued_at_utc) that belong to each day's headline version."""
    heads = headline_versions(frame)
    if heads.empty:
        return frame.iloc[0:0]
    return frame.merge(heads, on=["delivery_day", "issued_at_utc"], how="inner")


def _json(name: str) -> dict:
    path = PUBLISHED / name
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (ValueError, OSError):
        return {}


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

    out = {
        "status": _json("status.json"),
        "tomorrow": read("tomorrow.csv"),
        "forecasts": read("forecasts.csv"),
        "actuals": read("actuals.csv"),
        "scores": read("scores.csv"),
        "band": read("error_band.csv"),
        "shape": _json("shape_battery.json"),
        "shape_backtest": _json("shape_battery_backtest.json"),
        "prob_tomorrow": read("tomorrow_probabilistic.csv"),
        "drivers": read("drivers.csv"),
        "prob_backtest": _json("probabilistic_backtest.json"),
        "market": _json("market.json"),
        "walkforward": _json("market_walkforward.json"),
    }
    # the dataset may carry more than one feature set (the current model's and the fallback's); the app shows the current one.
    out["scores_all"] = out["scores"]
    for key in ("forecasts", "scores"):
        out[key] = default_set_rows(out[key], out["status"])
    return out


def default_set_rows(frame: pd.DataFrame, status: dict) -> pd.DataFrame:
    wanted = status.get("feature_set")
    if frame.empty or not wanted or "feature_set" not in frame:
        return frame
    kept = frame[frame["feature_set"] == wanted]
    return kept.reset_index(drop=True) if len(kept) else frame


def published_version() -> float:
    path = PUBLISHED / "status.json"
    return path.stat().st_mtime if path.exists() else 0.0


def data_or_stop() -> dict:
    data = load_published(published_version())
    if not data["status"]:
        st.error(NO_DATA)
        st.stop()
    return data


# ============================================================================ tomorrow's forecast
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
        for lo, hi, color, name in (("band_p10", "band_p90", BAND[0], LEGEND_BAND[0]), ("band_p25", "band_p75", BAND[1], LEGEND_BAND[1])):
            fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([by_hour[hi], by_hour[lo][::-1]]), fill="toself",
                                     fillcolor=color, line=dict(width=0), name=name, hoverinfo="skip"))
    if by_hour["actual_today"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=by_hour["actual_today"], name=LEGEND_TODAY, mode="lines",
                                 line=dict(color=GREY, width=2, dash="dot"), hovertemplate="%{y:.1f}"))
    fig.add_trace(go.Scatter(x=x, y=by_hour["forecast"], name=LEGEND_FORECAST, mode="lines",
                             line=dict(color=BLUE, width=2.5), hovertemplate="%{y:.1f}"))
    fig = style(fig, "EUR/MWh", "Hour (Paris)", height=340)
    fig.update_xaxes(dtick=3, range=[-0.3, 23.3])
    return fig


def probability_chart(hourly: pd.DataFrame) -> go.Figure:
    summary = negative_summary(hourly)
    shown = hourly["p_negative"].map(format_probability)
    fig = go.Figure(go.Bar(x=hourly["hour"], y=hourly["p_negative"], name="Probability of a negative price", marker_color=BLUE,
                           customdata=shown, hovertemplate="%{x}:00 Paris: %{customdata}<extra></extra>"))
    top = probability_axis_top(summary["max_p"])
    fig = style(fig, "Probability", "Hour (Paris)", height=240)
    fig.update_yaxes(range=[0, top], tickformat=".0%")
    fig.update_xaxes(dtick=3)
    if show_half_line(top):
        fig.add_hline(y=0.5, line=dict(color=GREY, width=1, dash="dot"))
    return fig


# ============================================================================ the brief (mirrors thermo_fr/forecast/brief.py)
PEAK_HOURS = range(8, 20)  # 08:00 to 20:00 Paris
NEGATIVE_THRESHOLD = 0.05
MIN_STEP = 0.5
AVERAGE_GROUP, REFIT_STEP = "model average", "model refit"
FRENCH_GROUP = "recent French prices (model following the market)"
LEGACY_GROUPS = {"recent French prices": FRENCH_GROUP}  # the group's name in drivers.csv files exported before 2026-10-14
GROUP_ORDER = [FRENCH_GROUP, "neighbour prices", "wind", "solar", "load and temperature", "nuclear", "calendar", AVERAGE_GROUP]
FUNDAMENTALS = [("French base, previous day", "EUR/MWh", "price_lag1_mean", 1.0),
                ("Load forecast", "GW", "load_fc_mw", 1000.0), ("Temperature", "C", "temp_c", 1.0), ("Wind estimate", "GW", "wind_proxy_mw", 1000.0),
                ("Solar estimate", "GW", "solar_proxy_mw", 1000.0), ("Latest nuclear output", "GW", "nuclear_d1_early_mw", 1000.0),
                ("Load minus wind, solar and nuclear", "GW", "residual_v2_mw", 1000.0)]
NEIGHBOUR_ROWS = [("DE", "de_lu"), ("BE", "be"), ("ES", "es"), ("IT-North", "it_north"), ("CH", "ch")]
GROUP_LABELS = {FRENCH_GROUP: "Recent French prices", "neighbour prices": "Neighbour prices", "wind": "Wind", "solar": "Solar",
                "load and temperature": "Load and temperature", "nuclear": "Nuclear", "calendar": "Calendar", REFIT_STEP: "Model refit"}


def _nan(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def base_of(drivers) -> float:
    if drivers is None or drivers.empty:
        return float("nan")
    return float(pd.to_numeric(drivers["contribution"], errors="coerce").sum())


def grouped(drivers) -> pd.Series:
    """Contribution per group from the published `group` column; every group of GROUP_ORDER present."""
    out = pd.Series(0.0, index=GROUP_ORDER, dtype=float)
    if drivers is None or drivers.empty:
        return out
    sums = pd.to_numeric(drivers["contribution"], errors="coerce").groupby(drivers["group"].replace(LEGACY_GROUPS)).sum()
    for group, total in sums.items():
        out[group] = float(total)
    return out


def model_forecast(curve) -> dict:
    if curve is None or curve.empty:
        return {"base": float("nan"), "peak": float("nan"), "offpeak": float("nan"), "hours": 0}
    forecast = pd.to_numeric(curve["forecast"], errors="coerce")
    peak = curve["hour"].astype(int).isin(list(PEAK_HOURS)).to_numpy()
    return {"base": float(forecast.mean()), "peak": float(forecast[peak].mean()) if peak.any() else float("nan"),
            "offpeak": float(forecast[~peak].mean()) if (~peak).any() else float("nan"), "hours": int(len(curve))}


def negative_hours(by_hour, threshold: float = NEGATIVE_THRESHOLD) -> list:
    """Hours at or above the threshold, from `p_negative` (the published, calibrated value)."""
    if by_hour is None or by_hour.empty or "p_negative" not in by_hour:
        return []
    values = pd.to_numeric(by_hour["p_negative"], errors="coerce")
    picked = by_hour.loc[values >= threshold, ["hour"]].assign(p=values[values >= threshold].to_numpy())
    return [{"hour": int(r["hour"]), "p": float(r["p"])} for _, r in picked.sort_values("hour").iterrows()]


def waterfall(previous, current) -> dict:
    if previous is None or previous.empty or current is None or current.empty:
        return {"start": base_of(previous), "end": base_of(current), "change": float("nan"), "steps": [], "complete": False}
    before, after = grouped(previous), grouped(current)
    groups = [g for g in GROUP_ORDER if g != AVERAGE_GROUP] + [g for g in after.index.union(before.index) if g not in GROUP_ORDER]
    steps = [{"group": g, "change": float(after.get(g, 0.0) - before.get(g, 0.0))} for g in groups]
    refit = float(after.get(AVERAGE_GROUP, 0.0) - before.get(AVERAGE_GROUP, 0.0))
    if abs(refit) > 1e-9:
        steps.append({"group": REFIT_STEP, "change": refit})
    start, end = base_of(previous), base_of(current)
    return {"start": start, "end": end, "change": end - start, "steps": steps, "complete": True}


def feature_value(drivers, feature: str) -> float:
    if drivers is None or drivers.empty:
        return float("nan")
    rows = drivers.loc[drivers["feature"] == feature, "value"]
    return float(pd.to_numeric(rows, errors="coerce").iloc[0]) if len(rows) else float("nan")


def fundamentals_table(current, previous) -> pd.DataFrame:
    rows = []
    for label, unit, feature, divisor in FUNDAMENTALS:
        now, then = feature_value(current, feature) / divisor, feature_value(previous, feature) / divisor
        rows.append({"quantity": label, "unit": unit, "tomorrow": now, "previous": then, "change": now - then, "fr_spread": float("nan")})
    for label, zone in NEIGHBOUR_ROWS:
        now, then = feature_value(current, f"price_{zone}_lag1"), feature_value(previous, f"price_{zone}_lag1")
        rows.append({"quantity": f"{label} D-1 base", "unit": "EUR/MWh", "tomorrow": now, "previous": then, "change": now - then,
                     "fr_spread": feature_value(current, f"spread_{zone}_lag1")})
    return pd.DataFrame(rows, columns=["quantity", "unit", "tomorrow", "previous", "change", "fr_spread"])


def _delta(current, previous, feature, divisor=1.0) -> float:
    return (feature_value(current, feature) - feature_value(previous, feature)) / divisor


def step_phrase(group, current, previous, day_names=None) -> str:
    def moved(name, feature, unit, divisor=1.0, digits=1):
        change = _delta(current, previous, feature, divisor)
        if _nan(change):
            return name
        if round(abs(change), digits) == 0:
            return f"{name} unchanged"
        return f"{name} {abs(change):.{digits}f} {unit} {'higher' if change > 0 else 'lower'}"

    if group == "wind":
        return moved("wind", "wind_proxy_mw", "GW", 1000.0)
    if group == "solar":
        return moved("solar", "solar_proxy_mw", "GW", 1000.0)
    if group == "nuclear":
        return moved("nuclear output", "nuclear_d1_early_mw", "GW", 1000.0)
    if group == "load and temperature":
        load, temp = _delta(current, previous, "load_fc_mw", 1000.0), _delta(current, previous, "temp_c")
        if not _nan(load) and (_nan(temp) or abs(load) >= 0.3):
            return moved("load", "load_fc_mw", "GW", 1000.0)
        if not _nan(temp):
            return moved("temperature", "temp_c", "C")
        return "load and temperature"
    if group == FRENCH_GROUP:
        return moved("French base price", "price_lag1_mean", "EUR/MWh", 1.0, 0)
    if group == "neighbour prices":
        return moved("German base price", "price_de_lu_lag1", "EUR/MWh", 1.0, 0)
    if group == "calendar":
        return f"calendar, {day_names[0]} to {day_names[1]}" if day_names else "calendar"
    return group


def summary_line(result, current, previous, day_names=None, min_step=MIN_STEP) -> str:
    if not result.get("complete"):
        return ""
    change = result["change"]
    head = ("Model forecast unchanged" if abs(change) < min_step
            else f"Model forecast {'up' if change > 0 else 'down'} {abs(change):.1f} EUR/MWh")
    top = sorted((s for s in result["steps"] if abs(s["change"]) >= min_step), key=lambda s: -abs(s["change"]))[:2]
    if not top:
        return head + "."
    if top[0]["group"] == FRENCH_GROUP:
        head += ", mostly the model following the market"
    return f"{head}: " + ", ".join(f"{s['group'].split(' (')[0]} {s['change']:+.1f}" for s in top) + "."


def drivers_of(drivers: pd.DataFrame, day: str, issued: str, feature_set: str) -> pd.DataFrame:
    if drivers is None or drivers.empty:
        return pd.DataFrame(columns=["feature", "group", "value", "contribution"])
    rows = drivers[(drivers["delivery_day"] == day) & (drivers["issued_at_utc"] == issued) & (drivers["feature_set"] == feature_set)]
    return rows.reset_index(drop=True)


def brief_from_published(data: dict) -> dict:
    """The brief's numbers from the published files: headline version of the next delivery day and of the day before."""
    status = data.get("status") or {}
    feature_set = status.get("feature_set", "honest_v2")
    versions = tomorrow_versions(data)
    out = {"feature_set": feature_set, "versions": versions}
    if not versions:
        return out
    day, headline = versions["day"], versions["headline"]
    previous_day = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    issued = str(headline["issued_at_utc"].iloc[0])
    out.update(day=day, previous_day=previous_day, issued=issued, premarket=versions["premarket"], model=versions.get("model", "gbm"),
               level=model_forecast(headline.groupby("hour").agg(forecast=("forecast", "mean")).reset_index()))
    prob = data.get("prob_tomorrow", pd.DataFrame())
    hourly = prob.groupby("hour").agg(p_negative=("p_negative", "mean")).reset_index() if not prob.empty and "p_negative" in prob else pd.DataFrame()
    out["hourly_probability"] = hourly
    out["negative_hours"] = negative_hours(hourly)
    heads = headline_versions(data["forecasts"]) if not data["forecasts"].empty else pd.DataFrame(columns=["delivery_day", "issued_at_utc"])
    previous_issued = heads.loc[heads["delivery_day"] == previous_day, "issued_at_utc"]
    current = drivers_of(data.get("drivers"), day, issued, feature_set)
    previous = drivers_of(data.get("drivers"), previous_day, str(previous_issued.iloc[0]), feature_set) if len(previous_issued) else pd.DataFrame()
    out.update(current_drivers=current, previous_drivers=previous, fundamentals=fundamentals_table(current, previous), waterfall=waterfall(previous, current))
    out["day_names"] = (pd.Timestamp(previous_day).strftime("%a"), pd.Timestamp(day).strftime("%a"))
    out["summary"] = summary_line(out["waterfall"], current, previous, out["day_names"])
    market = status.get("market") or {}
    out["market"] = live_gate(market.get("scored_windows", market.get("scored_days", 0)), MIN_MARKET_WINDOWS, "windows")
    return out


GROUP_ALSO = {FRENCH_GROUP: "; hourly lags and older days also count", "neighbour prices": "; other zones and spreads also count"}


def driver_label(group: str, current, previous, day_names) -> str:
    name = f"<b>{GROUP_LABELS.get(group, group.capitalize())}</b>"
    if group in (REFIT_STEP, "calendar"):
        return name
    return f"{name}<br><span style='font-size:11px'>{step_phrase(group, current, previous, day_names)}{GROUP_ALSO.get(group, '')}</span>"


def drivers_chart(result: dict, current, previous, day_names) -> go.Figure:
    """Horizontal bars: each group's step in EUR/MWh, largest on top, signed values, each label written above its bar so it keeps
    the full width on a phone; start and end level in the title."""
    steps = sorted((s for s in result["steps"] if round(s["change"], 1) != 0), key=lambda s: abs(s["change"]))
    values = [s["change"] for s in steps]
    rows = list(range(len(steps)))
    fig = go.Figure(go.Bar(y=rows, x=values, orientation="h", width=0.32, marker_color=[ORANGE if v > 0 else BLUE for v in values],
                           text=[f"{v:+.1f}" for v in values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x:+.1f} EUR/MWh<extra></extra>"))
    for row, step in zip(rows, steps):
        fig.add_annotation(text=driver_label(step["group"], current, previous, day_names), xref="paper", x=0, xanchor="left", y=row + 0.18,
                           yanchor="bottom", showarrow=False, align="left", font=dict(size=12))
    title = f"Base forecast {result['start']:.1f} to {result['end']:.1f} EUR/MWh ({result['change']:+.1f})"
    fig = style(fig, "", "EUR/MWh", height=max(220, 74 * len(steps) + 80), title=title)
    span = max([abs(v) for v in values] + [1.0])
    fig.update_xaxes(range=[-1.25 * span, 1.25 * span], showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=GREY, zerolinewidth=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, range=[-0.4, len(steps) - 0.05])
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def fundamental_tile(row) -> None:
    unit = "°C" if row["unit"] == "C" else row["unit"]
    digits = 0 if row["unit"] == "EUR/MWh" else 1
    value = "n/a" if _nan(row["tomorrow"]) else f"{row['tomorrow']:.{digits}f} {unit}"
    note = "no earlier value" if _nan(row["change"]) else f"{row['change']:+.{digits}f} {unit}"
    tile(row["quantity"], value, note)


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def pnl_phrase(pnl: float) -> str:
    return f"lost {abs(pnl):.2f} EUR/MWh" if pnl < 0 else f"made {pnl:.2f} EUR/MWh"


def live_market_text(market: dict) -> str:
    """The live record against EEX: the collecting line below MIN_MARKET_WINDOWS, the pooled aggregates from there."""
    gate = live_gate(market.get("scored_windows", market.get("scored_days", 0)), MIN_MARKET_WINDOWS, UNIT_EEX)
    if gate["ready"] and market.get("hit_rate") is not None:
        return RECORD_STRIP_READY.format(n=gate["n"], hit=100 * market["hit_rate"], pnl=market["mean_pnl_per_mwh"],
                                         model=market["model_mae"], market=market["market_mae"])
    return RECORD_STRIP.format(text=lower_first(gate["text"]))


# ============================================================================ pages
def calibration_split(backtest: dict, label: str = "holdout") -> dict | None:
    """The calibrated reliability bins of a window split at 40%: hours and largest gap below, hours and the largest bin above."""
    rel = pd.DataFrame((((backtest or {}).get(label) or {}).get("recalibration") or {}).get("calibrated", {}).get("reliability", []))
    if rel.empty:
        return None
    rel = rel.dropna(subset=["mean_forecast", "observed_frequency"])
    low, high = rel[rel["mean_forecast"] < 0.4], rel[rel["mean_forecast"] >= 0.4]
    out = {"total": int(rel["n"].sum()), "low": int(low["n"].sum()), "high": int(high["n"].sum()),
           "gap": float(100 * (low["observed_frequency"] - low["mean_forecast"]).abs().max()) if len(low) else float("nan"),
           "under": bool(len(high) and (high["observed_frequency"] > high["mean_forecast"]).all())}
    if len(high):
        top = high.loc[high["n"].idxmax()]
        out.update(fc=100 * float(top["mean_forecast"]), obs=100 * float(top["observed_frequency"]), n=int(top["n"]))
    return out


def findings(data: dict) -> None:
    st.subheader(FINDINGS_HEADER)
    sel, hold = (data["shape_backtest"] or {}).get("selection"), (data["shape_backtest"] or {}).get("holdout")
    with st.container(border=True):
        st.markdown(f"**{F1_TITLE}**")
        if sel and hold:
            b, h = sel["battery"], hold["battery"]
            paired = ((hold.get("paired") or {}).get("battery_value") or {}).get("d1") or {}
            interval = (F1_INTERVAL.format(diff=paired["mean_difference"], lo=paired["ci95_low"], hi=paired["ci95_high"])
                        if {"mean_difference", "ci95_low", "ci95_high"} <= set(paired) else "")
            st.markdown(F1_BODY.format(days=b["days"], model=b["model"]["eur_per_day"], d1=b["d1"]["eur_per_day"], hold_days=h["days"],
                                       hold_model=h["model"]["eur_per_day"], hold_d1=h["d1"]["eur_per_day"], interval=interval))
    with st.container(border=True):
        st.markdown(f"**{F2_TITLE}**")
        r = ((data["prob_backtest"] or {}).get("holdout") or {}).get("recalibration")
        split = calibration_split(data["prob_backtest"])
        if r and split:
            st.markdown(F2_BODY.format(events=r["events"], hours=r["hours"], bss=r["calibrated"]["bss_vs_clim"], high=split["high"]))
    with st.container(border=True):
        st.markdown(f"**{F3_TITLE}**")
        m = market_record(data["walkforward"])
        if m is None or m["share"] is None:
            st.markdown(NOT_AVAILABLE)
        else:
            st.markdown(F3_BODY.format(windows=m["windows"], hit=100 * m["hit_rate"], pnl_phrase=pnl_phrase(m["mean_pnl_per_mwh"]),
                                       share=m["share"]))
    link("track", DETAILS_LINK)


def overview() -> None:
    data = data_or_stop()
    st.title(TITLE)
    st.markdown(TAGLINE)
    findings(data)

    brief = brief_from_published(data)
    if "day" not in brief:
        st.info(NO_FORECAST)
        return
    day, level = brief["day"], brief["level"]
    stamp = pd.Timestamp(day)
    st.subheader(TOMORROW_HEADER.format(weekday=f"{stamp:%A}", day=f"{stamp.day} {stamp:%B %Y}"))
    issued = pd.Timestamp(brief["issued"]).tz_convert(PARIS)
    if brief["premarket"]:
        st.caption(ISSUED_PREMARKET.format(time=f"{issued:%H:%M}", date=day_text(issued.tz_localize(None))))
    else:
        st.warning(ISSUED_LATE.format(time=f"{issued:%H:%M}"))
    if stamp < pd.Timestamp.now(tz=PARIS).normalize().tz_localize(None):
        st.warning(STALE.format(day=day_text(day)))
    by_hour = tomorrow_table(data)
    low, high = by_hour.loc[by_hour["forecast"].idxmin()], by_hour.loc[by_hour["forecast"].idxmax()]
    negative = brief["negative_hours"]
    with tiles():
        tile("Base", eur(level["base"]), "EUR/MWh", help="Mean of the 24 hourly forecasts.")
        tile("Peak", eur(level["peak"]), "EUR/MWh", help="Mean of 08:00 to 20:00 Paris.")
        tile("Cheapest hour", f"{low['forecast']:.1f}", f"{int(low['hour']):02d}:00, EUR/MWh")
        tile("Most expensive hour", f"{high['forecast']:.1f}", f"{int(high['hour']):02d}:00, EUR/MWh")
        tile("Hours with a 5%+ chance of a negative price", str(len(negative)), help="Calibrated probability of 5% or more.", width=340)
    if not _nan(level["peak"]) and not _nan(level["base"]) and level["peak"] < level["base"]:
        st.caption(PEAK_BELOW_BASE)
    if negative:
        st.markdown(NEGATIVE_SOME.format(hours=", ".join(f"{h['hour']:02d}:00 ({format_probability(h['p'])})" for h in negative)))
    else:
        st.caption(NEGATIVE_NONE)
    st.caption(live_market_text(data["status"].get("market") or data.get("market") or {}))

    st.subheader(FORECAST_HEADER)
    previous = (stamp - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    show(tomorrow_chart(by_hour, day, previous))
    st.caption(FORECAST_CAPTION)
    with st.expander(HOURLY_EXPANDER):
        st.dataframe(pd.DataFrame({"Hour": [f"{int(h):02d}:00" for h in by_hour["hour"]], LEGEND_FORECAST: by_hour["forecast"].round(1),
                                   LEGEND_TODAY: by_hour["actual_today"].round(1)}), hide_index=True)
    hourly = brief.get("hourly_probability", pd.DataFrame())
    if not hourly.empty:
        with st.expander(PROBABILITY_EXPANDER):
            st.caption(negative_summary(hourly)["text"])
            prob = data.get("prob_tomorrow", pd.DataFrame())
            if "calibrated" in prob and len(prob) and not bool(prob["calibrated"].min()):
                st.caption(UNCALIBRATED)
            show(probability_chart(hourly))
    later = brief["versions"].get("later", pd.DataFrame())
    if brief["premarket"] and later is not None and not later.empty:
        st.caption(LATER_VERSIONS.format(times=", ".join(t[-5:] for t in later["issued_paris"])))

    st.subheader(CHANGED_HEADER)
    result = brief["waterfall"]
    if not result["complete"]:
        st.info(CHANGED_MISSING)
        return
    st.markdown(f"**{brief['summary']}**")
    show(drivers_chart(result, brief["current_drivers"], brief["previous_drivers"], brief["day_names"]))
    st.caption(CHANGED_NOTE)
    with st.expander(INPUTS_EXPANDER):
        table = brief["fundamentals"]
        main = table[table["quantity"].isin([label for label, *_ in FUNDAMENTALS])].reset_index(drop=True)
        with tiles():
            for _, row in main.iterrows():
                fundamental_tile(row)
        rows = table.set_index("quantity")
        values = ", ".join(f"{NEIGHBOUR_NAMES.get(label, label)} {eur(rows.loc[f'{label} D-1 base', 'tomorrow'], 0)} "
                           f"({'n/a' if _nan(rows.loc[f'{label} D-1 base', 'fr_spread']) else format(rows.loc[f'{label} D-1 base', 'fr_spread'], '+.0f')})"
                           for label, _ in NEIGHBOUR_ROWS)
        st.caption(NEIGHBOURS_LINE.format(values=values))
        st.caption(INPUTS_CAPTION)


def market_section(data: dict) -> None:
    st.subheader(MARKET_HEADER)
    m = market_record(data["walkforward"])
    w = data["walkforward"] or {}
    if m is None:
        st.info(NOT_AVAILABLE)
    else:
        st.markdown("**" + MARKET_TAKEAWAY.format(windows=m["windows"], hit=100 * m["hit_rate"], pnl_phrase=pnl_phrase(m["mean_pnl_per_mwh"])) + "**")
        fig = go.Figure(go.Bar(y=["Traded price", "Model forecast"], x=[m["market_mae"], m["model_mae"]], orientation="h",
                               marker_color=[GREY, BLUE], text=[f"{m['market_mae']:.1f}", f"{m['model_mae']:.1f}"], textposition="outside",
                               cliponaxis=False, hovertemplate="%{x:.1f} EUR/MWh<extra></extra>"))
        fig = style(fig, "", "EUR/MWh", height=200, title=MARKET_ERROR_AXIS)
        fig.update_layout(showlegend=False, hovermode="closest", bargap=0.4)
        fig.update_xaxes(range=[0, 1.2 * max(m["model_mae"], m["market_mae"])], showgrid=True, gridcolor=GRID)
        show(fig)
        with tiles():
            tile("Right on direction", f"{100 * m['hit_rate']:.0f}%", f"{m['hits']} of {m['windows']} windows")
            tile("P&L per window", f"{m['mean_pnl_per_mwh']:+.2f}", "EUR/MWh")
        base = w.get("base_only") or {}
        if w.get("share_model_beats_market") is not None and base.get("hit_rate") is not None:
            st.caption(MARKET_WALKFORWARD.format(windows=m["windows"], base=m["base_windows"], peak=m["peak_windows"], period=m["period"],
                                                 model_mae=m["model_mae"], market_mae=m["market_mae"], closer=100 * w["share_model_beats_market"],
                                                 base_hit=100 * base["hit_rate"], base_pnl=base["mean_pnl_per_mwh"]))
        if m["share"] is not None and m["same"] is not None:
            st.markdown(MARKET_CAUSE.format(share=m["share"], same=m["same"], windows=m["windows"]))
        for title, text in ((MARKET_WHY, quant_item3(w)), (MARKET_SIGNIFICANT.format(hit=100 * m["hit_rate"]), quant_item4(w))):
            if not pending(text):
                with st.expander(title):
                    st.markdown(text)
    market = data["status"].get("market") or data.get("market") or {}
    gate = live_gate(market.get("scored_windows", market.get("scored_days", 0)), MIN_MARKET_WINDOWS, UNIT_EEX)
    st.markdown(f"**{MARKET_LIVE}:** " + (live_market_text(market) if gate["ready"] else lower_first(gate["text"]) + "."))
    st.caption(MARKET_RULE + " " + EEX_ATTRIBUTION)


def error_section(data: dict) -> None:
    st.subheader(ERROR_HEADER)
    status = data["status"]
    previous_set = status.get("fallback_model_file", "").split("/")[-1].replace(".txt", "")
    record, tables = live_record(data.get("scores_all", data["scores"]), status.get("feature_set", ""), previous_set)
    head = record["default"]
    gate = live_gate(head.get("days_in_window", 0), MIN_LIVE_DAYS, UNIT_MAE)
    st.markdown(f"**{MARKET_LIVE}:** " + (ERROR_LIVE.format(mae=head["model_mae"], n=gate["n"]) if gate["ready"] else lower_first(gate["text"]) + "."))
    holdout = (status.get("model") or {}).get("frozen_holdout") or {}
    if holdout.get("mae") is not None:
        first, last = [pd.Timestamp(d.strip()) for d in holdout["window"].split(" to ")]
        st.markdown(ERROR_HOLDOUT.format(window=f"{day_text(first)} to {day_text(last)}", mae=holdout["mae"]))
    late = [d["delivery_day"] for d in head.get("late_only_days") or []]
    if late:
        st.caption(ERROR_NOT_SCORED.format(dates=", ".join(day_text(d) for d in late)))
    if gate["ready"]:
        scores = tables["default"].tail(30).copy()
        scores["rolling"] = scores["mae"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=scores["delivery_day"], y=scores["mae"], name="Daily MAE", marker_color=BLUE, hovertemplate="%{y:.1f}"))
        fig.add_trace(go.Scatter(x=scores["delivery_day"], y=scores["rolling"], name="7-day mean", mode="lines", line=dict(color=ORANGE, width=2),
                                 hovertemplate="%{y:.1f}"))
        show(style(fig, "EUR/MWh", "Delivery day", height=280))
        st.caption(ERROR_DAILY_CAPTION)
    forecasts, actuals = data["forecasts"], data["actuals"]
    if forecasts.empty or actuals.empty:
        return
    heads = headline_versions(forecasts)
    heads = heads[heads["premarket"].astype(bool)][["delivery_day", "issued_at_utc"]]  # days issued only after the window are not drawn
    joined = forecasts.merge(heads, on=["delivery_day", "issued_at_utc"]).merge(actuals[["timestamp_utc", "price"]], on="timestamp_utc")
    if joined.empty:
        return
    joined["t"] = pd.to_datetime(joined["timestamp_utc"], utc=True).dt.tz_convert(PARIS)
    days = sorted(joined["delivery_day"].unique())[-30:]
    joined = joined[joined["delivery_day"].isin(days)].sort_values("t")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=joined["t"], y=joined["price"], name="Auction price", mode="lines", line=dict(color=GREY, width=1.5),
                             hovertemplate="%{y:.1f}"))
    fig.add_trace(go.Scatter(x=joined["t"], y=joined["forecast"], name="Pre-market forecast", mode="lines", line=dict(color=BLUE, width=2),
                             hovertemplate="%{y:.1f}"))
    fig = style(fig, "EUR/MWh", "Delivery time (Paris)", height=300, title=ERROR_CHART.format(days=len(days)))
    fig.update_xaxes(tickformat="%d %b", dtick=86400000)
    show(fig)


def calibration_section(data: dict) -> None:
    st.subheader(CALIB_HEADER)
    backtest = data["prob_backtest"] or {}
    r = (backtest.get("holdout") or {}).get("recalibration")
    if r:
        st.markdown("**" + CALIB_TAKEAWAY.format(bss=r["calibrated"]["bss_vs_clim"], bss7=r["calibrated"]["bss_vs_last7"]) + "**")
    split = calibration_split(backtest)
    if split:
        st.markdown((CALIB_DETAIL if split["under"] else CALIB_DETAIL_MIXED).format(**split))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, width=1, dash="dot"), name="Perfect",
                             hoverinfo="skip"))
    for label, color, name in (("selection", BLUE, "Selection window"), ("holdout", ORANGE, "Holdout")):
        rel = pd.DataFrame((((backtest.get(label) or {}).get("recalibration") or {}).get("calibrated") or {}).get("reliability", []))
        if rel.empty:
            continue
        rel = rel.dropna(subset=["mean_forecast", "observed_frequency"])
        sizes = [6 + 3.2 * math.log10(max(int(n), 1)) for n in rel["n"]]  # sparse bins read as small points
        fig.add_trace(go.Scatter(x=rel["mean_forecast"], y=rel["observed_frequency"], mode="lines+markers", name=name,
                                 line=dict(color=color, width=1.5), marker=dict(size=sizes, line=dict(width=0)), customdata=rel["n"],
                                 hovertemplate="forecast %{x:.0%}, observed %{y:.0%}, %{customdata:,} hours<extra></extra>"))
    fig = style(fig, "Observed frequency", "Forecast probability", height=340, title="Reliability after recalibration")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_layout(hovermode="closest")
    show(fig)
    with tiles():
        for label, name in (("selection", "selection"), ("holdout", "holdout")):
            rr = (backtest.get(label) or {}).get("recalibration")
            if rr:
                tile(f"Brier skill, {name}", f"{rr['calibrated']['bss_vs_clim']:+.2f}", f"{rr['raw']['bss_vs_clim']:+.2f} before recalibration",
                     help=f"{day_text(rr['first_day'])} to {day_text(rr['last_day'])}, {rr['events']:,} negative hours out of {rr['hours']:,}")
    calibration = ((data["status"].get("probabilistic") or {}).get("calibration")) or {}
    st.caption(CALIB_NOTE.format(pairs=calibration.get("pairs_hours", 0)))
    live = ((data["status"].get("probabilistic") or {}).get("live")) or {}
    gate = live_gate(live.get("days", 0), MIN_LIVE_DAYS, UNIT_PROB)
    negative = live.get("negative") or {}
    text = (f"{gate['n']} days, {negative.get('events', 0)} negative hours, Brier {negative.get('brier', float('nan')):.4f}."
            if gate["ready"] else lower_first(gate["text"]) + ".")
    st.markdown(f"**{MARKET_LIVE}:** {text}")


def shape_section(data: dict) -> None:
    st.subheader(SHAPE_HEADER)
    backtest = data["shape_backtest"] or {}
    sel, hold = backtest.get("selection"), backtest.get("holdout")
    if sel and hold:
        st.markdown("**" + SHAPE_TAKEAWAY.format(sel_model=sel["battery"]["model"]["eur_per_day"], sel_d1=sel["battery"]["d1"]["eur_per_day"],
                                                 hold_model=hold["battery"]["model"]["eur_per_day"],
                                                 hold_d1=hold["battery"]["d1"]["eur_per_day"]) + "**")
    fig = go.Figure()
    for label, color, name in (("selection", BLUE, "Selection window"), ("holdout", ORANGE, "Holdout")):
        record = backtest.get(label)
        if not record:
            continue
        b = record["battery"]
        values = [b["perfect_eur_per_day"] if key == "perfect" else b[key]["eur_per_day"] for key, _ in SHAPE_METHODS]
        fig.add_trace(go.Bar(y=[n for _, n in SHAPE_METHODS], x=values, orientation="h", name=f"{name}, {b['days']} days", marker_color=color,
                             text=[f"{v:.1f}" for v in values], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x:.1f} EUR/day<extra></extra>"))
    if fig.data:
        fig = style(fig, "", "Battery value, EUR/day", height=380, title="Battery value by shape forecast")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, 1.18 * max(max(t.x) for t in fig.data)])
        fig.update_layout(hovermode="closest", barmode="group", bargap=0.25, bargroupgap=0.05)
        show(fig)
        rows = [{"Shape MAE (EUR/MWh)": name, "Selection": sel["shape"][key]["shape_mae"] if sel else None,
                 "Holdout": hold["shape"][key]["shape_mae"] if hold else None} for key, name in SHAPE_METHODS[:3]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"Selection": st.column_config.NumberColumn(format="%.2f"),
                                                                        "Holdout": st.column_config.NumberColumn(format="%.2f")})
    if hold:
        hs = hold["shape"]
        if (hs["model"]["shape_mae"] < hs["d1"]["shape_mae"] and hold["battery"]["model"]["eur_per_day"] < hold["battery"]["d1"]["eur_per_day"]
                and hs["model"]["spread_error"] > hs["d1"]["spread_error"]):
            st.markdown(SHAPE_WHY.format(m=hs["model"]["shape_mae"], d1=hs["d1"]["shape_mae"], ms=hs["model"]["spread_error"],
                                         ds=hs["d1"]["spread_error"]))
    found = re.search(r"(\d+)% round-trip", ((data["shape"] or {}).get("definitions") or {}).get("battery", ""))
    st.caption(SHAPE_NOTE.format(rt=f"{found.group(1)}%" if found else "the stated"))
    live = (data["shape"] or {}).get("live") or {}
    gate = live_gate(live.get("days", 0), MIN_LIVE_DAYS, UNIT_BATTERY)
    if gate["ready"]:
        b = live["battery"]
        text = SHAPE_LIVE_READY.format(n=gate["n"], value=b["model"]["eur_per_day"], share=100 * (b["model"]["share_of_perfect"] or 0),
                                       d1=b["d1"]["eur_per_day"])
    else:
        text = lower_first(gate["text"]) + "."
    st.markdown(f"**{MARKET_LIVE}:** {text} {SHAPE_LIVE_NOTE}")
    daily = pd.DataFrame(live.get("daily") or [])
    if not daily.empty:
        with st.expander("Live battery value per day" + ("" if gate["ready"] else f" (n = {gate['n']}, collecting)")):
            fig = go.Figure()
            x = pd.to_datetime(daily["delivery_day"]).dt.strftime("%d %b")
            for column, name, color in (("value_model", SHAPE_LIVE_SERIES, BLUE), ("value_d1", "Yesterday's shape", GREY),
                                        ("value_perfect", "Perfect foresight", AQUA)):
                fig.add_trace(go.Bar(x=x, y=daily[column], name=name, marker_color=color, hovertemplate="%{y:.1f} EUR"))
            fig = style(fig, "EUR per day", "Delivery day", height=300)
            fig.update_xaxes(type="category")
            fig.update_layout(barmode="group")
            show(fig)


def track_record() -> None:
    data = data_or_stop()
    st.title("Track record")
    m = market_record(data["walkforward"])
    if m is not None:
        st.markdown(TRACK_INTRO.format(period=m["period"]))
    market_section(data)
    error_section(data)
    calibration_section(data)
    shape_section(data)


def method() -> None:
    data = data_or_stop()
    st.title("Method")
    st.markdown(METHOD_MD)
    st.markdown(f"#### {METHOD_SOURCES}")
    for line in SOURCES + [EEX_ATTRIBUTION]:
        st.markdown(f"- {line}")
    st.caption(SOURCES_NOTE)
    written = pd.Timestamp(data["status"]["written_at_utc"]).tz_convert(PARIS) if data["status"].get("written_at_utc") else None
    if written is not None:
        st.caption(f"Data updated {day_text(written.tz_localize(None))}, {written:%H:%M} Paris.")


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon=":material/bolt:", layout="centered")
    PAGES.update(overview=st.Page(overview, title="Overview", icon=":material/bolt:", default=True),
                 track=st.Page(track_record, title="Track record", icon=":material/insights:"),
                 method=st.Page(method, title="Method", icon=":material/menu_book:"))
    current = st.navigation(list(PAGES.values()), position="top")
    current.run()
    st.divider()
    with tiles():
        for key, page in PAGES.items():
            if page.title != current.title:
                st.page_link(page, label=page.title, icon=page.icon)
    st.caption(AUTHOR_LINE)


if __name__ == "__main__":
    main()
