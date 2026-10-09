"""Backtest report under reports/forecast/: metrics tables, charts and the worst days."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .models import BENCHMARKS, MODELS  # noqa: E402

COLUMN_LABELS = {"naive_day": "Same hour D-1", "naive_week": "Same hour D-7", "gbm": "Gradient boosting", "linear": "Linear"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SET_LABELS = {"honest": "honest", "honest_base": "honest without the wind proxy", "extended": "extended (may use late information)"}


def write_report(results: dict, out=Path("reports/forecast"), sample_week: str | None = None, sources: dict | None = None,
                 comparison: dict | None = None) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for feature_set, r in results.items():
        bt = r["backtest"]
        bt.predictions.to_csv(out / f"predictions_{feature_set}.csv", index_label="timestamp_utc")
        r["worst_days"].to_csv(out / f"worst_days_{feature_set}.csv", index_label="delivery_day")
        summary[feature_set] = {
            "months": bt.months,
            "point_in_time": bt.point_in_time,
            "metrics_strict": r["metrics_strict"],
            "metrics_all": r["metrics_all"],
            "worst_summary": r["worst_summary"],
            "worst_days": json.loads(r["worst_days"].reset_index().to_json(orient="records", date_format="iso")),
        }
        mae_by_hour_chart(r["metrics_strict"], out / f"mae_by_hour_{feature_set}.png", feature_set)
        monthly_chart(r["metrics_strict"], out / f"mae_by_month_{feature_set}.png", feature_set)
    week = sample_week or default_sample_week(results)
    sample_week_chart(results, week, out / "sample_week.png")
    summary["sample_week"] = week
    if sources is not None:
        summary["sources"] = sources
    if comparison is not None:
        summary["price_comparison"] = comparison
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(summary), encoding="utf-8")
    return out / "report.md"


def default_sample_week(results: dict) -> str:
    """A Monday in the last full winter month of the test window, else the first Monday available."""
    preds = next(iter(results.values()))["backtest"].predictions
    days = preds[preds["strict"]]["delivery_day"].drop_duplicates().sort_values()
    mondays = days[days.dt.dayofweek == 0]
    january = mondays[mondays.dt.month == 1]
    pick = january.iloc[-1] if len(january) else mondays.iloc[0]
    return pick.strftime("%Y-%m-%d")


def mae_by_hour_chart(metrics: dict, path: Path, feature_set: str) -> None:
    if not metrics["by_hour"]:
        return
    by_hour = pd.DataFrame(metrics["by_hour"]).T.sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for column in list(BENCHMARKS) + list(MODELS):
        style = "--" if column in BENCHMARKS else "-"
        ax.plot(by_hour.index, by_hour[column], style, linewidth=1.8, label=COLUMN_LABELS[column])
    ax.set_xlabel("Delivery hour (Paris time)")
    ax.set_ylabel("MAE, EUR/MWh")
    ax.set_title(f"Forecast error by hour, {feature_set} feature set, {metrics['first_day']} to {metrics['last_day']}")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def monthly_chart(metrics: dict, path: Path, feature_set: str) -> None:
    if not metrics["by_month"]:
        return
    by_month = pd.DataFrame(metrics["by_month"]).T
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(by_month))
    for column in list(BENCHMARKS) + list(MODELS):
        style = "--" if column in BENCHMARKS else "-"
        ax.plot(x, by_month[column], style, marker="o", markersize=3, linewidth=1.6, label=COLUMN_LABELS[column])
    ax.set_xticks(list(x))
    ax.set_xticklabels(by_month.index, rotation=60, fontsize=8)
    ax.set_ylabel("MAE, EUR/MWh")
    ax.set_title(f"Monthly forecast error, {feature_set} feature set (retrained each month)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_week_chart(results: dict, monday: str, path: Path) -> None:
    start = pd.Timestamp(monday)
    end = start + pd.Timedelta(days=7)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    first = True
    for feature_set, r in results.items():
        preds = r["backtest"].predictions
        week = preds[(preds["delivery_day"] >= start) & (preds["delivery_day"] < end)]
        local = week.index.tz_convert("Europe/Paris")
        if first:
            ax.plot(local, week["actual"], color="black", linewidth=2, label="Actual day-ahead price")
            ax.plot(local, week["naive_day"], color="grey", linestyle="--", linewidth=1, label="Same hour D-1")
            first = False
        ax.plot(local, week["gbm"], linewidth=1.6, label=f"Gradient boosting, {feature_set}")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Forecast against actual, week of {monday}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def forecast_day_chart(curve: pd.DataFrame, date: str, path: Path, feature_set: str) -> None:
    """Hourly forecast for one delivery day, with the actual price when it is known."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hours = curve["hour"].to_numpy()
    ax.step(hours, curve["forecast"], where="mid", linewidth=2, label=f"Forecast ({feature_set} feature set)")
    if "actual" in curve and curve["actual"].notna().any():
        ax.step(hours, curve["actual"], where="mid", color="black", linewidth=1.5, label="Actual")
    if "naive_day" in curve:
        ax.step(hours, curve["naive_day"], where="mid", color="grey", linestyle="--", linewidth=1, label="Same hour D-1")
    ax.set_xlabel("Delivery hour (Paris time)")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"French day-ahead price, {date}: forecast as of 12:00 the day before")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _row(label: str, cells: list) -> str:
    return "| " + " | ".join([label] + [str(c) for c in cells]) + " |"


def metrics_table(metrics_by_set: dict[str, dict], slice_name: str = "all") -> str:
    columns = list(BENCHMARKS) + list(MODELS)
    lines = [_row("Feature set", [f"{COLUMN_LABELS[c]} MAE / RMSE" for c in columns] + ["GBM vs D-1", "GBM vs D-7", "Linear vs D-1"]),
             _row("---", ["---"] * (len(columns) + 3))]
    for feature_set, metrics in metrics_by_set.items():
        s = metrics["slices"][slice_name]
        imp = metrics["improvement_pct"]
        cells = [f"{s[c]['mae']} / {s[c]['rmse']}" if s[c]["mae"] is not None else "n/a" for c in columns]
        cells += [f"{imp['gbm'].get('naive_day', 'n/a')}%", f"{imp['gbm'].get('naive_week', 'n/a')}%", f"{imp['linear'].get('naive_day', 'n/a')}%"]
        lines.append(_row(SET_LABELS.get(feature_set, feature_set), cells))
    return "\n".join(lines)


def slice_table(metrics: dict) -> str:
    columns = list(BENCHMARKS) + list(MODELS)
    lines = [_row("Hours", ["Count"] + [f"{COLUMN_LABELS[c]} MAE" for c in columns]), _row("---", ["---"] * (len(columns) + 1))]
    for name, s in metrics["slices"].items():
        lines.append(_row(name.replace("_", " "), [s["gbm"]["hours"]] + [s[c]["mae"] if s[c]["mae"] is not None else "n/a" for c in columns]))
    return "\n".join(lines)


def hour_table(metrics: dict) -> str:
    columns = list(BENCHMARKS) + list(MODELS)
    lines = [_row("Hour", [COLUMN_LABELS[c] for c in columns]), _row("---", ["---"] * len(columns))]
    for hour, vals in sorted(metrics["by_hour"].items(), key=lambda kv: int(kv[0])):
        lines.append(_row(f"{int(hour):02d}", [vals[c] for c in columns]))
    return "\n".join(lines)


def worst_table(rows: list[dict]) -> str:
    header = ["Day", "Weekday", "Holiday", "MAE", "D-1 bench MAE", "Mean price", "Max", "Min", "Neg. hours",
              "Jump vs D-1", "Temp C", "Anomaly C", "Wind m/s", "Load fc MW", "Renew. fc MW"]
    lines = [_row(header[0], header[1:]), _row("---", ["---"] * (len(header) - 1))]
    for r in rows:
        day = str(r["delivery_day"])[:10]
        lines.append(_row(day, [
            DOW[int(r["dow"])], "yes" if r["holiday"] else "", f"{r['mae']:.1f}", f"{r['naive_mae']:.1f}", f"{r['mean_price']:.1f}",
            f"{r['max_price']:.0f}", f"{r['min_price']:.0f}", r["negative_hours"], f"{r['price_jump_vs_prev_day']:+.1f}",
            f"{r['temp_c']:.1f}", f"{r['temp_anomaly_c']:+.1f}", f"{r['wind100_ms']:.1f}", f"{r['load_fc_mw']:,.0f}",
            f"{r['renewables_fc_mw']:,.0f}" if r["renewables_fc_mw"] == r["renewables_fc_mw"] else "n/a",
        ]))
    return "\n".join(lines)


def to_markdown(summary: dict) -> str:
    sets = [k for k in summary if k in SET_LABELS]
    strict = {k: summary[k]["metrics_strict"] for k in sets}
    full = {k: summary[k]["metrics_all"] for k in sets}
    any_strict = next(iter(strict.values()))
    any_full = next(iter(full.values()))
    parts = [
        "# French day-ahead price forecast: backtest",
        "",
        "Hourly forecasts of the French day-ahead price made with information available at 12:00 Paris time on the day "
        "before delivery. Models are retrained at the start of every month on all earlier data and forecast that month. "
        "Errors in EUR/MWh.",
        "",
        "Feature sets:",
        "",
        "- **honest**: calendar, ENTSO-E day-ahead load forecast, Open-Meteo weather as forecast two days ahead "
        "(temperature, 100 m wind, radiation) and lagged prices (D-1, D-2, D-7). Everything passes the 12:00 gate.",
        "- **extended**: honest plus the ENTSO-E day-ahead wind and solar forecasts and the residual load built from them. "
        "ENTSO-E allows these to be published until 18:00 on D-1, after the auction, so **this set may use late information**.",
        "",
        f"## Headline, strict point-in-time rows ({any_strict['first_day']} to {any_strict['last_day']}, {any_strict['rows']:,} hours)",
        "",
        "Rows whose weather forecasts come from the as-issued archive. Improvement is the MAE reduction against the benchmark. "
        "The honest set passes the look-ahead check for every row; the extended set does not, by construction.",
        "",
        metrics_table(strict),
        "",
        f"## Full test window ({any_full['first_day']} to {any_full['last_day']}, {any_full['rows']:,} hours)",
        "",
        "Includes the early 2024 weeks where the weather inputs are the historical-forecast proxy (not point in time).",
        "",
        metrics_table(full),
        "",
    ]
    for k in sets:
        m = strict[k]
        parts += [
            f"## {SET_LABELS[k].capitalize()} feature set, by slice (strict rows)",
            "",
            f"Top price hours are those at or above {m['top_price_cut_eur_mwh']} EUR/MWh (95th percentile of the test window). "
            "Peak is 08:00 to 20:00 on weekdays.",
            "",
            slice_table(m),
            "",
            f"![MAE by hour, {k}](mae_by_hour_{k}.png)",
            "",
            f"![MAE by month, {k}](mae_by_month_{k}.png)",
            "",
            f"### {k.capitalize()}: MAE by delivery hour",
            "",
            hour_table(m),
            "",
            f"### {k.capitalize()}: twenty worst forecast days (gradient boosting, all test rows)",
            "",
            worst_table(summary[k]["worst_days"]),
            "",
            "Common features of these days: " + json.dumps(summary[k]["worst_summary"], indent=None),
            "",
        ]
    parts += [f"## Sample week", "", f"![Forecast against actual, week of {summary['sample_week']}](sample_week.png)", ""]
    if "price_comparison" in summary:
        parts += ["## ENTSO-E API prices against the CSV exports", "", "```", json.dumps(summary["price_comparison"], indent=2), "```", ""]
    return "\n".join(parts)
