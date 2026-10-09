"""Command line: fetch real data, fit it, run the offline demo, or forecast day-ahead prices."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data.dataset import daily_frame
from .data.quality import format_quality, quality_report
from .data.sources import SOURCE_NAMES, get_source
from .data.weather import ATTRIBUTION as OPEN_METEO_ATTRIBUTION
from .report import run


def fetch_series(args) -> tuple[pd.DataFrame, dict]:
    """Download load, prices and temperatures with the sources named on the command line."""
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.out) / "cache"
    options = {"cache_dir": cache_dir, "csv_dir": Path(args.csv_dir)}
    load_source = get_source(args.load_source, **options)
    price_source = load_source if args.price_source == args.load_source else get_source(args.price_source, **options)

    print(f"Fetching load from {load_source.name} ...")
    load = load_source.load(args.start, args.end)
    print(f"Fetching day-ahead prices from {price_source.name} ...")
    price = price_source.day_ahead_prices(args.start, args.end)

    from .data.weather import OpenMeteoSource

    # Open-Meteo end dates are inclusive; the load and price sources' are exclusive.
    last_day = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print("Fetching temperatures from Open-Meteo ...")
    temperature = OpenMeteoSource().fetch(args.start, last_day)

    hourly = pd.concat([temperature, load, price], axis=1).sort_index()
    sources = {
        "start": args.start,
        "end": args.end,
        "fetched_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "load": describe(load_source, "load"),
        "price": describe(price_source, "price"),
        "temperature": {"source": "open-meteo", "attribution": OPEN_METEO_ATTRIBUTION},
    }
    return hourly, sources


def describe(source, series: str) -> dict:
    """What to record about a source in sources.json."""
    return {
        "source": source.name,
        "attribution": source.attribution,
        "details": getattr(source, "details", {}).get(series, {}),
    }


def cmd_fetch(args) -> None:
    hourly, sources = fetch_series(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out / "hourly.csv", index_label="timestamp_utc")
    (out / "sources.json").write_text(json.dumps(sources, indent=2))

    quality = quality_report(hourly, args.start, args.end)
    (out / "quality.json").write_text(json.dumps(quality, indent=2))
    print(f"Saved {len(hourly):,} hourly rows to {out / 'hourly.csv'} (sources in {out / 'sources.json'})")
    print(format_quality(quality))


def cmd_fit(args) -> None:
    data = Path(args.data)
    hourly = pd.read_csv(data / "hourly.csv", index_col=0)
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    sources = json.loads((data / "sources.json").read_text()) if (data / "sources.json").exists() else None
    summary = run(daily_frame(hourly), args.out, sources=sources)
    print((Path(args.out) / "summary.md").read_text(encoding="utf-8"))
    return summary


def cmd_demo(args) -> None:
    from .synthetic import make_synthetic

    run(daily_frame(make_synthetic()), args.out)
    print("Synthetic data, true values: threshold 15.0 °C, 2,400 MW/°C, 4.00 EUR/MWh per °C\n")
    print((Path(args.out) / "summary.md").read_text(encoding="utf-8"))


def first_of_this_month() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-01")


def cmd_forecast_fetch(args) -> None:
    from .forecast.inputs import fetch_inputs, save_inputs

    hourly, sources, comparison = fetch_inputs(args.start, args.end, cache_dir=Path(args.cache_dir), csv_dir=Path(args.csv_dir))
    path = save_inputs(hourly, sources, comparison, out=Path(args.out))
    print(f"Saved {len(hourly):,} hourly rows to {path}")
    for column, cov in sources["coverage"].items():
        print(f"  {column:22s} {cov.get('hours', 0):>7,} hours  {cov.get('first', '')} to {cov.get('last', '')}")


def cmd_forecast_backtest(args) -> None:
    from .forecast.backtest import run_backtest
    from .forecast.inputs import load_inputs
    from .forecast.report import write_report

    data = Path(args.data)
    hourly = load_inputs(data)
    sources = json.loads((data.parent / "sources.json").read_text()) if (data.parent / "sources.json").exists() else None
    comparison_path = data.parent / "price_comparison.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else None
    results = run_backtest(hourly, args.test_start, args.test_end)
    path = write_report(results, out=Path(args.out), sample_week=args.sample_week, sources=sources, comparison=comparison)
    print(path.read_text(encoding="utf-8"))


def cmd_forecast(args) -> None:
    from .forecast.day import forecast_day, refresh_window
    from .forecast.inputs import load_inputs
    from .forecast.report import forecast_day_chart

    inputs = load_inputs(Path(args.data))
    fresh = None if args.no_refresh else refresh_window(args.date, cache_dir=Path(args.cache_dir))
    curve = forecast_day(args.date, inputs, fresh)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / f"day_{args.date}.csv")
    forecast_day_chart(curve, args.date, out / f"day_{args.date}.png", "honest")
    shown = curve[["hour", "forecast", "naive_day", "actual", "load_fc_mw", "temp_c"]]
    print(f"Day-ahead price forecast for {args.date} (EUR/MWh), information as of 12:00 Paris the day before:")
    print(shown.to_string(index=False, na_rep=""))
    print(f"Saved {out / f'day_{args.date}.csv'} and {out / f'day_{args.date}.png'}")


def cmd_timing_probe(args) -> None:
    from .forecast.probe import run_probe

    rows = run_probe(log_path=Path(args.log), cache_dir=Path(args.cache_dir) / "entsoe", delivery_day=args.date)
    print(rows.to_string(index=False))
    print(f"Appended to {args.log}")


def cmd_morning_run(args) -> None:
    from .forecast.jobs import morning_run, setup_logging
    from .forecast.store import Store

    setup_logging("morning-run", logs_dir=Path(args.logs_dir))
    store = Store(Path(args.db))
    try:
        summary = morning_run(store, args.date, kind=args.kind, inputs_path=Path(args.data), cache_dir=Path(args.cache_dir),
                              reports_dir=Path(args.reports_dir), feature_sets=tuple(args.feature_sets))
        if args.extend_history:
            from .forecast.publish import extend_history

            added = extend_history(Path(args.extend_history), store, summary["delivery_day"])
            print(f"history extended by {added} rows")
    finally:
        store.close()
    print(f"morning-run {summary['status']}: delivery day {summary['delivery_day']}, "
          f"forecasts {', '.join(summary['forecasts']) or 'none'}" + (f"; errors: {summary['errors']}" if summary["errors"] else ""))


def cmd_settle(args) -> None:
    from .forecast.jobs import settle, setup_logging
    from .forecast.store import Store

    setup_logging("settle", logs_dir=Path(args.logs_dir))
    store = Store(Path(args.db))
    try:
        summary = settle(store, cache_dir=Path(args.cache_dir), kind=args.kind)
        if args.extend_history:
            from .forecast.publish import extend_history

            for day in summary["actual_days"]:
                extend_history(Path(args.extend_history), store, day)
    finally:
        store.close()
    print(f"settle {summary['status']}: actual prices for {summary['actual_days'] or 'no new days'}, "
          f"{summary['scored']} forecast versions scored" + (f"; errors: {summary['errors']}" if summary["errors"] else ""))


def cmd_publish(args) -> None:
    from .forecast.publish import export_published
    from .forecast.store import Store

    store = Store(Path(args.db))
    try:
        last = store.runs(limit=1)
        last_run = last.iloc[0].to_dict() if len(last) else None
        status = export_published(store, Path(args.out), inputs_path=Path(args.data), days=args.days, last_run=last_run)
    finally:
        store.close()
    print(f"Published {status['forecast_versions']} forecast versions, {status['actual_days']} actual days, "
          f"{status['scored_versions']} scores and {status['history_rows']:,} history rows to {args.out}")


def cmd_import_published(args) -> None:
    from .forecast.publish import import_published
    from .forecast.store import Store

    store = Store(Path(args.db))
    try:
        summary = import_published(store, Path(args.src), inputs_path=Path(args.data))
    finally:
        store.close()
    print(f"Imported {summary} from {args.src}")


def cmd_dashboard(args) -> None:
    import subprocess
    import sys

    app = Path(__file__).parent / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port), "--server.headless", "true",
               "--browser.gatherUsageStats", "false", "--", "--db", args.db]
    print("Starting the dashboard at http://localhost:%d (Ctrl+C to stop)" % args.port)
    subprocess.run(command, check=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="thermo-fr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download load, prices and temperatures")
    fetch.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    fetch.add_argument("--end", required=True, help="YYYY-MM-DD, exclusive")
    fetch.add_argument("--out", default="data")
    fetch.add_argument("--load-source", choices=SOURCE_NAMES, default="rte", help="default: rte (no key needed)")
    fetch.add_argument(
        "--price-source", choices=SOURCE_NAMES, default="energy-charts", help="default: energy-charts (no key needed)"
    )
    fetch.add_argument("--csv-dir", default="data/csv", help="directory of ENTSO-E CSV exports for --*-source csv")
    fetch.add_argument("--cache-dir", default=None, help="raw response cache, default <out>/cache")
    fetch.set_defaults(func=cmd_fetch)

    fit = sub.add_parser("fit", help="Fit the models on downloaded data")
    fit.add_argument("--data", default="data")
    fit.add_argument("--out", default="reports")
    fit.set_defaults(func=cmd_fit)

    demo = sub.add_parser("demo", help="Run end to end on synthetic data, no keys needed")
    demo.add_argument("--out", default="reports/demo")
    demo.set_defaults(func=cmd_demo)

    ffetch = sub.add_parser("forecast-fetch", help="Download the day-ahead forecast inputs (ENTSO-E key needed)")
    ffetch.add_argument("--start", default="2021-01-01", help="YYYY-MM-DD, inclusive")
    ffetch.add_argument("--end", default=None, help="YYYY-MM-DD, exclusive; default: first day of the current month")
    ffetch.add_argument("--out", default="data/forecast")
    ffetch.add_argument("--cache-dir", default="data/cache")
    ffetch.add_argument("--csv-dir", default="data/csv", help="ENTSO-E price exports to compare the API prices with")
    ffetch.set_defaults(func=cmd_forecast_fetch)

    fbt = sub.add_parser("forecast-backtest", help="Walk-forward backtest of the day-ahead price forecast")
    fbt.add_argument("--data", default="data/forecast/inputs.csv")
    fbt.add_argument("--out", default="reports/forecast")
    fbt.add_argument("--test-start", default="2024-01-01", help="first month forecast out of sample")
    fbt.add_argument("--test-end", default="2026-01-01", help="exclusive")
    fbt.add_argument("--sample-week", default=None, help="Monday (YYYY-MM-DD) of the week to chart")
    fbt.set_defaults(func=cmd_forecast_backtest)

    fc = sub.add_parser("forecast", help="Hourly price forecast for one delivery day, as of 12:00 the day before")
    fc.add_argument("--date", required=True, help="delivery day, YYYY-MM-DD (Paris)")
    fc.add_argument("--data", default="data/forecast/inputs.csv")
    fc.add_argument("--out", default="reports/forecast")
    fc.add_argument("--cache-dir", default="data/cache")
    fc.add_argument("--no-refresh", action="store_true", help="use the stored inputs only, no download")
    fc.set_defaults(func=cmd_forecast)

    probe = sub.add_parser("timing-probe", help="Log which ENTSO-E day-ahead items already exist for tomorrow")
    probe.add_argument("--date", default=None, help="delivery day to probe, default tomorrow (Paris)")
    probe.add_argument("--log", default="data/timing_probe.csv")
    probe.add_argument("--cache-dir", default="data/cache")
    probe.set_defaults(func=cmd_timing_probe)

    morning = sub.add_parser("morning-run", help="Fetch tomorrow's inputs, log their timing, store the forecasts")
    morning.add_argument("--date", default=None, help="delivery day, default tomorrow (Paris)")
    morning.add_argument("--kind", default="scheduled", choices=("scheduled", "manual", "test"))
    morning.add_argument("--db", default="data/forecast.db")
    morning.add_argument("--data", default="data/forecast/inputs.csv")
    morning.add_argument("--cache-dir", default="data/cache")
    morning.add_argument("--reports-dir", default="reports/forecast", help="backtest predictions for the error band")
    morning.add_argument("--logs-dir", default="logs")
    morning.add_argument("--feature-sets", nargs="+", default=["honest", "extended"], choices=("honest", "extended"))
    morning.add_argument("--extend-history", default=None, metavar="INPUTS_CSV",
                         help="append the day's stored inputs to this history table after the run (used by the workflow)")
    morning.set_defaults(func=cmd_morning_run)

    stl = sub.add_parser("settle", help="Fetch actual prices and score the stored forecasts")
    stl.add_argument("--kind", default="scheduled", choices=("scheduled", "manual", "test"))
    stl.add_argument("--db", default="data/forecast.db")
    stl.add_argument("--cache-dir", default="data/cache")
    stl.add_argument("--logs-dir", default="logs")
    stl.add_argument("--extend-history", default=None, metavar="INPUTS_CSV", help="fold the settled prices into this history table")
    stl.set_defaults(func=cmd_settle)

    pub = sub.add_parser("publish", help="Export the public dataset (honest forecasts, actuals, scores) to published/")
    pub.add_argument("--out", default="published")
    pub.add_argument("--db", default="data/forecast.db")
    pub.add_argument("--data", default="data/forecast/inputs.csv")
    pub.add_argument("--days", type=int, default=90)
    pub.set_defaults(func=cmd_publish)

    imp = sub.add_parser("import-published", help="Load published/ into a store (used by the GitHub Actions workflow)")
    imp.add_argument("--from", dest="src", default="published")
    imp.add_argument("--db", default="data/forecast.db")
    imp.add_argument("--data", default="data/forecast/inputs.csv")
    imp.set_defaults(func=cmd_import_published)

    dash = sub.add_parser("dashboard", help="Open the local Streamlit dashboard")
    dash.add_argument("--db", default="data/forecast.db")
    dash.add_argument("--port", type=int, default=8501)
    dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    if args.command == "forecast-fetch" and args.end is None:
        args.end = first_of_this_month()
    args.func(args)
