"""Command line: fetch real data, fit it, run the offline demo, or forecast day-ahead prices."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data.dataset import daily_frame
from .data.quality import format_quality, quality_report
from .data.sources import SOURCE_NAMES, get_source
from .data.weather import ATTRIBUTION as OPEN_METEO_ATTRIBUTION
from .forecast.backtest import DEFAULT_SETS
from .forecast.features import FEATURE_SETS
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
    results = run_backtest(hourly, args.test_start, args.test_end, feature_sets=tuple(args.feature_sets))
    path = write_report(results, out=Path(args.out), sample_week=args.sample_week, sources=sources, comparison=comparison)
    print(path.read_text(encoding="utf-8"))


def cmd_forecast(args) -> None:
    from .forecast.day import forecast_day, refresh_window
    from .forecast.inputs import load_inputs
    from .forecast.report import forecast_day_chart

    inputs = load_inputs(Path(args.data))
    fresh = None if args.no_refresh else refresh_window(args.date, cache_dir=Path(args.cache_dir), wind_weights=Path(args.wind_weights),
                                                        solar_weights=Path(args.solar_weights))
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
        model_file = None if args.refit or not args.model_file else Path(args.model_file)
        fallback = Path(args.fallback_model_file) if args.fallback_model_file else None
        summary = morning_run(store, args.date, kind=args.kind, inputs_path=Path(args.data), cache_dir=Path(args.cache_dir),
                              reports_dir=Path(args.reports_dir), feature_sets=tuple(args.feature_sets), model_file=model_file, fallback_model_file=fallback,
                              wind_weights=Path(args.wind_weights), solar_weights=Path(args.solar_weights))
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
        status = export_published(store, Path(args.out), days=args.days, last_run=last_run)
    finally:
        store.close()
    print(f"Published {status['forecast_versions']} forecast versions, {status['actual_days']} actual days and "
          f"{status['scored_versions']} scores to {args.out}")


def cmd_refit_model(args) -> None:
    from .forecast.jobs import setup_logging
    from .forecast.refit import fetch_and_refit, refit_from_file

    setup_logging("refit-model", logs_dir=Path(args.logs_dir))
    sets = tuple(args.feature_sets)
    if args.from_file:
        metas = refit_from_file(Path(args.data), out_dir=Path(args.out), feature_sets=sets, probabilistic=not args.no_probabilistic)
    else:
        end = args.end or first_of_this_month()
        metas = fetch_and_refit(args.start, end, out_dir=Path(args.out), cache_dir=Path(args.cache_dir), csv_dir=Path(args.csv_dir),
                                inputs_path=Path(args.data), feature_sets=sets, probabilistic=not args.no_probabilistic)
    for feature_set, meta in metas.items():
        holdout = meta.get("holdout") or {}
        print(f"{feature_set}: model fitted on {meta['train_hours']:,} hours ({meta['train_from']} to {meta['train_to']}), saved as "
              f"{meta['model_file']} under {args.out}; holdout MAE {holdout.get('mae')} against benchmark {holdout.get('naive_mae')}")


def cmd_import_published(args) -> None:
    from .forecast.publish import import_published
    from .forecast.store import Store

    store = Store(Path(args.db))
    try:
        summary = import_published(store, Path(args.src))
    finally:
        store.close()
    print(f"Imported {summary} from {args.src}")


def cmd_market_add(args) -> None:
    from .forecast.market import MarketDataError, MarketRow, add_row, parse_window

    try:
        start, end = parse_window(args.window)
        row = MarketRow(args.date, args.product, start, end, args.open, args.high, args.low, args.close, args.vwap, args.source, args.note or "",
                        trade_date=args.trade_date or "", trades=args.trades, volume_mwh=args.volume_mwh)
        frame = add_row(row, Path(args.path))
    except MarketDataError as exc:
        raise SystemExit(f"market add: {exc}")
    print(f"Stored {args.product} row for {args.date}, window {start}-{end} Paris on {row.traded_on}, VWAP {args.vwap}; {len(frame)} row(s) in "
          f"{args.path} (git-ignored, never published)")


def cmd_market_paste(args) -> None:
    import sys

    from .forecast.market import MarketDataError, add_row, parse_paste

    text = args.text if args.text else sys.stdin.read()
    try:
        row = parse_paste(text, delivery_date=args.date or "", source=args.source)
        frame = add_row(row, Path(args.path))
    except MarketDataError as exc:
        raise SystemExit(f"market paste: {exc}")
    print(f"Parsed and stored {row.product} row for {row.delivery_date}, window {row.window_start}-{row.window_end} Paris on {row.traded_on}, "
          f"O {row.open} H {row.high} L {row.low} C {row.close} VWAP {row.vwap}; {len(frame)} row(s) in {args.path} (git-ignored, never published)")


def cmd_market_fetch(args) -> None:
    import logging

    from .forecast.jobs import setup_logging
    from .forecast.market import delivery_days_to_collect, run_fetcher

    setup_logging("market-fetch")
    log = logging.getLogger("thermo_fr.market")
    if args.date:
        days = list(args.date)
    elif args.backfill:
        today = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).normalize()
        days = [d.strftime("%Y-%m-%d") for d in pd.date_range(today - pd.Timedelta(days=args.backfill), today, freq="D")]
    else:
        days = delivery_days_to_collect()
    outcomes = run_fetcher(days, plugin=Path(args.plugin), market_path=Path(args.path), log_path=Path(args.log), raw_dir=Path(args.raw_dir),
                           spacing_s=args.spacing, log=log.info)
    counts = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    print("market fetch for " + ", ".join(days) + ": " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    for o in outcomes:
        if o.status == "error":
            print(f"  {o.delivery_date} {o.product}: {o.message}")
    if counts.get("error") and not counts.get("stored"):
        print("Nothing stored. Fall back to: thermo-fr market paste --date <delivery day> --text \"...\"")


def cmd_market_evaluate(args) -> None:
    from .forecast.market import evaluate, format_table, load_market
    from .forecast.store import Store

    rows = load_market(Path(args.path))
    if rows.empty:
        print(f"No market rows in {args.path}; add one with: thermo-fr market add ...")
        return
    store = Store(Path(args.db))
    try:
        result = evaluate(store, rows, feature_set=args.feature_set, bands=tuple(args.bands))
    finally:
        store.close()
    s = result["summary"]
    print(f"Forecast ({args.feature_set}) against EEX traded prices, {s['days']} day(s) with a forecast before the window, "
          f"{s['scored_days']} with an auction result:")
    print(format_table(result["table"]))
    if s["scored_days"]:
        print(f"\nHit rate {100 * s['hit_rate']:.0f}%, total P&L {s['total_pnl_per_mwh']:+.2f} EUR/MWh, mean {s['mean_pnl_per_mwh']:+.2f} per day "
              f"(entry at the close instead of the VWAP: mean {s['mean_pnl_per_mwh_close']:+.2f}); model MAE vs auction {s['model_mae']:.2f}, "
              f"market MAE vs auction {s['market_mae']:.2f}, model error below the market's on {100 * s['share_model_beats_market']:.0f}% of days.")
        print("\nNo-trade bands (in sample):")
        print(result["bands"].to_string(index=False, na_rep=""))
    for skipped in s["skipped"]:
        print(f"skipped {skipped['delivery_date']} {skipped['product']}: {skipped['reason']}")


def _outage_api(zone: str):
    from .data.entsoe_rest import EntsoeApi

    return EntsoeApi(zone=zone)


def cmd_outage_snapshot(args) -> None:
    import logging

    from .data.outages import snapshot
    from .forecast.jobs import setup_logging

    setup_logging("outage-snapshot")
    log = logging.getLogger("thermo_fr.outages")
    now = pd.Timestamp(args.now) if args.now else None
    manifest = snapshot(_outage_api(args.zone), out_dir=Path(args.out), now=now, zone=args.zone, days_back=args.days_back,
                        days_ahead=args.days_ahead, log=log.info)
    print(f"Snapshot {manifest['retrieved_at_utc']}: {manifest['documents']} notices in {len(manifest['files'])} file(s) under {args.out} "
          f"(period {manifest['period_start']} to {manifest['period_end']}, git-ignored)")


def cmd_nuclear_availability(args) -> None:
    from .data.outages import planned_nuclear

    frame = planned_nuclear(args.date, pd.Timestamp(args.as_of), snapshot_dir=Path(args.snapshots), installed_mw=args.installed_mw)
    if frame.attrs.get("snapshot") is None:
        print(f"No outage snapshot taken by {args.as_of} under {args.snapshots}; nothing can be said as of that time.")
        return
    print(f"Planned nuclear availability for {args.date} as of {args.as_of}, from the snapshot {frame.attrs['snapshot']} "
          f"({frame.attrs['notice_count']} active nuclear notices):")
    shown = frame.copy()
    shown.index = shown.index.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M")
    print(shown.to_string())
    print(f"Daily mean unavailable {frame['unavailable_mw'].mean():.0f} MW, available {frame['available_mw'].mean():.0f} MW of {args.installed_mw:,} MW.")


def cmd_prob_backtest(args) -> None:
    """Walk-forward of the quantile and event models; writes reports/prob and the aggregates file the public app reads."""
    from .forecast import probabilistic as pb
    from .forecast.features import build_features
    from .forecast.inputs import load_inputs

    hourly = load_inputs(Path(args.data))
    hourly = hourly[hourly.index < pd.Timestamp(args.test_end, tz="UTC") - pd.Timedelta(hours=2)]
    table = build_features(hourly, args.feature_set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"prob_raw_{args.label}.parquet"
    if args.reuse and raw_path.exists():
        pred = pd.read_parquet(raw_path)
    else:
        pred = pb.walk_forward_prob(table, args.walk_start or args.test_start, args.test_end, log=print).predictions
        pred.to_parquet(raw_path)
    if args.earlier:
        earlier = pd.read_parquet(args.earlier)
        pred = pd.concat([earlier[earlier["delivery_day"] < args.test_start], pred]).sort_index()
    ready = pb.prepare(pred)
    ready = ready[(ready["delivery_day"] >= pd.Timestamp(args.strict_from or args.test_start)) & (ready["delivery_day"] < pd.Timestamp(args.test_end))]
    ready.to_parquet(out / f"prob_ready_{args.label}.parquet")
    record = pb.backtest_record(ready, args.label)
    if args.publish:
        path = Path(args.publish)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing[args.label] = record
        path.write_text(json.dumps(existing, indent=2, default=str))
    overall = record["intervals"]["all"]
    print(f"Probabilistic backtest ({args.label}) strict rows {record['first_day']} to {record['last_day']}:")
    for m, r in overall.items():
        print(f"  {m:12s} pinball mean {r['pinball_mean']:>6}  coverage {100 * r['coverage_10_90']:5.1f}%  width {r['mean_width']:>6}")
    for event, r in record["events"].items():
        print(f"  {event}: events {r.get('events')}  BSS vs climatology {r.get('bss_vs_clim')}  vs last 7 days {r.get('bss_vs_last7')}  accepted {r.get('accepted')}")


def cmd_shape_backtest(args) -> None:
    from .forecast.features import build_features
    from .forecast.inputs import load_inputs
    from .forecast.shape import evaluate_shape, walk_forward_shape

    hourly = load_inputs(Path(args.data))
    hourly = hourly[hourly.index < pd.Timestamp(args.test_end, tz="UTC") - pd.Timedelta(hours=2)]  # nothing after the window enters a fit
    table = build_features(hourly, args.feature_set)
    level = None
    if args.level:
        frame = pd.read_csv(args.level, index_col=0, parse_dates=True)
        frame.index = pd.to_datetime(frame.index, utc=True)
        level = frame["gbm"]
    bt = walk_forward_shape(table, args.test_start, args.test_end, level=level, log=print)
    pred = bt.predictions[bt.predictions["delivery_day"] >= pd.Timestamp(args.strict_from or args.test_start)]
    result = evaluate_shape(pred, strict_only=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / f"shape_predictions_{args.label}.csv", index_label="timestamp_utc")
    result["battery_daily"].to_csv(out / f"battery_daily_{args.label}.csv", index=False)
    record = {"label": args.label, "feature_set": args.feature_set, "first_day": result["first_day"], "last_day": result["last_day"],
              "months": bt.months, "level_source": args.level or "D-1 mean", "shape": result["shape"], "battery": result["battery"],
              "written_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
    if args.publish:
        path = Path(args.publish)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing[args.label] = record
        path.write_text(json.dumps(existing, indent=2))
    print(f"Shape backtest ({args.label}, {args.feature_set}) on strict rows {result['first_day']} to {result['last_day']}, "
          f"{result['shape']['days']} days:")
    for method in ("model", "d1", "same_type"):
        s, b = result["shape"][method], result["battery"][method]
        print(f"  {method:10s} shape MAE {s['shape_mae']:>6}  spread error {s['spread_error']:>6}  cheapest-2 hit {100 * s['cheapest2_hit_rate']:.0f}%  "
              f"dearest-2 hit {100 * s['dearest2_hit_rate']:.0f}%  battery {b['eur_per_day']:>7} EUR/day = {100 * (b['share_of_perfect'] or 0):.0f}% "
              f"of perfect, traded {b['days_traded']} skipped {b['days_skipped']} losing {b['losing_days']}")
    print(f"  perfect foresight {result['battery']['perfect_eur_per_day']} EUR/day over {result['battery']['days']} days")


def cmd_schedule_step(args) -> None:
    from .forecast.schedule import main as schedule_main

    argv = ["--event", args.event, "--schedule", args.schedule or "", "--input", args.input or ""] + (["--now", args.now] if args.now else [])
    schedule_main(argv)


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
    fbt.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_SETS), choices=FEATURE_SETS,
                     help="honest_wind is the honest set before the solar proxy and the calendar structure; honest_base has neither proxy")
    fbt.set_defaults(func=cmd_forecast_backtest)

    fc = sub.add_parser("forecast", help="Hourly price forecast for one delivery day, as of 12:00 the day before")
    fc.add_argument("--date", required=True, help="delivery day, YYYY-MM-DD (Paris)")
    fc.add_argument("--data", default="data/forecast/inputs.csv")
    fc.add_argument("--out", default="reports/forecast")
    fc.add_argument("--cache-dir", default="data/cache")
    fc.add_argument("--no-refresh", action="store_true", help="use the stored inputs only, no download")
    fc.add_argument("--wind-weights", default="published/model/wind_proxy.json", help="wind proxy calibration written by refit-model")
    fc.add_argument("--solar-weights", default="published/model/solar_proxy.json", help="solar proxy calibration written by refit-model")
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
    morning.add_argument("--feature-sets", nargs="+", default=["honest_v2", "extended"], choices=("honest_v2", "honest", "extended"))
    morning.add_argument("--model-file", default="published/model/honest_v2.txt",
                         help="stored LightGBM model (its own feature set is predicted with it), the same file the GitHub workflow predicts "
                              "with (default: %(default)s)")
    morning.add_argument("--refit", action="store_true", help="ignore --model-file and fit the models live on the inputs history")
    morning.add_argument("--fallback-model-file", default="published/model/honest.txt",
                         help="used instead of --model-file when an input that model needs is missing for the day (recorded in data_status)")
    morning.add_argument("--wind-weights", default="published/model/wind_proxy.json", help="wind proxy calibration written by refit-model")
    morning.add_argument("--solar-weights", default="published/model/solar_proxy.json", help="solar proxy calibration written by refit-model")
    morning.set_defaults(func=cmd_morning_run)

    stl = sub.add_parser("settle", help="Fetch actual prices and score the stored forecasts")
    stl.add_argument("--kind", default="scheduled", choices=("scheduled", "manual", "test"))
    stl.add_argument("--db", default="data/forecast.db")
    stl.add_argument("--cache-dir", default="data/cache")
    stl.add_argument("--logs-dir", default="logs")
    stl.set_defaults(func=cmd_settle)

    refit = sub.add_parser("refit-model", help="Fetch the full history, fit the honest model, save model file and metadata")
    refit.add_argument("--start", default="2021-01-01")
    refit.add_argument("--end", default=None, help="exclusive; default first day of the current month")
    refit.add_argument("--out", default="published/model")
    refit.add_argument("--data", default="data/forecast/inputs.csv", help="where the fetched history is saved (local only)")
    refit.add_argument("--from-file", action="store_true", help="refit from --data without fetching")
    refit.add_argument("--feature-sets", nargs="+", default=["honest_v2", "honest"], choices=FEATURE_SETS,
                       help="one model file per set: the default model honest_v2 and the fallback honest (default: %(default)s)")
    refit.add_argument("--no-probabilistic", action="store_true", help="skip the quantile and event models (written for the first set)")
    refit.add_argument("--cache-dir", default="data/cache")
    refit.add_argument("--csv-dir", default="data/csv")
    refit.add_argument("--logs-dir", default="logs")
    refit.set_defaults(func=cmd_refit_model)

    pub = sub.add_parser("publish", help="Export the public dataset (honest forecasts, actuals, scores) to published/")
    pub.add_argument("--out", default="published")
    pub.add_argument("--db", default="data/forecast.db")
    pub.add_argument("--days", type=int, default=90)
    pub.set_defaults(func=cmd_publish)

    imp = sub.add_parser("import-published", help="Load published/ into a store (used by the GitHub Actions workflow)")
    imp.add_argument("--from", dest="src", default="published")
    imp.add_argument("--db", default="data/forecast.db")
    imp.set_defaults(func=cmd_import_published)

    mkt = sub.add_parser("market", help="EEX traded prices (collected locally, pasted or typed) and the forecast scored against them")
    mkt_sub = mkt.add_subparsers(dest="market_command", required=True)
    madd = mkt_sub.add_parser("add", help="append one traded-price row to data/market/eex_fr_da.csv (git-ignored)")
    madd.add_argument("--date", required=True, help="delivery date, YYYY-MM-DD")
    madd.add_argument("--product", required=True, choices=("base", "peak"))
    madd.add_argument("--window", required=True, help="trading window on the trade date, Paris time, e.g. 11:15-12:00")
    for name in ("open", "high", "low", "close", "vwap"):
        madd.add_argument(f"--{name}", required=True, type=float)
    madd.add_argument("--source", required=True, help='where the prices came from, e.g. "EEX via trader"')
    madd.add_argument("--note", default="")
    madd.add_argument("--trade-date", default="", help="the day the window was traded; default the day before delivery, Friday for Sat/Sun/Mon")
    madd.add_argument("--trades", type=int, default=None, help="trades in the window, if known")
    madd.add_argument("--volume-mwh", type=float, default=None, help="volume traded in the window, if known")
    madd.add_argument("--path", default="data/market/eex_fr_da.csv")
    madd.set_defaults(func=cmd_market_add)
    mpaste = mkt_sub.add_parser("paste", help="parse a pasted line such as 'FR DA Base EEX Trades 11:15-12:00 O: 100.00 H: 102.00 L: 99.00 "
                                              "C: 101.00 VWAP: 100.50' and store it (fallback when the fetch fails)")
    mpaste.add_argument("--date", default="", help="delivery date, YYYY-MM-DD (needed unless the text carries one)")
    mpaste.add_argument("--text", default="", help="the pasted text; read from stdin when omitted")
    mpaste.add_argument("--source", default="EEX via trader")
    mpaste.add_argument("--path", default="data/market/eex_fr_da.csv")
    mpaste.set_defaults(func=cmd_market_paste)
    mfetch = mkt_sub.add_parser("fetch", help="collect the window from EEX's public market data page with the local, git-ignored collector "
                                              "module (laptop Task Scheduler only, never GitHub Actions)")
    mfetch.add_argument("--date", nargs="*", default=None, help="delivery days to collect; default tomorrow (Friday: Saturday to Monday)")
    mfetch.add_argument("--backfill", type=int, default=0, help="collect the last N days of history instead (one-off)")
    mfetch.add_argument("--plugin", default="local/eex_fetch.py", help="the local collector module")
    mfetch.add_argument("--path", default="data/market/eex_fr_da.csv")
    mfetch.add_argument("--log", default="data/market/fetch_log.csv", help="the collection log shown in the dashboard's data status panel")
    mfetch.add_argument("--raw-dir", default="data/market/raw", help="where raw responses are kept (git-ignored)")
    mfetch.add_argument("--spacing", type=float, default=10.0, help="seconds between requests")
    mfetch.set_defaults(func=cmd_market_fetch)
    meval = mkt_sub.add_parser("evaluate", help="score every stored market row with the forecast that was live before its window")
    meval.add_argument("--path", default="data/market/eex_fr_da.csv")
    meval.add_argument("--db", default="data/forecast.db")
    meval.add_argument("--feature-set", default="honest_v2")
    meval.add_argument("--bands", nargs="+", type=float, default=[0.0, 1.0, 2.0, 5.0, 10.0], help="no-trade bands in EUR/MWh (in sample)")
    meval.set_defaults(func=cmd_market_evaluate)

    snap = sub.add_parser("outage-snapshot", help="Save today's ENTSO-E unavailability notices of French units, raw and paged (laptop only)")
    snap.add_argument("--out", default="data/entsoe/outage_snapshots", help="git-ignored folder; the retrieval time is in every file name")
    snap.add_argument("--zone", default="FR")
    snap.add_argument("--days-back", type=int, default=1)
    snap.add_argument("--days-ahead", type=int, default=360, help="the window with --days-back must stay under one year")
    snap.add_argument("--now", default=None, help="retrieval time to stamp the files with (tests); default the current time")
    snap.set_defaults(func=cmd_outage_snapshot)
    avail = sub.add_parser("nuclear-availability", help="Planned nuclear availability for a delivery day as of a time, from the snapshots")
    avail.add_argument("--date", required=True, help="delivery day, YYYY-MM-DD")
    avail.add_argument("--as-of", required=True, help="UTC time; only snapshots taken by then are used, e.g. 2026-10-13T08:05:00Z")
    avail.add_argument("--snapshots", default="data/entsoe/outage_snapshots")
    avail.add_argument("--installed-mw", type=float, default=63_020.0)
    avail.set_defaults(func=cmd_nuclear_availability)

    prb = sub.add_parser("prob-backtest", help="Walk-forward of the quantile and event models with their benchmarks; aggregates to published/")
    prb.add_argument("--data", default="data/forecast/inputs.csv")
    prb.add_argument("--feature-set", default="honest_v2", choices=FEATURE_SETS)
    prb.add_argument("--walk-start", default="2023-02-01", help="first month of the walk-forward (a year before the scored window feeds the trailing benchmarks)")
    prb.add_argument("--test-start", default="2024-02-17")
    prb.add_argument("--test-end", default="2026-07-01", help="exclusive")
    prb.add_argument("--strict-from", default=None)
    prb.add_argument("--earlier", default=None, help="an earlier raw parquet whose rows feed the trailing windows (for the holdout)")
    prb.add_argument("--reuse", action="store_true", help="reuse reports/prob/prob_raw_<label>.parquet instead of refitting")
    prb.add_argument("--out", default="reports/prob")
    prb.add_argument("--publish", default="published/probabilistic_backtest.json", help="aggregates only; '' to skip")
    prb.add_argument("--label", default="selection")
    prb.set_defaults(func=cmd_prob_backtest)

    shp = sub.add_parser("shape-backtest", help="Walk-forward of the shape model (price minus the day's base) with the battery backtest")
    shp.add_argument("--data", default="data/forecast/inputs.csv")
    shp.add_argument("--feature-set", default="honest_v2", choices=FEATURE_SETS)
    shp.add_argument("--test-start", default="2024-02-01")
    shp.add_argument("--test-end", default="2026-07-01", help="exclusive")
    shp.add_argument("--strict-from", default="2024-02-17", help="first delivery day of the scored window")
    shp.add_argument("--level", default="reports/forecast/predictions_honest_v2.csv",
                     help="walk-forward level predictions whose daily mean is the base forecast for the battery decision (gbm column)")
    shp.add_argument("--out", default="reports/shape")
    shp.add_argument("--publish", default="published/shape_battery_backtest.json", help="aggregates only; '' to skip")
    shp.add_argument("--label", default="selection", help="selection or holdout")
    shp.set_defaults(func=cmd_shape_backtest)

    sched = sub.add_parser("schedule-step", help="Which step a scheduled GitHub Actions run should perform, from the Paris clock")
    sched.add_argument("--event", required=True)
    sched.add_argument("--schedule", default="")
    sched.add_argument("--input", default="")
    sched.add_argument("--now", default=None, help="UTC time, ISO format (tests); default now")
    sched.set_defaults(func=cmd_schedule_step)

    dash = sub.add_parser("dashboard", help="Open the local Streamlit dashboard")
    dash.add_argument("--db", default="data/forecast.db")
    dash.add_argument("--port", type=int, default=8501)
    dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    if args.command == "forecast-fetch" and args.end is None:
        args.end = first_of_this_month()
    args.func(args)
