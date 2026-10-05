"""Command line: fetch real data, fit it, or run the offline demo."""

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

    args = parser.parse_args(argv)
    args.func(args)
