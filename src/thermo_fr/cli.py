"""Command line: fetch real data, fit it, or run the offline demo."""

import argparse
from pathlib import Path

import pandas as pd

from .data.dataset import daily_frame
from .report import run


def cmd_fetch(args) -> None:
    from .data.entsoe_client import EntsoeSource
    from .data.weather import OpenMeteoSource

    entsoe = EntsoeSource()
    load = entsoe.load(args.start, args.end)
    price = entsoe.day_ahead_prices(args.start, args.end)
    # Open-Meteo end dates are inclusive; ENTSO-E's are exclusive.
    last_day = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    temperature = OpenMeteoSource().fetch(args.start, last_day)

    hourly = pd.concat([temperature, load, price], axis=1).sort_index()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out / "hourly.csv", index_label="timestamp_utc")
    print(f"Saved {len(hourly):,} hourly rows to {out / 'hourly.csv'}")


def cmd_fit(args) -> None:
    hourly = pd.read_csv(Path(args.data) / "hourly.csv", index_col=0)
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    summary = run(daily_frame(hourly), args.out)
    print((Path(args.out) / "summary.md").read_text())
    return summary


def cmd_demo(args) -> None:
    from .synthetic import make_synthetic

    run(daily_frame(make_synthetic()), args.out)
    print("Synthetic data, true values: threshold 15.0 °C, 2,400 MW/°C, 4.00 EUR/MWh per °C\n")
    print((Path(args.out) / "summary.md").read_text())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="thermo-fr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download load, prices and temperatures")
    fetch.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    fetch.add_argument("--end", required=True, help="YYYY-MM-DD, exclusive")
    fetch.add_argument("--out", default="data")
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
