import json
from pathlib import Path

import pandas as pd
import pytest

from thermo_fr import cli
from thermo_fr.data.csv_source import CsvSource
from thermo_fr.data.energy_charts import EnergyChartsSource
from thermo_fr.data.rte_eco2mix import RteEco2mixSource
from thermo_fr.data.sources import SOURCE_NAMES, Source, get_source
from thermo_fr.report import to_markdown
from thermo_fr.synthetic import make_synthetic

FIXTURES = Path(__file__).parent / "fixtures"


def test_factory_builds_each_keyless_source(tmp_path):
    assert isinstance(get_source("energy-charts", cache_dir=tmp_path), EnergyChartsSource)
    assert isinstance(get_source("rte", cache_dir=tmp_path), RteEco2mixSource)
    assert isinstance(get_source("csv", csv_dir=FIXTURES), CsvSource)
    assert get_source("rte", cache_dir=tmp_path).cache.directory == tmp_path / "rte"
    with pytest.raises(ValueError):
        get_source("nope")


def test_entsoe_requires_key(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ENTSOE_API_KEY"):
        get_source("entsoe")


def test_sources_satisfy_protocol(tmp_path):
    for name in ("energy-charts", "rte", "csv"):
        source = get_source(name, cache_dir=tmp_path, csv_dir=FIXTURES)
        assert isinstance(source, Source)
        assert source.name == name and source.attribution


def test_markdown_lists_sources_and_attribution():
    summary = {
        "period": "2021-01-01 to 2025-12-31", "days": 1, "heating_threshold_c": 15.0,
        "load_gradient_mw_per_c": 2400.0, "load_gradient_se": 50.0, "load_r2": 0.9,
        "price_gradient_eur_mwh_per_c": 4.0, "price_gradient_se": 0.5, "price_r2": 0.5,
        "out_of_sample": {"test_days": 365, "train_end": "2024-12-31", "mae_mw": 1000.0, "mape_pct": 2.0},
        "sources": {
            "load": {"source": "rte", "attribution": "RTE via ODRE, Licence Ouverte."},
            "price": {"source": "energy-charts", "attribution": "Energy-Charts, CC BY 4.0."},
            "temperature": {"source": "open-meteo", "attribution": "Open-Meteo, CC BY 4.0."},
        },
    }
    text = to_markdown(summary)
    assert "## Data sources" in text
    assert "- Price: energy-charts. Energy-Charts, CC BY 4.0." in text
    assert "## Data sources" not in to_markdown({**summary, "sources": None})


def test_cli_fetch_mixes_sources_and_writes_sidecars(tmp_path, monkeypatch):
    synthetic = make_synthetic("2023-08-01", "2024-11-05")

    class FakeLoad:
        name, attribution, details = "rte", "RTE attribution", {"load": {"unit": "MW"}}

        def load(self, start, end):
            return synthetic["load_mw"]

        def day_ahead_prices(self, start, end):
            raise AssertionError("not used")

    class FakePrice:
        name, attribution, details = "energy-charts", "EC attribution", {}

        def load(self, start, end):
            raise AssertionError("not used")

        def day_ahead_prices(self, start, end):
            return synthetic["price_eur_mwh"]

    built = {}

    def fake_get_source(name, **options):
        built[name] = options
        return {"rte": FakeLoad(), "energy-charts": FakePrice()}[name]

    class FakeWeather:
        def fetch(self, start, end):
            return synthetic["temperature"]

    monkeypatch.setattr(cli, "get_source", fake_get_source)
    monkeypatch.setattr("thermo_fr.data.weather.OpenMeteoSource", FakeWeather)
    out = tmp_path / "data"
    cli.main([
        "fetch", "--start", "2023-08-01", "--end", "2024-11-01", "--out", str(out),
        "--load-source", "rte", "--price-source", "energy-charts", "--csv-dir", str(tmp_path / "csv"),
    ])
    assert built["rte"]["cache_dir"] == out / "cache"
    assert built["energy-charts"]["csv_dir"] == tmp_path / "csv"
    sources = json.loads((out / "sources.json").read_text())
    assert sources["load"]["source"] == "rte" and sources["price"]["source"] == "energy-charts"
    assert sources["load"]["details"] == {"load": {"unit": "MW"}}["load"]
    assert sources["temperature"]["source"] == "open-meteo"
    quality = json.loads((out / "quality.json").read_text())
    assert quality["expected_hours"] == len(pd.date_range("2023-08-01", "2024-11-01", freq="1h", inclusive="left"))
    assert set(quality["series"]) == {"temperature", "load_mw", "price_eur_mwh"}
    hourly = pd.read_csv(out / "hourly.csv", index_col=0)
    assert list(hourly.columns) == ["temperature", "load_mw", "price_eur_mwh"]

    cli.main(["fit", "--data", str(out), "--out", str(tmp_path / "reports")])
    summary = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert summary["sources"]["price"]["attribution"] == "EC attribution"
    assert "- Load: rte. RTE attribution" in (tmp_path / "reports" / "summary.md").read_text(encoding="utf-8")


def test_cli_choices_and_defaults():
    parser_defaults = {}

    def capture(args):
        parser_defaults.update(vars(args))

    import argparse

    original = cli.cmd_fetch
    cli.cmd_fetch = capture
    try:
        cli.main(["fetch", "--start", "2025-01-01", "--end", "2025-01-08"])
    finally:
        cli.cmd_fetch = original
    assert parser_defaults["load_source"] == "rte" and parser_defaults["price_source"] == "energy-charts"
    assert parser_defaults["csv_dir"] == "data/csv"
    with pytest.raises(SystemExit):
        cli.main(["fetch", "--start", "2025-01-01", "--end", "2025-01-08", "--load-source", "bogus"])
    assert SOURCE_NAMES == ("entsoe", "energy-charts", "rte", "csv")
