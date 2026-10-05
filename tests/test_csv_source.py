from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_fr.data.csv_source import CsvSource, classify, localize_cet_cest, read_entsoe_csv
from thermo_fr.data.dataset import daily_frame
from thermo_fr.data.sources import UnsupportedSeriesError

FIXTURES = Path(__file__).parent / "fixtures"


def test_localize_resolves_autumn_duplicate_by_file_order():
    naive = pd.Series(["27.10.2024 01:00", "27.10.2024 02:00", "27.10.2024 02:00", "27.10.2024 03:00"])
    utc = localize_cet_cest(naive).tz_convert("UTC")
    assert [t.strftime("%H:%M") for t in utc] == ["23:00", "00:00", "01:00", "02:00"]


def test_localize_marks_spring_nonexistent_hour_as_nat():
    naive = pd.Series(["31.03.2024 01:00", "31.03.2024 02:00", "31.03.2024 03:00"])
    out = localize_cet_cest(naive)
    assert out.isna().tolist() == [False, True, False]
    assert out[2].tz_convert("UTC").hour == 1


def test_hourly_load_file_old_header_autumn_dst():
    kind, series = read_entsoe_csv(FIXTURES / "load_hourly_autumn_dst.csv")
    assert kind == "load"
    assert str(series.index.tz) == "UTC"
    assert len(series) == 73 and not series.index.duplicated().any()
    assert series.index[0] == pd.Timestamp("2024-10-25 22:00", tz="UTC")
    # The two "02:00" rows on 27 October land on 00:00 and 01:00 UTC, in that order.
    assert series[pd.Timestamp("2024-10-27 00:00", tz="UTC")] == 40020.0
    assert series[pd.Timestamp("2024-10-27 01:00", tz="UTC")] == 40120.0
    assert np.isnan(series[pd.Timestamp("2024-10-28 05:00", tz="UTC")])  # "n/e" stays missing


def test_quarter_hour_load_file_new_header_spring_dst(tmp_path):
    kind, raw = read_entsoe_csv(FIXTURES / "load_quarter_hour_spring_dst.csv")
    assert kind == "load" and len(raw) == 71 * 4
    hourly = CsvSource(FIXTURES).load("2024-03-30", "2024-04-01")
    assert hourly.name == "load_mw" and len(hourly) == 48
    assert hourly[pd.Timestamp("2024-03-31 05:00", tz="UTC")] == pytest.approx(50500 + 22.5)
    assert hourly.notna().all()


def test_price_file_units_and_missing_marker():
    kind, series = read_entsoe_csv(FIXTURES / "price_hourly_autumn_dst.csv")
    assert kind == "price"
    assert series[pd.Timestamp("2024-10-27 00:00", tz="UTC")] == pytest.approx(61.0)
    assert np.isnan(series[pd.Timestamp("2024-10-26 12:00", tz="UTC")])


def test_csv_source_series_names_and_window():
    source = CsvSource(FIXTURES)
    load = source.load("2024-10-26", "2024-10-28")
    price = source.day_ahead_prices("2024-10-26", "2024-10-28")
    assert load.name == "load_mw" and price.name == "price_eur_mwh"
    assert len(load) == 48 and len(price) == 48
    assert load.index[0] == pd.Timestamp("2024-10-26", tz="UTC")
    assert source.details["load"]["files"] == ["load_hourly_autumn_dst.csv", "load_quarter_hour_spring_dst.csv"]


def test_both_dst_days_are_complete_after_conversion():
    source = CsvSource(FIXTURES)
    load = source.load("2024-03-29", "2024-10-30")
    daily = daily_frame(pd.DataFrame({"temperature": 10.0, "load_mw": load, "price_eur_mwh": 50.0}))
    assert daily.loc["2024-03-31", "hours"] == 23
    assert daily.loc["2024-10-27", "hours"] == 25


def test_semicolon_delimiter_and_time_column_alias(tmp_path):
    (tmp_path / "x.csv").write_text(
        'Time (CET/CEST);Actual Total Load [MW] - BZN|FR\n'
        '01.01.2024 00:00 - 01.01.2024 01:00;50000\n'
        '01.01.2024 01:00 - 01.01.2024 02:00;51000\n'
    )
    kind, series = read_entsoe_csv(tmp_path / "x.csv")
    assert kind == "load" and series.index[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")


def test_unknown_file_is_rejected(tmp_path):
    (tmp_path / "other.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not an ENTSO-E"):
        read_entsoe_csv(tmp_path / "other.csv")
    assert classify(["a", "b"]) is None


def test_missing_series_or_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        CsvSource(tmp_path / "nope").load("2024-01-01", "2024-01-02")
    (tmp_path / "price.csv").write_text((FIXTURES / "price_hourly_autumn_dst.csv").read_text())
    with pytest.raises(UnsupportedSeriesError, match="load"):
        CsvSource(tmp_path).load("2024-10-26", "2024-10-27")
