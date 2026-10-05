import numpy as np
import pandas as pd

from thermo_fr.data.quality import format_quality, quality_report


def frame(start, end):
    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    return pd.DataFrame({"temperature": 10.0, "load_mw": 50000.0, "price_eur_mwh": 80.0}, index=idx)


def test_clean_data_has_no_flags():
    report = quality_report(frame("2024-01-01", "2024-01-03"), "2024-01-01", "2024-01-03")
    assert report["expected_hours"] == 48
    assert report["series"]["load_mw"]["missing_hours"] == 0
    assert report["flags"] == []
    assert "missing      0 h (0.00%)" in format_quality(report)


def test_missing_hours_are_counted_not_filled():
    hourly = frame("2024-01-01", "2024-01-03")
    hourly.loc["2024-01-01 05:00":"2024-01-01 08:00", "load_mw"] = np.nan
    hourly = hourly.drop(hourly.index[-2:])  # rows absent altogether count as missing too
    report = quality_report(hourly, "2024-01-01", "2024-01-03")
    assert report["series"]["load_mw"]["missing_hours"] == 6
    assert report["series"]["load_mw"]["missing_share"] == round(6 / 48, 4)
    assert report["series"]["price_eur_mwh"]["missing_hours"] == 2
    assert any(f.startswith("load_mw: 12.5%") for f in report["flags"])
    assert hourly["load_mw"].isna().sum() == 4  # input untouched


def test_implausible_values_are_flagged():
    hourly = frame("2024-01-01", "2024-01-02")
    hourly.iloc[0, hourly.columns.get_loc("load_mw")] = -5.0
    hourly.iloc[1, hourly.columns.get_loc("load_mw")] = 150000.0
    hourly.iloc[2, hourly.columns.get_loc("price_eur_mwh")] = 5000.0
    report = quality_report(hourly, "2024-01-01", "2024-01-02")
    load = report["series"]["load_mw"]
    assert load["negative"] == 1 and load["below_plausible"] == 1 and load["above_plausible"] == 1
    assert report["series"]["price_eur_mwh"]["above_plausible"] == 1
    assert "load_mw: 1 negative values" in report["flags"]
    assert "price_eur_mwh: 1 values outside -500.0 to 4,000.0" in report["flags"]
