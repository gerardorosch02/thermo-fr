"""The solar alignment fix (branch solar-align): start-of-hour labels for Open-Meteo preceding-hour means in both paths,
the ENTSO-E relabelling rule, the clear-sky labelling test, and that the fix uses nothing published later than before."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from thermo_fr.config import City
from thermo_fr.data import hour_labels
from thermo_fr.data.solar_points import SOLAR_POINTS, SolarPointsSource
from thermo_fr.data.weather_forecast import OpenMeteoForecastSource
from thermo_fr.forecast import align, features, label_audit, solar_labels
from thermo_fr.forecast.timing import check_point_in_time, issue_times, premarket_issue_for, radiation_issued_issue, weather_issued_issue

NOW = pd.Timestamp("2026-10-17T12:00Z")
POINTS = SOLAR_POINTS[:3]


class FakeOpenMeteo:
    """Answers like the Open-Meteo APIs: radiation as the mean over the hour ending at each label (clear sky), temperature
    and wind as instant values."""

    def __init__(self, points=POINTS):
        self.where = {(p.lat, p.lon): p for p in points}
        self.calls = []

    def get(self, url, params):
        self.calls.append(params)
        times = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="1h", tz="UTC")
        lat, lon = params["latitude"], params["longitude"]
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for variable in params["hourly"].split(","):
            if variable.startswith("shortwave_radiation"):
                values = solar_labels.clear_sky_hourly(times - pd.Timedelta(hours=1), lat, lon).to_numpy()  # mean over [T-1, T)
            elif variable.startswith("temperature_2m"):
                values = 10 + np.sin(np.arange(len(times)) / 5.0)
            else:
                values = 20 + np.cos(np.arange(len(times)) / 7.0)
            hourly[variable] = [round(float(v), 3) for v in values]
        return SimpleNamespace(content=json.dumps({"hourly": hourly}).encode())


@pytest.fixture
def labels(monkeypatch):
    def use(value):
        monkeypatch.setattr(hour_labels, "HOUR_LABELS", value)
    return use


def test_the_default_is_as_logged_and_changes_nothing_logged(labels):
    assert hour_labels.HOUR_LABELS == hour_labels.AS_LOGGED
    idx = pd.date_range("2025-06-01", "2025-06-03", freq="1h", tz="UTC")
    assert (radiation_issued_issue(idx) == weather_issued_issue(idx)).all()
    assert (issue_times(idx, "solar_proxy") == issue_times(idx, "wind_proxy")).all()
    assert features.FEATURE_TIMINGS["radiation_wm2"] == "radiation_issued"


def test_preceding_hour_variables_move_to_the_start_of_their_hour_and_instant_ones_do_not(tmp_path, labels):
    city = (City("Toulouse", 43.60, 1.45, 1.0),)
    labels(hour_labels.AS_LOGGED)
    logged = OpenMeteoForecastSource(cities=city, cache_dir=tmp_path / "a", client=FakeOpenMeteo(city), now=NOW).fetch(
        "2026-06-10", "2026-06-13", kinds=("issued",))
    labels(hour_labels.START_OF_HOUR)
    fixed = OpenMeteoForecastSource(cities=city, cache_dir=tmp_path / "b", client=FakeOpenMeteo(city), now=NOW).fetch(
        "2026-06-10", "2026-06-13", kinds=("issued",))
    pd.testing.assert_series_equal(fixed["temp_fc_c"], logged["temp_fc_c"])  # instant: unchanged
    pd.testing.assert_series_equal(fixed["wind100_fc_ms"], logged["wind100_fc_ms"])
    rad, raw = fixed["radiation_fc_wm2"], logged["radiation_fc_wm2"]
    np.testing.assert_allclose(rad.iloc[:-1].to_numpy(), raw.iloc[1:].to_numpy())  # the value for [T, T+1) now sits at T
    assert rad.index[-1].hour == 23 and rad.iloc[-1] == 0.0  # the last hour of the range: night in France
    truth = solar_labels.clear_sky_hourly(rad.index, 43.60, 1.45)  # the mean over [T, T+1)
    assert (rad - truth).abs().max() < 1.0


def test_history_and_morning_run_paths_get_the_same_labels(tmp_path, labels):
    """The morning run fetches a short window, the history path a long range; on start-of-hour labels both equal the
    as-logged table converted by align_table, hour for hour."""
    labels(hour_labels.AS_LOGGED)
    history_logged = SolarPointsSource(points=POINTS, cache_dir=tmp_path / "h0", client=FakeOpenMeteo(), now=NOW).fetch_points(
        "2026-05-01", "2026-06-20")
    labels(hour_labels.START_OF_HOUR)
    history = SolarPointsSource(points=POINTS, cache_dir=tmp_path / "h1", client=FakeOpenMeteo(), now=NOW).fetch_points(
        "2026-05-01", "2026-06-20")
    window = SolarPointsSource(points=POINTS, cache_dir=tmp_path / "w", client=FakeOpenMeteo(), now=NOW).fetch_points(
        "2026-06-05", "2026-06-17")  # D-10 to D+1 around D = 2026-06-15, as fetch_fresh asks
    converted = hour_labels.align_table(history_logged, columns=list(history_logged.columns))
    pd.testing.assert_frame_equal(history, converted)
    pd.testing.assert_frame_equal(window, history.loc[window.index])
    assert (window.iloc[-1] == 0.0).all() and window.index[-1].hour == 23


# --- the clear-sky labelling test ------------------------------------------------------------------------------------------

def synthetic_solar(days=120, start="2025-03-01", late=None):
    """Radiation as Open-Meteo labels it (end of hour) and actual solar as ENTSO-E labels it (start of hour), clear sky
    with a day-to-day cloud factor; `late` = (first, last) days on which the actuals are published one hour late."""
    idx = pd.date_range(start, periods=24 * days, freq="1h", tz="UTC")
    clear = solar_labels.national_clear_sky(idx, POINTS)
    rng = np.random.default_rng(0)
    factor = pd.Series(np.repeat(rng.uniform(0.3, 1.0, days), 24), index=idx)
    radiation_logged = pd.Series(solar_labels.national_clear_sky(idx - pd.Timedelta(hours=1), POINTS).to_numpy(), index=idx) * factor
    truth = (clear * factor * 20.0).rename("solar_mw")
    published = truth.copy()
    if late is not None:
        rows = (idx.normalize() >= pd.Timestamp(late[0], tz="UTC")) & (idx.normalize() <= pd.Timestamp(late[1], tz="UTC"))
        published[rows] = truth.reindex(idx[rows] - pd.Timedelta(hours=1)).to_numpy()
    return radiation_logged.rename("rad"), truth, published, clear


def test_clear_sky_labelling_test_fails_as_logged_and_passes_on_start_of_hour_labels():
    radiation, truth, _, clear = synthetic_solar()
    logged = solar_labels.clear_sky_label_check(radiation, truth, clear)
    fixed = solar_labels.clear_sky_label_check(hour_labels.to_start_of_hour(radiation), truth, clear)
    assert not logged["passed"] and logged["radiation_minus_actual_h"] > 0.75
    assert fixed["passed"] and abs(fixed["radiation_minus_actual_h"]) <= 0.25
    half = solar_labels.clear_sky_label_check(radiation.shift(1), truth, clear)  # two hours apart: fails too
    assert not half["passed"]


@pytest.mark.skipif(not Path("data/forecast/inputs.csv").exists(), reason="needs the local inputs history (git-ignored)")
def test_clear_sky_labelling_on_the_real_inputs():
    from thermo_fr.forecast.inputs import load_inputs
    from thermo_fr.data.solar_points import POINT_COLUMNS

    table = load_inputs("data/forecast/inputs.csv")
    table = table[(table.index >= "2024-05-01") & (table.index < "2026-07-01")]
    clear = solar_labels.national_clear_sky(table.index)
    for columns in (POINT_COLUMNS, ["radiation_fc_wm2"]):
        logged = table[columns].mean(axis=1)
        assert not solar_labels.clear_sky_label_check(logged, table["solar_mw"], clear)["passed"]
        fixed = hour_labels.to_start_of_hour(logged)
        assert solar_labels.clear_sky_label_check(fixed, table["solar_mw"], clear)["passed"]


# --- amendment 2: the ENTSO-E relabelling rule ------------------------------------------------------------------------------

def test_relabel_rule_finds_a_late_period_and_repairs_it():
    _, truth, published, clear = synthetic_solar(days=200, start="2023-06-01", late=("2023-09-01", "2023-10-20"))
    repaired, days, offsets = solar_labels.relabel_entsoe_solar(published, clear)
    late = pd.date_range("2023-09-01", "2023-10-20", tz="UTC")
    assert set(days) <= set(pd.date_range("2023-08-29", "2023-10-23", tz="UTC"))  # the centred window blurs the edges by 3 days
    assert len(set(days) & set(late)) >= len(late) - 6
    inner = pd.date_range("2023-09-05", "2023-10-16", tz="UTC")
    rows = published.index.normalize().isin(inner)
    np.testing.assert_allclose(repaired[rows].to_numpy()[:-1], truth[rows].to_numpy()[:-1], atol=1e-9)
    assert abs(offsets.attrs["median_offset_h"]) < 0.3


def test_relabel_rule_leaves_a_clean_history_alone():
    _, truth, _, clear = synthetic_solar(days=200, start="2023-06-01")
    repaired, days, _ = solar_labels.relabel_entsoe_solar(truth, clear)
    assert len(days) == 0
    pd.testing.assert_series_equal(repaired, truth.astype(float))


def test_label_audit_sees_a_one_hour_shift_in_an_actual_series():
    idx = pd.date_range("2023-01-01", periods=24 * 365, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    reference = pd.Series(np.cumsum(rng.normal(size=len(idx))), index=idx)
    actual = reference.copy()
    rows = (idx >= "2023-06-01") & (idx < "2023-09-01")
    actual[rows] = reference.shift(1)[rows]  # one hour late
    lag = label_audit.monthly_lag(actual, reference)
    assert lag["2023-07"] == pytest.approx(1.0, abs=0.2) and lag["2023-02"] == pytest.approx(0.0, abs=0.2)


def test_align_inputs_recalibrates_the_proxy_on_the_corrected_basis(labels):
    from thermo_fr.data.solar_points import POINT_COLUMNS

    radiation, truth, published, clear = synthetic_solar(days=200, start="2023-06-01", late=("2023-09-01", "2023-10-20"))
    table = pd.DataFrame(index=radiation.index)
    for column in POINT_COLUMNS + ["radiation_fc_wm2", "radiation_proxy_wm2"]:
        table[column] = radiation
    table["solar_mw"] = published
    table["solar_proxy_mw"] = np.nan
    out, report, weights = align.align_inputs(table, log=lambda *_: None)
    assert report["relabelled_solar_day_count"] > 40 and weights is not None
    assert out["solar_proxy_mw"].notna().any()
    np.testing.assert_allclose(out["radiation_fc_wm2"].iloc[:-1].to_numpy(), radiation.iloc[1:].to_numpy())


# --- timing gate and perturbation: nothing published later than before ---------------------------------------------------

def day_hours(day):
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    return pd.date_range(start, start + pd.Timedelta(days=1), freq="1h", inclusive="left").tz_convert("UTC")


def test_start_of_hour_radiation_is_published_no_later_than_before(labels):
    for day in pd.date_range("2025-01-01", "2025-12-31"):
        hours = day_hours(day)
        labels(hour_labels.AS_LOGGED)
        before = radiation_issued_issue(hours).max()
        proxy_before = issue_times(hours, "solar_proxy").max()
        labels(hour_labels.START_OF_HOUR)
        after = radiation_issued_issue(hours).max()
        assert after <= before, day  # the same lead-day-2 run cycle at the latest
        assert issue_times(hours, "solar_proxy").max() <= proxy_before
        assert after <= premarket_issue_for(hours[:1])[0]


def test_honest_v2_and_lear_features_pass_the_premarket_gate_on_start_of_hour_labels(labels):
    labels(hour_labels.START_OF_HOUR)
    idx = pd.date_range("2024-01-01", "2026-07-01", freq="1h", tz="UTC", inclusive="left")
    honest = {f: features.FEATURE_TIMINGS[f] for f in features.FEATURES["honest_v2"]}
    check_point_in_time(idx, honest, deadline="premarket")
    check_point_in_time(idx, {"rad_d0": features.LEAR_BLOCK_TIMINGS["rad_d0"]}, deadline="premarket")


def test_perturbing_what_was_published_later_changes_nothing_on_start_of_hour_labels(labels):
    """Raw Open-Meteo radiation (labelled at the end of its hour). Every raw value published after the latest raw value the
    as-logged features of D used is replaced by noise: D's start-of-hour radiation does not move. The raw value for the
    hour after D's last label does move it (positive control), and it was published no later than before."""
    raw_index = pd.date_range("2025-06-01", "2025-06-30", freq="1h", tz="UTC", inclusive="left")
    raw = pd.Series(solar_labels.clear_sky_hourly(raw_index - pd.Timedelta(hours=1), 43.6, 1.45).to_numpy(), index=raw_index)
    for day in ("2025-06-10", "2025-06-15", "2025-06-21"):
        hours = day_hours(day)
        latest_before = weather_issued_issue(hours).max()  # what the as-logged features of D used
        base = hour_labels.to_start_of_hour(raw).reindex(hours)
        published = weather_issued_issue(raw.index)
        noisy = raw.where(published <= latest_before, raw + 500.0 + np.arange(len(raw)))
        assert (noisy != raw).any()
        pd.testing.assert_series_equal(hour_labels.to_start_of_hour(noisy).reindex(hours), base)
        used = hours[-1] + pd.Timedelta(hours=1)
        moved = raw.copy()
        moved[used] += 100.0
        assert hour_labels.to_start_of_hour(moved).reindex(hours).iloc[-1] == base.iloc[-1] + 100.0
        assert weather_issued_issue(pd.DatetimeIndex([used]))[0] <= latest_before


def test_hour_labels_setting_rejects_unknown_values(labels):
    labels("utc_end")
    with pytest.raises(ValueError):
        hour_labels.current()
