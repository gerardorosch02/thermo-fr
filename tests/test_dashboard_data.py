"""The dashboard's read layer: panels built from a small store, no Streamlit and no network."""

import numpy as np
import pandas as pd
import pytest

from test_store import curve_for
from thermo_fr.dashboard import data as q
from thermo_fr.forecast.store import Store

TOMORROW, TODAY = "2026-10-08", "2026-10-07"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "forecast.db"
    s = Store(path)
    c = curve_for(TOMORROW, 100)
    s.save_forecast(1, TOMORROW, "honest", pd.Timestamp("2026-10-07T07:00Z"), "gbm", "scheduled", 50, True, c)
    s.save_forecast(2, TOMORROW, "honest", pd.Timestamp("2026-10-07T09:00Z"), "gbm", "scheduled", 50, True, curve_for(TOMORROW, 110))
    band = pd.DataFrame({"feature_set": "honest", "hour": range(24), "n": 100, "mae": 5.0, "p10": -8.0, "p25": -3.0, "p50": 0.0, "p75": 3.0, "p90": 8.0})
    s.save_error_band(band, "test")
    today_curve = curve_for(TODAY, 90)
    s.save_actuals(pd.Series(95.0 + today_curve["hour"].to_numpy(), index=today_curve.index), pd.Series([TODAY] * 24))
    inputs = pd.DataFrame({"load_fc_mw": 50000.0, "solar_fc_mw": 3000.0, "wind_onshore_fc_mw": 4000.0, "wind_offshore_fc_mw": 500.0,
                           "temp_fc_c": 12.0, "wind100_fc_ms": 6.0, "radiation_fc_wm2": 100.0}, index=c.index)
    s.save_input_values(2, TOMORROW, inputs)
    s.save_input_values(1, TODAY, inputs.set_index(today_curve.index) - 1000.0)
    s.record_status(2, TOMORROW, "load_forecast", "present", hours=24, revision="1", hash_="a", now=pd.Timestamp("2026-10-07T07:00Z"))
    s.record_status(2, TOMORROW, "wind_solar_forecast", "absent", now=pd.Timestamp("2026-10-07T07:00Z"))
    s.record_status(2, TOMORROW, "weather_issued", "present", hours=20, hash_="w", now=pd.Timestamp("2026-10-07T07:00Z"))
    s.record_status(2, TOMORROW, "prices", "error", message="gateway 599", now=pd.Timestamp("2026-10-07T07:00Z"))
    s.close()
    return path


def test_forecast_panel_builds_hourly_table_with_band_and_todays_actual(db):
    panel = q.forecast_panel(db, TOMORROW, "honest", TODAY)
    assert panel["meta"]["forecast_id"] == 2 and len(panel["versions"]) == 2
    by_hour = panel["by_hour"]
    assert len(by_hour) == 24 and by_hour["forecast"].iloc[0] == 110.0
    assert by_hour["band_p90"].iloc[0] == 118.0 and by_hour["band_p10"].iloc[0] == 102.0
    assert by_hour["actual_other"].iloc[3] == 98.0 and by_hour["actual_own"].isna().all()
    empty = q.forecast_panel(db, "2026-10-09", "honest", TODAY)
    assert empty["meta"] is None and "by_hour" not in empty


def test_inputs_panel_means_and_change_against_today(db):
    panel = q.inputs_panel(db, TOMORROW, TODAY)
    means, previous = panel["tomorrow"]["means"], panel["today"]["means"]
    assert means["wind_fc_mw"] == 4500.0 and means["residual_fc_mw"] == 50000.0 - 4500.0 - 3000.0
    assert means["load_fc_mw"] - previous["load_fc_mw"] == pytest.approx(1000.0)
    assert q.daily_inputs(pd.DataFrame()) == {}


def test_status_panel_flags_missing_late_and_errors(db):
    panel = q.status_panel(db, TOMORROW)
    table = panel["table"]
    assert table.loc["load_forecast", "flag"] == "ok" and table.loc["load_forecast", "minutes_before_gate"] == 180.0
    assert table.loc["wind_solar_forecast", "flag"] == "missing"
    assert table.loc["prices", "flag"] == "source error"
    assert table.loc["weather_issued", "flag"] == "incomplete"
    assert q.flag_for({"status": "present", "hours": 24, "minutes_before_gate": -30.0}) == "late (after the gate)"
    summary = panel["summary"].set_index("item")
    assert summary.loc["load_forecast", "days"] == 1


def test_performance_panel_rolls_scores(db):
    s = Store(db)
    c = curve_for(TODAY, 90)
    fid = s.save_forecast(1, TODAY, "honest", pd.Timestamp("2026-10-06T07:00Z"), "gbm", "scheduled", 50, True, c)
    s.save_score(fid, TODAY, "honest", "2026-10-06T07:00:00Z", {"hours": 24, "mae": 5.0, "rmse": 6.0, "naive_mae": 7.0, "naive_rmse": 8.0, "model_won": True})
    s.close()
    panel = q.performance_panel(db, "honest")
    assert panel["share_won"] == 1.0 and panel["mae"] == 5.0 and panel["naive_mae"] == 7.0
    assert panel["scores"]["rolling_mae"].iloc[0] == 5.0 and len(panel["hourly"]) == 24
    assert q.performance_panel(db, "extended")["scores"].empty


def test_db_version_and_day_helpers(db):
    assert q.db_version(db) > 0 and q.db_version(db.parent / "missing.db") == 0.0
    assert q.today_and_tomorrow(pd.Timestamp("2026-10-07T08:00Z")) == (TODAY, TOMORROW)
    assert not np.isnan(q.db_version(db))
