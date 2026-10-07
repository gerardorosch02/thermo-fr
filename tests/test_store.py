"""The SQLite store: timing log, forecast versions that never overwrite, scoring and the error band."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast.store import ForecastExistsError, Store, error_band_from_predictions, score_curve, value_hash


def curve_for(day: str, level: float = 100.0) -> pd.DataFrame:
    start = pd.Timestamp(day, tz="Europe/Paris")
    end = (pd.Timestamp(day) + pd.DateOffset(days=1)).tz_localize("Europe/Paris")
    index = pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")
    hours = index.tz_convert("Europe/Paris").hour
    return pd.DataFrame({"hour": hours, "forecast": level + hours, "naive_day": level - 5.0 + hours}, index=index)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "forecast.db")
    yield s
    s.close()


def test_runs_are_recorded_with_status(store):
    run = store.start_run("morning-run", "test", "2026-10-08", now=pd.Timestamp("2026-10-07T07:00Z"))
    store.finish_run(run, "partial", "wind_solar_forecast: down", now=pd.Timestamp("2026-10-07T07:02Z"))
    runs = store.runs()
    assert len(runs) == 1 and runs.iloc[0]["status"] == "partial" and runs.iloc[0]["finished_at_utc"] == "2026-10-07T07:02:00Z"


def test_timing_log_records_first_appearance_and_revisions(store):
    day = "2026-10-08"  # gate is 12:00 Paris on the 7th = 10:00 UTC
    store.record_status(1, day, "load_forecast", "absent", now=pd.Timestamp("2026-10-07T06:00Z"))
    assert store.timing_log().empty
    store.record_status(1, day, "load_forecast", "present", hours=24, revision="1", hash_="aaa", now=pd.Timestamp("2026-10-07T07:00Z"))
    store.record_status(2, day, "load_forecast", "present", hours=24, revision="1", hash_="aaa", now=pd.Timestamp("2026-10-07T08:00Z"))
    store.record_status(3, day, "load_forecast", "present", hours=24, revision="2", hash_="bbb", now=pd.Timestamp("2026-10-07T09:00Z"))
    store.record_status(3, day, "wind_solar_forecast", "present", hours=24, revision="3", hash_="ccc", now=pd.Timestamp("2026-10-07T15:00Z"))
    log = store.timing_log().set_index("item")
    assert log.loc["load_forecast", "first_seen_utc"] == "2026-10-07T07:00:00Z"
    assert log.loc["load_forecast", "first_seen_paris"] == "2026-10-07 09:00"
    assert log.loc["load_forecast", "minutes_before_gate"] == 180.0
    assert log.loc["load_forecast", "changes"] == 1 and log.loc["load_forecast", "last_changed_utc"] == "2026-10-07T09:00:00Z"
    assert log.loc["load_forecast", "checks"] == 3
    assert log.loc["wind_solar_forecast", "minutes_before_gate"] == -300.0
    latest = store.latest_status(day)
    assert latest.loc["load_forecast", "value_hash"] == "bbb" and latest.loc["load_forecast", "revision"] == "2"
    summary = store.timing_summary().set_index("item")
    assert summary.loc["load_forecast", "days_seen_before_gate"] == 1 and summary.loc["wind_solar_forecast", "days_seen_before_gate"] == 0
    assert summary.loc["load_forecast", "days_with_changes"] == 1


def test_forecast_versions_are_kept_and_never_overwritten(store):
    day = "2026-10-08"
    first = store.save_forecast(1, day, "honest", pd.Timestamp("2026-10-07T07:00Z"), "gbm", "scheduled", 1000, True, curve_for(day, 100))
    second = store.save_forecast(2, day, "honest", pd.Timestamp("2026-10-07T09:00Z"), "gbm", "scheduled", 1000, True, curve_for(day, 110))
    with pytest.raises(ForecastExistsError):
        store.save_forecast(3, day, "honest", pd.Timestamp("2026-10-07T09:00Z"), "gbm", "scheduled", 1000, True, curve_for(day, 120))
    versions = store.forecast_versions(day, "honest")
    assert versions["forecast_id"].tolist() == [first, second]
    assert store.forecast_curve(first)["forecast"].iloc[0] == 100.0  # the first version is untouched
    meta, curve = store.latest_forecast(day, "honest")
    assert meta["forecast_id"] == second and curve["forecast"].iloc[0] == 110.0 and len(curve) == 24
    assert store.latest_forecast(day, "extended")[0] is None
    assert store.forecast_days() == [day]


def test_input_values_round_trip(store):
    day = "2026-10-08"
    c = curve_for(day)
    frame = pd.DataFrame({"load_fc_mw": 50000.0 + c["hour"], "solar_fc_mw": [np.nan] * 24}, index=c.index)
    assert store.save_input_values(1, day, frame) == 48
    back = store.latest_input_values(day)
    assert back["load_fc_mw"].iloc[5] == 50005.0 and back["solar_fc_mw"].isna().all()
    assert store.latest_input_values("2026-10-09").empty


def test_actuals_scores_and_model_won(store):
    day = "2026-10-08"
    c = curve_for(day, 100)
    fid = store.save_forecast(1, day, "honest", pd.Timestamp("2026-10-07T07:00Z"), "gbm", "scheduled", 10, True, c)
    actual = pd.Series(102.0 + c["hour"].to_numpy(), index=c.index)
    store.save_actuals(actual, pd.Series([day] * 24), now=pd.Timestamp("2026-10-07T13:00Z"))
    assert store.actual_days() == [day] and len(store.actuals_for(day)) == 24
    score = score_curve(store.forecast_curve(fid), store.actuals_for(day))
    assert score["mae"] == pytest.approx(2.0) and score["naive_mae"] == pytest.approx(7.0) and score["model_won"]
    assert score["rmse"] == pytest.approx(2.0)
    store.save_score(fid, day, "honest", "2026-10-07T07:00:00Z", score, now=pd.Timestamp("2026-10-07T13:00Z"))
    assert store.unscored_forecasts().empty
    latest = store.latest_scores("honest")
    assert len(latest) == 1 and latest.iloc[0]["model_won"] == 1
    assert score_curve(store.forecast_curve(fid), pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))) is None


def test_error_band_percentiles_by_hour():
    rng = np.random.default_rng(0)
    hours = np.tile(np.arange(24), 200)
    actual = 50.0 + rng.normal(0, 1, len(hours))
    predictions = pd.DataFrame({"hour": hours, "actual": actual, "gbm": actual + rng.normal(0, 1 + hours / 12, len(hours)),
                                "strict": [True] * (len(hours) - 24) + [False] * 24})
    band = error_band_from_predictions(predictions, "honest")
    assert list(band.columns) == ["feature_set", "hour", "n", "mae", "p10", "p25", "p50", "p75", "p90"]
    assert len(band) == 24 and band["n"].sum() == len(hours) - 24
    assert (band["p10"] < band["p50"]).all() and (band["p50"] < band["p90"]).all()
    assert band.set_index("hour").loc[23, "mae"] > band.set_index("hour").loc[0, "mae"]  # wider errors late in the day
    with pytest.raises(ValueError):
        error_band_from_predictions(predictions[predictions["strict"] == False], "honest")  # noqa: E712


def test_error_band_store_round_trip(store):
    band = pd.DataFrame({"feature_set": "honest", "hour": range(24), "n": 10, "mae": 5.0, "p10": -8.0, "p25": -3.0, "p50": 0.0, "p75": 3.0, "p90": 8.0})
    store.save_error_band(band, source="test", now=pd.Timestamp("2026-10-07T07:00Z"))
    back = store.error_band("honest")
    assert len(back) == 24 and back.loc[3, "p90"] == 8.0 and back.loc[3, "source"] == "test"
    assert store.error_band("extended").empty


def test_value_hash_ignores_nothing_but_rounds():
    assert value_hash([1.0001, 2.0]) == value_hash([1.0002, 2.0])
    assert value_hash([1.0, 2.0]) != value_hash([1.0, 2.5])
    assert value_hash([np.nan, np.nan]) is None and value_hash([]) is None
