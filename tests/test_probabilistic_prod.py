"""Production path of the probabilistic forecasts: model files, the store table, the morning run with and without the files."""

import json

import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from test_jobs import DAY, FakeEntsoe, FakeSolarPoints, FakeWeather, FakeWindPoints, NOW
from thermo_fr.forecast import models, probabilistic as pb
from thermo_fr.forecast.features import build_features
from thermo_fr.forecast.jobs import morning_run
from thermo_fr.forecast.refit import refit_model, refit_sets
from thermo_fr.forecast.store import Store
from thermo_fr.forecast.wind_proxy import calibrate, save_weights
from thermo_fr.forecast import solar_proxy

SMALL = {"n_estimators": 20, "num_leaves": 15}


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 20)
    return synthetic_inputs("2023-09-01", "2024-05-01", issued_from="2024-01-10")


def test_fit_load_predict_round_trip(tmp_path, table):
    history = table[table.index < "2024-04-01"]
    features = build_features(history, "honest_v2")
    meta = pb.fit_probabilistic(features, "honest_v2", tmp_path / "model", params=SMALL, now=pd.Timestamp("2024-04-01T03:00Z"), log=lambda *_: None)
    paths = pb.prob_paths(tmp_path / "model", "honest_v2")
    assert all(p.exists() for p in paths.values()) and isinstance(meta["conformal_margin"], float) and meta["holdout"]["hours"] > 24 * 10
    saved = json.loads(paths["meta"].read_text())
    assert saved["features"] == list(features.X.columns) and saved["event_features"][-1] == "spike_threshold" and saved["events"]["negative"]["events"] == 0
    loaded = pb.load_probabilistic(tmp_path / "model", "honest_v2")
    day_rows = features.info["delivery_day"] == pd.Timestamp("2024-03-15")
    threshold = pb.spike_threshold_for_day(history, "2024-03-15")
    frame = pb.predict_probabilistic(loaded, features.X[day_rows], threshold)
    assert list(frame.columns) == pb.PROB_COLUMNS and len(frame) == 24
    assert (frame["q10"] <= frame["q50"]).all() and (frame["q50"] <= frame["q90"]).all() and ((frame["q10"] - frame["lo"]).round(3) == frame["conformal_margin"].round(3)).all()
    assert frame["p_negative"].between(0, 1).all() and frame["p_spike"].between(0, 1).all() and (frame["spike_threshold"] - threshold).abs().lt(1e-3).all()
    assert pb.load_probabilistic(tmp_path / "elsewhere", "honest_v2") is None
    paths["spike"].unlink()
    assert pb.load_probabilistic(tmp_path / "model", "honest_v2") is None  # one missing file means no band at all


def test_store_round_trip_and_live_margin(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    from test_store import curve_for
    frames = []
    for i, day in enumerate(pd.date_range("2026-09-01", periods=25, freq="D")):
        d = day.strftime("%Y-%m-%d")
        curve = curve_for(d)
        fid = store.save_forecast(i, d, "honest_v2", pd.Timestamp(day - pd.Timedelta(days=1)).tz_localize("UTC") + pd.Timedelta(hours=8),
                                  "gbm:honest_v2.txt", "scheduled", 0, True, curve)
        prob = pd.DataFrame(index=curve.index)
        prob["q10"], prob["q50"], prob["q90"] = curve["forecast"] - 5, curve["forecast"], curve["forecast"] + 5
        prob["lo"], prob["hi"] = prob["q10"] - 2, prob["q90"] + 2
        prob["p_negative"], prob["p_spike"], prob["spike_threshold"], prob["conformal_margin"] = 0.05, 0.1, 150.0, 2.0
        store.save_probabilistic(fid, prob)
        actual = pd.Series(curve["forecast"].to_numpy() + (8 if i % 2 else -3), index=curve.index)  # alternately outside and inside the raw band
        store.save_actuals(actual, pd.Series([d] * len(curve)))
        from thermo_fr.forecast.store import score_curve
        store.save_score(fid, d, "honest_v2", pd.Timestamp(day - pd.Timedelta(days=1)).strftime("%Y-%m-%dT08:00:00Z"), score_curve(curve, actual))
        frames.append(fid)
    back = store.probabilistic_curve(frames[0])
    assert list(back.columns) == pb.PROB_COLUMNS and len(back) == 24 and back["conformal_margin"].iloc[0] == 2.0
    settled = store.probabilistic_settled("honest_v2", days=90)
    assert settled["delivery_day"].nunique() == 25 and settled["actual"].notna().all()
    margin = pb.live_margin(settled, min_days=20)
    assert margin == pytest.approx(3.0)  # scores are 3 on the outside days and -2 inside: the 80% quantile with 25 days is 3
    assert pb.live_margin(settled[settled["delivery_day"] < "2026-09-10"], min_days=20) is None
    calibration = pb.live_calibration(settled)
    assert calibration["days"] == 25 and 0 < calibration["coverage_10_90"] < 1 and calibration["negative"]["events"] == 0
    assert len(calibration["spike"]["reliability"]) == 10
    store.close()


def test_morning_run_adds_the_band_when_the_files_exist_and_records_absence_otherwise(tmp_path, table):
    before = table[table.index < "2024-04-01"]
    refit_sets(before, out_dir=tmp_path / "model", feature_sets=("honest_v2",), log=lambda *_: None, probabilistic=False)
    pb.fit_probabilistic(build_features(before, "honest_v2"), "honest_v2", tmp_path / "model", params=SMALL, log=lambda *_: None)
    save_weights(calibrate(before, before["wind_onshore_mw"] + before["wind_offshore_mw"]), tmp_path / "model" / "wind_proxy.json")
    save_weights(solar_proxy.calibrate(before, before["solar_mw"]), tmp_path / "model" / "solar_proxy.json")
    history = tmp_path / "inputs.csv"
    table[table.index < "2024-04-08"].to_csv(history, index_label="timestamp_utc")
    store = Store(tmp_path / "forecast.db")

    def run(model_dir):
        return morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table), now=NOW,
                           reports_dir=tmp_path / "none", wind_points=FakeWindPoints(table), wind_weights=tmp_path / "model" / "wind_proxy.json",
                           solar_points=FakeSolarPoints(table), solar_weights=tmp_path / "model" / "solar_proxy.json", feature_sets=("honest_v2",),
                           model_file=model_dir / "honest_v2.txt", fallback_model_file=None)

    summary = run(tmp_path / "model")
    assert summary["status"] == "ok"
    fid = summary["forecasts"]["honest_v2"]
    prob = store.probabilistic_curve(fid)
    assert len(prob) == 24 and ((prob["q10"] - prob["lo"]).round(3) == prob["conformal_margin"].round(3)).all() and prob["p_spike"].between(0, 1).all()
    status = store.status_for(DAY)
    row = status[status["item"] == "probabilistic"].iloc[-1]
    assert row["status"] == "present" and "metadata" in row["message"]
    # the point model alone, no probabilistic files: the forecast is stored, the band is recorded as absent, the run stays ok
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "honest_v2.txt").write_bytes((tmp_path / "model" / "honest_v2.txt").read_bytes())
    (plain / "honest_v2.json").write_bytes((tmp_path / "model" / "honest_v2.json").read_bytes())
    summary = run(plain) if False else morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table),
                                                  now=NOW + pd.Timedelta(hours=1), reports_dir=tmp_path / "none", wind_points=FakeWindPoints(table),
                                                  wind_weights=tmp_path / "model" / "wind_proxy.json", solar_points=FakeSolarPoints(table),
                                                  solar_weights=tmp_path / "model" / "solar_proxy.json", feature_sets=("honest_v2",),
                                                  model_file=plain / "honest_v2.txt", fallback_model_file=None)
    assert summary["status"] == "ok" and store.probabilistic_curve(summary["forecasts"]["honest_v2"]).empty
    status = store.status_for(DAY)
    assert (status[status["item"] == "probabilistic"]["status"].tolist()[-1]) == "absent"
    store.close()


def test_refit_sets_writes_the_probabilistic_files_for_the_first_set_only(tmp_path, table, monkeypatch):
    monkeypatch.setattr(pb, "REFIT_PROB_PARAMS", SMALL)
    history = table[table.index < "2024-04-01"]
    metas = refit_sets(history, out_dir=tmp_path / "model", feature_sets=("honest_v2", "honest"), log=lambda *_: None)
    assert "probabilistic" in metas["honest_v2"] and "probabilistic" not in metas["honest"]
    assert (tmp_path / "model" / "honest_v2_q90.txt").exists() and not (tmp_path / "model" / "honest_q90.txt").exists()
    assert (tmp_path / "model" / "honest_v2_probabilistic.json").exists()
