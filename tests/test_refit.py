"""Monthly refit: model file and metadata, and predicting with the stored model on a short window."""

import json

import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from test_jobs import DAY, NOW, FakeEntsoe, FakeSolarPoints, FakeWeather, FakeWindPoints
from thermo_fr.forecast import models
from thermo_fr.forecast.day import forecast_day
from thermo_fr.forecast.jobs import morning_run
from thermo_fr.forecast.models import predict_with
from thermo_fr.forecast.refit import refit_model
from thermo_fr.forecast.store import Store


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 40)
    return synthetic_inputs("2023-09-01", "2024-05-01", issued_from="2024-01-10")


def test_refit_writes_model_and_metadata_with_holdout_metrics(tmp_path, table):
    history = table[table.index < "2024-03-31T22:00Z"]  # up to the end of the Paris day 2024-03-31
    meta = refit_model(history, out_dir=tmp_path / "model", holdout_days=30, now=pd.Timestamp("2024-04-01T03:00Z"), log=lambda *_: None, params={})
    assert (tmp_path / "model" / "honest.txt").exists()
    saved = json.loads((tmp_path / "model" / "honest.json").read_text())
    assert saved == meta
    weights = json.loads((tmp_path / "model" / "wind_proxy.json").read_text())
    assert meta["wind_proxy"]["file"] == "wind_proxy.json" and meta["wind_proxy"]["capacity_mw"] == weights["fit"]["capacity_mw"]
    assert weights["fit"]["to"] == "2024-03-31 21:00:00+00:00" and weights["fit"]["mae_mw"] < 1000
    solar = json.loads((tmp_path / "model" / "solar_proxy.json").read_text())
    assert solar["proxy"] == "solar" and meta["solar_proxy"]["file"] == "solar_proxy.json" and solar["fit"]["mae_mw"] < 400
    assert solar["fit"]["from"] == "2024-01-10 00:00:00+00:00"  # the first hour with point forecasts
    assert {"wind_proxy_mw", "solar_proxy_mw", "day_type", "price_lag_same_type"} <= set(meta["features"])
    assert meta["train_from"] == "2023-09-01" and meta["train_to"] == "2024-03-31" and meta["fitted_at_utc"] == "2024-04-01T03:00:00Z"
    assert meta["train_hours"] == int(history["price_eur_mwh"].notna().sum())
    assert meta["features"][0] == "hour" and "load_fc_mw" in meta["features"] and "wind_fc_mw" not in meta["features"]
    check = meta["recent_check"]
    assert check["from"] == "2024-03-02" and check["to"] == "2024-03-31" and check["hours"] == 30 * 24 - 1
    assert check["mae"] < check["naive_mae"]  # the synthetic price is learnable
    assert check["excluded_from_stored_fit"] is False and "not the frozen holdout" in check["note"]
    assert meta["frozen_holdout"]["mae"] == 26.23 and "docs/experiments.md" in meta["frozen_holdout"]["source"]  # honest, copied from the log
    assert "params" in meta and meta["params"]["n_estimators"] == 40
    small = refit_model(history, out_dir=tmp_path / "small", holdout_days=0, log=lambda *_: None)
    assert small["params"]["n_estimators"] == 300 and small["params"]["num_leaves"] == 31  # REFIT_PARAMS by default


def test_refit_sets_writes_one_model_per_set_and_the_proxies_once(tmp_path, table):
    from thermo_fr.forecast.refit import refit_sets

    history = table[table.index < "2024-04-01"]
    metas = refit_sets(history, out_dir=tmp_path / "model", feature_sets=("honest", "honest_v2"), log=lambda *_: None)
    assert set(metas) == {"honest", "honest_v2"}
    for name in ("honest", "honest_v2"):
        assert (tmp_path / "model" / f"{name}.txt").exists() and json.loads((tmp_path / "model" / f"{name}.json").read_text())["feature_set"] == name
    v2 = json.loads((tmp_path / "model" / "honest_v2.json").read_text())
    assert {"nuclear_d1_early_mw", "price_de_lu_lag1", "spread_ch_lag1", "residual_v2_mw"} <= set(v2["features"])
    assert "pre-market" in v2["note"] and "wind_proxy" in metas["honest"] and "wind_proxy" not in v2
    assert (tmp_path / "model" / "wind_proxy.json").exists()


def test_stored_model_predicts_on_a_ten_day_window(tmp_path, table):
    refit_model(table[table.index < "2024-04-01"], out_dir=tmp_path / "model", holdout_days=0, log=lambda *_: None)
    window = table[(table.index >= "2024-03-30") & (table.index < "2024-04-12")]
    curve = forecast_day(DAY, window, None, feature_set="honest", model_file=tmp_path / "model" / "honest.txt", log=lambda *_: None)
    assert len(curve) == 24 and curve["forecast"].notna().all() and curve.attrs["train_hours"] == 0
    error = (curve["forecast"] - curve["actual"]).abs().mean()
    naive = (curve["naive_day"] - curve["actual"]).abs().mean()
    assert error < naive
    with pytest.raises(ValueError, match="not in the table"):
        predict_with(tmp_path / "model" / "honest.txt", curve[["hour"]])


def test_morning_run_with_a_model_file_needs_no_history(tmp_path, table):
    refit_model(table[table.index < "2024-04-01"], out_dir=tmp_path / "model", holdout_days=0, log=lambda *_: None)
    store = Store(tmp_path / "db.sqlite")
    summary = morning_run(store, DAY, kind="test", inputs_path=tmp_path / "absent.csv", entsoe=FakeEntsoe(table), weather=FakeWeather(table),
                          now=NOW, reports_dir=tmp_path / "none", feature_sets=("honest",), model_file=tmp_path / "model" / "honest.txt",
                          wind_points=FakeWindPoints(table), wind_weights=tmp_path / "model" / "wind_proxy.json",
                          solar_points=FakeSolarPoints(table), solar_weights=tmp_path / "model" / "solar_proxy.json")
    assert summary["status"] == "ok" and list(summary["forecasts"]) == ["honest"]
    meta, curve = store.latest_forecast(DAY, "honest")
    assert meta["model"] == "gbm:honest.txt" and meta["train_hours"] == 0 and len(curve) == 24
    store.close()


def test_the_stored_model_serves_its_own_feature_set_and_the_others_are_fitted_live(tmp_path, table):
    refit_model(table[table.index < "2024-04-01"], out_dir=tmp_path / "model", holdout_days=0, log=lambda *_: None)
    history = tmp_path / "inputs.csv"
    table[table.index < "2024-04-08"].to_csv(history, index_label="timestamp_utc")
    store = Store(tmp_path / "db.sqlite")
    summary = morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table), now=NOW,
                          reports_dir=tmp_path / "none", feature_sets=("honest", "extended"), model_file=tmp_path / "model" / "honest.txt",
                          wind_points=FakeWindPoints(table), wind_weights=tmp_path / "model" / "wind_proxy.json",
                          solar_points=FakeSolarPoints(table), solar_weights=tmp_path / "model" / "solar_proxy.json")
    assert summary["status"] == "ok" and set(summary["forecasts"]) == {"honest", "extended"}
    honest, _ = store.latest_forecast(DAY, "honest")
    extended, _ = store.latest_forecast(DAY, "extended")
    assert honest["model"] == "gbm:honest.txt" and honest["train_hours"] == 0
    assert extended["model"] == "gbm" and extended["train_hours"] > 0
    store.close()
