"""Monthly refit: model file and metadata, and predicting with the stored model on a short window."""

import json

import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from test_jobs import DAY, NOW, FakeEntsoe, FakeWeather
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
    assert meta["train_from"] == "2023-09-01" and meta["train_to"] == "2024-03-31" and meta["fitted_at_utc"] == "2024-04-01T03:00:00Z"
    assert meta["train_hours"] == int(history["price_eur_mwh"].notna().sum())
    assert meta["features"][0] == "hour" and "load_fc_mw" in meta["features"] and "wind_fc_mw" not in meta["features"]
    holdout = meta["holdout"]
    assert holdout["holdout_from"] == "2024-03-02" and holdout["holdout_to"] == "2024-03-31" and holdout["holdout_hours"] == 30 * 24 - 1
    assert holdout["mae"] < holdout["naive_mae"]  # the synthetic price is learnable
    assert "params" in meta and meta["params"]["n_estimators"] == 40
    small = refit_model(history, out_dir=tmp_path / "small", holdout_days=0, log=lambda *_: None)
    assert small["params"]["n_estimators"] == 300 and small["params"]["num_leaves"] == 31  # REFIT_PARAMS by default


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
                          now=NOW, reports_dir=tmp_path / "none", feature_sets=("honest",), model_file=tmp_path / "model" / "honest.txt")
    assert summary["status"] == "ok" and list(summary["forecasts"]) == ["honest"]
    meta, curve = store.latest_forecast(DAY, "honest")
    assert meta["model"] == "gbm:honest.txt" and meta["train_hours"] == 0 and len(curve) == 24
    store.close()
