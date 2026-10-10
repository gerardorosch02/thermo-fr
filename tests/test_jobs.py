"""morning-run and settle, offline: graceful failures, status rows, versions, settlement scoring."""

import logging

import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from thermo_fr.data.solar_points import POINT_COLUMNS as SOLAR_POINT_COLUMNS
from thermo_fr.data.wind_points import POINT_COLUMNS
from thermo_fr.forecast import models, solar_proxy, wind_proxy
from thermo_fr.forecast.inputs import NEIGHBOUR_PRICE_COLUMNS
from thermo_fr.forecast.jobs import day_hours, missing_features, morning_run, next_delivery_day, presence, settle, setup_logging
from thermo_fr.forecast.refit import refit_model
from thermo_fr.forecast.store import Store
from thermo_fr.forecast.features import FEATURES
from thermo_fr.forecast.wind_proxy import calibrate, save_weights

FEATURES_HONEST = FEATURES["honest"]

DAY = "2024-04-10"  # delivery day inside the synthetic table
NOW = pd.Timestamp("2024-04-09T07:30Z")  # 09:30 Paris on D-1


class FakeEntsoe:
    """Serves slices of the synthetic table; any item listed in `down` raises; `hide` blanks a delivery day."""

    def __init__(self, table: pd.DataFrame, down=(), hide=None, revision="1"):
        self.table, self.down, self.hide, self.revision = table, set(down), hide, revision
        self.details = {}

    def _slice(self, item, start, end, columns):
        if item in self.down:
            raise ConnectionError(f"{item} service unavailable")
        part = self.table.loc[start:end, columns].copy()
        part = part[part.index < pd.Timestamp(end, tz="UTC")]
        if self.hide:
            part.loc[part.index.isin(day_hours(self.hide))] = float("nan")
        self.details[item] = [{"revision": self.revision}]
        return part

    def load_forecast(self, start, end):
        return self._slice("load_forecast", start, end, "load_fc_mw")

    def wind_solar_forecast(self, start, end):
        return self._slice("wind_solar_forecast", start, end, ["solar_fc_mw", "wind_onshore_fc_mw", "wind_offshore_fc_mw"])

    def day_ahead_prices(self, start, end):
        return self._slice("prices", start, end, "price_eur_mwh")

    def nuclear_generation_actual(self, start, end):
        return self._slice("nuclear_actual", start, end, "nuclear_mw")

    def neighbour_prices(self, start, end):
        return self._slice("neighbour_prices", start, end, NEIGHBOUR_PRICE_COLUMNS)


class FakeWeather:
    def __init__(self, table, down=False):
        self.table, self.down = table, down

    def fetch(self, start, end, kinds=("issued",)):
        if self.down:
            raise TimeoutError("open-meteo timed out")
        part = self.table.loc[start:end, ["temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"]]
        return part[part.index < pd.Timestamp(end, tz="UTC")]


class FakeWindPoints:
    columns = POINT_COLUMNS

    def __init__(self, table, down=False):
        self.table, self.down = table, down

    def fetch_points(self, start, end):
        if self.down:
            raise TimeoutError("open-meteo timed out")
        part = self.table.loc[start:end, self.columns]
        return part[part.index < pd.Timestamp(end, tz="UTC")]


class FakeSolarPoints(FakeWindPoints):
    columns = SOLAR_POINT_COLUMNS


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    return synthetic_inputs("2023-09-01", "2024-05-01", issued_from="2024-01-10")


@pytest.fixture
def weights(table, tmp_path):
    """Wind proxy calibration fitted on the synthetic history before the delivery day."""
    before = table[table.index < "2024-04-01"]
    actual = before["wind_onshore_mw"] + before["wind_offshore_mw"]
    return save_weights(calibrate(before, actual), tmp_path / "model" / "wind_proxy.json")


@pytest.fixture
def solar_weights(table, tmp_path):
    before = table[table.index < "2024-04-01"]
    return save_weights(solar_proxy.calibrate(before, before["solar_mw"]), tmp_path / "model" / "solar_proxy.json")


def run(store, table, history, tmp_path, weights, entsoe=None, weather=None, now=NOW, **kw):
    solar = kw.pop("solar_weights", tmp_path / "model" / "solar_proxy.json")
    kw.setdefault("feature_sets", ("honest", "extended"))  # the older tests were written for these two sets
    return morning_run(store, DAY, kind="test", inputs_path=history, entsoe=entsoe or FakeEntsoe(table), weather=weather or FakeWeather(table),
                       now=now, reports_dir=tmp_path / "none", wind_points=kw.pop("wind_points", None) or FakeWindPoints(table),
                       wind_weights=weights, solar_points=kw.pop("solar_points", None) or FakeSolarPoints(table), solar_weights=solar, **kw)


@pytest.fixture
def history(table, tmp_path):
    """Stored inputs ending before the delivery day, so the fresh fetch must supply D."""
    path = tmp_path / "inputs.csv"
    table[table.index < "2024-04-08"].to_csv(path, index_label="timestamp_utc")
    return path


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "forecast.db")
    yield s
    s.close()


def test_next_delivery_day_is_tomorrow_in_paris():
    assert next_delivery_day(pd.Timestamp("2026-10-06T22:30Z")) == "2026-10-08"  # already the 7th in Paris
    assert next_delivery_day(pd.Timestamp("2026-10-06T21:30Z")) == "2026-10-07"


def test_presence_counts_complete_hours_only():
    hours = day_hours(DAY)
    frame = pd.DataFrame({"a": 1.0, "b": 2.0}, index=hours)
    assert presence(frame, DAY) == ("present", 24, presence(frame, DAY)[2])
    frame.loc[hours[:3], "b"] = float("nan")
    assert presence(frame, DAY)[:2] == ("present", 21)
    assert presence(pd.Series(float("nan"), index=hours), DAY) == ("absent", 0, None)


def test_morning_run_stores_both_feature_sets_and_the_timing_log(store, table, history, tmp_path, weights, solar_weights):
    entsoe = FakeEntsoe(table, revision="3")
    summary = run(store, table, history, tmp_path, weights, entsoe=entsoe)
    assert summary["status"] == "ok" and set(summary["forecasts"]) == {"honest", "extended"} and summary["errors"] == []
    assert summary["error_band"] == []
    status = store.latest_status(DAY)
    assert set(status.index) == {"load_forecast", "wind_solar_forecast", "prices", "weather_issued", "wind_points", "wind_proxy",
                                 "solar_points", "solar_proxy", "nuclear_actual", "neighbour_prices"}
    assert status.loc["wind_proxy", "status"] == "present" and status.loc["wind_proxy", "hours"] == 24
    assert status.loc["solar_proxy", "status"] == "present" and status.loc["solar_proxy", "hours"] == 24
    assert status.loc["load_forecast", "status"] == "present" and status.loc["load_forecast", "hours"] == 24
    assert status.loc["load_forecast", "revision"] == "3"
    log = store.timing_log().set_index("item")
    assert log.loc["load_forecast", "minutes_before_gate"] == 150.0  # 09:30 against the 12:00 gate
    honest_meta, honest = store.latest_forecast(DAY, "honest")
    extended_meta, extended = store.latest_forecast(DAY, "extended")
    assert len(honest) == 24 and len(extended) == 24 and honest["naive_day"].notna().all()
    assert honest_meta["passes_gate"] == 1 and extended_meta["passes_gate"] == 0
    assert honest_meta["issued_at_utc"] == "2024-04-09T07:30:00Z" and honest_meta["kind"] == "test"
    inputs = store.latest_input_values(DAY)
    assert len(inputs) == 24 and inputs["load_fc_mw"].notna().all() and inputs["wind_proxy_mw"].notna().all()
    # the proxies applied to the fresh point forecasts track the synthetic generation
    actual = (table.loc[inputs.index, "wind_onshore_mw"] + table.loc[inputs.index, "wind_offshore_mw"])
    assert (inputs["wind_proxy_mw"] - actual).abs().mean() < 1500
    assert inputs["solar_proxy_mw"].notna().all() and (inputs["solar_proxy_mw"] - table.loc[inputs.index, "solar_mw"]).abs().mean() < 400
    runs = store.runs()
    assert runs.iloc[0]["status"] == "ok" and runs.iloc[0]["delivery_day"] == DAY


def test_second_run_adds_a_version_and_counts_changes(store, table, history, tmp_path, weights, solar_weights):
    run(store, table, history, tmp_path, weights)
    bumped = table.copy()
    bumped.loc[day_hours(DAY), "load_fc_mw"] += 1000.0
    later = NOW + pd.Timedelta(hours=1)
    run(store, table, history, tmp_path, weights, entsoe=FakeEntsoe(bumped), now=later)
    versions = store.forecast_versions(DAY, "honest")
    assert versions["issued_at_utc"].tolist() == ["2024-04-09T07:30:00Z", "2024-04-09T08:30:00Z"]
    log = store.timing_log().set_index("item")
    assert log.loc["load_forecast", "changes"] == 1 and log.loc["load_forecast", "first_seen_utc"] == "2024-04-09T07:30:00Z"
    assert log.loc["prices", "changes"] == 0 and log.loc["load_forecast", "checks"] == 2


def test_wind_solar_down_gives_partial_run_with_honest_forecast_only(store, table, history, tmp_path, weights, solar_weights):
    summary = run(store, table, history, tmp_path, weights, entsoe=FakeEntsoe(table, down=("wind_solar_forecast",)))
    assert summary["status"] == "partial" and list(summary["forecasts"]) == ["honest"]
    status = store.latest_status(DAY)
    assert status.loc["wind_solar_forecast", "status"] == "error" and "unavailable" in status.loc["wind_solar_forecast", "message"]
    assert status.loc["forecast_extended", "status"] == "error"
    assert "wind_solar_forecast" not in store.timing_log()["item"].tolist()
    assert "wind_solar_forecast" in store.runs().iloc[0]["message"]


def test_everything_down_is_a_failed_run_not_a_crash(store, table, history, tmp_path, weights, solar_weights):
    entsoe = FakeEntsoe(table, down=("load_forecast", "wind_solar_forecast", "prices", "nuclear_actual", "neighbour_prices"))
    summary = run(store, table, history, tmp_path, weights, entsoe=entsoe, weather=FakeWeather(table, down=True),
                  wind_points=FakeWindPoints(table, down=True), solar_points=FakeSolarPoints(table, down=True))
    assert summary["status"] == "failed" and summary["forecasts"] == {}
    assert store.runs().iloc[0]["status"] == "failed"
    assert (store.latest_status(DAY)["status"] == "error").all()
    assert "no wind point forecasts" in store.latest_status(DAY).loc["wind_proxy", "message"]
    assert "no solar point forecasts" in store.latest_status(DAY).loc["solar_proxy", "message"]


def test_missing_proxy_weights_are_recorded_and_the_forecast_still_runs(store, table, history, tmp_path, solar_weights):
    summary = run(store, table, history, tmp_path, tmp_path / "absent.json", feature_sets=("honest",))
    assert summary["status"] == "partial" and list(summary["forecasts"]) == ["honest"]
    status = store.latest_status(DAY)
    assert status.loc["wind_points", "status"] == "present" and status.loc["solar_proxy", "status"] == "present"
    assert status.loc["wind_proxy", "status"] == "error" and "refit-model" in status.loc["wind_proxy", "message"]
    inputs = store.latest_input_values(DAY)
    assert inputs["wind_proxy_mw"].isna().all() and inputs["solar_proxy_mw"].notna().all()


def test_a_missing_model_file_is_recorded_and_the_forecast_is_fitted_live(store, table, history, tmp_path, weights, solar_weights):
    summary = run(store, table, history, tmp_path, weights, feature_sets=("honest",), model_file=tmp_path / "model" / "absent.txt")
    assert summary["status"] == "partial" and list(summary["forecasts"]) == ["honest"]
    status = store.latest_status(DAY)
    assert status.loc["model", "status"] == "error" and "refit-model" in status.loc["model", "message"]
    meta, _ = store.latest_forecast(DAY, "honest")
    assert meta["model"] == "gbm" and meta["train_hours"] > 0


def test_missing_load_forecast_for_the_day_is_recorded_as_absent(store, table, history, tmp_path, weights, solar_weights):
    summary = run(store, table, history, tmp_path, weights, entsoe=FakeEntsoe(table, hide=DAY))
    assert summary["status"] == "failed"
    status = store.latest_status(DAY)
    assert status.loc["load_forecast", "status"] == "absent" and status.loc["prices", "status"] == "absent"
    assert "not available" in status.loc["forecast_honest", "message"]


def test_error_band_is_refreshed_from_backtest_predictions(store, table, history, tmp_path, weights, solar_weights):
    reports = tmp_path / "reports"
    reports.mkdir()
    hours = list(range(24)) * 20
    pd.DataFrame({"hour": hours, "actual": 50.0, "gbm": [50.0 + (h - 12) / 4 for h in hours], "strict": True}).to_csv(reports / "predictions_honest.csv")
    summary = morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table), now=NOW,
                          reports_dir=reports, wind_points=FakeWindPoints(table), wind_weights=weights, solar_points=FakeSolarPoints(table),
                          solar_weights=solar_weights)
    assert summary["error_band"] == ["honest"]
    band = store.error_band("honest")
    assert len(band) == 24 and band.loc[0, "p50"] == pytest.approx(-3.0) and store.error_band("extended").empty


def test_settle_scores_every_version_against_actuals_and_benchmark(store, table, history, tmp_path, weights, solar_weights):
    entsoe = FakeEntsoe(table)
    run(store, table, history, tmp_path, weights, entsoe=entsoe)
    run(store, table, history, tmp_path, weights, entsoe=entsoe, now=NOW + pd.Timedelta(hours=2))
    settled = settle(store, entsoe=entsoe, now=pd.Timestamp("2024-04-09T13:00Z"), kind="test")
    assert settled["status"] == "ok" and settled["scored"] == 4
    assert set(settled["actual_days"]) >= {DAY, "2024-04-09"}
    scores = store.scores()
    assert len(scores) == 4 and scores["hours"].eq(24).all() and scores["naive_mae"].notna().all()
    assert set(scores["feature_set"]) == {"honest", "extended"}
    assert store.unscored_forecasts().empty
    again = settle(store, entsoe=entsoe, now=pd.Timestamp("2024-04-09T14:00Z"), kind="test")
    assert again["scored"] == 0
    status = store.latest_status(DAY)
    assert status.loc["prices", "status"] == "present"


def test_settle_with_prices_down_is_partial_or_failed(store, table, history, tmp_path, weights, solar_weights):
    run(store, table, history, tmp_path, weights)
    settled = settle(store, entsoe=FakeEntsoe(table, down=("prices",)), now=pd.Timestamp("2024-04-09T13:00Z"), kind="test")
    assert settled["status"] == "failed" and settled["scored"] == 0 and settled["errors"]
    assert store.latest_status(DAY).loc["prices", "status"] == "error"


def test_v2_model_falls_back_to_the_honest_model_when_its_inputs_are_missing(store, table, history, tmp_path, weights, solar_weights):
    before = table[table.index < "2024-04-01"]
    refit_model(before, out_dir=tmp_path / "model", feature_set="honest_v2", holdout_days=0, log=lambda *_: None, params={"n_estimators": 20})
    refit_model(before, out_dir=tmp_path / "model", feature_set="honest", holdout_days=0, log=lambda *_: None, params={"n_estimators": 20})
    v2, honest = tmp_path / "model" / "honest_v2.txt", tmp_path / "model" / "honest.txt"
    # every input arrives: the v2 model predicts
    summary = run(store, table, history, tmp_path, weights, feature_sets=("honest_v2",), model_file=v2, fallback_model_file=honest)
    versions = store.forecast_versions(DAY, "honest_v2")
    assert summary["status"] == "ok" and versions["model"].tolist() == ["gbm:honest_v2.txt"]
    status = store.status_for(DAY)
    assert {"nuclear_actual", "neighbour_prices"} <= set(status["item"]) and "fallback" not in set(status["status"])
    # nuclear generation does not arrive: the day's nuclear features are all missing, the honest model takes over, the run is partial
    entsoe = FakeEntsoe(table, down=("nuclear_actual",))
    summary = run(store, table, history, tmp_path, weights, entsoe=entsoe, feature_sets=("honest_v2",), model_file=v2, fallback_model_file=honest, now=NOW + pd.Timedelta(hours=1))
    versions = store.forecast_versions(DAY, "honest_v2")
    assert summary["status"] == "partial" and versions["model"].tolist()[-1] == "gbm:honest.txt:fallback"
    rows = store.status_for(DAY)
    fallback = rows[(rows["item"] == "model") & (rows["status"] == "fallback")]
    assert len(fallback) == 1 and "nuclear_d1_early_mw" in fallback.iloc[0]["message"] and "honest.txt" in fallback.iloc[0]["message"]
    assert "model: honest_v2.txt needs" in store.runs(limit=1).iloc[0]["message"]
    # the same failure without a usable fallback produces no v2 forecast and an error, never a prediction with missing features
    summary = run(store, table, history, tmp_path, weights, entsoe=entsoe, feature_sets=("honest_v2",), model_file=v2, fallback_model_file=None, now=NOW + pd.Timedelta(hours=2))
    assert summary["status"] == "failed" and any("inputs missing" in e for e in summary["errors"])
    assert len(store.forecast_versions(DAY, "honest_v2")) == 2


def test_missing_features_names_the_columns_without_values_on_the_day(table):
    assert missing_features(DAY, table, "honest_v2") == []
    blank = table.copy()
    blank.loc[blank.index >= "2024-04-07", "nuclear_mw"] = float("nan")
    assert set(missing_features(DAY, blank, "honest_v2")) == {"nuclear_d2_mw", "nuclear_d2_mean_mw", "nuclear_d1_early_mw", "residual_v2_mw"}
    assert missing_features(DAY, blank, "honest") == []
    assert missing_features("2030-01-01", table, "honest") == list(FEATURES_HONEST)


def test_setup_logging_writes_a_file_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTSOE_API_KEY", "SECRET-TOKEN")
    path = setup_logging("morning-run", logs_dir=tmp_path / "logs")
    logging.getLogger("thermo_fr.jobs").info("hello from the job")
    for handler in logging.getLogger("thermo_fr").handlers:
        handler.flush()
    text = path.read_text(encoding="utf-8")
    assert "hello from the job" in text and "SECRET" not in text and path.name.startswith("morning-run_")
