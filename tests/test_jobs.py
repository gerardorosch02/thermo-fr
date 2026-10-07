"""morning-run and settle, offline: graceful failures, status rows, versions, settlement scoring."""

import logging

import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from thermo_fr.forecast import models
from thermo_fr.forecast.jobs import day_hours, morning_run, next_delivery_day, presence, settle, setup_logging
from thermo_fr.forecast.store import Store

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


class FakeWeather:
    def __init__(self, table, down=False):
        self.table, self.down = table, down

    def fetch(self, start, end, kinds=("issued",)):
        if self.down:
            raise TimeoutError("open-meteo timed out")
        part = self.table.loc[start:end, ["temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"]]
        return part[part.index < pd.Timestamp(end, tz="UTC")]


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    return synthetic_inputs("2023-09-01", "2024-05-01", issued_from="2024-01-10")


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


def test_morning_run_stores_both_feature_sets_and_the_timing_log(store, table, history, tmp_path):
    entsoe = FakeEntsoe(table, revision="3")
    summary = morning_run(store, DAY, kind="test", inputs_path=history, entsoe=entsoe, weather=FakeWeather(table), now=NOW,
                          reports_dir=tmp_path / "none")
    assert summary["status"] == "ok" and set(summary["forecasts"]) == {"honest", "extended"} and summary["errors"] == []
    assert summary["error_band"] == []
    status = store.latest_status(DAY)
    assert set(status.index) == {"load_forecast", "wind_solar_forecast", "prices", "weather_issued"}
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
    assert len(inputs) == 24 and inputs["load_fc_mw"].notna().all()
    runs = store.runs()
    assert runs.iloc[0]["status"] == "ok" and runs.iloc[0]["delivery_day"] == DAY


def test_second_run_adds_a_version_and_counts_changes(store, table, history, tmp_path):
    morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table), now=NOW,
                reports_dir=tmp_path / "none")
    bumped = table.copy()
    bumped.loc[day_hours(DAY), "load_fc_mw"] += 1000.0
    later = NOW + pd.Timedelta(hours=1)
    morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(bumped), weather=FakeWeather(table), now=later,
                reports_dir=tmp_path / "none")
    versions = store.forecast_versions(DAY, "honest")
    assert versions["issued_at_utc"].tolist() == ["2024-04-09T07:30:00Z", "2024-04-09T08:30:00Z"]
    log = store.timing_log().set_index("item")
    assert log.loc["load_forecast", "changes"] == 1 and log.loc["load_forecast", "first_seen_utc"] == "2024-04-09T07:30:00Z"
    assert log.loc["prices", "changes"] == 0 and log.loc["load_forecast", "checks"] == 2


def test_wind_solar_down_gives_partial_run_with_honest_forecast_only(store, table, history, tmp_path):
    summary = morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table, down=("wind_solar_forecast",)),
                          weather=FakeWeather(table), now=NOW, reports_dir=tmp_path / "none")
    assert summary["status"] == "partial" and list(summary["forecasts"]) == ["honest"]
    status = store.latest_status(DAY)
    assert status.loc["wind_solar_forecast", "status"] == "error" and "unavailable" in status.loc["wind_solar_forecast", "message"]
    assert status.loc["forecast_extended", "status"] == "error"
    assert "wind_solar_forecast" not in store.timing_log()["item"].tolist()
    assert "wind_solar_forecast" in store.runs().iloc[0]["message"]


def test_everything_down_is_a_failed_run_not_a_crash(store, table, history, tmp_path):
    entsoe = FakeEntsoe(table, down=("load_forecast", "wind_solar_forecast", "prices"))
    summary = morning_run(store, DAY, kind="test", inputs_path=history, entsoe=entsoe, weather=FakeWeather(table, down=True), now=NOW,
                          reports_dir=tmp_path / "none")
    assert summary["status"] == "failed" and summary["forecasts"] == {}
    assert store.runs().iloc[0]["status"] == "failed"
    assert (store.latest_status(DAY)["status"] == "error").all()


def test_missing_load_forecast_for_the_day_is_recorded_as_absent(store, table, history, tmp_path):
    summary = morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table, hide=DAY), weather=FakeWeather(table),
                          now=NOW, reports_dir=tmp_path / "none")
    assert summary["status"] == "failed"
    status = store.latest_status(DAY)
    assert status.loc["load_forecast", "status"] == "absent" and status.loc["prices", "status"] == "absent"
    assert "not available" in status.loc["forecast_honest", "message"]


def test_error_band_is_refreshed_from_backtest_predictions(store, table, history, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    hours = list(range(24)) * 20
    pd.DataFrame({"hour": hours, "actual": 50.0, "gbm": [50.0 + (h - 12) / 4 for h in hours], "strict": True}).to_csv(reports / "predictions_honest.csv")
    summary = morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table), now=NOW,
                          reports_dir=reports)
    assert summary["error_band"] == ["honest"]
    band = store.error_band("honest")
    assert len(band) == 24 and band.loc[0, "p50"] == pytest.approx(-3.0) and store.error_band("extended").empty


def test_settle_scores_every_version_against_actuals_and_benchmark(store, table, history, tmp_path):
    entsoe = FakeEntsoe(table)
    morning_run(store, DAY, kind="test", inputs_path=history, entsoe=entsoe, weather=FakeWeather(table), now=NOW, reports_dir=tmp_path / "none")
    morning_run(store, DAY, kind="test", inputs_path=history, entsoe=entsoe, weather=FakeWeather(table), now=NOW + pd.Timedelta(hours=2),
                reports_dir=tmp_path / "none")
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


def test_settle_with_prices_down_is_partial_or_failed(store, table, history, tmp_path):
    morning_run(store, DAY, kind="test", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table), now=NOW,
                reports_dir=tmp_path / "none")
    settled = settle(store, entsoe=FakeEntsoe(table, down=("prices",)), now=pd.Timestamp("2024-04-09T13:00Z"), kind="test")
    assert settled["status"] == "failed" and settled["scored"] == 0 and settled["errors"]
    assert store.latest_status(DAY).loc["prices", "status"] == "error"


def test_setup_logging_writes_a_file_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTSOE_API_KEY", "SECRET-TOKEN")
    path = setup_logging("morning-run", logs_dir=tmp_path / "logs")
    logging.getLogger("thermo_fr.jobs").info("hello from the job")
    for handler in logging.getLogger("thermo_fr").handlers:
        handler.flush()
    text = path.read_text(encoding="utf-8")
    assert "hello from the job" in text and "SECRET" not in text and path.name.startswith("morning-run_")
