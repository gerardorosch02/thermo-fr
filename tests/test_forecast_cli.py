"""The forecast commands end to end on a synthetic inputs table, offline."""

import json

import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from thermo_fr import cli
from thermo_fr.forecast import models
from thermo_fr.forecast.day import forecast_day


@pytest.fixture
def inputs_file(tmp_path, monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 40)
    frame = synthetic_inputs("2023-09-01", "2024-05-01", issued_from="2024-01-10")
    data = tmp_path / "forecast"
    data.mkdir()
    frame.to_csv(data / "inputs.csv", index_label="timestamp_utc")
    (data / "sources.json").write_text(json.dumps({"note": "synthetic"}))
    return data / "inputs.csv"


def test_backtest_command_writes_the_report(inputs_file, tmp_path, capsys):
    out = tmp_path / "reports"
    cli.main([
        "forecast-backtest", "--data", str(inputs_file), "--out", str(out),
        "--test-start", "2024-02-01", "--test-end", "2024-04-01", "--sample-week", "2024-02-05",
    ])
    names = {p.name for p in out.iterdir()}
    assert {"report.md", "summary.json", "sample_week.png", "mae_by_hour_honest.png", "mae_by_month_extended.png",
            "predictions_honest.csv", "worst_days_extended.csv"} <= names
    summary = json.loads((out / "summary.json").read_text())
    assert summary["honest"]["months"] == ["2024-02", "2024-03"]
    assert summary["honest"]["metrics_strict"]["rows"] > 0 and summary["sources"] == {"note": "synthetic"}
    assert summary["honest"]["point_in_time"] is True and summary["extended"]["point_in_time"] is False
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "| honest |" in text and "| extended (may use late information) |" in text
    assert "twenty worst forecast days" in text
    assert "Day-ahead price forecast" not in capsys.readouterr().out  # that line belongs to the forecast command


def test_forecast_command_uses_stored_inputs_without_refresh(inputs_file, tmp_path, capsys):
    out = tmp_path / "reports"
    cli.main(["forecast", "--date", "2024-03-15", "--data", str(inputs_file), "--out", str(out), "--no-refresh"])
    curve = pd.read_csv(out / "day_2024-03-15.csv")
    assert len(curve) == 24 and curve["hour"].tolist() == list(range(24))
    assert curve["forecast"].notna().all() and curve["actual"].notna().all()
    assert (out / "day_2024-03-15.png").exists()
    printed = capsys.readouterr().out
    assert "information as of 12:00 Paris the day before" in printed


def test_forecast_refuses_days_without_issued_weather_or_load_forecast(inputs_file):
    inputs = pd.read_csv(inputs_file, index_col=0)
    inputs.index = pd.to_datetime(inputs.index, utc=True)
    with pytest.raises(ValueError, match="as issued are missing"):
        forecast_day("2023-12-15", inputs, log=lambda *_: None)
    gone = inputs.copy()
    day = gone.index.tz_convert("Europe/Paris").normalize().tz_localize(None) == pd.Timestamp("2024-03-15")
    gone.loc[day, "load_fc_mw"] = float("nan")
    with pytest.raises(ValueError, match="load forecast"):
        forecast_day("2024-03-15", gone, log=lambda *_: None)
    with pytest.raises(ValueError, match="No input hours"):
        forecast_day("2024-09-01", inputs, log=lambda *_: None)


def test_morning_run_defaults_to_the_published_model_and_refit_opts_out(monkeypatch, tmp_path):
    from pathlib import Path

    from thermo_fr.forecast import jobs

    calls = []

    def fake_run(store, date, **kw):
        calls.append(kw)
        return {"status": "ok", "delivery_day": "2026-10-10", "forecasts": {"honest": 1}, "errors": []}

    monkeypatch.setattr(jobs, "morning_run", fake_run)
    monkeypatch.setattr(jobs, "setup_logging", lambda *a, **k: None)
    db = str(tmp_path / "db.sqlite")
    cli.main(["morning-run", "--db", db, "--logs-dir", str(tmp_path)])
    cli.main(["morning-run", "--db", db, "--logs-dir", str(tmp_path), "--refit"])
    cli.main(["morning-run", "--db", db, "--logs-dir", str(tmp_path), "--model-file", "elsewhere/model.txt"])
    assert calls[0]["model_file"] == Path("published/model/honest.txt")  # the file the GitHub workflow predicts with
    assert calls[0]["solar_weights"] == Path("published/model/solar_proxy.json") and calls[0]["wind_weights"] == Path("published/model/wind_proxy.json")
    assert calls[1]["model_file"] is None
    assert calls[2]["model_file"] == Path("elsewhere/model.txt")


def test_forecast_merges_a_fresh_window_over_the_stored_table(inputs_file):
    inputs = pd.read_csv(inputs_file, index_col=0)
    inputs.index = pd.to_datetime(inputs.index, utc=True)
    history = inputs[inputs.index < "2024-03-10"]
    fresh = inputs[(inputs.index >= "2024-03-05") & (inputs.index < "2024-03-17")].copy()
    fresh.loc[fresh.index >= "2024-03-14T23:00Z", "price_eur_mwh"] = float("nan")  # the day's price is not known yet
    curve = forecast_day("2024-03-15", history, fresh, log=lambda *_: None)
    assert len(curve) == 24 and curve["actual"].isna().all() and curve["forecast"].notna().all()
