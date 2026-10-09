"""The published/ dataset: export, import into an empty store, and the history extension."""

import json

import pandas as pd
import pytest

from test_store import curve_for
from thermo_fr.forecast.publish import export_published, import_published
from thermo_fr.forecast.store import Store

NOW = pd.Timestamp("2026-10-09T12:00Z")
DAYS = ["2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10"]


def filled_store(path) -> Store:
    s = Store(path)
    for i, day in enumerate(DAYS):
        c = curve_for(day, 100 + i)
        first = s.save_forecast(1, day, "honest", pd.Timestamp(f"{DAYS[max(i - 1, 0)]}T06:00Z"), "gbm", "scheduled", 50, True, c)
        s.save_forecast(2, day, "honest", pd.Timestamp(f"{DAYS[max(i - 1, 0)]}T09:00Z"), "gbm", "scheduled", 50, True, curve_for(day, 105 + i))
        s.save_forecast(3, day, "extended", pd.Timestamp(f"{DAYS[max(i - 1, 0)]}T09:00Z"), "gbm", "scheduled", 50, False, c)
        if day <= "2026-10-09":
            actual = pd.Series(103.0 + c["hour"].to_numpy(), index=c.index)
            s.save_actuals(actual, pd.Series([day] * 24), now=NOW)
            s.save_score(first, day, "honest", f"{DAYS[max(i - 1, 0)]}T06:00:00Z",
                         {"hours": 24, "mae": 3.0, "rmse": 3.5, "naive_mae": 8.0, "naive_rmse": 9.0, "mae_below_baseline": True}, now=NOW)
    # a stale day outside the window
    old = curve_for("2026-06-01", 50)
    s.save_forecast(9, "2026-06-01", "honest", pd.Timestamp("2026-05-31T06:00Z"), "gbm", "scheduled", 50, True, old)
    band = pd.DataFrame({"feature_set": "honest", "hour": range(24), "n": 10, "mae": 5.0, "p10": -8.0, "p25": -3.0, "p50": 0.0, "p75": 3.0, "p90": 8.0})
    s.save_error_band(band, "test")
    return s


def test_export_writes_the_public_files_for_the_window(tmp_path):
    store = filled_store(tmp_path / "db.sqlite")
    out = tmp_path / "published"
    status = export_published(store, out, days=90, now=NOW)
    store.close()
    names = {p.name for p in out.iterdir()}
    assert names == {"forecasts.csv", "actuals.csv", "scores.csv", "tomorrow.csv", "error_band.csv", "status.json"}
    forecasts = pd.read_csv(out / "forecasts.csv")
    assert set(forecasts["delivery_day"]) == set(DAYS)  # June is outside the 90-day window, extended is never exported
    assert forecasts.groupby("delivery_day")["issued_at_utc"].nunique().eq(2).all()
    assert list(forecasts.columns) == ["delivery_day", "issued_at_utc", "kind", "train_hours", "timestamp_utc", "hour", "forecast", "naive_day"]
    tomorrow = pd.read_csv(out / "tomorrow.csv")
    assert tomorrow["delivery_day"].unique().tolist() == ["2026-10-10"] and tomorrow["issued_at_utc"].nunique() == 1
    assert tomorrow["forecast"].iloc[0] == 108.0  # the later (105 + 3) version
    assert len(pd.read_csv(out / "actuals.csv")) == 72 and len(pd.read_csv(out / "scores.csv")) == 3
    assert len(pd.read_csv(out / "error_band.csv")) == 24
    status_file = json.loads((out / "status.json").read_text())
    assert status_file == status and status["tomorrow"] == "2026-10-10" and status["forecast_versions"] == 8
    assert any("ENTSO-E" in a for a in status["attributions"]) and any("Open-Meteo" in a for a in status["attributions"])


def test_import_rebuilds_an_empty_store_and_is_idempotent(tmp_path):
    store = filled_store(tmp_path / "db.sqlite")
    out = tmp_path / "published"
    export_published(store, out, days=90, now=NOW)
    store.close()

    fresh = Store(tmp_path / "fresh.sqlite")
    summary = import_published(fresh, out)
    assert summary == {"forecast_versions": 8, "actual_rows": 72, "scores": 3, "error_band_rows": 24}
    meta, curve = fresh.latest_forecast("2026-10-10", "honest")
    assert meta["issued_at_utc"] == "2026-10-09T09:00:00Z" and curve["forecast"].iloc[0] == 108.0 and len(curve) == 24
    assert fresh.latest_scores("honest").shape[0] == 3 and len(fresh.error_band("honest")) == 24
    assert len(fresh.actuals_for("2026-10-08")) == 24
    again = import_published(fresh, out)
    assert again["forecast_versions"] == 0 and len(fresh.forecast_versions("2026-10-10", "honest")) == 2
    # exporting from the restored store gives the same public files
    again_out = tmp_path / "published2"
    export_published(fresh, again_out, days=90, now=NOW)
    for name in ("forecasts.csv", "actuals.csv", "scores.csv", "tomorrow.csv", "error_band.csv"):
        a, b = pd.read_csv(out / name), pd.read_csv(again_out / name)
        pd.testing.assert_frame_equal(a.sort_values(list(a.columns)).reset_index(drop=True), b.sort_values(list(b.columns)).reset_index(drop=True))
    fresh.close()


def test_import_tolerates_missing_or_empty_files(tmp_path):
    empty = tmp_path / "published"
    empty.mkdir()
    (empty / "forecasts.csv").write_text("")
    store = Store(tmp_path / "db.sqlite")
    assert import_published(store, empty) == {"forecast_versions": 0, "actual_rows": 0, "scores": 0, "error_band_rows": 0}
    store.close()
