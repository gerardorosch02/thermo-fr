"""The GitHub Actions path after a settle: versions imported from the public dataset carry no same-hour-previous-day values and no
quantile or spike columns (None in the store), and run #10 of 2026-10-12 crashed on them in the publish step. A second settle of the
same day changes nothing, and the workflow commits nothing when only the stamps in status.json moved."""

import json

import numpy as np
import pandas as pd
import pytest

from test_store import curve_for
from thermo_fr import cli
from thermo_fr.forecast.gitsync import _git
from thermo_fr.forecast.jobs import settle
from thermo_fr.forecast.probabilistic import live_calibration
from thermo_fr.forecast.publish import export_published, import_published, staged_changes_are_stamps_only, status_equivalent
from thermo_fr.forecast.shape import battery_backtest, live_frame, live_shape_record, previous_day_actual
from thermo_fr.forecast.store import Store, score_curve

DAY, PREVIOUS = "2026-10-13", "2026-10-12"


def imported_version(store, day=DAY, forecast_id=1, issued="2026-10-12T08:30Z"):
    """A version as import_published stores it: no naive_day, probabilistic rows with the two negative-price columns only."""
    curve = curve_for(day).drop(columns=["naive_day"])
    store.save_forecast(forecast_id, day, "honest_v2", pd.Timestamp(issued), "gbm:honest_v2.txt", "scheduled", 0, True, curve)
    prob = pd.DataFrame({"p_negative": 0.01, "p_negative_cal": 0.05}, index=curve.index)
    store.save_probabilistic(forecast_id, prob)
    return curve


def settled_day(store, curve, day=DAY, forecast_id=1):
    actual = pd.Series(90.0 + curve["hour"].to_numpy() * 1.5, index=curve.index)
    store.save_actuals(actual, pd.Series([day] * len(curve)))
    store.save_score(forecast_id, day, "honest_v2", "2026-10-12T08:30:00Z", score_curve(store.forecast_curve(forecast_id), actual))
    return actual


def test_live_frame_takes_the_d1_reference_from_the_previous_day_when_the_version_has_none(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    curve = imported_version(store)
    actual = settled_day(store, curve)
    assert store.forecast_curve(1)["naive_day"].isna().all()
    # the previous day is not settled yet: no D-1 value, no crash
    record = live_shape_record(store, "honest_v2")
    assert record["days"] == 1 and record["battery"]["model"]["days_traded"] == 1
    assert np.isnan(record["daily"].loc[0, "value_d1"]) and not record["daily"].loc[0, "traded_d1"]
    assert previous_day_actual(store, DAY) is None
    # yesterday's auction prices stored: the D-1 shape is yesterday's, moved forward by 24 hours
    previous_curve = curve_for(PREVIOUS)
    previous_actual = pd.Series(80.0 + previous_curve["hour"].to_numpy() * 3.0, index=previous_curve.index)
    store.save_actuals(previous_actual, pd.Series([PREVIOUS] * len(previous_curve)))
    d1 = previous_day_actual(store, DAY)
    frame = live_frame(store.forecast_curve(1), actual, DAY, d1=d1)
    expected = previous_actual.to_numpy() - previous_actual.mean()
    assert np.allclose(frame["shape_d1"].to_numpy(), expected) and frame["base_d1"].iloc[0] == pytest.approx(previous_actual.mean())
    record = live_shape_record(store, "honest_v2")
    assert record["daily"].loc[0, "traded_d1"] and record["shape"]["d1"]["shape_mae"] > 0
    # a version that carries its own naive_day keeps it, whatever d1 says
    own = live_frame(curve_for(DAY), actual, DAY, d1=d1)
    assert np.allclose(own["base_d1"], curve_for(DAY)["naive_day"].mean())
    store.close()


def test_battery_backtest_treats_an_object_column_of_none_as_missing():
    index = curve_for(DAY).index
    frame = pd.DataFrame({"delivery_day": pd.Timestamp(DAY), "actual": 50.0 + np.arange(24), "base_model": 60.0, "shape_model": np.arange(24) - 11.5,
                          "base_d1": pd.Series([None] * 24, index=index, dtype=object), "shape_d1": pd.Series([None] * 24, index=index, dtype=object)},
                         index=index)
    result = battery_backtest(frame, ("model", "d1"))
    assert result["daily"].loc[0, "traded_model"] and not result["daily"].loc[0, "traded_d1"] and np.isnan(result["daily"].loc[0, "value_d1"])


def test_live_calibration_accepts_imported_rows_without_quantiles_or_spike_columns(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    curve = imported_version(store)
    settled_day(store, curve)
    settled = store.probabilistic_settled("honest_v2")
    assert settled["spike_threshold"].isna().all() and settled["q10"].isna().all()
    live = live_calibration(settled)
    assert live["days"] == 1 and live["hours"] == 24 and live["negative"]["events"] == 0 and live["negative_calibrated_hours"] == 24
    assert "spike" not in live and "coverage_10_90" not in live
    store.close()


def test_export_after_a_settle_on_an_imported_store_succeeds(tmp_path):
    """The whole publish step of run #10, on a store built the way the workflow builds it."""
    store = Store(tmp_path / "db.sqlite")
    curve = imported_version(store)
    settled_day(store, curve)
    out = tmp_path / "published"
    status = export_published(store, out, days=90, now=pd.Timestamp("2026-10-12T21:44:50Z"), market_path=tmp_path / "no_market.csv")
    assert status["shape_battery"]["days"] == 1 and status["probabilistic"]["live"]["days"] == 1
    live = json.loads((out / "shape_battery.json").read_text())["live"]
    assert live["daily"][0]["traded_model"] is True and pd.isna(live["daily"][0]["value_d1"])  # no previous day in the store: no D-1 value
    store.close()


class PriceSource:
    """Serves hourly day-ahead prices for the settle, like the ENTSO-E client (prices only)."""

    def __init__(self, prices: pd.Series):
        self.prices, self.details, self.calls = prices, {}, 0

    def day_ahead_prices(self, start, end):
        self.calls += 1
        self.details["prices"] = [{"revision": "1"}]
        return self.prices[(self.prices.index >= pd.Timestamp(start, tz="UTC")) & (self.prices.index < pd.Timestamp(end, tz="UTC"))]


def test_a_second_settle_of_the_same_day_changes_nothing_but_the_stamps(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    curve = imported_version(store)
    index = pd.date_range(pd.Timestamp("2026-10-11", tz="Europe/Paris"), pd.Timestamp("2026-10-14", tz="Europe/Paris"), freq="1h",
                          inclusive="left").tz_convert("UTC")
    source = PriceSource(pd.Series(60.0 + (np.arange(len(index)) % 24) * 1.5, index=index, name="price_eur_mwh"))
    first = settle(store, entsoe=source, now=pd.Timestamp("2026-10-13T13:20Z"), kind="test")
    assert first["status"] == "ok" and first["scored"] == 1 and DAY in first["actual_days"]
    out1, out2 = tmp_path / "first", tmp_path / "second"
    status1 = export_published(store, out1, days=90, now=pd.Timestamp("2026-10-13T13:21Z"), market_path=tmp_path / "no_market.csv")
    second = settle(store, entsoe=source, now=pd.Timestamp("2026-10-13T14:21Z"), kind="test")  # the backup cron, an hour later
    assert second["status"] == "ok" and second["scored"] == 0 and second["actual_days"] == first["actual_days"]
    status2 = export_published(store, out2, days=90, now=pd.Timestamp("2026-10-13T14:22Z"), market_path=tmp_path / "no_market.csv")
    for name in ("forecasts.csv", "actuals.csv", "scores.csv", "tomorrow.csv", "probabilistic.csv", "shape_battery.json"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
    assert len(store.scores()) == 1 and store.actuals_for(DAY).shape == (24,)
    assert status1 != status2 and status_equivalent(status1, status2)
    store.close()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "published").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    status = {"written_at_utc": "2026-10-13T13:21:00Z", "last_run": {"command": "settle", "started_at_utc": "2026-10-13T13:20:00Z"},
              "actual_days": 6, "live_record": {"days": 1}}
    (repo / "published" / "status.json").write_text(json.dumps(status))
    (repo / "published" / "actuals.csv").write_text("delivery_day,price\n2026-10-13,50\n")
    _git(repo, "add", "published")
    _git(repo, "commit", "-q", "-m", "first")
    return repo, status


def test_staged_changes_are_stamps_only_and_the_cli_exit_code(tmp_path, monkeypatch, capsys):
    repo, status = _repo(tmp_path)
    # nothing staged: not an unchanged publish (the workflow's own quiet check handles that)
    assert staged_changes_are_stamps_only(repo) == (False, "staged: nothing")
    # only the stamps moved
    moved = {**status, "written_at_utc": "2026-10-13T14:22:00Z", "last_run": {"command": "settle", "started_at_utc": "2026-10-13T14:21:00Z"}}
    (repo / "published" / "status.json").write_text(json.dumps(moved))
    _git(repo, "add", "published")
    unchanged, why = staged_changes_are_stamps_only(repo)
    assert unchanged and "only the stamps" in why
    cli.main(["publish-unchanged", "--repo", str(repo), "--out", "published"])  # exit 0: the workflow unstages and commits nothing
    assert "nothing new to publish" in capsys.readouterr().out
    # a data file changed as well: a real publish
    (repo / "published" / "actuals.csv").write_text("delivery_day,price\n2026-10-13,50\n2026-10-14,40\n")
    _git(repo, "add", "published")
    unchanged, why = staged_changes_are_stamps_only(repo)
    assert not unchanged and "actuals.csv" in why
    with pytest.raises(SystemExit) as exc:
        cli.main(["publish-unchanged", "--repo", str(repo), "--out", "published"])
    assert exc.value.code == 1
    # status.json changed beyond its stamps (a new settled day in the live record): a real publish too
    _git(repo, "reset", "-q", "--hard")
    (repo / "published" / "status.json").write_text(json.dumps({**moved, "live_record": {"days": 2}}))
    _git(repo, "add", "published")
    assert staged_changes_are_stamps_only(repo) == (False, "published/status.json changed beyond its stamps")


def test_import_then_export_leaves_the_published_files_byte_for_byte(tmp_path):
    """Run #12 of 2026-10-13 re-committed scores.csv with an rmse one unit in the last place off: the default CSV float parser is inexact."""
    store = Store(tmp_path / "db.sqlite")
    curve = imported_version(store)
    actual = pd.Series([90.0 + h * 1.5 + (0.1 if h % 7 == 0 else 0.0) for h in curve["hour"]], index=curve.index)  # non-terminating errors
    store.save_actuals(actual, pd.Series([DAY] * len(curve)))
    store.save_score(1, DAY, "honest_v2", "2026-10-12T08:30:00Z", {"hours": 24, "mae": 16.345312500000002, "rmse": 22.771532775549314,
                                                                      "naive_mae": None, "naive_rmse": None, "mae_below_baseline": None})
    out = tmp_path / "published"
    export_published(store, out, days=90, now=pd.Timestamp("2026-10-13T06:18Z"), market_path=tmp_path / "no_market.csv")
    store.close()
    fresh = Store(tmp_path / "fresh.sqlite")
    import_published(fresh, out)
    again = tmp_path / "again"
    export_published(fresh, again, days=90, now=pd.Timestamp("2026-10-13T06:19Z"), market_path=tmp_path / "no_market.csv")
    fresh.close()
    for name in ("forecasts.csv", "actuals.csv", "scores.csv", "probabilistic.csv", "tomorrow.csv", "error_band.csv"):
        assert (out / name).read_bytes() == (again / name).read_bytes(), name
    assert "22.771532775549314" in (again / "scores.csv").read_text()
