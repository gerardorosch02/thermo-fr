"""A version's issue time is the run's actual time, never a scheduled time, so a delayed run cannot pass as pre-market."""

import pandas as pd

from test_jobs import run  # noqa: F401  (the fixtures table, history, weights, store come with it)
from test_jobs import DAY, FakeEntsoe, FakeWeather, FakeSolarPoints, FakeWindPoints, table, history, weights, solar_weights, store  # noqa: F401
from test_store import curve_for
from thermo_fr.forecast.jobs import morning_run
from thermo_fr.forecast.market import headline_version, premarket_flag
from thermo_fr.forecast.store import Store


def test_issue_time_is_the_wall_clock_when_no_time_is_given(store, table, history, tmp_path, weights, solar_weights):
    before = pd.Timestamp.now(tz="UTC")
    summary = morning_run(store, DAY, kind="scheduled", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table), now=None,
                          reports_dir=tmp_path / "none", wind_points=FakeWindPoints(table), wind_weights=weights,
                          solar_points=FakeSolarPoints(table), solar_weights=solar_weights, feature_sets=("honest",))
    after = pd.Timestamp.now(tz="UTC")
    assert summary["forecasts"], summary["errors"]
    versions = store.forecast_versions(DAY, "honest")
    issued = pd.Timestamp(versions.iloc[-1]["issued_at_utc"])
    assert before - pd.Timedelta(seconds=1) <= issued <= after + pd.Timedelta(seconds=1)
    # the run row carries the same clock
    started = pd.Timestamp(store.runs(limit=1).iloc[0]["started_at_utc"])
    assert abs((started - issued).total_seconds()) < 120
    # and a 2024 delivery day issued now is, of course, not a pre-market version: the flag follows the issue time, not the kind
    assert not premarket_flag([DAY], [issued])[0]


def test_a_late_version_never_replaces_a_pre_market_one(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    day = "2026-10-13"  # Tuesday; the window opens 11:15 Paris on 2026-10-12, 09:15Z
    s.save_forecast(1, day, "honest_v2", pd.Timestamp("2026-10-12T08:10Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve_for(day, 50.0))
    s.save_forecast(2, day, "honest_v2", pd.Timestamp("2026-10-12T10:20Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve_for(day, 70.0))
    picked = headline_version(s.forecast_versions(day, "honest_v2"))
    assert len(picked) == 1 and int(picked.iloc[0]["forecast_id"]) == 1 and bool(picked.iloc[0]["premarket"])
    # a day with only a late version shows that version, flagged as issued after the window
    other = "2026-10-14"
    s.save_forecast(3, other, "honest_v2", pd.Timestamp("2026-10-13T12:05Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve_for(other))
    picked = headline_version(s.forecast_versions(other, "honest_v2"))
    assert int(picked.iloc[0]["forecast_id"]) == 3 and not bool(picked.iloc[0]["premarket"])
    s.close()
