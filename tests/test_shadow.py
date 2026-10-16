"""The laptop-only S2 shadow: live-format inputs, the publication-time gate, the 11:15 guard, fallbacks, settlement and the
statistics at day 40, and that nothing reaches published/ or git."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from test_jobs import FakeEntsoe, FakeSolarPoints, FakeWeather, FakeWindPoints
from thermo_fr.forecast import models, shadow, solar_proxy
from thermo_fr.forecast.jobs import day_hours, morning_run
from thermo_fr.forecast.store import Store, iso
from thermo_fr.forecast.wind_proxy import calibrate, save_weights

DAY = "2024-04-10"
MORNING = pd.Timestamp("2024-04-09T08:20Z")  # 10:20 Paris on D-1, inside the pre-market window
WINDOWS = (364,)


@pytest.fixture(scope="module")
def table():
    return synthetic_inputs("2022-09-01", "2024-05-01", issued_from="2024-01-10")


@pytest.fixture
def setup(table, tmp_path, monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    monkeypatch.chdir(tmp_path)  # anything written relative to the working directory would land here

    def guard(original):
        def wrapped(*args, **kwargs):
            command = args[0] if args else kwargs.get("args", "")
            text = " ".join(command) if isinstance(command, (list, tuple)) else str(command)
            if any(word in text.split() for word in ("git", "gh")):
                raise AssertionError(f"the shadow must never call git or gh: {text}")
            return original(*args, **kwargs)
        return wrapped

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, guard(getattr(subprocess, name)))
    history = tmp_path / "inputs.csv"
    table[table.index < "2024-04-01"].to_csv(history, index_label="timestamp_utc")
    before = table[table.index < "2024-04-01"]
    wind = save_weights(calibrate(before, before["wind_onshore_mw"] + before["wind_offshore_mw"]), tmp_path / "model" / "wind_proxy.json")
    solar = save_weights(solar_proxy.calibrate(before, before["solar_mw"]), tmp_path / "model" / "solar_proxy.json")
    store = Store(tmp_path / "forecast.db")
    fresh = tmp_path / "shadow" / "fresh"

    def morning(now=MORNING, save=True, entsoe=None):
        return morning_run(store, DAY, kind="test", inputs_path=history, entsoe=entsoe or FakeEntsoe(table), weather=FakeWeather(table), now=now,
                           reports_dir=tmp_path / "none", wind_points=FakeWindPoints(table), wind_weights=wind,
                           solar_points=FakeSolarPoints(table), solar_weights=solar, feature_sets=("honest_v2",),
                           shadow_window_dir=fresh if save else None)

    yield {"store": store, "history": history, "fresh": fresh, "morning": morning, "tmp": tmp_path}
    store.close()


def run_shadow(s, issued="2024-04-09T08:30Z"):
    return shadow.shadow_run(s["store"], DAY, kind="test", inputs_path=s["history"], fresh_dir=s["fresh"], windows=WINDOWS,
                             now=pd.Timestamp(issued), clock=lambda: pd.Timestamp(issued))


def curves(store, model):
    v = pd.read_sql_query("SELECT version_id FROM shadow_versions WHERE model = ? ORDER BY version_id", store.conn, params=(model,))
    return [shadow.version_curve(store, i) for i in v["version_id"]]


def test_morning_run_saves_its_window_only_when_asked(setup):
    setup["morning"](save=False)
    assert not setup["fresh"].exists()
    summary = setup["morning"]()
    files = shadow.saved_windows(setup["fresh"])
    assert len(files) == 1 and files[0][0] == summary["run_id"]
    window = pd.read_csv(files[0][1], index_col=0)
    assert window.index.min() < "2024-04-01" and window.index.max() >= "2024-04-10"


def test_shadow_run_issues_s2_s1_and_lear_from_the_morning_inputs(setup):
    setup["morning"]()
    summary = run_shadow(setup)
    assert summary["status"] == "ok" and set(summary["versions"]) == {"S2", "S1", "LEAR"} and summary["before_window"]
    store = setup["store"]
    honest_id = store.conn.execute("SELECT honest_forecast_id FROM shadow_versions WHERE model = 'S2'").fetchone()[0]
    honest = store.forecast_curve(honest_id)["forecast"]
    lear, s1, s2 = curves(store, "LEAR")[0], curves(store, "S1")[0], curves(store, "S2")[0]
    assert len(lear) == 24 and lear.notna().all()
    np.testing.assert_allclose(s1.to_numpy(), ((honest + lear) / 2).to_numpy())
    np.testing.assert_allclose(s2.to_numpy(), s1.to_numpy())  # fewer than 14 settled days: equal weights
    assert run_shadow(setup)["status"] == "skipped"  # the same honest_v2 version is not shadowed twice
    assert not (setup["tmp"] / "published").exists()


def test_values_published_after_the_issue_time_change_nothing(setup, table):
    setup["morning"]()
    history = shadow.live_history(setup["history"], setup["fresh"])
    base = shadow.lear_live(history, DAY, windows=WINDOWS)
    changed = history.copy()
    local = changed.index.tz_convert("Europe/Paris")
    day = local.normalize().tz_localize(None)
    D = pd.Timestamp(DAY)
    price_cols = [c for c in changed.columns if c.startswith("price_")]
    changed.loc[day >= D, price_cols] += 500.0  # prices of D and later: published 13:00 on D-1 and after
    changed.loc[day > D, ["load_fc_mw", "temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"]] *= 3  # inputs of later days
    late_nuclear = changed.index + pd.Timedelta(hours=3) > pd.Timestamp("2024-04-09T08:05Z")
    changed.loc[late_nuclear, "nuclear_mw"] += 9999.0  # nuclear hours public after 10:05 Paris on D-1
    changed["solar_fc_mw"] = 1e6  # ENTSO-E wind and solar forecasts: never read
    np.testing.assert_array_equal(shadow.lear_live(changed, DAY, windows=WINDOWS).to_numpy(), base.to_numpy())
    moved = history.copy()
    moved.loc[day == D - pd.Timedelta(days=1), "price_eur_mwh"] += 50.0  # the D-1 price, public at 13:00 on D-2, does matter
    assert not np.allclose(shadow.lear_live(moved, DAY, windows=WINDOWS).to_numpy(), base.to_numpy())
    known = shadow.as_known(history, pd.Timestamp("2024-04-09T08:05Z"))
    assert known.loc[known.index.isin(day_hours(DAY)), "price_eur_mwh"].isna().all()
    assert known.loc[known.index.isin(day_hours(DAY)), "load_fc_mw"].notna().all()  # due 10:00 Paris on D-1


def test_versions_at_or_after_11_15_paris_do_not_count(setup):
    setup["morning"]()
    late = run_shadow(setup, issued="2024-04-09T09:15:00Z")  # 11:15 Paris
    assert not late["before_window"] and "does not count" in late["message"]
    assert shadow.counted_versions(setup["store"]).empty or (shadow.counted_versions(setup["store"])["delivery_day"] < shadow.SHADOW_START).all()
    versions = pd.read_sql_query("SELECT before_window FROM shadow_versions", setup["store"].conn)
    assert (versions["before_window"] == 0).all()
    setup["morning"](now=pd.Timestamp("2024-04-09T08:50Z"))
    on_time = run_shadow(setup, issued="2024-04-09T09:14:59Z")
    assert on_time["before_window"]


def test_missing_input_falls_back_to_honest_v2_and_is_counted(setup, table):
    setup["morning"](entsoe=FakeEntsoe(table, down={"neighbour_prices"}))  # CH and DE-LU prices of D-1 missing this morning
    s = setup["store"]
    # the history from inputs.csv ends before D-1, so LEAR has no D-1 neighbour prices: fallback
    summary = run_shadow(setup)
    assert summary["status"] == "fallback" and "LEAR" not in summary["versions"]
    honest_id = s.conn.execute("SELECT honest_forecast_id FROM shadow_versions WHERE model = 'S2'").fetchone()[0]
    np.testing.assert_array_equal(curves(s, "S2")[0].to_numpy(), s.forecast_curve(honest_id)["forecast"].to_numpy())
    assert s.conn.execute("SELECT fallback FROM shadow_versions WHERE model = 'S2'").fetchone()[0] == 1


def test_weights_use_settled_days_before_d_only(setup):
    setup["morning"]()
    s = setup["store"]
    shadow.ensure_schema(s)
    days = pd.bdate_range("2024-02-01", "2024-04-08")
    rows = [(d.strftime("%Y-%m-%d"), "state", 10.0, 5.0, "x") for d in days]
    s.conn.executemany("INSERT INTO shadow_errors VALUES (?, ?, ?, ?, ?)", rows)
    s.conn.commit()
    first = run_shadow(setup)
    assert first["w_lear"] == pytest.approx(2 / 3)  # LEAR's error half of honest_v2's
    s.conn.executemany("INSERT INTO shadow_errors VALUES (?, ?, ?, ?, ?)",
                       [(d, "live", 1.0, 999.0, "x") for d in ("2024-04-10", "2024-04-11", "2024-04-12")])  # D and later
    s.conn.commit()
    setup["morning"](now=pd.Timestamp("2024-04-09T08:50Z"))
    assert run_shadow(setup, issued="2024-04-09T09:00Z")["w_lear"] == pytest.approx(2 / 3)


def fake_day(store, day, honest_values, s2_values, fallback=0):
    """A pre-market honest_v2 version and a counted S2 and S1 version for `day`, written straight into the store."""
    shadow.ensure_schema(store)
    hours = day_hours(day)
    issued = pd.Timestamp(day).tz_localize("Europe/Paris") - pd.Timedelta(hours=14)  # 10:00 Paris on D-1
    curve = pd.DataFrame({"hour": hours.tz_convert("Europe/Paris").hour, "forecast": honest_values, "naive_day": np.nan}, index=hours)
    fid = store.save_forecast(None, day, "honest_v2", issued, "gbm:honest_v2.txt", "test", 0, True, curve)
    for model, values in (("S2", s2_values), ("S1", s2_values)):
        shadow._store_version(store, day, model, issued + pd.Timedelta(minutes=5), 1000 + fid, None, fid, pd.Series(values, index=hours),
                              bool(fallback), True, 0.5 if model == "S2" else None)
    store.save_actuals(pd.Series(50.0, index=hours), pd.Series([day] * len(hours)))


def test_settle_reports_counts_until_day_40_then_computes_once(tmp_path):
    store = Store(tmp_path / "forecast.db")
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-10-20", periods=41)]
    printed = []
    for i, day in enumerate(days[:39]):
        fake_day(store, day, 60.0, 60.0 if i == 3 else 55.0, fallback=int(i == 3))  # a fallback day's S2 is the honest_v2 curve
    out = shadow.shadow_settle(store, kind="test", log_fn=printed.append)
    assert out["settled_shadow_days"] == 39 and "result" not in out and "39 of 40" in printed[-1]
    assert store.conn.execute("SELECT COUNT(*) FROM shadow_result").fetchone()[0] == 0
    fake_day(store, days[39], 60.0, 55.0)
    fake_day(store, days[40], 60.0, 90.0)  # day 41 never enters the statistics
    out = shadow.shadow_settle(store, kind="test", log_fn=printed.append)
    result = out["result"]
    assert result["days"] == 40 and result["last_day"] == days[39]
    assert result["mean_difference_s2_minus_honest_v2"] == pytest.approx((39 * -5.0 + 0.0) / 40, abs=1e-3)
    assert result["fallback_rate"] == pytest.approx(1 / 40) and result["adopted"] is True and result["ci90"][1] <= 0
    again = shadow.shadow_settle(store, kind="test", log_fn=printed.append)
    assert again["result"] == result and "already computed" in printed[-1]
    store.close()


def test_shadow_tables_never_reach_the_public_export(tmp_path, monkeypatch):
    from thermo_fr.forecast.publish import export_published

    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "forecast.db")
    fake_day(store, "2026-10-20", 60.0, 55.0)
    out = tmp_path / "published"
    export_published(store, out_dir=out, now=pd.Timestamp("2026-10-21T12:00Z"), market_path=tmp_path / "none.csv")
    text = "\n".join(p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file())
    assert "S2" not in text and "shadow" not in text.lower()
    store.close()
