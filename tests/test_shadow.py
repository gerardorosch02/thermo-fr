"""The laptop-only S2 shadow: live-format inputs, the publication-time gate, the 11:15 guard, fallbacks, the model-file rule
(read-only git on the fetched origin/main ref), lost days, settlement and the statistics at day 40, and that nothing reaches
published/ or git beyond read-only commands."""

import json
import logging
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from test_jobs import FakeEntsoe, FakeSolarPoints, FakeWeather, FakeWindPoints
from thermo_fr.forecast import models, shadow, solar_proxy
from thermo_fr.forecast.jobs import day_hours, morning_run
from thermo_fr.forecast.refit import refit_model
from thermo_fr.forecast.store import Store, iso
from thermo_fr.forecast.wind_proxy import calibrate, save_weights

DAY = "2024-04-10"
MORNING = pd.Timestamp("2024-04-09T08:20Z")  # 10:20 Paris on D-1, inside the pre-market window
WINDOWS = (364,)


@pytest.fixture(scope="module")
def table():
    return synthetic_inputs("2022-09-01", "2024-05-01", issued_from="2024-01-10")


REAL_RUN = subprocess.run
GIT_ID = ["-c", "user.name=test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false"]


HELPER = {"active": False}  # the tests' own git set-up, not the shadow's calls


def git(repo, *args):
    HELPER["active"] = True
    try:
        out = REAL_RUN(["git", *GIT_ID, *args], cwd=str(repo), capture_output=True, text=True)
    finally:
        HELPER["active"] = False
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def publish_model(repo, model_dir, message="refit"):
    """Commit model_dir's honest_v2 files under published/model/ and point origin/main at the commit (no remote, no fetch)."""
    target = repo / "published" / "model"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("honest_v2.txt", "honest_v2.json"):
        shutil.copy(model_dir / name, target / name)
    git(repo, "add", "published")
    git(repo, "commit", "-q", "-m", message)
    commit = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/main", commit)
    return commit


@pytest.fixture(scope="module")
def model_dirs(table, tmp_path_factory):
    """The published honest_v2 file and a second, different one (as after a monthly refit)."""
    before = table[table.index < "2024-04-01"]
    first, second = tmp_path_factory.mktemp("model_a"), tmp_path_factory.mktemp("model_b")
    refit_model(before, out_dir=first, feature_set="honest_v2", holdout_days=0, log=lambda *_: None, params={"n_estimators": 20})
    refit_model(before, out_dir=second, feature_set="honest_v2", holdout_days=0, log=lambda *_: None, params={"n_estimators": 21})
    return first, second


@pytest.fixture
def setup(table, tmp_path, monkeypatch, model_dirs):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    monkeypatch.chdir(tmp_path)  # anything written relative to the working directory would land here
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    publish_model(repo, model_dirs[0])
    git_calls = []

    def guard(original):
        def wrapped(*args, **kwargs):
            command = args[0] if args else kwargs.get("args", "")
            words = list(command) if isinstance(command, (list, tuple)) else str(command).split()
            if HELPER["active"]:
                return original(*args, **kwargs)
            if "gh" in words:
                raise AssertionError(f"the shadow must never call gh: {words}")
            if "git" in words:
                sub = words[words.index("git") + 1]
                git_calls.append(sub)
                if sub not in ("rev-parse", "show", "log"):
                    raise AssertionError(f"the shadow runs read-only git only: {words}")
            return original(*args, **kwargs)
        return wrapped

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, guard(getattr(subprocess, name)))
    history = tmp_path / "inputs.csv"
    table[table.index < "2024-04-01"].to_csv(history, index_label="timestamp_utc")
    before = table[table.index < "2024-04-01"]
    model_file = model_dirs[0] / "honest_v2.txt"
    wind = save_weights(calibrate(before, before["wind_onshore_mw"] + before["wind_offshore_mw"]), tmp_path / "model" / "wind_proxy.json")
    solar = save_weights(solar_proxy.calibrate(before, before["solar_mw"]), tmp_path / "model" / "solar_proxy.json")
    store = Store(tmp_path / "forecast.db")
    fresh = tmp_path / "shadow" / "fresh"

    def morning(now=MORNING, save=True, entsoe=None, day=DAY, model=None):
        return morning_run(store, day, kind="test", inputs_path=history, entsoe=entsoe or FakeEntsoe(table), weather=FakeWeather(table), now=now,
                           reports_dir=tmp_path / "none", wind_points=FakeWindPoints(table), wind_weights=wind,
                           solar_points=FakeSolarPoints(table), solar_weights=solar, feature_sets=("honest_v2",),
                           model_file=model or model_file, fallback_model_file=None, shadow_window_dir=fresh if save else None)

    yield {"store": store, "history": history, "fresh": fresh, "morning": morning, "tmp": tmp_path, "repo": repo,
           "models": tmp_path / "shadow" / "models", "git_calls": git_calls, "model_dirs": model_dirs}
    store.close()


def run_shadow(s, issued="2024-04-09T08:30Z", day=DAY):
    return shadow.shadow_run(s["store"], day, kind="test", inputs_path=s["history"], fresh_dir=s["fresh"], windows=WINDOWS,
                             now=pd.Timestamp(issued), clock=lambda: pd.Timestamp(issued), repo=s["repo"], models_dir=s["models"])


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
    setup["morning"]()
    s = setup["store"]
    # the saved window of the morning lacks D's load forecast (as if it had not been published): LEAR cannot be issued
    path = shadow.saved_windows(setup["fresh"])[0][1]
    window = pd.read_csv(path, index_col=0)
    window.loc[window.index.isin(day_hours(DAY).strftime("%Y-%m-%d %H:%M:%S+00:00")), "load_fc_mw"] = np.nan
    assert window["load_fc_mw"].isna().sum() >= 24
    window.to_csv(path)
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
    assert "w_lear" not in first
    stored_weight = "SELECT w_lear FROM shadow_versions WHERE model = 'S2' ORDER BY version_id DESC LIMIT 1"
    assert s.conn.execute(stored_weight).fetchone()[0] == pytest.approx(2 / 3)  # LEAR's error half of honest_v2's
    s.conn.executemany("INSERT INTO shadow_errors VALUES (?, ?, ?, ?, ?)",
                       [(d, "live", 1.0, 999.0, "x") for d in ("2024-04-10", "2024-04-11", "2024-04-12")])  # D and later
    s.conn.commit()
    setup["morning"](now=pd.Timestamp("2024-04-09T08:50Z"))
    run_shadow(setup, issued="2024-04-09T09:00Z")
    assert s.conn.execute(stored_weight).fetchone()[0] == pytest.approx(2 / 3)


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


# --- the model-file rule (shadow_decision.md) ---------------------------------------------------------------------------------

def day_row(store, day=DAY):
    cols = [c[1] for c in store.conn.execute("PRAGMA table_info(shadow_day_models)")]
    row = store.conn.execute("SELECT * FROM shadow_day_models WHERE delivery_day = ?", (day,)).fetchone()
    return dict(zip(cols, row)) if row else None


def test_model_file_of_the_day_is_read_with_read_only_git_stored_by_sha_and_fixed_for_the_day(setup):
    setup["morning"]()
    first = run_shadow(setup)
    assert first["status"] == "ok"
    store, (dir_a, dir_b) = setup["store"], setup["model_dirs"]
    row = day_row(store)
    sha = shadow.sha256_file(dir_a / "honest_v2.txt")
    meta = json.loads((dir_a / "honest_v2.json").read_text())
    assert row["sha256"] == sha and first["model_sha256"] == sha
    assert (setup["models"] / f"{sha}.txt").read_bytes() == (dir_a / "honest_v2.txt").read_bytes()
    assert row["ref_commit"] == git(setup["repo"], "rev-parse", "origin/main") == row["model_commit"]
    assert row["ref_commit_time_utc"] and row["ref_age_hours"] is not None
    assert (row["fitted_at_utc"], row["train_from"], row["train_to"]) == (meta["fitted_at_utc"], meta["train_from"], meta["train_to"])
    assert row["lear_windows"] == "[364]" and row["lear_recalibrated_through"] == "2024-04-09" and row["changed"] == 0
    assert row["source"].startswith("repository ref")  # no publish worktree yet: the shared ref, read through the repository
    # the refit lands on origin/main between the two runs of the day: the day keeps its first file
    publish_model(setup["repo"], dir_b)
    setup["morning"](now=pd.Timestamp("2024-04-09T08:50Z"))
    second = run_shadow(setup, issued="2024-04-09T09:00Z")
    assert second["status"] == "ok" and day_row(store)["sha256"] == sha
    assert set(setup["git_calls"]) <= {"rev-parse", "show", "log"} and setup["git_calls"]
    assert not (setup["tmp"] / "published").exists()


def test_publish_worktree_is_used_when_it_exists(setup):
    git(setup["repo"], "worktree", "add", "-q", "--detach", str(setup["repo"] / ".publish-worktree"), "origin/main")
    setup["morning"]()
    assert run_shadow(setup)["status"] == "ok"
    assert day_row(setup["store"])["source"] == "publish worktree"


def test_a_day_whose_comparator_used_another_file_is_lost_not_counted(setup):
    publish_model(setup["repo"], setup["model_dirs"][1])  # origin/main has the refit, the laptop's checkout does not
    setup["morning"]()
    out = run_shadow(setup)
    store = setup["store"]
    assert out["status"] == "lost" and "model files differ" in out["message"]
    assert store.conn.execute("SELECT COUNT(*) FROM shadow_versions").fetchone()[0] == 0
    reasons = pd.read_sql_query("SELECT reason FROM shadow_lost_events", store.conn)["reason"]
    assert reasons.str.contains("model files differ").all()


def test_the_next_day_switches_to_the_refit_and_records_the_change(setup):
    store, (dir_a, dir_b) = setup["store"], setup["model_dirs"]
    setup["morning"]()
    assert run_shadow(setup)["status"] == "ok"
    publish_model(setup["repo"], dir_b)
    nxt = "2024-04-11"
    setup["morning"](now=pd.Timestamp("2024-04-10T08:20Z"), day=nxt, model=dir_b / "honest_v2.txt")
    out = run_shadow(setup, issued="2024-04-10T08:30Z", day=nxt)
    assert out["status"] == "ok"
    row = day_row(store, nxt)
    assert row["sha256"] == shadow.sha256_file(dir_b / "honest_v2.txt") and row["changed"] == 1


def test_unreadable_ref_uses_the_last_stored_file_or_loses_the_day(setup):
    store = setup["store"]
    setup["morning"]()
    assert run_shadow(setup)["status"] == "ok"
    git(setup["repo"], "update-ref", "-d", "refs/remotes/origin/main")
    nxt = "2024-04-11"
    setup["morning"](now=pd.Timestamp("2024-04-10T08:20Z"), day=nxt)
    assert run_shadow(setup, issued="2024-04-10T08:30Z", day=nxt)["status"] == "ok"
    row = day_row(store, nxt)
    assert row["source"].startswith("last stored file") and row["sha256"] == day_row(store)["sha256"]


def test_no_ref_and_no_stored_file_loses_the_day(setup):
    git(setup["repo"], "update-ref", "-d", "refs/remotes/origin/main")
    setup["morning"]()
    out = run_shadow(setup)
    assert out["status"] == "lost" and "no published honest_v2 file" in out["message"]


def test_read_only_git_refuses_anything_else(tmp_path):
    for args in (("fetch", "origin", "main"), ("pull",), ("checkout", "main"), ("commit", "-m", "x"), ("push",)):
        with pytest.raises(ValueError, match="read-only"):
            shadow.read_only_git(tmp_path, *args, runner=lambda *a, **k: pytest.fail("must not run"))


def test_line_endings_do_not_change_the_model_sha():
    assert shadow.sha256_bytes(b"tree\r\nTree=0\r\n") == shadow.sha256_bytes(b"tree\nTree=0\n")


# --- the LEAR weight stays out of every output before day 40 ------------------------------------------------------------------

def test_lear_weight_is_never_logged_or_returned(setup, caplog):
    s = setup["store"]
    shadow.ensure_schema(s)
    s.conn.executemany("INSERT INTO shadow_errors VALUES (?, ?, ?, ?, ?)",
                       [(d.strftime("%Y-%m-%d"), "state", 10.0, 5.0, "x") for d in pd.bdate_range("2024-02-01", "2024-04-08")])
    s.conn.commit()
    setup["morning"]()
    caplog.clear()  # only what the shadow run itself logs (the morning run logs its proxy weights, which are unrelated)
    with caplog.at_level(logging.DEBUG, logger="thermo_fr"):
        out = run_shadow(setup)
    assert any(r.name == "thermo_fr.shadow" for r in caplog.records)
    text = caplog.text.lower() + json.dumps(out, default=str).lower()
    run_message = s.conn.execute("SELECT message FROM runs WHERE command = 'shadow-run'").fetchone()[0] or ""
    for word in ("weight", "w_lear", "0.667", "0.333"):
        assert word not in text and word not in run_message.lower()


# --- the weight-state sort -----------------------------------------------------------------------------------------------------

def test_a_live_error_row_wins_over_a_state_row_for_the_same_day(tmp_path):
    store = Store(tmp_path / "forecast.db")
    shadow.ensure_schema(store)
    store.conn.executemany("INSERT INTO shadow_errors VALUES (?, ?, ?, ?, ?)",
                           [("2026-10-10", "state", 1.0, 2.0, "x"), ("2026-10-10", "live", 3.0, 4.0, "y"), ("2026-10-09", "state", 5.0, 6.0, "x")])
    store.conn.commit()
    frame = shadow.weight_errors(store)
    assert frame.loc["2026-10-10"].tolist() == [3.0, 4.0] and frame.loc["2026-10-09"].tolist() == [5.0, 6.0]
    assert frame.index.is_monotonic_increasing
    store.close()


# --- lost days in every settle line and in the statistics ---------------------------------------------------------------------

def test_lost_days_are_reported_in_settle_lines_and_at_day_40(tmp_path):
    store = Store(tmp_path / "forecast.db")
    expected = shadow.expected_days(shadow.SHADOW_START, "2026-12-31")
    lost = {expected[2], expected[5]}
    printed = []
    counted = [d for d in expected if d not in lost]
    for day in counted[:10]:
        fake_day(store, day, 60.0, 55.0)
    shadow.ensure_schema(store)
    store.conn.execute("INSERT INTO shadow_lost_events VALUES (?, 1, 'model files differ: comparator x, published y', 'z')", (expected[2],))
    store.conn.commit()
    out = shadow.shadow_settle(store, kind="test", log_fn=printed.append)
    assert out["lost_days"] == 2
    assert "10 of 40" in printed[-1] and "2 lost" in printed[-1] and "model files differ" in printed[-1] and "no shadow run recorded" in printed[-1]
    for day in counted[10:41]:
        fake_day(store, day, 60.0, 55.0)
    result = shadow.shadow_settle(store, kind="test", log_fn=printed.append)["result"]
    assert result["days"] == 40 and result["lost_days"] == 2
    assert {r["delivery_day"] for r in result["lost"]} == lost
    assert result["counted_days_by_model_file"] == [{"sha256": "unknown", "first_day": counted[0], "counted_days": 40}]
    store.close()
