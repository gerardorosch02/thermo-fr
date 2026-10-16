"""Model sprint machinery: LEAR (construction, timing, scaling, windows, perturbation), the combination weights (no outcome of
day D or later), the walk-forward options (asinh target, recency weights, training-window averaging), the sprint rule."""

import json

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from thermo_fr import cli
from thermo_fr.forecast import lear, models, sprint, timing
from thermo_fr.forecast.backtest import FitOptions, walk_forward
from thermo_fr.forecast.combine import boa_weights, combine, daily_mae, inverse_mae_weights
from thermo_fr.forecast.features import build_features
from thermo_fr.forecast.models import fit_predict
from thermo_fr.forecast.timing import LookaheadError, issue_times
from thermo_fr.forecast.transforms import asinh_forward, asinh_inverse, robust_scale

LONG = ("2022-01-01", "2024-05-01")


@pytest.fixture(scope="module")
def long_inputs():
    frame = synthetic_inputs(*LONG, issued_from="2023-06-01")
    # a price that encodes its own day and hour, so every lag vector can be checked by hand
    local = frame.index.tz_convert("Europe/Paris")
    day_number = (local.normalize().tz_localize(None) - pd.Timestamp("2022-01-01")).days
    frame["price_eur_mwh"] = 50 + 0.1 * np.asarray(day_number) + np.asarray(local.hour) + np.random.default_rng(1).normal(0, 3, len(frame))
    return frame


@pytest.fixture(scope="module")
def design(long_inputs):
    return lear.lear_design(long_inputs)


def hourly_value(frame, column, day, hour):
    local = frame.index.tz_convert("Europe/Paris")
    mask = (local.normalize().tz_localize(None) == pd.Timestamp(day)) & (local.hour == hour)
    return float(frame.loc[mask, column].mean())


# --- LEAR construction and clock changes ------------------------------------------------------------------------------------

def test_lear_vectors_are_the_lagged_days(long_inputs, design):
    X, Y, blocks = design
    assert X.shape[1] == sum(len(c) for c in blocks.values()) == 302
    D = pd.Timestamp("2023-11-15")
    for lag in (1, 2, 3, 7):
        for h in (0, 7, 18, 23):
            assert X.at[D, f"price_d{lag}_h{h:02d}"] == pytest.approx(hourly_value(long_inputs, "price_eur_mwh", D - pd.Timedelta(days=lag), h))
    for h in (5, 19):
        assert X.at[D, f"load_d0_h{h:02d}"] == pytest.approx(hourly_value(long_inputs, "load_fc_mw", D, h))
        assert X.at[D, f"load_d1_h{h:02d}"] == pytest.approx(hourly_value(long_inputs, "load_fc_mw", D - pd.Timedelta(days=1), h))
        assert X.at[D, f"load_d7_h{h:02d}"] == pytest.approx(hourly_value(long_inputs, "load_fc_mw", D - pd.Timedelta(days=7), h))
        assert X.at[D, f"ch_d1_h{h:02d}"] == pytest.approx(hourly_value(long_inputs, "price_ch_eur_mwh", D - pd.Timedelta(days=1), h))
        assert X.at[D, f"de_lu_d1_h{h:02d}"] == pytest.approx(hourly_value(long_inputs, "price_de_lu_eur_mwh", D - pd.Timedelta(days=1), h))
        assert Y.at[D, h] == pytest.approx(hourly_value(long_inputs, "price_eur_mwh", D, h))
    assert X.at[D, "dow_2"] == 1 and X.loc[D, [f"dow_{k}" for k in range(7)]].sum() == 1
    assert X.at[pd.Timestamp("2023-11-01"), "day_type_2"] == 1  # All Saints' Day
    table = build_features(long_inputs, "honest_v2")
    assert X.at[D, "nuclear_d1_early"] == pytest.approx(table.X.loc[table.info["delivery_day"] == D, "nuclear_d1_early_mw"].iloc[0])


def test_clock_change_days(long_inputs, design):
    X, Y, _ = design
    autumn, spring = pd.Timestamp("2023-10-29"), pd.Timestamp("2024-03-31")
    assert Y.at[autumn, 2] == pytest.approx(hourly_value(long_inputs, "price_eur_mwh", autumn, 2))  # mean of the two rows
    assert Y.at[spring, 2] == pytest.approx((Y.at[spring, 1] + Y.at[spring, 3]) / 2)  # missing hour filled inside vectors
    assert X.at[spring + pd.Timedelta(days=1), "price_d1_h02"] == pytest.approx(Y.at[spring, 2])
    assert X.notna().all(axis=1).loc["2022-02-01":"2024-04-30"].all()


def test_lear_timing_rules_pass_both_deadlines_and_late_inputs_stop_it(long_inputs, monkeypatch):
    idx = pd.date_range("2024-01-15", periods=24, freq="1h", tz="Europe/Paris").tz_convert("UTC")
    assert (issue_times(idx, "price_lag3") == pd.Timestamp("2024-01-11T12:00Z")).all()  # 13:00 Paris on D-4
    assert (issue_times(idx, "load_forecast_lag1") == pd.Timestamp("2024-01-13T09:00Z")).all()  # 10:00 Paris on D-2
    assert (issue_times(idx, "load_forecast_lag7") == pd.Timestamp("2024-01-07T09:00Z")).all()
    assert set(lear.BLOCK_TIMINGS.values()) <= set(timing.TIMINGS)
    _, _, blocks = lear.lear_design(long_inputs)
    year = pd.date_range("2023-01-01", "2024-04-30", freq="1h", tz="UTC")
    lear.assert_gate(year, blocks)
    from thermo_fr.forecast import features

    assert all(c in features.FEATURE_TIMINGS for c in features.LEAR_COLUMNS)
    monkeypatch.delitem(features.FEATURE_TIMINGS, "temp_d0_h05")
    with pytest.raises(KeyError, match="temp_d0_h05"):
        lear.assert_gate(year[:24], blocks)
    monkeypatch.setitem(features.FEATURE_TIMINGS, "temp_d0_h05", "weather_issued")
    monkeypatch.setitem(lear.BLOCK_TIMINGS, "load_d0", "wind_solar_forecast")  # an input published after the gate
    for h in range(24):
        monkeypatch.setitem(features.FEATURE_TIMINGS, f"load_d0_h{h:02d}", "wind_solar_forecast")
    with pytest.raises(LookaheadError):
        lear.lear_walk_forward(long_inputs, "2024-03-01", "2024-03-02", windows=(364,))


# --- scaling and windows from training rows only ----------------------------------------------------------------------------

def test_transform_and_scalers_use_training_rows_only(design):
    X, Y, blocks = design
    m, s = robust_scale([1, 2, 3, 4, 100])
    assert m == 3 and s == pytest.approx(1.4826)
    assert asinh_inverse(asinh_forward([5.0, -40.0], m, s), m, s) == pytest.approx([5.0, -40.0])
    assert robust_scale([7, 7, 7]) == (7.0, 1.0)
    first, last = lear.window_days(pd.Timestamp("2024-03-10"), 364)
    assert last == pd.Timestamp("2024-03-09") and first == pd.Timestamp("2023-03-12")
    assert lear.window_days(pd.Timestamp("2022-06-01"), 1456)[0] == lear.LEAR_START
    train = (X.index >= first) & (X.index <= last)
    scalers = lear.fit_scalers(X[train], Y[train], blocks)
    X2, Y2 = X.copy(), Y.copy()
    X2.loc[X2.index > last] += 1000.0
    Y2.loc[Y2.index > last] += 1000.0
    assert lear.fit_scalers(X2[train], Y2[train], blocks) == scalers
    assert scalers["price_d1"] == scalers["target"] and scalers["load_d0"] != scalers["target"]


def test_lear_forecast_for_d_uses_nothing_from_d_or_later(long_inputs):
    D = pd.Timestamp("2024-03-12")
    base = lear.lear_walk_forward(long_inputs, D.strftime("%Y-%m-%d"), (D + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), windows=(364,))
    assert base["lear"].notna().all() and len(base) == 24
    changed = long_inputs.copy()
    local = changed.index.tz_convert("Europe/Paris").normalize().tz_localize(None)
    later = local >= D
    changed.loc[later, "price_eur_mwh"] += 500.0  # prices of D and later
    changed.loc[local > D, ["load_fc_mw", "temp_fc_c", "price_ch_eur_mwh", "nuclear_mw"]] *= 3  # inputs of later days
    nuclear_late = (local == D - pd.Timedelta(days=1)) & (changed.index.tz_convert("Europe/Paris").hour >= 8)
    changed.loc[nuclear_late, "nuclear_mw"] += 9999  # D-1 hours ending after 08:00, known only after 10:05 on D-1
    again = lear.lear_walk_forward(changed, D.strftime("%Y-%m-%d"), (D + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), windows=(364,))
    np.testing.assert_allclose(again["lear"].to_numpy(), base["lear"].to_numpy(), rtol=0, atol=1e-9)
    moved = long_inputs.copy()
    moved.loc[local == D - pd.Timedelta(days=1), "price_eur_mwh"] += 50.0  # the D-1 price does matter
    third = lear.lear_walk_forward(moved, D.strftime("%Y-%m-%d"), (D + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), windows=(364,))
    assert not np.allclose(third["lear"].to_numpy(), base["lear"].to_numpy())


def test_lear_autumn_rows_share_the_hour_two_forecast(long_inputs):
    out = lear.lear_walk_forward(long_inputs, "2023-10-29", "2023-10-30", windows=(364,), recalibrate="monthly")
    assert len(out) == 25 and out["lear"].notna().all()
    two = out[out["hour"] == 2]["lear"]
    assert len(two) == 2 and two.iloc[0] == two.iloc[1]


# --- combination weights ----------------------------------------------------------------------------------------------------

def errors_table(days=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame({"a": rng.uniform(5, 10, days), "b": rng.uniform(10, 20, days)}, index=idx)


def test_inverse_mae_weights_by_hand_and_warm_up():
    errors = errors_table()
    days = errors.index
    w = inverse_mae_weights(errors, days, half_life=14, min_days=14)
    assert np.allclose(w.sum(axis=1), 1)
    assert (w.iloc[:5] == 0.5).all().all()  # fewer than 14 settled Tuesday-to-Saturday days
    D = days[40]
    past = errors[(errors.index <= D - pd.Timedelta(days=1)) & errors.index.dayofweek.isin([1, 2, 3, 4, 5])]
    decay = 0.5 ** ((D - past.index).days.to_numpy() / 14)
    ewma = (past.to_numpy() * decay[:, None]).sum(axis=0) / decay.sum()
    expected = (1 / ewma) / (1 / ewma).sum()
    assert w.loc[D].to_numpy() == pytest.approx(expected)
    assert w.loc[D, "a"] > w.loc[D, "b"]  # the model with the smaller recent error gets more weight


def test_online_weights_for_d_use_no_outcome_of_d_or_later():
    errors = errors_table(90)
    days = errors.index[20:]
    w = inverse_mae_weights(errors, days)
    for D in days[::7]:
        changed = errors.copy()
        changed.loc[changed.index >= D] = changed.loc[changed.index >= D] * 10 + 100
        assert inverse_mae_weights(changed, [D]).loc[D].to_numpy() == pytest.approx(w.loc[D].to_numpy())
    # Sunday and Monday deliveries never update the weights
    off = errors.copy()
    off.loc[off.index.dayofweek.isin([0, 6])] = 999.0
    assert np.allclose(inverse_mae_weights(off, days).to_numpy(), w.to_numpy())


def hourly_frame(days=50, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="1h", tz="UTC")
    actual = 60 + 10 * np.sin(np.arange(len(idx)) / 24) + rng.normal(0, 2, len(idx))
    frame = pd.DataFrame({"delivery_day": idx.tz_localize(None).normalize(), "actual": actual}, index=idx)
    frame["good"] = actual + rng.normal(0, 2, len(idx))
    frame["bad"] = actual + 8 + rng.normal(0, 6, len(idx))
    return frame


def test_boa_weights_move_to_the_better_model_and_never_see_d():
    frame = hourly_frame()
    days = pd.DatetimeIndex(sorted(frame["delivery_day"].unique()))
    w = boa_weights(frame, ["good", "bad"], days, eta=0.05)
    assert np.allclose(w.sum(axis=1), 1) and (w.iloc[0] == 0.5).all()
    assert w.iloc[-1]["good"] > 0.9
    D = days[30]
    changed = frame.copy()
    changed.loc[changed["delivery_day"] >= D, "actual"] += 300
    assert boa_weights(changed, ["good", "bad"], [D]).loc[D].to_numpy() == pytest.approx(w.loc[D].to_numpy())
    e = daily_mae(frame, ["good", "bad"])
    assert (e["good"] < e["bad"]).mean() > 0.9


def test_combine_renormalises_over_present_components():
    frame = pd.DataFrame({"delivery_day": pd.to_datetime(["2024-01-01"] * 3), "x": [10.0, 20.0, np.nan], "y": [30.0, np.nan, np.nan]})
    assert combine(frame, None, ["x", "y"]).tolist()[:2] == [20.0, 20.0] and np.isnan(combine(frame, None, ["x", "y"]).iloc[2])
    weights = pd.DataFrame({"x": [0.25], "y": [0.75]}, index=pd.to_datetime(["2024-01-01"]))
    assert combine(frame, weights, ["x", "y"]).iloc[0] == pytest.approx(25.0)


# --- walk-forward options -------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_table():
    return build_features(synthetic_inputs(), "honest")


def test_default_options_take_the_original_path(short_table, monkeypatch):
    calls = []
    original = fit_predict

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr("thermo_fr.forecast.backtest.fit_predict", spy)
    a = walk_forward(short_table, "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None)
    b = walk_forward(short_table, "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None, options=FitOptions())
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    assert all(not c for c in calls)  # no sample_weight, no extra arguments: the original call


def test_asinh_recency_and_windows_match_a_manual_fit(short_table):
    start, end = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-04-01")
    days = short_table.info["delivery_day"]
    train, test = days < start, (days >= start) & (days < end)
    opts = FitOptions(target_transform="asinh", recency_half_life=30, train_windows=(None, 60))
    got = walk_forward(short_table, "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None, options=opts).predictions["linear"]
    preds = []
    for window in (None, 60):
        mask = train if window is None else train & (days >= start - pd.Timedelta(days=60))
        y = short_table.y[mask]
        m, s = robust_scale(y.dropna())  # training targets only
        weight = 0.5 ** ((start - days[mask]).dt.days.to_numpy() / 30)
        z = pd.Series(asinh_forward(y, m, s), index=y.index)
        preds.append(asinh_inverse(fit_predict("linear", short_table.X[mask], z, short_table.X[test], sample_weight=weight), m, s))
    np.testing.assert_allclose(got.to_numpy(), np.mean(preds, axis=0))
    assert FitOptions(target_transform="asinh", train_windows=(None, 730, 365)).suffix == "+windows_all_730_365+asinh"


def test_asinh_parameters_ignore_the_forecast_month(short_table):
    opts = FitOptions(target_transform="asinh")
    a = walk_forward(short_table, "2024-03-01", "2024-04-01", models=("gbm",), log=lambda *_: None, options=opts).predictions["gbm"]
    changed = build_features(synthetic_inputs(), "honest")
    later = changed.info["delivery_day"] >= pd.Timestamp("2024-03-01")
    changed.y[later] = changed.y[later] * 5 + 1000
    b = walk_forward(changed, "2024-03-01", "2024-04-01", models=("gbm",), log=lambda *_: None, options=opts).predictions["gbm"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_forecast_backtest_cli_with_sprint_options(tmp_path, monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    data = tmp_path / "forecast"
    data.mkdir()
    synthetic_inputs().to_csv(data / "inputs.csv", index_label="timestamp_utc")
    out = tmp_path / "out"
    cli.main(["forecast-backtest", "--data", str(data / "inputs.csv"), "--out", str(out), "--test-start", "2024-03-01", "--test-end",
              "2024-04-01", "--feature-sets", "honest_v2", "--target-transform", "asinh", "--train-windows", "all,60"])
    summary = json.loads((out / "summary.json").read_text())
    assert "honest_v2+windows_all_60+asinh" in summary and (out / "report.md").exists()


# --- the sprint runner and its rule ----------------------------------------------------------------------------------------

def test_sprint_refuses_the_holdout(long_inputs):
    with pytest.raises(ValueError, match="holdout"):
        sprint.run_sprint(long_inputs, "2026-06-01", "2026-08-01")


def fake_sprint_frame(gains: dict, days=120, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=days * 24, freq="1h", tz="UTC")
    local = idx.tz_convert("Europe/Paris")
    actual = 60 + 20 * np.sin(np.arange(len(idx)) / 24) + rng.normal(0, 10, len(idx))
    frame = pd.DataFrame({"delivery_day": local.normalize().tz_localize(None), "hour": local.hour, "dow": local.dayofweek,
                          "holiday": 0, "is_peak": ((local.hour >= 8) & (local.hour < 20) & (local.dayofweek < 5)).astype(int),
                          "wind_mw": rng.uniform(0, 10000, len(idx)), "solar_mw": rng.uniform(0, 5000, len(idx)),
                          "actual": actual, "strict": True}, index=idx)
    frame["naive_day"] = frame["actual"].shift(24)
    frame["naive_week"] = frame["actual"].shift(168)
    noise = rng.normal(0, 15, len(idx))
    frame["honest_v2"] = actual + noise
    frame["lear"] = actual + rng.normal(0, 15, len(idx))
    for name, gain in gains.items():
        frame[name] = actual + noise * (1 - gain)  # a uniformly smaller error
    frame["w_lear"] = 0.5
    frame["lear_missing"] = False
    hourly = pd.DataFrame({"price_eur_mwh": actual}, index=idx)
    return frame, hourly


def test_selection_rule_on_a_synthetic_frame():
    frame, hourly = fake_sprint_frame({"S1": 0.3, "S2": 0.0, "S3": 0.2, "S4": 0.01})
    report = sprint.selection_report(frame, hourly)
    c = report["candidates"]
    assert c["S1"]["passes"] and c["S1"]["eligible_for_shadow"] and c["S1"]["gain"] > 0.25
    assert not c["S2"]["passes"] and not c["S4"]["conditions"]["gain_at_least_0.25"]
    assert report["to_shadow"] == "S1"
    assert set(report["diagnostics"]["S1"]) >= {"rmse", "by_month", "mae_without_april_2024", "regime_change_days", "lag", "paired_vs_baseline"}
    p = report["diagnostics"]["S1"]["paired_vs_baseline"]
    assert p["upper_one_sided_98_75"] < 0 and p["ci95"][0] <= p["upper_one_sided_98_75"]
    none = sprint.selection_report(*fake_sprint_frame({"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0}))
    assert none["to_shadow"] is None and "no candidate is distinguishable" in none["finding"]


def test_pipeline_perturbation_features_scalers_models_and_weights(long_inputs):
    """Prices of D and later, inputs of later days, and D-1 nuclear hours after 08:00 leave LEAR's features, scaling parameters
    and fitted models for D unchanged, and the S2 weight used on D."""
    D = pd.Timestamp("2024-03-12")
    changed = long_inputs.copy()
    local = changed.index.tz_convert("Europe/Paris")
    day = local.normalize().tz_localize(None)
    changed.loc[day >= D, "price_eur_mwh"] += 500.0
    changed.loc[day > D, ["load_fc_mw", "temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2", "price_ch_eur_mwh", "price_de_lu_eur_mwh"]] *= 3
    changed.loc[(day == D - pd.Timedelta(days=1)) & (local.hour >= 8), "nuclear_mw"] += 9999
    X1, Y1, blocks = lear.lear_design(long_inputs)
    X2, Y2, _ = lear.lear_design(changed)
    pd.testing.assert_frame_equal(X1.loc[:D], X2.loc[:D])
    first, last = lear.window_days(D, 364)
    fits = []
    for X, Y in ((X1, Y1), (X2, Y2)):
        train = (X.index >= first) & (X.index <= last)
        fits.append(lear.fit_window(X[train], Y[train], blocks))
    assert fits[0]["scalers"] == fits[1]["scalers"] and fits[0]["rows"] == fits[1]["rows"]
    for a, b in zip(fits[0]["models"], fits[1]["models"]):
        np.testing.assert_array_equal(a.coef_, b.coef_)
        assert a.intercept_ == b.intercept_
    # S2: the weight used on D does not move when the actual prices of D and later days change
    frame = hourly_frame(60)
    frame["strict"] = True
    frame["naive_day"] = np.nan
    base = frame.rename(columns={"good": "gbm"})[["delivery_day", "actual", "gbm", "strict"]]
    lear_part = pd.DataFrame({"lear": frame["bad"], "lear_w364": frame["bad"]}, index=frame.index)
    w1 = sprint.assemble(base, base, base, lear_part, lear_windows=(364,))
    late = base.copy()
    cut = pd.Timestamp("2024-02-10")
    late.loc[late["delivery_day"] >= cut, "actual"] += 400
    w2 = sprint.assemble(late, late, late, lear_part, lear_windows=(364,))
    on_d = w1["delivery_day"] == cut
    assert (w1.loc[on_d, "w_lear"] == w2.loc[on_d, "w_lear"]).all() and w1.loc[on_d, "w_lear"].iloc[0] < 0.5
    assert (w1.loc[on_d, "S2"] == w2.loc[on_d, "S2"]).all()


def test_fixed_seeds_make_runs_repeat(short_table):
    opts = FitOptions(target_transform="asinh", train_windows=(None, 60))
    a = walk_forward(short_table, "2024-03-01", "2024-04-01", models=("gbm",), log=lambda *_: None, options=opts).predictions["gbm"]
    b = walk_forward(short_table, "2024-03-01", "2024-04-01", models=("gbm",), log=lambda *_: None, options=opts).predictions["gbm"]
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    assert models.GBM_PARAMS["random_state"] == 7


def test_staged_commands_resume_and_report(long_inputs, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    data = tmp_path / "inputs.csv"
    long_inputs.to_csv(data, index_label="timestamp_utc")
    out = tmp_path / "sprint"
    common = ["--data", str(data), "--out", str(out), "--test-start", "2024-03-01", "--test-end", "2024-04-01"]
    with pytest.raises(ValueError, match="holdout"):
        cli.main(["sprint-stage", "gbm", "--data", str(data), "--out", str(out), "--test-end", "2026-08-01"])
    cli.main(["sprint-stage", "gbm", *common])
    cli.main(["sprint-stage", "lear", *common, "--recalibrate", "monthly"])
    assert (out / "lear_2024-03.csv").exists() and {"gbm_honest_v2.csv", "gbm_S3.csv", "gbm_S4.csv"} <= {p.name for p in out.iterdir()}
    stamp = (out / "lear_2024-03.csv").stat().st_mtime
    cli.main(["sprint-stage", "lear", *common, "--recalibrate", "monthly"])  # resumes: nothing redone
    assert (out / "lear_2024-03.csv").stat().st_mtime == stamp
    cli.main(["sprint-report", *common])
    report = json.loads((out / "sprint_report.json").read_text())
    assert set(report["candidates"]) == {"S1", "S2", "S3", "S4"} and "finding" in report
    frame = pd.read_csv(out / "sprint_predictions.csv")
    assert frame[["honest_v2", "lear", "S1", "S2", "S3", "S4"]].notna().all().all()
