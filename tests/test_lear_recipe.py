"""LEAR to the published recipe (docs/prereg/lear_recipe_prereg.md): the optional short-window rule, asinh slope and shared Gram
matrix leave the default model untouched; the rolling slope choice and every new forecast path use nothing from D onward."""

import warnings

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from thermo_fr.forecast import lear, lear_recipe, yardstick
from thermo_fr.forecast.transforms import asinh_forward, asinh_inverse, asinh_slope_forward, asinh_slope_inverse, slope_offset

LONG = ("2022-01-01", "2024-05-01")
DAY = pd.Timestamp("2024-03-12")


@pytest.fixture(scope="module")
def inputs():
    frame = synthetic_inputs(*LONG, issued_from="2023-06-01")
    local = frame.index.tz_convert("Europe/Paris")
    spikes = np.random.default_rng(2).standard_t(3, len(frame)) * 6  # heavy tails, so the asinh slope matters
    frame["price_eur_mwh"] = frame["price_eur_mwh"] + spikes + 0.02 * np.asarray((local.normalize().tz_localize(None) - pd.Timestamp(LONG[0])).days)
    return frame


@pytest.fixture(scope="module")
def design(inputs):
    return lear.lear_design(inputs)


def train_rows(design, day, window):
    X, Y, blocks = design
    first, last = lear.window_days(day, window)
    rows = (X.index >= first) & (X.index <= last)
    return X[rows], Y[rows], blocks


# --- the transform ----------------------------------------------------------------------------------------------------------

def test_slope_one_is_the_plain_asinh_bit_for_bit():
    x = np.random.default_rng(0).normal(40, 60, 500)
    assert np.array_equal(asinh_slope_forward(x, 35.0, 20.0, 1.0), asinh_forward(x, 35.0, 20.0))
    z = asinh_forward(x, 35.0, 20.0)
    assert np.array_equal(asinh_slope_inverse(z, 35.0, 20.0, 1.0), asinh_inverse(z, 35.0, 20.0))
    assert slope_offset(1.0) == 0.0


@pytest.mark.parametrize("c", [0.1, 0.3, 0.5, 0.7])
def test_slope_transform_formula_inverse_and_slope_at_the_origin(c):
    a = np.sqrt(1 / c**2 - 1)
    assert slope_offset(c) == pytest.approx(a)
    m, s = 35.0, 20.0
    x = np.array([-400.0, -5.0, 35.0, 36.0, 80.0, 3000.0])
    u = (x - m) / s
    expected = np.sign(u) * (np.arcsinh(np.abs(u) + a) - np.arcsinh(a))
    z = asinh_slope_forward(x, m, s, c)
    assert z == pytest.approx(expected) and z[2] == 0.0
    assert asinh_slope_inverse(z, m, s, c) == pytest.approx(x)
    assert asinh_slope_forward(2 * m - x, m, s, c) == pytest.approx(-z)  # odd around the median
    h = 1e-6
    assert (asinh_slope_forward(m + s * h, m, s, c) - asinh_slope_forward(m - s * h, m, s, c)) / (2 * h) == pytest.approx(c, rel=1e-4)
    assert np.all(np.diff(asinh_slope_forward(np.linspace(-500, 500, 101), m, s, c)) > 0)


def test_slope_must_be_in_the_unit_interval():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            slope_offset(bad)


def test_slope_applies_to_the_price_blocks_only(design):
    X, Y, blocks = train_rows(design, DAY, 364)
    scalers = lear.fit_scalers(X, Y, blocks)
    plain = pd.DataFrame(lear.transform(X, blocks, scalers), columns=X.columns)
    sloped = pd.DataFrame(lear.transform(X, blocks, scalers, slope=0.3), columns=X.columns)
    assert lear.SLOPE_BLOCKS == ("price_d1", "price_d2", "price_d3", "price_d7", "ch_d1", "de_lu_d1")
    for name, cols in blocks.items():
        same = np.array_equal(plain[cols].to_numpy(), sloped[cols].to_numpy())
        assert same == (name not in lear.SLOPE_BLOCKS), name
    m, s = scalers["ch_d1"]
    assert sloped["ch_d1_h07"].to_numpy() == pytest.approx(asinh_slope_forward(X["ch_d1_h07"].to_numpy(), m, s, 0.3))
    assert np.array_equal(lear.transform(X, blocks, scalers, slope=1.0), plain.to_numpy())


# --- the defaults are the model as it was -----------------------------------------------------------------------------------

def reference_fit_predict(X, Y, blocks, x_new, max_iter=2500):
    """The fit and forecast written out as the code stood before the optional arguments (lear.py at commit 0459b68)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso, LassoLarsIC

    scalers = lear.fit_scalers(X, Y, blocks)
    Xt = X.copy()
    for name, cols in blocks.items():
        if name != "calendar":
            Xt[cols] = asinh_forward(Xt[cols].to_numpy(), *scalers[name])
    Xt = Xt.to_numpy(dtype=float)
    new = x_new.copy()
    for name, cols in blocks.items():
        if name != "calendar":
            new[cols] = asinh_forward(new[cols].to_numpy(), *scalers[name])
    m, s = scalers["target"]
    out = []
    for h in lear.HOURS:
        yt = asinh_forward(Y[h].to_numpy(), m, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            alpha = LassoLarsIC(criterion="aic", max_iter=max_iter).fit(Xt, yt).alpha_
            model = Lasso(alpha=max(alpha, 1e-12), max_iter=max_iter, precompute=True).fit(Xt, yt)
        out.append(asinh_inverse(model.predict(new.to_numpy(dtype=float)), m, s)[0])
    return np.array(out)


def test_default_fit_is_bit_for_bit_the_original_and_ignores_the_short_rule_on_long_windows(design):
    X, Y, blocks = train_rows(design, DAY, 364)
    x_new = design[0].loc[[DAY]]
    expected = reference_fit_predict(X, Y, blocks, x_new)
    fitted = lear.fit_window(X, Y, blocks)
    assert set(fitted) == {"scalers", "rows", "models"} and fitted["rows"] == 364
    assert np.array_equal(lear.predict_window(fitted, x_new, blocks).loc[DAY].to_numpy(), expected)
    with_rule = lear.fit_window(X, Y, blocks, short_rule=lear.SHORT_RULE, slope=1.0, share_gram=False)
    assert np.array_equal(lear.predict_window(with_rule, x_new, blocks).loc[DAY].to_numpy(), expected)
    assert lear.LEAR_WINDOWS == (364, 728, 1092, 1456)


def test_shared_gram_matrix_gives_the_same_forecast(design):
    X, Y, blocks = train_rows(design, DAY, 364)
    x_new = design[0].loc[[DAY]]
    plain = lear.predict_window(lear.fit_window(X, Y, blocks), x_new, blocks)
    fitted = lear.fit_window(X, Y, blocks, share_gram=True)
    assert "offset" in fitted
    shared = lear.predict_window(fitted, x_new, blocks)
    np.testing.assert_allclose(shared.to_numpy(), plain.to_numpy(), rtol=0, atol=lear_recipe.GRAM_TOLERANCE)


# --- short windows ----------------------------------------------------------------------------------------------------------

def test_short_window_is_not_fitted_by_default_and_uses_the_target_variance_when_asked(design):
    X, Y, blocks = train_rows(design, DAY, 56)
    assert len(X) == 56 and len(X) <= X.shape[1] + 1
    assert lear.fit_window(X, Y, blocks) is None
    with pytest.raises(ValueError, match="short_rule"):
        lear.fit_window(X, Y, blocks, short_rule="something else")
    fitted = lear.fit_window(X, Y, blocks, short_rule=lear.SHORT_RULE, share_gram=True)
    assert fitted["rows"] == 56 and "offset" not in fitted  # short windows never take the shared Gram matrix
    forecast = lear.predict_window(fitted, design[0].loc[[DAY]], blocks)
    assert forecast.notna().all().all()
    assert lear.fit_window(X.iloc[: lear.MIN_SHORT_ROWS - 1], Y.iloc[: lear.MIN_SHORT_ROWS - 1], blocks, short_rule=lear.SHORT_RULE) is None


def test_short_window_penalty_is_the_old_scikit_learn_criterion(design):
    """scikit-learn up to 1.0.1: criterion = n * MSE / (var(y) + eps) + 2 * df along the LARS path; its minimiser is the penalty."""
    from sklearn.linear_model import LassoLarsIC, lars_path

    X, Y, blocks = train_rows(design, DAY, 84)
    scalers = lear.fit_scalers(X, Y, blocks)
    Xt = lear.transform(X, blocks, scalers)
    m, s = scalers["target"]
    fitted = lear.fit_window(X, Y, blocks, short_rule=lear.SHORT_RULE)
    for h in (3, 12, 19):
        yt = asinh_forward(Y[h].to_numpy(), m, s)
        assert lear.short_noise_variance(yt) == np.var(yt) + np.finfo("float64").eps
        Xc, yc = Xt - Xt.mean(axis=0), yt - yt.mean()
        alphas, _, coefs = lars_path(Xc, yc, Gram=None, alpha_min=0.0, method="lasso", max_iter=2500)
        mse = np.mean((yc[:, None] - Xc @ coefs) ** 2, axis=0)
        df = (np.abs(coefs) > np.finfo(coefs.dtype).eps).sum(axis=0)
        old = len(yc) * mse / (np.var(yc) + np.finfo("float64").eps) + 2 * df
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic = LassoLarsIC(criterion="aic", max_iter=2500, noise_variance=lear.short_noise_variance(yt)).fit(Xt, yt)
        assert ic.alpha_ == pytest.approx(alphas[np.argmin(old)], rel=1e-9)
        assert fitted["models"][h].alpha == pytest.approx(max(ic.alpha_, 1e-12))
        assert 0 < np.count_nonzero(fitted["models"][h].coef_) < len(yt)


# --- the rolling slope choice -----------------------------------------------------------------------------------------------

def slope_frame(days=70, start="2023-12-23", seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=24 * days, freq="1h", tz="UTC")
    day = pd.Series(pd.DatetimeIndex(index.tz_localize(None).normalize()), index=index)
    hour = pd.Series(np.asarray(index.hour), index=index)
    actual = pd.Series(rng.normal(60, 20, len(index)), index=index)
    forecasts = pd.DataFrame({c: actual + rng.normal(0, 5 + 10 * abs(c - 0.5), len(index)) for c in lear_recipe.SLOPES}, index=index)
    return forecasts, actual, day, hour


def test_rolling_choice_takes_the_lowest_mae_of_the_last_56_days_per_hour():
    forecasts, actual, day, hour = slope_frame()
    chosen, value = lear_recipe.rolling_slope_choice(forecasts, actual, day, hour)
    first = pd.Timestamp("2023-12-23") + pd.Timedelta(days=56)
    assert first == pd.Timestamp("2024-02-17")
    assert chosen[day < first].isna().all() and chosen[day >= first].notna().all()
    for D, h in [(first, 0), (first, 13), (first + pd.Timedelta(days=9), 23)]:
        window = (day >= D - pd.Timedelta(days=56)) & (day <= D - pd.Timedelta(days=1)) & (hour == h)
        assert window.sum() == 56
        errors = forecasts[window].sub(actual[window], axis=0).abs().mean()
        row = (day == D) & (hour == h)
        assert chosen[row].iloc[0] == errors.idxmin()
        assert value[row].iloc[0] == forecasts.loc[row, errors.idxmin()].iloc[0]


def test_rolling_choice_breaks_ties_towards_the_larger_slope():
    forecasts, actual, day, hour = slope_frame()
    forecasts[1.0] = forecasts[0.5]  # identical errors: the larger slope must win wherever 0.5 would
    chosen, _ = lear_recipe.rolling_slope_choice(forecasts, actual, day, hour)
    assert (chosen.dropna() != 0.5).all() and (chosen.dropna() == 1.0).any()
    same = pd.DataFrame({c: forecasts[0.5] for c in lear_recipe.SLOPES})
    chosen, value = lear_recipe.rolling_slope_choice(same, actual, day, hour)
    assert (chosen.dropna() == 1.0).all()


def test_rolling_choice_for_d_reads_nothing_from_d_or_later():
    forecasts, actual, day, hour = slope_frame()
    chosen, value = lear_recipe.rolling_slope_choice(forecasts, actual, day, hour)
    D = pd.Timestamp("2024-02-20")
    rng = np.random.default_rng(9)
    actual2, forecasts2 = actual.copy(), forecasts.copy()
    actual2[day >= D] = rng.normal(900, 300, int((day >= D).sum()))  # the prices of D and later
    forecasts2.loc[day > D] = rng.normal(900, 300, (int((day > D).sum()), forecasts.shape[1]))  # later forecasts
    chosen2, value2 = lear_recipe.rolling_slope_choice(forecasts2, actual2, day, hour)
    on = (day == D).to_numpy()
    assert np.array_equal(chosen2[on].to_numpy(), chosen[on].to_numpy()) and np.array_equal(value2[on].to_numpy(), value[on].to_numpy())
    forecasts3 = forecasts.copy()
    settled = ((day >= D - pd.Timedelta(days=56)) & (day < D)).to_numpy()
    forecasts3.loc[settled, 0.1] = actual[settled]  # a slope that was perfect on the settled days must be chosen for D
    assert (lear_recipe.rolling_slope_choice(forecasts3, actual, day, hour)[0][on] == 0.1).all()
    assert (chosen[on] != 0.1).any()


# --- the new forecast paths and the scrambling test -------------------------------------------------------------------------

def test_scramble_from_touches_only_what_is_unknown_at_the_gate(inputs):
    scrambled = lear_recipe.scramble_from(inputs, DAY, seed=1)
    local = inputs.index.tz_convert("Europe/Paris")
    days = local.normalize().tz_localize(None)
    before = days < DAY - pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(scrambled[before], inputs[before])
    eve = days == DAY - pd.Timedelta(days=1)
    late = eve & (local.hour >= 8)
    assert (scrambled.loc[late, "nuclear_mw"] != inputs.loc[late, "nuclear_mw"]).all()
    pd.testing.assert_series_equal(scrambled.loc[eve & (local.hour < 8), "nuclear_mw"], inputs.loc[eve & (local.hour < 8), "nuclear_mw"])
    pd.testing.assert_frame_equal(scrambled.loc[eve].drop(columns="nuclear_mw"), inputs.loc[eve].drop(columns="nuclear_mw"))
    on = days == DAY
    for column in ("load_fc_mw", "temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"):
        pd.testing.assert_series_equal(scrambled.loc[on, column], inputs.loc[on, column])
    for column in ("price_eur_mwh", "price_ch_eur_mwh", "price_de_lu_eur_mwh", "nuclear_mw", "wind_onshore_mw", "solar_mw"):
        assert (scrambled.loc[on, column] != inputs.loc[on, column]).all()
    later = days > DAY
    assert (scrambled.loc[later].to_numpy() != inputs.loc[later].to_numpy()).all()


@pytest.mark.parametrize("window, slope", [(56, 1.0), (84, 0.3), (364, 0.5)])
def test_new_paths_forecast_d_from_nothing_dated_d_or_later(inputs, design, window, slope):
    X, Y, blocks = design
    base = lear_recipe.forecast_days(X, Y, blocks, [DAY], window, slope)
    assert base.loc[DAY].notna().all()
    X2, Y2, _ = lear.lear_design(lear_recipe.scramble_from(inputs, DAY, seed=4))
    again = lear_recipe.forecast_days(X2, Y2, blocks, [DAY], window, slope)
    np.testing.assert_allclose(again.to_numpy(), base.to_numpy(), rtol=0, atol=1e-9)
    moved = inputs.copy()
    eve = moved.index.tz_convert("Europe/Paris").normalize().tz_localize(None) == DAY - pd.Timedelta(days=1)
    moved.loc[eve, "price_eur_mwh"] += 50.0  # the D-1 price is known at the gate and does matter
    X3, Y3, _ = lear.lear_design(moved)
    assert not np.allclose(lear_recipe.forecast_days(X3, Y3, blocks, [DAY], window, slope).to_numpy(), base.to_numpy())


def test_slope_changes_the_forecast_and_the_ar_design_keeps_prices_and_calendar(design):
    X, Y, blocks = design
    plain = lear_recipe.forecast_days(X, Y, blocks, [DAY], 364, 1.0)
    sloped = lear_recipe.forecast_days(X, Y, blocks, [DAY], 364, 0.3)
    assert not np.allclose(plain.to_numpy(), sloped.to_numpy())
    Xa, used = lear_recipe.ar_design(X, blocks)
    assert list(used) == ["price_d1", "price_d2", "price_d3", "price_d7", "calendar"] and Xa.shape[1] == 4 * 24 + 13
    ar = lear_recipe.forecast_days(Xa, Y, used, [DAY], 364, 1.0)
    assert ar.loc[DAY].notna().all() and not np.allclose(ar.to_numpy(), plain.to_numpy())


# --- the runner -------------------------------------------------------------------------------------------------------------

def test_selection_inputs_stop_before_the_holdout(inputs):
    cut = lear_recipe.selection_inputs(inputs, "2024-03-01")
    assert cut.index.max() == pd.Timestamp("2024-02-29T22:00Z")  # 23:00 Paris on 29 February
    with pytest.raises(ValueError, match="holdout"):
        lear_recipe.selection_inputs(inputs, "2026-07-02")


def test_chunks_are_written_once_and_resumed(inputs, tmp_path, monkeypatch):
    first = "2024-04-29"
    jobs = lear_recipe.job_list("full", [1.0, 0.5], [56], first, end="2024-05-01") + lear_recipe.job_list("ar", [1.0], [364], first, end="2024-05-01")
    assert len(jobs) == 3 and jobs[0] == ("full", 1.0, 56, pd.Period("2024-04", "M"))
    calls = []
    real = lear_recipe.forecast_days
    monkeypatch.setattr(lear_recipe, "forecast_days", lambda *a, **k: calls.append(a[4:]) or real(*a, **k))
    monkeypatch.setattr(lear_recipe, "chunk_days", lambda month, start, end=None: pd.DatetimeIndex(["2024-04-29", "2024-04-30"]))
    log = lambda *a, **k: None  # noqa: E731
    assert lear_recipe.run_jobs(inputs, tmp_path, jobs, first, shard=(0, 2), log=log) == 0
    assert len(calls) == 2  # this shard holds jobs 0 and 2
    assert lear_recipe.run_jobs(inputs, tmp_path, jobs, first, shard=(1, 2), budget_s=-1.0, log=log) == 1  # no time left: nothing runs
    assert lear_recipe.run_jobs(inputs, tmp_path, jobs, first, log=log) == 0
    assert len(calls) == 3  # only the missing chunk was run
    path = lear_recipe.chunk_path(tmp_path, "full", 0.5, 56, pd.Period("2024-04", "M"))
    assert path.name == "2024-04.csv" and path.parent.name == "w56" and path.parent.parent.name == "c0.5"
    table = pd.read_csv(path, index_col=0, parse_dates=[0])
    assert list(table.index) == [pd.Timestamp("2024-04-29"), pd.Timestamp("2024-04-30")] and table.shape == (2, 24)
    X, Y, blocks = lear.lear_design(lear_recipe.selection_inputs(inputs))
    direct = real(X, Y, blocks, [pd.Timestamp("2024-04-30")], 56, 0.5)
    np.testing.assert_allclose(table.loc["2024-04-30"].to_numpy(), direct.to_numpy()[0], rtol=0, atol=1e-9)


def test_window_average_and_hourly_layout():
    days = pd.date_range("2025-10-25", periods=3)  # the autumn clock change is on the 26th
    a = pd.DataFrame(np.arange(72, dtype=float).reshape(3, 24), index=days, columns=range(24))
    b = a + 10.0
    b.iloc[1] = np.nan  # a window with no forecast for a day drops out of that day's mean
    mean = lear_recipe.window_average([a, b])
    assert mean.loc[days[0], 5] == a.loc[days[0], 5] + 5.0 and mean.loc[days[1], 5] == a.loc[days[1], 5]
    index = pd.date_range("2025-10-24T22:00Z", "2025-10-27T23:00Z", freq="1h", inclusive="left")
    local = index.tz_convert("Europe/Paris")
    info = pd.DataFrame({"delivery_day": local.normalize().tz_localize(None), "hour": local.hour}, index=index)
    hourly = lear_recipe.to_hourly(info, a)
    assert len(hourly) == 73
    two = hourly[(info["delivery_day"] == days[1]) & (info["hour"] == 2)]
    assert len(two) == 2 and (two == a.loc[days[1], 2]).all()


# --- the rule ---------------------------------------------------------------------------------------------------------------

def rule_frame(days=120, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=24 * days, freq="1h", tz="UTC")
    day = pd.DatetimeIndex(index.tz_localize(None).normalize())
    actual = 60 + rng.normal(0, 20, len(index))
    frame = pd.DataFrame({"delivery_day": day, "hour": index.hour, "dow": day.dayofweek, "holiday": 0, "is_peak": 0, "actual": actual,
                          "strict": True, "wind_mw": rng.uniform(1000, 9000, len(index)), "solar_mw": rng.uniform(0, 5000, len(index))}, index=index)
    frame.loc[frame["delivery_day"] == pd.Timestamp("2025-01-01"), "holiday"] = 1
    frame["naive_day"], frame["naive_week"] = frame["actual"].shift(24), frame["actual"].shift(168)
    noise = rng.normal(0, 10, len(index))
    frame["L0"] = actual + noise
    frame["A"] = actual + 0.9 * noise  # about 0.8 better everywhere
    frame["B"] = actual + 0.99 * noise  # better, but by less than 0.25
    frame["C"] = actual + 0.9 * noise
    frame.loc[frame["dow"] >= 5, "C"] = actual[frame["dow"] >= 5] + 1.4 * noise[frame["dow"] >= 5]  # weekends much worse
    return frame


def test_rule_applies_the_three_conditions_and_picks_the_lowest_passing_mae():
    frame = rule_frame()
    out = lear_recipe.apply_rule(frame)
    assert out["baseline_mae"] == pytest.approx(yardstick.mae(frame, "L0"))
    a, b, c = (out["candidates"][k] for k in "ABC")
    assert a["passes"] and all(a["conditions"].values()) and a["gain"] == pytest.approx(0.1 * out["baseline_mae"])
    assert b["conditions"] == {"i_mae_at_least_0.25_below_L0": False, "ii_no_slice_worse_by_more_than_1.0": True,
                               "iii_one_sided_98.33_upper_below_zero": True} and not b["passes"]
    assert not c["conditions"]["ii_no_slice_worse_by_more_than_1.0"] and c["slice_differences"]["weekends"] > 1.0 and not c["passes"]
    assert a["paired_vs_L0"]["one_sided_level"] == pytest.approx(1 - 0.05 / 3)
    assert a["paired_vs_L0"]["upper_one_sided"] == pytest.approx(
        yardstick.paired_daily(frame, "A", "L0", one_sided_level=1 - 0.05 / 3)["upper_one_sided"])
    assert out["lear_r"] == "A" and "LEAR-R" in out["finding"]
    assert set(a["slice_differences"]) == {"weekends", "holidays", "windiest 10%", "sunniest 10%", "top 5% hours"}
    none = lear_recipe.apply_rule(frame, candidates=("B", "C"))
    assert none["lear_r"] is None and "null" in none["finding"]


def test_recent_volatility_is_the_spread_of_the_seven_days_before():
    price = pd.Series(np.arange(20, dtype=float) ** 2, index=pd.date_range("2025-01-01", periods=20))
    D = pd.Timestamp("2025-01-15")
    vol = lear_recipe.recent_volatility(price, [D, pd.Timestamp("2025-01-05")])
    assert vol.iloc[0] == pytest.approx(np.std(price["2025-01-08":"2025-01-14"].to_numpy(), ddof=1))
    assert np.isnan(vol.iloc[1])  # fewer than seven earlier days
    changed = price.copy()
    changed[changed.index >= D] = 1e6
    assert lear_recipe.recent_volatility(changed, [D]).iloc[0] == vol.iloc[0]


def test_reproduction_check_compares_at_the_printed_precision():
    frame = rule_frame()
    frame["honest_v2"], frame["S1"] = frame["L0"], frame["L0"]
    out = lear_recipe.reproduction(frame)
    assert not out["ok"] and out["honest_v2"]["expected"] == "15.7040" and out["L0"]["expected"] == "15.36"
    assert out["strict_hours"] == len(frame)


def test_monthly_gbm_files_give_the_frame_of_one_walk_forward(inputs, tmp_path, monkeypatch):
    from thermo_fr.forecast import models

    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 20)
    quiet = lambda *a, **k: None  # noqa: E731
    whole = lear_recipe.gbm_honest_v2(inputs, "2024-03-01", "2024-05-01", log=quiet)
    parts = lear_recipe.gbm_honest_v2(inputs, "2024-03-01", "2024-05-01", log=quiet, out=tmp_path)
    assert sorted(p.name for p in (tmp_path / "gbm").iterdir()) == ["honest_v2_2024-03.csv", "honest_v2_2024-04.csv"]
    assert len(parts) == len(whole) and (parts.index == whole.index).all()
    np.testing.assert_allclose(parts["gbm"].to_numpy(), whole["gbm"].to_numpy(), rtol=0, atol=1e-12)
    monkeypatch.setattr(lear_recipe, "walk_forward", lambda *a, **k: pytest.fail("a month on disk must not be refitted"))
    again = lear_recipe.gbm_honest_v2(inputs, "2024-03-01", "2024-05-01", log=quiet, out=tmp_path)
    assert np.array_equal(again["gbm"].to_numpy(), parts["gbm"].to_numpy())
