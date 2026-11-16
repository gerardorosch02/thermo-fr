"""The yardstick measurements (forecast/yardstick.py): each function against numbers worked out by hand."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from thermo_fr.forecast import yardstick
from thermo_fr.forecast.shape import block_bootstrap_mean


def hourly_frame(days=21, start="2025-09-22", seed=0) -> pd.DataFrame:
    """Hourly rows for whole UTC days (treated as delivery days), with an actual price, the D-1 and D-7 naives and a forecast."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=24 * days, freq="1h", tz="UTC")
    day = pd.DatetimeIndex(index.tz_localize(None).normalize())
    hour = np.asarray(index.hour)
    actual = 60 + 20 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 8, len(index))
    frame = pd.DataFrame({"delivery_day": day, "hour": hour, "actual": actual, "strict": True}, index=index)
    frame["naive_day"] = frame["actual"].shift(24)
    frame["naive_week"] = frame["actual"].shift(24 * 7)
    frame["model"] = frame["actual"] + rng.normal(0, 3, len(index))
    return frame


def test_lago_naive_takes_d1_tuesday_to_friday_and_d7_otherwise():
    frame = hourly_frame()
    lago = yardstick.lago_naive(frame)
    weekday = pd.DatetimeIndex(frame["delivery_day"]).dayofweek
    for k in range(7):
        rows = (weekday == k) & frame["naive_week"].notna().to_numpy()
        expected = frame["naive_day"] if k in (1, 2, 3, 4) else frame["naive_week"]
        assert (lago[rows] == expected[rows]).all() and rows.sum() == 48
    assert pd.Timestamp("2025-09-22").dayofweek == 0  # the frame starts on a Monday


def test_relative_mae_divides_by_each_naive_on_common_strict_rows():
    frame = hourly_frame()
    frame.loc[frame.index[-24:], "strict"] = False  # the last day is not scored
    out = yardstick.relative_mae(frame, "model")
    rows = frame[frame["strict"] & frame["naive_week"].notna()]
    assert out["hours"] == len(rows) == 24 * 13
    own = (rows["model"] - rows["actual"]).abs().mean()
    assert out["mae"] == pytest.approx(own)
    assert out["rmae_naive_day"] == pytest.approx(own / (rows["naive_day"] - rows["actual"]).abs().mean())
    assert out["rmae_naive_week"] == pytest.approx(own / (rows["naive_week"] - rows["actual"]).abs().mean())
    lago = yardstick.lago_naive(rows)
    assert out["rmae_naive_lago"] == pytest.approx(own / (lago - rows["actual"]).abs().mean())
    assert yardstick.relative_mae(frame, "naive_day")["rmae_naive_day"] == pytest.approx(1.0)


def test_ols_slope_by_hand():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    out = yardstick.ols_slope(x, 1.5 + 2.0 * x)
    assert out["slope"] == pytest.approx(2.0) and out["intercept"] == pytest.approx(1.5) and out["correlation"] == pytest.approx(1.0)
    assert out["n"] == 4
    assert np.isnan(yardstick.ols_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["slope"])  # no variation in x


def test_under_reaction_slope_is_two_when_the_forecast_moves_half_as_far():
    frame = hourly_frame()
    frame["half"] = frame["naive_day"] + 0.5 * (frame["actual"] - frame["naive_day"])
    frame["double"] = frame["naive_day"] + 2.0 * (frame["actual"] - frame["naive_day"])
    half, double = yardstick.under_reaction(frame, "half"), yardstick.under_reaction(frame, "double")
    for part in ("hourly", "daily"):
        assert half[part]["slope"] == pytest.approx(2.0) and half[part]["correlation"] == pytest.approx(1.0)
        assert double[part]["slope"] == pytest.approx(0.5)
    assert half["hourly"]["n"] == 24 * 20 and half["daily"]["n"] == 20  # the first day has no D-1
    # a forecast with independent noise around the truth: the slope falls below one
    assert yardstick.under_reaction(frame, "model")["hourly"]["slope"] < 1.0


def test_paired_daily_uses_the_project_bootstrap_and_the_one_sided_level():
    frame = hourly_frame()
    frame["worse"] = frame["model"] + 4.0 * np.sign(frame["model"] - frame["actual"])  # 4 further from the truth every hour
    out = yardstick.paired_daily(frame, "worse", "model", one_sided_level=1 - 0.05 / 3)
    assert out["days"] == 21 and out["mean"] == pytest.approx(4.0) and out["share_days_better"] == 0.0
    diff = (yardstick.daily_abs_error(frame, "worse") - yardstick.daily_abs_error(frame, "model")).to_numpy()
    assert out["ci95"] == pytest.approx(list(block_bootstrap_mean(diff, block=7, resamples=5000, seed=0, level=0.95)))
    assert out["upper_one_sided"] == pytest.approx(block_bootstrap_mean(diff, block=7, resamples=5000, seed=0, level=1 - 0.1 / 3)[1])
    noisy = yardstick.paired_daily(frame, "naive_day", "model", one_sided_level=0.9833)
    assert noisy["ci95"][0] < noisy["upper_one_sided"]  # the one-sided 98.33% end lies above the lower two-sided end
    assert yardstick.paired_daily(frame, "model", "worse")["mean"] == pytest.approx(-4.0)


def test_spearman_matches_scipy_with_ties_and_is_nan_for_a_flat_profile():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=24), rng.normal(size=24)
    b[5] = b[6] = b[7]
    assert yardstick.spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic)
    assert yardstick.spearman(a, a) == pytest.approx(1.0) and yardstick.spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(yardstick.spearman(a, np.ones(24)))


def test_spread_and_shape_by_hand():
    actual = np.arange(24, dtype=float)  # lowest four hours 0..3 (mean 1.5), highest four 20..23 (mean 21.5)
    assert yardstick.spread_of(actual, actual) == pytest.approx(20.0)
    assert yardstick.spread_of(-actual, actual) == pytest.approx(-20.0)
    forecast = actual.copy()
    forecast[[20, 21, 22, 23]] = [0.5, 1.5, 2.5, 3.5]  # the forecast's top four are 16..19, its bottom four 0, 20, 1, 21
    assert yardstick.spread_of(forecast, actual) == pytest.approx(np.mean([16, 17, 18, 19]) - np.mean([0, 20, 1, 21]))
    frame = hourly_frame(days=10)
    frame["perfect"], frame["reversed"] = frame["actual"], -frame["actual"]
    assert yardstick.shape(frame, "perfect") == {"days": 10, "days_with_rank_correlation": 10, "spearman_mean": pytest.approx(1.0),
                                                 "spearman_median": pytest.approx(1.0), "spread_capture": pytest.approx(1.0)}
    assert yardstick.shape(frame, "reversed")["spread_capture"] == pytest.approx(-1.0)
    d1 = yardstick.shape(frame, "naive_day")
    assert d1["days"] == 9 and 0 < d1["spread_capture"] < 1 and 0 < d1["spearman_mean"] < 1
    # the capture is a ratio of sums over days, not a mean of daily ratios
    rows = frame.dropna(subset=["naive_day"])
    earned = sum(yardstick.spread_of(p["naive_day"].to_numpy(), p["actual"].to_numpy()) for _, p in rows.groupby("delivery_day"))
    best = sum(yardstick.spread_of(p["actual"].to_numpy(), p["actual"].to_numpy()) for _, p in rows.groupby("delivery_day"))
    assert d1["spread_capture"] == pytest.approx(earned / best)


def test_split_at_the_quarter_hour_auction():
    frame = hourly_frame()  # 2025-09-22 to 2025-10-12
    parts = yardstick.split(frame)
    assert parts["before"]["delivery_day"].max() == pd.Timestamp("2025-09-30")
    assert parts["from"]["delivery_day"].min() == pd.Timestamp("2025-10-01")
    assert len(parts["before"]) + len(parts["from"]) == len(frame)


def test_tightness_is_the_peak_residual_minus_early_nuclear_in_gw():
    index = pd.date_range("2025-01-01", periods=72, freq="1h")
    day = index.normalize()
    load = pd.Series(50_000.0, index=index)
    load.iloc[18] = 70_000.0  # the peak of day 1
    wind = pd.Series(5_000.0, index=index)
    solar = pd.Series(1_000.0, index=index)
    nuclear = pd.Series(np.repeat([40_000.0, 45_000.0, 30_000.0], 24), index=index)
    solar.iloc[60] = np.nan  # day 3 is incomplete
    out = yardstick.tightness(load, wind, solar, nuclear, day)
    assert out.iloc[0] == pytest.approx((70_000 - 5_000 - 1_000 - 40_000) / 1000)
    assert out.iloc[1] == pytest.approx((50_000 - 6_000 - 45_000) / 1000)
    assert np.isnan(out.iloc[2])


def test_terciles_split_into_three_equal_groups():
    values = pd.Series(np.arange(9, dtype=float), index=pd.date_range("2025-01-01", periods=9))
    values.iloc[4] = np.nan
    out = yardstick.terciles(values)
    assert out.dropna().tolist() == [1, 1, 1, 2, 2, 3, 3, 3] and np.isnan(out.iloc[4])


def test_weather_effect_recovers_the_coefficients_by_tercile():
    rng = np.random.default_rng(5)
    n = 600
    table = pd.DataFrame({"wind_surprise_gw": rng.normal(0, 1.5, n), "solar_surprise_gw": rng.normal(0, 1.0, n)})
    group = pd.Series(np.repeat([1, 2, 3], n // 3))
    slope = np.where(group == 3, -6.0, -1.0)  # tight days react six times as strongly to wind
    table["error"] = 2.0 + slope * table["wind_surprise_gw"] - 3.0 * table["solar_surprise_gw"] + rng.normal(0, 0.01, n)
    out = yardstick.weather_effect(table, group)
    assert out["tercile_1"]["wind_eur_per_gw"] == pytest.approx(-1.0, abs=0.01)
    assert out["tercile_3"]["wind_eur_per_gw"] == pytest.approx(-6.0, abs=0.01)
    assert out["tercile_2"]["solar_eur_per_gw"] == pytest.approx(-3.0, abs=0.01)
    assert out["tercile_3"]["intercept"] == pytest.approx(2.0, abs=0.01) and out["tercile_3"]["r2"] > 0.999
    assert out["pooled"]["n"] == n and out["tercile_3"]["n"] == n // 3
    assert -6.0 < out["pooled"]["wind_eur_per_gw"] < -1.0 and out["pooled"]["r2"] < out["tercile_3"]["r2"]
    exact = yardstick.regress([1.0, 3.0, 5.0, 7.0], np.array([[0.0], [1.0], [2.0], [3.0]]))
    assert exact["coefficients"] == pytest.approx([2.0]) and exact["intercept"] == pytest.approx(1.0) and exact["r2"] == pytest.approx(1.0)
