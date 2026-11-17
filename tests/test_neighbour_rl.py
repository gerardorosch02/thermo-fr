"""Neighbours' load and residual load in LEAR (forecast/neighbour_rl.py): the data rules and the fitting path, on small synthetic tables."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from thermo_fr.forecast import lear, lear_recipe, neighbour_rl as nr  # noqa: E402
from thermo_fr.forecast.timing import delivery_days  # noqa: E402


def _synthetic(days: int = 80, seed: int = 0):
    """A small design (one price-lag block of 24 columns and two dummies), the 24 prices of each day and one zone table."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    zone = pd.DataFrame(rng.normal(50, 10, (days, 24)), index=index, columns=lear.HOURS)
    noise = rng.normal(0, 1, (days, 24))
    price = pd.DataFrame(40 + 0.8 * (zone.to_numpy() - 50) + noise, index=index, columns=lear.HOURS)
    lag = price.shift(1)
    cols = [f"price_d1_h{h:02d}" for h in lear.HOURS]
    design = pd.DataFrame(lag.to_numpy(), index=index, columns=cols)
    design["dow_0"] = (index.dayofweek == 0).astype(float)
    design["dow_1"] = (index.dayofweek == 1).astype(float)
    blocks = {"price_d1": cols, "calendar": ["dow_0", "dow_1"]}
    return design, price, blocks, zone


def test_fill_takes_the_same_hour_of_an_earlier_day_at_most_seven_days_back():
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    table = pd.DataFrame(1.0, index=index, columns=lear.HOURS)
    table.iloc[0, 5] = 7.0
    table.iloc[1:10, 5] = np.nan  # nine missing days after one value
    table.iloc[3, 8] = np.nan
    filled, count = nr._filled(table)
    assert (filled.iloc[1:8, 5] == 7.0).all()  # seven days back at most
    assert filled.iloc[8:10, 5].isna().all()
    assert filled.iloc[3, 8] == 1.0
    assert count == 8


def test_the_fill_never_reads_a_later_day():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    table = pd.DataFrame(np.nan, index=index, columns=lear.HOURS)
    table.iloc[4] = 9.0
    filled, count = nr._filled(table)
    assert filled.iloc[:4].isna().all().all() and count == 0


def test_residual_load_subtracts_wind_and_solar_and_counts_a_missing_offshore_series_as_zero():
    hours = pd.date_range("2024-01-09 23:00", periods=48, freq="h", tz="UTC")  # two Paris days
    info = pd.DataFrame({"delivery_day": delivery_days(hours), "hour": hours.tz_convert("Europe/Paris").hour}, index=hours)
    data = {}
    for z in nr.ZONES:
        data[f"load_fc_{z}_mw"] = 100.0
        data[f"solar_fc_{z}_mw"] = 10.0
        data[f"wind_onshore_fc_{z}_mw"] = 20.0
        data[f"wind_offshore_fc_{z}_mw"] = 5.0 if z == "de_lu" else np.nan
    neighbours = pd.DataFrame(data, index=hours)
    all_days = pd.date_range("2024-01-10", periods=2, freq="D")
    load, _ = nr.zone_tables(neighbours, info, all_days, "load")
    residual, _ = nr.zone_tables(neighbours, info, all_days, "residual")
    assert (load["es"] == 100.0).all().all()
    assert (residual["de_lu"] == 65.0).all().all()
    assert (residual["es"] == 70.0).all().all()
    with pytest.raises(ValueError):
        nr.zone_tables(neighbours, info, all_days, "prices")


def test_no_added_block_is_the_present_lear_fit():
    design, price, blocks, _ = _synthetic()
    day = design.index[-1]
    mine, kept = nr.fit_predict_day(design, price, blocks, {}, day, 60, start=design.index[1])
    present = lear._fit_predict_window(design, price, blocks, day, 60, design.index[1], [day])
    assert kept == {}
    np.testing.assert_array_equal(mine.to_numpy(), present.loc[day].to_numpy())


def test_a_zone_that_drives_the_price_is_kept_and_improves_the_forecast():
    design, price, blocks, zone = _synthetic()
    day = design.index[-1]
    with_zone, kept = nr.fit_predict_day(design, price, blocks, {"de_lu": zone}, day, 60, start=design.index[1])
    without, _ = nr.fit_predict_day(design, price, blocks, {}, day, 60, start=design.index[1])
    actual = price.loc[day]
    assert kept["de_lu"] >= 20  # of 24 hourly models
    assert (with_zone - actual).abs().mean() < 0.5 * (without - actual).abs().mean()


def test_the_model_for_hour_h_sees_only_hour_h_of_the_zone():
    design, price, blocks, zone = _synthetic()
    day = design.index[-1]
    base, _ = nr.fit_predict_day(design, price, blocks, {"de_lu": zone}, day, 60, start=design.index[1])
    changed = zone.copy()
    first, last = lear.window_days(day, 60, design.index[1])
    # hour 7 reordered across the training days: the zone's median and MAD (taken over all 24 hours together) stay the same
    changed.loc[first:last, 7] = zone.loc[first:last, 7].to_numpy()[::-1]
    again, _ = nr.fit_predict_day(design, price, blocks, {"de_lu": changed}, day, 60, start=design.index[1])
    others = [h for h in lear.HOURS if h != 7]
    np.testing.assert_array_equal(base[others].to_numpy(), again[others].to_numpy())
    assert base[7] != again[7]


def test_the_forecast_day_itself_and_later_days_do_not_enter_the_fit():
    design, price, blocks, zone = _synthetic()
    day = design.index[-10]
    base, _ = nr.fit_predict_day(design, price, blocks, {"de_lu": zone}, day, 60, start=design.index[1])
    price2, zone2 = price.copy(), zone.copy()
    price2.loc[day:] = 1e6  # the prices of D and later
    zone2.loc[day + pd.Timedelta(days=1):] = -1e6  # the zone's values after D
    again, _ = nr.fit_predict_day(design, price2, blocks, {"de_lu": zone2}, day, 60, start=design.index[1])
    np.testing.assert_array_equal(base.to_numpy(), again.to_numpy())


def test_a_day_with_a_missing_zone_value_has_no_forecast_and_a_training_day_with_one_is_dropped():
    design, price, blocks, zone = _synthetic()
    day = design.index[-1]
    hole = zone.copy()
    hole.loc[day, 3] = np.nan
    out, _ = nr.fit_predict_day(design, price, blocks, {"de_lu": hole}, day, 60, start=design.index[1])
    assert out.isna().all()
    hole = zone.copy()
    hole.loc[design.index[-20], 3] = np.nan
    out, _ = nr.fit_predict_day(design, price, blocks, {"de_lu": hole}, day, 60, start=design.index[1])
    assert out.notna().all()


def test_scramble_leaves_the_day_and_everything_before_it():
    hours = pd.date_range("2024-03-01", periods=24 * 6, freq="h", tz="UTC")
    neighbours = pd.DataFrame({"load_fc_de_lu_mw": np.arange(len(hours), dtype=float), "wind_offshore_fc_ch_mw": np.nan}, index=hours)
    day = pd.Timestamp("2024-03-03")
    noisy = nr.scramble_neighbours(neighbours, day, seed=1)
    kept = delivery_days(hours) <= day
    pd.testing.assert_series_equal(noisy.loc[kept, "load_fc_de_lu_mw"], neighbours.loc[kept, "load_fc_de_lu_mw"])
    assert (noisy.loc[~kept, "load_fc_de_lu_mw"] != neighbours.loc[~kept, "load_fc_de_lu_mw"]).all()
    assert noisy["wind_offshore_fc_ch_mw"].isna().all()


def test_the_named_scramble_days_and_the_job_list():
    days = nr.scramble_days()
    assert len(days) == 20 and set(pd.DatetimeIndex(nr.SCRAMBLE_NAMED)) <= set(days)
    assert days.min() >= nr.SCORED_START and days.max() < pd.Timestamp(nr.TEST_END)
    jobs = nr.job_list()
    assert len(jobs) == len(nr.months()) * len(nr.CANDIDATES) * len(nr.WINDOWS)
    assert nr.chunk_days(nr.months()[0])[0] == nr.SCORED_START
    assert nr.ONE_SIDED_LEVEL == pytest.approx(0.975) and nr.IDEALISED == ("I",)


def test_the_idealised_candidate_can_never_pass():
    index = pd.date_range("2024-02-17 23:00", periods=24 * 40, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    actual = 60 + rng.normal(0, 20, len(index))
    frame = pd.DataFrame({"delivery_day": delivery_days(index), "hour": index.tz_convert("Europe/Paris").hour, "actual": actual, "strict": True,
                          "dow": delivery_days(index).dayofweek, "holiday": (delivery_days(index).day == 1).astype(int), "is_peak": 0, "weather_point_in_time": True,
                          "wind_mw": rng.uniform(1000, 9000, len(index)), "solar_mw": rng.uniform(0, 5000, len(index)),
                          "L0": actual + rng.normal(0, 15, len(index)), "naive_day": actual + rng.normal(0, 25, len(index)),
                          "naive_week": actual + rng.normal(0, 30, len(index))}, index=index)
    frame["H"] = actual + rng.normal(0, 5, len(index))
    frame["I"] = actual + rng.normal(0, 5, len(index))
    rule = nr.apply_rule(frame)
    assert rule["candidates"]["H"]["passes"] is True
    assert "passes" not in rule["candidates"]["I"] and rule["candidates"]["I"]["idealised"] is True
    assert rule["candidates"]["I"]["verdict"].startswith("ceiling met")
    frame["I"] = frame["L0"] + 0.01
    assert nr.apply_rule(frame)["candidates"]["I"]["verdict"].startswith("ceiling not met")
    assert lear_recipe.TEST_END == nr.TEST_END
