"""French residual load in LEAR (forecast/french_rl.py): the series, the partial-history regressor and the rule, on small synthetic tables."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from thermo_fr.forecast import french_rl as fr, lear, neighbour_rl as nr  # noqa: E402
from thermo_fr.forecast.timing import delivery_days  # noqa: E402


def _synthetic(days: int = 90, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    zone = pd.DataFrame(rng.normal(50, 10, (days, 24)), index=index, columns=lear.HOURS)
    price = pd.DataFrame(40 + 0.8 * (zone.to_numpy() - 50) + rng.normal(0, 1, (days, 24)), index=index, columns=lear.HOURS)
    cols = [f"price_d1_h{h:02d}" for h in lear.HOURS]
    design = pd.DataFrame(price.shift(1).to_numpy(), index=index, columns=cols)
    design["dow_0"] = (index.dayofweek == 0).astype(float)
    design["dow_1"] = (index.dayofweek == 1).astype(float)
    return design, price, {"price_d1": cols, "calendar": ["dow_0", "dow_1"]}, zone


def _hourly(days: int = 6, start: str = "2024-04-28 22:00"):
    hours = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    info = pd.DataFrame({"delivery_day": delivery_days(hours), "hour": hours.tz_convert("Europe/Paris").hour}, index=hours)
    hourly = pd.DataFrame({"load_fc_mw": 100.0, "wind_proxy_mw": 20.0, "solar_proxy_mw": 10.0, "solar_fc_mw": 12.0,
                           "wind_onshore_fc_mw": 25.0, "wind_offshore_fc_mw": np.nan}, index=hours)
    before = np.asarray(info["delivery_day"] < fr.HONEST_START)
    hourly.loc[before, ["wind_proxy_mw", "solar_proxy_mw"]] = np.nan
    all_days = pd.DatetimeIndex(sorted(info["delivery_day"].unique()))
    return hourly, info, all_days


def test_the_four_french_series():
    hourly, info, all_days = _hourly()
    tables, replaced = fr.french_tables(hourly, info, all_days)
    early, late = all_days < fr.HONEST_START, all_days >= fr.HONEST_START
    assert tables["honest"][early].isna().all().all() and (tables["honest"][late] == 70.0).all().all()
    assert (tables["ideal"] == 63.0).all().all()  # a missing offshore value counts as zero
    assert (tables["q"][early] == 63.0).all().all() and (tables["q"][late] == 70.0).all().all()
    assert tables["m"][early].isna().all().all() and (tables["m"][late] == 63.0).all().all()
    assert replaced["load_fc_mw"] == 0


def test_q_never_reads_the_operator_forecast_from_the_honest_start():
    hourly, info, all_days = _hourly()
    changed = hourly.copy()
    late = np.asarray(info["delivery_day"] >= fr.HONEST_START)
    changed.loc[late, list(fr.OPERATOR_COLUMNS)] = 1e6
    a, _ = fr.french_tables(hourly, info, all_days)
    b, _ = fr.french_tables(changed, info, all_days)
    pd.testing.assert_frame_equal(a["q"], b["q"])
    pd.testing.assert_frame_equal(a["honest"], b["honest"])


def test_a_partial_regressor_with_too_few_days_is_left_out_and_gives_the_plain_fit():
    design, price, blocks, zone = _synthetic()
    day = design.index[-1]
    partial = zone.copy()
    partial.iloc[:-10] = np.nan  # nine training days with it
    out, kept = nr.fit_predict_day(design, price, blocks, {"fr": partial}, day, 80, start=design.index[1], partial=("fr",), min_partial_days=28)
    plain, _ = nr.fit_predict_day(design, price, blocks, {}, day, 80, start=design.index[1])
    assert kept == {}
    np.testing.assert_array_equal(out.to_numpy(), plain.to_numpy())
    missing_today = zone.copy()
    missing_today.loc[day, 5] = np.nan
    out, kept = nr.fit_predict_day(design, price, blocks, {"fr": missing_today}, day, 80, start=design.index[1], partial=("fr",))
    assert kept == {}
    np.testing.assert_array_equal(out.to_numpy(), plain.to_numpy())


def test_a_partial_regressor_keeps_the_days_before_it_exists_in_the_fit_and_helps():
    design, price, blocks, zone = _synthetic()
    day = design.index[-1]
    partial = zone.copy()
    partial.iloc[:40] = np.nan
    with_it, kept = nr.fit_predict_day(design, price, blocks, {"fr": partial}, day, 80, start=design.index[1], partial=("fr",))
    plain, _ = nr.fit_predict_day(design, price, blocks, {}, day, 80, start=design.index[1])
    actual = price.loc[day]
    assert kept["fr"] >= 20
    assert (with_it - actual).abs().mean() < 0.7 * (plain - actual).abs().mean()
    # a day without the regressor is still a training day: changing its price changes the forecast
    price2 = price.copy()
    price2.iloc[20] += 30.0
    again, _ = nr.fit_predict_day(design, price2, blocks, {"fr": partial}, day, 80, start=design.index[1], partial=("fr",))
    assert not np.allclose(again.to_numpy(), with_it.to_numpy())
    # without `partial` the same days are dropped, which is the neighbours test's rule
    dropped, _ = nr.fit_predict_day(design, price2, blocks, {"fr": partial}, day, 80, start=design.index[1])
    assert not np.allclose(dropped.to_numpy(), again.to_numpy())


def test_the_values_of_the_missing_days_do_not_matter_and_later_days_are_not_read():
    design, price, blocks, zone = _synthetic()
    day = design.index[-5]
    partial = zone.copy()
    partial.iloc[:40] = np.nan
    base, _ = nr.fit_predict_day(design, price, blocks, {"fr": partial}, day, 80, start=design.index[1], partial=("fr",))
    later = partial.copy()
    later.loc[day + pd.Timedelta(days=1):] = -1e6
    price2 = price.copy()
    price2.loc[day:] = 1e6
    again, _ = nr.fit_predict_day(design, price2, blocks, {"fr": later}, day, 80, start=design.index[1], partial=("fr",))
    np.testing.assert_array_equal(base.to_numpy(), again.to_numpy())


def test_candidates_windows_jobs_and_levels():
    assert fr.HONEST == ("P", "Q") and fr.IDEALISED == ("F", "FI", "M")
    assert fr.WINDOWS_OF["P"] == tuple(nr.WINDOWS) and fr.WINDOWS_OF["FI"] == (728,)
    assert fr.ONE_SIDED_LEVEL == pytest.approx(1 - 0.05 / 3)
    jobs = fr.job_list()
    assert len(jobs) == 2 * 4 * len(fr.months("P")) + len(fr.months("M")) + 2 * len(fr.months("F"))
    assert fr.chunk_days("P", fr.months("P")[0])[0] == fr.HONEST_START and fr.chunk_days("F", fr.months("F")[0])[0] == nr.SCORED_START
    assert len(pd.date_range(fr.RULE_START, pd.Timestamp(nr.TEST_END) - pd.Timedelta(days=1), freq="D")) == 763
    days = fr.scramble_days()
    assert len(days) == 20 and set(pd.DatetimeIndex(fr.SCRAMBLE_NAMED)) <= set(days)
    assert set(fr.extra_for("FI", {"ideal": 1}, {"de_lu": 2, "be": 3})) == {"fr", "de_lu", "be"}


def test_the_operator_forecast_of_the_day_is_put_back_only_for_that_day():
    hourly, info, _ = _hourly()
    scrambled = hourly.copy()
    scrambled[list(fr.OPERATOR_COLUMNS)] = -5.0
    day = pd.Timestamp("2024-05-01")
    out = fr.keep_operator_forecast_of(scrambled, hourly, day)
    same = np.asarray(info["delivery_day"] == day)
    assert (out.loc[same, "solar_fc_mw"] == 12.0).all() and (out.loc[~same, "solar_fc_mw"] == -5.0).all()


def _rule_frame(seed: int = 0):
    index = pd.date_range("2024-02-17 23:00", periods=24 * 200, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    actual = 60 + rng.normal(0, 20, len(index))
    days = delivery_days(index)
    frame = pd.DataFrame({"delivery_day": days, "hour": index.tz_convert("Europe/Paris").hour, "actual": actual, "strict": True,
                          "dow": days.dayofweek, "holiday": (days.day == 1).astype(int), "is_peak": 0, "weather_point_in_time": True,
                          "wind_mw": rng.uniform(1000, 9000, len(index)), "solar_mw": rng.uniform(0, 5000, len(index)),
                          "L0": actual + rng.normal(0, 15, len(index)), "naive_day": actual + rng.normal(0, 25, len(index)),
                          "naive_week": actual + rng.normal(0, 30, len(index))}, index=index)
    frame["F_w728"] = actual + rng.normal(0, 10, len(index))
    return frame, actual, rng


def test_the_rule_for_the_honest_candidates_and_the_neighbours():
    frame, actual, rng = _rule_frame()
    frame["P"] = actual + rng.normal(0, 5, len(frame))
    frame["Q"] = frame["L0"] + 0.01
    frame["FI_w728"] = actual + rng.normal(0, 4, len(frame))
    rule = frame[frame["delivery_day"] >= fr.RULE_START]
    out = fr.apply_rule(frame, rule)
    assert out["candidates"]["P"]["passes"] is True and out["candidates"]["Q"]["passes"] is False and out["taken"] == "P"
    assert out["neighbours"]["worth_building"] is True
    # with no honest pass the neighbours are parked whatever FI does
    frame["P"] = frame["L0"] + 0.02
    out = fr.apply_rule(frame, frame[frame["delivery_day"] >= fr.RULE_START])
    assert out["taken"] is None and out["neighbours"]["worth_building"] is False and out["neighbours"]["conditions"]["gain_at_least_0.25_below_F"]
    assert out["next"].startswith("null")
