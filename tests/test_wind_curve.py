"""The W wind curve option (wind_prereg.md, step4_spec.md): default unchanged, the curve, the last-hour rule, the timing gate,
the reviewer's walk-forward MAE reproduced, and that LEAR uses neither the wind proxy nor residual_v2."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_fr.data import hour_labels
from thermo_fr.data.wind_points import POINT_COLUMNS
from thermo_fr.forecast import features, timing, wind_proxy
from thermo_fr.forecast.timing import check_point_in_time, issue_times, premarket_issue_for


@pytest.fixture
def settings(monkeypatch):
    def use(curve=None, labels=None):
        if curve is not None:
            monkeypatch.setattr(wind_proxy, "WIND_CURVE", curve)
        if labels is not None:
            monkeypatch.setattr(hour_labels, "HOUR_LABELS", labels)
    return use


def speeds(start="2025-01-01", days=200, seed=0):
    idx = pd.date_range(start, periods=24 * days, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    base = 7 + 3 * np.sin(np.arange(len(idx)) / 37.0)
    return pd.DataFrame({c: np.clip(base + rng.normal(0, 2, len(idx)), 0, None) for c in POINT_COLUMNS}, index=idx)


def test_default_curve_is_the_existing_one():
    assert wind_proxy.WIND_CURVE == wind_proxy.CUBIC
    pts = speeds(days=5)
    np.testing.assert_array_equal(wind_proxy.capacity_factors(pts).to_numpy(), wind_proxy.power_curve(pts.to_numpy()))
    assert wind_proxy.spec_for() is wind_proxy.WIND
    actual = pd.Series(wind_proxy.power_curve(pts.to_numpy()) @ np.full(len(POINT_COLUMNS), 500.0), index=pts.index)
    w = wind_proxy.calibrate(pts, actual)
    pd.testing.assert_series_equal(wind_proxy.apply_weights(pts, w), wind_proxy.gen_proxy.apply_weights(pts, w, wind_proxy.WIND))


def test_smoothed_curve_and_the_hour_mean():
    assert wind_proxy.W_WEIGHTS.sum() == pytest.approx(1.0) and len(wind_proxy.W_NODES) == 31
    v = np.array([0.0, 3.0, 8.0, 12.0, 20.0, 30.0, np.nan])
    c = wind_proxy.smoothed_curve(v)
    assert np.isnan(c[-1]) and c[0] == 0.0 and 0 < c[1] < 0.01 and c[1] < c[2] < c[3] < 1.0
    assert 0.9 < c[4] < 1.0 and 0 < c[5] < 0.5  # the spread reaches past the cut-out speed at 20 and 30 m/s
    pts = speeds(days=3)
    nxt = wind_proxy.capacity_factors(pts, "W", "next")
    own = wind_proxy.capacity_factors(pts, "W", "own")
    sm = pd.DataFrame(wind_proxy.smoothed_curve(pts.to_numpy()), index=pts.index, columns=pts.columns)
    T = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    np.testing.assert_allclose(nxt.loc[T], (sm.loc[T] + sm.loc[T + pd.Timedelta(hours=1)]) / 2)
    last = pd.Timestamp("2025-01-01 22:00", tz="UTC")  # 23:00 Paris in winter: the last hour of the Paris day
    assert wind_proxy.last_paris_hour(pd.DatetimeIndex([last]))[0]
    np.testing.assert_allclose(own.loc[last], sm.loc[last])
    np.testing.assert_allclose(nxt.loc[last], (sm.loc[last] + sm.loc[last + pd.Timedelta(hours=1)]) / 2)
    mask = ~wind_proxy.last_paris_hour(pts.index)
    pd.testing.assert_frame_equal(own[mask], nxt[mask])


@pytest.mark.parametrize("day", ["2025-01-15", "2025-03-30", "2025-06-21", "2025-10-26"])
def test_fallback_uses_no_speed_valid_after_the_delivery_day(day):
    """With W_LAST_HOUR = "own", changing every speed valid at or after the end of D leaves D's factors unchanged;
    with "next", the speed one hour after D's last hour enters (the case the saved windows must cover)."""
    pts = speeds(start="2025-01-01", days=320)
    start = pd.Timestamp(day).tz_localize("Europe/Paris").tz_convert("UTC")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
    hours = (pts.index >= start) & (pts.index < end)
    later = pts.copy()
    later[pts.index >= end] += 9.0
    own_base, own_moved = wind_proxy.capacity_factors(pts, "W", "own"), wind_proxy.capacity_factors(later, "W", "own")
    pd.testing.assert_frame_equal(own_moved[hours], own_base[hours])
    nxt_base, nxt_moved = wind_proxy.capacity_factors(pts, "W", "next"), wind_proxy.capacity_factors(later, "W", "next")
    assert not np.allclose(nxt_moved[hours].to_numpy(), nxt_base[hours].to_numpy())


def test_w_wind_proxy_passes_the_premarket_gate_with_its_t_plus_1_speed(settings):
    settings(curve=wind_proxy.SMOOTHED, labels=hour_labels.START_OF_HOUR)
    idx = pd.date_range("2024-01-01", "2026-07-01", freq="1h", tz="UTC", inclusive="left")
    honest = {f: features.FEATURE_TIMINGS[f] for f in features.FEATURES["honest_v2"]}
    check_point_in_time(idx, honest, deadline="gate")
    check_point_in_time(idx, honest, deadline="premarket")
    w_issue = issue_times(idx, "wind_proxy")
    settings(curve=wind_proxy.CUBIC)
    cubic_issue = issue_times(idx, "wind_proxy")
    assert (w_issue >= cubic_issue).all()  # the T+1 speed is never earlier than the label's own
    days = timing.delivery_days(idx)
    assert (pd.Series(w_issue, index=idx).groupby(days).max().to_numpy() <= premarket_issue_for(idx).to_series().groupby(days).max().to_numpy()).all()
    # the last hour's T+1 speed (the end of D) comes from the 18Z run of D-2, like every other hour of D
    end = (days + pd.Timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
    run = (pd.DatetimeIndex(end) - pd.Timedelta(hours=48)).floor("6h")
    assert (run == (days - pd.Timedelta(days=2) + pd.Timedelta(hours=18)).tz_localize("UTC")).all()


def test_weights_fitted_with_w_are_applied_with_w(settings):
    pts = speeds(days=120)
    actual = pd.Series(wind_proxy.capacity_factors(pts, "W", "own").to_numpy() @ np.full(len(POINT_COLUMNS), 400.0), index=pts.index)
    w = wind_proxy.calibrate(pts, actual, curve="W", last_hour="own")
    assert w["curve"] == "W" and w["last_hour"] == "own"
    settings(curve=wind_proxy.CUBIC)
    np.testing.assert_allclose(wind_proxy.apply_weights(pts, w).to_numpy(), actual.to_numpy(), rtol=1e-6, atol=1e-3)


def test_lear_uses_neither_the_wind_proxy_nor_residual_v2():
    from thermo_fr.forecast import lear

    names = set(features.LEAR_BLOCK_TIMINGS) | set(getattr(lear, "BLOCK_TIMINGS", {})) | set(features.LEAR_COLUMNS)
    rules = set(features.LEAR_BLOCK_TIMINGS.values())
    assert not any("wind_proxy" in n or "residual" in n for n in names)
    assert not rules & {"wind_proxy", "residual_v2"}


@pytest.mark.skipif(not Path("data/forecast/inputs_aligned.csv").exists(), reason="needs the corrected inputs table (git-ignored)")
def test_reviewer_walk_forward_mae_is_reproduced():
    from thermo_fr.forecast.inputs import load_inputs

    inp = load_inputs("data/forecast/inputs_aligned.csv")
    actual = wind_proxy.actual_wind(inp)
    w, _ = wind_proxy.rolling_proxy(inp[POINT_COLUMNS], actual, curve="W", last_hour="next")
    rows = (inp.index >= pd.Timestamp("2024-04-30 22:00", tz="UTC")) & (inp.index < pd.Timestamp("2026-07-01", tz="UTC"))
    m = rows & actual.notna() & inp["wind_proxy_mw"].notna() & w.notna()
    assert int(m.sum()) == 18957
    assert round(float((actual[m] - w[m]).abs().mean()), 1) == 1258.2
    assert round(float((actual[m] - inp["wind_proxy_mw"][m]).abs().mean()), 1) == 1400.0
