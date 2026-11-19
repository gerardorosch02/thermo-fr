"""W's two changes on their own: the refit grid and the recency weights. No real price is read here."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import spread_w as sw


def test_refits_fall_every_seventh_day_from_the_anchor_and_cover_the_window():
    days = sw.refit_days("2026-10-11", "2027-01-18")
    assert ((days - pd.Timestamp(sw.GRID_ANCHOR)).days % 7 == 0).all()
    assert days[0] <= pd.Timestamp("2026-10-11") < days[0] + pd.Timedelta(days=7)
    assert days[-1] <= pd.Timestamp("2027-01-18") < days[-1] + pd.Timedelta(days=7)
    assert (np.diff(days.values) == np.timedelta64(7, "D")).all()


def test_a_single_day_gets_exactly_one_refit_and_it_is_not_after_the_day():
    for day in ("2025-10-26", "2026-07-01", "2026-07-07", "2026-07-08"):
        days = sw.refit_days(day, day)
        assert len(days) == 1 and days[0] <= pd.Timestamp(day)


def test_a_day_half_a_year_old_counts_half_and_yesterday_counts_almost_fully():
    refit = pd.Timestamp("2026-10-14")
    training = pd.Series(pd.to_datetime(["2026-10-13", "2026-04-17", "2025-10-19", "2021-01-01"]))
    w = sw.weights(training, refit)
    assert w[0] == pytest.approx(0.5 ** (1 / 180)) and w[1] == pytest.approx(0.5) and w[2] == pytest.approx(0.25)
    assert 0 < w[3] < 0.001 and (np.diff(w) < 0).all()


def test_a_training_day_on_or_after_the_refit_day_is_refused():
    with pytest.raises(ValueError):
        sw.weights(pd.Series(pd.to_datetime(["2026-10-14"])), pd.Timestamp("2026-10-14"))


def test_the_model_of_a_block_never_trains_on_the_block(monkeypatch):
    days = pd.date_range("2024-05-01", "2024-07-10", freq="D")
    index = pd.date_range("2024-05-01", periods=len(days) * 24, freq="h", tz="UTC")
    frame = pd.DataFrame({"delivery_day": np.repeat(days, 24)}, index=index)
    y = pd.Series(np.arange(len(index), dtype=float), index=index)
    X = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    seen = []

    class Recorder:
        def __init__(self, **_):
            pass

        def fit(self, X_train, y_train, sample_weight=None):
            seen.append((X_train.index.max(), len(X_train), sample_weight))
            self.level = float(y_train.iloc[-1])
            return self

        def predict(self, X_block):
            return np.full(len(X_block), self.level)

    monkeypatch.setattr(sw.lgb, "LGBMRegressor", Recorder)
    feat = {"frame": frame, "y": y, "sets": {"G": X}, "first_month": pd.Timestamp("2024-05-01")}
    pred = sw.fit_w(feat, "2024-07-01", "2024-07-10")
    scored = pred.dropna()
    assert set(frame.loc[scored.index, "delivery_day"]) == set(pd.date_range("2024-07-01", "2024-07-10"))
    starts = sw.refit_days("2024-07-01", "2024-07-10")
    assert len(seen) == len(starts)
    for (last_train, rows, w), start in zip(seen, starts):
        assert last_train < pd.Timestamp(start, tz="UTC") and len(w) == rows and w.max() < 1
    # each forecast equals the last target before its own block: nothing from inside the block leaked in
    day_of = frame.loc[scored.index, "delivery_day"]
    for start in starts:
        block = scored[(day_of >= start) & (day_of < start + pd.Timedelta(days=7))]
        assert len(block) and (block == y[frame["delivery_day"] < start].iloc[-1]).all()
