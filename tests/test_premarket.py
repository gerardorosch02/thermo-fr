"""The pre-market headline rule in the store, the published dataset and the public app."""

import pandas as pd
import pytest

from test_store import curve_for
from thermo_fr.forecast.market import headline_version, premarket_cutoff, premarket_flag, public_summary
from thermo_fr.forecast.publish import export_published
from thermo_fr.forecast.store import Store, score_curve

DAY = "2026-10-10"  # the window opens 11:15 Paris on 10-09 = 09:15Z (CEST)


def flat(day, level):
    c = curve_for(day)
    c["forecast"] = level
    return c


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.save_forecast(1, DAY, "honest", pd.Timestamp("2026-10-09T08:00Z"), "gbm:honest.txt", "scheduled", 0, True, flat(DAY, 70.0))  # 10:00 Paris
    s.save_forecast(2, DAY, "honest", pd.Timestamp("2026-10-09T08:40Z"), "gbm:honest.txt", "scheduled", 0, True, flat(DAY, 74.0))  # 10:40 Paris
    s.save_forecast(3, DAY, "honest", pd.Timestamp("2026-10-09T10:30Z"), "gbm:honest.txt", "scheduled", 0, True, flat(DAY, 82.0))  # 12:30 Paris
    s.save_forecast(4, "2026-10-11", "honest", pd.Timestamp("2026-10-10T10:30Z"), "gbm:honest.txt", "manual", 0, True, flat("2026-10-11", 60.0))
    for day, level in ((DAY, 51.26), ("2026-10-11", 45.0)):
        c = curve_for(day)
        s.save_actuals(pd.Series(level, index=c.index), pd.Series([day] * len(c)))
    for fid in (1, 2, 3, 4):
        v = s.forecast_versions(DAY if fid < 4 else "2026-10-11", "honest")
        row = v[v["forecast_id"] == fid].iloc[0]
        s.save_score(fid, row["delivery_day"], "honest", row["issued_at_utc"], score_curve(s.forecast_curve(fid), s.actuals_for(row["delivery_day"])))
    yield s
    s.close()


def test_cutoff_and_flag():
    assert premarket_cutoff(DAY) == pd.Timestamp("2026-10-09T09:15Z")
    assert premarket_cutoff("2026-12-10") == pd.Timestamp("2026-12-09T10:15Z")
    flags = premarket_flag([DAY, DAY, DAY], ["2026-10-09T08:00Z", "2026-10-09T09:15Z", "2026-10-09T10:30Z"])
    assert flags.tolist() == [True, False, False]  # exactly at the opening is not before it


def test_headline_is_the_last_premarket_version_else_the_latest(store):
    meta, curve, later = store.headline_forecast(DAY, "honest")
    assert meta["forecast_id"] == 2 and meta["premarket"] and curve["forecast"].iloc[0] == 74.0
    assert later["forecast_id"].tolist() == [3]
    meta, curve, later = store.headline_forecast("2026-10-11", "honest")
    assert meta["forecast_id"] == 4 and not meta["premarket"] and later["forecast_id"].tolist() == [4]
    assert store.headline_forecast("2026-10-12", "honest")[0] is None
    scores = store.headline_scores("honest")
    assert scores.set_index("delivery_day")["forecast_id"].to_dict() == {DAY: 2, "2026-10-11": 4}
    assert scores.set_index("delivery_day")["premarket"].to_dict() == {DAY: True, "2026-10-11": False}
    table = headline_version(pd.DataFrame({"delivery_day": [DAY, DAY], "issued_at_utc": ["2026-10-09T10:30Z", "2026-10-09T08:00Z"]}))
    assert table["issued_at_utc"].tolist() == ["2026-10-09T08:00Z"] and headline_version(pd.DataFrame(columns=["delivery_day", "issued_at_utc"])).empty


def test_published_tomorrow_is_the_premarket_version_and_files_carry_the_flag(store, tmp_path):
    out = tmp_path / "published"
    status = export_published(store, out, days=90, now=pd.Timestamp("2026-10-09T12:00Z"), market_path=tmp_path / "no_market.csv")
    tomorrow = pd.read_csv(out / "tomorrow.csv")  # the latest delivery day in the store is 10-11, whose only version came after the window
    assert tomorrow["delivery_day"].unique().tolist() == ["2026-10-11"] and tomorrow["forecast"].iloc[0] == 60.0
    only = Store(tmp_path / "only.sqlite")
    for fid, issued, level in ((1, "2026-10-09T08:00Z", 70.0), (2, "2026-10-09T08:40Z", 74.0), (3, "2026-10-09T10:30Z", 82.0)):
        only.save_forecast(fid, DAY, "honest", pd.Timestamp(issued), "gbm:honest.txt", "scheduled", 0, True, flat(DAY, level))
    export_published(only, tmp_path / "only_published", days=90, now=pd.Timestamp("2026-10-09T12:00Z"), market_path=tmp_path / "no_market.csv")
    only.close()
    tomorrow = pd.read_csv(tmp_path / "only_published" / "tomorrow.csv")
    assert tomorrow["issued_at_utc"].unique().tolist() == ["2026-10-09T08:40:00Z"] and tomorrow["forecast"].iloc[0] == 74.0
    forecasts = pd.read_csv(out / "forecasts.csv")
    assert forecasts.groupby("issued_at_utc")["premarket"].first().to_dict() == {
        "2026-10-09T08:00:00Z": 1, "2026-10-09T08:40:00Z": 1, "2026-10-09T10:30:00Z": 0, "2026-10-10T10:30:00Z": 0}
    scores = pd.read_csv(out / "scores.csv")
    assert set(scores.columns) >= {"premarket", "mae"} and scores["premarket"].tolist() == [1, 1, 0, 0]
    assert status["market"]["scored_days"] == 0


def test_public_summary_collects_data_until_twenty_days():
    result = {"summary": {"scored_days": 3, "hit_rate": 0.67, "mean_pnl_per_mwh": 1.0, "model_mae": 10.0, "market_mae": 2.0,
                          "share_model_beats_market": 0.0}}
    public = public_summary(result)
    assert public["min_days_to_show"] == 20 and public["note"] == "Versus the market: collecting data, 3 of 20 days."
    assert "mean_pnl_per_mwh" not in public and "hit_rate" not in public
    assert public_summary(result, min_days=3)["mean_pnl_per_mwh"] == 1.0


def test_public_app_helpers_pick_the_premarket_version():
    import streamlit_app as app

    rows = []
    for issued, level in (("2026-10-09T08:00:00Z", 70.0), ("2026-10-09T08:40:00Z", 74.0), ("2026-10-09T10:30:00Z", 82.0)):
        for hour in range(24):
            rows.append({"delivery_day": DAY, "issued_at_utc": issued, "kind": "scheduled", "train_hours": 0,
                         "timestamp_utc": f"2026-10-09T{22 + hour:02d}:00:00Z" if hour < 2 else f"2026-10-10T{hour - 2:02d}:00:00Z",
                         "hour": hour, "forecast": level, "naive_day": 60.0})
    forecasts = pd.DataFrame(rows)
    heads = app.headline_versions(forecasts)
    assert len(heads) == 1 and heads.loc[0, "issued_at_utc"] == "2026-10-09T08:40:00Z" and bool(heads.loc[0, "premarket"])
    data = {"forecasts": forecasts, "tomorrow": pd.DataFrame(), "actuals": pd.DataFrame(), "scores": pd.DataFrame(), "band": pd.DataFrame()}
    versions = app.tomorrow_versions(data)
    assert versions["day"] == DAY and versions["premarket"] and versions["headline"]["forecast"].iloc[0] == 74.0
    assert versions["later"]["issued_paris"].tolist() == ["2026-10-09 12:30"] and versions["later"]["daily_mean"].iloc[0] == 82.0
    by_hour = app.tomorrow_table(data)
    assert len(by_hour) == 24 and by_hour["forecast"].eq(74.0).all()
    only_late = data | {"forecasts": forecasts[forecasts["issued_at_utc"] == "2026-10-09T10:30:00Z"]}
    late = app.tomorrow_versions(only_late)
    assert not late["premarket"] and late["headline"]["forecast"].iloc[0] == 82.0 and late["later"].empty is False
    scores = pd.DataFrame({"delivery_day": [DAY] * 3, "issued_at_utc": ["2026-10-09T08:00:00Z", "2026-10-09T08:40:00Z", "2026-10-09T10:30:00Z"],
                           "mae": [10.0, 12.0, 30.0], "naive_mae": [20.0] * 3, "mae_below_baseline": [1, 1, 0]})
    assert app.headline_rows(scores)["mae"].tolist() == [12.0]
