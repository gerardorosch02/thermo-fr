"""The probabilistic rows travel through the public dataset: export, tomorrow's file, status aggregates, and the import round trip."""

import json

import pandas as pd

from test_store import curve_for
from thermo_fr.forecast.publish import export_published, import_published
from thermo_fr.forecast.store import Store, score_curve


def test_export_and_import_of_the_probabilistic_rows(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    day = "2026-10-13"
    curve = curve_for(day)
    fid = store.save_forecast(1, day, "honest_v2", pd.Timestamp("2026-10-12T08:00Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve)
    prob = pd.DataFrame(index=curve.index)
    prob["q10"], prob["q50"], prob["q90"] = curve["forecast"] - 6, curve["forecast"], curve["forecast"] + 6
    prob["lo"], prob["hi"] = prob["q10"] - 1.5, prob["q90"] + 1.5
    prob["p_negative"], prob["p_spike"], prob["spike_threshold"], prob["conformal_margin"] = 0.02, 0.3, 140.0, 1.5
    store.save_probabilistic(fid, prob)
    actual = pd.Series(curve["forecast"].to_numpy() + 2.0, index=curve.index)
    store.save_actuals(actual, pd.Series([day] * len(curve)))
    store.save_score(fid, day, "honest_v2", "2026-10-12T08:00:00Z", score_curve(curve, actual))
    out = tmp_path / "published"
    status = export_published(store, out, days=90, now=pd.Timestamp("2026-10-12T12:00Z"), market_path=tmp_path / "none.csv")
    exported = pd.read_csv(out / "probabilistic.csv")
    assert len(exported) == 24 and {"p_negative", "feature_set", "hour"} <= set(exported.columns)
    assert not {"q10", "lo", "hi", "p_spike", "spike_threshold"} & set(exported.columns)  # the band and the spike stay local, under evaluation
    tomorrow = pd.read_csv(out / "tomorrow_probabilistic.csv")
    assert len(tomorrow) == 24 and tomorrow["delivery_day"].unique().tolist() == [day]
    live = status["probabilistic"]["live"]
    assert live["days"] == 1 and live["negative"]["events"] == 0 and "coverage_10_90" not in live and "spike" not in live
    assert status["probabilistic"]["public_components"] == ["negative"] and "negative" in status["probabilistic"]["definitions"]
    fresh = Store(tmp_path / "fresh.sqlite")
    summary = import_published(fresh, out)
    assert summary["probabilistic_versions"] == 1
    back = fresh.probabilistic_curve(int(fresh.forecast_versions(day, "honest_v2").iloc[0]["forecast_id"]))
    assert len(back) == 24 and back["p_negative"].iloc[0] == 0.02 and back["q10"].isna().all() and back["p_spike"].isna().all()
    assert import_published(fresh, out)["probabilistic_versions"] == 0  # idempotent
    fresh.close()
    store.close()
