"""The market section of status.json is computed where the traded prices are and carried through elsewhere."""

import json

import pandas as pd

from test_store import curve_for
from thermo_fr.forecast.publish import export_published, market_status
from thermo_fr.forecast.store import Store


def test_actions_without_market_data_keeps_the_laptop_aggregates(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    day = "2026-10-13"
    store.save_forecast(1, day, "honest_v2", pd.Timestamp("2026-10-12T08:00Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve_for(day))
    out = tmp_path / "published"
    out.mkdir()
    # the laptop wrote these aggregates earlier (no traded prices here either, so they are stand-ins with the marker fields)
    laptop = {"scored_days": 21, "min_days_to_show": 20, "hit_rate": 0.43, "mean_pnl_per_mwh": -1.2, "computed_at_utc": "2026-11-20T11:00:00Z",
              "computed_on": "laptop", "counts": "settled delivery days ..."}
    (out / "status.json").write_text(json.dumps({"market": laptop}))
    status = export_published(store, out, days=90, now=pd.Timestamp("2026-11-21T12:00Z"), market_path=tmp_path / "no_market.csv")
    assert status["market"]["scored_days"] == 21 and status["market"]["hit_rate"] == 0.43 and status["market"]["carried_through"] is True
    assert "laptop" in status["market"]["carried_note"]
    # with no earlier aggregates at all the section says so, with the count definition
    fresh = market_status(store, tmp_path / "no_market.csv", previous=None)
    assert fresh["scored_days"] == 0 and "settled delivery days" in fresh["counts"] and fresh["min_days_to_show"] == 20
    # an older status.json without the marker fields is not carried through (it could be the zeros a previous Actions run wrote)
    old = market_status(store, tmp_path / "no_market.csv", previous={"scored_days": 0, "note": "No traded prices recorded yet."})
    assert old["scored_days"] == 0 and "carried_through" not in old
    store.close()
