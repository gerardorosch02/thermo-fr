"""market.json is written where the traded prices are and read everywhere else."""

import json

import pandas as pd

from test_store import curve_for
from thermo_fr.forecast.market import MarketRow, add_row
from thermo_fr.forecast.publish import MARKET_FILE, export_published, write_market_file
from thermo_fr.forecast.store import Store


def test_laptop_writes_market_json_and_a_runner_carries_it_into_status(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    day = "2026-10-13"
    store.save_forecast(1, day, "honest_v2", pd.Timestamp("2026-10-12T08:00Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve_for(day))
    market = tmp_path / "eex.csv"
    add_row(MarketRow(day, "base", "11:15", "12:00", 60.0, 62.0, 59.0, 61.0, 60.5, "test desk"), market)
    out = tmp_path / "published"
    status = export_published(store, out, days=90, now=pd.Timestamp("2026-10-12T12:00Z"), market_path=market)
    written = json.loads((out / MARKET_FILE).read_text())
    assert written["computed_on"] == "laptop" and written["carried_through"] is False and status["market"] == written
    # the workflow's runner: same published folder, no market file; it must not rewrite market.json and must carry it into status.json
    before = (out / MARKET_FILE).read_text()
    runner_status = export_published(store, out, days=90, now=pd.Timestamp("2026-10-13T13:00Z"), market_path=tmp_path / "absent.csv")
    assert (out / MARKET_FILE).read_text() == before
    assert runner_status["market"]["carried_through"] is True and runner_status["market"]["computed_at_utc"] == written["computed_at_utc"]
    assert write_market_file(store, out, tmp_path / "absent.csv") is None
    store.close()
