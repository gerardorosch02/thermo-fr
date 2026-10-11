"""market.json is written where the traded prices are and read everywhere else, and carries aggregates only."""

import json
import re

import pandas as pd

from test_store import curve_for
from thermo_fr.forecast.market import MIN_PUBLIC_DAYS, MarketRow, add_row, evaluate, load_market, trade_date_for
from thermo_fr.forecast.publish import MARKET_FILE, export_published, write_market_file
from thermo_fr.forecast.store import Store

# what the file may carry: counts, definitions, stamps, and (from MIN_PUBLIC_DAYS days) five aggregates; never a row or a price
ALWAYS = {"source", "scored_days", "min_days_to_show", "counts", "computed_at_utc", "computed_on", "carried_through"}
BELOW_MIN = ALWAYS | {"note"}
AGGREGATES = {"hit_rate", "mean_pnl_per_mwh", "model_mae", "market_mae", "share_model_beats_market"}
PER_DAY_COLUMNS = ("forecast", "entry_vwap", "entry_close", "auction", "pnl_per_mwh", "pnl_per_mwh_close", "cumulative_pnl_per_mwh",
                   "model_error", "market_error", "signal")


def seeded_store_and_market(tmp_path, days: int):
    """`days` fictional scored delivery days: a forecast issued the morning of the trade date, a flat auction result, one traded base row."""
    store = Store(tmp_path / "db.sqlite")
    market = tmp_path / "eex.csv"
    for i, day in enumerate(pd.date_range("2026-09-01", periods=days, freq="D").strftime("%Y-%m-%d")):
        curve = curve_for(day)
        curve["forecast"] = 55.0 + 0.11 * i
        issued = pd.Timestamp(trade_date_for(day)) + pd.Timedelta(hours=7)  # 09:00 Paris on the trade date, before the 11:15 window
        store.save_forecast(i + 1, day, "honest_v2", issued.tz_localize("UTC"), "gbm:honest_v2.txt", "scheduled", 0, True, curve)
        store.save_actuals(pd.Series(48.0 + 0.53 * i, index=curve.index), pd.Series([day] * len(curve)))
        vwap = round(50.0 + 0.37 * i, 2)
        add_row(MarketRow(day, "base", "11:15", "12:00", vwap - 0.5, vwap + 0.7, vwap - 0.9, vwap + 0.3, vwap, "test desk"), market)
    return store, market


def per_day_values(store, market) -> set[str]:
    table = evaluate(store, load_market(market), feature_set="honest_v2")["table"]
    values = set()
    for column in PER_DAY_COLUMNS:
        for v in table[column].dropna():
            values.add(f"{float(v):.2f}")
            values.add(f"{float(v):g}")
    return values


def assert_aggregates_only(text: str, store, market, allowed: set[str]):
    written = json.loads(text)
    assert set(written) <= allowed, set(written) - allowed
    assert all(not isinstance(v, (list, dict)) for v in written.values()), "a list or an object means per-day rows"
    numbers = set(re.findall(r"-?\d+(?:\.\d+)?", json.dumps({k: v for k, v in written.items() if k != "computed_at_utc"})))
    leaked = numbers & per_day_values(store, market)
    assert not leaked, f"per-day market values in market.json: {sorted(leaked)}"
    for word in ("delivery_date", "entry", "vwap", "auction", "table", "rows", "window_paris", "cumulative"):
        assert word not in {k.lower() for k in written}, word
    return written


def test_market_json_below_the_minimum_holds_only_the_count_and_the_definition(tmp_path):
    store, market = seeded_store_and_market(tmp_path, days=3)
    section = write_market_file(store, tmp_path / "published", market)
    written = assert_aggregates_only((tmp_path / "published" / MARKET_FILE).read_text(), store, market, BELOW_MIN)
    assert written == section and written["scored_days"] == 3 and written["min_days_to_show"] == MIN_PUBLIC_DAYS
    assert written["note"] == f"Versus the market: collecting data, 3 of {MIN_PUBLIC_DAYS} days."
    assert not (AGGREGATES & set(written))
    store.close()


def test_market_json_from_the_minimum_holds_five_aggregates_and_still_no_per_day_value(tmp_path):
    store, market = seeded_store_and_market(tmp_path, days=MIN_PUBLIC_DAYS + 1)
    write_market_file(store, tmp_path / "published", market)
    written = assert_aggregates_only((tmp_path / "published" / MARKET_FILE).read_text(), store, market, ALWAYS | AGGREGATES)
    assert written["scored_days"] == MIN_PUBLIC_DAYS + 1 and AGGREGATES <= set(written) and "note" not in written
    assert 0 <= written["hit_rate"] <= 1 and written["model_mae"] > 0 and written["market_mae"] > 0
    store.close()


def test_the_real_published_market_json_if_present_is_aggregates_only():
    """The file committed in this checkout, whatever the laptop last wrote."""
    from pathlib import Path

    path = Path("published") / MARKET_FILE
    if not path.exists():
        return
    written = json.loads(path.read_text())
    assert set(written) <= ALWAYS | AGGREGATES | {"note", "carried_note"}, set(written) - (ALWAYS | AGGREGATES | {"note", "carried_note"})
    assert all(not isinstance(v, (list, dict)) for v in written.values())
    if written["scored_days"] < MIN_PUBLIC_DAYS:
        assert not (AGGREGATES & set(written))


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
