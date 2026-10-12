"""The public market record scores the published pre-market version, never a version only the laptop holds."""

import pandas as pd

from test_market_file_carry import published_csv
from test_store import curve_for
from thermo_fr.forecast.market import LOCAL_LABEL, MarketRow, PublishedVersions, add_row, evaluate, load_market, sync_published_versions
from thermo_fr.forecast.publish import MARKET_FILE, write_market_file
from thermo_fr.forecast.store import Store


def test_only_days_with_a_published_premarket_version_are_counted(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    market = tmp_path / "eex.csv"
    a, b = "2026-10-13", "2026-10-14"  # Tuesday and Wednesday; windows open 11:15 Paris (09:15Z) the day before
    for day, level in ((a, 60.0), (b, 65.0)):
        curve = curve_for(day)
        store.save_actuals(pd.Series(52.0, index=curve.index), pd.Series([day] * len(curve)))
        add_row(MarketRow(day, "base", "11:15", "12:00", 55.5, 56.0, 55.0, 55.8, 55.6, "test desk"), market)
        # the laptop holds a pre-market version of both days
        store.save_forecast(1, day, "honest_v2", pd.Timestamp(f"{pd.Timestamp(day) - pd.Timedelta(days=1):%Y-%m-%d}T08:20Z"),
                            "gbm:honest_v2.txt", "scheduled", 0, True, curve_for(day, level))
    # the public site: day A has a pre-market version with a different forecast; day B only a version issued after the window
    public_a = curve_for(a, 80.0)
    late_b = curve_for(b, 30.0)
    archived = sync_published_versions(store, published_csv([(a, pd.Timestamp("2026-10-12T08:30Z"), public_a),
                                                              (b, pd.Timestamp("2026-10-13T10:23Z"), late_b)]))
    assert archived == 2 and sync_published_versions(store, published_csv([(a, pd.Timestamp("2026-10-12T08:30Z"), public_a)])) == 0
    public = evaluate(PublishedVersions(store), load_market(market), feature_set="honest_v2")
    assert public["summary"]["scored_days"] == 1 and public["summary"]["versions"] == PublishedVersions.LABEL
    row = public["table"].iloc[0]
    assert row["delivery_date"] == a and row["forecast"] == round(public_a["forecast"].mean(), 2)  # the published figure, not the local 60
    assert [s["delivery_date"] for s in public["summary"]["skipped"]] == [b] and "issued before the window" in public["summary"]["skipped"][0]["reason"]
    local = evaluate(store, load_market(market), feature_set="honest_v2")
    assert local["summary"]["scored_days"] == 2 and local["summary"]["versions"] == LOCAL_LABEL
    section = write_market_file(store, tmp_path / "published", market)
    assert section["scored_days"] == 1 and "published pre-market version" in section["counts"] and section["versions"] == PublishedVersions.LABEL
    # the archive outlives the public file: a forecasts.csv that no longer carries day A leaves A scorable
    sync_published_versions(store, published_csv([(b, pd.Timestamp("2026-10-13T10:23Z"), late_b)]))
    assert write_market_file(store, tmp_path / "published", market)["scored_days"] == 1
    assert (tmp_path / "published" / MARKET_FILE).exists()
    store.close()


def test_market_publish_reads_forecasts_csv_from_the_remote_branch_not_the_working_tree(tmp_path):
    from test_gitsync import git, make_repos
    from thermo_fr.cli import market_publish

    remote, laptop, actions = make_repos(tmp_path)
    store = Store(tmp_path / "db.sqlite")
    market = tmp_path / "eex.csv"
    day = "2026-10-13"
    curve = curve_for(day)
    store.save_actuals(pd.Series(52.0, index=curve.index), pd.Series([day] * len(curve)))
    add_row(MarketRow(day, "base", "11:15", "12:00", 55.5, 56.0, 55.0, 55.8, 55.6, "test desk"), market)
    # the workflow committed a pre-market version to main; the laptop's working tree holds a regenerated file with a late version only
    (actions / "published" / "forecasts.csv").write_text(published_csv([(day, pd.Timestamp("2026-10-12T08:30Z"), curve_for(day, 80.0))]))
    git(actions, "add", "published")
    git(actions, "commit", "-q", "-m", "Publish morning-run")
    git(actions, "push", "-q", "origin", "HEAD:main")
    git(laptop, "pull", "-q", "origin", "main")  # the laptop has the committed file, then regenerates it locally (a manual thermo-fr publish)
    (laptop / "published" / "forecasts.csv").write_text(published_csv([(day, pd.Timestamp("2026-10-12T10:00Z"), curve_for(day, 30.0))]))
    result = market_publish(store, laptop / "published", market, laptop, kind="test", log=_Quiet())
    assert result["status"] == "ok" and result["section"]["scored_days"] == 1
    assert store.published_versions(day, "honest_v2")["issued_at_utc"].tolist() == ["2026-10-12T08:30:00Z"]
    assert "published versions archived" not in result["message"]  # the message states the outcome; the archive count is in the log
    git(actions, "pull", "-q", "origin", "main")
    assert (actions / "published" / "market.json").exists() and (actions / "published" / "forecasts.csv").read_text().count("08:30") == 24
    store.close()


class _Quiet:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
