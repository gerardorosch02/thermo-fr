"""The public live record: pre-market days of the default set only, late-only days excluded and listed, previous model split off."""

import pandas as pd

from thermo_fr.forecast import publish


def scores_fixture() -> pd.DataFrame:
    """Scores of several versions: the previous set on 8 to 10 Oct (8 Oct late only), the default set from 13 Oct (13 Oct late only)."""
    rows = []

    def add(day, issued, feature_set, mae, naive, model):
        rows.append({"delivery_day": day, "issued_at_utc": issued, "feature_set": feature_set, "hours": 24, "mae": mae, "rmse": mae + 5,
                     "naive_mae": naive, "naive_rmse": naive + 5, "mae_below_baseline": int(mae < naive), "settled_at_utc": "x", "model": model})
    add("2026-10-08", "2026-10-07T10:05:33Z", "honest", 44.1, 81.2, "gbm")   # 12:05 Paris: late only
    add("2026-10-08", "2026-10-07T10:30:20Z", "honest", 44.1, 81.2, "gbm")
    add("2026-10-09", "2026-10-08T06:01:04Z", "honest", 30.0, 40.0, "gbm")   # 08:01 Paris: pre-market
    add("2026-10-09", "2026-10-08T07:00:46Z", "honest", 28.0, 40.0, "gbm")   # the last pre-market version is the one scored
    add("2026-10-09", "2026-10-08T12:33:49Z", "honest", 10.0, 40.0, "gbm")   # late: must not be used
    add("2026-10-10", "2026-10-09T08:00:52Z", "honest", 50.0, 45.0, "gbm")
    add("2026-10-10", "2026-10-09T17:55:04Z", "honest", 12.0, 45.0, "gbm:honest.txt")
    add("2026-10-13", "2026-10-12T10:23:39Z", "honest_v2", 20.0, 30.0, "gbm:honest_v2.txt")  # 12:23 Paris: late only
    add("2026-10-14", "2026-10-13T08:06:00Z", "honest_v2", 18.0, 30.0, "gbm:honest_v2.txt")  # 10:06 Paris: pre-market
    add("2026-10-14", "2026-10-13T08:31:00Z", "honest_v2", 16.0, 30.0, "gbm:honest_v2.txt")
    add("2026-10-15", "2026-10-14T08:06:00Z", "honest_v2", 34.0, 30.0, "gbm:honest_v2.txt")
    add("2026-10-15", "2026-10-14T10:30:00Z", "honest_v2", 5.0, 30.0, "gbm:honest_v2.txt")   # late: must not be used
    return pd.DataFrame(rows)


def test_headline_counts_pre_market_days_of_the_default_set_only_and_lists_late_only_days():
    record, tables = publish.live_record(scores_fixture(), "honest_v2", "honest")
    head = record["default"]
    assert head["status"] == "ok" and head["feature_set"] == "honest_v2"
    assert head["first_premarket_day"] == "2026-10-14" and head["days"] == 2 and head["days_in_window"] == 2
    assert head["model_mae"] == 25.0 and head["baseline_mae"] == 30.0 and head["share_below_baseline"] == 0.5  # (16 + 34) / 2: the last pre-market versions
    assert [d["delivery_day"] for d in head["late_only_days"]] == ["2026-10-13"] and head["late_only_days"][0]["versions"] == 1
    assert tables["default"]["issued_at_utc"].tolist() == ["2026-10-13T08:31:00Z", "2026-10-14T08:06:00Z"]
    previous = record["previous"]
    assert previous["feature_set"] == "honest" and previous["days"] == 2 and previous["first_premarket_day"] == "2026-10-09"
    assert previous["model_mae"] == 39.0 and previous["baseline_mae"] == 42.5  # (28 + 50) / 2 and (40 + 45) / 2
    assert [d["delivery_day"] for d in previous["late_only_days"]] == ["2026-10-08"]
    assert "late only" in record["rule"] and record["min_days_to_show"] == 1


def test_collecting_data_while_the_default_set_has_no_settled_pre_market_day():
    scores = scores_fixture()
    only_late = scores[~((scores["feature_set"] == "honest_v2") & scores["delivery_day"].isin(["2026-10-14", "2026-10-15"]))]
    record, tables = publish.live_record(only_late, "honest_v2", "honest")
    head = record["default"]
    assert head["status"] == "collecting data" and head["days"] == 0 and "model_mae" not in head
    assert [d["delivery_day"] for d in head["late_only_days"]] == ["2026-10-13"]
    assert tables["default"].empty
    # without a default-set first day, every previous-set day belongs to the previous row
    assert record["previous"]["days"] == 2 and [d["delivery_day"] for d in record["previous"]["late_only_days"]] == ["2026-10-08"]
    empty, _ = publish.live_record(pd.DataFrame(), "honest_v2", "honest")
    assert empty["default"]["status"] == "collecting data" and empty["previous"] is None


def test_previous_set_days_after_the_default_started_are_not_double_counted():
    scores = scores_fixture()
    extra = scores[scores["delivery_day"] == "2026-10-14"].assign(feature_set="honest")  # the fallback set also scored on a default day
    record, _ = publish.live_record(pd.concat([scores, extra]), "honest_v2", "honest")
    assert record["previous"]["days"] == 2 and record["previous"]["last_premarket_day"] == "2026-10-10"
    assert record["default"]["days"] == 2


def test_the_public_app_computes_the_same_record():
    import streamlit_app as app

    scores = scores_fixture()
    ours, our_tables = publish.live_record(scores, "honest_v2", "honest")
    theirs, their_tables = app.live_record(scores, "honest_v2", "honest")
    assert ours == theirs
    assert our_tables["default"]["issued_at_utc"].tolist() == their_tables["default"]["issued_at_utc"].tolist()
    assert app.late_only_text(theirs["default"]) == "Late only, not scored: 2026-10-13 (1 version(s), earliest issued 2026-10-12 12:23 Paris)."
    assert app.late_only_text(theirs["previous"]).startswith("Late only, not scored: 2026-10-08 (2 version(s)")
