"""The public dataset under published/, and the round trip that lets GitHub Actions keep it current.

GitHub Actions has no persistent disk, so the published folder is both the
output and the state: a workflow run imports it into a fresh SQLite store,
runs morning-run or settle, and exports again. The files are plain CSV so
git stores small deltas when rows are appended.

Files written by `export_published`:

    forecasts.csv      forecast versions of the last `days` days (honest_v2, the default, and honest, the fallback, told apart by
                       the feature_set column; the model column names the file that produced each version): delivery day,
                       issue time, hour, forecast and `premarket` (1 when issued before the 11:15 Paris market window of D-1)
    actuals.csv        actual day-ahead prices (the auction result) for the same window
    scores.csv         daily MAE and RMSE of each version against the auction result, with the same `premarket` flag. Since
                       2026-10-12 no same-hour-previous-day figure is published: the EEX traded price (market.json) is the only
                       benchmark of the level forecast; the store still computes that figure internally for the backtests
    tomorrow.csv       the headline forecast of the default model for the next delivery day, hourly: the last
                       version issued before the market window, else the latest
    error_band.csv     backtest error percentiles by hour (the dashboard's shaded band until the quantile band is made the default)
    market.json        the market aggregates, written only on the laptop (where the traded prices are) by `thermo-fr market publish`, which
                       commits and pushes that one file; the workflow never writes it and copies it into status.json unchanged
    drivers.csv        per version of the published sets and model feature: the feature's group (brief.py), the daily mean of the
                       feature and of its LightGBM contribution, plus the expected value row; the morning brief's "what changed"
                       and "why the model forecast moved" come from it. Public inputs only, nothing from EEX
    probabilistic.csv  per version and hour of the published sets: the probability of a negative price, calibrated (p_negative) with the
                       raw classifier output alongside (p_negative_raw) and a flag (calibrated); the quantile band and the spike probability
                       are computed and stored locally every day but not published, under evaluation);
                       tomorrow_probabilistic.csv the headline version's rows
    model/honest_v2_calibration.json, model/honest_v2_calibration_pairs.csv  the isotonic calibration of the negative-price probability
                       (breakpoints, fit date, pairs) and the out-of-sample pairs it is fitted on, refitted monthly (calibration.py)
    probabilistic_backtest.json  aggregates of the negative-price backtest on the selection window and the holdout, written by
                       `thermo-fr prob-backtest`
    model/honest_v2.txt   the default LightGBM model, refitted monthly by the workflow (see refit.py)
    model/honest_v2.json  its training period, fit date, features, a recent check (the last 30 days, not excluded from the stored
                          fit) and the frozen-holdout result copied from docs/experiments.md
    model/honest.txt      the fallback model (used when a v2 input is missing), refitted with it; honest.json its metadata
    status.json        when the dataset was written, the last run, the attributions and, under
                       "market", aggregates of the forecast against EEX traded prices (market.py):
                       days scored, hit rate, mean P&L per MWh, model and market error. Never a
                       traded price, and nothing below MIN_PUBLIC_DAYS scored days.

Only the honest_v2 and honest feature sets are published, and at most `days` days of actual
prices. The inputs history, the data status, the timing log and the traded
prices stay local.
"""

import json
from pathlib import Path

import pandas as pd

from .market import DEFAULT_PATH as MARKET_PATH
from .market import MIN_PUBLIC_DAYS
from .brief import group_of
from .calibration import load_calibration, public_calibration
from .probabilistic import live_calibration
from .shape import live_shape_record
from .market import evaluate as evaluate_market
from .market import PublishedVersions, headline_version, load_market, premarket_flag, public_summary
from .features import refuse_research
from .store import Store, iso, utc_now

PUBLIC_FEATURE_SET = "honest_v2"  # the default model's set (published/model/honest_v2.txt); honest.txt is the fallback
PUBLIC_FEATURE_SETS = ("honest_v2", "honest")  # both are exported, with a feature_set column; the public app shows the default
FALLBACK_FEATURE_SET = "honest"
DEFAULT_DIR = Path("published")

ATTRIBUTIONS = [
    "Day-ahead prices, load forecast and wind and solar forecasts: ENTSO-E Transparency Platform, "
    "https://transparency.entsoe.eu, reused with the source cited as required by its terms.",
    "French load history used in the thermosensitivity work: RTE eCO2mix via ODRE, "
    "https://opendata.reseaux-energies.fr, Licence Ouverte v2.0 (Etalab).",
    "Weather forecasts: Open-Meteo.com, https://open-meteo.com, CC BY 4.0.",
]


SHAPE_BACKTEST_FILE = "shape_battery_backtest.json"  # written by `thermo-fr shape-backtest`, aggregates only


PROB_BACKTEST_FILE = "probabilistic_backtest.json"  # written by `thermo-fr prob-backtest`, aggregates only


def probabilistic_status(store: Store, out_dir) -> dict:
    """Live calibration of the published set's probabilistic forecasts (aggregates only) plus a pointer to the backtest aggregates."""
    out = Path(out_dir)
    live = live_calibration(store.probabilistic_settled(PUBLIC_FEATURE_SET))
    public_live = {k: live[k] for k in ("days", "hours", "negative", "negative_calibrated_hours") if k in live}  # the band and the spike stay local
    meta_path = out / "model" / f"{PUBLIC_FEATURE_SET}_probabilistic.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {"live": public_live, "backtest_file": PROB_BACKTEST_FILE if (out / PROB_BACKTEST_FILE).exists() else None,
            "calibration": public_calibration(load_calibration(out / "model", PUBLIC_FEATURE_SET)),
            "model": {k: meta.get(k) for k in ("fitted_at_utc", "train_from", "train_to") if k in meta} if meta else {},
            "public_components": ["negative"],
            "definitions": {"negative": "probability that the hour's price is below zero, from a classifier on the same inputs as the point "
                                        "forecast; accepted on the selection window and confirmed on the holdout (docs/experiments.md)"}}


def public_probabilistic_record(record: dict) -> dict:
    """What the public backtest file carries: the negative-price event only (the interval and the spike stay in the local reports)."""
    return {"label": record.get("label"), "first_day": record.get("first_day"), "last_day": record.get("last_day"),
            "events": {"negative": record.get("events", {}).get("negative", {})}, "written_at_utc": record.get("written_at_utc")}


def shape_status(store: Store, out_dir) -> dict:
    """Write shape_battery.json (backtest aggregates plus the live record of the published set) and return the live aggregates for status."""
    out = Path(out_dir)
    backtest_path = out / SHAPE_BACKTEST_FILE
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}
    live = live_shape_record(store, PUBLIC_FEATURE_SET)
    daily = live.pop("daily", None)
    payload = {"backtest": backtest, "live": {**live, "daily": daily.to_dict(orient="records") if daily is not None else []},
               "definitions": {"shape": "each hour's price minus the day's mean (base)",
                               "battery": "1 MW / 2 MWh, 88% round-trip, one cycle a day, a 2-hour charge block before a 2-hour discharge block "
                                          "chosen on the forecast before the gate, settled at the auction result; skipped when the forecast "
                                          "spread does not cover the efficiency loss",
                               "benchmarks": "d1: yesterday's shape; same_type: the shape of the most recent earlier day of the same type"}}
    (out / "shape_battery.json").write_text(json.dumps(payload, indent=2, default=str))
    return {k: v for k, v in live.items() if k in ("days", "shape", "battery")}


MARKET_FILE = "market.json"  # written only where the traded prices are (the laptop); every other publisher reads it and carries it through


def _premarket(frame: pd.DataFrame):
    return premarket_flag(frame["delivery_day"], frame["issued_at_utc"])

LIVE_RECORD_RULE = ("Each delivery day counts once, scored against the auction result on the last version issued before the market window "
                    "opened (11:15 Paris on the day before); a day whose versions were all issued later is listed as late only and not scored. "
                    "The record is the current default model from its first pre-market day; the earlier days of the previous set are a "
                    "separate row. It states the model's own error only: the EEX traded price, in the Versus the market panel, is the one "
                    "benchmark of the level forecast.")


def _record_for(rows: pd.DataFrame, feature_set: str, days: int) -> tuple[dict, pd.DataFrame]:
    """One row per delivery day (its last pre-market version) with the aggregates of the last `days` of them; late-only days listed."""
    picked, late_only = [], []
    for day, group in rows.sort_values(["delivery_day", "issued_at_utc"]).groupby("delivery_day", sort=True):
        pre = group[group["premarket"].astype(bool)]
        if len(pre):
            picked.append(pre.iloc[-1])
        else:
            late_only.append({"delivery_day": day, "versions": int(len(group)), "first_issued_utc": str(group["issued_at_utc"].iloc[0])})
    table = pd.DataFrame(picked).reset_index(drop=True) if picked else rows.iloc[0:0]
    out = {"feature_set": feature_set, "days": int(len(table)), "late_only_days": late_only, "window_days": days}
    if len(table):
        recent = table.tail(days)
        out.update({"first_premarket_day": str(table["delivery_day"].iloc[0]), "last_premarket_day": str(table["delivery_day"].iloc[-1]),
                    "days_in_window": int(len(recent)), "model_mae": round(float(recent["mae"].mean()), 2),
                    "model_rmse": round(float(recent["rmse"].mean()), 2) if "rmse" in recent else None,
                    "scored_against": "the auction result; the EEX traded price is the only benchmark (Versus the market)"})
    return out, table


def live_record(scores: pd.DataFrame, default_set: str = "honest_v2", previous_set: str = "honest", days: int = 30) -> tuple[dict, dict]:
    """The public live record: (record, tables). record["default"] is the headline, record["previous"] the earlier set's own row or None."""
    needed = {"delivery_day", "issued_at_utc", "feature_set", "mae"}
    empty = {"feature_set": default_set, "days": 0, "late_only_days": [], "window_days": days, "status": "collecting data"}
    if scores is None or scores.empty or not needed <= set(scores.columns):
        return {"default": empty, "previous": None, "min_days_to_show": 1, "rule": LIVE_RECORD_RULE}, {"default": pd.DataFrame(), "previous": pd.DataFrame()}
    frame = scores.copy()
    frame["premarket"] = _premarket(frame)
    default, default_table = _record_for(frame[frame["feature_set"] == default_set], default_set, days)
    default["status"] = "ok" if default["days"] >= 1 else "collecting data"
    previous_rows = frame[frame["feature_set"] == previous_set]
    if default.get("first_premarket_day"):
        previous_rows = previous_rows[previous_rows["delivery_day"] < default["first_premarket_day"]]
    previous, previous_table = (_record_for(previous_rows, previous_set, days) if len(previous_rows) else (None, pd.DataFrame()))
    return ({"default": default, "previous": previous, "min_days_to_show": 1, "rule": LIVE_RECORD_RULE},
            {"default": default_table, "previous": previous_table})




def write_market_file(store: Store, out_dir, market_path=MARKET_PATH) -> dict | None:
    """Compute the market aggregates where the traded prices are and write published/market.json; None when there is no market file.

    The aggregates score the published pre-market versions archived in the store (see market.PublishedVersions); `thermo-fr market
    publish` refreshes that archive from forecasts.csv on origin/main before calling this.
    """
    rows = load_market(market_path)
    if rows.empty:
        return None
    section = market_status(store, market_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKET_FILE).write_text(json.dumps(section, indent=2, default=str))
    return section


def read_market_file(out_dir) -> dict | None:
    path = Path(out_dir) / MARKET_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def market_status(store: Store, market_path=MARKET_PATH, previous: dict | None = None) -> dict:
    """The public aggregates of the forecast against traded prices (see market.public_summary).

    The traded prices live only on the laptop. A machine without the market file (GitHub Actions) does not recompute the section: it
    carries the last aggregates the laptop published through unchanged, so the public count never drops to zero for lack of data.
    """
    rows = load_market(market_path)
    if rows.empty:
        if previous and previous.get("computed_at_utc"):
            return {**previous, "carried_through": True,
                    "carried_note": "this machine holds no traded prices; the aggregates are the last ones computed on the laptop"}
        return {"scored_days": 0, "scored_windows": 0, "min_days_to_show": MIN_PUBLIC_DAYS, "note": "No traded prices recorded yet.",
                "benchmark": "the only benchmark of the level forecast: the window VWAP of the traded future, scored at the auction result",
                "counts": "windows of settled delivery days whose published pre-market version was issued before the market window, scored "
                          "against the auction result"}
    result = evaluate_market(PublishedVersions(store), rows, feature_set=PUBLIC_FEATURE_SET)
    return {**public_summary(result), "computed_on": "laptop", "carried_through": False}


INTERNAL_COMPARISON_KEYS = ("naive_mae", "naive_rmse", "naive_day", "mae_below_baseline")  # computed for the backtests, never published


def public_model_meta(meta: dict | None, keys) -> dict:
    """The model metadata for status.json without the internal same-hour-previous-day figures of the recent check and the holdout."""
    if not meta:
        return {}
    out = {}
    for key in keys:
        value = meta.get(key)
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in INTERNAL_COMPARISON_KEYS}
        out[key] = value
    return out


def export_published(store: Store, out_dir=DEFAULT_DIR, days: int = 90, now=None, last_run: dict | None = None,
                     market_path=MARKET_PATH) -> dict:
    """Write the public files. Returns a summary with row counts."""
    refuse_research(PUBLIC_FEATURE_SETS + (PUBLIC_FEATURE_SET,), "publish")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    today = utc_now(now).tz_convert("Europe/Paris").normalize().tz_localize(None)
    cutoff = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

    sets_sql = ", ".join("?" for _ in PUBLIC_FEATURE_SETS)
    versions = pd.read_sql_query(
        "SELECT forecast_id, delivery_day, issued_at_utc, kind, train_hours, feature_set, model FROM forecasts"
        f" WHERE feature_set IN ({sets_sql}) AND delivery_day >= ? ORDER BY delivery_day, issued_at_utc",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    values = pd.read_sql_query(
        "SELECT v.forecast_id, v.timestamp_utc, v.hour, v.forecast FROM forecast_values v"
        f" JOIN forecasts f ON f.forecast_id = v.forecast_id WHERE f.feature_set IN ({sets_sql}) AND f.delivery_day >= ?",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    versions["premarket"] = premarket_flag(versions["delivery_day"], versions["issued_at_utc"]).astype(int) if len(versions) else []
    forecasts = versions.merge(values, on="forecast_id", how="inner").drop(columns=["forecast_id"])
    forecasts = forecasts.sort_values(["delivery_day", "issued_at_utc", "timestamp_utc"])
    forecasts.to_csv(out / "forecasts.csv", index=False)

    actuals = pd.read_sql_query(
        "SELECT delivery_day, timestamp_utc, hour, price FROM actuals WHERE delivery_day >= ? ORDER BY timestamp_utc",
        store.conn, params=(cutoff,),
    )
    actuals.to_csv(out / "actuals.csv", index=False)

    scores = pd.read_sql_query(
        "SELECT delivery_day, issued_at_utc, feature_set, hours, mae, rmse, settled_at_utc"
        f" FROM scores WHERE feature_set IN ({sets_sql}) AND delivery_day >= ? ORDER BY delivery_day, issued_at_utc",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    scores["premarket"] = premarket_flag(scores["delivery_day"], scores["issued_at_utc"]).astype(int) if len(scores) else []
    model_of = versions[["delivery_day", "issued_at_utc", "feature_set", "model"]].drop_duplicates()
    scores = scores.merge(model_of, on=["delivery_day", "issued_at_utc", "feature_set"], how="left") if len(scores) else scores.assign(model=[])
    scores.to_csv(out / "scores.csv", index=False)

    tomorrow_day = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = forecasts[forecasts["delivery_day"] >= today.strftime("%Y-%m-%d")]
    if not tomorrow.empty:
        latest_day = tomorrow["delivery_day"].max()
        day_versions = versions[versions["delivery_day"] == latest_day]
        preferred = day_versions[day_versions["feature_set"] == PUBLIC_FEATURE_SET]  # the default model's versions, else whatever exists
        day_versions = preferred if len(preferred) else day_versions
        headline = headline_version(day_versions).iloc[0]
        latest = tomorrow[(tomorrow["delivery_day"] == latest_day) & (tomorrow["issued_at_utc"] == headline["issued_at_utc"])
                          & (tomorrow["feature_set"] == headline["feature_set"])]
        latest.to_csv(out / "tomorrow.csv", index=False)
    else:
        pd.DataFrame(columns=forecasts.columns).to_csv(out / "tomorrow.csv", index=False)

    # public: the negative-price probability only; the quantile band and the spike probability stay in the local store, under evaluation
    prob = pd.read_sql_query(
        "SELECT f.delivery_day, f.issued_at_utc, f.feature_set, p.timestamp_utc, p.p_negative AS p_negative_raw, p.p_negative_cal"
        " FROM forecast_prob p JOIN forecasts f ON f.forecast_id = p.forecast_id"
        f" WHERE f.feature_set IN ({sets_sql}) AND f.delivery_day >= ? ORDER BY f.delivery_day, f.issued_at_utc, p.timestamp_utc",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    if len(prob):
        prob["hour"] = pd.to_datetime(prob["timestamp_utc"], utc=True).dt.tz_convert("Europe/Paris").dt.hour  # the public app groups by Paris hour
        prob["calibrated"] = prob["p_negative_cal"].notna().astype(int)
        prob["p_negative"] = prob["p_negative_cal"].fillna(prob["p_negative_raw"])  # what the public app shows: calibrated where it exists
    else:
        prob["hour"], prob["calibrated"], prob["p_negative"] = pd.Series(dtype=int), pd.Series(dtype=int), pd.Series(dtype=float)
    prob = prob[["delivery_day", "issued_at_utc", "feature_set", "timestamp_utc", "p_negative", "p_negative_raw", "calibrated", "hour"]]
    prob.to_csv(out / "probabilistic.csv", index=False)
    if not tomorrow.empty and len(prob):
        head_prob = prob[(prob["delivery_day"] == latest_day) & (prob["issued_at_utc"] == headline["issued_at_utc"])
                         & (prob["feature_set"] == headline["feature_set"])]
        head_prob.to_csv(out / "tomorrow_probabilistic.csv", index=False)
    else:
        pd.DataFrame(columns=prob.columns).to_csv(out / "tomorrow_probabilistic.csv", index=False)

    drivers = pd.read_sql_query(
        "SELECT f.delivery_day, f.issued_at_utc, f.feature_set, d.feature, d.value, d.contribution"
        " FROM forecast_drivers d JOIN forecasts f ON f.forecast_id = d.forecast_id"
        f" WHERE f.feature_set IN ({sets_sql}) AND f.delivery_day >= ? ORDER BY f.delivery_day, f.issued_at_utc, d.rowid",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    drivers.insert(4, "group", drivers["feature"].map(group_of) if len(drivers) else pd.Series(dtype=str))
    drivers.to_csv(out / "drivers.csv", index=False)

    band = pd.DataFrame()
    for band_set in (PUBLIC_FEATURE_SET, FALLBACK_FEATURE_SET):  # the default model's band, else the fallback model's
        band = pd.read_sql_query(
            "SELECT feature_set, hour, n, mae, p10, p25, p50, p75, p90 FROM error_band WHERE feature_set = ? ORDER BY hour",
            store.conn, params=(band_set,),
        )
        if len(band):
            break
    band.to_csv(out / "error_band.csv", index=False)

    status_path = out / "status.json"
    try:
        previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    except ValueError:
        previous_status = {}
    write_market_file(store, out, market_path)  # only where the traded prices are; elsewhere the existing file is read below
    fallback_meta = {}
    fallback_path = out / "model" / f"{FALLBACK_FEATURE_SET}.json"
    if fallback_path.exists():
        fallback_meta = json.loads(fallback_path.read_text())
    model_meta = {}
    meta_path = out / "model" / f"{PUBLIC_FEATURE_SET}.json"
    if meta_path.exists():
        model_meta = json.loads(meta_path.read_text())

    status = {
        "written_at_utc": iso(utc_now(now)),
        "feature_set": PUBLIC_FEATURE_SET,
        "published_feature_sets": list(PUBLIC_FEATURE_SETS),
        "default_model_file": f"model/{PUBLIC_FEATURE_SET}.txt",
        "fallback_model_file": f"model/{FALLBACK_FEATURE_SET}.txt",
        "model_note": "Every forecast version in forecasts.csv and scores.csv carries the model that produced it in the model column: "
                      "gbm:<file> for a stored model, with the suffix :fallback when the default model's inputs were missing and the "
                      "fallback model was used, and gbm for a model fitted live.",
        "window_days": days,
        "from_day": cutoff,
        "tomorrow": tomorrow_day,
        "forecast_versions": int(len(versions)),
        "actual_days": int(actuals["delivery_day"].nunique()) if len(actuals) else 0,
        "scored_versions": int(len(scores)),
        "live_record": live_record(scores, PUBLIC_FEATURE_SET, FALLBACK_FEATURE_SET)[0],
        "model": public_model_meta(model_meta, ("feature_set", "model_file", "fitted_at_utc", "train_from", "train_to", "train_hours",
                                               "recent_check", "frozen_holdout", "features")),
        "fallback_model": public_model_meta(fallback_meta, ("feature_set", "model_file", "fitted_at_utc", "train_from", "train_to",
                                                            "train_hours", "recent_check", "frozen_holdout")),
        "last_run": last_run or {},
        "market": market_status(store, market_path, previous=read_market_file(out) or previous_status.get("market")),
        "shape_battery": shape_status(store, out),
        "probabilistic": probabilistic_status(store, out),
        "attributions": ATTRIBUTIONS,
        "note": "Forecasts use only information available at 12:00 Paris time on the day before delivery. The level forecast has one "
                "benchmark: the EEX French day-ahead future traded before the auction, scored at the auction result, as aggregates "
                "under \"market\" once enough windows are recorded (hit rate, mean P&L per MWh, model and market error, window count). "
                "Forecast error (MAE) is the model's own error against the auction result, with no other comparison. Traded prices are "
                "not republished. Historical errors are shown as context, not as a probability forecast.",
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


STATUS_VOLATILE_KEYS = ("written_at_utc", "last_run")  # change on every export even when the dataset did not


def status_equivalent(old: dict, new: dict, ignore=STATUS_VOLATILE_KEYS) -> bool:
    """True when two status.json contents differ at most in the stamps (written_at_utc, last_run)."""
    def text(status: dict) -> str:  # serialised, so that NaN compares equal to NaN and nested order does not matter
        return json.dumps({k: v for k, v in status.items() if k not in ignore}, sort_keys=True, default=str)

    return text(old) == text(new)


def staged_changes_are_stamps_only(repo, out="published") -> tuple[bool, str]:
    """(True, why) when the staged changes in `repo` are status.json alone and it differs from HEAD only in its stamps.

    The workflow runs this after `git add published` so that a backup settle that arrives after the dispatched one, or any export
    that found nothing new, commits nothing: the data files are byte-identical in that case and only the stamps moved. Any git
    problem counts as a real change, so a publish is never dropped by mistake.
    """
    from .gitsync import _git

    status_path = f"{Path(out).as_posix()}/status.json"
    staged = _git(repo, "diff", "--cached", "--name-only", check=False)
    if staged.returncode != 0:
        return False, f"git diff failed: {staged.stderr.strip()}"
    names = staged.stdout.split()
    if names != [status_path]:
        return False, f"staged: {', '.join(names) if names else 'nothing'}"
    head = _git(repo, "show", f"HEAD:{status_path}", check=False)
    if head.returncode != 0:
        return False, f"no committed {status_path} to compare with"
    try:
        same = status_equivalent(json.loads(head.stdout), json.loads((Path(repo) / status_path).read_text(encoding="utf-8")))
    except (ValueError, OSError) as exc:
        return False, f"could not compare {status_path}: {exc}"
    if same:
        return True, f"only the stamps in {status_path} changed ({', '.join(STATUS_VOLATILE_KEYS)}); nothing new to publish"
    return False, f"{status_path} changed beyond its stamps"


def import_published(store: Store, in_dir=DEFAULT_DIR) -> dict:
    """Load the published files into a (usually empty) store.

    Numbers are parsed exactly (float_precision="round_trip"): pandas' default parser can land one unit in the last place off, which
    made an import followed by an export rewrite scores.csv with a different last digit (run #12 of 2026-10-13 committed exactly that),
    and a publish that changed nothing must leave the files byte for byte as they were.
    """
    src = Path(in_dir)
    summary = {"forecast_versions": 0, "actual_rows": 0, "scores": 0, "error_band_rows": 0, "probabilistic_versions": 0, "driver_versions": 0}
    forecasts_path = src / "forecasts.csv"
    if forecasts_path.exists() and forecasts_path.stat().st_size > 0:
        forecasts = pd.read_csv(forecasts_path, float_precision="round_trip")
        if not forecasts.empty:
            forecasts["timestamp_utc"] = pd.to_datetime(forecasts["timestamp_utc"], utc=True)
            if "feature_set" not in forecasts:  # files written when only one set was published
                forecasts["feature_set"] = FALLBACK_FEATURE_SET
            if "model" not in forecasts:
                forecasts["model"] = "gbm"
            forecasts["model"] = forecasts["model"].fillna("gbm")
            keys = ["delivery_day", "issued_at_utc", "kind", "train_hours", "feature_set", "model"]
            for (day, issued, kind, train_hours, feature_set, model), group in forecasts.groupby(keys, sort=True):
                if not store.forecast_versions(day, feature_set).query("issued_at_utc == @issued").empty:
                    continue
                curve = group.set_index("timestamp_utc")[[c for c in ("hour", "forecast", "naive_day") if c in group]].sort_index()
                store.save_forecast(None, day, feature_set, pd.Timestamp(issued), str(model), str(kind), int(train_hours), True, curve)
                summary["forecast_versions"] += 1
    actuals_path = src / "actuals.csv"
    if actuals_path.exists() and actuals_path.stat().st_size > 0:
        actuals = pd.read_csv(actuals_path, float_precision="round_trip")
        if not actuals.empty:
            index = pd.to_datetime(actuals["timestamp_utc"], utc=True)
            prices = pd.Series(actuals["price"].to_numpy(), index=index)
            summary["actual_rows"] = store.save_actuals(prices, actuals["delivery_day"].reset_index(drop=True))
    scores_path = src / "scores.csv"
    if scores_path.exists() and scores_path.stat().st_size > 0:
        scores = pd.read_csv(scores_path, float_precision="round_trip").rename(columns={"model_won": "mae_below_baseline"})  # files written before the rename
        if "feature_set" not in scores:
            scores["feature_set"] = FALLBACK_FEATURE_SET
        versions = pd.read_sql_query("SELECT forecast_id, delivery_day, issued_at_utc, feature_set FROM forecasts", store.conn)
        merged = scores.merge(versions, on=["delivery_day", "issued_at_utc", "feature_set"], how="inner")
        for _, r in merged.iterrows():
            below = r.get("mae_below_baseline")  # files written before 2026-10-12 carried the internal comparison columns
            store.save_score(int(r["forecast_id"]), r["delivery_day"], r["feature_set"], r["issued_at_utc"], {
                "hours": r["hours"], "mae": r["mae"], "rmse": r["rmse"], "naive_mae": r.get("naive_mae"), "naive_rmse": r.get("naive_rmse"),
                "mae_below_baseline": None if below is None or pd.isna(below) else bool(below),
            }, now=pd.Timestamp(r["settled_at_utc"]) if isinstance(r["settled_at_utc"], str) else None)
            summary["scores"] += 1
    prob_path = src / "probabilistic.csv"
    if prob_path.exists() and prob_path.stat().st_size > 0:
        prob = pd.read_csv(prob_path, float_precision="round_trip")
        if not prob.empty:
            versions = pd.read_sql_query("SELECT forecast_id, delivery_day, issued_at_utc, feature_set FROM forecasts", store.conn)
            merged = prob.merge(versions, on=["delivery_day", "issued_at_utc", "feature_set"], how="inner")
            for fid, group in merged.groupby("forecast_id"):
                if not store.probabilistic_curve(int(fid)).empty:
                    continue
                g = group.copy()
                if "p_negative_raw" in g:  # files written since the calibration: p_negative is the calibrated value, the raw one sits alongside
                    g["p_negative_cal"] = g["p_negative"].where(g.get("calibrated", 1) == 1)
                    g["p_negative"] = g["p_negative_raw"]
                frame = g.set_index(pd.to_datetime(g["timestamp_utc"], utc=True)).reindex(
                    columns=["q10", "q50", "q90", "lo", "hi", "p_negative", "p_spike", "spike_threshold", "p_negative_cal"])  # public files carry the probability only
                frame["conformal_margin"] = (frame["q10"] - frame["lo"]).round(4)
                store.save_probabilistic(int(fid), frame)
                summary["probabilistic_versions"] += 1
    drivers_path = src / "drivers.csv"
    if drivers_path.exists() and drivers_path.stat().st_size > 0:
        drivers = pd.read_csv(drivers_path, float_precision="round_trip")
        if not drivers.empty:
            versions = pd.read_sql_query("SELECT forecast_id, delivery_day, issued_at_utc, feature_set FROM forecasts", store.conn)
            merged = drivers.merge(versions, on=["delivery_day", "issued_at_utc", "feature_set"], how="inner")
            for fid, group in merged.groupby("forecast_id", sort=False):
                if not store.drivers(int(fid)).empty:
                    continue
                store.save_drivers(int(fid), group[["feature", "value", "contribution"]])
                summary["driver_versions"] += 1
    band_path = src / "error_band.csv"
    if band_path.exists() and band_path.stat().st_size > 0:
        band = pd.read_csv(band_path, float_precision="round_trip")
        if not band.empty:
            store.save_error_band(band, source=str(band_path))
            summary["error_band_rows"] = int(len(band))
    return summary

