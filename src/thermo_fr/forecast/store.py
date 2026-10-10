"""SQLite store for the scheduled forecast jobs and the dashboard.

One file, data/forecast.db, with these tables:

- runs           one row per job run (command, kind, delivery day, status, message)
- data_status    one row per run and input: present, absent or error, with hour
                 count, ENTSO-E revision number and a hash of the values
- timing_log     per delivery day and input, the first time it was seen, how
                 many minutes before the 12:00 Paris gate that was, and how many
                 times its values changed in later runs
- input_values   the hourly input values of a delivery day as seen by a run
- forecasts      one row per issued forecast version (delivery day, feature
                 set, issue time); a version is never updated or replaced
- forecast_values  the hourly curve of a version, with the same-hour-D-1 benchmark
- actuals        day-ahead prices fetched by settle
- scores         each version scored against the actuals and the benchmark
- error_band     per feature set and delivery hour, percentiles of the backtest
                 error, used by the dashboard as a historical error band

All timestamps are ISO 8601 strings in UTC; delivery days are YYYY-MM-DD in
Paris time.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

from .timing import gate_closure

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT NOT NULL, kind TEXT NOT NULL, delivery_day TEXT,
    started_at_utc TEXT NOT NULL, finished_at_utc TEXT, status TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS data_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, delivery_day TEXT NOT NULL, item TEXT NOT NULL,
    checked_at_utc TEXT NOT NULL, status TEXT NOT NULL, hours INTEGER, revision TEXT, value_hash TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS timing_log (
    delivery_day TEXT NOT NULL, item TEXT NOT NULL, first_seen_utc TEXT NOT NULL, first_seen_paris TEXT NOT NULL,
    minutes_before_gate REAL NOT NULL, last_hash TEXT, changes INTEGER NOT NULL DEFAULT 0, last_changed_utc TEXT,
    checks INTEGER NOT NULL DEFAULT 1, PRIMARY KEY (delivery_day, item));
CREATE TABLE IF NOT EXISTS input_values (
    run_id INTEGER NOT NULL, delivery_day TEXT NOT NULL, series TEXT NOT NULL, timestamp_utc TEXT NOT NULL, value REAL,
    PRIMARY KEY (run_id, series, timestamp_utc));
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, delivery_day TEXT NOT NULL, feature_set TEXT NOT NULL,
    issued_at_utc TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL, train_hours INTEGER, passes_gate INTEGER NOT NULL,
    UNIQUE (delivery_day, feature_set, issued_at_utc));
CREATE TABLE IF NOT EXISTS forecast_values (
    forecast_id INTEGER NOT NULL, timestamp_utc TEXT NOT NULL, hour INTEGER NOT NULL, forecast REAL NOT NULL, naive_day REAL,
    PRIMARY KEY (forecast_id, timestamp_utc));
CREATE TABLE IF NOT EXISTS forecast_prob (
    forecast_id INTEGER NOT NULL, timestamp_utc TEXT NOT NULL, q10 REAL, q50 REAL, q90 REAL, lo REAL, hi REAL, p_negative REAL, p_spike REAL,
    spike_threshold REAL, conformal_margin REAL, PRIMARY KEY (forecast_id, timestamp_utc));
CREATE TABLE IF NOT EXISTS actuals (
    timestamp_utc TEXT PRIMARY KEY, delivery_day TEXT NOT NULL, hour INTEGER NOT NULL, price REAL NOT NULL, fetched_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scores (
    forecast_id INTEGER PRIMARY KEY, delivery_day TEXT NOT NULL, feature_set TEXT NOT NULL, issued_at_utc TEXT NOT NULL,
    hours INTEGER NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL, naive_mae REAL, naive_rmse REAL, mae_below_baseline INTEGER,
    settled_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS error_band (
    feature_set TEXT NOT NULL, hour INTEGER NOT NULL, n INTEGER NOT NULL, mae REAL NOT NULL,
    p10 REAL NOT NULL, p25 REAL NOT NULL, p50 REAL NOT NULL, p75 REAL NOT NULL, p90 REAL NOT NULL,
    source TEXT, computed_at_utc TEXT NOT NULL, PRIMARY KEY (feature_set, hour));
"""

DEFAULT_PATH = Path("data/forecast.db")


class ForecastExistsError(RuntimeError):
    """A forecast for this delivery day, feature set and issue time is already stored."""


def utc_now(now=None) -> pd.Timestamp:
    return pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")


def iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def value_hash(values) -> str | None:
    """Short, stable hash of a list of hourly values (None when there are none)."""
    clean = [None if pd.isna(v) else round(float(v), 3) for v in values]
    if not clean or all(v is None for v in clean):
        return None
    return hashlib.sha1(json.dumps(clean).encode()).hexdigest()[:12]


class Store:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Rename the scores column of databases written before the wording change (model_won)."""
        columns = [r[1] for r in self.conn.execute("PRAGMA table_info(scores)")]
        if "model_won" in columns and "mae_below_baseline" not in columns:
            self.conn.execute("ALTER TABLE scores RENAME COLUMN model_won TO mae_below_baseline")
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # Runs

    def start_run(self, command: str, kind: str, delivery_day: str | None, now=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (command, kind, delivery_day, started_at_utc) VALUES (?, ?, ?, ?)",
            (command, kind, delivery_day, iso(utc_now(now))),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, message: str = "", now=None) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at_utc = ?, status = ?, message = ? WHERE run_id = ?",
            (iso(utc_now(now)), status, message, run_id),
        )
        self.conn.commit()

    def runs(self, limit: int = 50) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", self.conn, params=(limit,))

    # Data status and the timing log

    def record_status(self, run_id: int | None, delivery_day: str, item: str, status: str, hours: int = 0,
                      revision: str | None = None, hash_: str | None = None, message: str = "", now=None) -> None:
        """Log one check of one input and keep the timing log up to date."""
        checked = utc_now(now)
        self.conn.execute(
            "INSERT INTO data_status (run_id, delivery_day, item, checked_at_utc, status, hours, revision, value_hash, message)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, delivery_day, item, iso(checked), status, int(hours), revision, hash_, message[:500]),
        )
        if status == "present":
            row = self.conn.execute(
                "SELECT last_hash, changes FROM timing_log WHERE delivery_day = ? AND item = ?", (delivery_day, item)
            ).fetchone()
            if row is None:
                gate = gate_closure([delivery_day])[0]
                minutes = (gate - checked).total_seconds() / 60.0
                self.conn.execute(
                    "INSERT INTO timing_log (delivery_day, item, first_seen_utc, first_seen_paris, minutes_before_gate,"
                    " last_hash, changes, last_changed_utc, checks) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, 1)",
                    (delivery_day, item, iso(checked), checked.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M"),
                     round(minutes, 1), hash_),
                )
            elif hash_ is not None and hash_ != row["last_hash"]:
                self.conn.execute(
                    "UPDATE timing_log SET last_hash = ?, changes = changes + 1, last_changed_utc = ?, checks = checks + 1"
                    " WHERE delivery_day = ? AND item = ?",
                    (hash_, iso(checked), delivery_day, item),
                )
            else:
                self.conn.execute(
                    "UPDATE timing_log SET checks = checks + 1 WHERE delivery_day = ? AND item = ?", (delivery_day, item)
                )
        self.conn.commit()

    def status_for(self, delivery_day: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM data_status WHERE delivery_day = ? ORDER BY checked_at_utc", self.conn, params=(delivery_day,)
        )

    def latest_status(self, delivery_day: str) -> pd.DataFrame:
        """The most recent check of each item for a delivery day."""
        frame = self.status_for(delivery_day)
        if frame.empty:
            return frame
        return frame.sort_values("checked_at_utc").groupby("item").tail(1).set_index("item")

    def timing_log(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM timing_log ORDER BY delivery_day, item", self.conn)

    def timing_summary(self) -> pd.DataFrame:
        """Per input across all logged days: how many days, first-appearance statistics, revisions."""
        log = self.timing_log()
        columns = ["item", "days", "days_polled_before_gate", "earliest_min_before_gate", "median_min_before_gate",
                   "latest_min_before_gate", "days_seen_before_gate", "days_with_changes"]
        if log.empty:
            return pd.DataFrame(columns=columns)
        first_checks = pd.read_sql_query(
            "SELECT delivery_day, item, MIN(checked_at_utc) AS first_checked_utc FROM data_status GROUP BY delivery_day, item", self.conn
        )
        log = log.merge(first_checks, on=["delivery_day", "item"], how="left")
        gate = gate_closure(log["delivery_day"])
        first_checked = pd.to_datetime(log["first_checked_utc"], utc=True)
        log["polled_before_gate"] = (first_checked < gate).fillna(False).to_numpy()
        grouped = log.groupby("item")
        return pd.DataFrame({
            "days": grouped.size(),
            "days_polled_before_gate": grouped["polled_before_gate"].sum().astype(int),
            "earliest_min_before_gate": grouped["minutes_before_gate"].max(),
            "median_min_before_gate": grouped["minutes_before_gate"].median(),
            "latest_min_before_gate": grouped["minutes_before_gate"].min(),
            "days_seen_before_gate": grouped["minutes_before_gate"].apply(lambda s: int((s > 0).sum())),
            "days_with_changes": grouped["changes"].apply(lambda s: int((s > 0).sum())),
        }).reset_index()[columns]

    # Input values

    def save_input_values(self, run_id: int, delivery_day: str, frame: pd.DataFrame) -> int:
        rows = []
        for series in frame.columns:
            for ts, value in frame[series].items():
                rows.append((run_id, delivery_day, series, iso(ts), None if pd.isna(value) else float(value)))
        self.conn.executemany(
            "INSERT OR REPLACE INTO input_values (run_id, delivery_day, series, timestamp_utc, value) VALUES (?, ?, ?, ?, ?)", rows
        )
        self.conn.commit()
        return len(rows)

    def input_value_runs(self, delivery_day: str) -> pd.DataFrame:
        """The runs that stored inputs for a delivery day, oldest first, with their start times."""
        return pd.read_sql_query(
            "SELECT r.run_id, r.started_at_utc, r.kind FROM runs r WHERE r.run_id IN"
            " (SELECT DISTINCT run_id FROM input_values WHERE delivery_day = ?) ORDER BY r.started_at_utc",
            self.conn, params=(delivery_day,),
        )

    def input_values_for_run(self, run_id: int, delivery_day: str) -> pd.DataFrame:
        frame = pd.read_sql_query(
            "SELECT series, timestamp_utc, value FROM input_values WHERE run_id = ? AND delivery_day = ?",
            self.conn, params=(int(run_id), delivery_day),
        )
        if frame.empty:
            return pd.DataFrame()
        wide = frame.pivot(index="timestamp_utc", columns="series", values="value")
        wide.index = pd.to_datetime(wide.index, utc=True)
        return wide.sort_index()

    def latest_input_values(self, delivery_day: str) -> pd.DataFrame:
        """Hourly inputs of a delivery day from the most recent run that stored them (wide frame)."""
        run = self.conn.execute(
            "SELECT MAX(run_id) FROM input_values WHERE delivery_day = ?", (delivery_day,)
        ).fetchone()[0]
        if run is None:
            return pd.DataFrame()
        return self.input_values_for_run(run, delivery_day)

    # Forecasts

    def save_forecast(self, run_id: int | None, delivery_day: str, feature_set: str, issued_at, model: str, kind: str,
                      train_hours: int, passes_gate: bool, curve: pd.DataFrame) -> int:
        """Store one forecast version. Raises ForecastExistsError instead of overwriting."""
        try:
            cur = self.conn.execute(
                "INSERT INTO forecasts (run_id, delivery_day, feature_set, issued_at_utc, model, kind, train_hours, passes_gate)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, delivery_day, feature_set, iso(issued_at), model, kind, int(train_hours), int(passes_gate)),
            )
        except sqlite3.IntegrityError as exc:
            raise ForecastExistsError(f"{feature_set} forecast for {delivery_day} issued at {iso(issued_at)} already stored") from exc
        forecast_id = int(cur.lastrowid)
        self.conn.executemany(
            "INSERT INTO forecast_values (forecast_id, timestamp_utc, hour, forecast, naive_day) VALUES (?, ?, ?, ?, ?)",
            [
                (forecast_id, iso(ts), int(row["hour"]), float(row["forecast"]),
                 None if pd.isna(row.get("naive_day")) else float(row["naive_day"]))
                for ts, row in curve.iterrows()
            ],
        )
        self.conn.commit()
        return forecast_id

    def save_probabilistic(self, forecast_id: int, frame: pd.DataFrame) -> None:
        """The quantiles, conformal band and event probabilities of a version (index: valid hour UTC)."""
        columns = ["q10", "q50", "q90", "lo", "hi", "p_negative", "p_spike", "spike_threshold", "conformal_margin"]
        rows = [(int(forecast_id), iso(ts), *[None if pd.isna(row[c]) else float(row[c]) for c in columns]) for ts, row in frame.iterrows()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO forecast_prob (forecast_id, timestamp_utc, " + ", ".join(columns) + ") VALUES (?, ?, " + ", ".join("?" * len(columns)) + ")",
            rows,
        )
        self.conn.commit()

    def probabilistic_curve(self, forecast_id: int) -> pd.DataFrame:
        frame = pd.read_sql_query(
            "SELECT timestamp_utc, q10, q50, q90, lo, hi, p_negative, p_spike, spike_threshold, conformal_margin FROM forecast_prob"
            " WHERE forecast_id = ? ORDER BY timestamp_utc", self.conn, params=(int(forecast_id),))
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        return frame.set_index("timestamp_utc")

    def probabilistic_settled(self, feature_set: str, days: int = 90) -> pd.DataFrame:
        """The headline version's probabilistic curve of each settled day of the last `days`, joined with the actual prices."""
        scores = self.headline_scores(feature_set, days=days)
        frames = []
        for _, row in scores.iterrows():
            curve = self.probabilistic_curve(int(row["forecast_id"]))
            if curve.empty:
                continue
            actual = self.actuals_for(row["delivery_day"])
            curve["actual"] = actual.reindex(curve.index).to_numpy()
            curve["delivery_day"] = row["delivery_day"]
            frames.append(curve)
        return pd.concat(frames) if frames else pd.DataFrame(columns=["q10", "q50", "q90", "lo", "hi", "p_negative", "p_spike", "spike_threshold",
                                                                          "conformal_margin", "actual", "delivery_day"])

    def forecast_versions(self, delivery_day: str, feature_set: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM forecasts WHERE delivery_day = ?"
        params: tuple = (delivery_day,)
        if feature_set:
            query += " AND feature_set = ?"
            params += (feature_set,)
        return pd.read_sql_query(query + " ORDER BY issued_at_utc", self.conn, params=params)

    def forecast_curve(self, forecast_id: int) -> pd.DataFrame:
        frame = pd.read_sql_query(
            "SELECT timestamp_utc, hour, forecast, naive_day FROM forecast_values WHERE forecast_id = ? ORDER BY timestamp_utc",
            self.conn, params=(forecast_id,),
        )
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        return frame.set_index("timestamp_utc")

    def latest_forecast(self, delivery_day: str, feature_set: str) -> tuple[dict | None, pd.DataFrame]:
        versions = self.forecast_versions(delivery_day, feature_set)
        if versions.empty:
            return None, pd.DataFrame()
        meta = versions.iloc[-1].to_dict()
        return meta, self.forecast_curve(int(meta["forecast_id"]))

    def headline_forecast(self, delivery_day: str, feature_set: str) -> tuple[dict | None, pd.DataFrame, pd.DataFrame]:
        """The pre-market version (last issued before the market window), else the latest; plus the later versions.

        Returns (meta with a `premarket` flag, curve, later versions issued
        after the market window).
        """
        from .market import headline_version, premarket_flag

        versions = self.forecast_versions(delivery_day, feature_set)
        if versions.empty:
            return None, pd.DataFrame(), pd.DataFrame()
        versions = versions.copy()
        versions["premarket"] = premarket_flag(versions["delivery_day"], versions["issued_at_utc"])
        meta = headline_version(versions).iloc[0].to_dict()
        later = versions[~versions["premarket"]]
        return meta, self.forecast_curve(int(meta["forecast_id"])), later

    def forecast_days(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT delivery_day FROM forecasts ORDER BY delivery_day")]

    # Actuals and scores

    def save_actuals(self, prices: pd.Series, delivery_days: pd.Series, now=None) -> int:
        fetched = iso(utc_now(now))
        local = prices.index.tz_convert("Europe/Paris")
        rows = [
            (iso(ts), str(day)[:10], int(h), float(p), fetched)
            for ts, day, h, p in zip(prices.index, delivery_days, local.hour, prices.to_numpy())
            if not pd.isna(p)
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO actuals (timestamp_utc, delivery_day, hour, price, fetched_at_utc) VALUES (?, ?, ?, ?, ?)", rows
        )
        self.conn.commit()
        return len(rows)

    def actuals_for(self, delivery_day: str) -> pd.Series:
        frame = pd.read_sql_query(
            "SELECT timestamp_utc, price FROM actuals WHERE delivery_day = ? ORDER BY timestamp_utc", self.conn, params=(delivery_day,)
        )
        index = pd.to_datetime(frame["timestamp_utc"], utc=True)
        return pd.Series(frame["price"].to_numpy(), index=index, name="price")

    def actual_days(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT delivery_day FROM actuals ORDER BY delivery_day")]

    def save_score(self, forecast_id: int, delivery_day: str, feature_set: str, issued_at: str, score: dict, now=None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO scores (forecast_id, delivery_day, feature_set, issued_at_utc, hours, mae, rmse,"
            " naive_mae, naive_rmse, mae_below_baseline, settled_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (forecast_id, delivery_day, feature_set, issued_at, int(score["hours"]), float(score["mae"]), float(score["rmse"]),
             score.get("naive_mae"), score.get("naive_rmse"),
             None if score.get("mae_below_baseline") is None else int(score["mae_below_baseline"]), iso(utc_now(now))),
        )
        self.conn.commit()

    def unscored_forecasts(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT f.* FROM forecasts f LEFT JOIN scores s ON s.forecast_id = f.forecast_id WHERE s.forecast_id IS NULL"
            " ORDER BY f.delivery_day, f.issued_at_utc",
            self.conn,
        )

    def scores(self, days: int | None = None) -> pd.DataFrame:
        frame = pd.read_sql_query("SELECT * FROM scores ORDER BY delivery_day, feature_set, issued_at_utc", self.conn)
        if days and not frame.empty:
            cutoff = (pd.Timestamp(frame["delivery_day"].max()) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            frame = frame[frame["delivery_day"] > cutoff]
        return frame

    def latest_scores(self, feature_set: str, days: int = 30) -> pd.DataFrame:
        """The score of the last issued version per delivery day, for one feature set."""
        frame = self.scores()
        frame = frame[frame["feature_set"] == feature_set]
        if frame.empty:
            return frame
        last = frame.sort_values("issued_at_utc").groupby("delivery_day").tail(1).sort_values("delivery_day")
        return last.tail(days)

    def headline_scores(self, feature_set: str, days: int = 30) -> pd.DataFrame:
        """The score of each day's headline version: the pre-market one when it exists, else the latest."""
        from .market import headline_version

        frame = self.scores()
        frame = frame[frame["feature_set"] == feature_set]
        if frame.empty:
            return frame
        return headline_version(frame).sort_values("delivery_day").tail(days)

    # Error band

    def save_error_band(self, band: pd.DataFrame, source: str, now=None) -> None:
        computed = iso(utc_now(now))
        self.conn.executemany(
            "INSERT OR REPLACE INTO error_band (feature_set, hour, n, mae, p10, p25, p50, p75, p90, source, computed_at_utc)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (r["feature_set"], int(r["hour"]), int(r["n"]), float(r["mae"]), float(r["p10"]), float(r["p25"]),
                 float(r["p50"]), float(r["p75"]), float(r["p90"]), source, computed)
                for _, r in band.iterrows()
            ],
        )
        self.conn.commit()

    def error_band(self, feature_set: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM error_band WHERE feature_set = ? ORDER BY hour", self.conn, params=(feature_set,)
        ).set_index("hour")


def error_band_from_predictions(predictions: pd.DataFrame, feature_set: str, model: str = "gbm", strict_only: bool = True) -> pd.DataFrame:
    """Percentiles of the signed error (forecast minus actual) by delivery hour from a backtest.

    `predictions` is the frame written by the backtest (columns hour, actual,
    strict and one per model). The band describes past errors of the same model
    at the same hour; it is not a probability forecast for any particular day.
    """
    frame = predictions.dropna(subset=["actual", model])
    if strict_only and "strict" in frame:
        frame = frame[frame["strict"].astype(bool)]
    if frame.empty:
        raise ValueError("No rows to build an error band from.")
    error = frame[model] - frame["actual"]
    grouped = error.groupby(frame["hour"].astype(int))
    band = pd.DataFrame({
        "n": grouped.size(),
        "mae": grouped.apply(lambda s: float(s.abs().mean())),
        "p10": grouped.quantile(0.10),
        "p25": grouped.quantile(0.25),
        "p50": grouped.quantile(0.50),
        "p75": grouped.quantile(0.75),
        "p90": grouped.quantile(0.90),
    })
    band.index.name = "hour"
    band = band.reset_index()
    band.insert(0, "feature_set", feature_set)
    return band


def score_curve(curve: pd.DataFrame, actual: pd.Series) -> dict | None:
    """MAE and RMSE of a stored curve against actual prices, and the same for its benchmark."""
    joined = curve.join(actual.rename("actual"), how="inner").dropna(subset=["actual"])
    if joined.empty:
        return None
    err = joined["forecast"] - joined["actual"]
    out = {"hours": int(len(joined)), "mae": float(err.abs().mean()), "rmse": float((err**2).mean() ** 0.5)}
    naive = joined.dropna(subset=["naive_day"])
    if not naive.empty:
        nerr = naive["naive_day"] - naive["actual"]
        out["naive_mae"] = float(nerr.abs().mean())
        out["naive_rmse"] = float((nerr**2).mean() ** 0.5)
        out["mae_below_baseline"] = bool(out["mae"] < out["naive_mae"])
    else:
        out["naive_mae"] = out["naive_rmse"] = out["mae_below_baseline"] = None
    return out
