"""The scheduled jobs: morning-run and settle.

morning-run, several times during the morning of D-1:

1. asks ENTSO-E for the day-ahead load forecast, the wind and solar forecasts
   and the prices around the next delivery day D, and Open-Meteo for the
   weather forecasts issued two days ahead, including the point forecasts
   that feed the wind and solar generation proxies;
2. records for every input whether it is present for D, how many hours it
   has, its revision number and a hash of its values, so the timing log can
   say when each input first appeared and whether it changed between runs
   (this replaces the one-off timing probe);
3. stores the hourly inputs of D, then stores the forecast for D with the
   honest feature set, predicted with the stored model under published/model
   (the same file the GitHub Actions workflow uses, so a local run and the
   workflow give the same curve for the same inputs) or fitted live when no
   stored model is given, and, when the wind and solar forecasts exist, the
   extended set, which has no stored model and is always fitted live; every
   forecast is a new version stamped with its issue time;
4. refreshes the error band from the backtest predictions when they exist.

A source that is down is recorded as an error in data_status and the run
continues with what it has; the run status is "ok", "partial" or "failed".

settle, in the afternoon after the auction results are out: fetches the actual
prices for the days that have forecasts (and for today, so the dashboard can
show it), stores them, and scores every unscored forecast version against the
actuals and against the same-hour-previous-day benchmark stored with it.

Every message goes through the logging module; the command line attaches a
file handler under logs/. The ENTSO-E token is never part of any message.
"""

import json
import logging
import traceback
from pathlib import Path

import pandas as pd

from ..config import LOCAL_TZ
from ..data.entsoe_client import EntsoeSource
from ..data.solar_points import SolarPointsSource
from ..data.weather_forecast import OpenMeteoForecastSource
from ..data.wind_points import WindPointsSource
from .day import HISTORY_DAYS, forecast_day, merge_inputs
from .features import FEATURES, build_features
from .gen_proxy import apply_weights, load_weights
from .inputs import INPUT_COLUMNS, NEIGHBOUR_PRICE_COLUMNS, load_inputs
from .solar_proxy import SOLAR
from .store import Store, error_band_from_predictions, score_curve, value_hash
from .timing import delivery_days
from .wind_proxy import WIND

log = logging.getLogger("thermo_fr.jobs")

DEFAULT_WEIGHTS_PATH = WIND.default_path
DEFAULT_MODEL_FILE = Path("published/model/honest.txt")
INPUT_SERIES = ["load_fc_mw", "solar_fc_mw", "wind_onshore_fc_mw", "wind_offshore_fc_mw", "temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2",
                "wind_proxy_mw", "solar_proxy_mw", "nuclear_mw"] + NEIGHBOUR_PRICE_COLUMNS
WEATHER_SERIES = ["temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"]
FEATURE_SETS = ("honest", "extended")


def next_delivery_day(now=None) -> str:
    now = pd.Timestamp(now).tz_convert(LOCAL_TZ) if now is not None else pd.Timestamp.now(tz=LOCAL_TZ)
    return (now.normalize() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")


def day_hours(day: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(day, tz=LOCAL_TZ)
    end = (pd.Timestamp(day) + pd.DateOffset(days=1)).tz_localize(LOCAL_TZ)
    return pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")


def presence(frame: pd.DataFrame | pd.Series, day: str) -> tuple[str, int, str | None]:
    """status, hour count and value hash of the delivery day's rows in a series or frame."""
    hours = day_hours(day)
    part = frame.reindex(hours)
    if isinstance(part, pd.Series):
        part = part.to_frame()
    complete = part.dropna(how="any")
    if complete.empty:
        return "absent", 0, None
    values = [v for column in part.columns for v in part[column].to_numpy()]
    return "present", int(len(complete)), value_hash(values)


def fetch_fresh(day: str, cache_dir: Path, store: Store, run_id: int, entsoe=None, weather=None, now=None, wind_points=None,
                wind_weights=WIND.default_path, solar_points=None, solar_weights=SOLAR.default_path) -> pd.DataFrame:
    """Fetch every input around `day`, recording each one's status; failures are logged, not raised.

    The wind and solar proxies are derived from the point forecasts with the
    calibration weights in `wind_weights` and `solar_weights`; a missing
    weights file is recorded as an error for the item wind_proxy or
    solar_proxy and that column stays NaN.
    """
    target = pd.Timestamp(day)
    start = (target - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    entsoe = entsoe or EntsoeSource(cache_dir=Path(cache_dir) / "entsoe")
    weather = weather or OpenMeteoForecastSource(cache_dir=Path(cache_dir) / "open-meteo")
    wind_points = wind_points or WindPointsSource(cache_dir=Path(cache_dir) / "open-meteo")
    solar_points = solar_points or SolarPointsSource(cache_dir=Path(cache_dir) / "open-meteo")
    pieces = []

    def attempt(item: str, fetch, revision_of=lambda: None):
        try:
            result = fetch()
        except Exception as exc:  # noqa: BLE001  a source that is down must not stop the run
            message = f"{exc.__class__.__name__}: {exc}"
            log.error("%s failed: %s", item, message)
            store.record_status(run_id, day, item, "error", message=message, now=now)
            return
        status, hours, digest = presence(result, day)
        store.record_status(run_id, day, item, status, hours=hours, revision=revision_of(), hash_=digest, now=now)
        log.info("%s for %s: %s (%d hours, revision %s)", item, day, status, hours, revision_of())
        pieces.append(result)

    def entsoe_revision(item):
        def latest():
            meta = entsoe.details.get(item) or []
            return meta[-1].get("revision") if meta else None

        return latest

    attempt("load_forecast", lambda: entsoe.load_forecast(start, end), entsoe_revision("load_forecast"))
    attempt("wind_solar_forecast", lambda: entsoe.wind_solar_forecast(start, end), entsoe_revision("wind_solar_forecast"))
    attempt("prices", lambda: entsoe.day_ahead_prices(start, end), entsoe_revision("prices"))
    attempt("nuclear_actual", lambda: entsoe.nuclear_generation_actual(start, end), entsoe_revision("nuclear_actual"))
    attempt("neighbour_prices", lambda: entsoe.neighbour_prices(start, end))
    attempt("weather_issued", lambda: weather.fetch(start, end, kinds=("issued",))[WEATHER_SERIES])
    attempt("wind_points", lambda: wind_points.fetch_points(start, end))
    attempt("solar_points", lambda: solar_points.fetch_points(start, end))

    def proxy(spec, weights_path):
        def compute():
            points = next((p for p in pieces if isinstance(p, pd.DataFrame) and any(c in spec.columns for c in p.columns)), None)
            if points is None:
                raise ValueError(f"no {spec.name} point forecasts in this run")
            weights = load_weights(weights_path)
            if weights is None:
                raise FileNotFoundError(f"no {spec.name} proxy weights at {weights_path} (written by refit-model)")
            return apply_weights(points, weights, spec)

        return compute

    attempt("wind_proxy", proxy(WIND, wind_weights))
    attempt("solar_proxy", proxy(SOLAR, solar_weights))
    if not pieces:
        return pd.DataFrame(columns=INPUT_COLUMNS)
    fresh = pd.concat(pieces, axis=1).sort_index()
    return fresh.reindex(columns=INPUT_COLUMNS)


def refresh_error_band(store: Store, reports_dir=Path("reports/forecast"), now=None) -> list[str]:
    done = []
    for feature_set in FEATURE_SETS:
        path = Path(reports_dir) / f"predictions_{feature_set}.csv"
        if not path.exists():
            continue
        predictions = pd.read_csv(path)
        band = error_band_from_predictions(predictions, feature_set)
        store.save_error_band(band, source=str(path), now=now)
        done.append(feature_set)
    return done


def missing_features(day: str, inputs: pd.DataFrame, feature_set: str) -> list[str]:
    """The features of the set that have no value at all on the delivery day's rows (an input that did not arrive)."""
    table = build_features(inputs, feature_set)
    rows = table.info["delivery_day"] == pd.Timestamp(day).normalize()
    if not rows.any():
        return list(FEATURES[feature_set])
    return [f for f in table.X.columns if table.X.loc[rows, f].isna().all()]


def stored_model_feature_set(model_file) -> str:
    """The feature set a stored model was fitted on, from the metadata next to it (honest when there is none)."""
    meta = Path(model_file).with_suffix(".json")
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("feature_set", "honest")
        except (OSError, ValueError):
            pass
    return "honest"


def morning_run(store: Store, delivery_day: str | None = None, kind: str = "scheduled", inputs_path=Path("data/forecast/inputs.csv"),
                cache_dir=Path("data/cache"), reports_dir=Path("reports/forecast"), entsoe=None, weather=None, now=None,
                feature_sets=FEATURE_SETS, model_file=None, wind_points=None, wind_weights=WIND.default_path, solar_points=None,
                solar_weights=SOLAR.default_path, fallback_model_file=DEFAULT_MODEL_FILE) -> dict:
    """One morning run. Returns a summary dict; never raises for a source failure.

    If the stored model needs an input that did not arrive for the delivery
    day (every value of a feature missing on the day's rows), the forecast is
    made with `fallback_model_file` instead, provided that model does not need
    the missing features either; the fallback is recorded in data_status
    under the item "model" with status "fallback", the version's model name
    ends in ":fallback" and the run finishes as partial. Without a usable
    fallback that set's forecast is not produced: nothing predicts with
    missing features.

    With `model_file` (a LightGBM file written by refit) the forecast of the
    feature set that model was fitted on is a prediction with that model
    instead of a refit, so no inputs history is needed beyond the days
    fetched around the delivery day; the other feature sets are fitted live.
    A `model_file` that does not exist is recorded as an error for the item
    "model" and every set is fitted live, so the run still produces a
    forecast but is marked partial.
    """
    day = delivery_day or next_delivery_day(now)
    run_id = store.start_run("morning-run", kind, day, now=now)
    log.info("morning-run %s for delivery day %s (run %d)", kind, day, run_id)
    summary = {"run_id": run_id, "delivery_day": day, "kind": kind, "forecasts": {}, "errors": []}
    try:
        if model_file is not None and not Path(model_file).exists():
            message = f"no stored model at {model_file} (written by refit-model); the forecasts are fitted live instead"
            log.error("model: %s", message)
            store.record_status(run_id, day, "model", "error", message=message, now=now)
            model_file = None
        stored_for = stored_model_feature_set(model_file) if model_file is not None else None
        fresh = fetch_fresh(day, Path(cache_dir), store, run_id, entsoe=entsoe, weather=weather, now=now, wind_points=wind_points,
                            wind_weights=wind_weights, solar_points=solar_points, solar_weights=solar_weights)
        history = load_inputs(inputs_path) if Path(inputs_path).exists() else pd.DataFrame(columns=INPUT_COLUMNS)
        merged = merge_inputs(history, fresh)
        hours = day_hours(day)
        present = merged.reindex(hours)[INPUT_SERIES]
        store.save_input_values(run_id, day, present)
        issued_at = pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")
        for feature_set in feature_sets:
            try:
                stored = model_file if feature_set == stored_for else None
                predict_set, suffix = feature_set, ""
                if stored is not None:
                    missing = missing_features(day, merged, feature_set)
                    fallback = Path(fallback_model_file) if fallback_model_file is not None else None
                    usable = fallback is not None and fallback.exists() and fallback.resolve() != Path(stored).resolve()
                    if missing and usable:
                        fallback_set = stored_model_feature_set(fallback)
                        still_missing = [f for f in missing if f in FEATURES[fallback_set]]
                        if still_missing:
                            raise ValueError(f"inputs missing for {day}: {', '.join(missing)}; the fallback model {fallback.name} needs "
                                             f"{', '.join(still_missing)} too")
                        message = (f"{Path(stored).name} needs {', '.join(missing)}, missing for {day}; predicted with the fallback "
                                   f"{fallback.name} ({fallback_set}) instead")
                        log.warning("model: %s", message)
                        store.record_status(run_id, day, "model", "fallback", message=message, now=now)
                        stored, predict_set, suffix = fallback, fallback_set, ":fallback"
                    elif missing:
                        raise ValueError(f"inputs missing for {day}: {', '.join(missing)}; no fallback model available")
                curve = forecast_day(day, merged, None, log=log.info, feature_set=predict_set, model_file=stored)
                model_name = "gbm" if stored is None else f"gbm:{Path(stored).name}{suffix}"
                forecast_id = store.save_forecast(
                    run_id, day, feature_set, issued_at, model_name, kind, curve.attrs["train_hours"], curve.attrs["passes_gate"], curve
                )
                summary["forecasts"][feature_set] = forecast_id
                log.info("%s forecast for %s stored as version %d (issued %s)", feature_set, day, forecast_id, issued_at)
            except Exception as exc:  # noqa: BLE001
                message = f"{exc.__class__.__name__}: {exc}"
                log.error("%s forecast for %s not produced: %s", feature_set, day, message)
                store.record_status(run_id, day, f"forecast_{feature_set}", "error", message=message, now=now)
                summary["errors"].append(f"forecast_{feature_set}: {message}")
        summary["error_band"] = refresh_error_band(store, reports_dir, now=now)
    except Exception as exc:  # noqa: BLE001
        message = f"{exc.__class__.__name__}: {exc}"
        log.error("morning-run failed: %s\n%s", message, traceback.format_exc())
        store.finish_run(run_id, "failed", message, now=now)
        summary["status"] = "failed"
        summary["errors"].append(message)
        return summary
    status_rows = store.status_for(day)
    source_errors = status_rows[(status_rows["run_id"] == run_id) & (status_rows["status"].isin(["error", "fallback"]))]
    if not summary["forecasts"]:
        status = "failed"
    elif len(source_errors) or len(summary["forecasts"]) < len(feature_sets):
        status = "partial"
    else:
        status = "ok"
    message = "; ".join(source_errors["item"] + ": " + source_errors["message"].str[:80]) if len(source_errors) else ""
    store.finish_run(run_id, status, message, now=now)
    summary["status"] = status
    log.info("morning-run finished: %s %s", status, message)
    return summary


def settle(store: Store, cache_dir=Path("data/cache"), entsoe=None, now=None, kind: str = "scheduled") -> dict:
    """Fetch actual prices for today, tomorrow and every forecast day, and score what is unscored."""
    today = (pd.Timestamp(now).tz_convert(LOCAL_TZ) if now is not None else pd.Timestamp.now(tz=LOCAL_TZ)).normalize()
    run_id = store.start_run("settle", kind, None, now=now)
    summary = {"run_id": run_id, "scored": 0, "actual_days": [], "errors": []}
    entsoe = entsoe or EntsoeSource(cache_dir=Path(cache_dir) / "entsoe")
    wanted = set(store.forecast_days()) | {today.strftime("%Y-%m-%d"), (today + pd.DateOffset(days=1)).strftime("%Y-%m-%d")}
    already = set(store.actual_days())
    missing = sorted(d for d in wanted if d not in already or d >= today.strftime("%Y-%m-%d"))
    for day in missing:
        start = (pd.Timestamp(day) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")  # the Paris day starts on the UTC day before
        end = (pd.Timestamp(day) + pd.DateOffset(days=2)).strftime("%Y-%m-%d")
        try:
            prices = entsoe.day_ahead_prices(start, end)
        except Exception as exc:  # noqa: BLE001
            message = f"{exc.__class__.__name__}: {exc}"
            log.error("prices for %s failed: %s", day, message)
            store.record_status(run_id, day, "prices", "error", message=message, now=now)
            summary["errors"].append(f"prices {day}: {message}")
            continue
        status, hours, digest = presence(prices, day)
        meta = entsoe.details.get("prices") or []
        store.record_status(run_id, day, "prices", status, hours=hours, revision=meta[-1].get("revision") if meta else None,
                            hash_=digest, now=now)
        if status == "present":
            days = delivery_days(prices.index).strftime("%Y-%m-%d")
            keep = days == day
            store.save_actuals(prices[keep], pd.Series(days[keep]), now=now)
            summary["actual_days"].append(day)
            log.info("actual prices for %s stored (%d hours)", day, hours)
        else:
            log.info("actual prices for %s not published yet", day)
    for _, version in store.unscored_forecasts().iterrows():
        actual = store.actuals_for(version["delivery_day"])
        if actual.empty:
            continue
        score = score_curve(store.forecast_curve(int(version["forecast_id"])), actual)
        if score is None:
            continue
        store.save_score(int(version["forecast_id"]), version["delivery_day"], version["feature_set"], version["issued_at_utc"], score, now=now)
        summary["scored"] += 1
        log.info("scored %s %s issued %s: MAE %.2f (benchmark %.2f)", version["delivery_day"], version["feature_set"],
                 version["issued_at_utc"], score["mae"], score["naive_mae"] or float("nan"))
    status = "ok" if not summary["errors"] else ("partial" if summary["actual_days"] or summary["scored"] else "failed")
    store.finish_run(run_id, status, "; ".join(summary["errors"]), now=now)
    summary["status"] = status
    return summary


def setup_logging(name: str, logs_dir=Path("logs")) -> Path:
    """Console plus one file per command and day under logs/."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{name}_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.log"
    root = logging.getLogger("thermo_fr")
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(path.resolve()) for h in root.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        root.addHandler(stream)
    return path
