"""The weather collector against stand-in answers shaped like the ones the sources gave to the probe of 2026-11-20."""
import datetime as dt
import gzip
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("weather_collect", Path(__file__).resolve().parents[1] / "scripts" / "weather_collect.py")
wc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wc)
UTC = dt.timezone.utc


def source(runs_listed, calls, fail=()):
    """A stand-in for the network. Answers by URL; a POST gets one result per query, with as many runs as the query's range holds."""
    def fetch(url, body=None, **_):
        calls.append((url, body))
        ok = lambda payload: {"url": url, "status": 200, "body": payload if isinstance(payload, bytes) else json.dumps(payload).encode(), "seconds": 0.1,  # noqa: E731
                              "asked_utc": "2026-11-21T08:50:00Z", "answered_utc": "2026-11-21T08:50:01Z", "error": "", "attempts": 1, "request_sha256": None, "work_units": "7"}
        if any(word in url for word in fail):
            return {**ok(b"down"), "status": 503, "error": "HTTP 503"}
        if "/v1/forecasts/runs" in url:
            return ok({"dataProductId": url.split("dataProductId=")[1].split("&")[0], "snapshotId": "SNAP", "runs": [{"initTime": r} for r in runs_listed]})
        if url.endswith("/v1/forecasts"):
            results = []
            for q in body["queries"]:
                members = 31 if "gefs" in q["dataProductId"] else 51 if "ifs" in q["dataProductId"] else 1
                step = 1 if members == 1 else 3
                leads = list(range(0, q["maxLeadTimeHours"] + 1, step))
                if "initTime" in q:
                    inits = [q["initTime"]]
                else:
                    t, end, inits = dt.datetime.fromisoformat(q["initTimesAfter"].replace("Z", "+00:00")), dt.datetime.fromisoformat(q["initTimesBefore"].replace("Z", "+00:00")), []
                    t = t.replace(minute=0) + dt.timedelta(hours=1)
                    while t < end:
                        if t.hour % (6 if members == 1 else 24) == 0:
                            inits.append(wc.iso(t))
                        t += dt.timedelta(hours=1)
                forecasts = []
                for init in inits:
                    series = {"wind_u_100m": [3.0] * len(leads), "wind_v_100m": [4.0] * len(leads)}
                    f = {"initTime": init, "validTimes": ["t"] * len(leads), "leadTimeHours": [float(x) for x in leads]}
                    f.update({"data": series} if members == 1 else {"ensembleMembers": [{"member": m, "data": series} for m in range(members)]})
                    forecasts.append(f)
                results.append({"queryId": q["queryId"], "forecasts": forecasts, "snapshotId": "SNAP"})
            return ok({"results": results})
        if "meta.json" in url:
            return ok({"last_run_initialisation_time": 1795262400, "last_run_availability_time": 1795284000})
        if "single-runs" in url and "ecmwf_ifs&run=2024" in url:
            run = dt.datetime.strptime(url.split("run=")[1][:16], "%Y-%m-%dT%H:%M")
            if run.day == 3:
                return ok([{"hourly": {"time": ["t"] * 80, "wind_speed_100m": [None] * 80}}] * 17)  # a run the archive does not hold
            times = [(run + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(168)]
            return ok([{"hourly": {"time": times, "wind_speed_100m": [float(h % 7) for h in range(168)]}}] * 17)
        if "single-runs" in url:
            return ok([{"hourly": {"time": ["t"] * 3, "wind_speed_100m": [5.0, 6.0, None]}}] * 34)
        return ok(b"<Key>x/gfs.t00z.pgrb2.0p25.f010</Key><LastModified>2026-11-21T03:40:00.000Z</LastModified>")
    return fetch


def test_one_morning_is_recorded_once_with_every_answer_kept(tmp_path):
    calls, lines = [], []
    clock = lambda: dt.datetime(2026, 11, 21, 8, 50, tzinfo=UTC)  # noqa: E731  09:50 Paris: the slot before the gate
    out = wc.collect(tmp_path, clock=clock, fetch=source(["2026-11-21T00:00:00Z", "2026-11-20T18:00:00Z"], calls), log=lines.append)
    record = out["record"]
    assert out["status"] == "recorded" and record["slot"] == "gate" and record["gate_utc"] == "2026-11-21T09:05:00Z" and record["finished_by_gate"]
    assert all(p["run_used"] == "2026-11-21T00:00:00Z" and not p["fallback"] and p["points_answered"] == 34 for p in record["products"].values())
    assert [record["products"][p]["values"] for p in ("gfs", "gefs", "ifs_ens")] == [34 * 55, 34 * 17 * 31, 34 * 17 * 51]
    assert record["flags"] == [] and record["open_meteo_runs"]["icon_eu"] == {"points": 34, "hours_present_min": 2, "hours_present_max": 2}
    posts = [b for u, b in calls if b]
    assert all(len(b["queries"]) <= 10 for b in posts) and len(posts) == 12  # 34 points in fours of ten, three products
    folder = tmp_path / "live" / "2026-11-21" / "gate"
    assert all(wc.sha((folder / r["file"]).read_bytes()) == r["sha256"] for r in record["requests"]) and len(record["requests"]) == len(calls)
    assert json.loads((tmp_path / "live" / "manifest.jsonl").read_text().strip())["record_sha256"] == wc.sha((tmp_path / "live" / "2026-11-21" / "gate.json").read_bytes())
    again = wc.collect(tmp_path, clock=clock, fetch=source([], calls), log=lines.append)
    assert again["status"] == "already recorded" and len((tmp_path / "live" / "manifest.jsonl").read_text().splitlines()) == 1


def test_a_missing_midnight_run_is_replaced_by_the_last_one_and_said_so_and_a_dead_source_is_flagged(tmp_path):
    calls = []
    clock = lambda: dt.datetime(2026, 7, 1, 5, 30, tzinfo=UTC)  # noqa: E731  07:30 Paris in summer
    record = wc.collect(tmp_path, clock=clock, fetch=source(["2026-06-30T18:00:00Z", "2026-06-30T00:00:00Z"], calls, fail=("open-meteo", "amazonaws")), log=lambda *_: None)["record"]
    assert record["slot"] == "early" and record["gate_utc"] == "2026-07-01T08:05:00Z"
    assert all(p["run_used"] == "2026-06-30T00:00:00Z" and p["fallback"] for p in record["products"].values())
    assert sum("was not there" in f for f in record["flags"]) == 3 and any(f.startswith("open-meteo meta") for f in record["flags"]) and any(f.startswith("noaa listing") for f in record["flags"])


def test_history_is_resumable_and_packs_to_one_table(tmp_path):
    calls = []
    fetch = source([], calls)
    first = wc.history(tmp_path, "gfs", "de", dt.date(2021, 5, 1), dt.date(2021, 5, 4), fetch=fetch, log=lambda *_: None, limit=3)
    assert first == {"asked": 3, "saved": 3, "failed": 0, "stopped": False}
    rest = wc.history(tmp_path, "gfs", "de", dt.date(2021, 5, 1), dt.date(2021, 5, 4), fetch=fetch, log=lambda *_: None)
    assert rest["asked"] == 1 and rest["saved"] == 4 and len(calls) == 4  # two chunks of two days, 17 points in a ten and a seven
    assert wc.history(tmp_path, "gfs", "de", dt.date(2021, 5, 1), dt.date(2021, 5, 4), fetch=fetch, log=lambda *_: None)["asked"] == 0
    table = gzip.open(wc.pack(tmp_path, "gfs", "de", log=lambda *_: None), "rt").read().splitlines()
    assert table[0] == "init_utc,lead_h,point,speed_ms" and len(table) - 1 == 16 * 55 * 17 and "2021-05-01T00:00:00Z,0,de_NordseeKueste,5.00" in table
    wc.history(tmp_path, "gefs", "de", dt.date(2021, 5, 1), dt.date(2021, 5, 10), fetch=fetch, log=lambda *_: None)
    ens = gzip.open(wc.pack(tmp_path, "gefs", "de", log=lambda *_: None), "rt").read().splitlines()
    assert ens[0].split(",")[:4] == ["init_utc", "lead_h", "point", "m00"] and len(ens[0].split(",")) == 34 and len(ens) - 1 == 10 * 25 * 17


def test_the_ecmwf_runs_are_fetched_a_few_a_day_and_an_empty_run_is_not_counted_as_saved(tmp_path):
    calls = []
    fetch = source([], calls)
    first = wc.history_ecmwf(tmp_path, "de", dt.date(2024, 4, 1), dt.date(2024, 4, 6), fetch=fetch, log=lambda *_: None, limit=4)
    assert first == {"asked": 4, "saved": 3, "failed": 1, "left": 3, "stopped": False}  # 3 April is empty in the stand-in archive
    second = wc.history_ecmwf(tmp_path, "de", dt.date(2024, 4, 1), dt.date(2024, 4, 6), fetch=fetch, log=lambda *_: None, limit=4)
    assert (second["asked"], second["saved"], second["failed"]) == (3, 5, 1) and len(calls) == 7  # the empty run is asked again, the saved ones are not
    assert all("models=ecmwf_ifs" in u and "wind_speed_unit=ms" in u and u.count(",") == 32 for u, _ in calls)
    table = gzip.open(wc.pack_ecmwf(tmp_path, "de", log=lambda *_: None), "rt").read().splitlines()
    assert table[0] == "init_utc,lead_h,point,speed_ms" and len(table) - 1 == 5 * 72 * 17
    assert "2024-04-01T00:00:00Z,0,de_NordseeKueste,0.00" in table and "2024-04-06T00:00:00Z,71,de_OffOstsee,1.00" in table
