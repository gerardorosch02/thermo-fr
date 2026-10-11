"""The laptop's market publish records its outcome in the store (ok, skipped, failed) so the dashboard can show it."""

import subprocess
import time
from pathlib import Path

import pandas as pd

from test_gitsync import LOCAL, git, make_repos
from test_market_file_carry import seeded_store_and_market
from thermo_fr.cli import market_publish
from thermo_fr.dashboard.data import last_publish_run
from thermo_fr.forecast.store import Store


def test_publish_records_ok_skipped_and_failed_runs(tmp_path):
    remote, laptop, actions = make_repos(tmp_path)
    store, market = seeded_store_and_market(tmp_path, days=2)
    # ok: written and pushed with the local identity
    result = market_publish(store, laptop / "published", market, laptop, kind="scheduled", log=_Quiet())
    assert result["status"] == "ok" and result["pushed"] and result["identity"] == f"{LOCAL[0]} <{LOCAL[1]}>"
    last = last_publish_run(store)
    assert last["status"] == "ok" and last["kind"] == "scheduled" and "pushed to origin/main" in last["message"]
    git(actions, "pull", "-q", "origin", "main")
    assert git(actions, "log", "-1", "--format=%ae").stdout.strip() == LOCAL[1]
    # the stamp (computed_at_utc, one-second resolution) is the only change between two publishes of the same data
    time.sleep(1.1)
    again = market_publish(store, laptop / "published", market, laptop, kind="manual", log=_Quiet())
    assert again["status"] == "ok" and last_publish_run(store)["kind"] == "manual"
    # skipped: no traded prices
    skipped = market_publish(store, laptop / "published", tmp_path / "absent.csv", laptop, log=_Quiet())
    assert skipped["status"] == "skipped" and last_publish_run(store)["status"] == "skipped"
    # failed: the push is rejected twice; the failure is recorded, not raised
    time.sleep(1.1)

    def reject(cmd, **kw):
        if cmd[1] == "push":  # the other publisher lands a commit right before every push attempt
            git(actions, "pull", "-q", "--rebase", "origin", "main")
            (actions / "published" / "status.json").write_text(str(pd.Timestamp.now().value))
            git(actions, "commit", "-q", "-am", "Publish")
            git(actions, "push", "-q", "origin", "HEAD:main")
        return subprocess.run(cmd, **kw)

    failed = market_publish(store, laptop / "published", market, laptop, kind="scheduled", log=_Quiet(), runner=reject)
    assert failed["status"] == "failed" and "rejected twice" in failed["message"]
    last = last_publish_run(store)
    assert last["status"] == "failed" and "rejected twice" in last["message"]
    runs = store.runs()
    assert runs["command"].eq("market-publish").sum() == 4 and runs["finished_at_utc"].notna().all()
    store.close()


class _Quiet:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
