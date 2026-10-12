"""The laptop's market publish records its outcome in the store (ok, skipped, failed) so the dashboard can show it."""

import json
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


def test_publish_from_a_feature_branch_with_uncommitted_work_pushes_only_market_json_to_main(tmp_path):
    """The main working tree is on a feature branch with dirty and untracked files: the publish uses the worktree and leaves all of that alone."""
    remote, laptop, actions = make_repos(tmp_path)
    store, market = seeded_store_and_market(tmp_path, days=2)
    (laptop / "README.md").write_text("main version\n")
    git(laptop, "add", "README.md")
    git(laptop, "commit", "-q", "-m", "readme on main")
    git(laptop, "push", "-q", "origin", "HEAD:main")
    # a feature branch with a committed change, a dirty tracked file and an untracked file
    git(laptop, "checkout", "-q", "-b", "feature")
    (laptop / "feature.py").write_text("x = 1\n")
    git(laptop, "add", "feature.py")
    git(laptop, "commit", "-q", "-m", "feature work (must never reach main)")
    (laptop / "README.md").write_text("edited, uncommitted\n")
    (laptop / "scratch.txt").write_text("untracked\n")
    (laptop / "published" / "market.json").write_text('{"scored_days": 99}')  # a stale copy in the main tree: ignored by the publish
    feature_head = git(laptop, "rev-parse", "HEAD").stdout.strip()
    status_before = git(laptop, "status", "--porcelain").stdout

    result = market_publish(store, laptop / "published", market, laptop, kind="scheduled", log=_Quiet())
    assert result["status"] == "ok" and result["pushed"], result["message"]
    # main received exactly one new commit, touching market.json only, on top of the readme commit; the feature commit is not there
    git(actions, "pull", "-q", "origin", "main")
    log = git(actions, "log", "--format=%s", "main").stdout.splitlines()
    assert log[0].startswith("Market aggregates") and log[1] == "readme on main" and not any("feature work" in line for line in log)
    assert git(actions, "show", "--stat", "--format=", "HEAD").stdout.strip().splitlines()[0].split("|")[0].strip() == "published/market.json"
    assert not (actions / "feature.py").exists() and (actions / "README.md").read_text() == "main version\n"
    assert json.loads((actions / "published" / "market.json").read_text())["scored_days"] == 2
    # the laptop's main tree: same branch, same head, same dirty and untracked files, the stale copy still there
    assert git(laptop, "branch", "--show-current").stdout.strip() == "feature" and git(laptop, "rev-parse", "HEAD").stdout.strip() == feature_head
    assert git(laptop, "status", "--porcelain").stdout == status_before
    assert (laptop / "README.md").read_text() == "edited, uncommitted\n" and (laptop / "scratch.txt").exists()
    assert json.loads((laptop / "published" / "market.json").read_text())["scored_days"] == 99
    # the worktree exists, is excluded from status, and is reused by a second publish after another publisher moved main on
    assert (laptop / ".publish-worktree" / ".git").exists() and ".publish-worktree" not in status_before
    (actions / "published" / "status.json").write_text('{"written": "later"}')
    git(actions, "commit", "-q", "-am", "Publish settle")
    git(actions, "push", "-q", "origin", "HEAD:main")
    time.sleep(1.1)
    again = market_publish(store, laptop / "published", market, laptop, kind="scheduled", log=_Quiet())
    assert again["status"] == "ok" and again["pushed"]
    git(actions, "pull", "-q", "origin", "main")
    assert git(actions, "log", "--format=%s", "-3").stdout.splitlines()[:2] == [git(actions, "log", "--format=%s", "-1").stdout.strip(), "Publish settle"]
    assert (actions / "published" / "status.json").read_text() == '{"written": "later"}'
    assert git(laptop, "branch", "--show-current").stdout.strip() == "feature" and git(laptop, "status", "--porcelain").stdout == status_before
    store.close()


class _Quiet:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
