"""The laptop's one-file push: commits only the given path, rebases onto what another machine pushed first, retries once, never forces."""

import subprocess
from pathlib import Path

import pytest

from thermo_fr.forecast.gitsync import GitSyncError, commit_and_push


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True)


def make_repos(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    laptop, actions = tmp_path / "laptop", tmp_path / "actions"
    for clone in (laptop, actions):
        subprocess.run(["git", "clone", "-q", str(remote), str(clone)], check=True)
        git(clone, "config", "user.email", "t@example.com")
        git(clone, "config", "user.name", "test")
    (laptop / "published").mkdir()
    (laptop / "published" / "status.json").write_text("{}")
    (laptop / "published" / "market.json").write_text('{"scored_days": 1}')
    git(laptop, "add", "published")
    git(laptop, "commit", "-q", "-m", "seed")
    git(laptop, "push", "-q", "origin", "HEAD:main")
    git(actions, "pull", "-q", "origin", "main")
    return remote, laptop, actions


def test_push_rebases_onto_the_other_publisher_and_stages_only_its_file(tmp_path):
    remote, laptop, actions = make_repos(tmp_path)
    # the workflow pushes a new status.json first
    (actions / "published" / "status.json").write_text('{"written": "actions"}')
    git(actions, "commit", "-q", "-am", "Publish settle")
    git(actions, "push", "-q", "origin", "HEAD:main")
    # the laptop changes market.json and, separately, has an unrelated local edit that must not be committed
    (laptop / "published" / "market.json").write_text('{"scored_days": 2}')
    (laptop / "notes.txt").write_text("local only")
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return subprocess.run(cmd, **kw)

    result = commit_and_push(laptop, [Path("published/market.json")], "Market aggregates", runner=runner, log=lambda *_: None)
    assert result["committed"] and result["pushed"]
    assert all("--force" not in c and "-f" not in c[2:3] for c in calls)
    log = git(actions, "pull", "-q", "origin", "main") and git(actions, "log", "--oneline").stdout
    assert "Market aggregates" in log and "Publish settle" in log
    assert (actions / "published" / "status.json").read_text() == '{"written": "actions"}'
    assert (actions / "published" / "market.json").read_text() == '{"scored_days": 2}'
    assert not (actions / "notes.txt").exists()  # the unrelated file was never committed
    again = commit_and_push(laptop, [Path("published/market.json")], "Market aggregates", runner=runner, log=lambda *_: None)
    assert again == {"committed": False, "pushed": False, "attempts": 0}


def test_push_retries_once_when_rejected_then_gives_up(tmp_path):
    remote, laptop, actions = make_repos(tmp_path)
    (laptop / "published" / "market.json").write_text('{"scored_days": 3}')
    state = {"pushes": 0}

    def runner(cmd, **kw):
        if cmd[1] == "push":
            state["pushes"] += 1
            # another publisher lands a commit right before every push attempt
            (actions / "published" / "status.json").write_text(f'{{"n": {state["pushes"]}}}')
            git(actions, "commit", "-q", "-am", f"Publish {state['pushes']}")
            git(actions, "push", "-q", "origin", "HEAD:main")
        return subprocess.run(cmd, **kw)

    with pytest.raises(GitSyncError, match="rejected twice"):
        commit_and_push(laptop, [Path("published/market.json")], "Market aggregates", runner=runner, log=lambda *_: None)
    assert state["pushes"] == 2
