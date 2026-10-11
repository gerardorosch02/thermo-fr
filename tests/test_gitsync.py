"""The laptop's one-file push: commits only the given path, rebases onto what another machine pushed first, retries once, never forces."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from thermo_fr.forecast.gitsync import GitSyncError, commit_and_push, local_identity

LOCAL = ("Laptop Publisher", "100+laptop@users.noreply.example.com")  # the repository's own identity (made up)


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True)


def make_repos(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    laptop, actions = tmp_path / "laptop", tmp_path / "actions"
    for clone in (laptop, actions):
        subprocess.run(["git", "clone", "-q", str(remote), str(clone)], check=True)
        git(clone, "config", "user.email", LOCAL[1])
        git(clone, "config", "user.name", LOCAL[0])
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
    assert (again["committed"], again["pushed"], again["attempts"]) == (False, False, 0)


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


def test_identity_comes_from_the_local_config_even_when_global_config_and_environment_say_otherwise(tmp_path, monkeypatch):
    remote, laptop, actions = make_repos(tmp_path)
    # a global config and inherited environment variables carrying a personal address: neither may reach the commit
    leaky = tmp_path / "gitconfig"
    leaky.write_text("[user]\n\tname = Personal Name\n\temail = personal@example.com\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(leaky))
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "inherited@example.com")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "inherited@example.com")
    assert local_identity(laptop) == LOCAL
    (laptop / "published" / "market.json").write_text('{"scored_days": 4}')
    seen = []

    def runner(cmd, **kw):
        seen.append((cmd, kw.get("env") or {}))
        return subprocess.run(cmd, **kw)

    result = commit_and_push(laptop, [Path("published/market.json")], "Market aggregates", runner=runner, log=lambda *_: None)
    assert result["pushed"] and result["identity"] == f"{LOCAL[0]} <{LOCAL[1]}>"
    git(actions, "pull", "-q", "origin", "main")
    stamp = git(actions, "log", "-1", "--format=%an|%ae|%cn|%ce").stdout.strip()
    assert stamp == f"{LOCAL[0]}|{LOCAL[1]}|{LOCAL[0]}|{LOCAL[1]}"
    for cmd, env in seen:  # every git call runs without the inherited identity and with prompts disabled; the commit with the local one
        assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GCM_INTERACTIVE"] == "never"
        assert "inherited@example.com" not in env.values()
        if cmd[1] != "config":
            assert env["GIT_COMMITTER_EMAIL"] == LOCAL[1] and env["GIT_AUTHOR_EMAIL"] == LOCAL[1]
    assert "personal@example.com" not in git(actions, "log", "--format=%ae %ce").stdout


def test_missing_local_identity_is_refused_before_anything_is_committed(tmp_path):
    remote, laptop, actions = make_repos(tmp_path)
    git(laptop, "config", "--unset", "user.email")
    (laptop / "published" / "market.json").write_text('{"scored_days": 5}')
    head = git(laptop, "rev-parse", "HEAD").stdout
    with pytest.raises(GitSyncError, match="user.email is not set in the repository's local git config"):
        commit_and_push(laptop, [Path("published/market.json")], "Market aggregates", log=lambda *_: None)
    assert git(laptop, "rev-parse", "HEAD").stdout == head
    assert git(laptop, "diff", "--cached", "--name-only").stdout.strip() == ""  # nothing left staged either


def test_data_commit_skips_the_pre_commit_hook(tmp_path):
    remote, laptop, actions = make_repos(tmp_path)
    hooks = Path(git(laptop, "rev-parse", "--git-path", "hooks").stdout.strip())
    hooks = hooks if hooks.is_absolute() else laptop / hooks
    hooks.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "hook_ran"
    hook = hooks / "pre-commit"
    hook.write_text(f"#!/bin/sh\necho ran > '{marker.as_posix()}'\nexit 1\n", newline="\n")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # a code commit is still blocked by the hook
    (laptop / "code.py").write_text("x = 1\n")
    git(laptop, "add", "code.py")
    blocked = subprocess.run(["git", "commit", "-q", "-m", "code"], cwd=str(laptop), capture_output=True, text=True)
    assert blocked.returncode != 0 and marker.exists()
    marker.unlink()
    git(laptop, "reset", "-q", "code.py")
    # the one-file data commit is not
    (laptop / "published" / "market.json").write_text('{"scored_days": 6}')
    result = commit_and_push(laptop, [Path("published/market.json")], "Market aggregates", log=lambda *_: None)
    assert result["committed"] and result["pushed"] and not marker.exists()
    assert os.path.exists(laptop / "code.py") and "code.py" not in git(laptop, "show", "--stat", "HEAD").stdout
