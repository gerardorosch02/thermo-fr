"""Commit one or a few files and push them, rebasing onto whatever another machine pushed first. Never a force push.

Two publishers write to the repository: the GitHub Actions workflow (the public dataset under published/, every weekday morning and
every afternoon) and the laptop (published/market.json, after its settle). They touch different files, so a rebase of one onto the
other never conflicts; this module makes the laptop side safe anyway: it stages only the paths it is given, commits them, pulls with
rebase (autostash keeps unrelated local edits out of the way), pushes, and on a rejected push fetches and rebases once more before
giving up. There is no --force anywhere, and a rebase that stops is aborted so the working tree is left as it was.

The author and committer of the data commit are read from the repository's own (--local) git config and passed to every git call as
GIT_AUTHOR_* and GIT_COMMITTER_* variables, which override any other config level: a global or system setting can never put a
different name or email on the commit, and a repository without a local identity is refused before anything is committed. Every git
call also runs with terminal and credential-manager prompts disabled, so a scheduled run that cannot authenticate fails with a
message instead of hanging on a prompt nobody sees. The data commit is made with --no-verify: it changes one generated file, and the
local pre-commit hook (the test suite) is for code commits.
"""

import os
import subprocess
from pathlib import Path

NO_PROMPT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


class GitSyncError(RuntimeError):
    """A git step failed in a way the module does not retry."""


def _environment(env=None) -> dict:
    """The inherited environment without any GIT_AUTHOR_* or GIT_COMMITTER_* variable, prompts disabled, plus `env`."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_"))}
    return {**base, **NO_PROMPT_ENV, **(env or {})}


def _git(repo, *args, check=True, runner=subprocess.run, env=None):
    result = runner(["git", *args], cwd=str(repo), capture_output=True, text=True, env=_environment(env))
    if check and result.returncode != 0:
        raise GitSyncError(f"git {' '.join(args)}: {result.stderr.strip() or result.stdout.strip()}")
    return result


def local_identity(repo, runner=subprocess.run) -> tuple[str, str]:
    """The user.name and user.email of the repository's own config file; refused when either is missing there."""
    values = []
    for key in ("user.name", "user.email"):
        got = _git(Path(repo), "config", "--local", "--get", key, check=False, runner=runner)
        value = got.stdout.strip()
        if got.returncode != 0 or not value:
            raise GitSyncError(f"{key} is not set in the repository's local git config; refusing to commit with an identity from elsewhere "
                               f"(set it with: git config --local {key} ...)")
        values.append(value)
    return values[0], values[1]


def identity_env(name: str, email: str) -> dict:
    """Environment variables that fix the author and the committer, above every git config level."""
    return {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}


def fetch_file(repo, path: str, remote: str = "origin", branch: str = "main", runner=subprocess.run) -> str | None:
    """Fetch the remote branch and return the committed text of `path` on it (None when the branch has no such file).

    Reads the commit, not the working tree, so a locally regenerated copy of a public file can never pass as the published one.
    """
    repo = Path(repo)
    _git(repo, "fetch", remote, branch, runner=runner)
    shown = _git(repo, "show", f"{remote}/{branch}:{path}", check=False, runner=runner)
    if shown.returncode != 0:
        if "does not exist" in shown.stderr or "exists on disk, but not in" in shown.stderr or "invalid object name" in shown.stderr.lower():
            return None
        raise GitSyncError(f"git show {remote}/{branch}:{path}: {shown.stderr.strip()}")
    return shown.stdout


def commit_and_push(repo, paths, message: str, remote: str = "origin", branch: str = "main", runner=subprocess.run, log=print) -> dict:
    """Stage `paths`, commit them if they changed, rebase onto the remote branch, push; retry the rebase and push once.

    The identity comes from the repository's local config only (see local_identity); the commit skips the pre-commit hook.
    """
    repo = Path(repo)
    paths = [str(p) for p in paths]
    name, email = local_identity(repo, runner=runner)
    env = identity_env(name, email)
    _git(repo, "add", "--", *paths, runner=runner, env=env)
    staged = _git(repo, "diff", "--cached", "--quiet", "--", *paths, check=False, runner=runner, env=env)
    if staged.returncode == 0:
        log("nothing to commit")
        return {"committed": False, "pushed": False, "attempts": 0, "identity": f"{name} <{email}>"}
    _git(repo, "commit", "--no-verify", "-m", message, "--", *paths, runner=runner, env=env)
    log(f"committed {', '.join(paths)} as {name} <{email}> (data commit, hook skipped)")
    attempts = 0
    while True:
        attempts += 1
        pulled = _git(repo, "pull", "--rebase", "--autostash", remote, branch, check=False, runner=runner, env=env)
        if pulled.returncode != 0:
            _git(repo, "rebase", "--abort", check=False, runner=runner, env=env)
            if attempts >= 2:
                raise GitSyncError(f"rebase onto {remote}/{branch} failed twice: {pulled.stderr.strip()}")
            log("rebase stopped; fetching and retrying once")
            _git(repo, "fetch", remote, branch, runner=runner, env=env)
            continue
        pushed = _git(repo, "push", remote, f"HEAD:{branch}", check=False, runner=runner, env=env)
        if pushed.returncode == 0:
            log(f"pushed to {remote}/{branch} on attempt {attempts}")
            return {"committed": True, "pushed": True, "attempts": attempts, "identity": f"{name} <{email}>"}
        if attempts >= 2:
            raise GitSyncError(f"push to {remote}/{branch} rejected twice: {pushed.stderr.strip()}")
        log("push rejected (someone pushed first); rebasing and retrying once")
