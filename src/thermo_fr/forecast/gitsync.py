"""Commit one or a few files and push them, rebasing onto whatever another machine pushed first. Never a force push.

Two publishers write to the repository: the GitHub Actions workflow (the public dataset under published/, every weekday morning and
every afternoon) and the laptop (published/market.json, after its settle). They touch different files, so a rebase of one onto the
other never conflicts; this module makes the laptop side safe anyway: it stages only the paths it is given, commits them, pulls with
rebase (autostash keeps unrelated local edits out of the way), pushes, and on a rejected push fetches and rebases once more before
giving up. There is no --force anywhere, and a rebase that stops is aborted so the working tree is left as it was.
"""

import subprocess
from pathlib import Path


class GitSyncError(RuntimeError):
    """A git step failed in a way the module does not retry."""


def _git(repo, *args, check=True, runner=subprocess.run):
    result = runner(["git", *args], cwd=str(repo), capture_output=True, text=True)
    if check and result.returncode != 0:
        raise GitSyncError(f"git {' '.join(args)}: {result.stderr.strip() or result.stdout.strip()}")
    return result


def commit_and_push(repo, paths, message: str, remote: str = "origin", branch: str = "main", runner=subprocess.run, log=print) -> dict:
    """Stage `paths`, commit them if they changed, rebase onto the remote branch, push; retry the rebase and push once."""
    repo = Path(repo)
    paths = [str(p) for p in paths]
    _git(repo, "add", "--", *paths, runner=runner)
    staged = _git(repo, "diff", "--cached", "--quiet", "--", *paths, check=False, runner=runner)
    if staged.returncode == 0:
        log("nothing to commit")
        return {"committed": False, "pushed": False, "attempts": 0}
    _git(repo, "commit", "-m", message, "--", *paths, runner=runner)
    log(f"committed {', '.join(paths)}")
    attempts = 0
    while True:
        attempts += 1
        pulled = _git(repo, "pull", "--rebase", "--autostash", remote, branch, check=False, runner=runner)
        if pulled.returncode != 0:
            _git(repo, "rebase", "--abort", check=False, runner=runner)
            if attempts >= 2:
                raise GitSyncError(f"rebase onto {remote}/{branch} failed twice: {pulled.stderr.strip()}")
            log("rebase stopped; fetching and retrying once")
            _git(repo, "fetch", remote, branch, runner=runner)
            continue
        pushed = _git(repo, "push", remote, f"HEAD:{branch}", check=False, runner=runner)
        if pushed.returncode == 0:
            log(f"pushed to {remote}/{branch} on attempt {attempts}")
            return {"committed": True, "pushed": True, "attempts": attempts}
        if attempts >= 2:
            raise GitSyncError(f"push to {remote}/{branch} rejected twice: {pushed.stderr.strip()}")
        log("push rejected (someone pushed first); rebasing and retrying once")
