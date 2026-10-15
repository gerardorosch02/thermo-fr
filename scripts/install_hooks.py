"""Install the local git pre-commit hook that runs the test suite and blocks the commit on any failure.

Usage: python scripts/install_hooks.py

Local only: the hook lives in .git/hooks, which git never commits, and nothing in the GitHub Actions workflow refers to it. The hook
runs `python -m pytest -q` from the repository root with the interpreter that ran this installer; a non-zero exit blocks the commit. To
commit without the suite in an emergency, git's own --no-verify still works, and that choice shows in the terminal. The one automated
commit made on this machine, `thermo-fr market publish` (published/market.json, a generated data file), uses --no-verify by design:
the hook is for code commits.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

HOOK = """#!/bin/sh
# Installed by scripts/install_hooks.py: run the test suite before every commit and block it on any failure.
cd "$(git rev-parse --show-toplevel)" || exit 1
# git exports GIT_DIR, GIT_INDEX_FILE and others to hooks; tests that create temporary repositories must not inherit them,
# or their git commands act on this repository. PYTHONPATH=src tests this checkout's code, also from a linked worktree.
for name in $(env | grep -o '^GIT_[A-Za-z0-9_]*'); do unset "$name"; done
echo "pre-commit: running the test suite ..."
PYTHONPATH=src "{python}" -m pytest -q
status=$?
if [ $status -ne 0 ]; then
  echo "pre-commit: tests failed (exit $status); commit blocked"
  exit $status
fi
exit 0
"""


def install(repo: Path, python: str | None = None) -> Path:
    git_dir = Path(subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    hooks = git_dir / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    path = hooks / "pre-commit"
    interpreter = (python or sys.executable).replace("\\", "/")
    path.write_text(HOOK.format(python=interpreter), encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    written = install(root)
    print(f"pre-commit hook installed at {written} (runs python -m pytest -q with {sys.executable}; blocks the commit on failure)")
