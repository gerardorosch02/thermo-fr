"""The pre-commit hook installer writes an executable hook into a repository's hooks folder; nothing refers to it in the workflow."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import install_hooks  # noqa: E402


def test_installer_writes_the_hook_into_a_fresh_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = install_hooks.install(tmp_path, python="/usr/bin/python3")
    assert path == tmp_path / ".git" / "hooks" / "pre-commit" and path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh") and "python3" in text and "-m pytest -q" in text and "commit blocked" in text
    assert "\r" not in text
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "forecast.yml").read_text(encoding="utf-8")
    assert "install_hooks" not in workflow and "pre-commit" not in workflow
