"""Shared test helpers: a fake HTTP session and a no-op sleep so tests run offline and fast."""

import json
import os

import pytest

from thermo_fr.data import http


@pytest.fixture(autouse=True)
def _no_inherited_git_environment(monkeypatch):
    """Drop GIT_* variables inherited from the caller. Inside a pre-commit hook git exports GIT_DIR, GIT_INDEX_FILE and
    friends; a test that runs `git init` or `git config` on a temporary repository would otherwise act on the real one."""
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name, raising=False)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content: bytes | None = None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        if content is None:
            content = json.dumps(payload).encode() if payload is not None else b""
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.content)


class FakeSession:
    """Answers GETs from a handler(url, params) -> FakeResponse, or from a queue of responses."""

    def __init__(self, handler=None, responses=None):
        self.handler = handler
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, dict(params or {})))
        if self.handler is not None:
            return self.handler(url, dict(params or {}))
        return self.responses.pop(0)


@pytest.fixture
def sleeps(monkeypatch):
    """Replace time.sleep in the HTTP helper with a recorder, and freeze the throttle clock."""
    recorded = []
    monkeypatch.setattr(http.time, "sleep", lambda seconds: recorded.append(seconds))
    return recorded


@pytest.fixture
def client_factory(sleeps):
    def make(session, **kwargs):
        kwargs.setdefault("backoff", 1.0)
        kwargs.setdefault("retries", 3)
        return http.HttpClient(session=session, **kwargs)

    return make
