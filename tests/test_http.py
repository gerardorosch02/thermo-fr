import pytest
import requests

from conftest import FakeResponse, FakeSession
from thermo_fr.data.http import FileCache, HttpClient, HttpError, ServiceUnavailableError


def test_retries_on_503_then_succeeds(client_factory, sleeps):
    session = FakeSession(responses=[FakeResponse(503), FakeResponse(503), FakeResponse(200, {"ok": 1})])
    client = client_factory(session)
    assert client.get("https://x/y").json() == {"ok": 1}
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff between the attempts


def test_retries_on_429_and_honours_retry_after(client_factory, sleeps):
    session = FakeSession(responses=[FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {})])
    client_factory(session).get("https://x/y")
    assert sleeps == [7.0]


def test_gives_up_with_clear_message(client_factory, sleeps):
    session = FakeSession(responses=[FakeResponse(503)] * 4)
    with pytest.raises(ServiceUnavailableError, match="HTTP 503"):
        client_factory(session, retries=3).get("https://x/y")
    assert len(session.calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_backoff_is_capped(client_factory, sleeps):
    session = FakeSession(responses=[FakeResponse(503)] * 6)
    with pytest.raises(ServiceUnavailableError):
        client_factory(session, retries=5, max_wait=5.0).get("https://x/y")
    assert sleeps == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_non_transient_error_is_not_retried(client_factory, sleeps):
    session = FakeSession(responses=[FakeResponse(400, content=b"bad bzn")])
    with pytest.raises(HttpError, match="bad bzn"):
        client_factory(session).get("https://x/y")
    assert len(session.calls) == 1 and sleeps == []


def test_min_interval_spaces_requests(monkeypatch, sleeps):
    clock = iter([100.0, 110.0, 110.0])
    monkeypatch.setattr("thermo_fr.data.http.time.monotonic", lambda: next(clock))
    session = FakeSession(responses=[FakeResponse(200, {}), FakeResponse(200, {})])
    client = HttpClient(session=session, min_interval=30.0)
    client.get("https://x/1")
    client.get("https://x/2")
    assert sleeps == [20.0]  # 30 s interval minus the 10 s that already passed


def test_file_cache_round_trip(tmp_path):
    cache = FileCache(tmp_path / "c")
    calls = []

    def download():
        calls.append(1)
        return b"payload"

    assert cache.fetch("price_FR_2021-01-01.json", download) == b"payload"
    assert cache.fetch("price_FR_2021-01-01.json", download) == b"payload"
    assert len(calls) == 1
    assert (tmp_path / "c" / "price_FR_2021-01-01.json").read_bytes() == b"payload"


def test_file_cache_sanitises_keys(tmp_path):
    cache = FileCache(tmp_path)
    assert cache.path("a/b c?d=1").name == "a_b_c_d_1"


def test_timeouts_are_retried_like_connection_errors(client_factory, sleeps):
    from conftest import FakeResponse

    answers = iter([requests.exceptions.ReadTimeout("slow"), FakeResponse(200, payload={"ok": True})])

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, timeout=None, headers=None):
            self.calls += 1
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

    session = Session()
    client = client_factory(session)
    assert client.get("https://x.test/slow").json() == {"ok": True}
    assert session.calls == 2 and sleeps == [1.0]
