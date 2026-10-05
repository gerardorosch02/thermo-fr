"""Small HTTP helpers shared by the keyless data sources.

Two concerns live here so that every source behaves the same way:

- `HttpClient` spaces requests out to respect rate limits, retries on the
  transient statuses (429, 502, 503, 504) with exponential backoff, and gives
  up with a clear message when a service is down.
- `FileCache` keeps raw responses on disk under data/cache/ so that reruns do
  not download anything again.
"""

import re
import time
from pathlib import Path

import requests

RETRY_STATUSES = (429, 502, 503, 504)


class HttpError(RuntimeError):
    """A request failed for a reason that retrying will not fix."""


class ServiceUnavailableError(HttpError):
    """The service kept answering with a transient error until we ran out of retries."""


class HttpClient:
    """GET with a minimum interval between requests and exponential backoff.

    `min_interval` is the number of seconds to leave between two requests,
    which is how rate limits such as "2 requests per minute" are respected.
    `backoff` is the first wait after a transient failure; every further wait
    doubles, capped at `max_wait`. A Retry-After header, when present, wins.
    `retry_statuses` lists the HTTP statuses treated as transient; a source
    whose gateway answers with other codes (ENTSO-E uses 599) can extend it.
    """

    def __init__(
        self,
        session=None,
        min_interval: float = 0.0,
        retries: int = 5,
        backoff: float = 5.0,
        max_wait: float = 120.0,
        timeout: float = 60.0,
        user_agent: str = "thermo-fr/0.1 (research tool)",
        retry_statuses=RETRY_STATUSES,
    ):
        self.session = session or requests.Session()
        self.retry_statuses = tuple(retry_statuses)
        self.min_interval = min_interval
        self.retries = retries
        self.backoff = backoff
        self.max_wait = max_wait
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}
        self._last_request: float | None = None

    def _throttle(self) -> None:
        if self._last_request is not None and self.min_interval > 0:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

    def _wait_for_retry(self, attempt: int, response) -> float:
        header = response.headers.get("Retry-After") if response is not None else None
        if header and str(header).strip().isdigit():
            return min(float(header), self.max_wait)
        return min(self.backoff * (2**attempt), self.max_wait)

    def get(self, url: str, params: dict | None = None):
        """GET `url`, returning the response once it has a non-transient status."""
        last_status = None
        for attempt in range(self.retries + 1):
            self._throttle()
            self._last_request = time.monotonic()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout, headers=self.headers)
            except requests.exceptions.ConnectionError as exc:
                response, last_status = None, f"connection error ({exc.__class__.__name__})"
            else:
                last_status = f"HTTP {response.status_code}"
                if response.status_code < 400:
                    return response
                if response.status_code not in self.retry_statuses:
                    raise HttpError(f"{last_status} from {url}: {response.text[:300]}")
            if attempt < self.retries:
                time.sleep(self._wait_for_retry(attempt, response))
        raise ServiceUnavailableError(
            f"{url} still failing ({last_status}) after {self.retries + 1} attempts. "
            "The service looks down or is rate limiting us; try again later."
        )


class FileCache:
    """Raw responses stored as files, named after the request they came from."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", key)
        return self.directory / safe

    def get(self, key: str) -> bytes | None:
        path = self.path(key)
        return path.read_bytes() if path.exists() else None

    def put(self, key: str, content: bytes) -> Path:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def fetch(self, key: str, download) -> bytes:
        """Return the cached bytes for `key`, calling `download()` on a miss."""
        cached = self.get(key)
        if cached is not None:
            return cached
        content = download()
        self.put(key, content)
        return content
