"""Polite HTTP helpers: a generic User-Agent, timeouts, retries with backoff, resumable files.

Only the standard library is used. No personal detail is ever sent: the User-Agent names the
software, as the Poly Haven API terms ask, and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "lumina-hdri-fetch/0.1"
TIMEOUT_S = 60.0
RETRIES = 5
BACKOFF_S = 2.0


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _retry_delay(attempt: int, err: Exception) -> float:
    """Exponential backoff; honour Retry-After on 429/503 when the server sends one."""
    if isinstance(err, urllib.error.HTTPError) and err.code in (429, 503):
        after = err.headers.get("Retry-After") if err.headers else None
        if after and after.isdigit():
            return float(after)
    return BACKOFF_S * (2**attempt)


def _is_retryable(err: Exception) -> bool:
    if isinstance(err, urllib.error.HTTPError):
        return err.code in (408, 425, 429, 500, 502, 503, 504)
    return isinstance(err, (urllib.error.URLError, TimeoutError, ConnectionError))


def get_json(url: str) -> Any:
    """GET a JSON document with retries."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(_request(url), timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as err:
            last = err
            if not _is_retryable(err) or attempt == RETRIES - 1:
                break
            time.sleep(_retry_delay(attempt, err))
    raise RuntimeError(f"GET {url} failed after retries: {last}") from last


def md5_of(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, md5: str | None = None, size: int | None = None) -> bool:
    """Download url to dest unless a complete copy is already there. Returns True if fetched.

    Writes to dest.part first and renames on success, so an interrupted run never leaves a
    truncated file under the final name. A file with the expected md5 (or size) is skipped.
    """
    if dest.exists() and _is_complete(dest, md5, size):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            _stream(url, part)
            if not _is_complete(part, md5, size):
                raise OSError(f"checksum or size mismatch for {url}")
            part.replace(dest)
            return True
        except Exception as err:
            last = err
            retryable = _is_retryable(err) or isinstance(err, OSError)
            if not retryable or attempt == RETRIES - 1:
                break
            time.sleep(_retry_delay(attempt, err))
    part.unlink(missing_ok=True)
    raise RuntimeError(f"download {url} failed after retries: {last}") from last


def _stream(url: str, path: Path) -> None:
    with urllib.request.urlopen(_request(url), timeout=TIMEOUT_S) as resp, path.open("wb") as fh:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            fh.write(chunk)


def _is_complete(path: Path, md5: str | None, size: int | None) -> bool:
    if size is not None and path.stat().st_size != size:
        return False
    if md5 is not None:
        return md5_of(path) == md5
    return size is not None or path.stat().st_size > 0
