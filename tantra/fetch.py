"""Download a ready-made GGUF chat model into ``models/gguf``.

The repository ships no weights (see README, "Data, licensing and rights").
This module fetches the official Qwen GGUF builds from the Hugging Face Hub
with resumable downloads and verifies every byte: exact size, SHA-256 and the
GGUF magic number before the file is put in place.

Network access is only needed here; everything after the download is offline.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CHUNK_BYTES = 1024 * 1024
USER_AGENT = "tantra-llm/0.1 (offline Bengali tantra toolkit)"
GGUF_MAGIC = b"GGUF"

# Progress callback: (bytes_done, total_bytes or None).
ProgressFn = Callable[[int, int | None], None]


class FetchError(RuntimeError):
    """Download or verification failed; the message is user-facing."""


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable GGUF file, pinned to an exact size and SHA-256."""

    key: str
    repo: str
    filename: str
    size_bytes: int
    sha256: str
    label: str
    base_url: str = "https://huggingface.co"

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.repo}/resolve/main/{self.filename}"


# Sizes and hashes verified against https://huggingface.co/api/models/<repo>/tree/main
# (the LFS oid is the SHA-256 of the file content) in October 2026.
MODELS: dict[str, ModelSpec] = {
    "standard": ModelSpec(
        key="standard",
        repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size_bytes=1_117_320_736,
        sha256="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        label="Qwen2.5-1.5B-Instruct · Q4_K_M · 1.04 GiB",
    ),
    "small": ModelSpec(
        key="small",
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size_bytes=491_400_032,
        sha256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        label="Qwen2.5-0.5B-Instruct · Q4_K_M · 468 MiB",
    ),
}
DEFAULT_MODEL_KEY = "standard"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def gguf_magic_ok(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(len(GGUF_MAGIC)) == GGUF_MAGIC


def _request(url: str, *, headers: dict[str, str]) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})


def _open(url: str, *, headers: dict[str, str], timeout: float):
    return urllib.request.urlopen(_request(url, headers=headers), timeout=timeout)


def remote_file_info(
    repo: str, filename: str, *, base_url: str = "https://huggingface.co", timeout: float = 15.0
) -> tuple[int, str] | None:
    """Best-effort (size, sha256) from the Hub tree API. Returns None when unreachable."""
    url = f"{base_url}/api/models/{repo}/tree/main"
    try:
        with _open(url, headers={}, timeout=timeout) as response:
            entries = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    for entry in entries:
        if entry.get("path") != filename:
            continue
        lfs = entry.get("lfs") or {}
        size = int(entry.get("size") or lfs.get("size") or 0)
        digest = lfs.get("oid", "")
        if size and len(digest) == 64:
            return size, digest.lower()
        return None
    return None


def _verify(path: Path, expected_size: int, expected_sha256: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise FetchError(f"size mismatch for {path.name}: expected {expected_size} bytes, got {actual_size}")
    if not gguf_magic_ok(path):
        raise FetchError(f"{path.name} does not start with the GGUF magic bytes")
    if expected_sha256:
        actual_sha = sha256_of(path)
        if actual_sha != expected_sha256:
            raise FetchError(
                f"sha256 mismatch for {path.name}:\n    expected {expected_sha256}\n    actual   {actual_sha}"
            )


def download_file(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    timeout: float = 60.0,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Download ``url`` to ``dest`` (resumable, verified, atomic).

    A partial download is kept as ``<dest>.part`` and resumed with an HTTP Range
    request. The file is renamed into place only after size, sha256 (when given)
    and the GGUF magic all pass.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if dest.is_file():
        stale = (expected_size is not None and dest.stat().st_size != expected_size) or (
            expected_sha256 is not None and sha256_of(dest) != expected_sha256
        )
        if stale:
            dest.unlink()
        else:
            if on_progress:
                on_progress(dest.stat().st_size, expected_size or dest.stat().st_size)
            return dest

    done = part.stat().st_size if part.is_file() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    try:
        response = _open(url, headers=headers, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected_size is not None and done == expected_size:
            response = None  # the .part file is already complete
        elif exc.code == 404:
            raise FetchError(f"not found on the server: {url}") from exc
        else:
            raise FetchError(f"download failed (HTTP {exc.code}) for {url}") from exc
    except OSError as exc:
        raise FetchError(f"cannot reach {url}: {exc}") from exc

    append = False
    if response is not None:
        with response:
            status = getattr(response, "status", 200)
            if status == 206:
                append = True
            elif done:
                done = 0  # server ignored the Range header: start over
            total = expected_size
            if total is None:
                length = response.headers.get("Content-Length")
                base = done if append and length else 0
                total = base + int(length) if length else None
            with part.open("ab" if append else "wb") as out:
                try:
                    while chunk := response.read(CHUNK_BYTES):
                        out.write(chunk)
                        done += len(chunk)
                        if on_progress:
                            on_progress(done, total)
                except (OSError, http.client.HTTPException) as exc:
                    # Keep the .part file: the next run resumes from where this one stopped.
                    raise FetchError(f"connection lost after {done:,} bytes ({exc}); rerun to resume") from exc

    if expected_size is None:
        raise FetchError(f"cannot verify {dest.name}: no expected size known")
    if part.is_file() and 0 < done < expected_size:
        # Short response or dropped connection: keep the bytes we have so the next run resumes.
        raise FetchError(f"incomplete download: got {done:,} of {expected_size:,} bytes; rerun to resume")
    try:
        _verify(part, expected_size, expected_sha256 or "")
    except FetchError:
        part.unlink(missing_ok=True)  # poisoned data must never be resumed into place
        raise
    os.replace(part, dest)
    return dest


def ensure_model(
    spec: ModelSpec,
    out_dir: Path,
    *,
    force: bool = False,
    timeout: float = 60.0,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Make sure ``spec`` is present in ``out_dir``; download and verify it if not."""
    dest = out_dir / spec.filename
    if force and dest.exists():
        dest.unlink()
    expected_size, expected_sha = spec.size_bytes, spec.sha256
    remote = remote_file_info(spec.repo, spec.filename, base_url=spec.base_url, timeout=min(timeout, 15.0))
    if remote is not None:
        expected_size, expected_sha = remote  # upstream knows best if the file was republished
    return download_file(
        spec.url,
        dest,
        expected_size=expected_size,
        expected_sha256=expected_sha,
        timeout=timeout,
        on_progress=on_progress,
    )
