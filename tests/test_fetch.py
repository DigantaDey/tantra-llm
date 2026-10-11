"""Model download: resumable fetch, verification, and the fetch_model.py command line.

Every test runs against a local HTTP server (with Range support) so nothing touches
the network. The pinned hashes in :mod:`tantra.fetch` were taken from the Hugging Face
tree API and are asserted here so a silent edit of the table cannot go unnoticed.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tantra.cli import fetch as fetch_cli
from tantra.fetch import (
    DEFAULT_MODEL_KEY,
    MODELS,
    FetchError,
    ModelSpec,
    download_file,
    ensure_model,
    gguf_magic_ok,
    remote_file_info,
    sha256_of,
)

PAYLOAD = b"GGUF" + b"tantra-test-payload-" * 64
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeHub:
    """Tiny Hugging Face stand-in: a resolve endpoint (optionally with Range) and a tree API."""

    def __init__(
        self,
        payload: bytes,
        *,
        honor_range: bool = True,
        tree: dict | None = None,
        truncate_after: int | None = None,
    ):
        self.payload = payload
        self.honor_range = honor_range
        self.tree = tree
        self.truncate_after = truncate_after
        self.requests: list[str] = []
        self.range_headers: list[str | None] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests.append(self.path)
                outer.range_headers.append(self.headers.get("Range"))
                if self.path.endswith("/tree/main"):
                    body = json.dumps(outer.tree or []).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if not self.path.endswith("/model.gguf"):
                    self.send_error(404)
                    return
                if outer.truncate_after is not None:
                    cut, outer.truncate_after = outer.truncate_after, None
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(outer.payload)))
                    self.end_headers()
                    self.wfile.write(outer.payload[:cut])
                    self.wfile.flush()
                    self.connection.close()
                    return
                start = 0
                rng = self.headers.get("Range")
                if rng and outer.honor_range:
                    start = int(rng.removeprefix("bytes=").split("-")[0])
                body = outer.payload[start:]
                self.send_response(206 if start else 200)
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(outer.payload) - 1}/{len(outer.payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def hub():
    servers: list[FakeHub] = []

    def make(**kwargs) -> FakeHub:
        server = FakeHub(**kwargs)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()


def make_spec(hub: FakeHub, *, sha: str = PAYLOAD_SHA, size: int | None = None) -> ModelSpec:
    return ModelSpec(
        key="test",
        repo="tantra/test",
        filename="model.gguf",
        size_bytes=len(PAYLOAD) if size is None else size,
        sha256=sha,
        label="test model",
        base_url=hub.base_url,
    )


# ----------------------------------------------------------------- pinned facts
def test_pinned_models_are_real_qwen_builds():
    # Guards the table against typos: the pins were verified against the Hugging Face tree API.
    assert set(MODELS) == {"standard", "small"}
    standard, small = MODELS["standard"], MODELS["small"]
    assert standard.repo == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    assert standard.filename == "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    assert standard.size_bytes == 1_117_320_736
    assert standard.sha256 == "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
    assert small.repo == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert small.filename == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert small.size_bytes == 491_400_032
    assert small.sha256 == "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
    for spec in MODELS.values():
        assert spec.url.endswith(f"{spec.repo}/resolve/main/{spec.filename}")
        assert len(spec.sha256) == 64
    assert DEFAULT_MODEL_KEY == "standard"


# ----------------------------------------------------------------- download
def test_full_download_is_verified_and_renamed(hub, tmp_path):
    server = hub(payload=PAYLOAD)
    spec = make_spec(server)
    seen: list[tuple[int, int | None]] = []
    dest = download_file(
        spec.url,
        tmp_path / spec.filename,
        expected_size=spec.size_bytes,
        expected_sha256=spec.sha256,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / (spec.filename + ".part")).exists()
    assert seen and seen[-1] == (len(PAYLOAD), len(PAYLOAD))


def test_interrupted_download_resumes_with_a_range_request(hub, tmp_path):
    server = hub(payload=PAYLOAD)
    spec = make_spec(server)
    part = tmp_path / (spec.filename + ".part")
    part.write_bytes(PAYLOAD[:100])
    dest = download_file(spec.url, tmp_path / spec.filename, expected_size=spec.size_bytes, expected_sha256=spec.sha256)
    assert dest.read_bytes() == PAYLOAD
    assert server.range_headers == ["bytes=100-"]


def test_server_that_ignores_range_still_yields_a_good_file(hub, tmp_path):
    server = hub(payload=PAYLOAD, honor_range=False)
    spec = make_spec(server)
    (tmp_path / (spec.filename + ".part")).write_bytes(PAYLOAD[:100])
    dest = download_file(spec.url, tmp_path / spec.filename, expected_size=spec.size_bytes, expected_sha256=spec.sha256)
    assert dest.read_bytes() == PAYLOAD


def test_complete_part_file_is_not_downloaded_again(hub, tmp_path):
    server = hub(payload=PAYLOAD)
    spec = make_spec(server)
    (tmp_path / (spec.filename + ".part")).write_bytes(PAYLOAD)
    dest = download_file(spec.url, tmp_path / spec.filename, expected_size=spec.size_bytes, expected_sha256=spec.sha256)
    assert dest.exists() and dest.read_bytes() == PAYLOAD


def test_corrupt_download_is_deleted_and_reported(hub, tmp_path):
    tampered = PAYLOAD[:-1] + b"?"  # same size and GGUF magic, wrong hash
    server = hub(payload=tampered)
    spec = make_spec(server)
    with pytest.raises(FetchError, match="sha256 mismatch"):
        download_file(spec.url, tmp_path / spec.filename, expected_size=spec.size_bytes, expected_sha256=spec.sha256)
    assert not (tmp_path / spec.filename).exists()
    assert not (tmp_path / (spec.filename + ".part")).exists()  # poisoned cache removed


def test_size_mismatch_is_reported(hub, tmp_path):
    server = hub(payload=PAYLOAD)
    spec = make_spec(server, size=len(PAYLOAD) + 5)
    with pytest.raises(FetchError, match="incomplete download"):
        download_file(spec.url, tmp_path / spec.filename, expected_size=spec.size_bytes, expected_sha256=spec.sha256)


def test_truncated_transfer_keeps_the_partial_and_resumes(hub, tmp_path):
    server = hub(payload=PAYLOAD, truncate_after=100)
    spec = make_spec(server)
    dest = tmp_path / spec.filename
    with pytest.raises(FetchError, match="rerun to resume"):
        download_file(spec.url, dest, expected_size=spec.size_bytes, expected_sha256=spec.sha256)
    part = tmp_path / (spec.filename + ".part")
    assert not dest.exists()
    assert part.read_bytes() == PAYLOAD[:100]  # kept for resume
    out = download_file(spec.url, dest, expected_size=spec.size_bytes, expected_sha256=spec.sha256)
    assert out.read_bytes() == PAYLOAD  # second attempt resumes from byte 100
    assert server.range_headers[-1] == "bytes=100-"


def test_non_gguf_payload_is_rejected(hub, tmp_path):
    server = hub(payload=b"XXXX" + PAYLOAD[4:])
    spec = make_spec(server, sha=hashlib.sha256(b"XXXX" + PAYLOAD[4:]).hexdigest())
    with pytest.raises(FetchError, match="GGUF magic"):
        download_file(spec.url, tmp_path / spec.filename, expected_size=spec.size_bytes, expected_sha256=spec.sha256)


def test_missing_file_is_a_clear_error(hub, tmp_path):
    server = hub(payload=PAYLOAD)
    spec = make_spec(server)
    missing = ModelSpec(
        key=spec.key,
        repo=spec.repo,
        filename="absent.gguf",
        size_bytes=spec.size_bytes,
        sha256=spec.sha256,
        label=spec.label,
        base_url=spec.base_url,
    )
    with pytest.raises(FetchError, match="not found"):
        download_file(missing.url, tmp_path / missing.filename, expected_size=1, expected_sha256=missing.sha256)


def test_existing_verified_file_short_circuits(hub, tmp_path):
    server = hub(payload=PAYLOAD)
    spec = make_spec(server)
    dest = tmp_path / spec.filename
    dest.write_bytes(PAYLOAD)
    out = download_file(
        spec.url, dest, expected_size=spec.size_bytes, expected_sha256=spec.sha256, on_progress=lambda d, t: None
    )
    assert out == dest
    assert server.requests == []  # not a single HTTP request


def test_existing_corrupt_file_is_replaced(hub, tmp_path):
    server = hub(payload=PAYLOAD)
    spec = make_spec(server)
    dest = tmp_path / spec.filename
    dest.write_bytes(b"GGUF" + b"\x00" * (len(PAYLOAD) - 4))
    out = download_file(spec.url, dest, expected_size=spec.size_bytes, expected_sha256=spec.sha256)
    assert out.read_bytes() == PAYLOAD


# ----------------------------------------------------------------- ensure_model / API
def test_remote_metadata_is_used_when_available(hub, tmp_path):
    tree = [
        {"type": "file", "path": "other.bin", "size": 1},
        {"type": "file", "path": "model.gguf", "size": len(PAYLOAD), "lfs": {"oid": PAYLOAD_SHA, "size": len(PAYLOAD)}},
    ]
    server = hub(payload=PAYLOAD, tree=tree)
    spec = make_spec(server, sha="0" * 64)  # the pin is stale; the API knows the real hash
    info = remote_file_info(spec.repo, spec.filename, base_url=server.base_url)
    assert info == (len(PAYLOAD), PAYLOAD_SHA)
    dest = ensure_model(spec, tmp_path)
    assert dest.exists() and gguf_magic_ok(dest)
    assert sha256_of(dest) == PAYLOAD_SHA


def test_unreachable_api_falls_back_to_pinned_values(hub, tmp_path):
    server = hub(payload=PAYLOAD)  # tree endpoint returns [] -> no metadata
    spec = make_spec(server)
    dest = ensure_model(spec, tmp_path)
    assert dest.exists() and dest.read_bytes() == PAYLOAD


# ----------------------------------------------------------------- command line
def test_cli_lists_models(capsys):
    assert fetch_cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "standard" in out and "small" in out


def test_cli_downloads_and_reports_next_steps(hub, tmp_path, monkeypatch, capsys):
    server = hub(payload=PAYLOAD)
    spec = make_spec(server)
    monkeypatch.setitem(fetch_cli.MODELS, "test", spec)
    assert fetch_cli.main(["--size", "test", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert (tmp_path / spec.filename).read_bytes() == PAYLOAD
    assert "sha256 verified" in out and "python app.py" in out


def test_cli_reports_a_download_failure(hub, tmp_path, monkeypatch, capsys):
    server = hub(payload=b"")
    spec = make_spec(server, size=len(PAYLOAD))
    monkeypatch.setitem(fetch_cli.MODELS, "test", spec)
    assert fetch_cli.main(["--size", "test", "--out-dir", str(tmp_path)]) == 1
    assert "size mismatch" in capsys.readouterr().out
