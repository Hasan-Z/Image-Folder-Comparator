from __future__ import annotations

import functools
import http.server
import socket
import threading

import pytest

from imagecompare.core.downloader import (
    download_with_progress,
    is_already_downloaded,
    throttled_progress_cb,
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def local_file_server(tmp_path):
    """Spin up a tiny local HTTP server serving files from a temp
    directory, so downloader tests run fully offline and deterministically."""
    serve_dir = tmp_path / "served"
    serve_dir.mkdir()
    port = _free_port()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def _url_for(filename: str) -> str:
        return f"http://127.0.0.1:{port}/{filename}"

    yield serve_dir, _url_for

    httpd.shutdown()
    httpd.server_close()


def test_download_writes_full_file_content(tmp_path, local_file_server):
    serve_dir, url_for = local_file_server
    data = b"x" * (256 * 1024 * 3 + 123)  # spans multiple chunks plus a remainder
    (serve_dir / "model.bin").write_bytes(data)

    dest = tmp_path / "out" / "model.bin"
    result = download_with_progress(url_for("model.bin"), dest)

    assert result == dest
    assert dest.read_bytes() == data


def test_download_reports_progress_with_correct_totals(tmp_path, local_file_server):
    serve_dir, url_for = local_file_server
    data = b"y" * (256 * 1024 * 2 + 500)
    (serve_dir / "model.bin").write_bytes(data)

    calls: list[tuple[int, int]] = []
    dest = tmp_path / "model.bin"
    download_with_progress(
        url_for("model.bin"), dest, progress_cb=lambda d, t: calls.append((d, t))
    )

    assert len(calls) > 0
    # progress is monotonically non-decreasing
    downloaded_values = [c[0] for c in calls]
    assert downloaded_values == sorted(downloaded_values)
    # final call reports the complete file
    assert calls[-1][0] == len(data)
    assert calls[-1][1] == len(data)


def test_download_creates_parent_directories(tmp_path, local_file_server):
    serve_dir, url_for = local_file_server
    (serve_dir / "model.bin").write_bytes(b"abc")

    dest = tmp_path / "a" / "b" / "c" / "model.bin"
    download_with_progress(url_for("model.bin"), dest)

    assert dest.exists()
    assert dest.read_bytes() == b"abc"


def test_download_does_not_leave_partial_file_on_success(tmp_path, local_file_server):
    serve_dir, url_for = local_file_server
    (serve_dir / "model.bin").write_bytes(b"abc")

    dest = tmp_path / "model.bin"
    download_with_progress(url_for("model.bin"), dest)

    part_file = dest.with_name(dest.name + ".part")
    assert not part_file.exists()


def test_download_cleans_up_partial_file_on_failure(tmp_path):
    dest = tmp_path / "model.bin"
    with pytest.raises(OSError):
        download_with_progress("http://127.0.0.1:1/does-not-exist", dest, timeout=1.0)

    part_file = dest.with_name(dest.name + ".part")
    assert not part_file.exists()
    assert not dest.exists()


def test_download_404_raises_and_leaves_no_file(tmp_path, local_file_server):
    _serve_dir, url_for = local_file_server
    dest = tmp_path / "model.bin"

    with pytest.raises(OSError):
        download_with_progress(url_for("does-not-exist.bin"), dest)

    assert not dest.exists()


def test_is_already_downloaded_false_when_missing(tmp_path):
    assert is_already_downloaded(tmp_path / "missing.bin") is False


def test_is_already_downloaded_true_for_nonempty_file(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"some data")
    assert is_already_downloaded(p) is True


def test_is_already_downloaded_false_for_empty_file(tmp_path):
    p = tmp_path / "empty.bin"
    p.touch()
    assert is_already_downloaded(p) is False


def test_throttled_progress_cb_limits_call_volume():
    calls: list[tuple[int, int]] = []
    cb = throttled_progress_cb(lambda d, t: calls.append((d, t)), min_percent_step=10)

    total = 1000
    for downloaded in range(0, total + 1, 1):  # simulate 1001 tiny chunk updates
        cb(downloaded, total)

    # should collapse to roughly one call per 10% step, not 1001 calls
    assert len(calls) <= 12
    assert calls[-1] == (total, total)  # always reports the final state


def test_throttled_progress_cb_forwards_every_call_when_total_unknown():
    calls: list[tuple[int, int]] = []
    cb = throttled_progress_cb(lambda d, t: calls.append((d, t)), min_percent_step=10)

    for downloaded in range(1, 6):
        cb(downloaded, 0)

    assert len(calls) == 5
