"""Generic streaming file download with progress reporting.

Used to pre-populate large model-weight caches (e.g. CLIP) with visible
progress, since the underlying libraries (torch/open_clip) typically
either download silently or only print a console progress bar that never
reaches the web UI.

Kept dependency-free (stdlib `urllib` only) so it works regardless of
which optional extras are installed.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path

DownloadProgressCallback = Callable[[int, int], None]  # (bytes_downloaded, total_bytes)

_DEFAULT_CHUNK_SIZE = 256 * 1024  # 256 KiB


def is_already_downloaded(dest_path: str | Path) -> bool:
    """True if `dest_path` exists and is non-empty. A zero-byte file is
    treated as "not downloaded" -- it's what a failed/interrupted write
    would look like if something bypassed the temp-file-then-rename
    safety below."""
    p = Path(dest_path)
    return p.exists() and p.stat().st_size > 0


def download_with_progress(
    url: str,
    dest_path: str | Path,
    *,
    progress_cb: DownloadProgressCallback | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    timeout: float = 30.0,
) -> Path:
    """Download `url` to `dest_path`, streaming in chunks and reporting
    progress via `progress_cb(bytes_downloaded, total_bytes)`.

    `total_bytes` is 0 if the server didn't send a Content-Length header
    (progress can still be shown as a running byte count in that case).

    Downloads to a sibling '<name>.part' file first and only renames to
    the final destination on success, so a partial/failed download can
    never masquerade as a complete, valid cache entry.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        progress_cb(downloaded, total)
    except BaseException:
        # never leave a half-written file where a caller might mistake it
        # for a complete, valid cache entry
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(dest_path)
    return dest_path


def throttled_progress_cb(
    inner_cb: DownloadProgressCallback,
    *,
    min_percent_step: int = 1,
) -> DownloadProgressCallback:
    """Wrap a progress callback so it only fires when the completion
    percentage has advanced by at least `min_percent_step`, instead of on
    every chunk. Large downloads can produce thousands of chunks; calling
    a UI-facing callback on every single one adds needless overhead for
    no visible benefit.

    If total is unknown (0), every call is forwarded, since there's no
    percentage to throttle by -- the caller likely wants a running byte
    count instead.
    """
    last_percent = -1

    def _cb(downloaded: int, total: int) -> None:
        nonlocal last_percent
        if total <= 0:
            inner_cb(downloaded, total)
            return
        percent = min(100, (downloaded * 100) // total)
        is_final = downloaded >= total
        if is_final or percent - last_percent >= min_percent_step:
            last_percent = percent
            inner_cb(downloaded, total)

    return _cb
