"""In-memory background job manager.

Each comparison run gets a job_id and executes on a worker thread (image
comparison is CPU/IO bound, not async-friendly, so a plain thread is
simpler and more predictable than asyncio here). Progress is polled by
the frontend via GET /api/jobs/{id}.

ETA calculation: rather than naively extrapolating from the very first
tick (which is wildly inaccurate once Deep Rotation kicks in and
per-item cost jumps), we keep a rolling average of time-per-item over
the last N ticks and use that to project remaining time. This lets the
estimate self-correct as the job moves between cheap and expensive
phases.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from imagecompare.core.comparator import ComparisonCancelled, MatchCandidate, compare_folders
from imagecompare.core.scanner import scan_folder
from imagecompare.core.thumbnails import get_or_create_thumbnail
from imagecompare.models import ComparisonSummary, JobProgress, JobStatus, LogEntry, LogLevel

_ROLLING_WINDOW = 50
_PAUSE_POLL_SECONDS = 0.15
_LOG_MAXLEN = 500

logger = logging.getLogger("imagecompare.jobs")


def _resolve(folder: str) -> Path:
    return Path(folder).expanduser().resolve()


class JobLog:
    """A small ring buffer of timestamped log entries for a single job,
    surfaced to the UI via GET /api/jobs/{id}/logs. Also mirrors every
    entry to the standard `logging` module so server operators get the
    same visibility in stdout/log files without polling the API."""

    def __init__(self, job_id: str, maxlen: int = _LOG_MAXLEN) -> None:
        self._job_id = job_id
        self._entries: deque[LogEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, level: str, message: str) -> None:
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            level=LogLevel(level),
            message=message,
        )
        with self._lock:
            self._entries.append(entry)
        log_fn = {"info": logger.info, "warning": logger.warning, "error": logger.error}.get(
            level, logger.info
        )
        log_fn("[job %s] %s", self._job_id, message)

    def snapshot(self) -> list[LogEntry]:
        with self._lock:
            return list(self._entries)


class JobControl:
    """Shared pause/cancel signaling between the API layer (which sets
    these flags on request) and the worker thread (which checks them at
    every checkpoint())."""

    def __init__(self) -> None:
        self.pause_event = threading.Event()
        self.cancel_event = threading.Event()

    def checkpoint(self) -> None:
        if self.cancel_event.is_set():
            raise ComparisonCancelled()
        while self.pause_event.is_set():
            if self.cancel_event.is_set():
                raise ComparisonCancelled()
            time.sleep(_PAUSE_POLL_SECONDS)


@dataclass
class _JobState:
    job_id: str
    folder1: str = ""
    folder2: str = ""
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    processed_items: int = 0
    phase: str = ""
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    error_message: str | None = None
    results: list[MatchCandidate] = field(default_factory=list)
    summary: ComparisonSummary | None = None
    control: JobControl = field(default_factory=JobControl)
    log: JobLog = field(init=False)
    _tick_times: deque[float] = field(default_factory=lambda: deque(maxlen=_ROLLING_WINDOW))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.log = JobLog(self.job_id)

    def record_tick(self, processed: int, total: int, phase: str) -> None:
        with self._lock:
            self.processed_items = processed
            self.total_items = total
            self.phase = phase
            self._tick_times.append(time.monotonic())

    def snapshot(self) -> JobProgress:
        with self._lock:
            elapsed = (self.finished_at or time.monotonic()) - self.started_at
            percent = (
                min(100.0, (self.processed_items / self.total_items) * 100)
                if self.total_items
                else 0.0
            )
            eta = self._estimate_eta()
            # defensive clamp: processed should never outrun total, but if
            # a future phase mis-tracks total mid-stream, don't show
            # nonsense like "5486 / 3091" in the UI
            display_processed = (
                min(self.processed_items, self.total_items)
                if self.total_items
                else self.processed_items
            )
            return JobProgress(
                job_id=self.job_id,
                status=self.status,
                total_items=self.total_items,
                processed_items=display_processed,
                percent=round(percent, 1),
                elapsed_seconds=round(elapsed, 1),
                eta_seconds=round(eta, 1) if eta is not None else None,
                error_message=self.error_message,
                phase=self.phase,
            )

    def _estimate_eta(self) -> float | None:
        if self.status == JobStatus.DONE:
            return 0.0
        if self.status in (JobStatus.ERROR, JobStatus.CANCELLED, JobStatus.PAUSED):
            return None
        if len(self._tick_times) < 2 or not self.total_items:
            return None
        window = list(self._tick_times)
        span = window[-1] - window[0]
        ticks_in_window = len(window) - 1
        if span <= 0 or ticks_in_window <= 0:
            return None
        avg_seconds_per_item = span / ticks_in_window
        remaining_items = max(0, self.total_items - self.processed_items)
        return avg_seconds_per_item * remaining_items


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, _JobState] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        *,
        folder1: str,
        folder2: str,
        mode: str,
        recursive: bool,
        include_patterns: str,
        exclude_patterns: str,
        visual_threshold: float,
        filename_threshold: float,
        deep_rotation: bool,
        hash_size: int,
        thumb_cache_dir: str,
    ) -> str:
        job_id = uuid.uuid4().hex
        state = _JobState(
            job_id=job_id,
            folder1=str(_resolve(folder1)),
            folder2=str(_resolve(folder2)),
        )
        with self._lock:
            self._jobs[job_id] = state

        state.log.add(
            "info",
            f"Job created: '{state.folder1}' vs '{state.folder2}' "
            f"(mode={mode}, recursive={recursive}, deep_rotation={deep_rotation}, "
            f"visual_threshold={visual_threshold}%)",
        )

        thread = threading.Thread(
            target=self._run,
            args=(
                state,
                folder1,
                folder2,
                mode,
                recursive,
                include_patterns,
                exclude_patterns,
                visual_threshold,
                filename_threshold,
                deep_rotation,
                hash_size,
                thumb_cache_dir,
            ),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run(
        self,
        state: _JobState,
        folder1: str,
        folder2: str,
        mode: str,
        recursive: bool,
        include_patterns: str,
        exclude_patterns: str,
        visual_threshold: float,
        filename_threshold: float,
        deep_rotation: bool,
        hash_size: int,
        thumb_cache_dir: str,
    ) -> None:
        try:
            state.status = JobStatus.SCANNING
            state.record_tick(0, 1, "scanning folders")
            state.log.add("info", "Scanning folders...")
            state.control.checkpoint()
            images1 = scan_folder(
                folder1,
                recursive=recursive,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            images2 = scan_folder(
                folder2,
                recursive=recursive,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            state.log.add(
                "info",
                f"Scan complete: {len(images1)} image(s) in folder A, "
                f"{len(images2)} image(s) in folder B",
            )
            state.control.checkpoint()

            state.status = JobStatus.RUNNING

            def progress_cb(processed: int, total: int, phase: str) -> None:
                state.record_tick(processed, total, phase)

            def log_cb(level: str, message: str) -> None:
                state.log.add(level, message)

            matches = compare_folders(
                images1,
                images2,
                mode=mode,
                visual_threshold=visual_threshold,
                deep_rotation=deep_rotation,
                hash_size=hash_size,
                progress_cb=progress_cb,
                checkpoint_cb=state.control.checkpoint,
                log_cb=log_cb,
            )

            # Generate thumbnails for surviving matches only (not the full
            # scan) to keep this fast.
            if matches:
                state.log.add("info", f"Generating thumbnails for {len(matches)} match(es)...")
            for c in matches:
                state.control.checkpoint()
                get_or_create_thumbnail(c.image1.path, thumb_cache_dir)
                get_or_create_thumbnail(c.image2.path, thumb_cache_dir)

            state.results = matches
            elapsed = time.monotonic() - state.started_at

            scores = [m.visual_similarity for m in matches]
            exact_dup_count = sum(1 for m in matches if m.is_exact_duplicate)
            state.summary = ComparisonSummary(
                mode=mode,
                folder1=state.folder1,
                folder2=state.folder2,
                total_images_a=len(images1),
                total_images_b=len(images2),
                total_pairs_compared=len(images1) * len(images2),
                total_matches=len(matches),
                exact_duplicates=exact_dup_count,
                highest_similarity=max(scores) if scores else None,
                average_similarity=round(sum(scores) / len(scores), 2) if scores else None,
                lowest_similarity=min(scores) if scores else None,
                deep_rotation_used=deep_rotation,
                visual_threshold=visual_threshold,
                elapsed_seconds=round(elapsed, 1),
            )

            if scores:
                state.log.add(
                    "info",
                    f"Summary: {len(images1)} vs {len(images2)} image(s), "
                    f"{len(images1) * len(images2)} pair(s) compared, {len(matches)} match(es) "
                    f">= {visual_threshold}% ({exact_dup_count} exact duplicate(s)), similarity "
                    f"range {min(scores):.1f}-{max(scores):.1f}% "
                    f"(avg {sum(scores) / len(scores):.1f}%)",
                )
            else:
                state.log.add(
                    "info",
                    f"Summary: {len(images1)} vs {len(images2)} image(s), "
                    f"{len(images1) * len(images2)} pair(s) compared, 0 matches "
                    f">= {visual_threshold}%",
                )

            state.log.add("info", f"Job finished in {elapsed:.1f}s: {len(matches)} match(es)")
            state.status = JobStatus.DONE
            state.finished_at = time.monotonic()
        except ComparisonCancelled:
            state.log.add("warning", "Job stopped by user")
            state.status = JobStatus.CANCELLED
            state.finished_at = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            state.error_message = str(exc)
            state.log.add("error", f"Job failed: {exc}")
            state.status = JobStatus.ERROR
            state.finished_at = time.monotonic()

    def get_progress(self, job_id: str) -> JobProgress | None:
        state = self._jobs.get(job_id)
        return state.snapshot() if state else None

    def get_state(self, job_id: str) -> _JobState | None:
        return self._jobs.get(job_id)

    def pause_job(self, job_id: str) -> bool:
        state = self._jobs.get(job_id)
        if state is None or state.status not in (JobStatus.RUNNING, JobStatus.SCANNING):
            return False
        state.control.pause_event.set()
        state.status = JobStatus.PAUSED
        state.log.add("info", "Job paused by user")
        return True

    def resume_job(self, job_id: str) -> bool:
        state = self._jobs.get(job_id)
        if state is None or state.status != JobStatus.PAUSED:
            return False
        state.control.pause_event.clear()
        state.status = JobStatus.RUNNING
        state.log.add("info", "Job resumed by user")
        return True

    def cancel_job(self, job_id: str) -> bool:
        state = self._jobs.get(job_id)
        if state is None or state.status in (
            JobStatus.DONE,
            JobStatus.ERROR,
            JobStatus.CANCELLED,
        ):
            return False
        state.log.add("info", "Stop requested by user...")
        state.control.cancel_event.set()
        # wake the worker if it's currently blocked waiting out a pause
        state.control.pause_event.clear()
        return True


job_manager = JobManager()
