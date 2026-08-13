"""HTTP API routes."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from imagecompare.api.xlsx_export import build_report_workbook
from imagecompare.config import settings
from imagecompare.core import clip_embed, dino_embed
from imagecompare.core.scanner import scan_folder
from imagecompare.core.upload import (
    InvalidRelativePathError,
    cleanup_old_uploads,
    sanitize_relative_path,
)
from imagecompare.jobs.manager import job_manager
from imagecompare.models import (
    BrowseEntry,
    BrowseResponse,
    ClipStatusResponse,
    CompareRequest,
    ComparisonMode,
    DinoStatusResponse,
    JobLogsResponse,
    JobProgress,
    JobResultsResponse,
    JobStatus,
    MatchResult,
    ScanPreviewRequest,
    ScanPreviewResponse,
    UploadFolderResponse,
)

router = APIRouter(prefix="/api")

THUMB_CACHE_DIR = Path.home() / ".imagecompare" / "thumbs"
UPLOADS_DIR = Path.home() / ".imagecompare" / "uploads"
_UPLOAD_MAX_AGE_SECONDS = 6 * 60 * 60  # prune upload dirs older than this on each new upload
_UPLOAD_CHUNK_SIZE = 1024 * 1024  # stream to disk in 1 MiB chunks, don't buffer whole files
DRIVES_SENTINEL = "__DRIVES__"


def _list_roots() -> list[Path]:
    """Top-level entry points for the folder browser: drive letters on
    Windows, the filesystem root everywhere else."""
    if os.name == "nt":
        import string

        roots = []
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                roots.append(drive)
        return roots
    return [Path("/")]


def _job_folders(job_id: str) -> tuple[Path, Path]:
    state = job_manager.get_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return Path(state.folder1).resolve(), Path(state.folder2).resolve()


@router.post("/upload-folder", response_model=UploadFolderResponse)
async def upload_folder(files: list[UploadFile] = File(...)) -> UploadFolderResponse:  # noqa: B008
    """Receive a folder dropped in the browser (drag-and-drop) and save
    it to a server-side temp directory, which can then be used exactly
    like any other scanned folder.

    Browsers deliberately hide the real filesystem path of dropped
    files/folders, so this is the only way to make drag-and-drop work:
    the frontend walks the dropped directory tree client-side and
    uploads each file with its relative path preserved (via
    `FormData.append(name, file, relativePath)`), and this endpoint
    reconstructs that structure on disk.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    cleanup_old_uploads(UPLOADS_DIR, _UPLOAD_MAX_AGE_SECONDS)

    upload_id = uuid.uuid4().hex
    dest_root = UPLOADS_DIR / upload_id
    dest_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    try:
        for upload in files:
            if not upload.filename:
                continue
            try:
                rel_path = sanitize_relative_path(upload.filename)
            except InvalidRelativePathError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            target = dest_root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                while chunk := await upload.read(_UPLOAD_CHUNK_SIZE):
                    out.write(chunk)
            saved += 1
    except HTTPException:
        shutil.rmtree(dest_root, ignore_errors=True)
        raise

    if saved == 0:
        shutil.rmtree(dest_root, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No valid files in upload")

    return UploadFolderResponse(path=str(dest_root), file_count=saved)


@router.get("/browse", response_model=BrowseResponse)
def browse(path: str | None = Query(default=None)) -> BrowseResponse:
    """List subdirectories of `path` (or the user's home directory if
    omitted) so the frontend can offer a server-side 'Open Folder'
    picker without relying on browser folder-upload.

    A special `path=__DRIVES__` request returns the top-level entry
    points (drive letters on Windows, `/` elsewhere) — this is how the
    UI's 'This PC' shortcut and the 'up' chain from a drive root work.
    """
    if path == DRIVES_SENTINEL:
        drive_entries = [
            BrowseEntry(name=str(root), path=str(root), is_dir=True) for root in _list_roots()
        ]
        return BrowseResponse(current_path="This PC", parent_path=None, entries=drive_entries)

    target = Path(path).expanduser().resolve() if path else Path.home()

    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    entries: list[BrowseEntry] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            entries.append(BrowseEntry(name=child.name, path=str(child), is_dir=is_dir))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}") from exc

    # directories first, then files, alphabetically within each group
    entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))

    if target.parent == target:
        # we're at a filesystem root (e.g. "C:\" or "/") — offer the
        # drive/root picker as the next "up" step instead of a dead end
        parent = DRIVES_SENTINEL
    else:
        parent = str(target.parent)

    return BrowseResponse(current_path=str(target), parent_path=parent, entries=entries)


@router.post("/scan-preview", response_model=ScanPreviewResponse)
def scan_preview(req: ScanPreviewRequest) -> ScanPreviewResponse:
    """Cheap live-count preview for the include/exclude filter boxes."""
    try:
        images = scan_folder(
            req.folder,
            recursive=req.recursive,
            include_patterns=req.include_patterns,
            exclude_patterns=req.exclude_patterns,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sample = [img.relative_path for img in images[:8]]
    return ScanPreviewResponse(count=len(images), sample=sample)


@router.get("/clip-status", response_model=ClipStatusResponse)
def clip_status() -> ClipStatusResponse:
    """Pre-flight readiness check for CLIP mode, so the UI can warn the
    user before they start a job rather than only finding out mid-run."""
    status = clip_embed.get_status()
    return ClipStatusResponse(
        dependencies_installed=status.dependencies_installed,
        model_loaded=status.model_loaded,
        model_name=status.model_name,
        pretrained_tag=status.pretrained_tag,
        weights_cache_dir=status.weights_cache_dir,
        weights_cache_dir_size_mb=status.weights_cache_dir_size_mb,
        install_hint=status.install_hint,
    )


@router.get("/dino-status", response_model=DinoStatusResponse)
def dino_status() -> DinoStatusResponse:
    """Pre-flight readiness check for DINOv2 mode, so the UI can warn the
    user before they start a job rather than only finding out mid-run."""
    status = dino_embed.get_status()
    return DinoStatusResponse(
        dependencies_installed=status.dependencies_installed,
        model_loaded=status.model_loaded,
        model_name=status.model_name,
        weights_cache_dir=status.weights_cache_dir,
        weights_cache_dir_size_mb=status.weights_cache_dir_size_mb,
        install_hint=status.install_hint,
    )


_MODE_LABELS = {ComparisonMode.CLIP: "CLIP", ComparisonMode.DINO: "DINOv2"}
_MODE_MODULES = {ComparisonMode.CLIP: clip_embed, ComparisonMode.DINO: dino_embed}


@router.post("/compare")
def compare(req: CompareRequest) -> dict:
    for folder in (req.folder1, req.folder2):
        p = Path(folder).expanduser()
        if not p.exists() or not p.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a valid directory: {folder}")

    module = _MODE_MODULES.get(req.mode)
    if module is not None and not module.is_available():
        label = _MODE_LABELS[req.mode]
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} mode requires the optional '{req.mode.value}' extra, which isn't "
                f"installed. Install it with: {module.INSTALL_HINT}"
            ),
        )

    job_id = job_manager.create_job(
        folder1=req.folder1,
        folder2=req.folder2,
        mode=req.mode.value,
        recursive=req.recursive,
        include_patterns=req.include_patterns,
        exclude_patterns=req.exclude_patterns,
        visual_threshold=req.visual_threshold,
        filename_threshold=req.filename_threshold,
        deep_rotation=req.deep_rotation,
        hash_size=req.hash_size,
        thumb_cache_dir=str(THUMB_CACHE_DIR),
    )
    return {"job_id": job_id}


@router.get("/jobs/{job_id}", response_model=JobProgress)
def job_progress(job_id: str) -> JobProgress:
    progress = job_manager.get_progress(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return progress


@router.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
def job_logs(job_id: str) -> JobLogsResponse:
    state = job_manager.get_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobLogsResponse(job_id=job_id, entries=state.log.snapshot())


@router.post("/jobs/{job_id}/pause")
def pause_job(job_id: str) -> dict:
    if job_manager.get_state(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job_manager.pause_job(job_id):
        raise HTTPException(status_code=409, detail="Job cannot be paused right now")
    return {"status": "paused"}


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict:
    if job_manager.get_state(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job_manager.resume_job(job_id):
        raise HTTPException(status_code=409, detail="Job cannot be resumed right now")
    return {"status": "running"}


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    if job_manager.get_state(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job_manager.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job cannot be stopped right now")
    return {"status": "cancelling"}


@router.get("/jobs/{job_id}/results", response_model=JobResultsResponse)
def job_results(job_id: str) -> JobResultsResponse:
    state = job_manager.get_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status == JobStatus.CANCELLED:
        raise HTTPException(status_code=409, detail="Job was stopped before completion")
    if state.status not in (JobStatus.DONE, JobStatus.ERROR):
        raise HTTPException(status_code=409, detail="Job not finished yet")

    return _build_results_response(job_id, state)


def _build_results_response(job_id: str, state: object) -> JobResultsResponse:
    matches: list[MatchResult] = []
    for c in state.results:  # type: ignore[attr-defined]
        matches.append(
            MatchResult(
                image1_name=c.image1.name,
                image1_path=str(c.image1.path),
                image1_relative_path=c.image1.relative_path,
                image1_md5=c.image1_md5,
                image2_name=c.image2.name,
                image2_path=str(c.image2.path),
                image2_relative_path=c.image2.relative_path,
                image2_md5=c.image2_md5,
                visual_similarity=c.visual_similarity,
                filename_similarity=c.filename_similarity,
                is_exact_duplicate=c.is_exact_duplicate,
                best_rotation_angle=c.best_rotation_angle,
                thumb1_url=f"/api/thumb?path={_quote(str(c.image1.path))}",
                thumb2_url=f"/api/thumb?path={_quote(str(c.image2.path))}",
            )
        )

    return JobResultsResponse(
        job_id=job_id,
        status=state.status,  # type: ignore[attr-defined]
        total_matches=len(matches),
        matches=matches,
        summary=state.summary,  # type: ignore[attr-defined]
    )


@router.get("/jobs/{job_id}/export.xlsx")
def export_job_xlsx(job_id: str) -> StreamingResponse:
    """Export a finished job's summary, full match table (including MD5
    hashes, paths, and filenames), and activity log as a downloadable
    Excel workbook."""
    state = job_manager.get_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status == JobStatus.CANCELLED:
        raise HTTPException(status_code=409, detail="Job was stopped before completion")
    if state.status not in (JobStatus.DONE, JobStatus.ERROR):
        raise HTTPException(status_code=409, detail="Job not finished yet")

    results = _build_results_response(job_id, state)
    buffer = build_report_workbook(results, log_entries=state.log.snapshot())

    filename = f"image-compare-report-{job_id[:8]}.xlsx"
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _quote(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="")


def _validate_path_in_scanned_roots(job_id: str, raw_path: str) -> Path:
    folder1, folder2 = _job_folders(job_id)
    target = Path(raw_path).resolve()
    for root in (folder1, folder2):
        try:
            target.relative_to(root)
            return target
        except ValueError:
            continue
    raise HTTPException(status_code=403, detail="Path is outside the scanned folders")


@router.get("/thumb")
def get_thumb(path: str) -> FileResponse:
    """Serve a cached thumbnail for an image path. Thumbnails are cached
    under a content-hash filename; we regenerate on demand if missing."""
    from imagecompare.core.thumbnails import get_or_create_thumbnail

    src = Path(path)
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404, detail="Source image not found")
    thumb_path = get_or_create_thumbnail(src, THUMB_CACHE_DIR)
    return FileResponse(thumb_path, media_type="image/jpeg")


@router.get("/image")
def get_full_image(job_id: str, path: str) -> FileResponse:
    """Serve a full-resolution image for the preview modal, restricted to
    files inside the two folders that were actually scanned for this job.
    """
    target = _validate_path_in_scanned_roots(job_id, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if target.stat().st_size > settings.max_preview_bytes:
        raise HTTPException(status_code=413, detail="Image too large to preview")
    return FileResponse(target)
