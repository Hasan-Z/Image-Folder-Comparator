from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from imagecompare.core.scanner import scan_folder
from imagecompare.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Light Table" in res.text


def test_browse_home_directory(client):
    res = client.get("/api/browse")
    assert res.status_code == 200
    data = res.json()
    assert "current_path" in data
    assert isinstance(data["entries"], list)


def test_browse_invalid_path_returns_400(client, tmp_path):
    res = client.get("/api/browse", params={"path": str(tmp_path / "nope")})
    assert res.status_code == 400


def test_scan_preview_counts_images(client, folder_pair):
    f1, _f2 = folder_pair
    res = client.post("/api/scan-preview", json={"folder": str(f1), "recursive": False})
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 3  # circle_red, circle_orange, square_blue


def test_scan_preview_missing_folder_returns_400(client, tmp_path):
    res = client.post("/api/scan-preview", json={"folder": str(tmp_path / "missing")})
    assert res.status_code == 400


def test_compare_rejects_invalid_folder(client, tmp_path, folder_pair):
    f1, _f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={"folder1": str(f1), "folder2": str(tmp_path / "missing")},
    )
    assert res.status_code == 400


def test_clip_status_endpoint_reflects_unavailable(client):
    # torch/open_clip aren't installed in this test environment, so this
    # exercises the real "not available" path end-to-end
    res = client.get("/api/clip-status")
    assert res.status_code == 200
    data = res.json()
    assert data["dependencies_installed"] is False
    assert "pip install" in data["install_hint"]
    assert data["model_name"]
    assert data["weights_cache_dir_size_mb"] >= 0.0


def test_clip_status_endpoint_reflects_available(client, monkeypatch):
    from imagecompare.core import clip_embed

    monkeypatch.setattr(clip_embed, "is_available", lambda: True)
    res = client.get("/api/clip-status")
    assert res.status_code == 200
    assert res.json()["dependencies_installed"] is True


def test_compare_rejects_clip_mode_when_dependencies_missing(client, folder_pair):
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={"folder1": str(f1), "folder2": str(f2), "mode": "clip"},
    )
    assert res.status_code == 400
    assert "pip install" in res.json()["detail"]


def test_compare_accepts_clip_mode_when_dependencies_available(client, folder_pair, monkeypatch):
    from imagecompare.core import clip_embed

    monkeypatch.setattr(clip_embed, "is_available", lambda: True)
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={"folder1": str(f1), "folder2": str(f2), "mode": "clip"},
    )
    # accepted at the API layer (a job is created); the job itself will
    # still fail once it actually tries to load the (uninstalled) model,
    # which is a separate, already-covered failure path
    assert res.status_code == 200
    assert "job_id" in res.json()


def test_dino_status_endpoint_reflects_unavailable(client):
    # torch/transformers aren't installed in this test environment, so
    # this exercises the real "not available" path end-to-end
    res = client.get("/api/dino-status")
    assert res.status_code == 200
    data = res.json()
    assert data["dependencies_installed"] is False
    assert "pip install" in data["install_hint"]
    assert data["model_name"]
    assert data["weights_cache_dir_size_mb"] >= 0.0


def test_dino_status_endpoint_reflects_available(client, monkeypatch):
    from imagecompare.core import dino_embed

    monkeypatch.setattr(dino_embed, "is_available", lambda: True)
    res = client.get("/api/dino-status")
    assert res.status_code == 200
    assert res.json()["dependencies_installed"] is True


def test_compare_rejects_dino_mode_when_dependencies_missing(client, folder_pair):
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={"folder1": str(f1), "folder2": str(f2), "mode": "dino"},
    )
    assert res.status_code == 400
    assert "pip install" in res.json()["detail"]


def test_compare_accepts_dino_mode_when_dependencies_available(client, folder_pair, monkeypatch):
    from imagecompare.core import dino_embed

    monkeypatch.setattr(dino_embed, "is_available", lambda: True)
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={"folder1": str(f1), "folder2": str(f2), "mode": "dino"},
    )
    assert res.status_code == 200
    assert "job_id" in res.json()


def test_full_compare_flow(client, folder_pair):
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={
            "folder1": str(f1),
            "folder2": str(f2),
            "mode": "hash",
            "visual_threshold": 0.0,
        },
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    # poll until done
    status = None
    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        status = prog["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.05)
    assert status == "done"

    results = client.get(f"/api/jobs/{job_id}/results")
    assert results.status_code == 200
    data = results.json()
    assert data["total_matches"] > 0
    assert any(m["is_exact_duplicate"] for m in data["matches"])


def test_results_include_summary_report(client, folder_pair):
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={"folder1": str(f1), "folder2": str(f2), "visual_threshold": 0.0},
    )
    job_id = res.json()["job_id"]

    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.02)

    data = client.get(f"/api/jobs/{job_id}/results").json()
    summary = data["summary"]
    assert summary is not None
    assert summary["mode"] == "hash"
    assert summary["total_images_a"] == 3
    assert summary["total_images_b"] == 3
    assert summary["total_pairs_compared"] == 9
    assert summary["total_matches"] == data["total_matches"]
    assert summary["exact_duplicates"] == 1
    assert summary["highest_similarity"] == 100.0
    assert summary["lowest_similarity"] is not None
    assert summary["average_similarity"] is not None
    assert summary["deep_rotation_used"] is False
    assert summary["visual_threshold"] == 0.0
    assert summary["elapsed_seconds"] >= 0.0
    assert summary["folder1"] == str(f1.resolve())
    assert summary["folder2"] == str(f2.resolve())


def test_summary_reports_deep_rotation_flag(client, folder_pair):
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={
            "folder1": str(f1),
            "folder2": str(f2),
            "visual_threshold": 0.0,
            "deep_rotation": True,
        },
    )
    job_id = res.json()["job_id"]

    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.02)

    data = client.get(f"/api/jobs/{job_id}/results").json()
    assert data["summary"]["deep_rotation_used"] is True


def _run_to_completion(client, folder_pair, **extra_json):
    f1, f2 = folder_pair
    payload = {"folder1": str(f1), "folder2": str(f2), "visual_threshold": 0.0}
    payload.update(extra_json)
    res = client.post("/api/compare", json=payload)
    job_id = res.json()["job_id"]
    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    return job_id


def test_export_xlsx_returns_valid_workbook_with_expected_sheets(client, folder_pair):
    from io import BytesIO

    from openpyxl import load_workbook

    job_id = _run_to_completion(client, folder_pair)
    res = client.get(f"/api/jobs/{job_id}/export.xlsx")

    assert res.status_code == 200
    assert res.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "attachment" in res.headers["content-disposition"]
    assert ".xlsx" in res.headers["content-disposition"]

    wb = load_workbook(BytesIO(res.content))
    assert wb.sheetnames == ["Summary", "Matches", "Activity Log"]


def test_export_xlsx_matches_sheet_contains_md5_and_paths(client, folder_pair):
    from io import BytesIO

    from openpyxl import load_workbook

    job_id = _run_to_completion(client, folder_pair)
    res = client.get(f"/api/jobs/{job_id}/export.xlsx")
    wb = load_workbook(BytesIO(res.content))

    ws = wb["Matches"]
    headers = [c.value for c in ws[1]]
    assert "Folder A - MD5" in headers
    assert "Folder B - MD5" in headers
    assert "Folder A - Full Path" in headers
    assert "Exact Duplicate (MD5)" in headers

    md5_col_a = headers.index("Folder A - MD5")
    md5_col_b = headers.index("Folder B - MD5")
    dup_col = headers.index("Exact Duplicate (MD5)")

    data_rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(data_rows) > 0
    for row in data_rows:
        assert row[md5_col_a]  # non-empty MD5 string
        assert row[md5_col_b]
        assert row[dup_col] in ("Yes", "No")

    # the known exact-duplicate pair should have matching MD5s and "Yes"
    dup_rows = [r for r in data_rows if r[dup_col] == "Yes"]
    assert len(dup_rows) == 1
    assert dup_rows[0][md5_col_a] == dup_rows[0][md5_col_b]


def test_export_xlsx_summary_sheet_has_key_stats(client, folder_pair):
    from io import BytesIO

    from openpyxl import load_workbook

    job_id = _run_to_completion(client, folder_pair)
    res = client.get(f"/api/jobs/{job_id}/export.xlsx")
    wb = load_workbook(BytesIO(res.content))

    ws = wb["Summary"]
    labels = [row[0] for row in ws.iter_rows(min_row=3, max_col=1, values_only=True) if row[0]]
    assert "Comparison mode" in labels
    assert "Matches found" in labels
    assert "Exact duplicates (MD5)" in labels
    assert "Highest similarity" in labels


def test_export_xlsx_activity_log_sheet_has_entries(client, folder_pair):
    from io import BytesIO

    from openpyxl import load_workbook

    job_id = _run_to_completion(client, folder_pair)
    res = client.get(f"/api/jobs/{job_id}/export.xlsx")
    wb = load_workbook(BytesIO(res.content))

    ws = wb["Activity Log"]
    headers = [c.value for c in ws[1]]
    assert headers == ["Timestamp", "Level", "Message"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(rows) > 0
    assert any("Job created" in r[2] for r in rows)


def test_export_xlsx_404_for_unknown_job(client):
    res = client.get("/api/jobs/does-not-exist/export.xlsx")
    assert res.status_code == 404


def test_export_xlsx_409_for_unfinished_job(client, folder_pair):
    from imagecompare.jobs.manager import job_manager
    from imagecompare.models import JobStatus

    f1, f2 = folder_pair
    res = client.post(
        "/api/compare", json={"folder1": str(f1), "folder2": str(f2), "visual_threshold": 0.0}
    )
    job_id = res.json()["job_id"]

    state = job_manager.get_state(job_id)
    # force it into a paused/running-looking state deterministically
    state.control.pause_event.set()
    state.status = JobStatus.PAUSED

    export_res = client.get(f"/api/jobs/{job_id}/export.xlsx")
    assert export_res.status_code == 409

    # let it finish so the background thread doesn't linger mid-test
    state.control.pause_event.clear()
    state.status = JobStatus.RUNNING


def test_export_xlsx_409_for_cancelled_job(client, folder_pair):
    job_id = _run_to_completion(client, folder_pair)  # will already be done, so cancel a fresh one
    from imagecompare.jobs.manager import job_manager
    from imagecompare.models import JobStatus

    state = job_manager.get_state(job_id)
    state.status = JobStatus.CANCELLED

    res = client.get(f"/api/jobs/{job_id}/export.xlsx")
    assert res.status_code == 409


def test_job_progress_404_for_unknown_job(client):
    res = client.get("/api/jobs/does-not-exist")
    assert res.status_code == 404


def test_job_logs_endpoint_returns_entries(client, folder_pair):
    job_id = _create_and_pause(client, folder_pair)
    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.02)

    res = client.get(f"/api/jobs/{job_id}/logs")
    assert res.status_code == 200
    data = res.json()
    assert data["job_id"] == job_id
    assert len(data["entries"]) > 0
    assert any("Job created" in e["message"] for e in data["entries"])
    assert any("Job finished" in e["message"] for e in data["entries"])
    for entry in data["entries"]:
        assert entry["level"] in ("info", "warning", "error")
        assert entry["timestamp"]


def test_job_logs_endpoint_404_for_unknown_job(client):
    res = client.get("/api/jobs/does-not-exist/logs")
    assert res.status_code == 404


def test_job_logs_record_pause_and_resume(client, folder_pair):
    job_id = _create_and_pause(client, folder_pair)
    client.post(f"/api/jobs/{job_id}/pause")
    client.post(f"/api/jobs/{job_id}/resume")

    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.02)

    entries = client.get(f"/api/jobs/{job_id}/logs").json()["entries"]
    messages = [e["message"] for e in entries]
    assert any("paused" in m.lower() for m in messages)
    assert any("resumed" in m.lower() for m in messages)


def test_results_404_for_unknown_job(client):
    res = client.get("/api/jobs/does-not-exist/results")
    assert res.status_code == 404


def test_browse_drives_sentinel_returns_roots(client):
    res = client.get("/api/browse", params={"path": "__DRIVES__"})
    assert res.status_code == 200
    data = res.json()
    assert data["current_path"] == "This PC"
    assert data["parent_path"] is None
    assert len(data["entries"]) >= 1
    assert all(e["is_dir"] for e in data["entries"])


def test_browse_from_filesystem_root_points_up_to_drives(client):
    res = client.get("/api/browse", params={"path": "/"})
    assert res.status_code == 200
    data = res.json()
    assert data["parent_path"] == "__DRIVES__"


def _create_and_pause(client, folder_pair):
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare",
        json={"folder1": str(f1), "folder2": str(f2), "visual_threshold": 0.0},
    )
    return res.json()["job_id"]


def test_pause_resume_stop_unknown_job_returns_404(client):
    assert client.post("/api/jobs/does-not-exist/pause").status_code == 404
    assert client.post("/api/jobs/does-not-exist/resume").status_code == 404
    assert client.post("/api/jobs/does-not-exist/stop").status_code == 404


def test_resume_job_that_was_never_paused_returns_409(client, folder_pair):
    job_id = _create_and_pause(client, folder_pair)
    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    res = client.post(f"/api/jobs/{job_id}/resume")
    assert res.status_code == 409


def test_stop_already_done_job_returns_409(client, folder_pair):
    job_id = _create_and_pause(client, folder_pair)
    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    res = client.post(f"/api/jobs/{job_id}/stop")
    assert res.status_code == 409


def test_stop_running_job_via_api(client, folder_pair):
    job_id = _create_and_pause(client, folder_pair)
    res = client.post(f"/api/jobs/{job_id}/stop")
    # either it stopped successfully, or (on a very fast machine) the job
    # had already finished before the stop request landed — both are
    # acceptable outcomes for this race; what matters is no 500s / hangs.
    assert res.status_code in (200, 409)

    final_status = None
    for _ in range(200):
        final_status = client.get(f"/api/jobs/{job_id}").json()["status"]
        if final_status in ("done", "error", "cancelled"):
            break
        time.sleep(0.02)
    assert final_status in ("done", "cancelled")


def test_results_endpoint_rejects_cancelled_job(client, folder_pair):
    job_id = _create_and_pause(client, folder_pair)
    stop_res = client.post(f"/api/jobs/{job_id}/stop")
    if stop_res.status_code != 200:
        pytest.skip("job completed before stop could land")

    for _ in range(200):
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        if status == "cancelled":
            break
        time.sleep(0.02)

    res = client.get(f"/api/jobs/{job_id}/results")
    assert res.status_code == 409


def test_upload_folder_creates_files_with_correct_structure(client):
    files = [
        ("files", ("top.png", b"top-bytes", "image/png")),
        ("files", ("sub/inner.png", b"inner-bytes", "image/png")),
    ]
    res = client.post("/api/upload-folder", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["file_count"] == 2

    root = Path(data["path"])
    assert (root / "top.png").read_bytes() == b"top-bytes"
    assert (root / "sub" / "inner.png").read_bytes() == b"inner-bytes"


def test_upload_folder_result_is_directly_scannable(client):
    files = [
        ("files", ("a.png", b"aaaa", "image/png")),
        ("files", ("sub/b.png", b"bbbb", "image/png")),
    ]
    res = client.post("/api/upload-folder", files=files)
    root = res.json()["path"]

    images = scan_folder(root, recursive=True)
    names = {img.name for img in images}
    assert names == {"a.png", "b.png"}


def test_upload_folder_rejects_missing_files_field(client):
    res = client.post("/api/upload-folder")
    assert res.status_code in (400, 422)


def test_upload_folder_rejects_path_traversal(client):
    files = [("files", ("../../evil.png", b"data", "image/png"))]
    res = client.post("/api/upload-folder", files=files)
    assert res.status_code == 400


def test_upload_folder_traversal_attempt_leaves_no_partial_dir(client, tmp_path, monkeypatch):
    from imagecompare.api import routes

    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(routes, "UPLOADS_DIR", uploads_dir)

    files = [
        ("files", ("ok.png", b"data", "image/png")),
        ("files", ("../escape.png", b"data", "image/png")),
    ]
    res = client.post("/api/upload-folder", files=files)
    assert res.status_code == 400
    remaining = list(uploads_dir.iterdir()) if uploads_dir.exists() else []
    assert remaining == []


def test_upload_folder_generates_unique_paths_per_request(client):
    files = [("files", ("a.png", b"data", "image/png"))]
    res1 = client.post("/api/upload-folder", files=files)
    res2 = client.post("/api/upload-folder", files=files)
    assert res1.json()["path"] != res2.json()["path"]


def test_upload_folder_cleans_up_old_uploads(client, tmp_path, monkeypatch):
    from imagecompare.api import routes

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    old_dir = uploads_dir / "stale"
    old_dir.mkdir()
    old_time = time.time() - (100 * 24 * 60 * 60)
    os.utime(old_dir, (old_time, old_time))

    monkeypatch.setattr(routes, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(routes, "_UPLOAD_MAX_AGE_SECONDS", 60 * 60)

    files = [("files", ("a.png", b"data", "image/png"))]
    client.post("/api/upload-folder", files=files)

    assert not old_dir.exists()


def test_image_endpoint_rejects_path_outside_scanned_folders(client, folder_pair, tmp_path):
    f1, f2 = folder_pair
    res = client.post(
        "/api/compare", json={"folder1": str(f1), "folder2": str(f2), "visual_threshold": 0.0}
    )
    job_id = res.json()["job_id"]

    for _ in range(200):
        prog = client.get(f"/api/jobs/{job_id}").json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.05)

    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"not a real image but path check happens first")

    res = client.get("/api/image", params={"job_id": job_id, "path": str(outside_file)})
    assert res.status_code == 403
