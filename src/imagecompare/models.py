"""Pydantic schemas shared between the API layer and job manager."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ComparisonMode(str, Enum):
    HASH = "hash"
    CLIP = "clip"
    DINO = "dino"


class JobStatus(str, Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    DONE = "done"
    ERROR = "error"


class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LogEntry(BaseModel):
    timestamp: str
    level: LogLevel
    message: str


class JobLogsResponse(BaseModel):
    job_id: str
    entries: list[LogEntry]


class CompareRequest(BaseModel):
    folder1: str
    folder2: str
    mode: ComparisonMode = ComparisonMode.HASH
    recursive: bool = False
    include_patterns: str = ""
    exclude_patterns: str = ""
    visual_threshold: float = Field(default=70.0, ge=0, le=100)
    filename_threshold: float = Field(default=90.0, ge=0, le=100)
    deep_rotation: bool = False
    hash_size: int = Field(default=16, ge=4, le=32)


class ScanPreviewRequest(BaseModel):
    folder: str
    recursive: bool = False
    include_patterns: str = ""
    exclude_patterns: str = ""


class ScanPreviewResponse(BaseModel):
    count: int
    sample: list[str]


class BrowseEntry(BaseModel):
    name: str
    path: str
    is_dir: bool


class BrowseResponse(BaseModel):
    current_path: str
    parent_path: str | None
    entries: list[BrowseEntry]


class MatchResult(BaseModel):
    image1_name: str
    image1_path: str
    image1_relative_path: str
    image1_md5: str
    image2_name: str
    image2_path: str
    image2_relative_path: str
    image2_md5: str
    visual_similarity: float
    filename_similarity: float
    is_exact_duplicate: bool
    best_rotation_angle: int | None = None
    thumb1_url: str
    thumb2_url: str


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    total_items: int = 0
    processed_items: int = 0
    percent: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    error_message: str | None = None
    phase: str = ""


class ComparisonSummary(BaseModel):
    mode: str
    folder1: str
    folder2: str
    total_images_a: int
    total_images_b: int
    total_pairs_compared: int
    total_matches: int
    exact_duplicates: int
    highest_similarity: float | None = None
    average_similarity: float | None = None
    lowest_similarity: float | None = None
    deep_rotation_used: bool
    visual_threshold: float
    elapsed_seconds: float


class JobResultsResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_matches: int
    matches: list[MatchResult]
    summary: ComparisonSummary | None = None


class ClipStatusResponse(BaseModel):
    dependencies_installed: bool
    model_loaded: bool
    model_name: str
    pretrained_tag: str
    weights_cache_dir: str
    weights_cache_dir_size_mb: float
    install_hint: str


class DinoStatusResponse(BaseModel):
    dependencies_installed: bool
    model_loaded: bool
    model_name: str
    weights_cache_dir: str
    weights_cache_dir_size_mb: float
    install_hint: str


class UploadFolderResponse(BaseModel):
    path: str
    file_count: int
