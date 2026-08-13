"""Optional CLIP-based semantic similarity ("Accurate mode").

Heavy dependencies (torch, open_clip) are imported lazily so the rest of
the app works fine without them installed — hash mode remains fully
functional even if this module's dependencies are missing.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_model = None
_preprocess = None
_device = "cpu"

_MODEL_NAME = "ViT-B-32"
_PRETRAINED_TAG = "laion2b_s34b_b79k"

INSTALL_HINT = "pip install imagecompare[clip]"


class ClipUnavailableError(RuntimeError):
    """Raised when CLIP mode is requested but dependencies aren't installed."""


def is_available() -> bool:
    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def is_model_loaded() -> bool:
    return _model is not None


def default_weights_cache_dir() -> Path:
    """Best-effort guess at where model weights get cached (HF Hub /
    open_clip conventions). Used only for the progress heartbeat below --
    never required for correctness, since loading still works even if
    this guess is wrong for a given environment/version."""
    hf_home = os.environ.get("HF_HOME")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    if xdg_cache:
        return Path(xdg_cache) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class ClipStatus:
    """A snapshot of CLIP mode's readiness, for the pre-flight status
    check surfaced in the UI before a job is even started.

    `weights_cache_dir_size_mb` is a best-effort heuristic, not a
    guarantee this exact model's weights are present -- see
    default_weights_cache_dir()'s docstring. It's still useful signal:
    an empty/tiny cache dir strongly suggests nothing has been downloaded
    yet, so the UI can warn "first run will download weights".
    """

    dependencies_installed: bool
    model_loaded: bool
    model_name: str
    pretrained_tag: str
    weights_cache_dir: str
    weights_cache_dir_size_mb: float
    install_hint: str


def get_status() -> ClipStatus:
    cache_dir = default_weights_cache_dir()
    size_mb = round(_dir_size_bytes(cache_dir) / (1024 * 1024), 1) if cache_dir.exists() else 0.0
    return ClipStatus(
        dependencies_installed=is_available(),
        model_loaded=is_model_loaded(),
        model_name=_MODEL_NAME,
        pretrained_tag=_PRETRAINED_TAG,
        weights_cache_dir=str(cache_dir),
        weights_cache_dir_size_mb=size_mb,
        install_hint=INSTALL_HINT,
    )


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def _ensure_model_loaded() -> None:
    global _model, _preprocess, _device
    if _model is not None:
        return
    try:
        import open_clip
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without extras installed
        raise ClipUnavailableError(
            f"CLIP mode requires the optional 'clip' extra: {INSTALL_HINT}"
        ) from exc

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED_TAG
    )
    model.eval().to(_device)
    _model = model
    _preprocess = preprocess


def ensure_model_ready(
    *,
    cache_dir: str | Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
    log_cb: Callable[[str, str], None] | None = None,
    heartbeat_interval: float = 1.0,
    log_every_n_seconds: float = 5.0,
    _loader: Callable[[], None] | None = None,
) -> None:
    """Load the CLIP model, reporting periodic progress if it isn't
    cached locally yet (first run downloads several hundred MB).

    There's no dependency-version-stable hook into open_clip/torch/
    huggingface_hub's internal download progress, so instead of trying
    to intercept exact byte counts, a background thread reports elapsed
    time plus how much the weights cache directory has grown, while the
    (potentially slow) load happens on the calling thread. This degrades
    gracefully: even if the cache-directory guess is wrong for a given
    environment/version, the user still sees "still working" heartbeat
    messages instead of an apparently frozen job.

    `_loader` is an injection point for tests -- production code should
    never need to pass it.
    """
    if is_model_loaded():
        return

    loader = _loader or _ensure_model_loaded
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else default_weights_cache_dir()
    start_size = _dir_size_bytes(resolved_cache_dir)
    start_time = time.monotonic()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        last_logged = 0.0
        while not stop_event.wait(heartbeat_interval):
            elapsed = time.monotonic() - start_time
            grown = max(0, _dir_size_bytes(resolved_cache_dir) - start_size)
            msg = f"Loading CLIP model... {elapsed:.0f}s elapsed"
            if grown > 0:
                msg += f", {grown / (1024 * 1024):.0f}MB downloaded so far"
            if progress_cb is not None:
                progress_cb(msg)
            if log_cb is not None and elapsed - last_logged >= log_every_n_seconds:
                last_logged = elapsed
                log_cb("info", msg)

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        if log_cb is not None:
            log_cb("info", "Loading CLIP model (downloading weights on first use)...")
        loader()
    finally:
        stop_event.set()
        thread.join(timeout=2.0)

    if log_cb is not None:
        elapsed = time.monotonic() - start_time
        log_cb("info", f"CLIP model ready ({elapsed:.1f}s)")


def compute_embedding(image_path: str | Path) -> np.ndarray:
    """Compute a normalized CLIP embedding vector for a single image."""
    import numpy as np
    import torch
    from PIL import Image

    _ensure_model_loaded()
    assert _model is not None and _preprocess is not None

    with Image.open(image_path) as img:
        tensor = _preprocess(img.convert("RGB")).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = _model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().numpy().astype(np.float32)


def raw_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Raw cosine similarity between two normalized CLIP embeddings, in
    [-1, 1], with no calibration applied. Exposed separately from
    cosine_similarity() so callers that want the uncalibrated value (e.g.
    to log the actual observed similarity distribution for a job, to
    help tune `baseline`) don't need to recompute the dot product."""
    import numpy as np

    sim = float(np.dot(vec_a, vec_b))
    return max(-1.0, min(1.0, sim))


def calibrate_similarity(raw_sim: float, *, baseline: float = 0.5) -> float:
    """Apply the baseline calibration (see cosine_similarity()'s
    docstring) to an already-computed raw cosine similarity."""
    if raw_sim <= baseline:
        return 0.0
    adjusted = (raw_sim - baseline) / (1.0 - baseline)
    return min(100.0, adjusted * 100)


def cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    *,
    baseline: float = 0.5,
) -> float:
    """Cosine similarity between two normalized CLIP embeddings, mapped to
    a 0-100 % display score.

    Raw cosine similarity between CLIP image embeddings does NOT spread
    evenly across [-1, 1] the way a naive linear rescale assumes. CLIP's
    embedding space is anisotropic (embeddings cluster in a narrow cone
    rather than spreading over the full hypersphere -- documented as the
    "modality gap" / cone effect), so two *completely unrelated* images
    routinely score 0.5-0.8 raw cosine similarity, not near 0. A plain
    `(sim + 1) / 2 * 100` mapping would show unrelated photos as 75-90%
    "similar", which is misleading.

    `baseline` is treated as the approximate floor for unrelated pairs:
    scores at or below it map to 0%, and 1.0 (identical) maps to 100%,
    linearly in between. 0.5 is a reasonable default based on commonly
    reported behavior for CLIP ViT-B/32-scale models on natural images,
    but the ideal value can vary by model and image domain -- if scores
    still look miscalibrated for a particular photo set, this is the
    number to adjust. The activity log reports the actual observed raw
    similarity distribution for each CLIP-mode job to help with that.
    """
    return calibrate_similarity(raw_cosine_similarity(vec_a, vec_b), baseline=baseline)
