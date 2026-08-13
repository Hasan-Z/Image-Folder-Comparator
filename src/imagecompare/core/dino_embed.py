"""Optional DINOv2-based visual similarity ("Visual mode").

Why DINOv2 instead of (or alongside) CLIP: CLIP is trained to align
images with *text* descriptions, which is a different objective from
"how visually similar are these two photos" -- that alignment pressure
is part of why CLIP's image-image cosine similarities cluster tightly
(the "cone effect", handled with a calibration baseline in
clip_embed.py). DINOv2 is trained purely on images with no text
involved, and is explicitly benchmarked on image retrieval / near-
duplicate detection -- the actual task this tool needs. Its embeddings
are also more naturally separated, needing a much gentler calibration,
and the "small" variant is meaningfully lighter than CLIP's image tower
(~22M vs ~88M parameters), so it's a genuinely better fit here, not just
an alternative.

Heavy dependencies (torch, transformers) are imported lazily so the rest
of the app works fine without them installed — hash mode remains fully
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
_processor = None
_device = "cpu"

_MODEL_ID = "facebook/dinov2-small"

INSTALL_HINT = "pip install imagecompare[dino]"


class DinoUnavailableError(RuntimeError):
    """Raised when DINOv2 mode is requested but dependencies aren't installed."""


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def is_model_loaded() -> bool:
    return _model is not None


def default_weights_cache_dir() -> Path:
    """Best-effort guess at where model weights get cached (Hugging Face
    Hub conventions -- transformers uses the same cache as huggingface_hub).
    Used only for the progress heartbeat below -- never required for
    correctness, since loading still works even if this guess is wrong
    for a given environment/version."""
    hf_home = os.environ.get("HF_HOME")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    if xdg_cache:
        return Path(xdg_cache) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class DinoStatus:
    """A snapshot of DINOv2 mode's readiness, for the pre-flight status
    check surfaced in the UI before a job is even started.

    `weights_cache_dir_size_mb` is a best-effort heuristic, not a
    guarantee this exact model's weights are present -- see
    default_weights_cache_dir()'s docstring.
    """

    dependencies_installed: bool
    model_loaded: bool
    model_name: str
    weights_cache_dir: str
    weights_cache_dir_size_mb: float
    install_hint: str


def get_status() -> DinoStatus:
    cache_dir = default_weights_cache_dir()
    size_mb = round(_dir_size_bytes(cache_dir) / (1024 * 1024), 1) if cache_dir.exists() else 0.0
    return DinoStatus(
        dependencies_installed=is_available(),
        model_loaded=is_model_loaded(),
        model_name=_MODEL_ID,
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
    global _model, _processor, _device
    if _model is not None:
        return
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:  # pragma: no cover - exercised only without extras installed
        raise DinoUnavailableError(
            f"DINOv2 mode requires the optional 'dino' extra: {INSTALL_HINT}"
        ) from exc

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
    model = AutoModel.from_pretrained(_MODEL_ID)
    model.eval().to(_device)
    _model = model


def ensure_model_ready(
    *,
    cache_dir: str | Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
    log_cb: Callable[[str, str], None] | None = None,
    heartbeat_interval: float = 1.0,
    log_every_n_seconds: float = 5.0,
    _loader: Callable[[], None] | None = None,
) -> None:
    """Load the DINOv2 model, reporting periodic progress if it isn't
    cached locally yet (first run downloads roughly 85-90MB -- notably
    smaller than CLIP's weights).

    See clip_embed.ensure_model_ready() for the full rationale behind
    the heartbeat-based approach; this mirrors it exactly.

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
            msg = f"Loading DINOv2 model... {elapsed:.0f}s elapsed"
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
            log_cb("info", "Loading DINOv2 model (downloading weights on first use)...")
        loader()
    finally:
        stop_event.set()
        thread.join(timeout=2.0)

    if log_cb is not None:
        elapsed = time.monotonic() - start_time
        log_cb("info", f"DINOv2 model ready ({elapsed:.1f}s)")


def compute_embedding(image_path: str | Path) -> np.ndarray:
    """Compute a normalized DINOv2 embedding vector for a single image,
    using the [CLS] token from the final hidden state -- the standard
    choice for DINOv2 image-level retrieval/similarity tasks."""
    import numpy as np
    import torch
    from PIL import Image

    _ensure_model_loaded()
    assert _model is not None and _processor is not None

    with Image.open(image_path) as img:
        inputs = _processor(images=img.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)

    return cls_embedding.squeeze(0).cpu().numpy().astype(np.float32)


def raw_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Raw cosine similarity between two normalized DINOv2 embeddings, in
    [-1, 1], with no calibration applied. Exposed separately from
    cosine_similarity() so callers that want the uncalibrated value (e.g.
    to log the actual observed similarity distribution for a job, to
    help tune `baseline`) don't need to recompute the dot product."""
    import numpy as np

    sim = float(np.dot(vec_a, vec_b))
    return max(-1.0, min(1.0, sim))


def calibrate_similarity(raw_sim: float, *, baseline: float = 0.3) -> float:
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
    baseline: float = 0.3,
) -> float:
    """Cosine similarity between two normalized DINOv2 embeddings, mapped
    to a 0-100 % display score.

    DINOv2 embeddings are more naturally separated than CLIP's (see this
    module's docstring), so the calibration baseline can be lower -- but
    a small calibration is still applied for consistency with the hash
    and CLIP modes, and because some residual clustering is common with
    most learned embedding spaces. 0.3 is an approximation; if scores
    still look off for a particular photo set, this is the number to
    adjust (see config.py's `dino_similarity_baseline`). The activity
    log reports the actual observed raw similarity distribution for each
    DINOv2-mode job to help with that.
    """
    return calibrate_similarity(raw_cosine_similarity(vec_a, vec_b), baseline=baseline)
