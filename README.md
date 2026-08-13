# Light Table — Image Folder Comparator

Compare two folders of images and find similar or duplicate photos between
them — e.g. *"image1 from Folder A is 90% similar to image10 from Folder B"*
— with a FastAPI backend and a self-contained web UI. Runs entirely on your
own machine; nothing leaves your computer.

![status](https://img.shields.io/badge/tests-182%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Optional: DINOv2 / CLIP visual modes](#optional-dinov2-or-clip-visualsemantic-modes)
- [How it works](#how-it-works)
- [API](#api)
- [Development](#development)
- [Project layout](#project-layout)
- [Notes & limitations](#notes--limitations)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Three comparison modes**
  - **Hash (fast)** — perceptual-hash based, near-instant, great for
    near-duplicates (resized, recompressed, watermarked).
  - **DINOv2 (visual)** *(optional extra)* — a self-supervised vision model
    purpose-built for image-to-image similarity/retrieval (no text involved
    in training, unlike CLIP), catching the same subject under different
    angles/lighting/crops. Smaller and CPU-friendlier than CLIP (~22M vs
    ~88M parameters for the image tower) and generally the better default
    for "are these two photos visually similar" — **recommended over CLIP**
    for this tool's actual use case.
  - **CLIP (semantic)** *(optional extra)* — trained to align images with
    text descriptions, so it's better at matching by subject/scene/concept
    than by close visual likeness. Useful as a second opinion, but its
    embedding space is less naturally separated for pure image-image
    comparison (see the calibration note below).
- **Deep Rotation** — optional 360°, 1°-step rotation sweep to catch
  duplicates that were simply rotated. Runs for every pair when enabled
  (the expensive part — rotating and re-hashing — only happens once per
  image, reused across every comparison against it, so it stays efficient
  without needing to skip any pairs).
- **MD5 exact-duplicate detection** — byte-identical files are flagged
  instantly and skip the more expensive visual comparison. The actual MD5
  hex digest for every image is available in the results and the Excel
  export, not just a yes/no flag.
- **Filename similarity** — a second, independent signal (configurable
  threshold, default 90%) comparing filenames via `difflib`.
- **Recursive folder scanning** with include/exclude glob filters
  (`*.jpg *.png`) and a live match-count preview.
- **Three ways to pick a folder** — type a path, browse it with an in-app
  server-side directory picker, or **drag a folder from your OS file
  manager straight onto the page** (browsers hide the real filesystem path
  of dropped files, so this uploads the contents to a server-side temp
  directory and treats it like any other scanned folder).
- **Real-time progress** — the progress bar and activity log update per
  image (not just per phase), showing exactly which file is being
  processed, with a self-correcting ETA. **Pause/Resume and Stop**
  controls take effect at the next checkpoint.
- **Activity log** — a collapsible panel showing a timestamped trace of
  everything a job did: dependency checks, scan counts, per-phase
  milestones, pause/resume/stop events, model loading, a raw
  similarity-distribution readout for CLIP/DINOv2 jobs (useful for tuning
  calibration against your own photos), and a final summary — also
  mirrored to the server's stdout/log via the standard `logging` module.
- **Summary report** after every comparison — images scanned, pairs
  compared, matches found, exact duplicates, similarity range, and elapsed
  time.
- **Export to Excel** — every comparison can be downloaded as a `.xlsx`
  workbook with a Summary sheet, a full Matches sheet (file names, full
  paths, relative paths, MD5 hashes, similarity scores, exact-duplicate
  flags, rotation angles), and the complete Activity Log — auto-filterable
  columns, frozen header rows, ready to open in Excel or Google Sheets.
- **Mirror-pair deduplication** — comparing a folder against itself (or
  two overlapping folders) no longer reports the same underlying file
  pair twice with the "from A / from B" roles swapped.
- **Side-by-side full-resolution preview** for manual visual verification,
  with keyboard navigation (← → to flip between matches, Esc to close).

## Quickstart

### Windows — one-click launchers

No manual venv setup needed — both scripts create/reuse a `.venv`, run `pip
install` on every launch (fast and idempotent when nothing's changed —
typically 1-3 seconds), and then start the server at `http://127.0.0.1:8000`.

**Command Prompt:**
```bat
run.bat
```

**PowerShell:**
```powershell
.\run.ps1
```

Useful flags for `run.ps1`:
```powershell
.\run.ps1 -WithDino           # install the DINOv2 extra (recommended)
.\run.ps1 -WithClip           # install the CLIP extra
.\run.ps1 -Port 8080          # serve on a different port
.\run.ps1 -Reload             # auto-restart on code changes (dev)
.\run.ps1 -ForceReinstall     # force pip to reinstall all packages from scratch
```
`run.bat` supports `--with-dino` and `--with-clip` for the same purpose.

If PowerShell blocks the script with an "execution policy" error, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS / Linux / manual setup

```bash
git clone https://github.com/your-org/image-compare.git
cd image-compare
python -m venv .venv && source .venv/bin/activate
pip install -e .
imagecompare
# → serving on http://127.0.0.1:8000
```

Or run directly with uvicorn (useful for development, enables auto-reload):

```bash
pip install -e ".[dev]"
uvicorn imagecompare.main:app --reload
```

Open `http://127.0.0.1:8000`, enter (drag-and-drop, or browse to) two
folder paths, adjust settings, and click **Run comparison**.

## Optional: DINOv2 or CLIP ("visual"/"semantic" modes)

Both pull in `torch` (plus `transformers` for DINOv2, or `open_clip_torch`
for CLIP), which are heavy, so they're kept as optional extras:

```bash
pip install -e ".[dino]"   # recommended: smaller, purpose-built for this task
pip install -e ".[clip]"   # optional second opinion / text-aligned matching
```

If neither is installed, Hash mode still works fully — selecting DINOv2 or
CLIP in the UI shows a live readiness check (dependencies installed?
weights already cached?) via `GET /api/dino-status` / `GET /api/clip-status`,
and `POST /api/compare` rejects a mode's requests up front with a clear,
actionable error if its extra isn't installed, rather than letting a job
fail partway through.

The first time either mode runs, it needs to download its model weights
(DINOv2-small: roughly 85-90MB; CLIP ViT-B/32: several hundred MB). There's
no version-stable hook into these libraries' internal download progress, so
instead of trying to intercept exact byte counts, the job's progress panel
and activity log show periodic heartbeat updates (elapsed time, plus how
much the weights cache directory has grown) while the download/load is in
flight — so the job never looks frozen during a large first-time download,
even though the percentage isn't exact.

**A note on similarity score calibration:** raw cosine similarity between
learned image embeddings does not spread evenly across the full range the
way hash-mode's hamming-distance percentage does — completely unrelated
photos routinely score well above 0 raw cosine similarity (a documented
property of these embedding spaces, more pronounced for CLIP than DINOv2).
A naive linear rescale of that would show unrelated images as deceptively
"similar". Both `clip_embed.cosine_similarity()` and
`dino_embed.cosine_similarity()` correct for this with a calibration
baseline (`clip_similarity_baseline` / `dino_similarity_baseline` in
`config.py`, defaulting to `0.5` and `0.3` respectively): scores at or
below the baseline map to 0%, and a perfect match (1.0) maps to 100%. These
defaults are reasonable approximations for natural photos, but the ideal
value can vary by image domain — **every CLIP/DINOv2 job logs the actual
observed raw similarity distribution** (min / p25 / p50 / p75 / max) for
that run, so you can tune these constants against real data from your own
photos rather than guessing.

## How it works

```
scan folder(s)                 →  respects recursive + include/exclude filters
   │
   ├─ MD5 per image             →  exact duplicates flagged as 100% instantly
   ├─ perceptual hash / CLIP / DINOv2 signature per image (one at a time,
   │  progress reported incrementally — not "compute everything, then report")
   │
   ▼
[optional] Deep Rotation:
   every folder-B image is pre-rotated through 360° once (reused for every
   folder-A comparison against it, so the expensive step never repeats)
   │
   ▼
N × M similarity matrix        →  hamming distance (hash) or calibrated
   │                                cosine similarity (CLIP / DINOv2)
   ▼
mirror-pair deduplication      →  when folders overlap, the same real pair
   │                                otherwise appears twice with roles swapped
   ▼
filename similarity attached to every surviving pair
   │
   ▼
filtered by visual threshold, sorted descending, thumbnails generated
   │
   ▼
summary + activity log + optional Excel export
```

See the module docstrings in `src/imagecompare/core/` for the detailed
rationale behind each step.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/browse?path=` | List subdirectories for the folder picker |
| `POST` | `/api/upload-folder` | Drag-and-drop folder upload → server-side path |
| `POST` | `/api/scan-preview` | Live count of images matching current filters |
| `GET` | `/api/clip-status` | CLIP mode readiness (deps installed? weights cached?) |
| `GET` | `/api/dino-status` | DINOv2 mode readiness (deps installed? weights cached?) |
| `POST` | `/api/compare` | Start a background comparison job → `job_id` |
| `GET` | `/api/jobs/{id}` | Poll progress (status, percent, ETA) |
| `GET` | `/api/jobs/{id}/logs` | Timestamped activity log for the job |
| `POST` | `/api/jobs/{id}/pause` | Pause a running job |
| `POST` | `/api/jobs/{id}/resume` | Resume a paused job |
| `POST` | `/api/jobs/{id}/stop` | Cancel a running/paused job |
| `GET` | `/api/jobs/{id}/results` | Fetch ranked matches once done |
| `GET` | `/api/jobs/{id}/export.xlsx` | Download the full report as an Excel workbook |
| `GET` | `/api/thumb?path=` | Cached thumbnail for an image |
| `GET` | `/api/image?job_id=&path=` | Full-resolution image, path-validated |

Full request/response schemas are in `src/imagecompare/models.py`.

## Development

```bash
pip install -e ".[dev]"

pytest                          # run the test suite (182 tests)
pytest --cov=src/imagecompare   # with coverage
ruff check src/ tests/          # lint
black src/ tests/               # format
mypy src/                       # type check
```

Test images are generated on the fly with Pillow (see `tests/conftest.py`) —
no binary fixtures are checked into the repo. CLIP/DINOv2-dependent code
paths are tested via dependency injection (fake loaders, synthetic
embedding vectors) so the full suite runs without installing `torch`.

CI runs the same checks on Python 3.10–3.12 via GitHub Actions
(`.github/workflows/ci.yml`).

## Project layout

```
src/imagecompare/
├── core/               # framework-agnostic business logic
│   ├── scanner.py          # folder walking, recursive + include/exclude filters
│   ├── hashing.py          # perceptual hash + Deep Rotation sweep
│   ├── md5util.py          # exact-duplicate detection
│   ├── filename_sim.py     # filename similarity scoring
│   ├── clip_embed.py       # optional CLIP embeddings + calibration
│   ├── dino_embed.py       # optional DINOv2 embeddings + calibration
│   ├── downloader.py       # generic streaming download w/ progress (model weights)
│   ├── upload.py           # drag-and-drop upload helpers (path sanitization, cleanup)
│   ├── thumbnails.py       # thumbnail generation + caching
│   └── comparator.py       # orchestrates the full pipeline above
├── jobs/
│   └── manager.py          # background job manager: progress, ETA, pause/cancel, logs
├── api/
│   ├── routes.py           # FastAPI routes
│   └── xlsx_export.py      # Excel report builder
├── web/                # Jinja2 template + static CSS/JS (no build step)
├── config.py           # settings (env-overridable via IMAGECOMPARE_*)
├── models.py           # pydantic schemas
└── main.py             # FastAPI app entry point
tests/                  # pytest suite, one file per module (182 tests)
run.bat / run.ps1       # one-click Windows launchers
```

## Notes & limitations

- Folder paths are read directly from the machine running the server —
  this is designed to run locally, not as a public multi-tenant service.
  `/api/image` restricts previews to files inside the two folders that were
  actually scanned for a given job.
- Comparison is `O(N × M)`; this is fine up to a few thousand images per
  folder. Very large libraries would benefit from an approximate-nearest-
  neighbor index (e.g. FAISS) — not implemented here, flagged as a possible
  future improvement.
- Deep Rotation is only available in Hash mode.
- Drag-and-drop folder upload copies file bytes to the server over
  loopback HTTP (browsers can't expose real filesystem paths of dropped
  files); for very large folders, typing/browsing to the path directly
  avoids that copy.

## Contributing

Bug reports, feature requests, and PRs are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, guidelines, and what to run
before opening a PR.

## License

MIT — see [LICENSE](LICENSE).
