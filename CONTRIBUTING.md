# Contributing

Thanks for considering a contribution! This is a small project, so the bar
is mostly: tests pass, lint is clean, and the change is scoped.

## Setup

```bash
git clone https://github.com/your-org/image-compare.git
cd image-compare
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a PR

```bash
ruff check src/ tests/
black src/ tests/
mypy src/
pytest --cov=src/imagecompare
```

CI runs the same checks on Python 3.10–3.12 (mypy is currently
non-blocking in CI while type coverage is filled in — please still try to
keep new code typed).

## Guidelines

- **Tests are required** for new behavior. Fixtures generate images with
  Pillow at test time (`tests/conftest.py`) rather than checking in binary
  files — please follow that pattern for new fixtures.
- **Keep `core/` framework-agnostic.** Modules under `src/imagecompare/core/`
  shouldn't import FastAPI or anything HTTP-related — they should work as a
  standalone library. HTTP concerns live in `api/`.
- **Heavy optional dependencies stay optional.** CLIP mode (`torch`,
  `open_clip_torch`) and DINOv2 mode (`torch`, `transformers`) are imported
  lazily inside `core/clip_embed.py` / `core/dino_embed.py` so the rest of
  the app works without them installed. Follow the same pattern if you add
  another optional backend — both modules are a good template (dependency
  check, status snapshot, calibration, heartbeat-based load progress).
- **Commit messages**: short, imperative subject line (`Add exclude-pattern
  precedence test`, not `Added` or `Adding`).
- **One logical change per PR** where reasonable — makes review and
  bisecting easier.

## Reporting bugs / requesting features

Open an issue with:
- What you expected vs. what happened
- Steps to reproduce (folder structure, settings used, if relevant)
- Python version and OS

## Code of conduct

Be respectful and constructive. Assume good faith.
