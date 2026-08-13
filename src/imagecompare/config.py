"""Central configuration and defaults for imagecompare.

Values here can be overridden via environment variables (prefixed
``IMAGECOMPARE_``) or a ``.env`` file, using pydantic-settings.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMAGECOMPARE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8000

    # --- scanning defaults ---
    default_extensions: tuple[str, ...] = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.webp",
        "*.gif",
        "*.tiff",
    )

    # --- comparison defaults ---
    default_visual_threshold: float = 70.0
    default_filename_threshold: float = 90.0
    default_hash_size: int = 16  # phash bit resolution (16 -> 256-bit hash)
    default_mode: str = "hash"  # "hash" | "clip" | "dino"
    clip_similarity_baseline: float = 0.5  # see clip_embed.cosine_similarity docstring
    dino_similarity_baseline: float = 0.3  # see dino_embed.cosine_similarity docstring

    # --- deep rotation ---
    deep_rotation_step_degrees: int = 1

    # --- thumbnails ---
    thumbnail_size: tuple[int, int] = (256, 256)
    thumbnail_dir_name: str = "thumbs"

    # --- misc ---
    max_preview_bytes: int = 25 * 1024 * 1024  # refuse to stream absurdly large files


settings = Settings()
