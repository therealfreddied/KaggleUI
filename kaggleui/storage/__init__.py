"""Remote destinations for artifacts (local disk, rclone, Hugging Face, GitHub releases)."""

from __future__ import annotations

from .base import RemoteEntry, StorageError, StorageProvider
from .manager import StorageManager, build_provider, redact

__all__ = ["RemoteEntry", "StorageError", "StorageManager", "StorageProvider", "build_provider", "redact"]
