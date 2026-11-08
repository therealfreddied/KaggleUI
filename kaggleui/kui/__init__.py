"""Local ``kui`` client: profiles, management API, Kaggle adapter and a small TUI."""

from __future__ import annotations

from .client import KuiError, ManagementClient
from .config import ProfileStore

__all__ = ["KuiError", "ManagementClient", "ProfileStore"]
