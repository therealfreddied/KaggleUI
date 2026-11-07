"""Durable workbench state: deployment sets, storage connections, jobs and artifacts.

Everything here survives a server restart within the session: the stores write JSON atomically next
to ``state.json`` and long tasks live in a sqlite job table instead of in-memory dicts.
"""

from __future__ import annotations

from .store import WorkbenchStore

__all__ = ["WorkbenchStore"]
