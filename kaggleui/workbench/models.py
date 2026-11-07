"""Workbench wire models.

The persisted shapes live in :mod:`kaggleui.schemas` because they are part of the API contract the SPA
mirrors by hand; defining them there also keeps ``Settings`` importable without a cycle. This module is
the single import point for workbench code and re-exports them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import (
    ArtifactFile,
    ArtifactManifest,
    CacheProfile,
    DeploymentSet,
    OptimizationProfile,
    Placement,
    StorageConnection,
    StorageSettings,
)

__all__ = [
    "ArtifactFile",
    "ArtifactManifest",
    "CacheProfile",
    "DeploymentSet",
    "OptimizationProfile",
    "Placement",
    "StorageConnection",
    "StorageSettings",
]


class DeploymentRecord(BaseModel):
    """A deployment spec the server can rebuild after a restart (llama-swap launch helper reads it)."""

    id: str
    spec: dict = Field(default_factory=dict)
    served_name: str = ""
    created_at: float = 0.0
    runtime: dict = Field(default_factory=dict)
