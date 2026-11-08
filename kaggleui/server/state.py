"""Process-wide state shared by every route."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..catalog.imports import ModelImports
from ..catalog.service import CatalogService
from ..config import Paths
from ..deploy.manager import DeploymentManager
from ..hosts import HostAdapter, detect
from ..interfaces.manager import InterfaceManager
from ..optimize import Optimizer
from ..schemas import Preset, Settings
from ..storage import StorageManager
from ..system.probe import Probe
from ..tunnels.manager import TunnelManager
from ..workbench.jobs import JobStore
from ..workbench.registry import ArtifactRegistry
from ..workbench.residency import Residency
from ..workbench.store import WorkbenchStore
from .auth import Auth
from .events import EventBus
from .management import ManagementTokens
from .updates import Updates


@dataclass
class AppState:
    paths: Paths
    settings: Settings
    port: int
    listen_host: str = "127.0.0.1"
    mock: bool = False
    fake_hw: str | None = None
    preset: Preset | None = None
    secrets: dict[str, bool] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    video_jobs: dict[str, dict] = field(default_factory=dict)
    world_jobs: dict[str, dict] = field(default_factory=dict)

    auth: Auth = field(init=False)
    bus: EventBus = field(init=False)
    probe: Probe = field(init=False)
    catalog: CatalogService = field(init=False)
    imports: ModelImports = field(init=False)
    tunnels: TunnelManager = field(init=False)
    manager: DeploymentManager = field(init=False)
    interfaces: InterfaceManager = field(init=False)
    updates: Updates = field(init=False)
    workbench: WorkbenchStore = field(init=False)
    jobs: JobStore = field(init=False)
    residency: Residency = field(init=False)
    artifacts: ArtifactRegistry = field(init=False)
    storage: StorageManager = field(init=False)
    optimize: Optimizer = field(init=False)
    host: HostAdapter = field(init=False)
    management: ManagementTokens = field(init=False)
    auto_backup_task: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.auth = Auth(self.paths.home / "session.key")
        self.bus = EventBus()
        self.probe = Probe(self.paths, fake=self.fake_hw)
        self.imports = ModelImports(self.paths.models / "imports", self.paths.envs)
        self.catalog = CatalogService(self.probe.snapshot, imports_root=self.imports.root)
        self.tunnels = TunnelManager(self.paths, on_change=lambda t: self.bus.publish("tunnel", t.info()))
        self.workbench = WorkbenchStore(self.paths.home / "workbench")
        self.workbench.load()
        self.jobs = JobStore(self.paths.state_file.parent / "jobs.sqlite3")
        self.jobs.recover_interrupted()
        self.manager = DeploymentManager(self)
        self.manager.restore()
        self.residency = Residency(self)
        self.artifacts = ArtifactRegistry(self.paths.models / "artifacts")
        self.storage = StorageManager(self.paths, self.workbench)
        self.optimize = Optimizer(self)
        self.interfaces = InterfaceManager(self)
        self.updates = Updates(self)
        self.host = detect()
        self.management = ManagementTokens(self.paths.home / "management.json")

    def main_url(self) -> str:
        t = self.tunnels.get("main")
        return t.url.rstrip("/") if t and t.url else f"http://127.0.0.1:{self.port}"

    def local_url(self) -> str:
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(self.listen_host, self.listen_host)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"
