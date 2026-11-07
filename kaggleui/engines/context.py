"""Per-deployment context handed to engines: logging, progress, GPU pinning, process tracking."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Paths
from ..proc import ManagedProcess
from ..system.probe import HardwareProfile
from ..workbench.jobs import JobStore
from ..workbench.store import WorkbenchStore

ProgressFn = Callable[[str, "float | None", str], None]


@dataclass
class EngineContext:
    paths: Paths
    dep_id: str
    served_name: str
    gpus: list[int]
    hw: HardwareProfile
    log: Callable[[str], None]
    progress: ProgressFn
    internal_key: str
    main_port: int
    processes: list[ManagedProcess] = field(default_factory=list)
    jobs: JobStore | None = None
    workbench: WorkbenchStore | None = None

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": os.environ.get("HF_HOME", str(self.paths.hf_home)),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpus)
        if os.environ.get("HF_TOKEN"):
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]
        env.update(extra or {})
        return env

    def log_path(self, name: str) -> str:
        return str(self.paths.logs / f"{self.dep_id}-{name}.log")


class DeployError(Exception):
    """A deployment failed with a message meant for the user."""
