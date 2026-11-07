"""Where KaggleUI is running: Kaggle notebooks, Colab, or a plain machine.

The adapter only reports what the host offers; paths and features stay identical, so a Colab session
behaves like a local one minus Kaggle's persistent output and accelerator toggle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Protocol

from . import secretstore
from .config import ON_COLAB, ON_KAGGLE

HostName = Literal["kaggle", "colab", "local"]

@dataclass
class Capabilities:
    secrets: Literal["kaggle", "colab", "env", "none"]
    drive: bool
    accelerator_toggle: bool
    persistent_disk: bool

@dataclass
class SessionInfo:
    host: HostName
    run_type: str | None
    session_limit_s: float | None

class HostAdapter(Protocol):
    name: HostName

    def capabilities(self) -> Capabilities: ...

    def session_info(self) -> SessionInfo: ...

    def load_secrets(self, refresh: bool = False) -> dict[str, bool]: ...

    def startup_guidance(self) -> list[str]: ...

class KaggleHost:
    name: HostName = "kaggle"

    def capabilities(self) -> Capabilities:
        return Capabilities(secrets="kaggle", drive=True, accelerator_toggle=True, persistent_disk=True)

    def session_info(self) -> SessionInfo:
        return SessionInfo(host=self.name, run_type=os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
                           session_limit_s=9 * 3600 if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive"
                           else 12 * 3600)

    def load_secrets(self, refresh: bool = False) -> dict[str, bool]:
        return secretstore.load_attached_secrets(refresh)

    def startup_guidance(self) -> list[str]:
        return ["Kaggle sessions end when the notebook stops; /kaggle/working is saved as output."]

class ColabHost:
    name: HostName = "colab"

    def capabilities(self) -> Capabilities:
        return Capabilities(secrets="colab", drive=True, accelerator_toggle=False, persistent_disk=False)

    def session_info(self) -> SessionInfo:
        return SessionInfo(host=self.name, run_type=os.environ.get("COLAB_RELEASE_TAG"), session_limit_s=None)

    def load_secrets(self, refresh: bool = False) -> dict[str, bool]:
        return secretstore.load_attached_secrets(refresh)

    def startup_guidance(self) -> list[str]:
        return ["Colab runtimes are interactive: everything under /tmp and /kaggleui is lost on disconnect.",
                "Mount Drive manually or attach a storage connection before converting large models."]

class LocalHost:
    name: HostName = "local"

    def capabilities(self) -> Capabilities:
        return Capabilities(secrets="env", drive=False, accelerator_toggle=False, persistent_disk=False)

    def session_info(self) -> SessionInfo:
        return SessionInfo(host=self.name, run_type=None, session_limit_s=None)

    def load_secrets(self, refresh: bool = False) -> dict[str, bool]:
        return secretstore.load_attached_secrets(refresh)

    def startup_guidance(self) -> list[str]:
        return []

def detect() -> HostAdapter:
    if ON_KAGGLE:
        return KaggleHost()
    if ON_COLAB:
        return ColabHost()
    return LocalHost()

