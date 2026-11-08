"""Profiles for the local client.

Four independent kinds of configuration are kept apart on purpose: an instance (URL + management
key), a workspace (repository), a notebook (Kaggle kernel source) and a Kaggle credential. A profile
is only a set of references. Nothing here rotates Kaggle accounts or pools quota.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

CONFIG_VERSION = 1
SECTIONS = ("instances", "workspaces", "notebooks", "credentials")
PROFILE_KEYS = ("instance", "workspace", "notebook", "credential")

class ProfileError(Exception):
    pass

def config_dir() -> Path:
    override = os.environ.get("KAGGLEUI_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "kaggleui"

def _write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

class ProfileStore:
    def __init__(self, path: Path | None = None):
        self.path = path or (config_dir() / "config.json")
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        blank = {"version": CONFIG_VERSION, "active": "default", "profiles": {},
                 **{section: {} for section in SECTIONS}}
        if not self.path.exists():
            return blank
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise ProfileError(f"cannot read {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ProfileError(f"{self.path} is not a profile document")
        if int(data.get("version") or 1) > CONFIG_VERSION:
            raise ProfileError("profile file was written by a newer kui; upgrade before editing it")
        merged = {**blank, **data}
        for section in (*SECTIONS, "profiles"):
            if not isinstance(merged.get(section), dict):
                merged[section] = {}
        return merged

    def save(self) -> None:
        _write_private(self.path, self.data)

    def active_name(self, override: str | None = None) -> str:
        return override or str(self.data.get("active") or "default")

    def profile(self, name: str | None = None) -> dict[str, Any]:
        key = self.active_name(name)
        profile = self.data["profiles"].get(key)
        if profile is None:
            if name:
                raise ProfileError(f"no profile named {name}")
            profile = {"instance": "local"}
            self.data["profiles"][key] = profile
            self.save()
        return profile

    def resolve(self, name: str | None = None) -> dict[str, Any]:
        """A profile with each referenced section expanded; secrets are present here only in memory."""
        key = self.active_name(name)
        profile = self.profile(key)
        out: dict[str, Any] = {"profile": key}
        for section in SECTIONS:
            ref = profile.get(section.rstrip("s") if section != "credentials" else "credential")
            out[section.rstrip("s") if section != "credentials" else "credential"] = (
                dict(self.data[section].get(ref, {})) if ref else {})
        return out

    def profile_names(self) -> list[str]:
        return sorted(self.data["profiles"])

    def add_profile(self, name: str, *, instance: str | None = None, workspace: str | None = None,
                    notebook: str | None = None, credential: str | None = None) -> dict:
        entry = self.data["profiles"].get(name, {})
        for key, value in (("instance", instance), ("workspace", workspace),
                           ("notebook", notebook), ("credential", credential)):
            if value:
                if not self.data[key + "s"].get(value):
                    raise ProfileError(f"no {key} named {value}")
                entry[key] = value
        self.data["profiles"][name] = entry
        self.save()
        return entry

    def set_section(self, section: str, name: str, values: dict[str, Any], *, make_active: bool = False) -> None:
        if section not in SECTIONS:
            raise ProfileError(f"unknown profile section {section}")
        clean = {k: v for k, v in values.items() if v not in (None, "")}
        current = self.data[section].get(name, {})
        current.update(clean)
        self.data[section][name] = current
        if make_active:
            self.data["active"] = name
        self.save()

    def set_active(self, name: str) -> None:
        if not self.data["profiles"].get(name):
            raise ProfileError(f"no profile named {name}")
        self.data["active"] = name
        self.save()

    def public(self) -> dict[str, Any]:
        def redact_rows(section: str) -> dict[str, dict]:
            out = {}
            for name, values in self.data[section].items():
                out[name] = {k: ("***" if "token" in k or "key" == k else v)
                             for k, v in values.items() if k not in {"token"}}
            return out

        return {"path": str(self.path), "active": self.data.get("active"),
                "profiles": self.data["profiles"],
                "instances": redact_rows("instances"), "workspaces": self.data["workspaces"],
                "notebooks": self.data["notebooks"], "credentials": redact_rows("credentials")}

