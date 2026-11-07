"""Choose which GGUF files a repo needs and where they land.

Pure functions over the Hub's ``siblings`` list, so the rules are testable without network access.
"""

from __future__ import annotations

import re
from typing import Any

_SHARD = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
_QUANT = re.compile(r"(IQ\d+_[A-Z0-9]+|Q\d+_K_[MSL]|Q\d+_K|Q\d+_\d|Q\d+|BF16|F16|F32)")
PREFERRED_QUANTS = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q4_0")


def quant_of(filename: str) -> str | None:
    matches = _QUANT.findall(filename)
    return matches[-1].upper() if matches else None


def _size(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("size") or 0)
    except (TypeError, ValueError):
        return 0


def is_projector(path: str) -> bool:
    lowered = path.lower()
    return "mmproj" in lowered or "projector" in lowered


def candidates(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Single-file GGUF candidates (shards are grouped separately)."""
    return [f for f in files if str(f.get("rfilename", "")).lower().endswith(".gguf")
            and not is_projector(str(f.get("rfilename", "")))
            and not _SHARD.match(str(f.get("rfilename", "")))]


def shard_groups(files: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Complete shard groups keyed by their shared base name; incomplete groups are dropped."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in files:
        match = _SHARD.match(str(entry.get("rfilename", "")))
        if not match:
            continue
        groups.setdefault(match.group("base").lower(), []).append(entry)
    complete: dict[str, list[dict[str, Any]]] = {}
    for base, members in groups.items():
        total = {int(_SHARD.match(str(m["rfilename"])).group("total")) for m in members}  # type: ignore[union-attr]
        if len(total) != 1:
            continue
        index = {int(_SHARD.match(str(m["rfilename"])).group("index")) for m in members}  # type: ignore[union-attr]
        if index != set(range(1, next(iter(total)) + 1)):
            continue
        complete[base] = sorted(members, key=lambda m: int(_SHARD.match(str(m["rfilename"])).group("index")))  # type: ignore[union-attr]
    return complete


def pick(files: list[dict[str, Any]], *, quant: str | None = None) -> list[dict[str, Any]]:
    """Return the files to download, in order. Raises ``ValueError`` when nothing fits."""
    groups = shard_groups(files)
    singles = candidates(files)

    def group_quant(members: list[dict[str, Any]]) -> str | None:
        return quant_of(str(members[0]["rfilename"]))

    options: list[tuple[str, str | None, list[dict[str, Any]]]] = []
    for members in groups.values():
        options.append(("sharded", group_quant(members), members))
    for entry in singles:
        options.append(("single", quant_of(str(entry["rfilename"])), [entry]))
    if not options:
        raise ValueError("This repo has no GGUF file to download")

    if quant:
        wanted = quant.upper()
        matching = [o for o in options if o[1] == wanted]
        if not matching:
            available = sorted({o[1] for o in options if o[1]})
            raise ValueError(f"quant {quant!r} is not in this repo; available: {', '.join(available) or 'unknown'}")
        options = matching
    else:
        for preferred in PREFERRED_QUANTS:
            matching = [o for o in options if o[1] == preferred]
            if matching:
                options = matching
                break

    options.sort(key=lambda o: (o[0] != "sharded", sum(_size(m) for m in o[2])))
    return options[0][2]


def file_list(files: list[dict[str, Any]], *, quant: str | None = None) -> list[dict[str, str]]:
    """The ``{path, size}`` list :func:`kaggleui.llamacpp.fetch` expects."""
    return [{"path": str(f["rfilename"]), "size": str(_size(f))} for f in pick(files, quant=quant)]


def projector(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in files:
        if str(entry.get("rfilename", "")).lower().endswith(".gguf") and is_projector(str(entry.get("rfilename", ""))):
            return entry
    return None


def needs_split(size_bytes: int, free_bytes: int) -> bool:
    """GGUF shards keep every file under 1 GB-ish; a single file needs room for the whole thing."""
    return size_bytes > max(free_bytes, 0)
