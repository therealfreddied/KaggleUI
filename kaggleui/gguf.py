"""Minimal GGUF metadata reader.

Only the header and the metadata key/value table are parsed; tensors are never touched, so this is
cheap enough to run before a model is launched. MoE detection reads the real metadata instead of
guessing from a repo or file name.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)

# GGUF value type ids -> struct format (None = variable length).
_SCALARS: dict[int, str] = {
    0: "<B",   # uint8
    1: "<b",   # int8
    2: "<H",   # uint16
    3: "<h",   # int16
    4: "<I",   # uint32
    5: "<i",   # int32
    6: "<f",   # float32
    8: "str",  # string
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}
_BOOL = 7
_ARRAY = 9


class GGUFError(Exception):
    """The file is not a readable GGUF."""


def _read_scalar(f: BinaryIO, vtype: int) -> Any:
    fmt = _SCALARS.get(vtype)
    if fmt is None:
        raise GGUFError(f"unknown GGUF value type {vtype}")
    if fmt == "str":
        return _read_string(f)
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))[0]


def _read_string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    if length > 1 << 24:
        raise GGUFError(f"implausible string length {length}")
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype == _BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == _ARRAY:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if count > 1 << 24:
            raise GGUFError(f"implausible array length {count}")
        return [_read_value(f, etype) for _ in range(count)]
    return _read_scalar(f, vtype)


def read_metadata(path: Path) -> dict[str, Any]:
    """Return the metadata key/value table of a GGUF file."""
    path = Path(path)
    try:
        with path.open("rb") as f:
            head = f.read(4)
            if head != MAGIC:
                raise GGUFError(f"{path.name} is not a GGUF file")
            (version,) = struct.unpack("<I", f.read(4))
            if version not in SUPPORTED_VERSIONS:
                raise GGUFError(f"unsupported GGUF version {version}")
            struct.unpack("<Q", f.read(8))  # tensor count
            (kv_count,) = struct.unpack("<Q", f.read(8))
            if kv_count > 1 << 20:
                raise GGUFError(f"implausible metadata count {kv_count}")
            meta: dict[str, Any] = {}
            for _ in range(kv_count):
                key = _read_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                meta[key] = _read_value(f, vtype)
            return meta
    except (OSError, struct.error) as exc:
        raise GGUFError(f"unreadable GGUF metadata in {path.name}: {exc}") from exc


def architecture(meta: dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    value = meta.get("general.architecture")
    return str(value) if value else None


def context_length(meta: dict[str, Any] | None) -> int | None:
    """Largest declared context window in GGUF metadata (``<arch>.context_length``)."""
    if not meta:
        return None
    windows = [v for k, v in meta.items()
               if k.endswith(".context_length") and isinstance(v, int) and not isinstance(v, bool) and v > 0]
    return max(windows) if windows else None


def is_moe(meta: dict[str, Any] | None) -> bool:
    """True when the metadata declares experts — the llama.cpp ``--cpu-moe`` precondition."""
    if not meta:
        return False
    return any(isinstance(k, str) and k.endswith(".expert_count") and isinstance(v, int) and v > 0
               for k, v in meta.items())


def expert_count(meta: dict[str, Any] | None) -> int | None:
    if not meta:
        return None
    for key, value in meta.items():
        if isinstance(key, str) and key.endswith(".expert_count") and isinstance(value, int):
            return value
    return None


def layer_count(meta: dict[str, Any] | None) -> int | None:
    arch = architecture(meta)
    if not meta or not arch:
        return None
    value = meta.get(f"{arch}.block_count")
    return int(value) if isinstance(value, int) else None


def has_magic(path: Path) -> bool:
    try:
        with Path(path).open("rb") as f:
            return f.read(4) == MAGIC
    except OSError:
        return False
