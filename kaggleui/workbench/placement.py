"""Resolve a :class:`~kaggleui.schemas.Placement` against the machine and validate it.

The resolver is engine-agnostic: llama.cpp compiles the result to argv, any other engine only consumes
the fields it understands. Errors here become ``DeployError`` messages the panel shows verbatim.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..schemas import DeploymentSpec, Placement
from ..system.probe import HardwareProfile

KV_TYPES = ("f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
_KV_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass
class ResolvedPlacement:
    gpus: list[int] = field(default_factory=list)
    gpu_layers: int = 0
    split_mode: str = "layer"
    tensor_split: list[float] | None = None
    main_gpu: int | None = None
    cpu_moe_layers: int | None = None
    cpu_moe: bool = False
    flash_attn: str = "auto"
    kv_k: str | None = None
    kv_v: str | None = None
    kv_offload: bool = True
    mmap: bool = True
    mlock: bool = False
    threads: int = 1
    threads_batch: int | None = None
    parallel: int = 1
    tensor_overrides: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "gpus": list(self.gpus), "gpu_layers": self.gpu_layers, "split_mode": self.split_mode,
            "tensor_split": list(self.tensor_split) if self.tensor_split else None,
            "main_gpu": self.main_gpu, "cpu_moe_layers": self.cpu_moe_layers, "cpu_moe": self.cpu_moe,
            "flash_attn": self.flash_attn, "kv_k": self.kv_k, "kv_v": self.kv_v,
            "kv_offload": self.kv_offload, "mmap": self.mmap, "mlock": self.mlock,
            "threads": self.threads, "threads_batch": self.threads_batch, "parallel": self.parallel,
            "tensor_overrides": list(self.tensor_overrides), "notes": list(self.notes),
        }


def _placement(spec: DeploymentSpec) -> Placement:
    return spec.placement or Placement()


def _gpu_indices(hw: HardwareProfile) -> list[int]:
    return [g.index for g in hw.gpus]


def validate(spec: DeploymentSpec, hw: HardwareProfile, *, arch: dict[str, Any] | None) -> list[str]:
    """Return human-readable errors; an empty list means the placement can be deployed."""
    p = _placement(spec)
    errors: list[str] = []
    present = _gpu_indices(hw)

    if p.tensor_split is not None:
        if len(p.tensor_split) != len(p.devices or present):
            errors.append(f"tensor_split has {len(p.tensor_split)} values but the placement uses "
                          f"{len(p.devices or present)} GPUs")
        elif any(v < 0 for v in p.tensor_split) or sum(p.tensor_split) <= 0:
            errors.append("tensor_split values must be >= 0 and sum to more than 0")

    devices = p.devices if p.devices is not None else present
    for index in devices:
        if index not in present:
            errors.append(f"GPU {index} not present on this machine")
    if p.devices is not None:
        if p.devices != sorted(p.devices):
            errors.append("devices must be listed in ascending order")
        if len(set(p.devices)) != len(p.devices):
            errors.append("devices must not repeat")
    if len(devices) < 2 and p.tensor_split is not None:
        errors.append("tensor_split needs at least two devices")
    if p.main_gpu is not None and p.main_gpu not in devices:
        errors.append(f"main_gpu {p.main_gpu} is not one of the placement's devices")

    if p.cpu_moe and p.cpu_moe_layers is not None:
        errors.append("cpu_moe and cpu_moe_layers are mutually exclusive")
    if p.cpu_moe or p.cpu_moe_layers is not None:
        if arch is None:
            errors.append("CPU MoE offload needs the GGUF metadata; download the model first")
        else:
            from ..gguf import is_moe, layer_count

            if not is_moe(arch):
                errors.append("This GGUF is not a mixture-of-experts model")
            elif p.cpu_moe_layers is not None:
                layers = layer_count(arch)
                if layers is not None and not 1 <= p.cpu_moe_layers <= layers:
                    errors.append(f"cpu_moe_layers must be between 1 and {layers}")

    if p.mlock and not p.mmap:
        errors.append("mlock requires mmap")
    if p.parallel < 1:
        errors.append("parallel must be at least 1")
    for value in (p.kv_cache_type_k, p.kv_cache_type_v):
        if value and not _KV_RE.match(value):
            errors.append(f"invalid KV cache type '{value}'")
    for value in p.tensor_overrides:
        if not value or "=" not in value:
            errors.append(f"override_tensor '{value}' must look like 'ATTN_V=CPU'")
    return errors


def resolve(spec: DeploymentSpec, hw: HardwareProfile, *, arch: dict[str, Any] | None,
            threads_default: int | None = None, assigned: list[int] | None = None) -> ResolvedPlacement:
    """Compute the effective placement. Assumes :func:`validate` returned no errors.

    ``assigned`` is the GPU set the deployment actually got (``CUDA_VISIBLE_DEVICES``); llama.cpp then
    numbers those devices from zero, so everything below is expressed in visible positions.
    """
    p = _placement(spec)
    present = _gpu_indices(hw)
    devices = list(p.devices) if p.devices is not None else list(assigned if assigned else present)
    out = ResolvedPlacement(gpus=devices)

    if p.main_gpu is not None and p.main_gpu in devices:
        out.main_gpu = devices.index(p.main_gpu)
    if len(devices) < 2:
        out.split_mode = "none"
        if p.split_mode != "none":
            out.notes.append("Single device: layer split disabled")
        if p.main_gpu is not None:
            out.notes.append("One device: it is the main GPU already")
    else:
        out.split_mode = p.split_mode
        if p.tensor_split:
            out.tensor_split = [round(float(v), 1) for v in p.tensor_split]
    if out.split_mode == "row" and out.tensor_split:
        out.notes.append("row split ignores tensor_split weights")

    if p.gpu_layers == "auto":
        out.gpu_layers = 999 if devices else 0
    else:
        out.gpu_layers = int(p.gpu_layers)
        if devices and out.gpu_layers == 0:
            out.notes.append("gpu_layers=0 keeps every layer on the CPU")

    out.cpu_moe = p.cpu_moe
    out.cpu_moe_layers = p.cpu_moe_layers
    out.tensor_overrides = list(p.tensor_overrides)
    out.flash_attn = p.flash_attn
    out.kv_k = p.kv_cache_type_k
    out.kv_v = p.kv_cache_type_v
    out.kv_offload = p.kv_offload
    out.mmap = p.mmap
    out.mlock = p.mlock
    out.parallel = max(1, p.parallel)
    if p.mlock and "simulated" in (hw.platform or ""):
        out.notes.append("mlock is a no-op on simulated hardware")

    cpus = threads_default or (os.cpu_count() or 4)
    out.threads = p.threads if p.threads else max(1, cpus // out.parallel)
    out.threads_batch = p.batch_threads
    if out.parallel > 1:
        out.notes.append(f"parallel={out.parallel} splits the context across slots")
    for value in (p.kv_cache_type_k, p.kv_cache_type_v):
        if value and value not in KV_TYPES:
            out.notes.append(f"KV cache type '{value}' is not a known llama.cpp type; passing it through")
    return out
