"""Base-model resolution for training: full safetensors only, never a GGUF."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..engines.download import hf_download

SAFE_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*", "*.py"]
GGUF_HINT = "Use the GGUF export, not training"

def _is_gguf_variant(variant: str | None) -> bool:
    if not variant:
        return False
    upper = variant.upper()
    return upper.endswith(".GGUF") or upper in {"GGUF", "Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M",
                                                "Q5_K_S", "Q6_K", "Q8_0", "F16"}

async def resolve_base(base: Any, paths: Any, *, log=print, progress=None, dep_id: str = "train") -> Path:
    """Return a local directory holding the fp16/bf16 safetensors for ``base`` (a ``ModelRef``)."""
    variant = getattr(base, "variant", None)
    if _is_gguf_variant(variant):
        from ..engines.context import DeployError

        raise DeployError(GGUF_HINT)
    source = getattr(base, "source", "hf")
    repo = getattr(base, "id", "")
    if source not in ("hf", "curated"):
        from ..engines.context import DeployError

        raise DeployError(f"Training needs an HF base model; {source} is not supported")
    shim = SimpleNamespace(
        paths=paths, dep_id=dep_id, served_name=dep_id, gpus=[], processes=[],
        log=log, progress=progress or (lambda *a, **k: None),
        env=lambda extra=None: dict(extra or {}),
    )
    # hf_download writes into the shared HF cache; the worker loads from that snapshot, so no copy.
    snap = Path(await hf_download(shim, repo, allow_patterns=SAFE_PATTERNS))
    if not (snap / "config.json").exists():
        raise RuntimeError(f"{repo} has no config.json; is it a full transformers repo?")
    return snap

def latest_checkpoint(scratch: Path) -> Path | None:
    """Highest-step ``checkpoint-N`` under ``scratch``; resume needs the newest one."""
    candidates: list[tuple[int, Path]] = []
    if not scratch.exists():
        return None
    for path in scratch.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists():
            candidates.append((step, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None
