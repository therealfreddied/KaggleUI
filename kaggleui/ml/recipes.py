"""Training recipe: validated hyperparameters plus the exact worker command they produce."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..schemas import ModelRef

METHODS = ("lora", "qlora")
QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")
DEFAULTS = {
    "r": 16, "alpha": 32, "dropout": 0.05, "epochs": 1, "lr": 2e-4, "seq_length": 2048,
    "batch_size": 1, "gradient_accumulation_steps": 16, "seed": 42,
}
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class RecipeError(ValueError):
    """The recipe cannot be trained as written."""

class TrainingRun(BaseModel):
    schema_version: int = 1
    id: str = ""
    base: ModelRef
    dataset_artifact: str
    method: Literal["lora", "qlora"] = "lora"
    output_name: str = ""
    r: int = DEFAULTS["r"]
    alpha: int = DEFAULTS["alpha"]
    dropout: float = DEFAULTS["dropout"]
    epochs: float = DEFAULTS["epochs"]
    lr: float = DEFAULTS["lr"]
    seq_length: int = DEFAULTS["seq_length"]
    batch_size: int = DEFAULTS["batch_size"]
    gradient_accumulation_steps: int = DEFAULTS["gradient_accumulation_steps"]
    seed: int = DEFAULTS["seed"]
    target_modules: list[str] = Field(default_factory=lambda: list(TARGET_MODULES))
    resume: bool = False
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("r")
    @classmethod
    def _rank(cls, value: int) -> int:
        if not 4 <= value <= 256 or value % 4:
            raise ValueError("r must be a multiple of 4 between 4 and 256")
        return value

    @field_validator("epochs")
    @classmethod
    def _epochs(cls, value: float) -> float:
        if not 1 <= value <= 10:
            raise ValueError("epochs must be between 1 and 10")
        return value

    @field_validator("seq_length")
    @classmethod
    def _seq(cls, value: int) -> int:
        if not 128 <= value <= 8192:
            raise ValueError("seq_length must be between 128 and 8192")
        return value

    @field_validator("lr")
    @classmethod
    def _lr(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("lr must be greater than 0")
        return value

    @field_validator("base")
    @classmethod
    def _base(cls, value: ModelRef) -> ModelRef:
        if value.variant and value.variant.upper().endswith(("GGUF", "Q4_K_M", "Q5_K_M", "Q8_0")):
            raise ValueError("Use the GGUF export, not training")
        return value

    @model_validator(mode="after")
    def _name(self) -> TrainingRun:
        if not self.output_name:
            base = self.base.id.rstrip("/").split("/")[-1]
            self.output_name = f"{base}-{self.dataset_artifact.split('@')[0]}-{self.method}"
        if self.hyperparameters:
            merged = {**DEFAULTS, **{k: v for k, v in self.hyperparameters.items() if k in DEFAULTS}}
            for key, value in merged.items():
                setattr(self, key, value)
        return self

    def worker_spec(self, *, dataset_path: Path, model_path: Path, out_dir: Path, scratch: Path,
                    python: str | None = None) -> dict[str, Any]:
        """Everything the child process needs, serialized as plain JSON."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.id,
            "model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "out_dir": str(out_dir),
            "scratch": str(scratch),
            "output_name": self.output_name,
            "method": self.method,
            "python": python or f"{sys.version_info.major}.{sys.version_info.minor}",
            "lora": {"r": self.r, "alpha": self.alpha, "dropout": self.dropout,
                     "target_modules": list(self.target_modules)},
            "sft": {"per_device_train_batch_size": self.batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "learning_rate": self.lr, "num_train_epochs": self.epochs,
                    "max_seq_length": self.seq_length, "save_steps": 100, "seed": self.seed},
            "resume": self.resume,
        }

    def dump(self) -> str:
        return json.dumps(self.worker_spec(dataset_path=Path(), model_path=Path(), out_dir=Path(),
                                          scratch=Path()).copy(), default=str)

def worker_argv(spec_path: Path | str) -> list[str]:
    return [sys.executable, "-m", "kaggleui.ml.train_worker", "--spec", str(spec_path)]

def worker_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or {})
    env.update({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
    return env

def estimate_vram(run: TrainingRun, *, params_b: float, bits: int = 16) -> int:
    """Rough bytes: weights + fp32 adapters/gradients at batch 1, plus activations."""
    weights = params_b * 1e9 * (bits / 32 if run.method == "qlora" else 2)
    trainable = params_b * 1e9 * min(0.02, run.r / 512)
    activations = run.seq_length * run.batch_size * 2 * 4096 * 2
    return int(weights + trainable * 18 + activations * 6)
