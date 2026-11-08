"""The in-venv body of merge/GGUF jobs (imports torch, peft and transformers lazily)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from .recipes import QUANTS
from .train_worker import emit, log, progress

CONVERTER_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/b12189/convert_hf_to_gguf.py"

def merge(base: Path, adapter: Path, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress("loading", 0.05, f"loading {base}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu",
                                                 low_cpu_mem_usage=True)
    progress("merging", 0.4, "applying adapter")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    progress("saving", 0.8, f"writing {out}")
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)

def build_imatrix(f16: Path, calibration: Path, out: Path, imatrix_bin: str) -> Path:
    dest = out / "imatrix.dat"
    progress("imatrix", 0.55, "building the importance matrix")
    subprocess.run([imatrix_bin, "-m", str(f16), "-f", str(calibration), "-o", str(dest)], check=True)
    return dest

def to_gguf(model_dir: Path, out: Path, quant: str, quantize_bin: str | None, scratch: Path,
            *, imatrix_bin: str | None = None, calibration: Path | None = None) -> Path:
    if quant not in QUANTS:
        raise SystemExit(f"unknown quant {quant}")
    out.mkdir(parents=True, exist_ok=True)
    script = scratch / "convert_hf_to_gguf.py"
    if not script.is_file():
        progress("converter", 0.05, "downloading convert_hf_to_gguf.py")
        with urllib.request.urlopen(CONVERTER_URL, timeout=60) as response:
            script.write_bytes(response.read())
    f16 = out / "model-f16.gguf"
    progress("converting", 0.1, f"converting {model_dir.name} to f16")
    subprocess.run([sys.executable, str(script), str(model_dir), "--outfile", str(f16), "--outtype", "f16"],
                   check=True)
    if quant == "F16":
        return f16
    if not quantize_bin:
        raise SystemExit("llama-quantize was not found in the pinned llama.cpp release")
    dest = out / f"model-{quant}.gguf"
    matrix = None
    if calibration is not None:
        if not imatrix_bin:
            raise SystemExit("llama-imatrix was not found in the pinned llama.cpp release")
        matrix = build_imatrix(f16, calibration, out, imatrix_bin)
    progress("quantizing", 0.6, f"quantizing to {quant}")
    argv = [quantize_bin, str(f16), str(dest), quant]
    if matrix is not None:
        argv += ["--imatrix", str(matrix)]
    subprocess.run(argv, check=True)
    return dest

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggleui.ml.conversion")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    out = Path(spec["out_dir"])
    calibration = Path(spec["calibration"]) if spec.get("calibration") else None
    matrix_args = {"imatrix_bin": spec.get("imatrix_bin"), "calibration": calibration}
    try:
        if spec["mode"] == "merge":
            merge(Path(spec["base_path"]), Path(spec["adapter_path"]), out)
        elif spec["mode"] == "convert":
            merged = out / "merged"
            merge(Path(spec["base_path"]), Path(spec["adapter_path"]), merged)
            to_gguf(merged, out, spec["quant"], spec.get("quantize_bin"), Path(spec["scratch"]), **matrix_args)
        else:
            to_gguf(Path(spec["model_path"]), out, spec["quant"], spec.get("quantize_bin"),
                    Path(spec["scratch"]), **matrix_args)
    except subprocess.CalledProcessError as exc:
        emit({"error": f"{exc.cmd[0]} exited with {exc.returncode}"})
        return exc.returncode or 1
    except SystemExit as exc:
        emit({"error": str(exc)})
        return 2
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    log(f"wrote {len(files)} files to {out}")
    emit({"progress": 1.0, "stage": "done", "message": "artifact written", "artifact": {"dir": str(out), "files": files}})
    return 0

if __name__ == "__main__":
    sys.exit(main())

