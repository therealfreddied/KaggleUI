"""The training body, executed inside the isolated venv created by ``train_worker``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..workbench.runner import EVENT_TAG


def emit(payload: dict) -> None:
    print(EVENT_TAG + json.dumps(payload), flush=True)

def load_samples(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        turns = row.get("messages") if isinstance(row, dict) else row
        rows.append({"messages": turns})
    return rows

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggleui.ml.training")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        emit({"error": f"training dependencies are missing: {exc}"})
        return 2

    emit({"message": "loading dataset", "progress": 0.02, "stage": "preparing"})
    samples = load_samples(Path(spec["dataset_path"]))
    tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(sample: dict) -> dict:
        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = Dataset.from_list([render(s) for s in samples])
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    quant = None
    if spec["method"] == "qlora":
        if not cuda:
            emit({"error": "QLoRA needs a CUDA GPU; pick LoRA on this host"})
            return 3
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    lora = spec["lora"]
    cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                     target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM")
    emit({"message": f"loading {spec['model_path']}", "progress": 0.1, "stage": "loading"})
    model = AutoModelForCausalLM.from_pretrained(spec["model_path"], torch_dtype=torch.bfloat16 if bf16 else None,
                                                 quantization_config=quant, device_map="auto",
                                                 trust_remote_code=True)
    model.config.use_cache = False
    sft = spec["sft"]
    config = SFTConfig(output_dir=str(out / "checkpoints"), per_device_train_batch_size=sft["per_device_train_batch_size"],
                       gradient_accumulation_steps=sft["gradient_accumulation_steps"],
                       learning_rate=sft["learning_rate"], num_train_epochs=sft["num_train_epochs"],
                       max_seq_length=sft["max_seq_length"], bf16=bf16, fp16=cuda and not bf16,
                       gradient_checkpointing=True, save_steps=sft["save_steps"], seed=sft["seed"],
                       report_to=[], logging_steps=10, dataset_text_field="text")
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, peft_config=cfg, tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=spec.get("resume_from") or None)
    emit({"message": "saving adapter", "progress": 0.95, "stage": "saving"})
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    emit({"progress": 1.0, "stage": "done", "message": "adapter written", "artifact": {"dir": str(out), "files": files}})
    return 0

if __name__ == "__main__":
    sys.exit(main())

