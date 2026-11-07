"""Teacher data collection: prompts answered by a local deployment (or opt-in HF Inference)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ..engines.context import DeployError

if TYPE_CHECKING:
    from ..deploy.manager import Deployment
    from ..server.state import AppState

HF_INFERENCE = "https://api-inference.huggingface.co/models/"

def _estimate(text: str) -> int:
    return len(text) // 4

async def collect(teacher: Deployment, prompts: list[str], *, max_new_tokens: int = 512,
                  temperature: float = 0.7, budget_tokens: int = 200_000, progress=None) -> dict:
    """Answer each prompt in order, stopping cleanly as soon as the token budget would be exceeded."""
    if teacher.backend is None:
        raise DeployError(f"{teacher.served_name} is not ready")
    samples: list[dict[str, Any]] = []
    spent = 0
    for index, prompt in enumerate(prompts, start=1):
        if spent + _estimate(prompt) + max_new_tokens > budget_tokens:
            raise DeployError(f"teacher budget of {budget_tokens} tokens exhausted after {index - 1} prompts")
        response = await teacher.backend.chat({"model": teacher.served_name, "temperature": temperature,
                                               "max_tokens": max_new_tokens,
                                               "messages": [{"role": "user", "content": prompt}]})
        try:
            answer = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise DeployError(f"teacher returned an unexpected payload for prompt {index}") from None
        if not isinstance(answer, str) or not answer.strip():
            raise DeployError(f"teacher returned no text for prompt {index}")
        spent += _estimate(prompt) + _estimate(answer)
        samples.append({"messages": [{"role": "user", "content": prompt},
                                     {"role": "assistant", "content": answer}]})
        if progress:
            progress(index / len(prompts), f"answered {index}/{len(prompts)} prompts")
    return {"samples": samples, "tokens": spent, "teacher": teacher.served_name}

async def remote_collect(state: AppState, repo: str, prompts: list[str], *, max_new_tokens: int = 512,
                         temperature: float = 0.7) -> dict:
    """HF Inference API fallback. Only when explicitly enabled; never a paid endpoint."""
    if not state.settings.ml.teacher_remote:
        raise DeployError("Remote teachers are disabled (Settings → Model Studio → teacher_remote)")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise DeployError("Remote teachers need HF_TOKEN")
    import httpx

    samples = []
    async with httpx.AsyncClient(timeout=120) as client:
        for prompt in prompts:
            response = await client.post(HF_INFERENCE + repo, headers={"Authorization": f"Bearer {token}"},
                                         json={"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens,
                                                                                "temperature": temperature,
                                                                                "return_full_text": False}})
            if response.status_code != 200:
                raise DeployError(f"HF Inference returned {response.status_code}: {response.text[:200]}")
            data = response.json()
            text = data[0].get("generated_text") if isinstance(data, list) and data else None
            if not text:
                raise DeployError("HF Inference returned no text")
            samples.append({"messages": [{"role": "user", "content": prompt},
                                         {"role": "assistant", "content": text}]})
    return {"samples": samples, "tokens": sum(_estimate(s["messages"][0]["content"]) +
                                              _estimate(s["messages"][1]["content"]) for s in samples),
            "teacher": repo}

def check_collision(teacher_name: str, student_name: str) -> None:
    if teacher_name.strip().lower() == student_name.strip().lower():
        raise DeployError("The teacher and the trained model cannot share a name")

def validate_samples(samples: list[dict[str, Any]]) -> list[str]:
    """Cheap structural check before writing teacher data to disk."""
    errors = []
    for index, row in enumerate(samples, start=1):
        turns = row.get("messages") if isinstance(row, dict) else None
        if not turns or turns[-1].get("role") != "assistant":
            errors.append(f"sample {index}: missing assistant answer")
    return errors

