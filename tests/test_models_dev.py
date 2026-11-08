"""models.dev fallback: open-weights indexing, exact joins and gap-only application."""

from __future__ import annotations

import httpx

from kaggleui.catalog import models_dev
from kaggleui.catalog.service import CatalogService

DOC = {
    "vercel": {"models": {
        "meta/llama-3.1-8b": {"name": "Llama 3.1 8B", "open_weights": True, "attachment": True,
                              "family": "llama", "knowledge": "2023-12", "release_date": "2024-07-23",
                              "limit": {"context": 128000, "output": 8192}},
        "meta/llama-3.1-8b-instruct": {"name": "Llama 3.1 8B Instruct", "open_weights": True,
                                       "limit": {"context": 200000}},
    }},
    "some-api": {"models": {
        "hidden/closed-model": {"name": "Closed Model", "open_weights": False,
                         "limit": {"context": 1000000, "output": 64000}},
        "shared/shared-model": {"name": "Shared Model", "open_weights": True, "limit": {"context": 32000}},
    }},
    "other-api": {"models": {
        "shared/shared-model": {"name": "Shared Model", "open_weights": True, "family": "shared",
                         "limit": {"context": 65536, "output": 4096}},
    }},
}


def test_index_skips_closed_models_and_merges_duplicate_weights():
    idx = models_dev.index(DOC)
    assert "hidden/closed-model" not in idx
    assert models_dev.lookup(idx, repo="hidden/closed-model") is None
    shared = models_dev.lookup(idx, repo="shared/shared-model")
    assert shared["context_length"] == 65536
    assert shared["max_output_tokens"] == 4096
    assert shared["family"] == "shared"


def test_lookup_prefers_weights_url_then_repo_then_name():
    idx = models_dev.index(DOC)
    url = "https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/model.safetensors"
    assert models_dev.huggingface_repo(url) == "meta-llama/Llama-3.1-8B"
    assert models_dev.lookup(idx, url=url)["context_length"] == 128000
    assert models_dev.lookup(idx, repo="meta-llama/Llama-3.1-8B")["max_output_tokens"] == 8192
    assert models_dev.lookup(idx, ref="llama3.1:8b")["context_length"] == 128000


def test_apply_fills_gaps_only_and_keeps_published_facts():
    entry = models_dev.lookup(models_dev.index(DOC), repo="shared/shared-model")
    item = {"context_length": 131072}
    models_dev.apply(item, entry)
    assert item["context_length"] == 131072  # the model's own metadata wins
    assert item["max_output_tokens"] == 4096
    assert item["models_dev"]["family"] == "shared"
    assert models_dev.apply({}, None) == {}


async def test_search_enrichment_joins_hf_and_ollama(monkeypatch):
    async def fake_fetch(_client):
        return models_dev.index(DOC)

    monkeypatch.setattr(models_dev, "fetch", fake_fetch)
    service = CatalogService(lambda: None)
    try:
        items = [
            {"source": "hf", "id": "meta-llama/Llama-3.1-8B", "name": "Llama-3.1-8B", "modality": "text"},
            {"source": "ollama", "id": "llama3.1", "name": "llama3.1", "variant": "8b", "modality": "text"},
            {"source": "attached", "id": "/kaggle/input/m.gguf", "name": "m", "modality": "text"},
        ]
        await service._apply_models_dev(items)
        assert items[0]["context_length"] == 128000
        assert items[0]["max_output_tokens"] == 8192
        assert "models_dev" not in items[2]
    finally:
        await service.close()


async def test_offline_models_dev_does_not_break_search(hw, monkeypatch):
    async def fail(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    service = CatalogService(lambda: hw("t4x2"))
    source = service.curated["text"][0]
    row = service._from_curated(source, "text")
    row["source"] = "hf"
    row["context_length"] = 32768
    original_context = row["context_length"]

    async def fake_hf(*_args, **_kwargs):
        return [row]

    async def no_enrich(*_args, **_kwargs):
        return False

    monkeypatch.setattr(service.client, "get", fail)
    monkeypatch.setattr(service, "_hf_items", fake_hf)
    monkeypatch.setattr(service, "_enrich", no_enrich)
    try:
        result = await service.search("text", source="hf", live=True, show_unsupported=True)
        assert result["items"]
        assert result["items"][0]["context_length"] == original_context
        assert "models_dev" not in result["items"][0]
        assert await service._models_dev_index(0.1) == {}
    finally:
        await service.close()

def test_ollama_tag_matches_a_dash_style_model_id():
    idx = models_dev.index({"alibaba": {"models": {
        "qwen3:8b": {"name": "Qwen3 8B", "open_weights": True, "limit": {"context": 40960, "output": 8192}},
        "qwen3-8b": {"name": "Qwen3 8B dash", "open_weights": True, "limit": {"context": 131072}},
    }}})
    # Both spellings normalize to the same key, so the larger shared limit wins.
    assert models_dev.lookup(idx, ref="qwen3:8b")["context_length"] == 131072


async def test_imported_items_keep_their_context_window(hw):
    hardware = hw("t4x2")
    service = CatalogService(lambda: hardware)
    try:
        item = service.imported({"path": "/models/imported", "name": "Imported", "modality": "text",
                                 "engine": "transformers", "context_length": 65536})
        assert item["context_length"] == 65536
    finally:
        await service.close()
