"""GGUF file selection and llama-swap config bookkeeping (no network, no binaries)."""

import pathlib

from kaggleui.engines import llamaswap
from kaggleui.engines.llamacpp_models import file_list, needs_split, pick, quant_of
from kaggleui.schemas import DeploymentSpec, Settings
from kaggleui.server.state import AppState

FILES = [
    {"rfilename": "README.md", "size": 100},
    {"rfilename": "mmproj-model-f16.gguf", "size": 500},
    {"rfilename": "Qwen-Q8_0.gguf", "size": 900},
    {"rfilename": "Qwen-Q4_K_M.gguf", "size": 400},
    {"rfilename": "Qwen-Q2_K.gguf", "size": 300},
    {"rfilename": "Big-Q4_K_M-00001-of-00002.gguf", "size": 700},
    {"rfilename": "Big-Q4_K_M-00002-of-00002.gguf", "size": 700},
    {"rfilename": "Broken-Q5_K_M-00001-of-00003.gguf", "size": 700},
]


def test_prefers_q4_k_m_and_groups_complete_shards():
    picked = pick(FILES)
    assert [f["rfilename"] for f in picked] == [
        "Big-Q4_K_M-00001-of-00002.gguf", "Big-Q4_K_M-00002-of-00002.gguf"]
    assert [f["rfilename"] for f in pick(FILES, quant="Q8_0")] == ["Qwen-Q8_0.gguf"]
    assert file_list(FILES, quant="Q4_K_M")[0] == {"path": "Big-Q4_K_M-00001-of-00002.gguf", "size": "700"}


def test_unknown_quant_lists_what_is_available():
    try:
        pick(FILES, quant="Q3_K_S")
    except ValueError as exc:
        assert "Q8_0" in str(exc) and "Q4_K_M" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_quant_and_split_helpers():
    assert quant_of("model-IQ4_XS.gguf") == "IQ4_XS"
    assert quant_of("model.Q4_K_M.gguf") == "Q4_K_M"
    assert quant_of("plain.gguf") is None
    assert needs_split(3_000, 1_000)
    assert not needs_split(3_000, 5_000)


def test_swap_instance_rebuilds_models_from_records(paths):
    settings = Settings()
    state = AppState(paths=paths, settings=settings, port=8996, mock=True, fake_hw="t4x2")
    store = state.workbench
    spec = DeploymentSpec.model_validate({"modality": "text", "lifecycle": "llama_swap",
                                          "model": {"source": "hf", "id": "org/m-GGUF", "engine": "llamacpp"}})
    store.put_deployment("dep_a", spec, "alpha", created_at=1.0,
                         runtime={"argv": ["/bin/llama-server", "--model", "/m/a.gguf", "--port", "${PORT}"],
                                  "served_name": "alpha", "gpus": [0]})
    store.put_deployment("dep_b", spec, "beta", created_at=2.0)  # no runtime yet: not served

    llamaswap.sync_from_records(state)
    inst = llamaswap.instance_for(paths)

    assert sorted(inst.models) == ["alpha"]
    written = pathlib.Path(inst.config_path).read_text()
    assert "alpha:" in written and "/m/a.gguf" in written
    assert "beta:" not in written

