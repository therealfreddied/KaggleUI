"""Placement validation and the argv llama.cpp receives for it."""

import yaml

from kaggleui.engines.llamacpp_launch import server_argv, swap_model_entry, swap_yaml
from kaggleui.schemas import DeploymentSpec
from kaggleui.workbench import placement as place


def _spec(placement: dict | None = None, **extra) -> DeploymentSpec:
    body = {"modality": "text", "model": {"source": "hf", "id": "org/model-GGUF", "engine": "llamacpp"}, **extra}
    if placement is not None:
        body["placement"] = placement
    return DeploymentSpec.model_validate(body)


MOE = {"general.architecture": "qwen3moe", "qwen3moe.block_count": 48, "qwen3moe.expert_count": 128}
DENSE = {"general.architecture": "llama", "llama.block_count": 32}


def test_device_choice_maps_to_visible_positions(hw):
    profile = hw("t4x2")
    spec = _spec({"devices": [1], "main_gpu": 1})
    assert place.validate(spec, profile, arch=DENSE) == []
    resolved = place.resolve(spec, profile, arch=DENSE, assigned=[1])
    assert resolved.gpus == [1]
    assert resolved.main_gpu == 0  # CUDA_VISIBLE_DEVICES=1 renumbers the card to zero


def test_bad_split_and_duplicate_devices_are_refused(hw):
    profile = hw("t4x2")
    errors = place.validate(_spec({"devices": [1, 0]}), profile, arch=DENSE)
    assert any("ascending" in e for e in errors)
    errors = place.validate(_spec({"devices": [0, 1], "tensor_split": [0.5]}), profile, arch=DENSE)
    assert any("tensor_split" in e for e in errors)
    errors = place.validate(_spec({"devices": [0], "tensor_split": [1.0]}), profile, arch=DENSE)
    assert any("two devices" in e for e in errors)


def test_cpu_moe_needs_metadata_and_a_moe_model(hw):
    profile = hw("t4x2")
    assert any("metadata" in e for e in place.validate(_spec({"cpu_moe": True}), profile, arch=None))
    assert any("mixture-of-experts" in e for e in place.validate(_spec({"cpu_moe": True}), profile, arch=DENSE))
    assert place.validate(_spec({"cpu_moe_layers": 12}), profile, arch=MOE) == []


def test_mmap_conflict_and_kv_types(hw):
    profile = hw("t4x2")
    errors = place.validate(_spec({"mlock": True, "mmap": False}), profile, arch=DENSE)
    assert any("mlock" in e for e in errors)
    resolved = place.resolve(_spec({"kv_cache_type_k": "q8_0", "kv_cache_type_v": "totally-not-kv"}),
                             profile, arch=DENSE)
    assert resolved.kv_k == "q8_0"
    assert any("not a known llama.cpp type" in n for n in resolved.notes)


def test_argv_compiles_the_resolved_placement(hw, paths):
    profile = hw("t4x2")
    spec = _spec({"devices": [0, 1], "tensor_split": [0.7, 0.3], "gpu_layers": 20, "flash_attn": "on",
                  "threads": 6, "parallel": 2}, options={"ctx": 4096})
    resolved = place.resolve(spec, profile, arch=DENSE, assigned=[0, 1])
    argv = server_argv(bin_path="/x/llama-server", model_path="/m/a.gguf", host="127.0.0.1", port=8111,
                       served_name="qwen", spec=spec, placed=resolved)
    assert argv[0] == "/x/llama-server"
    for flag, value in (("--n-gpu-layers", "20"), ("--tensor-split", "0.7,0.3"), ("--threads", "6"),
                        ("--parallel", "2"), ("--ctx-size", "4096"), ("--flash-attn", "on"),
                        ("--alias", "qwen"), ("--port", "8111")):
        assert argv[argv.index(flag) + 1] == value
    assert "--metrics" in argv and "--jinja" in argv


def test_swap_config_is_valid_yaml_and_keeps_the_port_macro(hw):
    profile = hw("t4x2")
    spec = _spec({"devices": [0]})
    resolved = place.resolve(spec, profile, arch=DENSE, assigned=[0])
    argv = server_argv(bin_path="/x/llama-server", model_path="/m/spaced dir/a.gguf", host="127.0.0.1",
                       port="${PORT}", served_name="qwen", spec=spec, placed=resolved)
    config = yaml.safe_load(swap_yaml({"qwen": swap_model_entry(served_name="qwen", cmd=argv, ttl=0)}))
    cmd = config["models"]["qwen"]["cmd"]
    assert "${PORT}" in cmd
    assert "/m/spaced dir/a.gguf" in cmd
    assert config["models"]["qwen"]["ttl"] == 0
