"""GGUF metadata reader: header, typed values and MoE detection used by placement."""

import struct

import pytest

from kaggleui.gguf import GGUFError, has_magic, is_moe, layer_count, read_metadata

UINT32, STRING, BOOL, ARRAY, UINT64, FLOAT32 = 4, 8, 7, 9, 10, 6


def _gguf(meta: dict[str, tuple[int, object]], version: int = 3) -> bytes:
    def scalar(vtype: int, value) -> bytes:
        if vtype == STRING:
            raw = str(value).encode()
            return struct.pack("<Q", len(raw)) + raw
        if vtype == BOOL:
            return struct.pack("<B", 1 if value else 0)
        fmt = {UINT32: "<I", UINT64: "<Q", FLOAT32: "<f"}[vtype]
        return struct.pack(fmt, value)

    body = b""
    for key, (vtype, value) in meta.items():
        body += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", vtype)
        if vtype == ARRAY:
            body += struct.pack("<I", UINT32) + struct.pack("<Q", len(value))
            body += b"".join(struct.pack("<I", v) for v in value)
        else:
            body += scalar(vtype, value)
    return b"GGUF" + struct.pack("<IQQ", version, 0, len(meta)) + body


def test_reads_strings_numbers_bools_and_arrays(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf({
        "general.architecture": (STRING, "qwen3moe"),
        "general.name": (STRING, "Qwen3 MoE"),
        "qwen3moe.block_count": (UINT32, 48),
        "qwen3moe.expert_count": (UINT32, 128),
        "qwen3moe.head_count": (UINT64, 32),
        "general.alignment": (UINT32, 32),
        "some.flag": (BOOL, True),
        "some.scale": (FLOAT32, 0.5),
        "some.sizes": (ARRAY, [1, 2, 3]),
    }))
    meta = read_metadata(path)
    assert meta["general.architecture"] == "qwen3moe"
    assert meta["some.flag"] is True
    assert meta["some.scale"] == pytest.approx(0.5)
    assert meta["some.sizes"] == [1, 2, 3]
    assert meta["qwen3moe.head_count"] == 32


def test_moe_and_layer_helpers(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf({"general.architecture": (STRING, "qwen3moe"),
                            "qwen3moe.block_count": (UINT32, 48),
                            "qwen3moe.expert_count": (UINT32, 128)}))
    meta = read_metadata(path)
    assert is_moe(meta)
    assert layer_count(meta) == 48

    dense = tmp_path / "d.gguf"
    dense.write_bytes(_gguf({"general.architecture": (STRING, "llama"), "llama.block_count": (UINT32, 32)}))
    assert not is_moe(read_metadata(dense))
    assert layer_count(read_metadata(dense)) == 32
    assert not is_moe(None) and layer_count(None) is None


def test_rejects_non_gguf_and_future_versions(tmp_path):
    bogus = tmp_path / "x.gguf"
    bogus.write_bytes(b"NOTGGUF" + b"\0" * 16)
    assert not has_magic(bogus)
    with pytest.raises(GGUFError):
        read_metadata(bogus)

    future = tmp_path / "future.gguf"
    future.write_bytes(_gguf({"general.architecture": (STRING, "llama")}, version=99))
    assert has_magic(future)
    with pytest.raises(GGUFError, match="version"):
        read_metadata(future)


def test_truncated_file_is_an_error_not_a_crash(tmp_path):
    path = tmp_path / "cut.gguf"
    path.write_bytes(_gguf({"general.architecture": (STRING, "llama")})[:20])
    with pytest.raises(GGUFError):
        read_metadata(path)
    with pytest.raises(GGUFError):
        read_metadata(tmp_path / "missing.gguf")
