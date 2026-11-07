"""Host detection, capabilities and the launch notebooks agree with the installer."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from kaggleui import hosts
from kaggleui.config import ON_COLAB, ON_KAGGLE

ROOT = Path(__file__).resolve().parent.parent
INSTALL = "https://raw.githubusercontent.com/UltraFEmotes/KaggleUI/main/install.sh"

def test_detect_prefers_kaggle(monkeypatch):
    monkeypatch.setattr(hosts, "ON_KAGGLE", True)
    monkeypatch.setattr(hosts, "ON_COLAB", True)
    assert hosts.detect().name == "kaggle"

def test_detect_colab(monkeypatch):
    monkeypatch.setattr(hosts, "ON_KAGGLE", False)
    monkeypatch.setattr(hosts, "ON_COLAB", True)
    host = hosts.detect()
    assert host.name == "colab"
    assert host.capabilities().persistent_disk is False
    assert host.capabilities().secrets == "colab"
    assert any("Drive" in line for line in host.startup_guidance())

def test_local_host_guidance_is_empty(monkeypatch):
    monkeypatch.setattr(hosts, "ON_KAGGLE", False)
    monkeypatch.setattr(hosts, "ON_COLAB", False)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    host = hosts.detect()
    assert host.name == "local"
    assert host.startup_guidance() == []
    assert host.session_info().session_limit_s is None

def test_colab_load_secrets_uses_userdata(monkeypatch):
    colab = types.ModuleType("google.colab")

    class Userdata:
        @staticmethod
        def get(label):
            if label == "HF_TOKEN":
                return "hf_x"
            raise Exception("SecretNotFoundError")

    colab.userdata = Userdata
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(hosts, "ON_KAGGLE", False)
    monkeypatch.setattr(hosts, "ON_COLAB", True)
    assert hosts.detect().load_secrets()["HF_TOKEN"] is True

def test_notebooks_install_through_the_same_script():
    colab = json.loads((ROOT / "notebooks" / "colab.ipynb").read_text())
    kaggle = json.loads((ROOT / "notebooks" / "KaggleUI.ipynb").read_text())
    assert colab["nbformat"] == 4
    code = "\n".join("".join(cell["source"]) for cell in colab["cells"] if cell["cell_type"] == "code")
    assert INSTALL in code and "kaggleui.launch()" in code
    kaggle_code = "\n".join("".join(cell["source"]) for cell in kaggle["cells"] if cell["cell_type"] == "code")
    assert INSTALL in kaggle_code and "kaggleui.launch()" in kaggle_code
    assert "colab.ipynb" in "".join(kaggle["cells"][0]["source"])

def test_detect_matches_config_flags():
    assert hosts.detect().name != "kaggle" or ON_KAGGLE
    assert not ON_COLAB or not ON_KAGGLE

