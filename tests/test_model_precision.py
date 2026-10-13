"""The precision a run records must be the precision that loaded."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.model as model_module


class _Capture:
    """Stands in for AutoModelForCausalLM: records the load kwargs and
    returns a model whose config may or may not carry a quantization
    config, as a pre-quantised checkpoint's does."""

    def __init__(self, quantized: bool):
        self.kwargs = None
        self.quantized = quantized

    def from_pretrained(self, model_id, **kwargs):
        self.kwargs = kwargs
        config = SimpleNamespace(quantization_config={"quant_method": "fp8"} if self.quantized else None)
        return SimpleNamespace(config=config, eval=lambda: None)


@pytest.fixture
def no_download(monkeypatch):
    monkeypatch.setattr(model_module.AutoTokenizer, "from_pretrained", lambda *a, **k: "tok")


def _load(monkeypatch, precision, quantized_checkpoint):
    capture = _Capture(quantized_checkpoint)
    monkeypatch.setattr(model_module, "AutoModelForCausalLM", capture)
    model_module.load_model("some/model", precision=precision)
    return capture.kwargs


def test_bf16_casts_and_does_not_quantise(monkeypatch, no_download):
    kwargs = _load(monkeypatch, "bf16", quantized_checkpoint=False)
    assert kwargs["dtype"] is torch.bfloat16
    assert "quantization_config" not in kwargs


def test_nf4_quantises_at_load_time(monkeypatch, no_download):
    kwargs = _load(monkeypatch, "nf4", quantized_checkpoint=False)
    assert kwargs["quantization_config"].load_in_4bit
    assert "dtype" not in kwargs


def test_fp8_loads_the_checkpoint_as_published(monkeypatch, no_download):
    kwargs = _load(monkeypatch, "fp8", quantized_checkpoint=True)
    assert "dtype" not in kwargs and "quantization_config" not in kwargs


def test_fp8_refuses_an_unquantised_checkpoint(monkeypatch, no_download):
    with pytest.raises(ValueError, match="not an fp8 checkpoint"):
        _load(monkeypatch, "fp8", quantized_checkpoint=False)


def test_unknown_precision_is_rejected(monkeypatch, no_download):
    with pytest.raises(ValueError, match="unknown precision"):
        model_module.load_model("some/model", precision="int8")


def test_run_metadata_records_model_and_precision(tmp_path):
    spec = importlib.util.spec_from_file_location("eval_script", Path("scripts/eval.py"))
    eval_script = importlib.util.module_from_spec(spec)
    sys.modules["eval_script"] = eval_script
    spec.loader.exec_module(eval_script)
    questions = tmp_path / "q.json"
    questions.write_text("[]")
    meta = eval_script.run_metadata(str(questions), None, "fp8", "none", "Qwen/Qwen3-32B-FP8")
    assert meta["model_id"] == "Qwen/Qwen3-32B-FP8"
    assert meta["precision"] == model_module.PRECISIONS["fp8"]
