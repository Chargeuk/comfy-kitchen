"""Exhaustive FP8 byte decoding and stride/rounding regression coverage."""

import pytest
import torch
from types import SimpleNamespace

import comfy_kitchen.backends.mps.fp8 as fp8_backend

from comfy_kitchen.backends.mps.fp8 import (
    MPSFP8UnsupportedError,
    dequantize_per_tensor_fp8,
)


FORMATS = [torch.float8_e4m3fn, torch.float8_e5m2]
OUTPUTS = [torch.float16, torch.bfloat16, torch.float32]
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Metal shader execution requires MPS",
)


def _mps_fp8(cpu):
    # No Tensor.to monkeypatch is needed to place compressed bytes on MPS.
    return cpu.view(torch.uint8).to("mps").view(cpu.dtype)


@requires_mps
@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("scale_value", [1.0, -0.0, 0.1234567, 10000.0, float("inf"), float("nan")])
def test_every_byte(dtype, output_type, scale_value):
    cpu = torch.arange(256, dtype=torch.uint8).view(dtype)
    scale = torch.tensor(scale_value, dtype=torch.float32)
    expected = cpu.to(output_type) * scale.to(output_type)
    actual = dequantize_per_tensor_fp8(_mps_fp8(cpu), scale.to("mps"), output_type).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = (expected == 0) & ~torch.isnan(expected)
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@requires_mps
@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand", "empty", "scalar"])
def test_layouts(dtype, output_type, layout):
    raw = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    if layout == "transpose":
        raw = raw.T
    elif layout == "slice":
        raw = raw[1::3, 2::2]
    elif layout == "expand":
        raw = raw[:1].expand(7, 16)
    elif layout == "empty":
        raw = raw[:0]
    elif layout == "scalar":
        raw = raw[0, 1]
    # Construct layout ON DEVICE; .to can otherwise materialise a dense copy.
    base = torch.arange(256, dtype=torch.uint8).reshape(16, 16).to("mps")
    if layout == "transpose":
        base = base.T
    elif layout == "slice":
        base = base[1::3, 2::2]
    elif layout == "expand":
        base = base[:1].expand(7, 16)
    elif layout == "empty":
        base = base[:0]
    elif layout == "scalar":
        base = base[0, 1]
    scale = torch.tensor([[0.1234567]], dtype=torch.float32)
    expected = raw.view(dtype).to(output_type) * scale.to(output_type)
    actual = dequantize_per_tensor_fp8(base.view(dtype), scale.to("mps"), output_type).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_cpu_declines():
    with pytest.raises(MPSFP8UnsupportedError, match="requires MPS"):
        dequantize_per_tensor_fp8(torch.zeros(2).to(FORMATS[0]), torch.tensor(1.0))


def test_disable_environment(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_DISABLE_MPS", "1")
    with pytest.raises(MPSFP8UnsupportedError, match="disabled by environment"):
        dequantize_per_tensor_fp8(torch.zeros(2).to(FORMATS[0]), torch.tensor(1.0))


@requires_mps
def test_unsupported_contracts():
    x = _mps_fp8(torch.zeros(2).to(FORMATS[0]))
    with pytest.raises(MPSFP8UnsupportedError, match="one floating-point scale"):
        dequantize_per_tensor_fp8(x, torch.ones(2, device="mps"))
    with pytest.raises(MPSFP8UnsupportedError, match="inference-only"):
        dequantize_per_tensor_fp8(x, torch.ones((), device="mps", requires_grad=True))


@requires_mps
def test_launch_failure_cached_for_fallback(monkeypatch):
    calls = []
    def failing_kernel(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected unsupported shader launch")
    monkeypatch.setattr(fp8_backend, "_launch_error", None)
    monkeypatch.setattr(fp8_backend, "_get_library", lambda: SimpleNamespace(decode_bfloat=failing_kernel))
    x = _mps_fp8(torch.zeros(2).to(FORMATS[0]))
    scale = torch.ones((), device="mps")
    with pytest.raises(MPSFP8UnsupportedError, match="launch failed"):
        dequantize_per_tensor_fp8(x, scale)
    with pytest.raises(MPSFP8UnsupportedError, match="disabled after launch failure"):
        dequantize_per_tensor_fp8(x, scale)
    assert len(calls) == 1


@requires_mps
@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_launch_oom_is_not_fallback(monkeypatch, error_type):
    error = error_type("MPS backend out of memory (injected)")
    def failing_kernel(*args, **kwargs):
        raise error
    monkeypatch.setattr(fp8_backend, "_launch_error", None)
    monkeypatch.setattr(fp8_backend, "_get_library", lambda: SimpleNamespace(decode_bfloat=failing_kernel))
    x = _mps_fp8(torch.zeros(2).to(FORMATS[0]))
    with pytest.raises(error_type) as caught:
        dequantize_per_tensor_fp8(x, torch.ones((), device="mps"))
    assert caught.value is error
    assert fp8_backend._launch_error is None
