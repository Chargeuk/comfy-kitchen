"""NVFP4 Metal decoding: byte formats, tiled scales, rounding and fallback."""

from types import SimpleNamespace

import pytest
import torch

import comfy_kitchen as ck
import comfy_kitchen.backends.eager.quantization as eager_quantization
import comfy_kitchen.backends.mps as mps_backend
import comfy_kitchen.backends.mps.nvfp4 as nvfp4_backend
from comfy_kitchen.backends.eager.quantization import dequantize_nvfp4 as reference_decode
from comfy_kitchen.backends.mps.nvfp4 import MPSNVFP4UnsupportedError, dequantize_nvfp4
from comfy_kitchen.float_utils import to_blocked


OUTPUTS = [torch.float16, torch.bfloat16, torch.float32]
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Metal shader execution requires MPS",
)


def _to_mps_fp8(tensor):
    return tensor.view(torch.uint8).to("mps").view(tensor.dtype)


def _example(rows=3, cols=80):
    packed = (torch.arange(rows * (cols // 2), dtype=torch.int64) % 256).to(torch.uint8)
    packed = packed.reshape(rows, cols // 2)
    scales = torch.arange(rows * (cols // 16), dtype=torch.int64).reshape(rows, cols // 16)
    # Positive finite FP8 values with distinct block positions.
    scales = (scales % 96 + 8).to(torch.uint8)
    return packed, torch.tensor(0.1234567), to_blocked(scales, flatten=False).view(torch.float8_e4m3fn)


def _assert_exact(actual, expected):
    actual = actual.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@requires_mps
@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("hi_first", [True, False])
@pytest.mark.parametrize("scale_value", [1.0, -0.0, -0.0312345, 0.1234567, 10000.0, float("inf"), float("nan")])
def test_all_nibbles_and_all_fp8_scale_bytes(output_type, hi_first, scale_value):
    # Every E2M1 value is paired with every E4M3 byte, including signed zeros,
    # subnormals, NaNs and scales which overflow the FP16 intermediate.
    packed = torch.tensor([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF], dtype=torch.uint8)
    packed = packed.repeat(256, 1)
    scale_bytes = torch.arange(256, dtype=torch.uint8).reshape(256, 1)
    scales = to_blocked(scale_bytes, flatten=False).view(torch.float8_e4m3fn)
    tensor_scale = torch.tensor(scale_value)
    expected = reference_decode(packed, tensor_scale, scales, output_type, hi_first)
    actual = dequantize_nvfp4(packed.to("mps"), tensor_scale.to("mps"), _to_mps_fp8(scales), output_type, hi_first)
    _assert_exact(actual, expected)


@requires_mps
@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("rows,cols", [(1, 16), (31, 48), (32, 64), (33, 80), (127, 160), (128, 64), (129, 80), (257, 272)])
def test_padded_scale_tiles(output_type, rows, cols):
    packed, tensor_scale, scales = _example(rows, cols)
    expected = reference_decode(packed, tensor_scale, scales, output_type)
    actual = dequantize_nvfp4(packed.to("mps"), tensor_scale.to("mps"), _to_mps_fp8(scales), output_type)
    _assert_exact(actual, expected)


def _strided_bytes(tensor, device, layout):
    if layout == "transpose":
        return tensor.T.contiguous().to(device).T
    if layout == "slice":
        backing = torch.zeros((tensor.shape[0] * 2 + 3, tensor.shape[1] * 2 + 4), dtype=torch.uint8, device=device)
        result = backing[1:1 + tensor.shape[0] * 2:2, 2:2 + tensor.shape[1] * 2:2]
        result.copy_(tensor.to(device))
        return result
    if layout == "expand":
        return tensor[:1].to(device).expand_as(tensor)
    return tensor.to(device)


@requires_mps
@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
@pytest.mark.parametrize("scale_device", ["cpu", "mps"])
def test_byte_safe_strided_inputs(output_type, layout, scale_device):
    packed, tensor_scale, scales = _example(129, 80)
    cpu_packed = _strided_bytes(packed, "cpu", layout)
    cpu_scales = _strided_bytes(scales.view(torch.uint8), "cpu", layout).view(scales.dtype)
    mps_packed = _strided_bytes(packed, "mps", layout)
    device_scales = _strided_bytes(scales.view(torch.uint8), scale_device, layout).view(scales.dtype)
    # A one-element view with a nonzero storage offset.
    scale_storage = torch.tensor([9.0, tensor_scale.item(), 8.0], device=scale_device)
    device_tensor_scale = scale_storage[1:2].reshape(1, 1)
    expected = reference_decode(cpu_packed, tensor_scale, cpu_scales, output_type)
    actual = dequantize_nvfp4(mps_packed, device_tensor_scale, device_scales, output_type)
    _assert_exact(actual, expected)


@requires_mps
@pytest.mark.parametrize("output_type", OUTPUTS)
def test_scale_mutations_are_observed(output_type):
    packed, scale, scales = _example(129, 80)
    device_packed = packed.to("mps")
    device_scale = scale.to("mps")
    device_scales = _to_mps_fp8(scales)
    for factor in (1.0, -0.5, 2.0):
        changed_scale = scale * factor
        device_scale.copy_(changed_scale)
        changed_scales = (scales.view(torch.uint8) + 1).view(scales.dtype)
        device_scales.view(torch.uint8).copy_(changed_scales.view(torch.uint8))
        expected = reference_decode(packed, changed_scale, changed_scales, output_type)
        actual = ck.dequantize_nvfp4(device_packed, device_scale, device_scales, output_type)
        _assert_exact(actual, expected)
        scales = changed_scales


@requires_mps
@pytest.mark.parametrize("rows,cols", [(0, 16), (3, 0), (0, 0)])
def test_empty_weights(rows, cols):
    packed = torch.empty((rows, cols // 2), device="mps", dtype=torch.uint8)
    scales = torch.empty(0, dtype=torch.uint8, device="mps").view(torch.float8_e4m3fn)
    actual = dequantize_nvfp4(packed, torch.tensor(1.0), scales)
    assert actual.shape == (rows, cols)
    assert actual.dtype == torch.bfloat16


def test_cpu_declines():
    with pytest.raises(MPSNVFP4UnsupportedError, match="requires MPS"):
        dequantize_nvfp4(*_example())


def test_disable_environment(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_DISABLE_MPS", "1")
    with pytest.raises(MPSNVFP4UnsupportedError, match="disabled by environment"):
        dequantize_nvfp4(*_example())


@requires_mps
def test_unsupported_contracts():
    packed, tensor_scale, scales = _example()
    packed, scales = packed.to("mps"), _to_mps_fp8(scales)
    with pytest.raises(MPSNVFP4UnsupportedError, match="2D uint8"):
        dequantize_nvfp4(packed[:, :-1], tensor_scale, scales)
    with pytest.raises(MPSNVFP4UnsupportedError, match="2D uint8"):
        dequantize_nvfp4(packed.unsqueeze(0), tensor_scale, scales)
    with pytest.raises(MPSNVFP4UnsupportedError, match="one floating-point tensor scale"):
        dequantize_nvfp4(packed, torch.ones(2), scales)
    with pytest.raises(MPSNVFP4UnsupportedError, match="E4M3FN block scales"):
        dequantize_nvfp4(packed, tensor_scale, scales.view(torch.uint8))
    with pytest.raises(MPSNVFP4UnsupportedError, match="padded weight shape"):
        dequantize_nvfp4(packed, tensor_scale, scales.view(torch.uint8)[:-1].view(scales.dtype))
    with pytest.raises(MPSNVFP4UnsupportedError, match="output dtype"):
        dequantize_nvfp4(packed, tensor_scale, scales, torch.int32)
    with pytest.raises(MPSNVFP4UnsupportedError, match="inference-only"):
        dequantize_nvfp4(packed, tensor_scale.requires_grad_(), scales)


def test_compile_failure_cached(monkeypatch):
    calls = []
    def failing_compile(source):
        calls.append(source)
        raise RuntimeError("injected unsupported shader compiler")
    monkeypatch.setattr(nvfp4_backend, "_library", None)
    monkeypatch.setattr(nvfp4_backend, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", failing_compile)
    for _ in range(2):
        with pytest.raises(MPSNVFP4UnsupportedError, match="compilation failed"):
            nvfp4_backend._get_library()
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_compile_oom_not_cached(monkeypatch, error_type):
    error = error_type("MPS backend out of memory (injected)")
    def failing_compile(source):
        raise error
    monkeypatch.setattr(nvfp4_backend, "_library", None)
    monkeypatch.setattr(nvfp4_backend, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", failing_compile)
    with pytest.raises(error_type) as caught:
        nvfp4_backend._get_library()
    assert caught.value is error
    assert nvfp4_backend._compile_error is None


@requires_mps
def test_launch_failure_cached(monkeypatch):
    calls = []
    def failing_kernel(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected unsupported shader launch")
    monkeypatch.setattr(nvfp4_backend, "_launch_error", None)
    monkeypatch.setattr(nvfp4_backend, "_get_library", lambda: SimpleNamespace(decode_nvfp4_bfloat=failing_kernel))
    packed, scale, scales = _example()
    args = packed.to("mps"), scale, _to_mps_fp8(scales)
    with pytest.raises(MPSNVFP4UnsupportedError, match="launch failed"):
        dequantize_nvfp4(*args)
    with pytest.raises(MPSNVFP4UnsupportedError, match="disabled after launch failure"):
        dequantize_nvfp4(*args)
    assert len(calls) == 1


@requires_mps
@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_launch_oom_does_not_fallback(monkeypatch, error_type):
    error = error_type("MPS backend out of memory (injected)")
    def failing_kernel(*args, **kwargs):
        raise error
    monkeypatch.setattr(nvfp4_backend, "_launch_error", None)
    monkeypatch.setattr(nvfp4_backend, "_get_library", lambda: SimpleNamespace(decode_nvfp4_bfloat=failing_kernel))
    packed, scale, scales = _example()
    # Exercise the registry wrapper too: it must not intercept an OOM.
    with pytest.raises(error_type) as caught:
        mps_backend.dequantize_nvfp4(packed.to("mps"), scale, _to_mps_fp8(scales))
    assert caught.value is error
    assert nvfp4_backend._launch_error is None


def test_cpu_fallback_uses_captured_eager_not_patched_function(monkeypatch):
    packed, scale, scales = _example()
    expected = reference_decode(packed, scale, scales)
    def unavailable(*args, **kwargs):
        raise MPSNVFP4UnsupportedError("injected unsupported shader")
    def patched(*args, **kwargs):
        raise AssertionError("compatibility wrapper was reentered")
    monkeypatch.setattr(nvfp4_backend, "dequantize_nvfp4", unavailable)
    monkeypatch.setattr(eager_quantization, "dequantize_nvfp4", patched)
    monkeypatch.setattr(mps_backend.eager, "dequantize_nvfp4", patched)
    _assert_exact(mps_backend.dequantize_nvfp4(packed, scale, scales), expected)


@requires_mps
@pytest.mark.parametrize("scale_device", ["cpu", "mps"])
def test_registry_dispatch_and_fallback(monkeypatch, scale_device):
    packed, scale, scales = _example()
    expected = reference_decode(packed, scale, scales)
    kwargs = dict(qx=packed.to("mps"), per_tensor_scale=scale.to(scale_device),
                  block_scales=_to_mps_fp8(scales) if scale_device == "mps" else scales,
                  output_type=torch.bfloat16)
    assert ck.registry.get_capable_backend("dequantize_nvfp4", kwargs) == "mps"
    _assert_exact(ck.dequantize_nvfp4(**kwargs), expected)
    def unavailable(*args, **kwargs):
        raise MPSNVFP4UnsupportedError("injected unsupported shader")
    monkeypatch.setattr(nvfp4_backend, "dequantize_nvfp4", unavailable)
    result = ck.dequantize_nvfp4(**kwargs)
    assert result.device.type == "mps"
    _assert_exact(result, expected)


def test_wrapper_rejects_scale_gradients():
    packed, scale, scales = _example()
    with pytest.raises(NotImplementedError, match="scale gradients"):
        mps_backend.dequantize_nvfp4(packed, scale.requires_grad_(), scales)
