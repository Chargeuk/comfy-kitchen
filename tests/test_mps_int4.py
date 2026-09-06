"""Rotated-basis INT4 decoder: byte order, cast rounding, strides and failure."""

from types import SimpleNamespace

import pytest
import torch

import comfy_kitchen.backends.mps.int4 as int4_backend
from comfy_kitchen.backends.mps.int4 import MPSINT4UnsupportedError, unpack_int4_scaled


OUTPUTS = [torch.float16, torch.bfloat16, torch.float32]
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Metal shader execution requires MPS",
)


def reference(packed, scales, output_type):
    # Match the installed AppleSilicon-FP8 W4A16 path exactly.
    packed = packed.view(torch.int8)
    lo = (packed << 4) >> 4
    hi = packed >> 4
    decoded = torch.stack((lo, hi), dim=-1).to(output_type)
    return decoded.reshape(packed.shape[0], packed.shape[1] * 2) * scales.to(output_type).reshape(-1, 1)


def assert_exact(actual, expected):
    actual = actual.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@requires_mps
@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("storage_type", [torch.int8, torch.uint8])
def test_all_bytes_and_scale_rounding(output_type, storage_type):
    scales = torch.tensor([1., -0., .1234567, -.0312345, 1e-8, 10000., float("inf"), float("nan")])
    packed = torch.arange(256, dtype=torch.uint8).repeat(scales.numel(), 1).view(storage_type)
    actual = unpack_int4_scaled(packed.to("mps"), scales.to("mps"), output_type)
    assert_exact(actual, reference(packed, scales, output_type))


def strided(tensor, device, layout):
    if layout == "transpose":
        return tensor.T.contiguous().to(device).T
    if layout == "slice":
        backing = torch.zeros((tensor.shape[0] * 2 + 3, tensor.shape[1] * 2 + 4), dtype=tensor.dtype, device=device)
        result = backing[1:1 + tensor.shape[0] * 2:2, 2:2 + tensor.shape[1] * 2:2]
        result.copy_(tensor.to(device))
        return result
    if layout == "expand":
        return tensor[:1].to(device).expand_as(tensor)
    raise ValueError(layout)


@requires_mps
@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
@pytest.mark.parametrize("scale_device", ["cpu", "mps"])
def test_strides_offsets_and_row_scales(output_type, layout, scale_device):
    packed = (torch.arange(11 * 35) % 256).to(torch.uint8).reshape(11, 35).view(torch.int8)
    scales = torch.linspace(-.172, .819, 24)[1:23:2].reshape(1, 11)
    actual = unpack_int4_scaled(strided(packed, "mps", layout), scales.to(scale_device), output_type)
    assert_exact(actual, reference(strided(packed, "cpu", layout), scales, output_type))


@requires_mps
@pytest.mark.parametrize("output_type", OUTPUTS)
def test_scale_mutation_and_chunked_rows(output_type):
    packed = torch.arange(256, dtype=torch.uint8).repeat(17, 1).view(torch.int8)
    scales = torch.linspace(-.245, .938, 17)
    device_packed, device_scales = packed.to("mps"), scales.to("mps")
    for factor in (1., -.5, 2.):
        device_scales.copy_(scales * factor)
        chunks = [unpack_int4_scaled(device_packed[i:i + 5], device_scales[i:i + 5], output_type) for i in range(0, 17, 5)]
        assert_exact(torch.cat(chunks), reference(packed, scales * factor, output_type))


@requires_mps
@pytest.mark.parametrize("rows,columns", [(0, 17), (3, 0), (0, 0)])
def test_empty(rows, columns):
    packed = torch.empty((rows, columns), device="mps", dtype=torch.int8)
    actual = unpack_int4_scaled(packed, torch.empty(rows))
    assert actual.shape == (rows, columns * 2)
    assert actual.dtype == torch.bfloat16


def test_cpu_declines():
    with pytest.raises(MPSINT4UnsupportedError, match="requires MPS"):
        unpack_int4_scaled(torch.zeros((2, 4), dtype=torch.int8), torch.ones(2))


def test_disable_environment(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_DISABLE_MPS", "1")
    with pytest.raises(MPSINT4UnsupportedError, match="disabled by environment"):
        unpack_int4_scaled(torch.zeros((2, 4), dtype=torch.int8), torch.ones(2))


@requires_mps
def test_unsupported_contracts():
    packed = torch.zeros((2, 4), dtype=torch.int8, device="mps")
    with pytest.raises(MPSINT4UnsupportedError, match="2D packed"):
        unpack_int4_scaled(packed.unsqueeze(0), torch.ones(2))
    with pytest.raises(MPSINT4UnsupportedError, match="2D packed"):
        unpack_int4_scaled(packed.float(), torch.ones(2))
    with pytest.raises(MPSINT4UnsupportedError, match="scale per row"):
        unpack_int4_scaled(packed, torch.ones(1))
    with pytest.raises(MPSINT4UnsupportedError, match="scale per row"):
        unpack_int4_scaled(packed, torch.ones(2, dtype=torch.int32))
    with pytest.raises(MPSINT4UnsupportedError, match="output dtype"):
        unpack_int4_scaled(packed, torch.ones(2), torch.int32)
    with pytest.raises(MPSINT4UnsupportedError, match="inference-only"):
        unpack_int4_scaled(packed, torch.ones(2, requires_grad=True))


def test_compile_failure_cached(monkeypatch):
    calls = []
    def fail(source):
        calls.append(source)
        raise RuntimeError("injected unsupported compiler")
    monkeypatch.setattr(int4_backend, "_library", None)
    monkeypatch.setattr(int4_backend, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", fail)
    for _ in range(2):
        with pytest.raises(MPSINT4UnsupportedError, match="compilation failed"):
            int4_backend._get_library()
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_compile_oom_not_cached(monkeypatch, error_type):
    error = error_type("MPS backend out of memory (injected)")
    def fail(source):
        raise error
    monkeypatch.setattr(int4_backend, "_library", None)
    monkeypatch.setattr(int4_backend, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", fail)
    with pytest.raises(error_type) as caught:
        int4_backend._get_library()
    assert caught.value is error
    assert int4_backend._compile_error is None


@requires_mps
def test_launch_failure_cached(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected unsupported launch")
    monkeypatch.setattr(int4_backend, "_launch_error", None)
    monkeypatch.setattr(int4_backend, "_get_library", lambda: SimpleNamespace(unpack_int4_bfloat=fail))
    packed = torch.zeros((2, 4), dtype=torch.int8, device="mps")
    with pytest.raises(MPSINT4UnsupportedError, match="launch failed"):
        unpack_int4_scaled(packed, torch.ones(2))
    with pytest.raises(MPSINT4UnsupportedError, match="disabled after launch failure"):
        unpack_int4_scaled(packed, torch.ones(2))
    assert len(calls) == 1


@requires_mps
@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_launch_oom_not_cached(monkeypatch, error_type):
    error = error_type("MPS backend out of memory (injected)")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(int4_backend, "_launch_error", None)
    monkeypatch.setattr(int4_backend, "_get_library", lambda: SimpleNamespace(unpack_int4_bfloat=fail))
    packed = torch.zeros((2, 4), dtype=torch.int8, device="mps")
    with pytest.raises(error_type) as caught:
        unpack_int4_scaled(packed, torch.ones(2))
    assert caught.value is error
    assert int4_backend._launch_error is None
