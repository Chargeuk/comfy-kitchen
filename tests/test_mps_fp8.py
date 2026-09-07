"""Exhaustive FP8 byte decoding and stride/rounding regression coverage."""

import pytest
import torch
from types import SimpleNamespace

import comfy_kitchen.backends.mps.fp8 as fp8_backend

from comfy_kitchen.backends.mps.fp8 import (
    MPSFP8UnsupportedError,
    decode_fp8,
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


@requires_mps
@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("output_type", OUTPUTS)
def test_unscaled_every_byte(dtype, output_type):
    cpu = torch.arange(256, dtype=torch.uint8).view(dtype)
    expected = cpu.to(output_type)
    actual = decode_fp8(_mps_fp8(cpu), output_type).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@requires_mps
@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand", "offset", "empty", "scalar"])
def test_unscaled_layouts(dtype, output_type, layout):
    def view(raw):
        if layout == "transpose":
            return raw.T
        if layout == "slice":
            return raw[1::3, 2::2]
        if layout == "expand":
            return raw[:1].expand(7, 16)
        if layout == "offset":
            return raw[3:]
        if layout == "empty":
            return raw[:0]
        return raw[0, 1]
    raw = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    cpu = view(raw).view(dtype)
    device = view(raw.to("mps")).view(dtype)
    actual = decode_fp8(device, output_type)
    assert actual.shape == cpu.shape and actual.dtype == output_type
    torch.testing.assert_close(actual.cpu(), cpu.to(output_type), rtol=0, atol=0, equal_nan=True)


@requires_mps
def test_unscaled_does_not_allocate_scale_or_call_scaled_decoder(monkeypatch):
    cpu = torch.arange(256, dtype=torch.uint8).view(FORMATS[0])
    x = _mps_fp8(cpu)
    def unexpected(*args, **kwargs):
        raise AssertionError("unscaled decoding must not create or prepare a scale")
    monkeypatch.setattr(torch, "ones", unexpected)
    monkeypatch.setattr(fp8_backend, "dequantize_per_tensor_fp8", unexpected)
    torch.testing.assert_close(decode_fp8(x).cpu(), cpu.float(), rtol=0, atol=0, equal_nan=True)


def test_unscaled_cpu_declines():
    with pytest.raises(MPSFP8UnsupportedError, match="requires MPS"):
        decode_fp8(torch.zeros(2).to(FORMATS[0]))


def test_unscaled_disable_environment(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_DISABLE_MPS", "1")
    with pytest.raises(MPSFP8UnsupportedError, match="disabled by environment"):
        decode_fp8(torch.zeros(2).to(FORMATS[0]))


def test_unscaled_uint32_guard(monkeypatch):
    monkeypatch.setattr(torch.mps, "compile_shader", lambda source: None, raising=False)
    fake = SimpleNamespace(device=torch.device("mps"), dtype=FORMATS[0], layout=torch.strided,
                           requires_grad=False, numel=lambda: 2**32)
    with pytest.raises(MPSFP8UnsupportedError, match="shader index range"):
        decode_fp8(fake)


@requires_mps
def test_unscaled_unsupported_contracts_and_gradients():
    x = _mps_fp8(torch.zeros(2).to(FORMATS[0]))
    with pytest.raises(MPSFP8UnsupportedError, match="unsupported FP8 input"):
        decode_fp8(x.view(torch.uint8))
    with pytest.raises(MPSFP8UnsupportedError, match="output dtype"):
        decode_fp8(x, torch.int32)
    with pytest.raises(MPSFP8UnsupportedError, match="inference-only"):
        decode_fp8(x.requires_grad_())


def test_shared_compile_failure_cached(monkeypatch):
    calls = []
    def failing_compile(source):
        calls.append(source)
        raise RuntimeError("injected unsupported compiler")
    monkeypatch.setattr(fp8_backend, "_library", None)
    monkeypatch.setattr(fp8_backend, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", failing_compile, raising=False)
    for _ in range(2):
        with pytest.raises(MPSFP8UnsupportedError, match="compilation failed"):
            fp8_backend._get_library()
    assert len(calls) == 1
    for dtype in ("float", "half", "bfloat"):
        assert f"kernel void decode_{dtype}(" in calls[0]
        assert f"kernel void decode_unscaled_{dtype}(" in calls[0]


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_shared_compile_oom_not_cached(monkeypatch, error_type):
    error = error_type("MPS backend out of memory (injected)")
    def failing_compile(source):
        raise error
    monkeypatch.setattr(fp8_backend, "_library", None)
    monkeypatch.setattr(fp8_backend, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", failing_compile, raising=False)
    with pytest.raises(error_type) as caught:
        fp8_backend._get_library()
    assert caught.value is error
    assert fp8_backend._compile_error is None


@requires_mps
def test_unscaled_launch_failure_disables_both_paths(monkeypatch):
    calls = []
    def failing_kernel(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected unsupported unscaled launch")
    monkeypatch.setattr(fp8_backend, "_launch_error", None)
    monkeypatch.setattr(fp8_backend, "_get_library", lambda: SimpleNamespace(decode_unscaled_float=failing_kernel))
    x = _mps_fp8(torch.zeros(2).to(FORMATS[0]))
    with pytest.raises(MPSFP8UnsupportedError, match="launch failed"):
        decode_fp8(x)
    with pytest.raises(MPSFP8UnsupportedError, match="disabled after launch failure"):
        decode_fp8(x)
    with pytest.raises(MPSFP8UnsupportedError, match="disabled after launch failure"):
        dequantize_per_tensor_fp8(x, torch.ones(()))
    assert len(calls) == 1


@requires_mps
@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_unscaled_launch_oom_is_not_fallback(monkeypatch, error_type):
    error = error_type("MPS backend out of memory (injected)")
    def failing_kernel(*args, **kwargs):
        raise error
    monkeypatch.setattr(fp8_backend, "_launch_error", None)
    monkeypatch.setattr(fp8_backend, "_get_library", lambda: SimpleNamespace(decode_unscaled_float=failing_kernel))
    x = _mps_fp8(torch.zeros(2).to(FORMATS[0]))
    with pytest.raises(error_type) as caught:
        decode_fp8(x)
    assert caught.value is error
    assert fp8_backend._launch_error is None


@requires_mps
@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("output_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["offset", "stride"])
def test_unscaled_vector_large_all_bytes_with_odd_tail(dtype, output_type, layout):
    count = fp8_backend._UNSCALED_VECTOR_MIN_ELEMENTS + 3
    raw = torch.arange(256, dtype=torch.uint8).repeat((count + 255) // 256)[:count]
    if layout == "offset":
        backing = torch.cat([torch.zeros(1, dtype=torch.uint8), raw, torch.zeros(1, dtype=torch.uint8)])
        cpu_raw, device_raw = backing[1:-1], backing.to("mps")[1:-1]
    else:
        backing = raw.repeat_interleave(2)
        cpu_raw, device_raw = backing[1::2], backing.to("mps")[1::2]
    expected = cpu_raw.view(dtype).to(output_type)
    actual = decode_fp8(device_raw.view(dtype), output_type).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@pytest.mark.parametrize("output_type", OUTPUTS)
@pytest.mark.parametrize("delta", [-1, 0, 3])
def test_unscaled_vector_threshold_and_launch_arguments(monkeypatch, output_type, delta):
    # Metadata-only fakes verify branch/launch arithmetic without allocating
    # large tensors or using the GPU; real byte accuracy is checked above.
    count = fp8_backend._UNSCALED_VECTOR_MIN_ELEMENTS + delta
    _check_unscaled_launch(monkeypatch, output_type, count)


@pytest.mark.parametrize("count", [2**31, 2**32 - 1])
def test_unscaled_vector_unsigned_count_bit_pattern(monkeypatch, count):
    _check_unscaled_launch(monkeypatch, torch.float16, count)


def _check_unscaled_launch(monkeypatch, output_type, count):
    monkeypatch.setattr(torch.mps, "compile_shader", lambda source: None, raising=False)
    calls = []
    raw = SimpleNamespace(contiguous=lambda: raw)
    fake = SimpleNamespace(device=torch.device("mps"), dtype=FORMATS[0], layout=torch.strided,
                           requires_grad=False, shape=(count,), numel=lambda: count, view=lambda _: raw)
    result = object()
    def capture(label):
        def kernel(*args, **kwargs):
            calls.append((label, args, kwargs))
        return kernel
    typename = fp8_backend._OUTPUTS[output_type]
    library = SimpleNamespace(**{f"decode_unscaled_{typename}": capture("scalar"),
                                 f"decode_unscaled4_{typename}": capture("vector")})
    monkeypatch.setattr(fp8_backend, "_launch_error", None)
    monkeypatch.setattr(fp8_backend, "_get_library", lambda: library)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: result)
    assert decode_fp8(fake, output_type) is result
    vector = output_type != torch.float32 and count >= fp8_backend._UNSCALED_VECTOR_MIN_ELEMENTS
    label, args, kwargs = calls[0]
    assert label == ("vector" if vector else "scalar")
    assert args[0] is raw and args[1] is result
    assert kwargs["threads"] == ((count + 3) // 4 if vector else count)
    assert kwargs["arg_casts"] == ({2: "int32", 3: "int32"} if vector else {2: "int32"})
    if vector:
        assert len(args) == 4 and args[3] & 0xFFFFFFFF == count
        assert -(2**31) <= args[3] <= 2**31 - 1
    else:
        assert len(args) == 3
