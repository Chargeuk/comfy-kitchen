"""Exact FP32 multiply/add rounding and safe fallback for the INT8 epilogue."""

from types import SimpleNamespace

import pytest
import torch

import comfy_kitchen.backends.mps.int8 as epilogue_backend
from comfy_kitchen.backends.mps.int8 import MPSINT8EpilogueUnsupportedError, int8_epilogue

TYPES = [torch.float32, torch.float16, torch.bfloat16]
requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")


def reference(y, scale, bias, dtype):
    result = y.float() * scale.float().reshape(-1)
    if bias is not None:
        result = result + bias.float().reshape(-1)
    return result.to(dtype)


def assert_exact(actual, expected):
    actual = actual.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@requires_mps
@pytest.mark.parametrize("input_dtype", TYPES)
@pytest.mark.parametrize("output_dtype", TYPES)
@pytest.mark.parametrize("scalar_scale", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
def test_dtypes_scaling_and_bias(input_dtype, output_dtype, scalar_scale, has_bias):
    generator = torch.Generator().manual_seed(1842)
    y = (torch.randn(7, 17, generator=generator) * 61.27).to(input_dtype)
    scale = torch.rand(1 if scalar_scale else 17, generator=generator) * 0.01337
    bias = torch.randn(17, generator=generator).to(input_dtype) if has_bias else None
    actual = int8_epilogue(y.to("mps"), scale.to("mps"), None if bias is None else bias.to("mps"), output_dtype)
    assert_exact(actual, reference(y, scale, bias, output_dtype))


@requires_mps
@pytest.mark.parametrize("output_dtype", TYPES)
def test_separate_fp32_multiply_add_not_fma(output_dtype):
    # (1 + 2**-23) * (1 - 2**-23) rounds to1 in FP32. With -1 bias,
    # separate ops produce0; FMA would instead produce -2**-46.
    y = torch.tensor([[1 + 2**-23, 129.9375, 131072.0, 480000.0]])
    scale = torch.tensor([1 - 2**-23, 0.001234567, 0.001, 0.001])
    bias = torch.tensor([-1.0, -0.031234, 0.0, 0.0])
    actual = int8_epilogue(y.to("mps"), scale.to("mps"), bias.to("mps"), output_dtype)
    assert_exact(actual, reference(y, scale, bias, output_dtype))
    assert torch.isfinite(actual).all()


@requires_mps
@pytest.mark.parametrize("output_dtype", TYPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_special_values(output_dtype, has_bias):
    y = torch.tensor([[0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-35, 1e35]])
    scale = torch.tensor([-1.0, 1.0, 0.0, -0.0, 1.0, 1e-5, 1e5])
    bias = torch.tensor([-0.0, -0.0, 0.0, 0.0, 1.0, 1e-40, -1e35]) if has_bias else None
    y, scale = y.to("mps"), scale.to("mps")
    bias = None if bias is None else bias.to("mps")
    actual = int8_epilogue(y, scale, bias, output_dtype)
    # MPS flushes FP32 subnormals in the prior separate PyTorch operations
    # too. Compare these exceptional values to the existing device path;
    # ordinary finite-value and FMA-rounding tests retain exact CPU checks.
    assert_exact(actual, reference(y, scale, bias, output_dtype).cpu())


@requires_mps
@pytest.mark.parametrize("output_dtype", TYPES)
def test_chunk_store_strided_scales_and_storage_offset(output_dtype):
    y = torch.arange(21, dtype=torch.float32).reshape(3, 7).to("mps")
    # Contiguous chunk with nonzero input/output storage offsets.
    y = torch.cat([y, y], dim=0)[3:]
    out_storage = torch.full((4, 13), -13.0, device="mps", dtype=output_dtype)
    out = out_storage[1:]
    scale = torch.arange(14, dtype=torch.float32, device="mps")[1::2] / 10
    bias = torch.arange(14, dtype=torch.float32, device="mps")[::2]
    actual = int8_epilogue(y, scale, bias, output_dtype, out=out, column_offset=3)
    assert actual is out
    expected = torch.full((3, 13), -13.0, dtype=output_dtype)
    expected[:, 3:10] = reference(y.cpu(), scale.cpu(), bias.cpu(), output_dtype)
    assert_exact(actual, expected)
    assert torch.equal(out_storage[0].cpu(), torch.full((13,), -13.0, dtype=output_dtype))


@requires_mps
def test_scale_and_bias_mutation_not_cached():
    y = torch.ones((3, 7), device="mps")
    scale = torch.ones(7, device="mps")
    bias = torch.zeros(7, device="mps")
    int8_epilogue(y, scale, bias)
    scale.mul_(3)
    bias.add_(2)
    assert_exact(int8_epilogue(y, scale, bias), torch.full((3, 7), 5, dtype=torch.float16))


@requires_mps
def test_overlapping_output_declines_before_launch():
    storage = torch.arange(26, device="mps", dtype=torch.float32)
    y, out = storage[:14].view(2, 7), storage.view(2, 13)
    before = storage.clone()
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="overlap"):
        int8_epilogue(y, torch.ones(7, device="mps"), output_type=torch.float32, out=out, column_offset=3)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@requires_mps
@pytest.mark.parametrize("shape", [(0, 7), (3, 0), (0, 0)])
def test_empty(shape):
    result = int8_epilogue(torch.empty(shape, device="mps"), torch.ones((), device="mps"))
    assert result.shape == shape and result.dtype == torch.float16


def test_cpu_declines():
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="requires MPS"):
        int8_epilogue(torch.ones((2, 7)), torch.ones(7))


def test_environment_disables(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_DISABLE_MPS", "1")
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="disabled by environment"):
        int8_epilogue(torch.ones((2, 7)), torch.ones(7))


@requires_mps
def test_unsupported_contracts():
    y, scale = torch.ones((2, 7), device="mps"), torch.ones(7, device="mps")
    for invalid_y in (y.T, y.unsqueeze(0), y.to(torch.int32)):
        with pytest.raises(MPSINT8EpilogueUnsupportedError):
            int8_epilogue(invalid_y, scale)
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="scales"):
        int8_epilogue(y, scale.half())
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="scales"):
        int8_epilogue(y, scale[:3])
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="bias"):
        int8_epilogue(y, scale, scale[:3])
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="inference-only"):
        int8_epilogue(y.requires_grad_(), scale)
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="output matrix"):
        int8_epilogue(y.detach(), scale, out=torch.empty((2, 7), device="mps", dtype=torch.float16), column_offset=1)
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="offset"):
        int8_epilogue(y.detach(), scale, column_offset=-1)


def test_compile_failure_cached(monkeypatch):
    calls = []
    def fail(source):
        calls.append(source)
        raise RuntimeError("injected unsupported compiler")
    monkeypatch.setattr(epilogue_backend, "_library", None)
    monkeypatch.setattr(epilogue_backend, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", fail)
    for _ in range(2):
        with pytest.raises(MPSINT8EpilogueUnsupportedError, match="compilation failed"):
            epilogue_backend._get_library()
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_compile_oom_not_cached(monkeypatch, error_type):
    error = error_type("MPS out of memory (injected)")
    def fail(source):
        raise error
    monkeypatch.setattr(epilogue_backend, "_library", None)
    monkeypatch.setattr(epilogue_backend, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", fail)
    with pytest.raises(error_type) as caught:
        epilogue_backend._get_library()
    assert caught.value is error
    assert epilogue_backend._compile_error is None


@requires_mps
def test_launch_failure_cached(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected unsupported launch")
    monkeypatch.setattr(epilogue_backend, "_launch_error", None)
    monkeypatch.setattr(epilogue_backend, "_get_library", lambda: SimpleNamespace(epilogue_float_half=fail))
    args = torch.ones((2, 7), device="mps"), torch.ones(7, device="mps")
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="launch failed"):
        int8_epilogue(*args)
    with pytest.raises(MPSINT8EpilogueUnsupportedError, match="disabled after"):
        int8_epilogue(*args)
    assert len(calls) == 1


@requires_mps
@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_launch_oom_propagates(monkeypatch, error_type):
    error = error_type("MPS out of memory (injected)")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(epilogue_backend, "_launch_error", None)
    monkeypatch.setattr(epilogue_backend, "_get_library", lambda: SimpleNamespace(epilogue_float_half=fail))
    with pytest.raises(error_type) as caught:
        int8_epilogue(torch.ones((2, 7), device="mps"), torch.ones(7, device="mps"))
    assert caught.value is error
    assert epilogue_backend._launch_error is None


@requires_mps
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("convrot", [False, True])
def test_bounded_linear_chunks_match_existing_mps_epilogue(monkeypatch, dtype, convrot):
    from comfy_kitchen.backends.eager import quantization

    generator = torch.Generator().manual_seed(71)
    x = torch.randn(7, 256, generator=generator).to(dtype).to("mps")
    weight = torch.randint(-127, 128, (11, 256), generator=generator, dtype=torch.int8).to("mps")
    scale = torch.rand(11, generator=generator).to("mps") * 0.001
    bias = torch.randn(11, generator=generator).to(dtype).to("mps")
    monkeypatch.setattr(quantization, "_FALLBACK_CHUNK_BYTES", 256 * 4 * 3)
    def run():
        return quantization._int8_linear_dequant(x, weight, scale, bias, dtype, convrot, 256)
    monkeypatch.setenv("COMFY_KITCHEN_DISABLE_MPS", "1")
    expected = run()
    monkeypatch.delenv("COMFY_KITCHEN_DISABLE_MPS")
    torch.testing.assert_close(run(), expected, rtol=0, atol=0)
