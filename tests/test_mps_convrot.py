"""SIMD regular ConvRot basis, FP32 safety and inference fallback contracts."""

from types import SimpleNamespace

import pytest
import torch

import comfy_kitchen.backends.mps.convrot as convrot
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
TYPE_PAIRS = [(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16),
              (torch.float32, torch.float32), (torch.float16, torch.float32),
              (torch.bfloat16, torch.float32)]


@pytest.fixture(autouse=True)
def enable_experimental_rotation(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_MPS_ROTATION", "1")


def test_default_disabled(monkeypatch):
    monkeypatch.delenv("COMFY_KITCHEN_MPS_ROTATION")
    assert convrot.try_rotate(None, 256) is None


@requires_mps
@pytest.mark.parametrize("group", [64, 256])
@pytest.mark.parametrize("input_dtype,output_dtype", TYPE_PAIRS)
def test_every_basis_vector(group, input_dtype, output_dtype):
    actual = convrot.try_rotate(torch.eye(group, device="mps", dtype=input_dtype), group, output_dtype)
    assert actual is not None
    torch.testing.assert_close(actual.cpu(), _build_hadamard(group).to(output_dtype), rtol=0, atol=0)


@requires_mps
@pytest.mark.parametrize("group", [64, 256])
@pytest.mark.parametrize("input_dtype,output_dtype", TYPE_PAIRS)
@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (65, 17)])
def test_random_and_multiple_groups(group, input_dtype, output_dtype, shape):
    generator = torch.Generator().manual_seed(361)
    x = torch.randn(shape[0], shape[1] * group, generator=generator).to(input_dtype)
    reference = _rotate_activation(x.float(), _build_hadamard(group), group).to(output_dtype)
    actual = convrot.try_rotate(x.to("mps"), group, output_dtype)
    assert actual is not None and actual.dtype == output_dtype and actual.shape == x.shape
    tolerance = {torch.float32: 3e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[output_dtype]
    torch.testing.assert_close(actual.cpu(), reference, rtol=tolerance, atol=tolerance)


@requires_mps
@pytest.mark.parametrize("group", [64, 256])
def test_fp16_rotation_retains_wide_values_for_safe_int8(group):
    x = (_build_hadamard(group)[0].sign() * 30000).half().reshape(1, group).to("mps")
    actual = convrot.try_rotate(x, group, torch.float32)
    expected = torch.zeros((1, group))
    expected[0, 0] = 30000 * group**0.5
    assert actual is not None and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    # Existing INT8 overflow regression: final scale restores a finite half.
    y = torch.nn.functional.linear(actual, torch.ones((4, group), device="mps")) * 0.001
    torch.testing.assert_close(y.half().cpu(), torch.full((1, 4), 30 * group**0.5, dtype=torch.float16), rtol=0, atol=0)


@requires_mps
@pytest.mark.parametrize("group", [64, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_large_finite_constant_not_overflowed_by_unnormalized_butterfly(group, dtype):
    # A regular normalized Hadamard preserves constants. Normalizing only
    # after a256-element butterfly would unnecessarily grow3e37 by16 first.
    x = torch.full((3, group * 5), 3e37, dtype=dtype, device="mps")
    actual = convrot.try_rotate(x, group)
    assert actual is not None and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), x.cpu(), rtol=3e-6 if dtype == torch.float32 else .01, atol=0)


@requires_mps
@pytest.mark.parametrize("group", [64, 256])
def test_high_dynamic_range_basis(group):
    x = (_build_hadamard(group)[5].sign() * 1e35).reshape(1, group).to("mps")
    actual = convrot.try_rotate(x, group)
    reference = _rotate_activation(x.cpu(), _build_hadamard(group), group)
    assert actual is not None and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), reference, rtol=3e-6, atol=1e30)


@requires_mps
def test_contiguous_storage_offset_and_default_dtype():
    x = torch.randn(5, 256, dtype=torch.float16, device="mps")[2:]
    actual = convrot.try_rotate(x, 256)
    assert actual is not None and actual.dtype == x.dtype
    reference = _rotate_activation(x.cpu().float(), _build_hadamard(256), 256).half()
    torch.testing.assert_close(actual.cpu(), reference, rtol=.002, atol=.002)


@requires_mps
def test_unsupported_inputs_return_none():
    assert convrot.try_rotate(torch.randn(16, 256, device="mps").T, 256) is None
    assert convrot.try_rotate(torch.randn(3, 257, device="mps"), 256) is None
    assert convrot.try_rotate(torch.randn(3, 256, device="mps"), 16) is None
    assert convrot.try_rotate(torch.randn(3, 256, device="mps"), 256, torch.float16) is None
    assert convrot.try_rotate(torch.ones(3, 256, device="mps", dtype=torch.int32), 256) is None
    assert convrot.try_rotate(torch.randn(3, 256, device="mps", requires_grad=True), 256) is None
    assert convrot.try_rotate(torch.empty(0, 256, device="mps"), 256) is None
    assert convrot.try_rotate(torch.tensor(1.0, device="mps"), 256) is None


def test_cpu_declines():
    assert convrot.try_rotate(torch.randn(3, 256), 256) is None


def test_environment_disables(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_DISABLE_MPS", "1")
    assert convrot.try_rotate(None, 256) is None


def test_uint32_guard():
    fake = SimpleNamespace(dtype=torch.float32, device=torch.device("mps"), ndim=2,
                           shape=(2**24, 256), layout=torch.strided, requires_grad=False,
                           is_contiguous=lambda: True, numel=lambda: 2**32)
    assert convrot.try_rotate(fake, 256) is None


def test_compile_failure_cached(monkeypatch):
    calls = []
    def fail(source):
        calls.append(source)
        raise RuntimeError("injected unsupported compiler")
    monkeypatch.setattr(convrot, "_library", None)
    monkeypatch.setattr(convrot, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", fail)
    assert convrot._get_library() is None
    assert convrot._get_library() is None
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_compile_oom_propagates(monkeypatch, error_type):
    error = error_type("MPS out of memory (injected)")
    def fail(source):
        raise error
    monkeypatch.setattr(convrot, "_library", None)
    monkeypatch.setattr(convrot, "_compile_error", None)
    monkeypatch.setattr(torch.mps, "compile_shader", fail)
    with pytest.raises(error_type) as caught:
        convrot._get_library()
    assert caught.value is error and convrot._compile_error is None


@requires_mps
def test_launch_failure_cached(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected unsupported launch")
    monkeypatch.setattr(convrot, "_launch_error", None)
    monkeypatch.setattr(convrot, "_get_library", lambda: SimpleNamespace(rotation_float_float_256=fail))
    x = torch.ones((3, 256), device="mps")
    assert convrot.try_rotate(x, 256) is None
    assert convrot.try_rotate(x, 256) is None
    assert len(calls) == 1


@requires_mps
@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
def test_launch_oom_propagates(monkeypatch, error_type):
    error = error_type("MPS out of memory (injected)")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(convrot, "_launch_error", None)
    monkeypatch.setattr(convrot, "_get_library", lambda: SimpleNamespace(rotation_float_float_256=fail))
    with pytest.raises(error_type) as caught:
        convrot.try_rotate(torch.ones((3, 256), device="mps"), 256)
    assert caught.value is error and convrot._launch_error is None
