"""Native CUDA must accept valid offset views and incomplete FP8 vectors."""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.constraints import validate_function_call
from comfy_kitchen.float_utils import fp4_x2_to_f32
from .conftest import cuda_backend_available, get_capable_backends


@pytest.mark.skipif(not cuda_backend_available(), reason="compiled CUDA backend required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("offset", [0, 1, 3])
def test_native_fp8_offset_and_tail_roundtrip(dtype, fp8_dtype, length, offset):
    # Offset and length vary independently: cloning an odd view must not hide
    # the separate final-vector overrun that used to affect contiguous inputs.
    source = torch.linspace(-3, 3, length + offset, device="cuda", dtype=dtype)[offset:]
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    with ck.use_backend("eager"):
        reference = ck.quantize_per_tensor_fp8(source, scale, fp8_dtype)
    with ck.use_backend("cuda"):
        quantized = ck.quantize_per_tensor_fp8(source, scale, fp8_dtype)
    assert torch.equal(quantized.view(torch.uint8), reference.view(torch.uint8))

    # Quantized output itself is allocated aligned; create an independently
    # offset FP8 view to exercise the dequantizer's input-address guard.
    storage = torch.empty(length + offset, device="cuda", dtype=fp8_dtype)
    storage.view(torch.uint8)[offset:] = reference.view(torch.uint8)
    view = storage[offset:]
    with ck.use_backend("eager"):
        expected = ck.dequantize_per_tensor_fp8(view, scale, output_type=dtype)
    with ck.use_backend("cuda"):
        actual = ck.dequantize_per_tensor_fp8(view, scale, output_type=dtype)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_triton_nvfp4_capability_and_public_dispatch():
    from comfy_kitchen.backends.triton import _build_constraints
    constraints = _build_constraints()["quantize_nvfp4"]
    assert constraints.min_compute_capability == (10, 0)
    # Other required tensor arguments are supplied for the positive hardware
    # contract; the real registry/public calls below use the actual GPU.
    x = torch.linspace(-4, 4, 32 * 128, device="cuda", dtype=torch.bfloat16).reshape(32, 128)
    scale = x.abs().amax().float() / (448 * 6)
    args = {"x": x, "per_tensor_scale": scale}
    rejected = validate_function_call(constraints, args, compute_capability=(8, 9))
    assert not rejected.success and rejected.failed_param == "__hardware__"
    assert validate_function_call(constraints, args, compute_capability=(10, 0)).success
    backends = get_capable_backends("quantize_nvfp4", "cuda")
    if torch.cuda.get_device_capability() < (10, 0):
        assert "triton" not in backends
    else:
        assert "triton" in backends
    assert "eager" in backends
    with ck.use_backend("eager"):
        reference, reference_scales = ck.quantize_nvfp4(x, scale)
    for backend in backends:
        with ck.use_backend(backend):
            packed, scales = ck.quantize_nvfp4(x, scale)
        assert packed.shape == reference.shape and scales.shape == reference_scales.shape
        torch.testing.assert_close(scales.float(), reference_scales.float(), rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(fp4_x2_to_f32(packed), fp4_x2_to_f32(reference), rtol=1e-2, atol=2.0)
        assert packed.dtype == torch.uint8 and torch.isfinite(scales.float()).all()
    public, public_scales = ck.quantize_nvfp4(x, scale)
    assert public.shape == reference.shape
    torch.testing.assert_close(public_scales.float(), reference_scales.float(), rtol=1e-5, atol=1e-3)
    torch.cuda.synchronize()
