"""Native MPS registry dispatch and compatibility fallbacks."""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import mps


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mps_fp8_registry_and_compile_fallback(monkeypatch):
    raw = torch.arange(256, dtype=torch.uint8)
    x = raw.to("mps").view(torch.float8_e4m3fn)
    scale = torch.tensor(0.5, device="mps")
    kwargs = dict(x=x, scale=scale, output_type=torch.bfloat16)
    assert ck.registry.get_capable_backend("dequantize_per_tensor_fp8", kwargs) == "mps"
    ref = raw.view(torch.float8_e4m3fn).bfloat16() * 0.5
    result = ck.dequantize_per_tensor_fp8(**kwargs)
    torch.testing.assert_close(result.cpu(), ref, rtol=0, atol=0, equal_nan=True)

    def unavailable(*args, **kwargs):
        raise mps.fp8.MPSFP8UnsupportedError("test compilation failure")

    monkeypatch.setattr(mps.fp8, "dequantize_per_tensor_fp8", unavailable)
    result = ck.dequantize_per_tensor_fp8(**kwargs)
    torch.testing.assert_close(result.cpu(), ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mps_int8_dispatch_half_overflow():
    x = torch.ones((1, 1024), device="mps", dtype=torch.float16)
    weight = torch.full((4, 1024), 127, device="mps", dtype=torch.int8)
    scale = torch.tensor(0.001, device="mps")
    assert ck.registry.get_capable_backend("int8_linear", dict(x=x, weight=weight, weight_scale=scale)) == "mps"
    out = ck.int8_linear(x, weight, scale, out_dtype=torch.float16)
    torch.testing.assert_close(out.cpu(), torch.full((1, 4), 130.048).half(), rtol=0, atol=0)
