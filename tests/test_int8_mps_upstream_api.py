"""Upstream activations and residuals must survive the custom MPS/fallback path."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", [None, "swiglu", "rms_norm"])
@pytest.mark.parametrize("convrot", [False, True])
def test_fallback_activation_and_residual(monkeypatch, device, dtype, activation, convrot):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    if device == "cpu":
        monkeypatch.setattr(quantization, "_device_has_int8_mm", lambda _: False)
    torch.manual_seed(21)
    k, n = 256, 7
    x = torch.randn(2, 3, k * (2 if activation == "swiglu" else 1), device=device, dtype=dtype)
    weight = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    scale = torch.rand(n, device=device) * 0.001
    bias = torch.randn(n, device=device, dtype=dtype)
    norm_weight = torch.randn(k, device=device, dtype=dtype)
    # Residual dtype differs deliberately: the API promises the output dtype.
    residual = torch.randn(2, 3, n, device=device, dtype=torch.float32)
    residual_scale = torch.randn(n, device=device, dtype=torch.float32)
    if activation == "swiglu":
        gate, up = x.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
    elif activation == "rms_norm":
        activated = torch.nn.functional.rms_norm(x, (k,), norm_weight, eps=1e-5)
    else:
        activated = x
    plain = ck.int8_linear(activated, weight, scale, bias, dtype, convrot=convrot)
    expected = torch.addcmul(residual.to(dtype), plain, residual_scale.to(dtype))
    actual = ck.int8_linear(
        x, weight, scale, bias, dtype, convrot=convrot,
        input_act=activation,
        input_act_weight=norm_weight if activation == "rms_norm" else None,
        input_act_eps=1e-5 if activation == "rms_norm" else 0.0,
        residual=residual, residual_scale=residual_scale,
    )
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="residual requires residual_scale"):
        ck.int8_linear(activated, weight, scale, bias, dtype, residual=residual)
