"""Upstream activations and residuals must survive the custom MPS/fallback path."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", [None, "gelu_tanh", "swiglu", "situ", "rms_norm"])
@pytest.mark.parametrize("convrot", [False, True])
def test_fallback_activation_and_residual(monkeypatch, device, dtype, activation, convrot):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    if device == "cpu":
        monkeypatch.setattr(quantization, "_device_has_int8_mm", lambda _: False)
    torch.manual_seed(21)
    k, n = 256, 7
    x = torch.randn(2, 3, k * (2 if activation in ("swiglu", "situ") else 1), device=device, dtype=dtype)
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
    elif activation == "situ":
        hidden, gate = x.chunk(2, dim=-1)
        activated = (hidden / 25.0).tanh() * 25.0
        activated *= (gate / 4.0).tanh() * 4.0 * torch.sigmoid(gate)
    elif activation == "gelu_tanh":
        activated = torch.nn.functional.gelu(x, approximate="tanh")
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


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("convrot", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_modulated_projection_keeps_apple_fallback(monkeypatch, device, dtype, convrot, normalize):
    """New upstream modulation APIs must retain the fork's overflow-safe GEMM."""
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    if device == "cpu":
        monkeypatch.setattr(quantization, "_device_has_int8_mm", lambda _: False)
    torch.manual_seed(22)
    k, n = 256, 7
    x = torch.randn(2, 3, k, device=device, dtype=dtype)
    weight = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    weight_scale = torch.rand(n, device=device) * 0.001
    bias = torch.randn(n, device=device, dtype=dtype)
    modulation = torch.randn(1, 1, k, device=device, dtype=dtype) * 0.1
    norm_weight = torch.randn(k, device=device, dtype=dtype)
    normalized = torch.nn.functional.rms_norm(x, (k,), norm_weight, 1e-5) if normalize else x
    expected = ck.int8_linear(
        normalized * (1.0 + modulation), weight, weight_scale, bias, dtype, convrot=convrot
    )
    if normalize:
        actual = ck.int8_linear_rms_modulated(
            x, norm_weight, 1e-5, modulation, weight, weight_scale, bias, dtype, convrot=convrot
        )
    else:
        actual = ck.int8_linear_modulated(
            x, modulation, weight, weight_scale, bias, dtype, convrot=convrot
        )
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
