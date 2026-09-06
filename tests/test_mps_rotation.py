"""Correctness and eligibility tests; no performance assertions."""
import pytest
import torch
from types import SimpleNamespace

from comfy_kitchen.backends.mps import rotation
from comfy_kitchen.backends.mps.rotation import try_regular_rotation
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


@pytest.fixture(autouse=True)
def enable_experimental_rotation(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_MPS_ROTATION", "1")


def test_default_requests_fallback(monkeypatch):
    monkeypatch.delenv("COMFY_KITCHEN_MPS_ROTATION")
    assert try_regular_rotation(SimpleNamespace(), 256) is None


def test_cpu_requests_fallback():
    assert try_regular_rotation(torch.randn(2, 16), 16) is None


def test_environment_disable(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_MPS_ROTATION", "0")
    assert try_regular_rotation(torch.randn(2, 16), 16) is None


def test_compile_failure_is_cached(monkeypatch, caplog):
    calls = []

    def fail_compile(source):
        calls.append(source)
        raise RuntimeError("test compiler unavailable")

    monkeypatch.setattr(rotation, "_LIBRARIES", {})
    monkeypatch.setattr(rotation, "_FAILED", set())
    monkeypatch.setattr(torch.mps, "compile_shader", fail_compile, raising=False)
    monkeypatch.setenv("COMFY_KITCHEN_MPS_ROTATION", "1")
    # Eligibility-only fake avoids requiring an Apple GPU for fallback tests.
    x = SimpleNamespace(
        device=torch.device("mps"), dtype=torch.float32, ndim=2,
        shape=(2, 16), is_contiguous=lambda: True, requires_grad=False,
        numel=lambda: 32,
    )
    assert try_regular_rotation(x, 16) is None
    assert try_regular_rotation(x, 16) is None
    assert len(calls) == 1
    assert sum("using fallback" in record.message for record in caplog.records) == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group", [4, 16, 64, 256])
@pytest.mark.parametrize("rows,groups", [(1, 1), (3, 5)])
def test_regular_rotation(dtype, group, rows, groups):
    # Include partial threadgroups and multiple groups per row.
    torch.manual_seed(77)
    x = torch.randn(rows, group * groups, dtype=dtype).to("mps")
    output = try_regular_rotation(x, group)
    assert output is not None, "eligible MPS shader failed to compile or dispatch"
    reference = _rotate_activation(
        x.cpu().float(), _build_hadamard(group), group
    ).to(dtype)
    tolerance = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(output.cpu(), reference, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_ineligible_mps_inputs():
    assert try_regular_rotation(torch.empty(0, 16, device="mps"), 16) is None
    assert try_regular_rotation(torch.randn(3, 17, device="mps"), 16) is None
    assert try_regular_rotation(torch.randn(3, 8, device="mps"), 8) is None
    assert try_regular_rotation(torch.randn(16, 16, device="mps").T, 16) is None
    assert try_regular_rotation(torch.randn(3, 16, device="mps", requires_grad=True), 16) is None
    assert try_regular_rotation(torch.empty(65, 256, device="mps"), 64) is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_basis_order_and_fp16_overflow():
    # Exact basis tests expose permutations/sign changes hidden by norm checks.
    x = torch.eye(16, device="mps")
    torch.testing.assert_close(try_regular_rotation(x, 16).cpu(), _build_hadamard(16))
    # Constant inputs are unchanged by this *regular* normalized Hadamard.
    x = torch.full((2, 256), 30000.0, dtype=torch.float16, device="mps")
    output = try_regular_rotation(x, 256)
    assert output is not None
    torch.testing.assert_close(output.cpu(), x.cpu(), atol=0, rtol=0)
