"""Bounded-memory regular ConvRot Hadamard rotation for Apple GPUs.

This is deliberately NOT a replacement for rotation with an arbitrary supplied
matrix. Call only where the basis is known to be int8_utils._build_hadamard.
No M5-only tensor operations or additional runtime dependencies are used.
"""
from __future__ import annotations

import logging
import os
import threading

import torch

_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_LIBRARIES: dict[torch.dtype, object] = {}
_FAILED: set[torch.dtype] = set()
_TYPES = {torch.float16: "half", torch.bfloat16: "bfloat", torch.float32: "float"}
_GROUPS = (4, 16, 64, 256)
# Paired BF16 M4 Max measurements: 256/16,384 elements benefited; 1M/8M
# elements regressed versus MPS matmul. Keep larger workloads on that path.
# This conservative bound is not a claim about end-to-end inference speed.
_MAX_ELEMENTS = 16384


def _shader(scalar: str) -> str:
    return r"""
#include <metal_stdlib>
using namespace metal;
kernel void regular_rotation(
    device const SCALAR* x [[buffer(0)]],
    device SCALAR* y [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& group_size [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float values[256];
  values[lane] = index < count ? float(x[index]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]].
  // Applying H4 independently along each base-four digit gives its Kronecker
  // powers, with the exact regular-basis ordering used by Kitchen.
  for (uint stride = 1; stride < group_size; stride *= 4) {
    uint digit = (lane / stride) % 4;
    uint base = lane - digit * stride;
    float a = values[base];
    float b = values[base + stride];
    float c = values[base + 2 * stride];
    float d = values[base + 3 * stride];
    float result;
    if (digit == 0) result = a + b + c - d;
    else if (digit == 1) result = a + b - c + d;
    else if (digit == 2) result = a - b + c + d;
    else result = -a + b + c + d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[lane] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (index < count) y[index] = SCALAR(values[lane] * rsqrt(float(group_size)));
}
""".replace("SCALAR", scalar)


def try_regular_rotation(x: torch.Tensor, group_size: int) -> torch.Tensor | None:
    """Return the regular-basis rotation, or None to request existing fallback.

    Unsupported layouts are not silently copied. FP32 intermediates avoid
    FP16 butterfly overflow; output preserves the input dtype and shape.
    Inference only: raw Metal dispatch does not register an autograd formula.
    Opt in with COMFY_KITCHEN_MPS_ROTATION=1; small-shape timing wins vary.
    """
    if os.environ.get("COMFY_KITCHEN_DISABLE_MPS") == "1" or os.environ.get("COMFY_KITCHEN_MPS_ROTATION", "0").lower() not in ("1", "true", "on"):
        return None
    if (
        x.device.type != "mps"
        or x.dtype not in _TYPES
        or group_size not in _GROUPS
        or x.ndim == 0
        or x.shape[-1] % group_size
        or not x.is_contiguous()
        or x.requires_grad
        or x.numel() == 0
        or x.numel() > _MAX_ELEMENTS
        or not hasattr(torch.mps, "compile_shader")
    ):
        return None
    with _LOCK:
        if x.dtype in _FAILED:
            return None
        library = _LIBRARIES.get(x.dtype)
        if library is None:
            try:
                library = torch.mps.compile_shader(_shader(_TYPES[x.dtype]))
                _LIBRARIES[x.dtype] = library
            except Exception as error:
                _FAILED.add(x.dtype)
                _LOG.warning("MPS regular rotation unavailable for %s; using fallback: %s", x.dtype, error)
                return None
    output = torch.empty_like(x)
    try:
        # Full threadgroups are mandatory: every lane participates in barriers.
        library.regular_rotation(
            x, output, x.numel(), group_size,
            threads=((x.numel() + 255) // 256) * 256,
            group_size=256,
        )
    except Exception as error:
        with _LOCK:
            first_failure = x.dtype not in _FAILED
            _FAILED.add(x.dtype)
        if first_failure:
            _LOG.warning("MPS regular rotation dispatch failed for %s; using fallback: %s", x.dtype, error)
        return None
    return output
