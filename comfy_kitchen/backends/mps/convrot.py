# SPDX-License-Identifier: Apache-2.0
"""Regular ConvRot rotation using SIMD shuffles and FP32 registers.

One SIMD group handles 64 or 256 elements. Its two/eight registers per lane
replace the older threadgroup-memory butterfly and its barriers. This helper
is only for Kitchen's regular Hadamard basis, never an arbitrary rotation.
"""

import logging
import os
import threading

import torch

from .fp8 import _OUTPUTS, _is_out_of_memory

_logger = logging.getLogger(__name__)
_lock = threading.Lock()
_library = None
_compile_error = None
_launch_error = None


def _shader_source():
    source = r"""
#include <metal_stdlib>
using namespace metal;
inline float h4_value(float a, float b, float c, float d, uint digit) {
  if (digit == 0u) return a + b + c - d;
  if (digit == 1u) return a + b - c + d;
  if (digit == 2u) return a - b + c + d;
  return -a + b + c + d;
}
"""
    for group in (64, 256):
        registers = group // 32
        for input_type in _OUTPUTS.values():
            for output_type in dict.fromkeys(("float", input_type)):
                source += f"""
kernel void rotation_{input_type}_{output_type}_{group}(
  device const {input_type}* x [[buffer(0)]],
  device {output_type}* y [[buffer(1)]],
  uint index [[thread_position_in_grid]],
  uint lane [[thread_index_in_simdgroup]]) {{
  uint base = (index / 32u) * {group}u;
  float values[{registers}];
  #pragma unroll
  // Apply the exact power-of-two normalization before the butterfly to
  // avoid the unnecessary sqrt(group) expansion of large FP32/BF16 values.
  for (uint j = 0u; j < {registers}u; ++j)
    values[j] = float(x[base + lane + j * 32u]) * {1 / group**0.5}f;
  // Base-four digits wholly inside a SIMD group's lane index.
  #pragma unroll
  for (uint stride = 1u; stride < 16u; stride *= 4u) {{
    uint digit = (lane / stride) % 4u;
    uint first_lane = lane - digit * stride;
    #pragma unroll
    for (uint j = 0u; j < {registers}u; ++j) {{
      float a = simd_shuffle(values[j], ushort(first_lane));
      float b = simd_shuffle(values[j], ushort(first_lane + stride));
      float c = simd_shuffle(values[j], ushort(first_lane + 2u * stride));
      float d = simd_shuffle(values[j], ushort(first_lane + 3u * stride));
      values[j] = h4_value(a, b, c, d, digit);
    }}
  }}
  // Stride16 spans two registers and the low/high16 lanes of each.
  #pragma unroll
  for (uint j = 0u; j < {registers}u; j += 2u) {{
    uint low_lane = lane % 16u;
    float a = simd_shuffle(values[j], ushort(low_lane));
    float b = simd_shuffle(values[j], ushort(low_lane + 16u));
    float c = simd_shuffle(values[j + 1u], ushort(low_lane));
    float d = simd_shuffle(values[j + 1u], ushort(low_lane + 16u));
    values[j] = h4_value(a, b, c, d, lane / 16u);
    values[j + 1u] = h4_value(a, b, c, d, lane / 16u + 2u);
  }}
"""
                if group == 256:
                    source += r"""
  // Stride64 is entirely register-local (even and odd register sets).
  #pragma unroll
  for (uint j = 0u; j < 2u; ++j) {
    float a = values[j];
    float b = values[j + 2u];
    float c = values[j + 4u];
    float d = values[j + 6u];
    values[j] = h4_value(a, b, c, d, 0u);
    values[j + 2u] = h4_value(a, b, c, d, 1u);
    values[j + 4u] = h4_value(a, b, c, d, 2u);
    values[j + 6u] = h4_value(a, b, c, d, 3u);
  }
"""
                source += f"""
  #pragma unroll
  for (uint j = 0u; j < {registers}u; ++j)
    y[base + lane + j * 32u] = {output_type}(values[j]);
}}
"""
    return source


def _get_library():
    global _library, _compile_error
    if _library is not None:
        return _library
    if _compile_error is not None:
        return None
    with _lock:
        if _library is None and _compile_error is None:
            try:
                _library = torch.mps.compile_shader(_shader_source())
            except Exception as error:
                if _is_out_of_memory(error):
                    raise
                _compile_error = error
                _logger.warning("MPS SIMD ConvRot compilation unavailable; using fallback: %s", error)
    return _library


def try_rotate(x, group_size, output_dtype=None):
    """Return a regular-basis rotation, or None to select the existing path.

    Output defaults to input dtype. INT8 callers with FP16 activations should
    explicitly request FP32 output, preserving values above 65504 until scale
    application after GEMM. All butterfly intermediate arithmetic is FP32.
    Inputs must be contiguous, inference-only, and contain complete groups.
    OOM propagates; it must not trigger a potentially larger fallback.
    """
    global _launch_error
    # The butterfly changes summation order. Small layer-level differences
    # compound in SAM detections, so this remains an explicit experiment.
    if (os.environ.get("COMFY_KITCHEN_DISABLE_MPS") == "1"
            or os.environ.get("COMFY_KITCHEN_MPS_ROTATION", "0").lower() not in ("1", "true", "on")):
        return None
    output_dtype = x.dtype if output_dtype is None else output_dtype
    if (x.device.type != "mps" or not hasattr(torch.mps, "compile_shader")
            or x.dtype not in _OUTPUTS or output_dtype not in (x.dtype, torch.float32)
            or group_size not in (64, 256) or x.ndim == 0 or x.shape[-1] % group_size
            or x.layout != torch.strided or not x.is_contiguous()
            or (torch.is_grad_enabled() and x.requires_grad)
            or x.numel() == 0 or x.numel() > 2**32 - 1 or _launch_error is not None):
        return None
    library = _get_library()
    if library is None:
        return None
    output = torch.empty(x.shape, dtype=output_dtype, device=x.device)
    kernel = getattr(library, f"rotation_{_OUTPUTS[x.dtype]}_{_OUTPUTS[output_dtype]}_{group_size}")
    try:
        # A complete SIMD group always participates, with no padded lanes.
        kernel(x, output, threads=(x.numel() // group_size) * 32, group_size=32)
    except RuntimeError as error:
        if _is_out_of_memory(error):
            raise
        with _lock:
            if _launch_error is None:
                _logger.warning("MPS SIMD ConvRot launch unavailable; using fallback: %s", error)
            _launch_error = error
        return None
    return output
