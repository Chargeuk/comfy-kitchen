# SPDX-License-Identifier: Apache-2.0
"""MPS FP8 decoding without FP8 arithmetic or full-size integer indices.

Only byte-domain operations touch the input. In particular, noncontiguous
inputs are copied as uint8 on their own device, never through the CPU.
"""

import os
import threading

import torch


class MPSFP8UnsupportedError(RuntimeError):
    """The caller should select its existing compatibility fallback."""


_FORMATS = {torch.float8_e4m3fn: 0, torch.float8_e5m2: 1}
_OUTPUTS = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_lock = threading.Lock()
_library = None
_compile_error = None
_launch_error = None


def _is_out_of_memory(error):
    # MPS versions may raise plain RuntimeError instead of OutOfMemoryError.
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()

_DECODE = r"""
#include <metal_stdlib>
using namespace metal;

inline float decode_fp8_byte(uchar byte, uint format) {
    uint magnitude = uint(byte) & 127u;
    float value;
    if (format == 0u) {
        uint exponent = magnitude >> 3;
        uint mantissa = magnitude & 7u;
        if (magnitude == 127u) value = as_type<float>(0x7fc00000u);
        else if (exponent == 0u) value = ldexp(float(mantissa), -9);
        else value = ldexp(float(8u + mantissa), int(exponent) - 10);
    } else {
        uint exponent = magnitude >> 2;
        uint mantissa = magnitude & 3u;
        if (exponent == 31u)
            value = as_type<float>(mantissa == 0u ? 0x7f800000u : 0x7fc00000u);
        else if (exponent == 0u) value = ldexp(float(mantissa), -16);
        else value = ldexp(float(4u + mantissa), int(exponent) - 17);
    }
    return as_type<float>(as_type<uint>(value) | ((uint(byte) & 128u) << 24));
}
"""


def _shader_source():
    source = _DECODE
    for typename in _OUTPUTS.values():
        source += f"""
kernel void decode_{typename}(
    device const uchar* input [[buffer(0)]],
    device const {typename}* scale [[buffer(1)]],
    device {typename}* output [[buffer(2)]],
    constant uint& format [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {{
    // Match eager: cast both operands to output dtype BEFORE multiplying.
    {typename} decoded = {typename}(decode_fp8_byte(input[i], format));
    output[i] = {typename}(float(decoded) * float(scale[0]));
}}
"""
    return source


def _get_library():
    global _library, _compile_error
    if _library is not None:
        return _library
    if _compile_error is not None:
        raise MPSFP8UnsupportedError("FP8 Metal shader compilation failed") from _compile_error
    with _lock:
        if _library is None:
            try:
                _library = torch.mps.compile_shader(_shader_source())
            except Exception as error:
                if _is_out_of_memory(error):
                    raise
                _compile_error = error
                raise MPSFP8UnsupportedError("FP8 Metal shader compilation failed") from error
    return _library


def dequantize_per_tensor_fp8(x, scale, output_type=torch.bfloat16):
    """Decode FP8 and apply a one-element scale in one Metal dispatch.

    This inference-only operation deliberately rejects unsupported contracts;
    callers can retain their eager/compatibility implementation as fallback.
    No scalar value is read back from the GPU, and no FP8 cast is dispatched.
    """
    global _launch_error
    if os.environ.get("COMFY_KITCHEN_DISABLE_MPS", "0") == "1":
        raise MPSFP8UnsupportedError("native MPS backend disabled by environment")
    if x.device.type != "mps" or not hasattr(torch.mps, "compile_shader"):
        raise MPSFP8UnsupportedError("FP8 shader requires MPS and compile_shader")
    if x.dtype not in _FORMATS or output_type not in _OUTPUTS:
        raise MPSFP8UnsupportedError("unsupported FP8 input or floating output dtype")
    if scale.numel() != 1 or scale.dtype not in _OUTPUTS:
        raise MPSFP8UnsupportedError("FP8 shader requires one floating-point scale")
    if x.layout != torch.strided or scale.layout != torch.strided:
        raise MPSFP8UnsupportedError("FP8 shader requires strided tensors")
    if torch.is_grad_enabled() and (x.requires_grad or scale.requires_grad):
        raise MPSFP8UnsupportedError("FP8 shader is inference-only")
    if x.numel() > 2**32 - 1:
        raise MPSFP8UnsupportedError("FP8 tensor exceeds shader index range")
    if _launch_error is not None:
        raise MPSFP8UnsupportedError("FP8 Metal shader disabled after launch failure") from _launch_error
    shape = torch.broadcast_shapes(x.shape, scale.shape)
    output = torch.empty(shape, dtype=output_type, device=x.device)
    if x.numel() == 0:
        return output
    # Equal-size dtype view preserves arbitrary strides and storage offsets.
    raw = x.view(torch.uint8).contiguous()
    rounded_scale = scale.to(device=x.device, dtype=output_type).contiguous()
    kernel = getattr(_get_library(), "decode_" + _OUTPUTS[output_type])
    try:
        kernel(raw, rounded_scale, output, _FORMATS[x.dtype], threads=x.numel(), arg_casts={3: "int32"})
    except RuntimeError as error:
        # A LUT fallback allocates MORE memory: never select it after an OOM.
        if _is_out_of_memory(error):
            raise
        with _lock:
            _launch_error = error
        raise MPSFP8UnsupportedError("FP8 Metal shader launch failed") from error
    return output


def decode_fp8(x, output_type=torch.float32):
    """Unscaled decode entry point for AppleSilicon-FP8 interoperability."""
    scale = torch.ones((), dtype=output_type, device=x.device)
    return dequantize_per_tensor_fp8(x, scale, output_type)
