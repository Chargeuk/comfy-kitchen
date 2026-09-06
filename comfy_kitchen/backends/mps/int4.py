# SPDX-License-Identifier: Apache-2.0
"""Signed INT4 unpack and row-scale application for rotated-basis W4A16.

This helper does not rotate weights, quantize activations, or cache decoded
weights. It preserves AppleSilicon-FP8's scale-cast then product-cast rounding.
"""

import os
import threading

import torch


class MPSINT4UnsupportedError(RuntimeError):
    """The caller should retain its existing INT4 unpack fallback."""


_OUTPUTS = {torch.float16: "half", torch.bfloat16: "bfloat", torch.float32: "float"}
_lock = threading.Lock()
_library = None
_compile_error = None
_launch_error = None


def _is_out_of_memory(error):
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _shader_source():
    source = "#include <metal_stdlib>\nusing namespace metal;\n"
    for typename in _OUTPUTS.values():
        source += f"""
kernel void unpack_int4_{typename}(
    device const uchar* packed [[buffer(0)]],
    device const {typename}* scales [[buffer(1)]],
    device {typename}* output [[buffer(2)]],
    constant uint& packed_columns [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {{
    uint byte = uint(packed[i]);
    // Kitchen packs even columns in the low nibble, signed range [-8, 7].
    int lo = int(byte & 15u);
    int hi = int(byte >> 4);
    lo -= (lo & 8) << 1;
    hi -= (hi & 8) << 1;
    float scale = float(scales[i / packed_columns]);
    output[2u * i] = {typename}(float(lo) * scale);
    output[2u * i + 1u] = {typename}(float(hi) * scale);
}}
"""
    return source


def _get_library():
    global _library, _compile_error
    with _lock:
        if _library is not None:
            return _library
        if _compile_error is not None:
            raise MPSINT4UnsupportedError("INT4 Metal shader compilation failed") from _compile_error
        try:
            _library = torch.mps.compile_shader(_shader_source())
        except Exception as error:
            if _is_out_of_memory(error):
                raise
            _compile_error = error
            raise MPSINT4UnsupportedError("INT4 Metal shader compilation failed") from error
    return _library


def unpack_int4_scaled(qweight, wscales, output_type=torch.bfloat16):
    """Decode [N, K/2] signed nibbles and apply N row scales on MPS.

    The returned [N, K] weight remains in its stored (possibly rotated) basis.
    Accepts int8/uint8 byte storage, including offsets and strided views. Scale
    values are never read back to the CPU and mutations are observed each call.
    Unsupported shaders request fallback; an OOM is propagated unchanged.
    """
    global _launch_error
    if os.environ.get("COMFY_KITCHEN_DISABLE_MPS", "0") == "1":
        raise MPSINT4UnsupportedError("native MPS backend disabled by environment")
    if qweight.device.type != "mps" or not hasattr(torch.mps, "compile_shader"):
        raise MPSINT4UnsupportedError("INT4 shader requires MPS and compile_shader")
    if qweight.ndim != 2 or qweight.dtype not in (torch.int8, torch.uint8):
        raise MPSINT4UnsupportedError("INT4 shader requires 2D packed int8/uint8 weights")
    if output_type not in _OUTPUTS:
        raise MPSINT4UnsupportedError("unsupported INT4 floating output dtype")
    if wscales.dtype not in _OUTPUTS or wscales.numel() != qweight.shape[0]:
        raise MPSINT4UnsupportedError("INT4 shader requires one floating-point scale per row")
    if qweight.layout != torch.strided or wscales.layout != torch.strided:
        raise MPSINT4UnsupportedError("INT4 shader requires strided tensors")
    if torch.is_grad_enabled() and wscales.requires_grad:
        raise MPSINT4UnsupportedError("INT4 shader is inference-only")
    if qweight.numel() > (2**32 - 1) // 2:
        raise MPSINT4UnsupportedError("INT4 output exceeds shader index range")
    if _launch_error is not None:
        raise MPSINT4UnsupportedError("INT4 Metal shader disabled after launch failure") from _launch_error
    output = torch.empty((qweight.shape[0], qweight.shape[1] * 2), dtype=output_type, device=qweight.device)
    if qweight.numel() == 0:
        return output
    raw = qweight.view(torch.uint8).contiguous()
    # Match the existing W4A16 expression: unpack.to(dtype) * scales.to(dtype).
    scales = wscales.to(device=qweight.device, dtype=output_type).contiguous()
    kernel = getattr(_get_library(), "unpack_int4_" + _OUTPUTS[output_type])
    try:
        kernel(raw, scales, output, qweight.shape[1], threads=raw.numel(), arg_casts={3: "int32"})
    except RuntimeError as error:
        if _is_out_of_memory(error):
            raise
        with _lock:
            _launch_error = error
        raise MPSINT4UnsupportedError("INT4 Metal shader launch failed") from error
    return output
