# SPDX-License-Identifier: Apache-2.0
"""Inference-only NVFP4 decoding on Metal, without CPU weight expansion.

The compressed weights and E4M3 block scales are accessed as bytes. Each
thread decodes 16 weights and shares one directly addressed cuBLAS-swizzled
scale, avoiding repeated scale decoding, unpacked integer tensors and a
separate scale-unswizzle operation.
"""

import os
import threading

import torch

from .fp8 import _DECODE, _OUTPUTS, _is_out_of_memory


class MPSNVFP4UnsupportedError(RuntimeError):
    """The caller should select its existing compatibility fallback."""


_lock = threading.Lock()
_library = None
_compile_error = None
_launch_error = None


def _shader_source():
    source = _DECODE + r"""
inline float decode_e2m1_nibble(uint nibble) {
    uint magnitude = nibble & 7u;
    float value = magnitude < 2u ? float(magnitude) * 0.5f
        : ldexp(float(2u + (magnitude & 1u)), int(magnitude >> 1) - 2);
    return as_type<float>(as_type<uint>(value) | ((nibble & 8u) << 28));
}
"""
    for typename in _OUTPUTS.values():
        source += f"""
kernel void decode_nvfp4_{typename}(
    device const uchar* input [[buffer(0)]],
    device const {typename}* tensor_scale [[buffer(1)]],
    device const uchar* block_scales [[buffer(2)]],
    device {typename}* output [[buffer(3)]],
    constant uint& packed_cols [[buffer(4)]],
    constant uint& scale_col_tiles [[buffer(5)]],
    constant uint& hi_first [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {{
    uint row = i / (packed_cols / 8u);
    uint block_col = i % (packed_cols / 8u);
    // Inverse address of to_blocked: tiles of 128 rows by 4 scales,
    // whose flattened inner order is [row % 32, row / 32, col % 4].
    uint tile = (row / 128u) * scale_col_tiles + block_col / 4u;
    uint scale_index = tile * 512u + (row % 32u) * 16u
        + ((row % 128u) / 32u) * 4u + block_col % 4u;
    {typename} block_scale = {typename}(decode_fp8_byte(block_scales[scale_index], 0u));
    // Eager rounds both scale operands, then their product, BEFORE the
    // final multiplication. Do not collapse these into one FP32 product.
    {typename} total_scale = {typename}(float(tensor_scale[0]) * float(block_scale));
    #pragma unroll
    for (uint pair = 0u; pair < 8u; ++pair) {{
        uint packed_index = i * 8u + pair;
        uint byte = uint(input[packed_index]);
        uint first = hi_first != 0u ? byte >> 4 : byte & 15u;
        uint second = hi_first != 0u ? byte & 15u : byte >> 4;
        output[2u * packed_index] = {typename}(decode_e2m1_nibble(first) * float(total_scale));
        output[2u * packed_index + 1u] = {typename}(decode_e2m1_nibble(second) * float(total_scale));
    }}
}}
"""
    return source


def _get_library():
    global _library, _compile_error
    if _library is not None:
        return _library
    if _compile_error is not None:
        raise MPSNVFP4UnsupportedError("NVFP4 Metal shader compilation failed") from _compile_error
    with _lock:
        if _library is None:
            try:
                _library = torch.mps.compile_shader(_shader_source())
            except Exception as error:
                if _is_out_of_memory(error):
                    raise
                _compile_error = error
                raise MPSNVFP4UnsupportedError("NVFP4 Metal shader compilation failed") from error
    return _library


def dequantize_nvfp4(
    qx, per_tensor_scale, block_scales, output_type=torch.bfloat16, hi_first=True
):
    """Decode 2D packed NVFP4 weights using tensor and E4M3 block scales.

    Both nibble orders and arbitrarily strided byte/scale views are supported.
    The block-scale storage must have exactly the padded cuBLAS layout size.
    This is weight decoding, not native NVIDIA-style FP4 matrix arithmetic.
    """
    global _launch_error
    if os.environ.get("COMFY_KITCHEN_DISABLE_MPS", "0") == "1":
        raise MPSNVFP4UnsupportedError("native MPS backend disabled by environment")
    if qx.device.type != "mps" or not hasattr(torch.mps, "compile_shader"):
        raise MPSNVFP4UnsupportedError("NVFP4 shader requires MPS and compile_shader")
    if qx.dtype != torch.uint8 or qx.ndim != 2 or qx.shape[1] % 8:
        raise MPSNVFP4UnsupportedError("NVFP4 shader requires 2D uint8 weights with columns divisible by 8")
    if output_type not in _OUTPUTS:
        raise MPSNVFP4UnsupportedError("unsupported floating output dtype")
    if per_tensor_scale.numel() != 1 or per_tensor_scale.dtype not in _OUTPUTS:
        raise MPSNVFP4UnsupportedError("NVFP4 shader requires one floating-point tensor scale")
    if block_scales.dtype != torch.float8_e4m3fn:
        raise MPSNVFP4UnsupportedError("NVFP4 shader requires E4M3FN block scales")
    if any(t.layout != torch.strided for t in (qx, per_tensor_scale, block_scales)):
        raise MPSNVFP4UnsupportedError("NVFP4 shader requires strided tensors")
    if any(t.device.type not in ("cpu", "mps") for t in (per_tensor_scale, block_scales)):
        raise MPSNVFP4UnsupportedError("NVFP4 scales must be on CPU or MPS")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (per_tensor_scale, block_scales)):
        raise MPSNVFP4UnsupportedError("NVFP4 shader is inference-only")
    rows, packed_cols = qx.shape
    scale_col_tiles = (packed_cols // 8 + 3) // 4
    expected_scales = ((rows + 127) // 128) * scale_col_tiles * 512
    if block_scales.numel() != expected_scales:
        raise MPSNVFP4UnsupportedError("NVFP4 block scale storage does not match padded weight shape")
    if qx.numel() * 2 > 2**32 - 1 or block_scales.numel() > 2**32 - 1:
        raise MPSNVFP4UnsupportedError("NVFP4 tensor exceeds shader index range")
    if _launch_error is not None:
        raise MPSNVFP4UnsupportedError("NVFP4 Metal shader disabled after launch failure") from _launch_error
    output = torch.empty((rows, packed_cols * 2), dtype=output_type, device=qx.device)
    if qx.numel() == 0:
        return output
    raw = qx.contiguous()
    # The equal-size byte view comes BEFORE copying, including CPU -> MPS.
    raw_scales = block_scales.view(torch.uint8).to(qx.device).contiguous()
    rounded_scale = per_tensor_scale.to(device=qx.device, dtype=output_type).contiguous()
    kernel = getattr(_get_library(), "decode_nvfp4_" + _OUTPUTS[output_type])
    try:
        kernel(
            raw, rounded_scale, raw_scales, output, packed_cols, scale_col_tiles, int(hi_first),
            threads=qx.numel() // 8, arg_casts={4: "int32", 5: "int32", 6: "int32"},
        )
    except RuntimeError as error:
        # CPU fallback can allocate much more memory; never choose it on OOM.
        if _is_out_of_memory(error):
            raise
        with _lock:
            _launch_error = error
        raise MPSNVFP4UnsupportedError("NVFP4 Metal shader launch failed") from error
    return output
