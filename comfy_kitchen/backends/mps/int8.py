# SPDX-License-Identifier: Apache-2.0
"""Fused FP32 scaling, bias and output cast for the safe MPS INT8 fallback.

This changes only the GEMM epilogue: it does not lower activation, rotation or
accumulator precision. The multiplication rounds to FP32 before adding bias,
matching the existing separate PyTorch operations rather than using an FMA.
"""

import os
import threading

import torch

from .fp8 import _OUTPUTS, _is_out_of_memory


class MPSINT8EpilogueUnsupportedError(RuntimeError):
    """The caller should retain the existing PyTorch epilogue."""


_lock = threading.Lock()
_library = None
_compile_error = None
_launch_error = None


def _shader_source():
    source = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
"""
    for input_name in _OUTPUTS.values():
        for output_name in _OUTPUTS.values():
            source += f"""
kernel void epilogue_{input_name}_{output_name}(
    device const {input_name}* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device {output_name}* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& output_columns [[buffer(5)]],
    constant uint& column_offset [[buffer(6)]],
    constant uint& scalar_scale [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {{
    uint column = index % columns;
    float value = float(input[index]) * scale[scalar_scale != 0u ? 0u : column];
    if (has_bias != 0u) value = value + bias[column];
    output[(index / columns) * output_columns + column_offset + column] = {output_name}(value);
}}
"""
    return source


def _get_library():
    global _library, _compile_error
    if _library is not None:
        return _library
    if _compile_error is not None:
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue Metal compilation failed") from _compile_error
    with _lock:
        if _library is None:
            try:
                _library = torch.mps.compile_shader(_shader_source())
            except Exception as error:
                if _is_out_of_memory(error):
                    raise
                _compile_error = error
                raise MPSINT8EpilogueUnsupportedError("INT8 epilogue Metal compilation failed") from error
    return _library


def int8_epilogue(y, weight_scale, bias=None, output_type=torch.float16, *, out=None, column_offset=0):
    """Compute ``(y.float() * scale + bias.float()).to(output_type)`` on MPS.

    Optional ``out`` is a contiguous full output matrix; ``column_offset``
    places this GEMM chunk without allocating another final tensor. Scale and
    bias refer only to the supplied chunk, not the full output matrix.
    Unsupported contracts raise for a caller-owned PyTorch fallback; OOM is
    deliberately not converted to an unsupported-operation exception.
    """
    global _launch_error
    if os.environ.get("COMFY_KITCHEN_DISABLE_MPS") == "1":
        raise MPSINT8EpilogueUnsupportedError("native MPS backend disabled by environment")
    if y.device.type != "mps" or not hasattr(torch.mps, "compile_shader"):
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue requires MPS and compile_shader")
    if y.dtype not in _OUTPUTS or output_type not in _OUTPUTS or y.ndim != 2:
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue requires a 2D floating GEMM output")
    if not y.is_contiguous() or y.layout != torch.strided:
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue requires contiguous strided GEMM output")
    rows, columns = y.shape
    if weight_scale.dtype != torch.float32 or weight_scale.numel() not in (1, columns):
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue requires FP32 scalar or per-channel scales")
    if bias is not None and (bias.dtype not in _OUTPUTS or bias.numel() != columns):
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue bias must match the chunk's output channels")
    tensors = [y, weight_scale] + ([] if bias is None else [bias]) + ([] if out is None else [out])
    if any(t.layout != torch.strided for t in tensors):
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue requires strided tensors")
    if any(t.device.type not in ("cpu", "mps") for t in tensors):
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue scales and bias must be on CPU or MPS")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue is inference-only")
    if not isinstance(column_offset, int) or column_offset < 0:
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue requires a nonnegative column offset")
    if out is None:
        if column_offset:
            raise MPSINT8EpilogueUnsupportedError("INT8 epilogue column offset requires an output matrix")
        out = torch.empty(y.shape, dtype=output_type, device=y.device)
    elif (out.device != y.device or out.dtype != output_type or out.ndim != 2 or not out.is_contiguous()
          or out.shape[0] != rows or column_offset + columns > out.shape[1]):
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue output matrix cannot hold this chunk")
    elif any(torch._C._overlaps(out, tensor) for tensor in (y, weight_scale, bias) if tensor is not None):
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue output must not overlap its inputs")
    if y.numel() > 2**32 - 1 or out.numel() > 2**32 - 1:
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue exceeds shader index range")
    if _launch_error is not None:
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue disabled after Metal launch failure") from _launch_error
    if y.numel() == 0:
        return out
    scale = weight_scale.to(device=y.device).reshape(-1).contiguous()
    rounded_bias = scale if bias is None else bias.to(device=y.device, dtype=torch.float32).reshape(-1).contiguous()
    kernel = getattr(_get_library(), f"epilogue_{_OUTPUTS[y.dtype]}_{_OUTPUTS[output_type]}")
    try:
        kernel(y, scale, rounded_bias, out, columns, out.shape[1], column_offset,
               int(scale.numel() == 1), int(bias is not None), threads=y.numel(),
               arg_casts={i: "int32" for i in range(4, 9)})
    except RuntimeError as error:
        if _is_out_of_memory(error):
            raise
        with _lock:
            _launch_error = error
        raise MPSINT8EpilogueUnsupportedError("INT8 epilogue Metal launch failed") from error
    return out
