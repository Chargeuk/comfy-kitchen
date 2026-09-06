"""Apple MPS inference operations, with byte-safe quantization fallbacks."""

import logging
import os
import sys
from dataclasses import replace

import torch

from comfy_kitchen.backends import eager
# Capture the original implementation before compatibility nodes monkeypatch
# the eager module; fallback must never redispatch back into this backend.
from comfy_kitchen.backends.eager.quantization import dequantize_nvfp4 as _cpu_dequantize_nvfp4
from comfy_kitchen.registry import registry

from . import fp8, nvfp4

logger = logging.getLogger(__name__)
int8_linear = eager.int8_linear
_warned_fp8 = False
_warned_nvfp4 = False


def dequantize_per_tensor_fp8(x, scale, output_type=torch.bfloat16):
    global _warned_fp8
    if torch.is_grad_enabled() and x.requires_grad:
        raise NotImplementedError("MPS FP8 decoding does not support input gradients")
    try:
        return fp8.dequantize_per_tensor_fp8(x, scale, output_type)
    except fp8.MPSFP8UnsupportedError as error:
        if not _warned_fp8:
            logger.warning("MPS FP8 shader unavailable; using byte lookup: %s", error)
            _warned_fp8 = True
        # Only the 256-element table is decoded on CPU, never the weights.
        table = torch.arange(256, dtype=torch.uint8).view(x.dtype).to(output_type).to(x.device)
        raw = x.view(torch.uint8).contiguous().long()
        return table[raw] * scale.to(device=x.device, dtype=output_type)


def dequantize_nvfp4(qx, per_tensor_scale, block_scales, output_type=torch.bfloat16, hi_first=True):
    global _warned_nvfp4
    if torch.is_grad_enabled() and any(t.requires_grad for t in (per_tensor_scale, block_scales)):
        raise NotImplementedError("MPS NVFP4 decoding does not support scale gradients")
    try:
        return nvfp4.dequantize_nvfp4(qx, per_tensor_scale, block_scales, output_type, hi_first)
    except nvfp4.MPSNVFP4UnsupportedError as error:
        if not _warned_nvfp4:
            logger.warning("MPS NVFP4 shader unavailable; using CPU compatibility decode: %s", error)
            _warned_nvfp4 = True
        cpu_scales = block_scales.view(torch.uint8).to("cpu").view(block_scales.dtype)
        result = _cpu_dequantize_nvfp4(
            qx.to("cpu"), per_tensor_scale.to("cpu"), cpu_scales, output_type, hi_first
        )
        return result.to(qx.device)


if not torch.backends.mps.is_available():
    registry.mark_unavailable("mps", "MPS device unavailable")
elif os.environ.get("COMFY_KITCHEN_DISABLE_MPS") == "1":
    registry.mark_unavailable("mps", "disabled by COMFY_KITCHEN_DISABLE_MPS=1")
else:
    capabilities = {
        name: replace(registry.get_constraints("eager", name), default_devices=frozenset({"mps"}))
        for name in ("int8_linear", "dequantize_per_tensor_fp8", "dequantize_nvfp4")
    }
    # Small scales may remain on CPU while compressed weights live on MPS.
    nvfp4_constraints = capabilities["dequantize_nvfp4"]
    nvfp4_params = dict(nvfp4_constraints.params)
    for name in ("per_tensor_scale", "block_scales"):
        nvfp4_params[name] = replace(nvfp4_params[name], devices=frozenset({"cpu", "mps"}))
    capabilities["dequantize_nvfp4"] = replace(nvfp4_constraints, params=nvfp4_params)
    registry.register("mps", sys.modules[__name__], capabilities)
