"""Apple MPS inference operations, with byte-safe FP8 fallback."""

import logging
import os
import sys
from dataclasses import replace

import torch

from comfy_kitchen.backends import eager
from comfy_kitchen.registry import registry

from . import fp8

logger = logging.getLogger(__name__)
int8_linear = eager.int8_linear
_warned_fp8 = False


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


if not torch.backends.mps.is_available():
    registry.mark_unavailable("mps", "MPS device unavailable")
elif os.environ.get("COMFY_KITCHEN_DISABLE_MPS") == "1":
    registry.mark_unavailable("mps", "disabled by COMFY_KITCHEN_DISABLE_MPS=1")
else:
    capabilities = {
        name: replace(registry.get_constraints("eager", name), default_devices=frozenset({"mps"}))
        for name in ("int8_linear", "dequantize_per_tensor_fp8")
    }
    registry.register("mps", sys.modules[__name__], capabilities)
