#!/usr/bin/env python3
"""Paired INT4 unpack+scale microbenchmarks at the seven observed SAM shapes.

This compares the installed AppleSilicon-FP8 expression with Kitchen's fused
decoder, not complete model inference. No checkpoint or custom node is loaded.
Run without another GPU workload; --max-seconds is bounded to 60 seconds.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from benchmark_mps import Thermal, command, measure_case
from comfy_kitchen.backends.mps.int4 import unpack_int4_scaled


# (decoded rows N, decoded columns K, calls per observed SAM inference)
SAM_SHAPES = [(3072, 1024, 32), (1024, 1024, 32), (4736, 1024, 32),
              (1024, 4736, 32), (256, 256, 30), (2048, 256, 12), (256, 2048, 12)]


def existing_unpack_scale(packed, scales, dtype):
    lo = (packed << 4) >> 4
    hi = packed >> 4
    decoded = torch.stack((lo, hi), dim=-1).to(dtype)
    decoded = decoded.reshape(packed.shape[0], packed.shape[1] * 2)
    return decoded * scales.to(device=packed.device, dtype=dtype).reshape(-1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 30 or not 0 <= args.warmups <= 10 or not 0 < args.max_seconds <= 60:
        parser.error("Use 1–30 pairs, 0–10 warmups and at most 60 seconds")
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable; refusing CPU timings")
    if os.environ.get("COMFY_KITCHEN_DISABLE_MPS") == "1":
        parser.error("Native MPS disabled by COMFY_KITCHEN_DISABLE_MPS")
    thermal = Thermal()
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(20260907)
    report = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "platform": platform.platform(), "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
              "torch": torch.__version__, "source_commit": command("git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"),
              "dtype": args.dtype, "seed": 20260907, "pairs": args.pairs, "warmups": args.warmups,
              "max_seconds": args.max_seconds, "thermal_before": thermal.snapshot(), "cases": [],
              "caveats": ["Microbenchmarks of unpack+scale only, not end-to-end model timings.",
                          "All samples retained; the whole pair is rejected at serious/critical thermal pressure.",
                          "Thermal pressure is not temperature; nominal does not establish identical clocks.",
                          "Drift >20% between first/last half medians is flagged, not corrected.",
                          "Time budget checked between shapes/pairs; initial compilation can exceed it."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_seconds
    with torch.inference_mode():
        for rows, columns, model_calls in SAM_SHAPES:
            if time.monotonic() >= deadline or (thermal.snapshot()["state"] or 0) >= 2:
                break
            packed = torch.randint(-128, 128, (rows, columns // 2), generator=generator, dtype=torch.int8).to("mps")
            scales = (torch.rand(rows, generator=generator) * .07123 - .02345).to("mps")
            functions = {"existing_unpack_scale": lambda: existing_unpack_scale(packed, scales, dtype),
                         "fused_unpack_scale": lambda: unpack_int4_scaled(packed, scales, dtype)}
            torch.testing.assert_close(functions["existing_unpack_scale"]().cpu(), functions["fused_unpack_scale"]().cpu(), rtol=0, atol=0)
            case = measure_case("int4_unpack_scale", functions, thermal, args, deadline,
                                decoded_shape=[rows, columns], packed_shape=list(packed.shape),
                                observed_sam_calls=model_calls, correctness="exact")
            old = case["summary"]["existing_unpack_scale"]
            new = case["summary"]["fused_unpack_scale"]
            if old is not None and new is not None:
                case["median_speedup"] = old["median_ms"] / new["median_ms"]
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps({"shape": [rows, columns], "median_speedup": case.get("median_speedup")}), flush=True)
            del packed, scales
    report["thermal_after"] = thermal.snapshot()
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
