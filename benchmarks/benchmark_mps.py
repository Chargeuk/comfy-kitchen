#!/usr/bin/env python3
"""Short, paired MPS microbenchmarks; no models, downloads or GPU contention.

Run from an otherwise idle Mac, e.g.:
    python benchmarks/benchmark_mps.py --output /tmp/kitchen-mps.json

Alternates AB/BA order, retains all samples, and rejects samples observed at
serious/critical thermal pressure. Thermal pressure is NOT chip temperature:
nominal pressure cannot establish that clocks are identical between samples.
These results measure operations, not complete ComfyUI generation speed.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from comfy_kitchen.backends.mps.fp8 import dequantize_per_tensor_fp8
from comfy_kitchen.backends.mps.rotation import try_regular_rotation
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=3).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


class Thermal:
    def __init__(self):
        self.read = None
        try:
            self.foundation = ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
            self.objc = ctypes.CDLL(ctypes.util.find_library("objc"))
            self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
            self.objc.objc_getClass.restype = ctypes.c_void_p
            self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
            self.objc.sel_registerName.restype = ctypes.c_void_p
            send = ctypes.cast(self.objc.objc_msgSend, ctypes.c_void_p).value
            send_pointer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
            send_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(send)
            process = send_pointer(self.objc.objc_getClass(b"NSProcessInfo"), self.objc.sel_registerName(b"processInfo"))
            selector = self.objc.sel_registerName(b"thermalState")
            if process:
                self.read = lambda: int(send_integer(process, selector))
        except (OSError, AttributeError, TypeError):
            pass

    def snapshot(self):
        if self.read is not None:
            state = self.read()
            return {"source": "NSProcessInfo.thermalState", "state": state,
                    "label": {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}.get(state, "unknown")}
        return {"source": "pmset", "state": None, "detail": command("pmset", "-g", "therm"),
                "caveat": "Thermal pressure unavailable; samples cannot be screened reliably."}


def summary(samples):
    if not samples:
        return None
    median = statistics.median(samples)
    half = max(1, len(samples) // 2)
    first, last = statistics.median(samples[:half]), statistics.median(samples[-half:])
    drift = last / first - 1 if first else None
    return {"count": len(samples), "median_ms": median, "min_ms": min(samples), "max_ms": max(samples),
            "mad_ms": statistics.median(abs(x - median) for x in samples),
            "first_half_median_ms": first, "last_half_median_ms": last,
            "drift_fraction": drift, "drift_flag": len(samples) >= 4 and abs(drift or 0) > 0.20}


def measure_case(name, functions, thermal, args, deadline, **details):
    labels = list(functions)
    result = {"name": name, **details, "thermal_before": thermal.snapshot(), "samples": []}
    if (result["thermal_before"]["state"] or 0) >= 2:
        result["stopped_for_thermal_pressure"] = True
        result["summary"] = {label: None for label in labels}
        return result
    for _ in range(args.warmups):
        for fn in functions.values():
            fn()
    torch.mps.synchronize()
    for pair in range(args.pairs):
        if time.monotonic() >= deadline:
            result["stopped_at_time_budget"] = True
            break
        for label in labels if pair % 2 == 0 else labels[::-1]:
            before = thermal.snapshot()
            torch.mps.synchronize()
            start = time.perf_counter_ns()
            output = functions[label]()
            torch.mps.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            after = thermal.snapshot()
            rejected = any(s["state"] is not None and s["state"] >= 2 for s in (before, after))
            result["samples"].append({"pair": pair, "path": label, "ms": elapsed,
                                      "thermal_before": before, "thermal_after": after,
                                      "rejected_thermal": rejected})
            del output
        if any(s["rejected_thermal"] for s in result["samples"][-2:]):
            result["stopped_for_thermal_pressure"] = True
            break
    result["thermal_after"] = thermal.snapshot()
    # Reject the WHOLE pair if either side experienced pressure.
    bad_pairs = {s["pair"] for s in result["samples"] if s["rejected_thermal"]}
    for sample in result["samples"]:
        sample["accepted_pair"] = sample["pair"] not in bad_pairs
    result["summary"] = {label: summary([s["ms"] for s in result["samples"]
                                         if s["path"] == label and s["accepted_pair"]]) for label in labels}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    if not 1 <= args.pairs <= 30 or not 0 <= args.warmups <= 10 or not 0 < args.max_seconds <= 60:
        parser.error("Use 1–30 pairs, 0–10 warmups and a time budget of at most 60 seconds")
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable (a sandbox may require GPU access permission)")
    if os.environ.get("COMFY_KITCHEN_DISABLE_MPS") == "1":
        parser.error("Native MPS is explicitly disabled; unset COMFY_KITCHEN_DISABLE_MPS for this comparison")
    thermal = Thermal()
    dtype = getattr(torch, args.dtype)
    try:
        kitchen_version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        kitchen_version = "source-only"
    result = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "metadata": {"platform": platform.platform(), "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
                           "torch": torch.__version__, "kitchen_installed_version": kitchen_version,
                           "source_commit": command("git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"),
                           "device": "mps", "dtype": args.dtype, "seed": 12345, "pairs": args.pairs,
                           "warmups": args.warmups, "max_seconds": args.max_seconds,
                           "rotation_environment": os.environ.get("COMFY_KITCHEN_MPS_ROTATION", "default")},
              "thermal_before": thermal.snapshot(), "cases": [],
              "caveats": ["Microbenchmarks, not end-to-end inference.",
                          "Thermal state is pressure, not temperature or clock telemetry.",
                          "Drift >20% between first/last half medians is flagged, not automatically corrected.",
                          "Time budget checked between bounded test cases/pairs; shader compilation may exceed it."]}
    deadline = time.monotonic() + args.max_seconds
    generator = torch.Generator(device="cpu").manual_seed(12345)
    with torch.inference_mode():
        for size in (1024, 4096):
            if time.monotonic() >= deadline or (thermal.snapshot()["state"] or 0) >= 2:
                break
            raw = torch.randint(0, 126, (size, size), generator=generator, dtype=torch.uint8).to("mps")
            x = raw.view(torch.float8_e4m3fn)
            scale = torch.tensor(0.1234567, device="mps")
            lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(dtype).to("mps")
            functions = {"lut": lambda: lut[x.view(torch.uint8).long()] * scale.to(dtype),
                         "native": lambda: dequantize_per_tensor_fp8(x, scale, dtype)}
            torch.testing.assert_close(functions["lut"]().cpu(), functions["native"]().cpu(), rtol=0, atol=0)
            result["cases"].append(measure_case("fp8_decode_scale", functions, thermal, args, deadline,
                                                shape=list(x.shape), correctness="exact", format="E4M3FN"))
        for elements in (256, 16384, 1048576):
            if time.monotonic() >= deadline or (thermal.snapshot()["state"] or 0) >= 2:
                break
            x = torch.randn((elements // 256, 256), generator=generator).to(device="mps", dtype=dtype)
            h = _build_hadamard(256, "mps", dtype)
            def dense():
                return _rotate_activation(x, h, 256)
            def selected():
                value = try_regular_rotation(x, 256)
                return dense() if value is None else value
            eligible = try_regular_rotation(x, 256) is not None
            # Never bypass the implementation's measured large-shape cutoff.
            torch.testing.assert_close(dense().float().cpu(), selected().float().cpu(), rtol=0.02, atol=0.04)
            result["cases"].append(measure_case("regular_rotation", {"dense": dense, "selected": selected},
                                                thermal, args, deadline, shape=list(x.shape), group_size=256,
                                                selected_path="metal" if eligible else "dense_fallback",
                                                correctness="rtol=0.02,atol=0.04"))
    result["thermal_after"] = thermal.snapshot()
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
