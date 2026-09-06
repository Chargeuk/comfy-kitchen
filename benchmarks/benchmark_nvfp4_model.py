#!/usr/bin/env python3
"""No-download NVFP4 layer and native MiniMax H3 text-encoder validation.

Use installed Kitchen for a baseline, then --kitchen-source for a candidate.
Layer mode reads only the selected layer and compares CPU-reference decoding
against the selected MPS dispatch, alternating AB/BA. Encode mode requires a
local checkpoint and retains ComfyUI's packed-weight, per-layer cast policy.
Run each baseline/candidate in a fresh process, with no other GPU benchmark.
"""
import argparse
import collections
import datetime
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

from model_validation import Thermal, compare

DEFAULT_COMFY = Path.home() / "Documents/comfyui/ComfyUI"
MODEL_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


def read_header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 32 * 1024**2:
            raise ValueError("Unexpectedly large safetensors header")
        return json.loads(stream.read(size)), 8 + size


def read_tensor(path, header, offset, name):
    entry = header[name]
    start, end = entry["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(offset + start)
        raw = bytearray(stream.read(end - start))
    if len(raw) != end - start:
        raise ValueError(f"Incomplete checkpoint tensor: {name}")
    dtype = {"U8": torch.uint8, "F32": torch.float32, "BF16": torch.bfloat16,
             "F16": torch.float16, "F8_E4M3": torch.float8_e4m3fn}[entry["dtype"]]
    return torch.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


def metrics(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    delta = a - b
    return {"exact": torch.equal(a, b), "finite": bool(torch.isfinite(b).all()),
            "max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
            "relative_l2": (delta.norm() / a.norm().clamp_min(1e-12)).item()}


def snapshot():
    import psutil
    return {"available_ram_bytes": psutil.virtual_memory().available,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_bytes": torch.mps.driver_allocated_memory()}


def write_report(args, report):
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


def track_dispatch(ck):
    counts, shapes = collections.Counter(), collections.Counter()
    for backend_name, module in ck.registry._backends.items():
        if module is None:
            continue
        for name in ("dequantize_nvfp4", "scaled_mm_nvfp4", "quantize_nvfp4",
                     "dequantize_per_tensor_fp8", "dequantize_int8_simple",
                     "dequantize_int8_simple_dtype", "dequantize_int8_embedding"):
            original = getattr(module, name, None)
            if original is None:
                continue
            label = f"{backend_name}.{name}"
            def wrapped(*args, _original=original, _label=label, **kwargs):
                counts[_label] += 1
                first = next((v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)), None)
                if first is not None:
                    shapes[f"{_label}:{tuple(first.shape)}:{first.dtype}:{first.device}"] += 1
                return _original(*args, **kwargs)
            setattr(module, name, wrapped)
    mps = ck.registry._backends.get("mps")
    native = getattr(mps, "nvfp4", None)
    if native is not None:
        original_native = native.dequantize_nvfp4
        def native_decode(*args, **kwargs):
            try:
                output = original_native(*args, **kwargs)
            except Exception:
                counts["metal.nvfp4_failed"] += 1
                raise
            counts["metal.nvfp4_completed"] += 1
            return output
        native.dequantize_nvfp4 = native_decode
    return counts, shapes


def layer_benchmark(args, report, header, offset, ck, reference_decode, thermal):
    prefix = args.layer + "."
    names = [prefix + suffix for suffix in ("weight", "weight_scale", "weight_scale_2")]
    names.append(prefix + "comfy_quant")
    if prefix + "pre_quant_scale" in header:
        names.append(prefix + "pre_quant_scale")
    read_bytes = sum(header[n]["data_offsets"][1] - header[n]["data_offsets"][0] for n in names)
    if read_bytes > args.max_read_mib * 1024**2:
        raise ValueError("Selected layer exceeds --max-read-mib; choose a smaller layer")
    tensors = {name: read_tensor(args.model_path, header, offset, name) for name in names}
    qcpu, bcpu, scpu = [tensors[name] for name in names[:3]]
    quantization = json.loads(tensors[prefix + "comfy_quant"].numpy().tobytes())
    if quantization["format"] != "nvfp4":
        raise ValueError("Selected layer is not NVFP4")
    dtype = getattr(torch, args.dtype)
    # Serialized NVFP4 block scales are already in Kitchen's blocked layout.
    reference = reference_decode(qcpu, scpu, bcpu, dtype)
    q = qcpu.to("mps")
    block = bcpu.view(torch.uint8).to("mps").view(torch.float8_e4m3fn)
    scale = scpu.to("mps")
    def cpu_detour():
        return reference_decode(q.cpu(), scale.cpu(), block.view(torch.uint8).cpu().view(torch.float8_e4m3fn), dtype).to("mps")
    def selected():
        return ck.dequantize_nvfp4(q, scale, block, dtype)
    decoded = selected()
    if decoded.device.type != "mps":
        raise RuntimeError("Selected decoder did not return an MPS tensor")
    report.update({"layer": args.layer, "read_bytes": read_bytes,
                   "layer_quantization": quantization,
                   "packed_shape": list(q.shape), "decoded_shape": list(decoded.shape),
                   "dequant_correctness": metrics(reference, decoded), "samples": []})
    torch.testing.assert_close(reference, decoded.cpu(), rtol=0, atol=0)
    generator = torch.Generator(device="cpu").manual_seed(20260906)
    x = torch.randn((args.tokens, decoded.shape[1]), generator=generator).to(device="mps", dtype=dtype)
    pre = tensors.get(prefix + "pre_quant_scale")
    if pre is not None:
        x = x * pre.to(device="mps", dtype=dtype)
    expected_linear = torch.nn.functional.linear(x, reference.to("mps"))
    actual_linear = torch.nn.functional.linear(x, decoded)
    report["linear_correctness"] = metrics(expected_linear, actual_linear)
    report["awq_pre_quant_scale_applied"] = pre is not None
    torch.testing.assert_close(expected_linear, actual_linear, rtol=0, atol=0)
    torch.save({"linear": actual_linear.cpu(), "decoded_sample": decoded[:64, :128].cpu()}, args.output / "tensors.pt")
    del decoded, reference, expected_linear, actual_linear
    functions = {"cpu_detour": cpu_detour, "selected_dispatch": selected}
    for function in functions.values():
        function()
    torch.mps.synchronize()
    deadline = time.monotonic() + args.max_seconds
    for pair in range(args.pairs):
        if time.monotonic() >= deadline:
            break
        if (thermal.snapshot()["state"] or 0) >= 2:
            break
        for label in list(functions) if pair % 2 == 0 else list(functions)[::-1]:
            before = thermal.snapshot()
            torch.mps.synchronize()
            start = time.perf_counter()
            output = functions[label]()
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            after = thermal.snapshot()
            report["samples"].append({"pair": pair, "path": label, "seconds": elapsed,
                                      "thermal_before": before, "thermal_after": after,
                                      "memory_after": snapshot()})
            del output
        write_report(args, report)
    bad = {s["pair"] for s in report["samples"] if any((s[k]["state"] or 0) >= 2 for k in ("thermal_before", "thermal_after"))}
    report["summary"] = {}
    for label in functions:
        values = [s["seconds"] for s in report["samples"] if s["path"] == label and s["pair"] not in bad]
        if values:
            half = max(1, len(values) // 2)
            drift = statistics.median(values[-half:]) / statistics.median(values[:half]) - 1
            report["summary"][label] = {"accepted_pairs": len(values), "median_seconds": statistics.median(values),
                                         "drift_fraction": drift, "drift_flag": len(values) >= 4 and abs(drift) > .2}
    report["rejected_pairs"] = sorted(bad)


def encode_benchmark(args, report, thermal, counts, shapes):
    import comfy.sd
    if args.model_path.stat().st_size != report["expected_file_bytes"]:
        raise ValueError("Local checkpoint size does not match its header; copy may be incomplete")
    # Two packed copies plus 8 GiB reserve is a conservative preflight, not a
    # claim that macOS will provide a fixed amount of usable GPU memory.
    required = 2 * args.model_path.stat().st_size + 8 * 1024**3
    if snapshot()["available_ram_bytes"] < required:
        raise RuntimeError(f"Insufficient available RAM for bounded encode preflight: need {required / 1024**3:.1f} GiB")
    if str(args.model_path.resolve()).startswith("/Volumes/"):
        raise ValueError("Full encode requires the local copy; layer mode may read the network model")
    torch.mps.synchronize()
    start = time.perf_counter()
    clip = comfy.sd.load_clip([str(args.model_path)], clip_type=comfy.sd.CLIPType.MINIMAX,
                             model_options={"dtype": getattr(torch, args.dtype), "initial_device": torch.device("cpu")})
    torch.mps.synchronize()
    report["load_seconds"] = time.perf_counter() - start
    if clip.patcher.load_device.type != "mps":
        raise RuntimeError("ComfyUI selected a non-MPS text-encoder device; refusing CPU timings")
    storage = collections.Counter()
    for parameter in clip.cond_stage_model.parameters():
        raw = getattr(parameter, "_qdata", parameter)
        storage[f"{type(parameter).__name__}:{raw.dtype}:{raw.device}"] += raw.numel() * raw.element_size()
    # Modules provide the public format marker; some Tensor subclasses expose
    # their layout under a different private name across Kitchen versions.
    quant_modules = [m for m in clip.cond_stage_model.modules() if getattr(m, "quant_format", None) == "nvfp4"]
    if not quant_modules or any(not hasattr(m.weight, "_qdata") for m in quant_modules):
        raise RuntimeError("Model did not retain packed NVFP4 weights; refusing expanded-model timing")
    report["raw_parameter_storage_bytes_after_load"] = dict(storage)
    report["nvfp4_modules"] = len(quant_modules)
    report["load_dispatch_counts"] = dict(counts)
    report["memory_after_load"] = snapshot()
    tokens = clip.tokenize(args.prompt)
    report["token_count"] = sum(len(batch) for batches in tokens.values() for batch in batches)
    if report["token_count"] > 64:
        raise ValueError("Use at most 64 tokens for this bounded text-only benchmark")
    report["runs"] = []
    write_report(args, report)
    deadline = time.monotonic() + args.max_seconds
    for index in range(args.runs):
        if time.monotonic() >= deadline:
            break
        before = thermal.snapshot()
        if (before["state"] or 0) >= 2:
            break
        if snapshot()["available_ram_bytes"] < 8 * 1024**3:
            raise RuntimeError("Less than 8 GiB available RAM before encode; stopping")
        counts.clear()
        shapes.clear()
        torch.mps.synchronize()
        start = time.perf_counter()
        output = clip.encode_from_tokens(tokens, return_dict=True)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        after = thermal.snapshot()
        saved = {k: v.detach().cpu() for k, v in output.items() if isinstance(v, torch.Tensor)}
        if not saved or not all(bool(torch.isfinite(v).all()) for v in saved.values()):
            raise RuntimeError("Text encoding returned no tensors or non-finite values")
        torch.save(saved, args.output / "tensors.pt")
        report["runs"].append({"index": index, "seconds": elapsed, "cold": index == 0,
                               "thermal_before": before, "thermal_after": after,
                               "rejected_thermal": any((s["state"] or 0) >= 2 for s in (before, after)),
                               "memory_after": snapshot(), "dispatch_counts": dict(counts),
                               "dispatch_shapes": dict(shapes),
                               "output_shapes": {k: list(v.shape) for k, v in saved.items()}, "all_finite": True})
        write_report(args, report)
        print(json.dumps(report["runs"][-1]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("header", "layer", "encode"), default="layer")
    parser.add_argument("--comfy", type=Path, default=DEFAULT_COMFY)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--kitchen-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layer", default="model.layers.0.self_attn.k_proj")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-read-mib", type=int, default=128)
    parser.add_argument("--prompt", default="A red sailboat on a peaceful lake.")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
        return
    if args.output is None:
        parser.error("--output is required")
    if not 1 <= args.pairs <= 30 or not 1 <= args.runs <= 4 or not 1 <= args.tokens <= 64 or not 0 < args.max_seconds <= 1200:
        parser.error("Use 1–30 pairs, 1–4 runs, 1–64 tokens, and at most 1200 seconds")
    args.model_path = args.model_path or args.comfy / "models/text_encoders" / MODEL_NAME
    header, offset = read_header(args.model_path)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "mode": args.mode, "model": str(args.model_path), "file_bytes": args.model_path.stat().st_size,
              "expected_file_bytes": offset + max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"),
              "dtype": args.dtype, "prompt": args.prompt, "header_bytes": offset - 8,
              "checkpoint_tensor_dtypes": dict(collections.Counter(v["dtype"] for k, v in header.items() if k != "__metadata__")),
              "caveats": ["Same quantized checkpoint, not comparison to original unquantized model.",
                          "Thermal pressure is not temperature; nominal does not establish equal clock speeds.",
                          "Memory snapshots are not peak-memory measurements.",
                          "Time budget checked between pairs/runs; one encode may exceed it."]}
    if args.mode == "header":
        write_report(args, report)
        print(json.dumps(report, indent=2))
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.comfy))
    if args.kitchen_source:
        sys.path.insert(0, str(args.kitchen_source))
    sys.argv = [sys.argv[0]]
    global torch
    import torch
    torch.set_grad_enabled(False)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing CPU fallback timings")
    import comfy_kitchen as ck
    from comfy_kitchen.backends.eager.quantization import dequantize_nvfp4 as reference_decode
    import comfy.sd
    package = args.comfy / "custom_nodes/ComfyUI-AppleSilicon-FP8"
    spec = importlib.util.spec_from_file_location("validation_apple_fp8", package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    counts, shapes = track_dispatch(ck)
    thermal = Thermal()
    report.update({"kitchen_source": ck.__file__, "torch": torch.__version__, "thermal_start": thermal.snapshot(),
                   "memory_start": snapshot()})
    with torch.inference_mode():
        if args.mode == "layer":
            layer_benchmark(args, report, header, offset, ck, reference_decode, thermal)
            report["dispatch_counts"] = dict(counts)
            report["dispatch_shapes"] = dict(shapes)
        else:
            encode_benchmark(args, report, thermal, counts, shapes)
    report["thermal_end"] = thermal.snapshot()
    report["memory_end"] = snapshot()
    write_report(args, report)
    print(json.dumps({"output": str(args.output), "summary": report.get("summary")}), flush=True)


if __name__ == "__main__":
    main()
