#!/usr/bin/env python3
"""Compare an explicitly supplied previous FP8 decoder with this checkout.

The node comparison includes the old FP32 intermediate and final dtype cast.
All comparisons use unscaled FP8 weights, not native FP8 matrix multiplication.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

from benchmark_mps import Thermal, measure_case, command
import torch
from comfy_kitchen.backends.mps import fp8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=20)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--max-seconds', type=float, default=45)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    args = parser.parse_args()
    if not 1 <= args.pairs <= 30 or not 0 <= args.warmups <= 10 or not 0 < args.max_seconds <= 60:
        parser.error('Use 1–30 pairs, 0–10 warmups and at most 60 seconds')
    if not torch.backends.mps.is_available():
        parser.error('MPS unavailable; refusing CPU timings')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('previous_mps_fp8', args.baseline_module)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    thermal = Thermal()
    dtype = getattr(torch, args.dtype)
    result = {'baseline_module': str(args.baseline_module.resolve()),
              'candidate_module': fp8.__file__, 'torch': torch.__version__,
              'chip': command('sysctl', '-n', 'machdep.cpu.brand_string'),
              'dtype': args.dtype, 'format': 'E4M3FN', 'seed': 20260907,
              'pairs': args.pairs, 'warmups': args.warmups, 'max_seconds': args.max_seconds, 'cases': [],
              'note': 'Operation-only AB/BA timings. Nominal thermal pressure does not prove equal clocks.'}
    deadline = time.monotonic() + args.max_seconds
    generator = torch.Generator(device='cpu').manual_seed(result['seed'])
    with torch.inference_mode():
        for shape in ((320,), (320, 320), (1280, 1280), (5120, 1280)):
            if time.monotonic() >= deadline or (thermal.snapshot()['state'] or 0) >= 2:
                break
            raw = torch.randint(0, 127, shape, dtype=torch.uint8, generator=generator).to('mps')
            x = raw.view(torch.float8_e4m3fn)
            for name, old in (
                ('decoder_only', lambda: baseline.decode_fp8(x, dtype)),
                ('node_conversion', lambda: baseline.decode_fp8(x).to(dtype)),
            ):
                functions = {'mps4': old, 'candidate': lambda: fp8.decode_fp8(x, dtype)}
                torch.testing.assert_close(old().cpu(), functions['candidate']().cpu(), rtol=0, atol=0)
                result['cases'].append(measure_case(name, functions, thermal, args, deadline,
                                                   shape=list(shape), correctness='exact'))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{key: case[key] for key in ('name', 'shape', 'summary')}
                      for case in result['cases']], indent=2))


if __name__ == '__main__':
    main()
