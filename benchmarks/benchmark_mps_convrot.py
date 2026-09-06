#!/usr/bin/env python3
"""Alternating dense/SIMD ConvRot timings on representative SAM shapes."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from benchmark_mps import Thermal, measure_case
from comfy_kitchen.backends.mps.convrot import try_rotate
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--wide', action='store_true', help='FP32 output/rotation reference, as used by INT8 with FP16 input')
    parser.add_argument('--pairs', type=int, default=20)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--max-seconds', type=float, default=45)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 30 or not 0 <= args.warmups <= 10 or not 0 < args.max_seconds <= 60:
        parser.error('Use 1–30 pairs, 0–10 warmups and at most 60 seconds')
    if not torch.backends.mps.is_available():
        parser.error('MPS required; refusing CPU timing')
    os.environ['COMFY_KITCHEN_MPS_ROTATION'] = '1'
    dtype = getattr(torch, args.dtype)
    output_dtype = torch.float32 if args.wide else dtype
    thermal = Thermal()
    report = {'dtype': str(dtype), 'output_dtype': str(output_dtype), 'torch': torch.__version__,
              'cases': [], 'notes': 'Rotation only, not model latency. Nominal thermal pressure does not prove fixed clocks.'}
    generator = torch.Generator().manual_seed(1773)
    deadline = time.monotonic() + args.max_seconds
    with torch.inference_mode():
        for group, rows, columns in ((256, 201, 256), (256, 5184, 1024),
                                     (256, 5184, 2048), (64, 5184, 4736)):
            if time.monotonic() >= deadline or (thermal.snapshot()['state'] or 0) >= 2:
                break
            x = torch.randn(rows, columns, generator=generator).to(dtype).to('mps')
            h = _build_hadamard(group, 'mps', output_dtype)
            def dense():
                return _rotate_activation(x.to(output_dtype), h, group)
            def simd():
                result = try_rotate(x, group, output_dtype=output_dtype)
                if result is None:
                    raise RuntimeError('SIMD path declined; refusing to time fallback')
                return result
            reference, actual = dense(), simd()
            tolerance = {torch.float32: 3e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[output_dtype]
            torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
            error = {'exact': torch.equal(actual.cpu(), reference.cpu()),
                     'max_abs': (actual.float() - reference.float()).abs().max().item()}
            del reference, actual
            result = measure_case('ConvRot', {'dense': dense, 'simd': simd}, thermal, args, deadline,
                                  shape=[rows, columns], group=group, correctness=error)
            report['cases'].append(result)
            print(json.dumps({'shape': [rows, columns], 'summary': result['summary']}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
