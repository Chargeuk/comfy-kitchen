#!/usr/bin/env python3
"""Fixed-input, no-download ComfyUI model validation; run one model per process.

Baseline uses installed Kitchen. Candidate: --kitchen-source /path/to/fork.
Compare: --compare BASELINE_DIR CANDIDATE_DIR (CPU only).
"""
import argparse
import collections
import ctypes
import ctypes.util
import importlib.util
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

DEFAULT_COMFY = Path.home() / 'Documents/comfyui/ComfyUI'
MODELS = {
    'sam-fp16': 'sam3.1_multiplex_fp16.safetensors',
    'sam-int8': 'sam3.1_multiplex_int8_convrot_selective_fp16clip_fp32source.safetensors',
    'sam-int4': 'sam3.1_multiplex_w4a4_convrot_selective_fp16clip_fp32source.safetensors',
    'pony-fp8': 'pony-20-real-dream-float8_e4m3fn.safetensors',
}


class Thermal:
    """Read native thermal pressure without importing either Kitchen version."""
    def __init__(self):
        self.read = None
        try:
            self.foundation = ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
            self.objc = ctypes.CDLL(ctypes.util.find_library('objc'))
            self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
            self.objc.objc_getClass.restype = ctypes.c_void_p
            self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
            self.objc.sel_registerName.restype = ctypes.c_void_p
            send = ctypes.cast(self.objc.objc_msgSend, ctypes.c_void_p).value
            pointer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
            integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(send)
            process = pointer(self.objc.objc_getClass(b'NSProcessInfo'), self.objc.sel_registerName(b'processInfo'))
            selector = self.objc.sel_registerName(b'thermalState')
            if process:
                self.read = lambda: int(integer(process, selector))
        except (OSError, AttributeError, TypeError):
            pass

    def snapshot(self):
        state = self.read() if self.read else None
        return {'source': 'NSProcessInfo.thermalState', 'state': state,
                'label': {0: 'nominal', 1: 'fair', 2: 'serious', 3: 'critical'}.get(state, 'unavailable')}


def compare(paths):
    import torch
    torch.set_grad_enabled(False)
    aa, bb = [torch.load(Path(p) / 'tensors.pt', map_location='cpu', weights_only=True) for p in paths]
    result = {}
    for key in sorted(aa.keys() | bb.keys()):
        if key not in aa or key not in bb or aa[key].shape != bb[key].shape:
            result[key] = {'incompatible': True}
            continue
        a, b = aa[key].float(), bb[key].float()
        delta = a - b
        result[key] = {'shape': list(a.shape), 'finite': bool(torch.isfinite(b).all()),
                       'max_abs': delta.abs().max().item(), 'mean_abs': delta.abs().mean().item(),
                       'rmse': delta.square().mean().sqrt().item(),
                       'relative_l2': (delta.norm() / a.norm().clamp_min(1e-12)).item()}
        if 'mask' in key:
            x, y = a > 0, b > 0
            result[key]['binary_iou'] = ((x & y).sum() / (x | y).sum().clamp_min(1)).item()
            result[key]['both_empty'] = not bool((x | y).any())
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODELS, default='sam-fp16')
    ap.add_argument('--model-path', type=Path)
    ap.add_argument('--comfy', type=Path, default=DEFAULT_COMFY)
    ap.add_argument('--kitchen-source', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--runs', type=int, default=2)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--skip-apple-patches', action='store_true')
    ap.add_argument('--inspect-only', action='store_true', help='Load and record parameter storage, without inference')
    ap.add_argument('--profile-convrot', action='store_true', help='Intrusive per-operation MPS timings for INT8/INT4 paths')
    ap.add_argument('--compare', nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        return
    if args.output is None:
        ap.error('--output is required for inference')
    if args.runs < 1:
        ap.error('--runs must be positive')
    model_path = args.model_path or args.comfy / 'models/checkpoints' / MODELS[args.model]
    if not model_path.is_file():
        ap.error(f'Model absent; no download attempted: {model_path}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(args.comfy))
    if args.kitchen_source:
        sys.path.insert(0, str(args.kitchen_source))
    sys.argv = [sys.argv[0]]  # Keep harness arguments out of Comfy's parser.
    import torch
    torch.set_grad_enabled(False)
    import numpy as np
    from PIL import Image, ImageDraw
    import comfy_kitchen as ck
    import comfy.sd
    import comfy.model_management as mm
    import nodes
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; run outside restricted sandbox. Refusing CPU timing.')
    if not args.skip_apple_patches:
        package = args.comfy / 'custom_nodes/ComfyUI-AppleSilicon-FP8'
        spec = importlib.util.spec_from_file_location('validation_apple_fp8', package / '__init__.py',
                                                     submodule_search_locations=[str(package)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    from comfy_kitchen.backends.eager import quantization as qmod
    counts = collections.Counter()
    shapes = collections.Counter()
    profile = collections.defaultdict(lambda: {'count': 0, 'seconds': 0.0, 'shapes': {}})
    def profile_call(label, fn, *a, **kw):
        if not args.profile_convrot:
            return fn(*a, **kw)
        sync()
        started = time.perf_counter()
        result = fn(*a, **kw)
        sync()
        elapsed = time.perf_counter() - started
        item = profile[label]
        item['count'] += 1
        item['seconds'] += elapsed
        desc = '|'.join(f'{tuple(v.shape)}:{v.dtype}' for v in a[:2] if isinstance(v, torch.Tensor))
        shape = item['shapes'].setdefault(desc, {'count': 0, 'seconds': 0.0})
        shape['count'] += 1
        shape['seconds'] += elapsed
        return result
    # Registry resolves backend attributes at call time. Wrap both exported and
    # implementation-module entries, preserving original object identity.
    for name in ('int8_linear', '_int8_linear_dequant', '_mps_int8_linear',
                 'dequantize_per_tensor_fp8', 'quantize_per_tensor_fp8',
                 'dequantize_int8_convrot_weight_dtype', 'dequantize_nvfp4', 'dequantize_mxfp8',
                 'convrot_w4a4_linear', 'int4_linear', 'dequantize_convrot_w4a4_weight'):
        wrappers = {}
        for module in (qmod, *ck.registry._backends.values()):
            if module is None or not hasattr(module, name):
                continue
            original = getattr(module, name)
            if id(original) not in wrappers:
                def wrapper(*a, _orig=original, _name=name, **kw):
                    counts[_name] += 1
                    if a and isinstance(a[0], torch.Tensor):
                        shapes[f'{_name}:{tuple(a[0].shape)}:{a[0].dtype}'] += 1
                    if len(a) > 1 and isinstance(a[1], torch.Tensor):
                        shapes[f'{_name}:weight={tuple(a[1].shape)}:{a[1].dtype}'] += 1
                    if _name == '_int8_linear_dequant':
                        return profile_call(_name + ':inclusive', _orig, *a, **kw)
                    return _orig(*a, **kw)
                wrappers[id(original)] = wrapper
            setattr(module, name, wrappers[id(original)])
    if args.profile_convrot:
        # Only plain operands: an outer QuantizedTensor call dispatches to a
        # separately profiled floating-point linear operation.
        _linear = torch.nn.functional.linear
        def profiled_linear(*a, **kw):
            if len(a) > 1 and all(isinstance(a[i], torch.Tensor) and not hasattr(a[i], '_layout_cls') for i in (0, 1)):
                return profile_call('torch.nn.functional.linear', _linear, *a, **kw)
            return _linear(*a, **kw)
        torch.nn.functional.linear = profiled_linear
    mps_modules = tuple('comfy_kitchen.backends.mps.' + name for name in ('fp8', 'rotation', 'int4', 'convrot', 'int8'))
    for module_name in mps_modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass
    if args.profile_convrot:
        for module in (qmod, sys.modules.get('comfy_kitchen.backends.eager.convrot_w4a4')):
            if module is not None and hasattr(module, '_rotate_activation'):
                original = module._rotate_activation
                module._rotate_activation = lambda *a, _o=original, **kw: profile_call('_rotate_activation', _o, *a, **kw)
        for name, module in list(sys.modules.items()):
            if name.startswith('validation_apple_fp8._patches') and hasattr(module, '_unpack_int4_signed_fast'):
                original = module._unpack_int4_signed_fast
                module._unpack_int4_signed_fast = lambda *a, _o=original, **kw: profile_call('_unpack_int4_signed_fast', _o, *a, **kw)
    for module_name, module in list(sys.modules.items()):
        if not (module_name.startswith('validation_apple_fp8._patches') or
                module_name in mps_modules):
            continue
        for name in ('decode_fp8', '_w4a16_linear_mps', 'try_regular_rotation',
                     'unpack_int4_scaled', 'try_rotate', 'int8_epilogue'):
            original = getattr(module, name, None)
            if original is None:
                continue
            label = f'{module_name}.{name}'
            def tracked(*a, _orig=original, _label=label, **kw):
                counts[_label] += 1
                if a and isinstance(a[0], torch.Tensor):
                    desc = '|'.join(f'{tuple(v.shape)}:{v.dtype}' for v in a[:2] if isinstance(v, torch.Tensor))
                    shapes[f'{_label}:{desc}'] += 1
                result = profile_call(_label + ':inclusive', _orig, *a, **kw) if '_w4a16_linear_mps' in _label else _orig(*a, **kw)
                if result is not None:
                    counts[_label + ':success'] += 1
                return result
            setattr(module, name, tracked)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'model': str(model_path), 'kitchen': ck.__file__, 'torch': torch.__version__,
              'apple_patches': not args.skip_apple_patches, 'seed': 20260906,
              'runs': [], 'notes': 'First run cold; subsequent runs warm. Same checkpoint across backends isolates kernel changes.'}
    if args.profile_convrot:
        report['profile_note'] = 'Intrusive profiling: each operation synchronizes MPS before and after timing; timings are not comparable to normal runs.'
    report['thermal_start'] = subprocess.run(['/usr/bin/pmset', '-g', 'therm'], capture_output=True, text=True).stdout
    thermal = Thermal()
    report['thermal_pressure_start'] = thermal.snapshot()
    def sync():
        torch.mps.synchronize()
    torch.manual_seed(report['seed'])
    sync()
    start = time.perf_counter()
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        str(model_path), output_vae=args.model == 'pony-fp8', output_clip=True)
    sync()
    report['load_seconds'] = time.perf_counter() - start
    storage = collections.Counter()
    for parameter in model.model.diffusion_model.parameters():
        raw = getattr(parameter, '_qdata', parameter)
        storage[f'{type(parameter).__name__}:{raw.dtype}:{raw.device.type}'] += raw.numel()
    report['loaded_storage_elements'] = dict(storage)
    report['model_compute_dtype'] = str(model.model.get_dtype())
    if args.inspect_only:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'loaded_storage_elements': dict(storage), 'model_compute_dtype': report['model_compute_dtype']}), flush=True)
        return
    if args.model.startswith('sam'):
        from comfy_extras.nodes_sam3 import _extract_text_prompts
        im = Image.new('RGB', (512, 512), '#cce5ff')
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 360, 512, 512), fill='#80b86b')
        draw.ellipse((155, 115, 355, 315), fill='#d92e2e', outline='#812323', width=5)
        draw.line((255, 315, 255, 450), fill='white', width=3)
        im.save(args.output / 'input.png')
        image = torch.from_numpy(np.array(im).copy()).float().div(255).unsqueeze(0)
        text = 'a red balloon'
        conditioning = nodes.CLIPTextEncode().encode(clip, text)[0]
        mm.load_model_gpu(model)
        dtype, device = model.model.get_dtype(), mm.get_torch_device()
        frame = comfy.utils.common_upscale(image.movedim(-1, 1), 1008, 1008, 'bilinear', crop='disabled').to(device=device, dtype=dtype)
        emb, mask, _ = _extract_text_prompts(conditioning, device, dtype)[0]
        def run():
            return model.model.diffusion_model(frame, text_embeddings=emb, text_mask=mask, threshold=0.5, orig_size=(512, 512))
    else:
        text = 'a watercolor painting of a red sailboat on a peaceful lake, mountains, daylight, no people'
        positive = nodes.CLIPTextEncode().encode(clip, text)[0]
        negative = nodes.CLIPTextEncode().encode(clip, 'text, watermark, blurry')[0]
        latent = nodes.EmptyLatentImage().generate(512, 512, 1)[0]
        def run():
            out = nodes.common_ksampler(model, report['seed'], args.steps, 5.0, 'euler', 'normal', positive, negative, latent)[0]
            return {'latent': out['samples'], 'image': vae.decode(out['samples'])}
    report['text'] = text
    sync()
    report['preparation_counts'] = dict(counts)
    report['preparation_shapes'] = dict(shapes)
    for index in range(args.runs):
        pressure_before = thermal.snapshot()
        if (pressure_before['state'] or 0) >= 2:
            raise RuntimeError('Serious thermal pressure; stopping rather than collecting distorted timings')
        counts.clear()
        shapes.clear()
        profile.clear()
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            output = run()
        sync()
        elapsed = time.perf_counter() - start
        tensors = {k: v.detach().float().cpu() for k, v in output.items() if isinstance(v, torch.Tensor)}
        profile_json = {k: {'count': v['count'], 'seconds': v['seconds'], 'shapes': dict(v['shapes'])} for k, v in profile.items()}
        report['runs'].append({'seconds': elapsed, 'counts': dict(counts), 'shapes': dict(shapes),
                               **({'profile': profile_json} if args.profile_convrot else {}),
                               'thermal_before': pressure_before, 'thermal_after': thermal.snapshot(),
                               'all_finite': all(bool(torch.isfinite(v).all()) for v in tensors.values()),
                               'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                               'mps_driver_bytes': torch.mps.driver_allocated_memory()})
        report['thermal_end'] = subprocess.run(['/usr/bin/pmset', '-g', 'therm'], capture_output=True, text=True).stdout
        torch.save(tensors, args.output / 'tensors.pt')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'run': index, 'seconds': elapsed, 'all_finite': report['runs'][-1]['all_finite']}), flush=True)
    if 'image' in tensors:
        Image.fromarray((tensors['image'][0].clamp(0, 1).numpy() * 255).astype('uint8')).save(args.output / 'output.png')
    elif 'masks' in tensors and tensors['masks'].numel():
        idx = int(tensors['scores'][0].reshape(-1).argmax())
        Image.fromarray((tensors['masks'][0, idx].sigmoid().numpy() * 255).astype('uint8')).save(args.output / 'top_mask.png')


if __name__ == '__main__':
    main()
