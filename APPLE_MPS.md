# Apple MPS branch

This branch combines two upstream works while keeping their responsibilities
separate:

- [Comfy-Org/comfy-kitchen#145](https://github.com/Comfy-Org/comfy-kitchen/pull/145)
  supplies the bounded-memory floating-point GEMM fallback used when PyTorch has
  no INT8 GEMM for a device. This is the active INT8/ConvRot path on M1-M4.
- [Comfy-Org/comfy-kitchen#107](https://github.com/Comfy-Org/comfy-kitchen/pull/107)
  supplies the experimental fused Metal 4 INT8 kernels. The older fallback from
  that PR is intentionally not included because #145 supersedes it.

The fused implementation is gated to an identified M5-or-newer Apple chip.
Metal shader compilation by itself is not treated as proof of support because
the shader can compile on older hardware without providing correct cooperative
INT8 TensorOps. Unknown chips fail closed to the floating-point fallback.

For ConvRot inputs, the dispatcher rotates the activation once before trying the
fused kernel. If the fused path is unavailable, it calls the fallback with
`convrot=False`, preventing a second rotation.

The branch is based on comfy-kitchen v0.2.33 (`e9ea99c`) and identifies itself
as `0.2.33+chargeuk.mps5`. Install the tested commit after ComfyUI's requirements
so ComfyUI's PyPI pin does not replace it.

## M4 backend

The `mps` registry backend provides INT8 linear, FP8 and NVFP4 dequantization. INT8
weights remain compressed; FP16 activations widen to FP32 before rotation and
GEMM so an unscaled intermediate cannot overflow before the output scale.
BF16 inputs retain their existing GEMM precision. A fused Metal epilogue applies
FP32 scale, then bias, then output casting without intermediate tensor copies.
It deliberately does not contract multiplication and addition into an FMA.
Weight/output temporaries remain chunked; widening FP16 activations adds an
activation-sized FP32 allocation. The single-chunk path avoids a redundant
full-output allocation and copy.

Opt-in regular-Hadamard rotation for contiguous inference inputs with groups
64/256 uses a SIMD-shuffle Metal kernel with FP32 registers. INT8 FP16 widening
is fused into that rotation. It is faster, but changed summation order caused
small numerical differences that compounded through SAM's detections. It remains
disabled by default. The older threadgroup-memory rotation is retained for
other eligible small inputs under the same opt-in. Both helpers use only
Kitchen's known regular basis, never an arbitrary supplied H.

The AppleSilicon-FP8 W4A16 bridge uses a fused signed INT4 unpack/row-scale
decoder, plus the SIMD rotation only when opted in. Weight-scale and product rounding match the
existing node; its floating-point activation/GEMM policy is unchanged. This is
not native INT4 matrix multiplication and introduces no activation quantization.

FP8 E4M3FN/E5M2 decoding reads uint8 storage directly, preserving eager
cast-before-scale rounding and FP16/BF16/FP32 outputs. Strided input copies are
performed in byte form on the GPU, avoiding full-size int64 gather indices and
CPU weight copies. Unsupported shader contracts use a byte-lookup fallback;
input gradients are explicitly unsupported. CUDA/CPU behavior is unchanged.

Unscaled FP8 decoding has its own entry point without a dummy scale allocation
or multiplication. FP16/BF16 outputs use four bytes per thread at 1,638,400
elements and above; smaller tensors and FP32 outputs retain scalar decoding.
The Apple-node bridge decodes raw parameters directly to the
requested compute dtype, avoiding an intermediate FP32 tensor and extra cast.
QuantizedTensor scaling and weight/bias functions retain their existing behavior.

FP8 checkpoint format does not automatically imply FP8 runtime storage. On M4,
ComfyUI normally expands a model that fits into FP16. Its existing startup option
`--fp8_e4m3fn-unet` requests FP8 diffusion-model storage instead; with this node
and Kitchen, weights are decoded for floating-point computation at use. This is
not native FP8 GEMM and does not require `--supports-fp8-compute`. The storage
flag is session-wide for automatic diffusion-model dtype selection, not a
per-checkpoint preference or a text-encoder/VAE precision setting.

NVFP4 decoding reads packed E2M1 weights and directly addresses cuBLAS-swizzled
E4M3 block scales in Metal. Both nibble orders, byte-safe strided inputs and
FP16/BF16/FP32 outputs are supported. It preserves the eager implementation's
separate rounding of the tensor scale, block scale, scale product and output.
Unsupported shaders retain CPU compatibility decoding; out-of-memory errors
propagate instead of triggering that more memory-hungry fallback. Scale gradients
are explicitly unsupported. This is NVFP4 storage with floating-point compute,
not Blackwell-style native FP4 matrix multiplication.

AppleSilicon-FP8 patches the eager backend, so the higher-priority MPS registry
path can bypass its CPU NVFP4 decoder without another custom-node or ComfyUI
patch. ComfyUI still owns model loading and when weights are expanded. No
full-model decoded-weight cache, new activation quantization, or fused NVFP4
matrix multiplication is introduced. MXFP8 retains its existing fallback.

`integrations/applesilicon-fp8-kitchen.patch` is the small companion patch for
AppleSilicon-FP8 v1.3.2 (`74734a1`): its shared FP8 decoder and INT4 W4A16 path use
Kitchen when present and retain their original fallbacks otherwise. It also
requests direct-to-compute-dtype raw FP8 conversion, with regression tests for
both FP8 formats, signed zeros and weight/bias functions. Apply with `git apply --check`
then `git apply` in that custom-node checkout. It does not remove the node's
other compatibility fixes and does not modify core ComfyUI.

Set `COMFY_KITCHEN_DISABLE_MPS=1` before startup to disable this new backend and
all direct shader hooks. The safe widened INT8 fallback remains active. Set
`COMFY_KITCHEN_MPS_ROTATION=1` to opt into experimental SIMD/legacy rotation;
this trades bit-identical model results for additional speed. Registry-level
backend overrides alone do not disable direct Apple-node/eager helper calls.
Enable Python DEBUG logging for `comfy_kitchen.dispatch` for registry choices.

## Validation and benchmarking

Run `python -m pytest tests/test_mps_fp8.py tests/test_mps_nvfp4.py tests/test_mps_rotation.py
tests/test_mps_dispatch.py tests/test_int8_fallback.py tests/test_int8_mps.py`
outside a sandbox that blocks Metal device access.
The ConvRot additions have dedicated `test_mps_int4.py`,
`test_mps_int8_epilogue.py`, and `test_mps_convrot.py` suites.

Run `python benchmarks/benchmark_mps.py --output /tmp/kitchen-mps.json` while
other GPU work is idle. It alternates AB/BA order, saves every timing, records
macOS thermal pressure, rejects serious/critical-pressure pairs and flags drift.
Nominal pressure is not proof of identical clocks or zero throttling. Report
operation timings separately from end-to-end model generation.

`benchmark_mps_int4.py` measures unpack/scale and `benchmark_mps_convrot.py`
measures rotation on observed SAM shapes. `model_validation.py --profile-convrot`
adds intrusive synchronized stage timings: these diagnose bottlenecks but must
not be compared with ordinary inference latency. All scripts use local models
or synthetic tensors without downloading anything.

`model_validation.py --model pony-fp8 --pony-fp8-storage` forwards ComfyUI's
actual storage flag and records both storage and compute dtype. Add
`--profile-fp8` only for intrusive diagnostics, not ordinary inference timings.
`benchmark_mps_fp8_unscaled.py --baseline-module PATH --output PATH` compares
this decoder with an explicit previous-version module using alternating pairs.
Keep the old module or extract it from the rollback wheel before updating.

`benchmarks/benchmark_nvfp4_model.py` tests a user-supplied local MiniMax H3 Qwen3-VL
NVFP4 text encoder without downloading anything. `--mode layer` reads only a
selected layer, checks decoding and AWQ-scaled linear outputs against a CPU
decode reference, and alternates CPU/native timings. `--mode encode` exercises
the native ComfyUI loader and text encoder, checks that weights remain packed,
and saves raw outputs, thermal pressure and dispatch counts. Use installed
Kitchen for the baseline and `--kitchen-source PATH` for the candidate. Memory
snapshots are not peak-memory measurements. Compare output directories with
`--compare BASELINE_DIR CANDIDATE_DIR`.

Measured results and limits are recorded in [M4_VALIDATION.md](M4_VALIDATION.md)
for the original FP8/INT8 work, [M4_NVFP4_VALIDATION.md](M4_NVFP4_VALIDATION.md)
for NVFP4, and [M4_CONVROT_VALIDATION.md](M4_CONVROT_VALIDATION.md) for the mps4
INT8/INT4 conversion improvements and opt-in SIMD experiment. The forced-FP8
storage follow-up is documented in [M4_FP8_VALIDATION.md](M4_FP8_VALIDATION.md).

The M4 implementation is original code informed by the regular-basis math and
the approaches in [AppMana's MPS branch](https://github.com/AppMana/forks-comfy-kitchen-m1-m4/tree/mps-backend),
[MLX](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels), and
[AppleSilicon-FP8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8).
No MLX runtime bridge, decoded-weight cache, or experimental fused GEMM is
enabled in this milestone. M5-only cooperative INT8 gates remain unchanged.
