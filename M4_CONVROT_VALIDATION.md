# M4 ConvRot performance validation — 2026-09-07

Candidate: `0.2.33+chargeuk.mps4`, compared with installed `mps3` (`315eb8e`).
Machine: Apple M4 Max, 128 GiB, macOS 26.4, Python 3.12.7, PyTorch 2.14.0.
ComfyUI 0.34.5 and AppleSilicon-FP8 1.3.2 remain unchanged apart from the
documented optional Kitchen bridge in the node's INT4 implementation.

## Enabled improvements

- INT8: one Metal operation applies FP32 output scaling, bias and dtype
  conversion, including direct writes into bounded output chunks. Single-chunk
  calls avoid an extra full-output allocation/copy. The FP16-to-FP32 rotation
  and GEMM overflow safeguards are retained.
- INT4: one Metal operation unpacks signed nibbles and applies row scales,
  preserving the existing node's scale-cast and product-cast rounding. Its
  W4A16 activation and floating-point GEMM policy is unchanged.
- Compressed model weights remain compressed between calls. No decoded-weight
  cache, new activation quantization, model download or extra node pack.

These are M4-compatible Metal conversion kernels, not native INT8/INT4 MMA.
M5-only cooperative matrix-kernel gates are unchanged.

## Final model results

Same local SAM3.1 checkpoint and fixed red-balloon input for each old/new pair;
1008×1008 detector input and all 200 raw masks/boxes/scores compared. Each model
was run in fresh processes in **old/new/new/old** order, four runs per process.
The first run is excluded from warm medians: six warm samples per version.

| Model | Old warm median | New warm median | Latency reduction | Throughput ratio |
|---|---:|---:|---:|---:|
| INT8 ConvRot | 1.174896 s | 0.951931 s | 18.98% | 1.234× |
| INT4 ConvRot (W4A16 path) | 0.884945 s | 0.858150 s | 3.03% | 1.031× |

All raw model outputs are **exactly equal** across these final comparisons.
Every candidate run completes 182 native INT8 epilogues or 182 native INT4
decodes, respectively. Experimental rotation completes zero calls by default.
All 32 model runs recorded nominal thermal pressure before and after execution.
Thermal pressure is not a temperature/clock measurement or proof of zero
throttling. Early exploratory timings varied; the conservative final ABBA
results above are the headline measurements.

Warm samples in seconds, in execution order within each version:

- INT8 old: 1.173607, 1.181388, 1.174806, 1.174987, 1.177492, 1.162821.
- INT8 new: 0.985204, 0.955792, 0.951927, 0.948281, 0.951302, 0.951936.
- INT4 old: 0.892716, 0.889390, 0.875471, 0.884993, 0.884897, 0.882143.
- INT4 new: 0.863254, 0.859039, 0.857262, 0.855964, 0.860651, 0.855990.

These measure the native SAM detector, not full workflow loading, network model
access, text encoding, or every possible diffusion/video model. Other ConvRot
models and layer shapes can have different gains.

## Regression and installed-package checks

- Final full suite: **1,017 passed, 1,608 hardware-specific skipped**. One
  pre-existing RMSNorm dtype warning; no test failures.
- Dedicated new coverage: 39 INT4 decoder tests, 68 INT8 epilogue tests and
  60 experimental rotation tests. Includes exact rounding, signed zeros,
  byte packing, offsets/strides, mutation, bounded chunk writes, alias rejection,
  FP16 overflow, unsupported compilation/launch and OOM propagation.
- Existing FP8/NVFP4 tests remain passing. Separate multiply/add rounding is
  retained; extreme FP32 subnormals follow the prior MPS flush-to-zero behavior.
- Installed wheel's 49 Python source files byte-match the checkout. `pip check`
  passes. Both local models were rerun from the installed wheel and again match
  their old outputs exactly, with all 182 native operations completing per run.
- Normal ComfyUI startup on temporary localhost port 8189 reports mps4 and MPS,
  with AppleSilicon-FP8 loading successfully. The temporary server was stopped;
  the normal launcher and ComfyUI core are unchanged.

## SIMD rotation experiment — opt-in only

A new regular-Hadamard SIMD-shuffle implementation supports groups 64/256,
uses FP32 registers, and can fuse FP16 input widening. Normalization occurs on
load to avoid unnecessary overflow in large finite FP32/BF16 inputs.

It is faster than dense rotation on the tested shapes. For FP16 input with
FP32 output, paired 20-sample medians include 0.468→0.291 ms at 5184×1024 and
2.036→0.866 ms at 5184×4736 (group 64). A separate 5184×2048 result was flagged
for timing drift and is not used for a stable speed claim.

However, butterfly summation changes rounding. When enabled throughout SAM,
small layer differences compound: INT8 mask-logit binary IoU was 0.99120 with
up to 17.5-pixel raw-box difference; INT4 IoU was 0.99278 with up to 4.375-pixel
raw-box difference. These compare all 200 raw proposals, not just the selected
detection. There is no quality-equivalence claim.

Therefore rotation remains **disabled by default**. Explicitly set
`COMFY_KITCHEN_MPS_ROTATION=1` before startup to experiment. The launch script
does not set it. `COMFY_KITCHEN_DISABLE_MPS=1` disables all new direct helpers
while retaining the safe existing floating-point fallback.

## Reproduction and rollback

- `benchmarks/model_validation.py`: local-model inference and raw-output
  comparison; `--profile-convrot` adds intrusive stage timings, not normal latency.
- `benchmarks/benchmark_mps_int4.py`: seven observed SAM weight shapes, exact
  unpack/scale comparisons and alternating thermal-screened timings. FP16
  operation-only speedups were 1.29–4.16×; these are not whole-model speedups.
- `benchmarks/benchmark_mps_convrot.py`: paired experimental rotation timings.
- Raw local artifacts: `outputs/convrot/` in the task workspace; final model
  directories are `final-sam-{int8,int4}-{baseline,candidate}-{a,b}`.
- Node bridge: `integrations/applesilicon-fp8-kitchen.patch`, checked against
  AppleSilicon-FP8 commit `74734a1`; it keeps the node's original fallbacks.
- Previous wheel preserved at
  `/Users/danielstapleton/Documents/comfyui/rollback/mps3/comfy_kitchen-0.2.33+chargeuk.mps3-py3-none-any.whl`.

Install the rollback wheel with `pip install --no-deps --force-reinstall` in
ComfyUI's venv and restart. The node bridge automatically falls back when the
new helpers are absent. No ComfyUI core changes or launcher changes are needed.
