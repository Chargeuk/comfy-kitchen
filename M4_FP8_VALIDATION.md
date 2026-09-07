# M4 FP8 storage performance — 2026-09-07

Candidate: `0.2.33+chargeuk.mps5`, versus `mps4` (`f98bf1f`).
Apple M4 Max, 128 GiB, macOS 26.4, Python 3.12.7, PyTorch 2.14.0,
ComfyUI 0.34.5, AppleSilicon-FP8 1.3.2 with the companion Kitchen patch.

## What changed

- A dedicated unscaled Metal FP8 decoder removes the unit-scale tensor,
  its GPU fill and the unnecessary scale multiplication.
- Large unscaled FP16/BF16 outputs decode four bytes per thread. The measured
  cutoff is 1,638,400 elements; smaller tensors and FP32 output remain scalar.
  Byte loads support unaligned offsets, with bounds checks for odd tails.
- The Apple node's raw FP8 parameter conversion requests the final compute
  dtype directly instead of decoding FP32 and then casting again.
- Scaled decoding, the shared FP8 byte decoder used by NVFP4, QuantizedTensor
  scaling, weight/bias functions, INT8/INT4 safety paths and default rotation
  policy are unchanged. No decoded-weight cache or new activation quantization.

These changes accelerate FP8 storage with floating-point computation. They do
not add native FP8 matrix multiplication to M4.

## Full Pony workflow

Local checkpoint: `pony-20-real-dream-float8_e4m3fn.safetensors`. Fixed seed
20260906, watercolor sailboat prompt, 512×512, eight Euler steps, CFG 5,
including VAE decode. Loading and text encoding are outside the timed region.

The checkpoint is plain E4M3 FP8, without per-tensor quantization metadata.
Normal ComfyUI loading chooses FP16 storage on this Mac. The test explicitly
forwards the real `--fp8_e4m3fn-unet` startup option and verifies that all
2,567,463,684 diffusion parameters remain FP8 on MPS while inference uses FP16.
Every eight-step FP8 run completes 13,440 native Kitchen decodes.

Final FP8 processes ran in **old/new/new/old** order, four runs each. Excluding
the first run of each process gives six warm samples per version. FP16 controls
were repeated in two separate processes, also with six warm samples combined.

| Storage / implementation | Warm median | Diffusion parameter storage |
|---|---:|---:|
| Normal FP16 storage | 2.511494 s | 4.78 GiB |
| FP8 storage, previous mps4 | 3.140649 s | 2.39 GiB |
| FP8 storage, optimized mps5 | 2.584854 s | 2.39 GiB |

Optimized FP8 has **17.70% lower latency** than the previous FP8 path (1.215×
throughput), leaving about 2.92% overhead versus the FP16 control. The saved
image and latent tensors are exactly equal across all four final FP8 processes
and the original FP16 control. This is not a comparison with an unquantized
original checkpoint: both modes use the same already-FP8 model file.

Warm samples, seconds:

- Old: 3.239318, 3.210640, 3.198127, 3.074438, 3.083171, 3.071552.
- New: 2.598346, 2.582655, 2.591315, 2.586928, 2.579283, 2.582780.
- FP16: 2.481116, 2.512575, 2.507630, 2.510413, 2.546753, 2.564139.

All 16 final FP8 runs and eight FP16 controls recorded nominal thermal pressure
before and after execution. This is not temperature/clock telemetry or proof
of zero throttling. The old-version groups differed by about 4%, so the combined
ABBA median is reported rather than the most favorable single comparison.
Dispatch-counting overhead is present equally in the validation harness.

Parameter storage is not peak workflow memory: activations, decoded weights,
text encoders, VAE and allocator caches still need additional memory. No global
startup defaults were changed; FP16 remains marginally faster for this model.

## Iteration and operation-level measurements

The first unscaled/direct-dtype implementation, before four-byte decoding,
ran at a 2.913 s exploratory warm median. Four-byte decoding then improved the
full workflow further. Eight-byte decoding was investigated but not retained;
small tensors did not show a consistent vectorization benefit.

Final FP16 microbenchmarks use 20 alternating old/new pairs per case. Examples
including the node's former FP32 intermediate and final cast:

| Weight shape | Previous conversion | New conversion |
|---|---:|---:|
| 320×320 | 0.1787 ms | 0.1471 ms |
| 1280×1280 | 0.3409 ms | 0.2008 ms |
| 5120×1280 | 0.6153 ms | 0.3231 ms |

All final microbenchmark pairs were thermally accepted and none exceeded the
20% first-half/last-half drift threshold. These are operation-only results,
with appreciable timing variability, not whole-model speedups. Separate
four-byte probes also showed repeatable large-tensor BF16 gains; full-model
timing here covers Pony's FP16 computation, not a BF16 diffusion workflow.

## Regression checks

- Full Kitchen suite: **1,089 passed, 1,608 hardware-specific skipped**. One
  pre-existing RMSNorm dtype warning, no failures.
- Focused FP8/NVFP4 suite: **247 passed**, including 144 FP8 tests. New coverage
  includes both formats, every byte, special values, signed zeros, strides,
  offsets, odd tails, dispatch thresholds, uint32 boundaries and failure/OOM
  handling. Scaled and shared NVFP4 decoding remain unchanged.
- Apple node conversion suite: **18 passed**, including exact outputs and
  function-input dtypes for raw weight/bias conversion, plus existing mixed
  quantization and embedding regressions. All 18 also pass with native MPS
  kernels disabled, exercising the compatibility fallback.
- Installed wheel's 49 Python files byte-match the source. `pip check` passes.
- Installed-wheel Pony rerun: 2.574 s warm, all 13,440 native decodes complete,
  saved image/latent exactly match the old baseline. The final ABBA result,
  rather than this isolated rerun, remains the headline measurement.
- Two-step intrusive profiling reduced inclusive node conversion time from
  1.293 s to 0.590 s across 3,360 calls. Nested decoder time was 0.711 s and
  0.563 s respectively; these figures must not be added together.
- ComfyUI startup with the actual storage flag reports mps5, PyTorch 2.14.0
  and MPS; AppleSilicon-FP8 loads successfully. The temporary localhost:8189
  server was stopped. Core ComfyUI and the normal launcher remain unchanged.

## Reproduction and rollback

- `benchmarks/model_validation.py --model pony-fp8 --pony-fp8-storage --output DIR`
  tests forced FP8 storage; omit `--pony-fp8-storage` for normal dtype policy.
  Use `--kitchen-source PATH` for a source candidate, otherwise installed Kitchen.
  `--compare OLD_DIR NEW_DIR` compares the saved raw tensors.
- `--profile-fp8` adds intrusive synchronized decoder/conversion timings. Nested
  inclusive timings must not be summed or compared with ordinary run latency.
- `benchmarks/benchmark_mps_fp8_unscaled.py --baseline-module PATH --output PATH`
  compares an explicitly preserved older decoder with the current checkout.
- Companion node changes and tests are reproducible through
  `integrations/applesilicon-fp8-kitchen.patch`, based on node commit `74734a1`.
- Raw local workflow and final microbenchmark results are in task workspace
  `outputs/fp8-iteration/`. Vectorization probes are in
  `outputs/convrot/fp8_vector_probe*.json`; these are exploratory artifacts.
- Previous wheel is preserved at
  `/Users/danielstapleton/Documents/comfyui/rollback/mps4/comfy_kitchen-0.2.33+chargeuk.mps4-py3-none-any.whl`.
  Reinstall with `pip install --no-deps --force-reinstall` in ComfyUI's venv
  and restart. The node's direct-dtype request also works with mps4; restoring
  the exact pre-test timing baseline additionally requires reverting its
  one-line `ops_bias_fp8._to_compute` change.

For normal use, pass `--fp8_e4m3fn-unet` to `start.sh` only when FP8 diffusion
storage is desired. It is session-wide for automatic diffusion dtype selection,
not a VAE/text-encoder flag. Do not add `--supports-fp8-compute` for this path.
