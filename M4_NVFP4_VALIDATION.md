# M4 NVFP4 validation — 2026-09-07

Apple M4 Max, 128 GiB, macOS 26.4, PyTorch 2.14.0, ComfyUI 0.34.5.
Baseline: `09c9cc8`, installed `0.2.33+chargeuk.mps2` plus AppleSilicon-FP8 1.3.2.
Candidate: `0.2.33+chargeuk.mps3`, same ComfyUI and custom node.

## Implementation and correctness

The new MPS `dequantize_nvfp4` reads E2M1 packed weights and directly addresses
swizzled E4M3 block scales. One Metal thread decodes a 16-weight block, sharing
the scale across eight packed bytes. Tensor/block scale rounding, both nibble
orders, signed zeros and output FP16/BF16/FP32 match CPU eager decoding.
Scale buffers are read each call; no decoded-weight cache is introduced.

- Final full suite: **850 passed, 1608 hardware-specific skipped**, one existing
  RMSNorm dtype warning.
- Focused MPS/INT8 regression suite: **228 passed**.
- NVFP4 suite: **103 passed**, including all 16 E2M1 values against all 256 E4M3
  scale bytes, special global scales, tile boundaries, strided/expanded inputs,
  scale mutations, CPU/MPS scale buffers, fallback, compilation/launch failures
  and propagation of out-of-memory errors.
- Both real checkpoint layers below decode exactly to CPU-reference values.
  Their MPS linear outputs are also exact; the down projection includes its
  stored AWQ input-smoothing vector.

## Real-layer paired measurements

Checkpoint: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, 15,687,142,551 bytes.
It contains 350 NVFP4 linears, an INT8 embedding and BF16 vision/norm parameters.
Its metadata requests floating-point matrix multiplication, with compressed
weight storage. AWQ preprocessing remains owned by ComfyUI.
The requested model was copied from the user's mounted share into ComfyUI's local
`models/text_encoders` directory. Rsync completed successfully; the complete
file size and safetensors header match the source. The source was not modified.

BF16 output, 20 alternating AB/BA pairs per layer, synchronized GPU timings:

| Decoded weight shape | Existing CPU detour | Native MPS | Decode ratio |
|---|---:|---:|---:|
| K projection: 1024 × 5120 | 3.3942 ms | 0.3542 ms | 9.58× |
| Down projection: 5120 × 25600 | 78.1187 ms | 2.5709 ms | 30.39× |

All pairs recorded nominal thermal pressure; none crossed the 20% drift flag.
File reads, reference checking and compilation are outside paired timings.
These are decoding measurements, **not full-model or video-generation speedups**.

A separate GPU-only prototype comparison favored one thread per 16-weight block
over one per packed pair: 0.3806 → 0.2773 ms on K, 1.2160 → 1.1592 ms on down.
Both variants matched CPU results. Its absolute timings are not interchangeable
with the CPU/native comparison above: execution order, cache state and integration
overhead differ. The production implementation keeps only the block variant.

## Complete text encoder

Native ComfyUI `CLIPType.MINIMAX`, BF16 compute, fixed nine-token prompt:
`A red sailboat on a peaceful lake.` Each process loads the local checkpoint,
runs one cold encode and two warm encodes. Process order was baseline, candidate,
candidate, baseline (ABBA); no GPU benchmarks ran concurrently.

| Warm encode | Four samples (seconds) | Median |
|---|---|---:|
| Existing CPU NVFP4 decode | 20.7149, 20.9928, 20.0696, 20.0998 | 20.4073 s |
| Native Metal NVFP4 decode | 0.7183, 0.7092, 0.7060, 0.7001 | 0.7076 s |

That is **28.84× faster warm text encoding for this prompt and checkpoint**.
All saved conditioning tensors and token tags are bit-for-bit identical across
the four processes. Every candidate encode completed all **350 NVFP4 Metal
decodes**, with no native-decoder failures or activation quantization. The
existing INT8 embedding path still ran once per encode. This rules out a skipped
encoder or cached-conditioning result as the cause of the improvement.

All observed thermal states were nominal. Cold encodes were 32.91/22.34 s for
baseline and 2.97/3.06 s for candidate, but disk/OS caches and initialization
make those unsuitable for the headline ratio. Warm timing excludes model loading.
Both builds retained packed weights and reported 14.71 GiB live MPS allocations
after encoding; this is not a peak-memory measurement. These results do not
measure MiniMax video generation, vision input, long context, or model quality
against an unquantized checkpoint.

A separate 42-token descriptive prompt was checked with two warm samples per
build. Baseline: 18.6280/18.9538 s; installed mps3: 0.6633/0.6536 s. Medians
were **18.7909 → 0.6585 s (28.54×)**, with exact conditioning and token tags,
350 successful Metal decodes per candidate encode, and nominal thermal pressure.
This longer-prompt check was baseline then candidate, not a second ABBA experiment.

## Installation validation

The CPU/MPS wheel was built with `setup.py bdist_wheel --no-cuda` and installed
without changing dependencies. `pip check` passed. All 46 installed Kitchen
Python files byte-match the source checkout. Normal ComfyUI startup reports
`0.2.33+chargeuk.mps3` and the MPS backend's NVFP4 capability. With the existing
extra-model-path configuration loaded, the model name resolves to the new local
copy rather than the network share. The temporary startup-validation server was
stopped after checking it; the normal launcher is unchanged.

## Scope and reproduction

`benchmarks/benchmark_nvfp4_model.py --mode layer --model-path MODEL --output OUT`
uses the installed Kitchen. Add `--kitchen-source PATH` for the candidate and
`--layer model.layers.0.mlp.down_proj --tokens 32` for the larger AWQ layer.
Layer mode reads selected tensors only; no downloads occur.

For full text encoding use `--mode encode` and a complete local checkpoint.
The harness checks RAM and packed-weight retention, separates cold/warm runs,
records dispatch and thermal state, and saves raw outputs. Use
`--compare BASELINE_DIR CANDIDATE_DIR` to compare those outputs on CPU.

Thermal pressure is not temperature/clock telemetry. Nominal does not prove
identical clocks or an otherwise idle laptop; desktop/video activity existed.
Memory snapshots are not peak-memory measurements. Accuracy comparisons use
the same quantized checkpoint, not an unavailable original BF16 model.

No native FP4 matrix arithmetic, fused NVFP4 GEMM, activation quantization,
full-model decoded cache, MXFP8 enhancement or new INT8 speedup is claimed.
Existing overflow-safe INT8 behavior and M5-only cooperative-kernel gates remain.
No core ComfyUI or additional custom-node changes are required for this stage.

The original Metal implementation uses the existing local FP8 decoder and
Kitchen's documented blocked-scale layout. Related reference implementations:
[MLX Metal quantized kernels](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fp_quantized.h)
and [NVFP4 format description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).

## Rollback

Reinstall the saved `comfy_kitchen-0.2.33+chargeuk.mps2-py3-none-any.whl` with
`python -m pip install --no-deps --force-reinstall PATH_TO_WHEEL`, then restart
ComfyUI. The user's previous wheel is in the ComfyUI installation's
`rollback/mps2/` directory. The existing AppleSilicon-FP8 CPU NVFP4 compatibility
path remains available when the new MPS backend is disabled.
