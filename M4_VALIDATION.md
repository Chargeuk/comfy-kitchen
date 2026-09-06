# M4 validation — 2026-09-06

Apple M4 Max, 128 GiB, macOS 26.4, PyTorch 2.14.0, ComfyUI 0.34.5.
Baseline: `957bf68`, package `0.2.33+chargeuk.mps1`. Candidate: `mps2`.

## Correctness

- Full suite: 746 passed, 1608 hardware-specific skipped, one existing dtype warning.
- After making rotation opt-in: focused FP8/rotation/dispatch/fallback suite 125 passed.
- Two reproduced FP16 overflow cases fixed: unscaled linear `inf` -> 130.048;
  ConvRot intermediate `inf` -> 480 (rounded to the requested output dtype).
- SAM FP16 and INT4: saved outputs exactly equal to baseline. INT4 continues
  through AppleSilicon-FP8 W4A16, 182 calls per run; no new INT4 kernel is claimed.
- SAM INT8: same top box and confidence (0.98610); top-mask IoU against FP16
  reference 0.99987. Lower-confidence mask logits differ because of FP32 math.
- Pony FP8 checkpoint: saved image and latent exactly equal to baseline.
  Inspection shows diffusion weights load into FP16 MPS parameters; this is a
  compatibility test, NOT runtime FP8-kernel coverage.
- One existing Qwen FP8 weight matrix (3072x3072) and its stored scale were read
  from the user's share. Native decode exactly matched CPU reference, including
  a scale mutation. No complete model download or copy was made.
- Synthetic ConvRot M=N=K=4096: finite output, exact agreement with explicit
  FP32 rotation/GEMM reference before final FP16 cast.
- Built wheel contains the new MPS modules; installed files match source.
  ComfyUI startup reports the MPS backend and `0.2.33+chargeuk.mps2`.

## Measurements (not universal speed claims)

Thermal-screened SAM INT8 ABBA, two warm samples from each of four processes:
baseline median 1.138624 s, candidate 1.189381 s: **4.46% slower**. This is the
cost of the safer FP32 path on this workload. All recorded native thermal states
were nominal. The synthetic image depicts a red balloon; timings include native
SAM detector work, not a video tracking benchmark.

FP8 E4M3 decode+scale to BF16, 20 alternating pairs:

| Shape | LUT median | Native median | Ratio |
|---|---:|---:|---:|
| 1024x1024 | 0.4401 ms | 0.2535 ms | 1.74x |
| 4096x4096 | 1.2894 ms | 0.4643 ms | 2.78x |

Outputs were exact; native decode avoids the 8-byte-per-element index buffer.
Thermal pressure remained nominal and no case crossed the 20% drift flag.
Pressure is not a temperature/clock measurement and cannot rule out all throttling.

Rotation results varied: early short tests favored the shader on small shapes,
but the final paired 1x256 test slightly favored dense matmul. Large shapes were
consistently unsuitable for the shader. Therefore rotation is **off by default**,
opt-in only, and limited to at most 16,384 elements.

## Limits and reproduction

No complete large-video generation, broad LoRA workflow matrix, training-gradient
support, native FP8 arithmetic, or new fused GEMM is claimed. Scale mutation is
covered; it is not equivalent to an end-to-end LoRA regression suite. No persistent
decoded-weight cache was introduced. Model files were not converted or modified.

`benchmarks/model_validation.py` runs fixed local SAM/Pony cases, saves raw
tensors and thermal records, and compares baseline/candidate output directories.
Use `--comfy PATH` and `--kitchen-source PATH` as needed. Its default checkpoint
names describe the user's existing test set; missing files fail without download.
`benchmarks/benchmark_mps.py` saves paired operation measurements to JSON.
For experimental rotation measurements explicitly set `COMFY_KITCHEN_MPS_ROTATION=1`.

## Rollback

Keep the mps1 wheel. Install it with `python -m pip install --no-deps
--force-reinstall PATH_TO_MPS1_WHEEL`, then restart ComfyUI. The companion Apple
node patch detects that the MPS module is absent and uses its original decoder.
To remove that patch too, check then reverse `integrations/applesilicon-fp8-kitchen.patch`
with `git apply --reverse --check` and `git apply --reverse` in the node checkout.
No destructive checkout/reset is needed.
