# Apple MPS INT8 branch

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
as `0.2.33+chargeuk.mps1`. Install the tested commit after ComfyUI's requirements
so ComfyUI's PyPI pin does not replace it.
