# NVIDIA NIM

**Status: blocked, CPU-only.** A NIM is a GPU-accelerated container. libarc has AVX2 / NEON kernels and no GPU code,
and Arc Transcriber Space's optional backend uses ONNX Runtime's CPU provider only. A NIM of today's image would not
use the GPU.

Already NIM-shaped: `GET /v1/health/live`, `/v1/health/ready`, `/v1/models`, an OpenAI-compatible transcription
route, environment configuration, one port, non-root, no network at run time.

A listing would need:

1. **A GPU path** for one Arc model. Shortest first:
   - Arc Transcriber Space: its encoder and RNNT already run through ONNX Runtime behind `c/src/arc_nft2_be.h`;
     switch that backend to the CUDA or TensorRT execution provider.
   - Arc Transcriber (full): the Parakeet encoder port described in [docs/LIBARC.md](../../../docs/LIBARC.md#not-ported-yet),
     with a CUDA encoder.
2. NVIDIA base images, NGC distribution, NVIDIA's scanning and partner process.
3. Whissle's commercial terms (Arc is CC BY-NC 4.0 today).

Arc Mini is too small for a GPU to pay off (RTF about 0.05 on 4 CPU threads); the NIM case is the full model.
