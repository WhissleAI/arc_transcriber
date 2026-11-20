# NVIDIA NIM

**Status: blocked on a GPU path.** A NIM is NVIDIA's packaging of a model as a GPU-accelerated inference container
(distributed through NGC, run on NVIDIA GPUs). libarc, which this image runs, is CPU-only: AVX2 / NEON kernels, no
CUDA, and the optional Arc Transcriber Space backend uses ONNX Runtime's CPU provider only (see `../../README.md`,
"GPU"). Packaging today's image as a NIM would ship a GPU container that does not use the GPU.

## What is already NIM-shaped

- Health routes `GET /v1/health/live` and `GET /v1/health/ready`, model listing `GET /v1/models` (OpenAI list shape).
- An OpenAI-compatible `POST /v1/audio/transcriptions`, plus Arc's fields in `verbose_json`; a realtime WebSocket.
- Configuration by environment variables, one port, non-root user, no network at run time, weights either mounted or
  bundled.

## What a NIM listing would need (to confirm with NVIDIA's partner team)

- A **GPU inference path** for at least one Arc model. Options, shortest first:
  1. **Arc Transcriber Space** (`arc_nft2_*`): its FastConformer encoder and RNNT nets already run through ONNX Runtime
     behind `c/src/arc_nft2_be.h`; switching that backend to ONNX Runtime's CUDA or TensorRT execution provider is a
     contained change (the rest of libarc stays on CPU, which is cheap: the encoder dominates the time).
  2. **Arc Transcriber (full)**: the Parakeet-TDT 0.6B v2 encoder port that `c/README.md` plans ("Plan: Arc Transcriber
     (full backbone)"), with a CUDA / TensorRT encoder; arc_audio already runs this model on GPUs in Python, which is
     where the published full-model accuracy comes from.
- A container built on NVIDIA's base images with the CUDA runtime, and per-GPU optimised profiles if TensorRT engines
  are used.
- Distribution through an NGC organisation, NVIDIA's security scanning, documentation and a model card, under a
  partner agreement (NVIDIA Inception / partner network).
- Licensing compatible with NVIDIA AI Enterprise distribution: Arc is CC BY-NC 4.0 today, so this needs Whissle's
  commercial terms.

Arc Mini itself is small enough (about 0.05 real-time factor on 4 CPU threads, `c/README.md`) that a GPU would not pay
for itself; the NIM case is the full model.
