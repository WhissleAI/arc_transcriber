# NVIDIA NIM

**Status: GPU path available via the Python package; NIM packaging not yet validated.**

Arc Transcriber (full, Parakeet-TDT 0.6B v2 encoder) runs on NVIDIA GPUs through PyTorch CUDA.
[`Dockerfile.cuda`](../../Dockerfile.cuda) serves it with `python -m arc_transcriber.serve`: NIM-style health routes
(`/v1/health/live`, `/v1/health/ready`), `/v1/models`, the OpenAI-compatible transcription route, one port, non-root,
environment configuration, and no network at run time when the weights are bundled. How to build it:
[deploy/README.md](../../README.md#gpu-image-nvidia-cuda). The image is **untested on a GPU**.

Only the C library (libarc) and its `arc-serve` image are CPU-only. Arc Mini also runs on CPU on-device and on a GPU
through the Python package, but it is small enough that a GPU rarely pays off.

A listing still needs:

1. A run on an NVIDIA GPU (for example AWS g6 / L4 or g5 / A10G): readiness, a transcript matching the CPU run on
   `examples/example_synth.wav`, latency and memory.
2. NVIDIA base images, NGC distribution, NVIDIA's scanning and partner process.
3. Whissle's commercial terms (Arc is CC BY-NC 4.0 today).
