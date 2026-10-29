# Arc C API (coming in 1.1.x)

A C11 inference library (`libarc`, one header `arc.h`) for both Arc backbones will live in this folder:

- **CPU build**: no dependencies beyond libc, libm and threads; edge and server backbones.
- **CUDA build** (`-DARC_CUDA=ON`): GPU inference, server (Parakeet) backbone first, then edge; several live streams per engine.

It reads the same packed weights as the Python package (see `docs/WEIGHTS_FORMAT.md`) and returns the same output schema.
Until then, use the Python package (`arc_audio`).
