# Deploying Arc outside PyTorch

This note is for engineers porting Arc inference to other runtimes (MLX / Core ML on Mac, ONNX Runtime Mobile or ExecuTorch on
Android, CUDA / ORT CUDA / TensorRT in the cloud). The Python package `arc_audio` is the reference implementation; every port should be
checked against the golden outputs in `tests/golden/` with `scripts/check_parity.py`.

## Backbones and targets

| backbone | `--backbone` (aliases) | what runs | target |
|---|---|---|---|
| **Arc Mini** | `mini` (`edge`) | streaming speech encoder + transcript decoder + Arc (all 4-bit packed) | Mac, Android, browsers, small cloud |
| **Arc Transcriber** (full) | `full` (`parakeet`, `server`) | NVIDIA Parakeet-TDT 0.6B v2 encoder + adapters + transcript decoder + Arc | cloud GPU (CUDA fp16, ORT CUDA, TensorRT), Mac (MLX) for files |
| Arc Transcriber Edge | (coming) | a 4-bit version of the full model | Android / Mac / cloud |

Golden reference outputs: `tests/golden/mini.json` (Arc Mini) and `tests/golden/full.json` (Arc Transcriber) match the published
weight files; check every port against them.

Budgets for Arc Mini on device: **download <= 50 MB**, **resident RAM < 200 MB**, real time on 2 mobile big cores.
Today (PyTorch reference, weights unpacked to float32): download 71 MB, RAM about 1.0 GB (files) / 1.6 GB (live). The gap is closed by
the items in "Reducing RAM" below; none of them needs new weights.

## Weight files

See `docs/WEIGHTS_FORMAT.md` for the exact byte layout. Summary: safetensors; each 4-bit matrix `W` is stored as `W.q` (uint8, two codes
per byte, low nibble first), `W.s` (float32 scale per group of 32) and `W.z` (uint8, two 4-bit zero points per byte), with the layout in
the header (`int4_layout`). Dequantised value = `(code - zero) * scale` in float32 (bit exact).

Mapping to runtime formats:

- **ONNX Runtime `MatMulNBits`** (com.microsoft, bits=4, block_size=32): `B` = our codes regrouped as `[N, K/32, 16]` uint8 with the same
  low-nibble-first order, `scales` = `W.s` (float32 or cast to float16), `zero_points` = `W.z` (packed uint8). Our groups run along the
  input (reduction) dimension of each output row, which is MatMulNBits' layout (N = output rows, K = inputs). The dequantisation formula
  is identical, so ORT reproduces our weights exactly when scales stay float32. Embedding tables: `GatherBlockQuantized` with the same
  blocks. Convolutions: the weight is flattened `[out, in * k]` (im2col order); run them as MatMulNBits after im2col or dequantise once at
  load. The two tensors with `axis: -2` (`trunk.adapt_down`, `trunk.adapt_up`) are batched per-domain matrices: transpose to put the
  grouped axis last before packing.
- **MLX affine 4-bit** (`mx.quantize`, group_size 32): MLX stores `w = scale * q + bias`. Use `scale = s` and `bias = -zero * s`. This is
  algebraically identical; in float16 it is not bit exact (rounding of `bias`), so expect tiny differences. MLX packs 8 codes per uint32
  little-endian, low bits first: repack pairs of our bytes in order.
- **Core ML**: `constexpr_blockwise_shift_scale` (iOS 18 / macOS 15) with int4 data, `offset = zero`, `scale = s`, block size 32 on the
  input axis.
- **ExecuTorch**: 4-bit groupwise linear (`torchao` int4 weight-only, group 32, asymmetric) matches the format; feed scales and zero points
  directly.
- **TensorRT / CUDA (full model)**: the Parakeet encoder is run in float16 (that is how the full model was evaluated); the decoder and Arc
  can use int4 weight-only GEMM with group 32 or be dequantised to float16 at load.

Tensors not on the 4-bit grid are stored as float32 / float16 and must stay in that precision: `word.tok_E` (word-channel token table,
float16, 21 MB; it is **not** the same table as the decoder embedding, so it cannot be shared) and the speech encoder's first convolution.
`heads.safetensors` / `adapters.safetensors` are float32. Do not requantise any packed tensor: the model was tuned to this exact grid.

## Pipeline and streaming state

Audio is 16 kHz mono float32. Everything advances in 100 ms frames (Arc's frame rate is 10 Hz).

| component | runs | state kept between chunks | bounded by | where to cap it |
|---|---|---|---|---|
| speech encoder (Mini) | every chunk, causal with 240 ms look-ahead | per-layer key/value windows (last 16 frames of 20 ms per layer), convolution tails, < 80 samples of PCM | fixed | already fixed |
| frame pooler | every encoder frame | the open 100 ms frame | fixed | |
| Arc network | every 100 ms frame | constant-size recurrent state per layer + per-speaker / per-segment pooled vectors | fixed | |
| Arc read-outs (`Engine`) | every frame | frame-probability history for `frames` output, open spans, speaker track bank, pending name / number reads | grows with session length | keep only the last N seconds of frame history (needed: about 20 s for pending reads); drop `frames` output in production |
| word store | when transcript words arrive | final and interim words | grows | keep the last 60 s of words (names wait at most ~17 s for words) |
| utterance segmenter | every frame | energy history (300 frames) + open utterance (<= 15 s) | fixed | |
| transcript decoder | once per closed utterance (<= 15 s), plus partials | encoder states of that utterance, beam-4 KV cache | per utterance | partials: 1 beam, every 1 s, last 7 s only |
| context prompt | per utterance | up to 96 prompt tokens, 64 earlier utterances for `auto_prev` (only 40 words used) | fixed | |
| event head | every frame, 300 ms look-ahead | last 3 frames of features | fixed | |
| device-directed | per utterance | features of the open utterance | fixed | |
| Parakeet features (full) | every chunk of 10 s (live) / 30 s (files) with 2 s / 4 s context | audio of the current chunk + context, projected 80 ms frames | grows (projected frames) | keep the last 2 s of projected frames |
| stream audio buffer | every chunk | last 30 s of PCM | fixed | 15 s + pad is enough when partials are off |

## Reducing RAM (Mini, no accuracy change)

1. **Native 4-bit runtime, no PyTorch.** Keep weights packed and use 4-bit GEMM kernels (MatMulNBits / MLX / Core ML). Packed weights
   are about 45 MB + 21 MB float16 table; PyTorch alone costs several hundred MB.
2. **mmap the weight files** instead of reading them into heap memory.
3. **Rolling windows** for every growing structure in the table above (frame history, word store, utterance list, projected frames).
4. **16-bit activations** (float16 / bfloat16) for the encoder, decoder and Arc network; keep the 4-bit weights and float32 scales.
   Verify with the golden checks (transcripts identical, probabilities within tolerance).
5. **Beam search**: share the encoder cross-attention keys/values across beams and keep one KV cache per beam prefix; a low-power
   mode with 2 beams (or greedy) for finals cuts decoder memory and time (small accuracy cost; measure on your data).
6. **Lazy optional heads**: load the event head (6.7 MB) and device-directed head only when requested (`events=None` skips them);
   skip partial transcripts and word alignment for partials when not displayed.
7. Keep only the tokenizer pieces needed at run time (vocabulary and special tokens); the tokenizer JSON can be converted to a compact
   binary form.

## Verifying a port

```bash
python scripts/check_parity.py --backbone mini --candidate my_port_output.json          # compare with tests/golden/mini.json
pytest -q tests/test_golden.py                                                          # the reference itself (PyTorch)
```

The golden files hold, for the bundled example clip: the full `predict()` JSON (default settings), and per-stage checks: pooled
encoder features (shape, mean, standard deviation, first values, sha256 of the float16 bytes), per-read-out frame probabilities
(shape, mean, first values), transcript decoder output text and tag spans per utterance. A port should reproduce the transcript text
and tags exactly and the numeric stages within the tolerances in `scripts/check_parity.py`.
