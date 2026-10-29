# Arc packed weight format (`*.q4.safetensors`)

This format is used by model revisions `v1.1.1`, `v1.1.2` and `v1.2.0` (Arc Mini 1.1.1, Arc Transcriber 1.1.1 and 1.2.0).

Files: `mini/model.q4.safetensors` (Arc), `mini/moonshine-streaming-tiny/model.q4.safetensors` (speech encoder + transcript decoder),
`full/model.q4.safetensors` (Arc, same bytes as mini), `full/decoder/model.q4.safetensors` (transcript decoder only).
Plain (unpacked) files: `mini/heads.safetensors`, `full/adapters.safetensors` (float32).

## Container
Standard safetensors (8-byte little-endian header length, JSON header, raw little-endian tensor data; each tensor's data is a
contiguous byte range given by `data_offsets`, so a reader can mmap the file). Header `__metadata__`:

- `format`: `"pt"`
- `group_size`: `"32"`
- `int4_layout`: a JSON string `{ "<tensor name>": {"shape": [...], "axis": -1 | -2, "rc": [R, C, pad], "dtype": "float32" | "float16"} }`

## Packed tensor `W` (every name listed in `int4_layout`)
Three entries replace the original tensor `W`:

| entry | dtype | shape | content |
|---|---|---|---|
| `W.q` | U8 | `[n * 32 / 2]` | 4-bit codes, two per byte |
| `W.s` | F32 | `[n]` | scale per group |
| `W.z` | U8 | `[ceil(n / 2)]` | 4-bit zero point per group, two per byte |

with `n = R * (C + pad) / 32` groups.

- **Nibble order:** element `2i` is the LOW nibble of byte `i` (`b & 0x0F`), element `2i+1` the HIGH nibble (`b >> 4`). The same order
  holds for `W.z` (an odd group count leaves the last high nibble 0).
- **Groups:** the tensor is viewed as a matrix `[R, C]` and each row is zero-padded on the right to `C + pad` (a multiple of 32).
  Group `g` covers row `g / ((C+pad)/32)`, columns `32 * (g mod ((C+pad)/32))` ... `+31`. Codes are in row-major order over that
  padded matrix: code index `k = r * (C + pad) + c`.
  - `axis == -1` (all Linear / Conv / Embedding weights): `R = shape[0]`, `C = prod(shape[1:])` (a conv weight `[out, in, k]` is
    flattened to `[out, in * k]`, row-major).
  - `axis == -2` (only `trunk.adapt_down` `[16, 512, 64]` and `trunk.adapt_up` `[16, 64, 512]`): the reduction axis -2 is moved last
    (`[d, a, b] -> [d, b, a]`), then `R = d * b`, `C = a`. After dequantising, move the axis back (`[d, b, a] -> [d, a, b]`).
- **Dequantisation (bit-exact):** `w[r, c] = ((float)code - (float)zp[g]) * s[g]` computed in IEEE float32 (one subtraction of two
  small integers, exact, then one float32 multiply), then drop the pad columns and reshape to `shape`. If `dtype` is `float16`, round
  the float32 result to float16 (round-to-nearest-even). This reproduces the evaluated weights bit for bit.
- Tensors not listed in `int4_layout` are stored as is (F32 or F16) under their own names (biases, norms, small tensors, and
  `word.tok_E` [32768, 320] F16 and `model.encoder.embedder.conv1.weight`, which are not on the 4-bit grid).

## Tied weights
`moonshine-streaming-tiny/config.json` has `tie_word_embeddings: true`: the decoder output projection `proj_out.weight` is not stored;
it equals `model.decoder.embed_tokens.weight` (packed).

## Reference implementation
`arc_audio/weights.py` (`load_q4`) (unpacked tensors are
byte-equal to the evaluated float32 / float16 weights).
