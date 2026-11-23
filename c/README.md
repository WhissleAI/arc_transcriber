# libarc: Arc in C

A dependency-free C11 library (libc, libm, threads) that runs **Arc Mini** from the published release folder
(`WhissleAI/arc-v1.1`, `mini/`) with no Python, PyTorch or ONNX. One header: [`include/arc.h`](include/arc.h).

- Offline: transcript with inline tags and word times, linked names, numbers with values, utterance tags
  (device-directed, intent, act), speakers, sound events, per-frame probabilities.
- Streaming in three modes: Arc's own transcript, heads only, or another recogniser's words.
- **Arc Transcriber Space** (`nft2`) too, when built with ONNX Runtime (optional).
- Not supported: Arc Transcriber (full, Parakeet backbone). See [docs/LIBARC.md](../docs/LIBARC.md#not-ported-yet).
- Checked against `arc_audio`: identical transcripts, tags, names and numbers on the golden clip and 609 clip runs.

Library version 1.5.0, ABI 2 (`ARC_VERSION`, `ARC_ABI_VERSION`; compare with `arc_version()` / `arc_abi_version()`).

## Build

```bash
make -C c                                    # build/libarc.a + tools (arc-transcribe, arc-score, arc-parity, arc-trim, arc-stream)
make -C c test                               # unit tests, no weights needed (CI runs these on Linux and macOS)
make -C c parity MINI=/path/arc-v1.1/mini    # golden parity: prints PARITY OK
make -C c ARC_ORT=/path/to/onnxruntime       # also Arc Transcriber Space (c/build/nft2/)
```

Get the weights with `hf download WhissleAI/arc-v1.1 --include "mini/*" --local-dir arc-v1.1` (gated: accept the
terms on Hugging Face first).

## Offline

```c
#include "arc.h"
char err[512];
arc_engine* e = arc_create("arc-v1.1/mini", 4, err, sizeof err);
arc_name_entry kb[] = {{"Theo Brandt", "person", "Harbor City Hawks", NULL, 0},
                       {"Harbor City Hawks", "team", "Harbor City Hawks", NULL, 0}};
arc_set_names(e, kb, 2);                                /* link to this list; also biases the decoder */
arc_options o = {.domain = "nba", .trunk = 1, .events = 1, .bias_names = 1, .names = 1, .numbers = 1};
arc_result* r = arc_transcribe(e, pcm, n, &o);          /* 16 kHz mono float, any length */
/* r->utts[i]: text, tags, words, intent, act, device_directed;  r->names[i]: heard, entity, state
   r->numbers[i]: heard, kind, value, value2, unit;  r->events[i]: sound / channel events */
arc_result_free(r);
arc_destroy(e);
```

```bash
build/arc-transcribe arc-v1.1/mini clip.wav --domain nba --names examples/example_names.json [--json] [--lm car.arclm --lm-weight 0.3]
```

Other calls: `arc_set_context` (names with kinds, terms, topic, earlier text), `arc_set_bias`, `arc_set_lm` (n-gram
LM fusion, [docs/LIBARC.md](../docs/LIBARC.md#n-gram-lm)), `arc_frames_compute`,
`arc_memory*` (carry state across calls), `arc_speakers` / `arc_diarization`, `arc_number_date`, and `arc_score`
(log P(text | audio) for candidate texts, for catalogs and commands; `build/arc-score`). Results own their memory. One
engine runs one call at a time: use one engine per thread.

## Streaming: `arc_stream`

Live audio in any chunk size; records out through one callback. Nothing is re-encoded.

| mode | loads | gives |
|---|---|---|
| `ARC_STREAM_FULL` | everything | PARTIAL and FINAL transcript records (finals identical to `arc_transcribe`), tags |
| `ARC_STREAM_HEADS` | no decoder (`ARC_NO_DECODER`) | utterances (times, device-directed, intent, act), events, frames; no text |
| `ARC_STREAM_EXTERNAL` | no decoder | the same, with another recogniser's words grouped into Arc's utterances |

```c
arc_engine* e = arc_create_ex("arc-v1.1/mini", 2, ARC_NO_DECODER, err, sizeof err);
arc_stream_config c = {.mode = ARC_STREAM_EXTERNAL, .domain = "assistant", .on_output = on_output, .frames = 1};
arc_stream* s = arc_stream_open(e, &c);
arc_stream_push(s, pcm, n);                         /* as audio arrives */
arc_ext_word w[] = {{"turn", 0.42f, 0.61f}, {"it", 0.61f, 0.70f}, {"up", 0.70f, 0.95f}};
arc_stream_words(s, w, 3, 1);                       /* the recogniser's final words, in stream time */
arc_stream_covered(s, 0.95f);                       /* its finals reach 0.95 s */
arc_stream_finish(s);
arc_stream_close(s);
```

| record | when |
|---|---|
| `PARTIAL` | text so far (FULL mode) |
| `ENDPOINT` | Arc's segmenter closed an utterance (EXTERNAL: ask the recogniser for its finals now) |
| `FINAL` | the utterance: text, words, inline tags, read-out |
| `TAGS` | provisional at the first non-speech frame; settled 0.5 s later |
| `EVENT`, `FRAME` | sound events; every 100 ms frame's probabilities |

Timing: a frame is final about 0.1 s after the read-out delay (`delay_ms`, default 300). A FINAL comes after the
segmenter's 0.4 s pause. In FULL mode the final decode runs inside the push that closes the utterance: push from a
worker thread, not the audio callback. `build/arc-stream MINI file.wav [--mode heads|external]` reports RTF, CPU,
peak RSS and latency per record type.

## Arc Transcriber Space: `arc_nft2_*`

ONNX Runtime 1.23 (CPU) runs the encoder and RNNT networks; libarc does everything else. Without `ARC_ORT`,
`arc_nft2_create` fails with a message.

```bash
make -C c ARC_ORT=/path/to/onnxruntime-linux-x64-1.23.2
c/build/nft2/arc-nft2 arc-transcriber-space/nft2 question.wav --names moon_names.json
```

```c
arc_nft2* m = arc_nft2_create("arc-transcriber-space/nft2", 4, err, sizeof err);
arc_name_entry kb[] = {{.name = "Tycho", .kind = "lunar_feature"}};
arc_nft2_set_names(m, kb, 1);              /* biasing, decoder prompt and linking */
arc_result* r = arc_nft2_transcribe(m, pcm, n, NULL);
arc_result_free(r);
arc_nft2_destroy(m);
```

Streams (`arc_nft2_stream_*`) add live events, echo cancelling and Arc Tagger output; see
[docs/LIBARC.md](../docs/LIBARC.md#arc-transcriber-space-reference) and [`examples/nft2_app.c`](examples/nft2_app.c).

## Speed and memory

| | Arc Mini (Apple M3 Pro, 4 threads) |
|---|---|
| real-time factor, transcript only / everything | 0.026 / 0.046 |
| load | 30-90 ms (weights memory-mapped) |
| peak memory, 30 s clip with everything | 180 MB (grows with clip length: about 1 GB for 8 minutes) |

Raspberry Pi numbers: [raspberry-pi/](../raspberry-pi/README.md). Arc Transcriber Space and full check results:
[docs/LIBARC.md](../docs/LIBARC.md).

## Layout

| path | what |
|---|---|
| `include/arc.h` | public API |
| `src/arc_api.c`, `arc_engine.*`, `arc_link.*` | engine, transcript, heads and read-outs, names and numbers |
| `src/arc_moon.*`, `arc_tok.*`, `arc_align.*` | Moonshine encoder / decoder, tokenizer, word times |
| `src/arc_lm.*` | n-gram LM (ARCLM1 loader, shallow fusion); libc only |
| `src/arc_nft2*` | Arc Transcriber Space; `arc_nft2_be.h` is its encoder backend interface |
| `src/arc_store.*`, `arc_kernels.c`, `arc_weights.c` | weights, 4-bit kernels (AVX2 / NEON), thread pool |
| `tools/` | CLI tools and the reference dumpers / comparers used for parity |
| `tests/` | unit tests and fixtures |

## Licence

Part of arc_transcriber: CC BY-NC 4.0 plus the Whissle terms in [LICENSE](../LICENSE). Moonshine-streaming-tiny inside
`mini/` is MIT; the metaphone in `src/arc_ntext.c` follows jellyfish (MIT, see [NOTICE](../NOTICE)).
