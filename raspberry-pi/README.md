# Arc Mini on a Raspberry Pi

Arc Mini through libarc ([`c/`](../c/README.md)) on a Raspberry Pi 4 or 5 (64-bit Raspberry Pi OS or Ubuntu): one
header, a static library, no Python. Offline files and live streams (Arc's transcript, heads only, or another
recogniser's words).

## Install

```bash
git clone https://github.com/WhissleAI/arc_transcriber && cd arc_transcriber
raspberry-pi/install.sh        # tuned build, unit tests, install to ~/.local, fetch Arc Mini
```

By hand: `make -C c PI=5` (`PI=4` for a Pi 4), then
`hf download WhissleAI/arc-v1.1 --include "mini/*" --local-dir models` (gated: accept the terms on Hugging Face and
log in first). Cross-compile from x86-64: `make -C c CC=aarch64-linux-gnu-gcc CPU=cortex-a76`.

## Use

```c
#include "arc.h"
arc_engine* e = arc_create_ex("arc-v1.1/mini", 2, ARC_NO_DECODER, err, sizeof err);   /* heads only */
arc_stream_config c = {.mode = ARC_STREAM_HEADS, .domain = "assistant", .on_output = on_output, .events = 1};
arc_stream* s = arc_stream_open(e, &c);
arc_stream_push(s, pcm, n);          /* 16 kHz mono float, as it arrives */
arc_stream_finish(s); arc_stream_close(s); arc_destroy(e);
```

| example | does |
|---|---|
| [`examples/hello.c`](examples/hello.c) | transcribes a WAV |
| [`examples/stream.c`](examples/stream.c) | streams from stdin: `arecord -q -f S16_LE -r 16000 -c 1 -t raw \| ./stream arc-v1.1/mini --heads` |

Build either with `cc X.c -I c/include c/build/libarc.a -lm -pthread`. Modes and records:
[c/README.md](../c/README.md#streaming-arc_stream). `bench.sh` reproduces the table below.

## Measured on a Raspberry Pi 5

Pi 5 (4 GB), Debian 13, GCC 14, **no cooler**; libarc from arc_transcriber 1.2.1 with `WhissleAI/arc-v1.1` v1.2.0 `mini/`
weights. 40 LibriSpeech test-clean clips (276 s), 100 ms chunks. Latencies are p50 / p90 ms after the end of speech.

| stream | threads | RTF | peak RSS | final | settled tags |
|---|---|---|---|---|---|
| heads only | 1 | 0.26 | 66 MB | 322 / 422 (endpoint) | 1022 / 1222 |
| heads only | 4 | 0.13 | 64 MB | 313 / 413 | 1009 / 1209 |
| full, partials every 1 s | 4 | 0.37 | 136 MB | 689 / 1762 | 1010 / 1209 |
| full, no partials | 4 | 0.24 | 129 MB | 712 / 1720 | 1009 / 1209 |

- Settled tags come about 1 s after speech: the 0.3 s read-out delay plus the 0.5 s pause that closes an utterance.
- In full mode the final decode runs inside the push that closes the utterance (up to 2-3 s): push from a worker
  thread. Heads-only pushes take under 35 ms.
- WER (full stream, 4 threads, Whisper's English normaliser): 6.7 % on 174 of the 200 test-clean clips.
- **Use a cooler** for sustained work: without one the Pi 5 reaches 82-89 °C within minutes at 2-4 threads and
  throttles. `bench.sh` prints temperature and throttling per row.
