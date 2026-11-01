# Arc Mini on a Raspberry Pi

Arc Mini through the dependency-free C library ([`c/`](../c/README.md)) on a Raspberry Pi 4 or 5 (64-bit Raspberry Pi OS
or Ubuntu): one header, a static library, no Python. Offline transcription, and live streams in three modes: Arc's own
transcript, heads only (tags and events without loading the transcript decoder), or another recogniser's words merged
with Arc's tags.

## Install

```bash
git clone https://github.com/WhissleAI/arc_transcriber && cd arc_transcriber
raspberry-pi/install.sh        # tuned build for this board, unit tests, install to ~/.local, fetch Arc Mini
```

By hand: `make -C c PI=5` (`PI=4` for a Pi 4), then `hf download WhissleAI/arc-v1.1 --include "mini/*" --local-dir models`
(the model repo is gated: accept its terms on Hugging Face and log in first). Cross-compiling from x86-64:
`make -C c CC=aarch64-linux-gnu-gcc CPU=cortex-a76`.

## C API

```c
#include "arc.h"

char err[512];
arc_engine* e = arc_create_ex("arc-v1.1/mini", 2, ARC_NO_DECODER, err, sizeof err);   /* heads only */
arc_stream_config c = {.mode = ARC_STREAM_HEADS, .domain = "assistant", .on_output = on_output, .events = 1};
arc_stream* s = arc_stream_open(e, &c);
arc_stream_push(s, pcm, n);          /* 16 kHz mono float, as it arrives; records come to on_output */
arc_stream_finish(s);
arc_stream_close(s);
arc_destroy(e);
```

[`examples/hello.c`](examples/hello.c) transcribes a WAV; [`examples/stream.c`](examples/stream.c) streams from stdin
(`arecord -q -f S16_LE -r 16000 -c 1 -t raw | ./stream arc-v1.1/mini --heads`). Both build with
`cc X.c -I c/include c/build/libarc.a -lm -pthread`. Modes, records and timing: [c/README.md](../c/README.md#streaming-arc_stream).

## Measured on a Raspberry Pi 5

Preliminary (work in progress). Pi 5 (4 GB), Debian 13, GCC 14, wall supply, **no cooler**; libarc from this repo
(arc_transcriber 1.2.1) with the `mini/` weights of `WhissleAI/arc-v1.1` v1.2.0. `bench.sh` reproduces the stream rows.

| | 1 thread | 4 threads |
|---|---|---|
| Offline, everything (transcript, trunk, events, word times, names, numbers), RTF: before → now (identical output) | 0.429 → 0.286 | 0.199 → 0.131 |
| Stream, heads only (no decoder): RTF, peak RSS | 0.26, 66 MB | |
| Stream, full (partials every 0.4 s): RTF, peak RSS | | 0.47, 126 MB |

**A Pi 5 needs a cooler** for sustained use: without one it reaches 83–85 °C and throttles in back-to-back runs.
`bench.sh` prints the temperature and throttling flags for every row.
