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

Pi 5 (4 GB), Debian 13, GCC 14, wall supply, **no cooler**; libarc from this repo (arc_transcriber 1.2.1) with the
`mini/` weights of `WhissleAI/arc-v1.1` v1.2.0. Streams: 40 LibriSpeech test-clean clips (276 s), 100 ms chunks,
`arc-stream` (what `bench.sh` runs), each run started below 62 °C. Latencies are p50 / p90 in ms after the end of
speech (first partial: after its start); "frame lag" is how late a FRAME record comes after the frame's end.

| | threads | RTF | CPU s per audio s | peak RSS | first partial | final | settled tags | frame lag | SoC max °C |
|---|---|---|---|---|---|---|---|---|---|
| heads only | 1 | 0.26 | 0.26 | 66 MB | – | 322 / 422 (endpoint) | 1022 / 1222 | 391 | 82 |
| heads only | 2 | 0.16 | 0.31 | 66 MB | – | 320 / 420 | 1013 / 1212 | 383 | 84 ¹ |
| heads only | 4 | 0.13 | 0.46 | 64 MB | – | 313 / 413 | 1009 / 1209 | 380 | 86 ¹ |
| full, partials every 1 s | 1 | 0.75 | 0.75 | 136 MB | 937 | 1137 / 2014 | 1022 / 1222 | 393 | 85 ¹ |
| full, partials every 1 s | 2 | 0.49 | 0.89 | 136 MB | 882 | 763 / 1847 | 1013 / 1213 | 384 | 87 ¹ |
| full, partials every 1 s | 4 | 0.37 | 1.19 | 136 MB | 856 | 689 / 1762 | 1010 / 1209 | 381 | 87 ¹ |
| full, partials every 0.4 s | 4 | 0.59 | 1.89 | 135 MB | 625 | 719 / 1804 | 1010 / 1213 | 381 | 89 ¹ |
| full, no partials | 4 | 0.24 | 0.75 | 129 MB | – | 712 / 1720 | 1009 / 1209 | 381 | 87 ¹ |
| heads only, 200 ms chunks | 1 | 0.19 | 0.19 | 67 MB | – | 431 / 532 | 1032 / 1331 | 446 | 81 |

¹ the soft temperature limit or throttling was active during the run (clock down to 1.5 GHz).

- Settled tags come ~1 s after speech: the default read-out delay (0.3 s) plus the 0.5 s pause that closes an utterance.
  Provisional tags (the same values, sent at the first frame without speech) and other read-out delays
  (`--delay-ms 0..800`): to be measured.
- In full mode a final decode runs inside the `arc_stream_push` that closes the utterance (up to 2–3 s on 1–4 threads):
  push from a worker thread, not the audio callback. Heads-only pushes take under 35 ms.
- Offline, everything (transcript, trunk, events, word times, names, numbers), RTF before → after the kernel changes
  (identical output): 0.429 → 0.286 at 1 thread, 0.199 → 0.131 at 4 (examples/example_synth.wav).
- WER (full stream, 4 threads, Whisper's English normaliser): 6.7 % on 174 of the 200 test-clean clips; the full 200, a
  noisy copy of the set and beam 1 vs 4: to be measured.

**A Pi 5 needs a cooler** for sustained use: without one it reaches 82–89 °C within a few minutes at 2–4 threads and
throttles. `bench.sh` prints the temperature and throttling flags for every row.
