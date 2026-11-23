# libarc reference

Details behind [c/README.md](../c/README.md): how the C port was checked, the Arc Transcriber Space API, the n-gram
LM, and what is not ported yet.

## How Arc Mini was checked

Every part of libarc is compared with the matching part of `arc_audio`.

| part | C source | result |
|---|---|---|
| weights (plain + packed 4-bit) | `arc_store.c` | bit-exact, 486/486 tensors |
| tokenizer (BPE, byte fallback, 768 special tokens) | `arc_tok.c` | 320/320 cases |
| Moonshine-streaming-tiny encoder | `arc_moon.c` | states within 2.4e-5 |
| decoder, beam search (4) / greedy, name bias | `arc_moon.c` | 153/153 utterances, identical tokens |
| segmentation, inline-tag read-out, context prompt | `arc_text.c`, `arc_api.c` | 170/170 transcripts |
| word times (alignment heads + DTW) | `arc_align.c` | 8,744/8,744 words to 1 ms |
| device-directed head, sound / channel events | `arc_engine.c` | 153/153 directed, 144/144 events |
| word channel | `arc_wordch.c` | within 4.5e-6 |
| tag head (intent, act, device-directed) | `arc_engine.c` | 153/153; max abs p diff 6.5e-5 |
| names (pointer head, rescorer, snap, alias / partial check) | `arc_link.c`, `arc_ntext.c`, `arc_alias.c` | identical (below) |
| numbers and dates | `arc_link.c`, `arc_num.c`, `arc_date.c` | identical (below); dates 169/169 |
| output tidying | `arc_tidy.c` | 98/98 |
| speakers (slot machine, TrackBank diarization, roles) | `arc_speaker.c` | 16/16 unit checks; server parity |

- **Golden parity**: `make -C c parity MINI=/path/arc-v1.1/mini` checks `tests/golden/mini.json`. Transcript, inline
  tags, linked names and number values are identical; probabilities are within tolerance. Passes on x86-64 and arm64.
- **Names, numbers and word times**: `tools/dump_names.py` (arc_audio) vs `build/arc-transcribe --json`, compared by
  `tools/compare_names.py`: 609 clip runs, 8,744 words, 756 names, 443 numbers, 0 differences. Clips cover the example,
  synthetic passages and voice commands, book requests, long recordings and an 8-minute file, across 13 domains and a
  misspelt list.
- **Unit tests without weights** (`make -C c test`, CI): numbers 3215/3215, name text 7849/7849, tidy 98/98, dates
  169/169, aliases 606/606, speakers 16/16, n-gram LM 48/48.
- **Scores and n-best**: 56/56 `arc_score` values within 2e-2 nats, 5/5 n-best lists identical.
- Precision: kernels sum in float32 in a different order than PyTorch. Decisions (tokens, tags, links, values) matched
  everywhere tested; probabilities agree to about 1e-4.

Reference data for the checks that need weights is made by `tools/dump_*.py` from clips of your own.

## Arc Transcriber Space reference

`arc_nft2_*` runs `Arc.predict(wav, transcript="nft2", ...)` in C. ONNX Runtime runs only the networks in
`nft2/encoder_onnx/` (encoder, RNNT decoder and joint) behind `src/arc_nft2_be.h`, so a native backend can replace it
later. libarc does the log-mel front end, the streaming loop, greedy RNNT with names biasing, the Arc decoder,
tagging and linking.

| call / field | what |
|---|---|
| `arc_nft2_set_context` | `dom` (default `"space"`), `prev` / `auto_prev`, `kb` with kinds, session memory |
| `arc_nft2_options` | `domain` (default `"assistant"`), `no_bias`, `no_events`, `no_guard`, `almost`, `nbest`, `live`, `playback_threshold` (0.7), `playback_drop`, `echo_decode` |
| `arc_nft2_stream_open / feed / flush / close` | `feed` returns each utterance once closed (0.4 s pause); `flush` returns the whole stream |
| `arc_nft2_stream_events` | with `options.live`: `ARC_EV_PARTIAL`, `ARC_EV_ENTITY`, `ARC_EV_EOT`, `ARC_EV_BARGE_IN`, `ARC_EV_FINAL` |
| `arc_nft2_stream_set_names`, `arc_nft2_stream_end_turn` | swap the list mid-stream; close the open utterance now |
| `arc_nft2_stream_feed_ref` | the device's playback, for echo cancelling (512 ms echo path) and `playback_leak` |
| `arc_nft2_set_tagger` | Arc Tagger on each final (`make ARC_WITH_ARCTAG=/path/to/arc-tagger-c`); `arc_utt.tagger_json` |
| `arc_word` | `conf`, `alts` / `alt_p`, `disfl` |
| `arc_utt` | `conf`, `clean_text`, `quantities`, `n_dropped`, `alts` / `alt_scores`, `playback_leak` / `playback`, `tagger_json` |

Clips up to 20 s are one utterance; longer ones are cut by the segmenter. Streams pad each utterance with silence so
the RNNT emits its last word. Names lists follow [NAMES_LISTS.md](NAMES_LISTS.md).

**Checks** (`tools/dump_nft2.py` vs `build/nft2/arc-nft2 --dump`, compared by `tools/compare_nft2.py`; 23 clips,
156 s): with the same log-mel frames, all 23 clips match on every discrete output, offline and streamed. With
PyTorch's own front end, 14/23 offline and 17/23 streamed are identical; the rest move a word boundary by one 80 ms
frame (one clip reads one extra word). Linked entities and device-directed decisions match on all 23. Post-processing,
echo and barge-in have their own tests (`tests/test_post.c`, `tests/test_echo.c`).

**Speed** (x86-64 Linux, 4 threads, ONNX Runtime 1.23.2, shared machine):

| | libarc | arc_audio |
|---|---|---|
| load | 2.0-2.4 s | 6.4 s |
| 8.9 s recording | 1.86 s | 2.2 s |
| all 23 clips (156 s) | RTF 0.31 | |
| peak RSS | 845 MB | 2.1 GB |

Most of the time is the encoder in ONNX Runtime.

## N-gram LM

`arc_set_lm(engine, path, weight, topk)` loads a token n-gram LM for shallow fusion; `arc_set_lm_weight` changes the
weight (cheap, per request), `path` NULL removes it, weight 0 keeps it loaded but off (output identical to no LM). The
maths, when it helps and recommended weights: [USAGE.md](USAGE.md#n-gram-lm). It runs after the suppressed ids and
before the repeat guard, in beam search and greedy decoding (finals and stream partials).

**File format `ARCLM1`** (little endian, natural logs):

| offset | field |
|---|---|
| 0 | `char magic[8]` = `"ARCLM1\0\0"` |
| 8 | `int32 order` (1-8) |
| 12 | `int32 vocab` (the decoder's: 32768 for Arc Mini) |
| 16 | `float32 unscored` (<= 0): a typical text piece's ln p; a scored token with no unigram costs twice that |
| 20 | `int32 cap`: table slots, a power of 2 |
| 24 | `uint8 scored[vocab]`: 1 for ids the LM scores (text pieces, `<s>` = 1, `</s>` = 2), 0 for tags and markers; zero padded to a multiple of 8 bytes |
| then | `cap` entries `{uint64 key, float32 logp, float32 bow}`; key 0 = empty slot (at least one) |

- An entry is an n-gram of ids `g`: `logp` = ln P(g[-1] | g[:-1]) and `bow` = ln of its backoff weight as a context
  (0 if none). A positive `logp` marks an entry that is only a context.
- `key` = FNV-1a 64 over the ids as little-endian int32 bytes, then `^= n * 0x9E3779B97F4A7C15`, then `| 1`. Its slot
  is `(key ^ key >> 32) & (cap - 1)`, then linear probing.
- P(t | h): the longest stored (h_k, t), plus the `bow` of each longer context h_k that was looked up and missed (ARPA
  backoff). Sentences are `<s> ... </s>`; tags and markers are skipped in the history.
- The loader checks the magic and version, header ranges, the exact file size, finite values, odd keys and an empty
  slot, and refuses a vocabulary other than the decoder's. The file is memory-mapped: engines loading the same file
  share its pages.

`arc_audio.lm` reads and writes the same format (`ArcLM`, `write_lm`). Checks: `tests/test_lm.c` (format, backoff,
fusion maths, every load error) and `tests/test_lm.py` (Python and libarc give the same transcript and tags on the
golden clip at weights 0, 0.3, 1 and 3; weight 0 is byte-identical to no LM).

## Not ported yet

arc_audio has these and libarc does not:

- in Arc Mini streams: early commit and names / numbers linking (linking needs a whole utterance's word times);
- the builtin fallback for external words, the `moonshine_release` and `parakeet` transcript tiers;
- `Stream.add_voice_segment` / `voice_roles`, segments, bio events, trunk span tags, the `frames` output of `predict`;
- **Arc Transcriber (full)**: the Parakeet-TDT 0.6B v2 encoder (24 conformer layers, about 600 M parameters).

Text handling is exact for ASCII; non-ASCII follows Python for the cases the tests cover.

A full-backbone port needs, in order: the 128-bin log-mel front end, the FastConformer encoder (with a one-time
conversion of the `.nemo` weights), Arc's adapters (`full/adapters.safetensors`) and the decoder from `full/decoder`
through the existing decoder code, then `tests/golden/full.json` through `arc-parity --backbone full`. It is about 30x
Mini's compute per second of audio, so a CPU port suits offline use and a GPU path would follow. Not started.
