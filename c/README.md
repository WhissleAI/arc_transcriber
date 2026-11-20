# libarc: a native C API for Arc Mini

A dependency-free C11 library (libc, libm, threads) that runs **Arc Mini**, the on-device model of Arc v1, from
the release folder as published (`WhissleAI/arc-v1.1`, `mini/`), with no Python, PyTorch or ONNX. One header,
[`include/arc.h`](include/arc.h). It produces what `Arc.predict` produces offline for Arc Mini: the transcript with
inline tags and word times, names linked to a list, numbers with values, utterance tags (device-directed, intent,
dialogue act), sound events and per-frame probabilities. It also runs **Arc Transcriber Space** (NFT2). For that model,
ONNX Runtime runs the encoder and RNNT networks, and it is an optional build dependency; see
[Arc Transcriber Space](#arc-transcriber-space-nft2-arc_nft2_). The full **Arc Transcriber** backbone (Parakeet) is not
supported yet; see [the plan](#plan-arc-transcriber-full-backbone).

**Verified against** `arc_audio` in this repository and the published `mini/` weights (`WhissleAI/arc-v1.1`). Newer code
or weights mean re-running the checks below.

## Build and use

```
make -C c            # build/libarc.a, build/arc-transcribe, build/arc-score, build/arc-parity, build/arc-trim, tests
make -C c test       # unit tests that need no weights
make -C c ARC_ORT=/path/to/onnxruntime   # also Arc Transcriber Space (c/build/nft2/), see below
```

```c
#include "arc.h"
char err[512];
arc_engine* e = arc_create("arc-v1.1/mini", 4, err, sizeof err);
const char* al[] = {"Brandt"};
arc_name_entry kb[] = {{"Marcus Delaney", "person", "Harbor City Hawks", NULL, 0},
                       {"Theo Brandt", "person", "Harbor City Hawks", al, 1},
                       {"Harbor City Hawks", "team", "Harbor City Hawks", NULL, 0}};
arc_set_names(e, kb, 3);                               /* the list to link to; also biases the decoder */
arc_options o = {.domain = "nba", .trunk = 1, .events = 1, .bias_names = 1, .names = 1, .numbers = 1};
arc_result* r = arc_transcribe(e, pcm, n, &o);         /* 16 kHz mono float, any length */
/* r->utts[i]: text, tags, words (text, start, end, tag), intent, act, device_directed
   r->names[i]: heard "Brandt", entity "Theo Brandt", state "linked", p, words
   r->numbers[i]: heard "102 to 98", kind "score", value 102, value2 98, unit, state "read_text"
   r->events[i]: sound / channel events */
arc_result_free(r);
arc_destroy(e);
```

The command-line tool runs the same calls:

```
build/arc-transcribe arc-v1.1/mini clip.wav [more.wav ...] --domain nba --names examples/example_names.json [--json]
```

`--names` takes a list in the format of `examples/example_names.json`; `--json` writes one line per file with the
transcript (word times and tags), names, numbers and events in arc_audio's keys.

### Candidate verification: `arc_score` and the n-best

An app that has candidates for what was said (catalog titles, contacts, commands) can ask Arc which one the audio
supports. `arc_score` returns the decoder's teacher-forced log P(text | audio) for each text, with the encoder run once
(reused from the last `arc_transcribe` of the same samples) and the texts walked as a prefix tree, so a shared "read …"
is decoded once. Arc writes names inside inline tags, so a text may mark its name with braces: `ARC_SCORE_TAGS` scores
it plain and inside ENTITY_PERSON / ENTITY_WORK and keeps the best; `ARC_SCORE_PERSON` only inside ENTITY_PERSON (one
sequence per text: a cheap first pass). The prompt holds only the context's domain, so the score is what the audio
supports, not what a list expects. Sums are not length-fair: compare texts of similar length, or add a per-token bonus.

```c
const char* texts[] = {"read {Treasure Island.}", "read {Sherlock Holmes.}", "Read Trej Island."};
float lp[3]; int ntok[3];
arc_score(e, pcm, n, texts, 3, ARC_SCORE_TAGS, lp, ntok);      /* lp: -5.9, -34.0, -8.9 (nats) */
```

`arc_utt.alts` / `alt_scores` / `n_alts` are the final decode's finished beams (best first, distinct texts, beam
search's length-normalised scores): free alternatives for a catalog or a parser to pick from.

```
build/arc-score arc-v1.1/mini clip.wav --tags --topic assistant --nbest "read {Treasure Island.}" "Read Trej Island."
```

Speed: scoring 25 one-sequence texts after the decode takes ~70 ms on 4 threads (M3 Pro), ~210 ms with all three tag
variants each; the person-tag pass then all three for the best 5 comes to about 120 ms (from those two measurements).

Other calls: `arc_set_context` (the decoder's context prompt: names with kinds, terms, topic, earlier text),
`arc_set_bias` / `arc_set_bias_strength` (decoder name bias), `arc_frames_compute` (per-frame probabilities of the
shipped read-outs), `arc_memory*` (carry the trunk's state across calls, save / load it), `arc_set_threads`. Results
own all their memory; one engine runs one call at a time (use one engine per thread).

## Streaming: `arc_stream`

Live audio in any chunk size, records out through one callback (on the caller's thread, inside the push). The encoder
is streamed (each layer keeps its keys and values; a frame is emitted once its 240 ms look-ahead has arrived), the
trunk carries its state, and the read-out runs every 100 ms frame, so nothing is re-encoded or re-windowed. Three
modes, one API:

| mode | loads | gives |
|---|---|---|
| `ARC_STREAM_FULL` | everything | Arc's transcript: PARTIAL records (greedy, from the streamed states), FINAL ones (the same decode as `arc_transcribe`), tags |
| `ARC_STREAM_HEADS` | no decoder (`arc_create_ex(..., ARC_NO_DECODER, ...)`) | utterances (times, device-directed, intent, act), events, frames; no text |
| `ARC_STREAM_EXTERNAL` | no decoder | the same, with another recogniser's words (`arc_stream_words` with times, or `arc_stream_text` without) grouped into Arc's utterances |

```c
arc_engine* e = arc_create_ex("arc-v1.1/mini", 2, ARC_NO_DECODER, err, sizeof err);
arc_stream_config c = {.mode = ARC_STREAM_EXTERNAL, .domain = "assistant", .on_output = on_output, .frames = 1};
arc_stream* s = arc_stream_open(e, &c);
arc_stream_push(s, pcm, n);                         /* as audio arrives */
arc_ext_word w[] = {{"turn", 0.42f, 0.61f}, {"it", 0.61f, 0.70f}, {"up", 0.70f, 0.95f}};
arc_stream_words(s, w, 3, 1);                       /* the recogniser's final words, in stream time */
arc_stream_covered(s, 0.95f);                       /* its final result reaches 0.95 s: earlier utterances need no more words */
arc_stream_finish(s);
arc_stream_close(s);
```

Records: `PARTIAL` (text so far), `ENDPOINT` (Arc's segmenter closed an utterance: in EXTERNAL mode, the moment to
ask the recogniser for its final words), `FINAL` (the utterance: text, words, inline tags, read-out),
`TAGS` (`tags_final` 0: the utterance's read-out at its first frame without speech, which changes only if speech
resumes; 1: settled when the read-out closes the utterance, 0.5 s without speech), `EVENT`, and `FRAME` (every 100 ms:
every column of `arc_frames_compute`, with `speech`, `device_directed` and `command_complete` at hand).

Timing. A frame is final when it is sent: the read-out delay plus about 0.1 s after it. The delay is how far the trunk
reads ahead of the frame it labels; Arc was trained with 0, 0.1, 0.2, 0.3 (default), 0.5 and 0.8 s
(`arc_stream_config.delay_ms`). An utterance's tags come at its first non-speech frame (provisional), settled 0.5 s
later; its FINAL text comes after the segmenter's 0.4 s pause (plus the 0.15 s pad the decode reads, in FULL mode).

Checked (`build/test_stream MINI file.wav ...`, chunks of random sizes): FULL finals identical to `arc_transcribe`
(text, times, inline tags), settled tags, directed head and events identical to `arc_transcribe` with the trunk,
HEADS the same without the decoder, FRAME columns identical to `arc_frames_compute`, and EXTERNAL fed Arc's own
aligned words groups them into the same utterances (bit-identical on aarch64; x86 within 1e-5 through the AVX2 GELU).
`build/arc-stream MINI file.wav ... [--mode heads|external]` runs files as live audio and reports RTF, CPU, peak RSS
and the latency of each kind of record.

## What runs, and how it was checked

| Part | arc_audio | Here | Matches the reference |
|---|---|---|---|
| Weights: plain + packed 4-bit (`int4_layout`, groups of 32, zero point + float32 scale) | `weights.py` | `arc_store.c` | bit-exact (486/486 tensors) |
| Tokenizer: SentencePiece-style BPE, byte fallback, 768 special tokens | HF tokenizers | `arc_tok.c` | 320/320 encode / decode cases |
| Moonshine-streaming-tiny encoder (CMVN + asinh + linear, 2 causal stride-2 convs, 6 sliding-window attention layers) | transformers | `arc_moon.c` | states within 2.4e-5 |
| Decoder (6 layers, partial interleaved RoPE, cross-attention + learned positions, gated SiLU MLP, tied output), HF-identical beam search (4 beams) / greedy, Arc's logits processors and trie name bias | transformers, `transcribe.py`, `bias.py` | `arc_moon.c` | 153/153 utterances: identical tokens |
| Utterance segmentation, loop fix, inline-tag readout, difflib span mapping, context prompt | `transcribe.py`, `tagdec.py`, `ctxprompt.py` | `arc_text.c`, `arc_api.c` | 170/170 transcripts (times, text, tags) with and without a context; 152/152 with name bias |
| **Word times**: teacher-forced cross-attention of the alignment heads, z-scored, median-filtered, DTW | `align.py` | `arc_align.c` | 8,744/8,744 words (609 runs): same text, start and end to 1 ms, same tags |
| Features pooled to 10 Hz (mean + max, via float16), layer-2 tap; device-directed head; sound / channel events | `api.py`, `mst.py`, `events.py` | `arc_engine.c` | 153/153 directed; 144/144 events |
| Trunk: 5 rotating-state SSM layers, FiLM on domain + delay, MLPs, domain adapter | `model.py`, `seq/` | `arc_engine.c` | (through the read-outs) |
| **Word channel**: word pieces + trunk states, causal word encoder, rule arrival, frame attention added to the trunk | `wordch.py` | `arc_wordch.c` | fused states within 4.5e-6 of arc_audio's change |
| Tag head (keys P(E)+r, per-query projections, priors, exclusive-group softmax), speaker / utterance state machine, utterance pooling | `model.py`, `engine.py` | `arc_engine.c` | intent, act, device_directed 153/153; max \|p − p_ref\| 6.5e-5 |
| **Names**: decoder tag spans → pointer head (NameEncoder char CNN, span text embedding, similarity features) → rescorer (fuzzy + metaphone similarity, team / recent terms) → name policy; list snapping of unlinked names and capitalised phrases | `engine.py`, `model.py`, `rescorer.py`, `namesim.py`, `textlayer.py`, `kbsnap.py` | `arc_link.c`, `arc_ntext.c` | see below |
| **Numbers**: decoder tag spans → number words → value heads (word tokens, unit) and the words' own value; extend_number | `engine.py`, `numvalue.py`, `numlib.py`, `textlayer.py`, `api.py` | `arc_link.c`, `arc_num.c` | see below |
| Output tidying (word edges, names trimmed to the mention, de-duplication; number words) | `tidy.py` | `arc_tidy.c` | 98/98 cases |
| **Dates**: a date number covers its month ("March third", "the third of March", with or without a year) and names the date (year / month / day, ISO) | `written.py` (parse_date, date_iso), `api.py` (extend_date), `tidy.py` | `arc_date.c` | 169/169 checks |

**Golden parity** (the release's own port check, `tests/golden/mini.json` with `scripts/check_parity.py`):

```
make -C c parity MINI=/path/arc-v1.1/mini      # PARITY OK
```

Transcript text, inline tags, linked names (heard, entity, state) and number values (heard, kind, value, value2) are
identical; pooled features, event features and event probabilities match to the printed precision; the per-frame
probabilities of every shipped read-out are within tolerance. It also passes with an x86-64 build.

**Names, numbers and word times against arc_audio** (`tools/dump_names.py` runs `Arc.predict`, `build/arc-transcribe
--json` runs libarc, `tools/compare_names.py` compares every field: names' heard text, entity, state, tag, source,
words, times to 1.5 ms, rescorer p, span p and the pointer head's own choice and p to 2e-3; numbers' heard text, kind,
unit, value, value2, state, kind head and times): **609 clip runs, all identical** (8,744 words, 756 names, 443
numbers; 0 differences):

| Set | Clips | Domain(s) | List |
|---|---|---|---|
| `examples/example_synth.wav` | 1 | nba | `examples/example_names.json` |
| Synthetic passages with names and numbers (6 texts × 3 voices) | 18 | assistant | 10 names with kinds, groups, aliases |
| Synthetic voice commands (67 × 2 voices) | 134 | assistant | same |
| Synthetic book requests ("read Moby Dick") | 200 | assistant | 21 titles and authors |
| Long recordings (dialogue, narration, commands clean / clipped / noisy / phone / mumbled, barge-in, music) | 9 | meeting | same as passages |
| An 8-minute concatenation (two trunk chunks of 300 s) | 1 × 2 | nba, talk | passages + example lists |
| Example + passages + 2 long recordings | 21 × 10 | nfl, esports, news, soccer, cricket, stream_chat, unknown, nhl, meeting, court_trial | combined list |
| Example + 6 passages with a misspelt list ("Theodore Brant", "Harbour City Hawks") | 7 × 5 | esports, news, meeting, nba, assistant | misspelt |

Paths seen in these runs: linked by the rescorer, not in the list, abstain, snapped from capitalised phrases; numbers
from the words (read_text) and from the value heads (read, 35 records); 9 number kinds. Two paths were not reached
end to end (they are covered by the unit tests of their parts): the name policy's fuzzy-text link (`via: "text"`) and
the snap of an unlinked decoder name.

**Unit tests without weights** (`make -C c test`, run by CI on Linux and macOS):

```
build/test_num   tests/data/num.jsonl     3215/3215   numlib / numvalue / textlayer (tools/dump_num.py)
build/test_ntext tests/data/ntext.jsonl   7849/7849   rescorer, namesim, ExactNames, kbsnap, metaphone, difflib, NFKD (tools/dump_ntext.py)
build/test_tidy  tests/data/tidy.jsonl      98/98     tidy.py (tools/dump_tidy.py)
build/test_date  tests/data/date.jsonl    169/169    spoken dates: written.parse_date / date_iso, api.extend_date (tools/dump_date.py)
```

The metaphone is checked against `jellyfish.metaphone` (an arc_audio dependency) on 1,850 cases and on all 471,967
words of a system word list. Note: `rescorer.py`'s fallback copy of metaphone (used only without jellyfish) differs
from jellyfish on words ending in "mb", "gn" or "gh"; libarc follows jellyfish.

**Checks that need reference data** (made by `tools/dump_*.py` with arc_audio from clips of your own, and the weights):

```
tools/verify_store.py FILE  vs  build/store_hash FILE        weights, bit for bit
build/test_tok   tokenizer.json  tok_cases.tsv                 tools/dump_tok.py
build/test_moon  moonshine-dir   moon-ref-dir                  tools/dump_moon.py   (encoder states, beam tokens)
build/test_transcribe mini wav-dir transcribe.jsonl            tools/dump_transcribe.py (+ dump_bias.py)
build/test_full  mini wav-dir ref_assistant.json [words]       Arc.predict(domain='assistant')
tools/compare_names.py ref.jsonl cand.jsonl                    tools/dump_names.py vs build/arc-transcribe --json
```

Precision notes: the trunk's SSM runs as the step recurrence (mathematically the chunked parallel form arc_audio uses;
float rounding differs at the 1e-5 level); kernels accumulate in float32 in a different order than PyTorch. Decisions
(tokens, tags, links, values) have matched everywhere tested; probabilities agree to about 1e-4.

## Speed and memory (Apple M3 Pro, 4 threads, other work running)

152 clips, 333 s of audio:

| | Real-time factor |
|---|---|
| Transcript only | 0.026 |
| + word times | 0.028 |
| Everything (trunk, events, word channel, names, numbers) | 0.046 |
| ... with a 10-name list | 0.048 |

Load: 30–90 ms (weights are memory-mapped; packed 4-bit matrices are used in place, dequantised a row at a time,
never expanded; the event and device-directed heads load on first use). Peak memory for a 30 s clip with everything:
180 MB. The encoder runs over the whole clip at once (as arc_audio's offline path does), so peak memory grows with
clip length (about 1 GB for 8 minutes with the trunk on); chunked encoding is future work. Files: 68 MB (`mini/`);
`build/arc-trim` copies a safetensors file keeping only given tensor prefixes (libarc reads `trunk. head. diar.
ent. word.` from `model.q4.safetensors`).

Scores and beams are checked against arc_audio (`tests/test_score.c` with `tools/dump_score_ref.py`): 56/56 scores
within 2e-2 nats (worst 3.6e-4) and 5/5 n-best lists identical on stretches of `examples/example_synth.wav`
(`tests/data/score_ref.jsonl`), 425/425 and 25/25 on 25 spoken book requests. Note for Python references: transformers'
MoonshineStreaming decoder (5.19) adds its positional embeddings to `encoder_outputs.last_hidden_state` in place, so
calling the model twice with one encoder output drifts every call after the first; copy the states per call.

```
build/test_score arc-v1.1/mini ../examples tests/data/score_ref.jsonl
```

## Arc Transcriber Space (NFT2): `arc_nft2_*`

**Arc Transcriber Space** (`WhissleAI/arc-transcriber-space`, folder `nft2/`) is Arc for questions about the Moon and
space. It uses a cache-aware streaming FastConformer encoder (17 layers, d 512, int8) with an RNNT head for the words.
The Arc decoder reads the same encoder (through Arc's adapters) to tag spans, and Arc's names / numbers / events heads
run on top. libarc runs it as `Arc.predict(wav, domain, transcript="nft2", names=..., context=...)` does with the ONNX
encoder:

- libarc's own C does the log-mel front end, the chunked streaming loop with its pre-encode cache, greedy RNNT decoding
  with names-list biasing (`BoostTrie`, alpha 1.5), the sentencepiece pieces and word times, Arc's adapters
  (`MemProj`, `FrameProj` arc / tap, `pool_range`), the context prompt (`build_prompt_v2`), the Arc decoder, tagging the
  RNNT words (char-level alignment, one type per surface, names-list spans, decoder casing), and the trunk and linking.
- **ONNX Runtime** (C API, CPU) runs only the networks in `nft2/encoder_onnx/`: `encoder_first.onnx` / `encoder.onnx`
  (weights in `encoder.weights`, memory-mapped once and shared by both sessions) and the RNNT `decoder.onnx`,
  `joint_enc.onnx`, `joint_net.onnx`. They sit behind a small internal interface
  ([`src/arc_nft2_be.h`](src/arc_nft2_be.h)), so a native C backend can replace ONNX Runtime later without changing
  anything else.

ONNX Runtime is optional. The default `make` stays dependency-free, and `arc_nft2_create` then fails with a message.

```
# ONNX Runtime 1.23 for your platform, from github.com/microsoft/onnxruntime/releases (onnxruntime-<os>-<arch>-1.23.x.tgz)
make -C c ARC_ORT=/path/to/onnxruntime-linux-x64-1.23.2      # builds into c/build/nft2/
c/build/nft2/arc-nft2 arc-transcriber-space/nft2 question.wav --names moon_names.json
```

```c
arc_nft2* m = arc_nft2_create("arc-transcriber-space/nft2", 4, err, sizeof err);
const char* al[] = {"Oceanus Procellarum"};
arc_name_entry kb[] = {{.name = "Tycho", .kind = "lunar_feature"},
                       {.name = "Ocean of Storms", .kind = "lunar_feature", .aliases = al, .n_aliases = 1}};
arc_nft2_set_names(m, kb, 2);              // biasing, the decoder prompt and linking; kept until changed
arc_result* r = arc_nft2_transcribe(m, pcm, n, NULL);   // 16 kHz mono float
// r->utts[i].text: the RNNT words with the decoder's casing; .words[k].tag: ENTITY_LUNAR, ENTITY_SPACECRAFT, ...
// r->utts[i].decoder_text: the Arc decoder's own text; .directed / .directed_head: the device-directed head
// r->names[i].entity: the list's name when linked ("Ocean of Storms" for "oceanus procellarum"), .state
arc_result_free(r);
arc_nft2_destroy(m);
```

- **Context** (`arc_nft2_set_context`): `dom` (NULL: the model's default `"space"`, `""`: none), `prev` or `auto_prev`
  (the last minute's text), `kb` with kinds (default: the names list itself), and session memory (`speakers`,
  `numbers`, `entities`), as in arc_audio's `context=` dict.
- **Options** (`arc_nft2_options`): `domain` for the trunk (default `"assistant"`), `no_bias`, `no_events`.
- **Clips**: up to 20 s are one utterance; longer clips are cut by the segmenter (with 0.15 s of padding). The features
  for names, numbers, events and the device-directed head come from 30 s chunks with 4 s of context, as on the server.
- **Streams** (`arc_nft2_stream_open / feed / flush / close`): feed any amount of audio. `feed` returns each
  utterance once the segmenter closes it (0.4 s into a pause). `flush` returns the whole stream with names, numbers,
  events and the device-directed head. Each streamed utterance is cut 0.15 s after its speech and then padded with
  silence to 0.5 s past its last frame (`transcription.nft2.tail_pad_s`), as arc_audio's streams do: the RNNT emits a
  word's last pieces a few hundred ms after it is spoken, and without the padding a stream loses its last word
  ("near Sinus me"). Offline decoding (`arc_nft2_transcribe`) has no such padding.
- **Names lists**: names and aliases are matched after spoken-form normalization (`Apollo 12` → `apollo twelve`;
  every character other than a–z, digits, `'`, `-` and space is dropped). Always set `kind` (`lunar_feature`, `site`,
  `celestial`, `spacecraft`, `mission`, `crew`, ...). Give aliases for names that are not said as written: `Artemis II`
  needs `"Artemis 2"`, and non-ASCII names need an ASCII alias. The full guide is
  [docs/NAMES_LISTS.md](../docs/NAMES_LISTS.md).

### App read-outs, live events and echo

`arc_abi_version()` / `arc_version()` report the header's `ARC_ABI_VERSION` (2) and `ARC_VERSION` the library was built
from; compare them with the header you compile against.

- **Dates** (1.3.1): a number of kind `date` with a month name keeps the month in `heard` ("March third", "the third
  of March") and `arc_number_date(r, i)` gives its `year` / `month` / `day` (0: not said) and `iso` ("--03-03",
  "2024-03-03", "2024"; NULL for a day alone), as arc_audio's `date` / `iso` fields. NULL when the number names no date.
  A function, not new `arc_number` fields, so the ABI stays 2.

- **Words** (`arc_word`): `conf` (the RNNT's unbiased probability of the word's least likely token), `alts` / `alt_p`
  (the runner-up spelling; with `options.nbest = 4`, also the words other beam hypotheses have there) and `disfl`
  (`filler` / `repeat` / `fragment`).
- **Utterances** (`arc_utt`): `conf`, `clean_text` (without disfluencies), `quantities` (`arc_quantity`: type, value,
  unit, relative, word span; "seventy five percent" -> percent 75), `n_dropped` (words the guard removed), `alts` /
  `alt_scores` with `nbest` (other texts, the beam's share), `playback_leak` / `playback` with a reference, and
  `tagger_json` with Arc Tagger.
- **Guard** (on; `options.no_guard = 1` turns it off): no words where the audio has no speech-level energy, no
  low-confidence repeats, no low-confidence words in the trailing silence.
- **Sound-alike names** (`options.almost = 1`): names records with `state = "almost"`, `entity` NULL, `candidate` /
  `candidate_kb_index` / `candidate_p`, for uncertain heard words (confidence below 0.7, not common vocabulary) that sound
  like a list entry.
- **Live streams** (`options.live = 1`): the open utterance is decoded one 560 ms encoder chunk at a time with the
  encoder caches and the RNNT state carried over, so nothing is decoded twice. `arc_nft2_stream_events` returns the
  events of the last call: `ARC_EV_PARTIAL` (text, `n_stable`), `ARC_EV_ENTITY` (a list name, an "almost" name or a
  completed quantity in the stable words), `ARC_EV_EOT` (end-of-turn probability, every feed), `ARC_EV_BARGE_IN` and
  `ARC_EV_FINAL`. `arc_nft2_stream_set_names` swaps the names list mid-stream (the open utterance's bias switches at
  once); `arc_nft2_stream_end_turn` closes the open utterance now. Finals are still decoded per utterance, as before.
- **Echo / self-speech** (`arc_nft2_stream_feed_ref`): the device's playback as sent to the speaker, with its stream
  time. A partitioned-block frequency-domain NLMS canceller (blocks of 256 samples, 32 partitions = 512 ms of echo
  path, double-talk freeze) runs on the microphone. Each utterance gets `playback_leak` (the share of its energy the
  playback explains); at `options.playback_threshold` (0.7) or above it is marked `playback` (left out with
  `playback_drop`). `ARC_EV_BARGE_IN`: speech the playback does not explain (leak below 0.5 over the last 0.3 s) while
  the device plays, once the canceller has seen 1 s of playback. `echo_decode = 1` decodes finals from the residual.
- **Arc Tagger** (`make ARC_WITH_ARCTAG=/path/to/arc-tagger-c`, which needs its built `build/libarctag.a`):
  `arc_nft2_set_tagger(m, "arc-tagger-c/model", "kids", 1, err, sizeof err)` runs the tagger (native backend, int8) on
  each final's words with their times and confidences; `arc_utt.tagger_json` is its JSON.
- **Example**: [`examples/nft2_app.c`](examples/nft2_app.c) (`build/nft2/nft2-app-example MODEL mic.wav
  [playback.wav]`) uses only `arc.h` and `libarc.a`: a live stream with a names list and an optional echo reference,
  printing events. `make` builds it with the library.

arc_audio has the same code, and the C is checked against it: `tests/test_post.c` (guard, disfluencies, quantities,
sound-alike names, end of turn; 98 cases from `tools/dump_post.py`), `tests/test_echo.c` (the canceller's leak and
barge-in on 22 windows from `tools/dump_echo.py`, regenerated from `examples/example_synth.wav`), and
`compare_nft2.py` on the clip set with `--almost --nbest 4` (confidences, alternatives, n-best texts, quantities and
almost records match with the same log-mel frames).

### How it was checked

`tools/dump_nft2.py` runs arc_audio (`Arc.predict`, `transcript="nft2"`, ONNX encoder) on a manifest of clips and
writes every stage. `build/nft2/arc-nft2 --manifest ... --json --dump DIR` writes the same stages from libarc, and
`tools/compare_nft2.py` compares the two. The test set has 23 clips (156 s): the 4 bundled synthetic samples with the
example names list, 7 real recordings of lunar-geology conversations (narrowband radio), 4 children's questions, 7
synthetic questions, and a 34.7 s clip that exercises segmentation and the 30 s feature chunks.

- **The same log-mel frames give the same output everywhere.** With `dump_nft2.py --mel-from DIR`, the reference
  encodes libarc's log-mel frames, and all 23 of 23 clips match on every discrete output: RNNT token ids and frames,
  decoder prompt and tokens, decoder text, transcript words with times and tags, names (heard, entity, state, type,
  times), numbers, events and the device-directed decision. The encoder output and tap are bit-identical. Floats after
  them differ by at most 2.4e-6 (decoder memory), 4.3e-6 (pooled features) and 6e-5 (tap features), from summation
  order. Probabilities match to 4 decimals.
- **End to end against PyTorch's front end**: 14 of 23 clips are identical on every output. arc_audio computes the
  log-mel with `torch.stft` (float32 MKL FFT). libarc's FFT is closer to an exact float64 log-mel (max error 1e-5 to
  2.4e-4, where torch's is 5e-5 to 5.8e-4), and the two differ by at most 6e-4. The encoder quantizes its activations
  dynamically to int8, which turns that into up to 0.8 in the encoder output. The visible effects: 6 clips move one
  word boundary by one 80 ms RNNT frame (the same words, tags and links), 1 real clip reads one extra word ("Alright.")
  at the start of an utterance, and 2 real clips shift the boundaries of narrowband / telephone channel events. Linked
  entities and device-directed decisions are the same on all 23 clips. The same spread appears between any two FFT
  implementations, so it is not specific to C.
- **Streams** (`dump_nft2.py --stream`: `Arc.stream(transcript="nft2")` fed 100 ms at a time; against
  `arc-nft2 --stream`, compared with `compare_nft2.py --stream`). With the same log-mel frames, all 23 of 23 clips
  match: every utterance's RNNT tokens, prompt, decoder tokens and text, and the words with times and tags. Linked
  names (heard, entity, state) match too. With torch's front end, 17 of 23 clips are identical; the other 6 move word
  boundaries by one RNNT frame, and their texts are the same. Streams keep the last word on the bundled samples:
  "... near Sinus Medii?" and "Why is Tycho so bright?".
- Offline, `sample_3` ("Why is Tycho so bright?", 2.2 s, one utterance with no padding) gives an empty RNNT transcript
  in both arc_audio and libarc. The decoder reads it correctly (`decoder_text`, ENTITY_LUNAR). Streamed, it is
  transcribed in full.
- Arc Mini is unchanged: on this branch, `arc-parity`, `arc-transcribe --json` and `arc-stream` give byte-identical
  output to `main`, and `make test` passes. An AddressSanitizer build runs the offline and stream paths clean, with no
  leaks.

### Speed and memory (x86-64 Linux, 4 threads, ONNX Runtime 1.23.2, a shared machine with other jobs running)

| | libarc | arc_audio (Python, same ONNX encoder) |
|---|---|---|
| load | 2.0–2.4 s | 6.4 s |
| 3.2 s question | 0.73 s | 0.95 s |
| 8.9 s recording | 1.86 s | 2.2 s |
| 34.7 s recording | 12.1 s | 13.1–14.1 s |
| all 23 clips (156 s) | RTF 0.31 | |
| peak RSS | 845 MB | 2.1 GB |

Most of the time goes to the encoder in ONNX Runtime: about 1.1 s of the 1.86 s for the 8.9 s recording. The Arc decoder
takes 0.44 s and the RNNT 0.09 s. A clip longer than 20 s is encoded twice, once per utterance and once for the 30 s
feature chunks, as arc_audio does.

## Not ported yet

arc_audio has these and libarc does not: in streams (other than live NFT2 events), early commit and names / numbers (linking needs the word times
of a whole utterance; EXTERNAL words get the utterance read-out, not word-sourced names), the builtin fallback for
external words, the
`moonshine_release` and `parakeet` transcript tiers, speaker read-outs and diarization beyond the slot state machine
(`voice` / `fused` modes, roles), segments, bio events, trunk span tags (`spans`), the `frames` output of `predict`
other than `arc_frames_compute`, and the full backbone. Text handling is exact for ASCII; non-ASCII text follows
Python for the cases the tests cover (accented Latin names fold as NFKD does; lower-casing is ASCII-only).

## Plan: Arc Transcriber (full backbone)

The full model is Arc's same heads on a bigger encoder; nothing in it is TDT decoding. From `arc_audio/server.py`
and the release (`WhissleAI/arc-v1.1`, `full/`):

- **Encoder**: NVIDIA Parakeet-TDT 0.6B v2's FastConformer encoder, unmodified (`nvidia/parakeet-tdt-0.6b-v2`,
  `parakeet-tdt-0.6b-v2.nemo`, 2.47 GB, CC-BY-4.0; Arc downloads it at run time). Only the encoder and the
  preprocessor are used: the TDT decoder and joint are not (they appear only in the optional `transcript="parakeet"`
  tier through NeMo). Architecture: 128-bin log-mel front end (pre-emphasis 0.97, STFT n_fft 512, hop 160, window
  400, the checkpoint's mel filterbank and window, log(x + 2^-24), per-feature mean / std over the clip); 8×
  subsampling by stride-2 conv2d (256 channels) and a linear to 1024; 24 conformer layers of d 1024 (half-step
  SiLU feed-forward 4096, relative-position multi-head self-attention, 8 heads, with bias_u / bias_v and a rel-shift,
  convolution module: pointwise + GLU, depthwise k 9, batch norm, SiLU, pointwise; second feed-forward; layer norm);
  80 ms frames. arc_audio reads the weights in float16 and runs them through transformers' `ParakeetEncoder`, in
  30 s chunks with 4 s of context on each side (`ServerFeatures`).
- **Arc's adapters** (`full/adapters.safetensors`, 22.5 MB): `MemProj` (LN, 1024 → 1024 GELU → 320, LN) turns the
  encoder output into the transcript decoder's memory, with a position correction from the decoder's own position
  table (`mem_with_pos`, 4× frame rate offset 2); `FrameProj` "arc" (LN, 1024 → 512) and "tap" (from layer 12, 1024 →
  320) give frames that are pooled to 10 Hz (`pool_range`, mean + max) and, for "arc", an MLP to the trunk's 640
  input features.
- **Decoder and heads**: `full/decoder/` (15.6 MB, a Moonshine-style decoder with its own tokenizer) and
  `full/model.q4.safetensors` (37 MB: trunk, tag head, entity heads, word channel, as in Mini) — the same code paths as
  Mini in libarc (`arc_moon.c` decoder, `arc_engine.c`, `arc_link.c`, `arc_wordch.c`), parameterised by the full
  config (`hop_s` 0.08 for word times, decoder dimensions from its config).

What a C port needs, in order:

1. A log-mel front end matching `LogMel` (FFT and Hann window exist in `arc_base`; add the filterbank from the
   checkpoint and the normalisation), checked against `Backbone.mel` frame by frame.
2. The FastConformer encoder (`arc_parakeet.c`): conv2d subsampling, relative positional encoding and rel-shift,
   the conformer block, batch norm folded into the depthwise conv at load. Weights: a conversion tool from the `.nemo`
   (a tar holding `model_weights.ckpt`, a PyTorch pickle: convert once with Python to safetensors, encoder only,
   float16 or Arc's 4-bit layout; 4-bit makes it about 350 MB). Checked layer by layer against
   `transformers.ParakeetEncoder` (tap at layer 12 and the output), then against `ServerFeatures` chunking.
3. The adapters and pooling (`MemProj`, `FrameProj`, `pool_range`, `mem_with_pos`), then the decoder from
   `full/decoder` through the existing `arc_moon` decoder (it reads its sizes from the config: it must stop assuming
   the tiny model's 320 / 6 layers), and the word-time alignment with 80 ms hops.
4. `tests/golden/full.json` through the same `arc-parity` (`--backbone full`), then the names / numbers comparison.

Cost: about 600 M encoder parameters (2.4 GB float32, 1.2 GB float16, ~350 MB 4-bit) and roughly 30× Mini's compute
per second of audio, which is why arc_audio runs it on a GPU; a CPU C port is practical for offline use, and a CUDA
or Metal path would follow. Not started here: the weights need about 5 GB of free disk to fetch and convert.

## Layout

`include/arc.h` public API · `src/arc_api.c` engine, context, bias, transcript · `arc_moon.*` Moonshine encoder /
decoder / beam search · `arc_tok.*` tokenizer · `arc_text.*` segmentation, tag readout, context prompt ·
`arc_align.*` word times · `arc_engine.*` features, heads, trunk, read-outs, stages · `arc_wordch.*` word channel ·
`arc_link.*` names and numbers · `arc_ntext.*` name text matching · `arc_num.*` number values · `arc_tidy.*` output
tidying · `arc_result.c` results · `arc_store.*` Arc weights and 4-bit kernels · `arc_base.*`, `arc_kernels.c`,
`arc_nft2.*` Arc Transcriber Space (front end, streaming
encoder loop, RNNT greedy + biasing, adapters, finals) · `arc_nft2_be.h` its encoder / RNNT backend interface, with
`arc_nft2_ort.c` (ONNX Runtime) and `arc_nft2_none.c` (none) · `arc_weights.c` shared C helpers (JSON, safetensors mmap, thread pool, GEMM with NEON / AVX2) ·
`tools/` arc-transcribe, arc-parity, arc-trim, arc-nft2, the reference dumpers and compare_nft2.py · `tests/` tests and fixtures.

## Licence

Like the rest of this repository, libarc is part of arc_transcriber: CC BY-NC 4.0 plus the Whissle terms in
[LICENSE](../LICENSE). Arc's weights are CC BY-NC 4.0 (Whissle); Moonshine-streaming-tiny inside `mini/` is MIT; the
metaphone in `src/arc_ntext.c` follows jellyfish (MIT, see NOTICE).
