# arc_transcriber

A command-line tool and Python library for **Arc**, an all-in-one streaming audio model. Give it audio (a file or live), and
optionally your list of names and some context. One model returns, every 100 ms:

- a **transcript** with **inline tags** on names, numbers and entities;
- **names linked to your own list**, and **numbers with their values** (scores, clocks, money, dates, durations, ...);
- **speakers** and their **roles**, per-utterance **acts / intents**, segment labels;
- **sound and channel events** (silence, telephone / narrowband audio, degraded audio, clipping; more labels as experimental);
- **device-directed speech** (is this utterance addressed to an assistant?);
- **context memory**: the decoder can take your names, terms, a topic, and what was said earlier.

Arc v1 has two models, with one API and one output schema:

| model | `--backbone` | download | runs on | use it for |
|---|---|---|---|---|
| **Arc Mini** | `mini` (default; alias `edge`) | about 71 MB | CPU (files at about 0.3x real time on 4 cores), GPU optional | apps, laptops, on-device, lowest latency |
| **Arc Transcriber** (full) | `full` (aliases `parakeet`, `server`) | about 77 MB + NVIDIA Parakeet-TDT 0.6B v2 encoder (about 2.5 GB, on first use) | GPU recommended, CPU for files | best transcript and name accuracy |

**Arc Transcriber Space** (`backbone="nft2"`, model `WhissleAI/arc-transcriber-space`) is a CPU model for voice apps
about the Moon and space flight, with names-list linking: see [Arc Transcriber Space (NFT2)](#arc-transcriber-space-nft2).

Coming next: **Arc Transcriber Edge**, a 4-bit version of the full model for Android, Mac and cloud.

- **Model weights, model card and evaluation results:** https://huggingface.co/WhissleAI/arc-v1.1 (downloaded on first use)
- **This repo:** inference code (`arc_audio`) and the `arc-transcribe` CLI, CC BY-NC 4.0 with the Whissle terms (non-commercial only; commercial licences: ksingla@whissle.ai). Bugs go in [Issues](../../issues).
- English, 16 kHz mono (other rates are resampled).

## Install

```bash
pip install git+https://github.com/WhissleAI/arc_transcriber
pip install "arc-transcriber[live] @ git+https://github.com/WhissleAI/arc_transcriber"     # + websockets, sounddevice (live / mic)
```

The first run downloads `WhissleAI/arc-v1.1` into the Hugging Face cache (only the folder of the model you
use). The model repository asks you to accept its terms (approval is automatic): open https://huggingface.co/WhissleAI/arc-v1.1
while logged in, accept, then `hf auth login` (or set `HF_TOKEN`). Arc Transcriber (full) also downloads `nvidia/parakeet-tdt-0.6b-v2`
(public, CC-BY-4.0); NeMo is not needed. Python >= 3.10, torch >= 2.1, transformers >= 5.19.

Benchmarks against other systems are on the [model card](https://huggingface.co/WhissleAI/arc-v1.1).

## CLI

```bash
arc-transcribe meeting.wav                                                     # Arc Mini, text with inline tags
arc-transcribe game.wav --domain nba --names roster.txt                        # link names to your list
arc-transcribe game.wav --names roster.txt --format json -o game.json          # full records (JSON)
arc-transcribe call.wav --backbone full --device cuda                          # Arc Transcriber (full) on a GPU
arc-transcribe call.wav --device auto                                          # cuda > mps (Apple GPU) > cpu
arc-transcribe meeting.wav --written                                           # written form: "2:30 p.m.", "$45", "March 3"
arc-transcribe talk.wav --context terms.txt --topic "product launch"           # decoder context: names / terms / topic
arc-transcribe talk.wav --auto-context                                         # context memory: feed back what was said
arc-transcribe room.wav --events all                                           # include experimental sound-event labels
arc-transcribe game.wav --format srt -o game.srt                               # or --format vtt
arc-transcribe game.wav --live                                                 # stream the file at real-time pace
arc-transcribe --mic --domain talk                                             # microphone (needs sounddevice), Ctrl-C to stop

export DEEPGRAM_API_KEY=...                                                    # optional external ASR words (keys from the environment)
arc-transcribe game.wav --asr deepgram --names roster.txt
arc-transcribe game.wav --asr deepgram --demo                                  # no key, no network: simulated messages
```

Try it on the bundled synthetic clip (fictional names): `arc-transcribe examples/example_synth.wav --domain nba --names examples/example_names.txt`.

| option | |
|---|---|
| `--backbone` | `mini` (default) or `full`; `edge` = `mini`, `parakeet` / `server` = `full` |
| `--domain` | `nba` `nhl` `nfl` `cricket` `soccer` `esports` `news` `talk` `stream_chat` `meeting` `court_appellate` `court_trial` `assistant` `assistant_noisy` `short_clips` `unknown` (default) |
| `--names` | your name list for linking: `.txt` (one name per line, optionally `name<TAB>kind`) or `.json` (strings or `{"name", "kind", "aliases", "group"}`) |
| `--context`, `--topic`, `--auto-context` | decoder context: terms (`.txt`, `.json` or `a,b,c`), a topic hint, and feeding back earlier text |
| `--events` | `default` (stable + beta labels), `all` (adds experimental labels), `none` |
| `--no-snap` | turn off the fuzzy / phonetic snap of mentions to `--names` |
| `--asr` | `builtin` (default), `deepgram`, `verbit` (external transcript words) |
| `--device` | `cpu` (default), `cuda`, `mps` (Apple GPU; falls back to `cpu` with a warning when MPS is missing) or `auto` (cuda > mps > cpu) |
| `--format` | `txt` (default), `json`, `srt`, `vtt`; `-o FILE` (default stdout) |
| `--written` | txt / srt / vtt show the written transcript (numbers as digits, names in your list's spelling) instead of inline tags |
| `--live`, `--mic`, `--duration S`, `--fast` | live mode from a file at real-time pace, or from the microphone |
| `--model`, `--revision` | another Hugging Face repo or a local folder (default `WhissleAI/arc-v1.1`), or a git revision of the model repository (default: the one this package pins) |

Live progress goes to stderr; the result goes to stdout or `-o`.

## Output

`--format txt` prints one line per utterance, with the speaker and inline tags after each name / number:

```
[   2.40-   7.00] S1: [Marcus Delaney](name: Marcus Delaney) brings the ball up the floor fourth quarter, [three minutes](duration: 180 s) left.
[  15.40-  19.40] S1: That puts the Harbor City Hawks ahead, [102 to 98.](score: 102-98)
```

`--format json` (and `predict()` in Python) gives the full records:

| key | content |
|---|---|
| `transcript` | utterances `{start, end, text, words, tags, device_directed: {p, directed, threshold, source, status}}`: the text as spoken ("two thirty", "forty five dollars"); `device_directed` is the raw head score (see [Device-directed](#device-directed-which-number)) |
| `transcript_written` | the same utterances in written form, the text to show: numbers from Arc's number tags as digits ("2:30 p.m.", "$45", "€12.50", "March 3, 2024", "12.5%", "21st", "555-1234", "3–1" in sports domains) and linked names in your list's spelling. A rewritten span is one word `{text, start, end, written_from, kind}` |
| `transcript_rewritten` | linked names replaced by the full entity name; numbers unchanged (use `transcript_written` to display) |
| `names` | `heard`, `tag`, `entity` (your list's spelling) or null, `state` (`linked` / `not_in_list` / `abstain`), `start`, `end`, `via` (`snap` = filled by the name-list snap; `alias` / `partial` = linked by the list check, with the linker's own decision in `linker_state`), `candidates` (the linker's ranked list entries `[{entity, p}]`, best first, with `p_not_in_list`), `snap_candidates` (`[{entity, score}]` from the snap) |
| `numbers` | `heard`, `kind`, `value` (+ `value2` for pairs), `unit`, `start`, `end`; dates also `date: {year, month, day}` and `iso` (`"--03-03"` for "March third", `"2024-03-03"`, `"2024"`) |
| `speakers`, `diarization` | per-speaker role and tags; `{speaker, start, end}` segments |
| `utterances`, `segments`, `spans`, `events` | per-utterance acts / intents, segment labels, word-span tags, instant events (speaker change, ...) |
| `audio_events` | `{label, start, end, p, status}` (status `stable` / `beta` / `experimental`) |
| `bio_events` | cough / throat clear / breath (off in the `assistant` domains by default: see `masks`) |

## Python

```python
from arc_audio import Arc, load_audio

arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone="mini", device="cpu", threads=4)  # backbone="full": Arc Transcriber
names = ["Marcus Delaney", {"name": "Theo Brandt", "aliases": ["Brandt"]}, {"name": "Harbor City Hawks", "kind": "team"}]

# whole file
out = arc.predict("game.wav", domain="nba", names=names, events="default")

# context for the decoder: names, free-form terms, a topic, earlier text; auto_prev = context memory
arc.set_context(names=names, terms=["pick and roll", "Whissle"], topic="nba", auto_prev=True)

# live: 100 ms chunks of float32 mono 16 kHz audio
wav = load_audio("game.wav")
s = arc.stream(domain="nba", names=names)
for i in range(0, len(wav), 1600):
    new = s.feed(wav[i:i + 1600])            # new transcript lines, names, numbers, audio_events, device_directed, ...
s.set_context(names=["Theo Brandt"], previous_text="...")    # context can change at any time (utterances closed after the call)
final = s.flush()
```

Push-to-talk and voice-agent turns: keep one stream open and end each turn with `end_turn()` (alias `finalize()`). It closes the
open utterance at once and (by default) waits for its final transcript, which is in the returned result; the stream stays open for
the next turn, with the same keys as `feed()`. `end_turn(context=...)` sets the decoder context for the next turn.

```python
with arc.stream(domain="assistant", names=contacts) as s:      # close() on exit: stops the worker thread
    for chunk in turn_audio:
        s.feed(chunk)
    out = s.end_turn()                                           # out["transcript"]: this turn's final(s)
    s.set_context(previous_text=out["transcript"][-1]["text"])   # or end_turn(context=...)
```

- Utterances close on a 0.4 s pause; one shorter than `min_s` (1.0 s, e.g. "Pause.") closes after `short_pause_s` (0.8 s) of
  silence. `arc.stream(segmenter=dict(short_pause_s=0.6))` (or `pause_s`, `min_s`, `max_s`; `short_pause_s=None` keeps short
  utterances open until `max_s` (15 s) or `flush()`); `predict(..., segmenter=...)` too.
- `switch_interval_s`: the stream does not change the interpreter's `sys.setswitchinterval` unless asked
  (`arc.stream(..., switch_interval_s=True)` uses the model's 0.5 ms; a float sets it). The previous value comes back on `flush()`,
  `close()`, the `with` block's end, garbage collection of the stream, or interpreter exit.
- `live_hold=True` (or `dict(name_p=0.5, min_span_s=0.3)`): the `live` view holds back first guesses: a name guess with `p` under
  `name_p`, and any name / number guess on a span open for less than `min_span_s`, show as `held: true` with no guess.
- `command_complete_hold_s=0.5` (stream and `predict()`): a `discourse.assistant.command_complete` event is released only after that
  much time with no speech; if the speaker goes on ("set an alarm at nine am | on next Sunday") it waits for the real end. The event
  keeps `t` (when Arc fired) and adds `released_t` and `hold_off_s`. Default 0: released at once.

- `predict(...)["transcript_written"]` is the written form; `arc.transcribe(wav, written=True)` adds `text_written` / `words_written` to
  each utterance; each `stream.feed()` / `flush()` result has `transcript_written` for its new utterances (and `flush()` also
  `transcript_written_all` for the session). `arc_audio.written.to_written("two thirty p.m.")` formats plain text.
- `device="auto"` picks cuda, then mps (Apple GPU), then cpu; `device="mps"` falls back to cpu with a warning when MPS is not there.
- `predict(..., events="all")` turns on experimental event labels; `events=None` turns events off.
- `predict(..., kb_snap=False)` turns off the name-list snap; `context=dict(kb=[...], kb_type=[...], dom=..., prev=...)` sets the
  decoder context for one call. The snap ignores command words and stopwords ("read", "play", "open", "the", ...): they are not
  matched to a list entry on their own, and are trimmed from the edges of a heard span ("Read Frankenstein" is matched as
  "Frankenstein").
- Names: before a name is left `not_in_list` / `abstain`, it is checked against the list's names, aliases and partial names with
  case, punctuation and spacing ignored ("DeMarco" = "De'Marco Hale", "Comets" = "Riverside Comets"); a partial name links
  only when one entry has it and the entry's kind fits the span. `alias_link=False` turns this off. `name_candidates=5` sets how many
  ranked candidates each record carries (0 = none).
- `masks` (predict and stream): per-domain output masks. By default (`masks="default"`) the `assistant` and `assistant_noisy`
  domains leave out profanity (`safety.profanity.*`, experimental) and bio events (`bio_events`, `bio_ev.*` audio events), even with
  `include_experimental=True` / `events="all"`, because they fire on clean command speech. `masks=None` masks
  nothing; `masks=["bio."]` masks a list of tag prefixes in this call; `masks={"assistant": [...], "talk": [...]}` replaces the
  table. `arc.output_masks(domain)` shows what is masked; `arc.tags(domain, masks=...)` lists the tags that remain. Raw `frames`
  probability arrays for the bio head are not masked.

### Device-directed: which number

Arc gives two device-directed numbers. They come from different read-outs and are on different scales, so they disagree on the
same clip and must not be compared with each other or mixed.

| field | what it is | scale and threshold |
|---|---|---|
| `utterances[].tags["discourse.assistant.device_directed"]` (and `live["utt"]["device_directed"]` while the utterance is open) | an utterance tag from the Arc network, read over Arc's own utterance (closed after 0.5 s without speech) | a probability of the tag after the per-domain prior shift (`calibrated=True`); no separate fit on held-out data; no fixed threshold in the package |
| `transcript[].device_directed` `{p, directed, threshold, source: "head", status}` | the device-directed head: a linear read-out of the speech features, averaged over the transcript utterance | raw score; `directed = p >= threshold` with the model's threshold (Arc Mini 0.5, status `beta`; Arc Transcriber 0.85, status `experimental`) |

**Use the utterance tag** as the device-directed score: it is the same number on every runtime (the C library's `device_directed`;
its `directed_head` is the head), it is in the live view before the utterance ends, and it is one of the model's shipped tags. Treat `transcript[].device_directed`
as a raw auxiliary score. Neither number is calibrated as a probability for always-listening use, and on multi-talker speech (people
talking to each other near the device) neither separates commands from other speech well; for now, require the device's name in
the utterance as well as a score above your own threshold, chosen on your own recordings.
### Written form and scoring

Arc's decoder writes numbers as they are spoken, and tags each one with its kind; the written form is built from those tags, not by a
second model. Rules: only number spans change; a span whose value is not certain from its words keeps its words ("two thirty" with no
time tag, no "at" and no "p.m."; "twelve fifty" with no currency); single-digit counts, durations and ordinals stay words ("two kids",
"first"), as in common style guides (`spell_small=False` in `arc_audio.written` writes them as digits). Words Arc already wrote as
digits are left as they are.

**What to score.** For word error rate against written references, or any check that looks for digits, score
`" ".join(u["text"] for u in out["transcript_written"])` (or `arc-transcribe --written --format txt`). The spoken-form `transcript`
is right for references written as words. With Whisper's `EnglishTextNormalizer` the two give nearly the same WER on plain speech,
but the normalizer does not undo every spoken form ("two thirty" -> "230", "March third" -> "march 3rd", while "2:30" -> "2 30"
and "March 3" -> "march 3"), so number-heavy sets score better on `transcript_written`.

### External ASR words (optional)

Arc can read the words of an external ASR service as its transcript input; it still listens to the audio for everything else. The
words are grouped into utterances with the same audio-based rule as the built-in transcript.

| `--asr` | needs (environment variables) | modes |
|---|---|---|
| `deepgram` | `DEEPGRAM_API_KEY` | file (pre-recorded REST) or `--live` (websocket) |
| `verbit` | `VERBIT_WS_URL` (websocket URL of a booked live order) and `VERBIT_API_KEY` | live only |
| either, with `--demo` | nothing (no network): messages are simulated from Arc's own transcript | file or `--live` |

Live modes with a real service need `pip install "arc-transcriber[live]"` (websockets); `--mic` needs it too (sounddevice). In Python:
`arc_transcriber.asr.deepgram_file`, `DeepgramLive`, `VerbitLive`, `deepgram_live_words`, `verbit_words`, then
`arc.predict(wav, transcript="external", words=words)` or `stream.add_words(words)`. The adapters are tested offline with recorded
message fixtures (`tests/test_asr_adapters.py`).

### Cloud streaming recognisers (Deepgram, AssemblyAI, Speechmatics)

[`arc_audio/integrations/`](arc_audio/integrations/README.md) streams the audio to a cloud recogniser and merges its
words into an Arc EXTERNAL stream (Arc's tags, directedness, end of turn and acts on the provider's words), with
finalise-at-end, reconnects, timeouts and a fallback to Arc's own decoder:

```bash
pip install "arc-transcriber[integrations] @ git+https://github.com/WhissleAI/arc_transcriber"
export DEEPGRAM_API_KEY=...          # or ASSEMBLYAI_API_KEY, SPEECHMATICS_API_KEY
python -m arc_audio.integrations.deepgram clip.wav
```

| provider | key | notes |
|---|---|---|
| Deepgram nova-3 | `DEEPGRAM_API_KEY` | interim results, endpointing, `Finalize` at the end of the input |
| AssemblyAI Universal Streaming | `ASSEMBLYAI_API_KEY` | per-word finals inside a turn, `ForceEndpoint` at the end; billed by session time |
| Speechmatics real-time | `SPEECHMATICS_API_KEY` | `max_delay` 0.7 s, `ForceEndOfUtterance` at the end; EU endpoint by default |

## Arc Transcriber Space (NFT2)

**Arc Transcriber Space** is an Arc model for voice apps about the Moon and space flight: questions to a lunar map or a mission
guide, crew and mission-control audio. It runs on the `nft2` backbone, a streaming FastConformer whose RNNT decoder writes the
final words, with the Arc layer (Arc decoder, Arc net, event and device-directed heads) on the same encoder. It links what is
said to the **names list** your app sends (craters and other lunar features, sites, spacecraft, missions, crew, planets, ...) and
tags those mentions as `entity.mention.lunar_feature`, `entity.mention.spacecraft`, `entity.mention.mission`,
`entity.mention.celestial`, `entity.mention.person`, .... The encoder runs on CPU with onnxruntime (int8, streaming export).

- **Model weights and model card:** https://huggingface.co/WhissleAI/arc-transcriber-space (downloaded on first use, the `nft2/`
  folder; CC BY-NC 4.0 with the Whissle terms, commercial licences: ksingla@whissle.ai)
- **Names lists:** [docs/NAMES_LISTS.md](docs/NAMES_LISTS.md): entry format, kinds, normalization, aliases, ordering.
- English, 16 kHz mono.

### Install

```bash
pip install "arc-transcriber[space] @ git+https://github.com/WhissleAI/arc_transcriber"   # + onnxruntime, onnx, sentencepiece
python examples/quickstart_space.py                                                     # offline + streaming on the model repo's sample clips
```

`Arc.from_pretrained("WhissleAI/arc-transcriber-space", backbone="nft2")` downloads only the `nft2/` folder; a local folder
works too (the repo root, or the `nft2/` folder itself). The ONNX encoder is the default; `encoder="torch"` with a local NFT2
`.nemo` (`ARC_NFT2_NEMO=/path/model.nemo`) runs the encoder in torch instead.

### Offline

```python
from arc_audio import Arc, load_audio

arc = Arc.from_pretrained("WhissleAI/arc-transcriber-space", backbone="nft2", device="cpu", threads=4)
NAMES = [
    {"name": "Tycho", "kind": "lunar_feature"},
    {"name": "Sea of Tranquility", "kind": "lunar_feature", "aliases": ["Mare Tranquillitatis"]},
    {"name": "Apollo 12", "kind": "mission"},
    {"name": "Orion", "kind": "spacecraft"},
    {"name": "Christina Koch", "kind": "crew"},
]
out = arc.predict(load_audio("question.wav"), domain="assistant", transcript="nft2", names=NAMES)
for n in out["names"]:
    print(n["heard"], n["tag"], n["entity"], n["state"])     # Tycho entity.mention.lunar_feature Tycho linked
```

`transcript="nft2"` is the RNNT transcript of the backbone, biased toward the names list (`bias_names=False` turns the biasing
off). Mentions the decoder tags that are not in the list are reported with `state: "not_in_list"`.

### Streaming

```python
s = arc.stream(domain="assistant", transcript="nft2", names=NAMES)
wav = load_audio("session.wav")
for i in range(0, len(wav), 1600):                 # 100 ms chunks of float32 mono 16 kHz audio
    new = s.feed(wav[i:i + 1600])                  # finals of the utterances closed so far: transcript, names, numbers, ...
    if screen_changed:
        s.set_context(names=NEW_NAMES)             # takes effect for the utterances that close after the call
final = s.flush()
s.close()
```

Utterances close on a 0.4 s pause (or at 15 s); each final is the RNNT words of the utterance with the decoder's tags and the
linked names. For partials, early entities, end of turn and echo handling on CPU use `arc.live_stream` (below).

Note: the model is built not to transcribe the app's own spoken answers (text-to-speech playback leaking into the microphone),
so synthetic TTS voices, especially female Kokoro-style voices, can be partly or fully ignored; test with real voices, not TTS.
Real children's speech is transcribed normally.

### Live events, echo and barge-in

`arc.live_stream(...)` decodes the open utterance as it arrives (one 560 ms encoder chunk at a time, nothing decoded twice)
and returns events with each feed. It is the Python form of libarc's `arc_nft2_stream_*` with `options.live`.

```python
ls = arc.live_stream(names=NAMES, context={"dom": "space"}, almost_names=True)
ls.feed_reference(tts_pcm, t=0.0)                  # optional: what the device plays, t = stream time of its first sample
for i in range(0, len(wav), 1600):
    out = ls.feed(wav[i:i + 1600])                 # {"utterances": [closed utterances], "events": [...]}
    for e in out["events"]:
        ...                                        # e["type"]: partial / entity / eot / barge_in / final
ls.set_names(NEW_NAMES)                            # hot list: the open utterance's bias switches at once
ls.end_turn()                                      # close the open utterance now (push-to-talk release)
ls.flush()
```

| event | when | fields |
|---|---|---|
| `partial` | the open utterance's words changed | `text`, `n_stable` (leading words no later partial changes), `start`, `end` |
| `entity` | a list name, a sound-alike or a quantity appears in the stable words | `kind` (`name` / `almost` / `quantity`), `text`, `entity`, `kb_index`, `p`; quantities `qtype`, `value`, `unit`, `relative` |
| `eot` | every feed while an utterance is open, and at each final | `p`: end-of-turn probability from the pause so far and the last word ("go to the" waits longer than "roger") |
| `barge_in` | speech that the playback does not explain while the device plays (needs `feed_reference`) | `p` = 1 - playback_leak of the last 0.3 s |
| `final` | an utterance closed | `text`; `kind` `playback` (or `dropped` with `playback_drop=True`) when its `playback_leak` is at least 0.7; `p` = playback_leak |

With a reference, a partitioned-block frequency-domain NLMS echo canceller (512 ms echo path) runs on the microphone, and each
utterance gets `playback_leak` (0-1, the share of its energy the playback explains) and `playback`. `echo_decode=True` decodes
the finals from the echo-cancelled audio. `stream.feed_reference(pcm, t)` on an ordinary `arc.stream(...)` adds
`playback_leak` / `playback` to its transcript utterances. Align `t` with the time the audio reaches the speaker; the canceller
learns up to 512 ms of further delay. It needs about a second of playback before its scores are reliable.

### Per-word confidence, alternatives, quantities and sound-alike names

`transcript="nft2"` utterances also carry:

| key | content |
|---|---|
| `words[i].conf` | the RNNT's probability of the word's least likely token, unbiased (a word written because of the names bias reads low) |
| `words[i].alts` | other spellings heard for the word `[{text, p}]`: the runner-up token, and with `nbest=4` the words other beam hypotheses have there |
| `words[i].disfl` | `filler` / `repeat` / `fragment`; `clean_text` is the text without them |
| `alts` | with `predict(..., nbest=4)`: other texts of the utterance `[{text, p}]` from an RNNT beam with the same names boosting (the words stay the greedy decode's) |
| `quantities` | typed values in the words as said: `{type: percent / distance / speed / duration / degrees / temperature / mass / power / energy / voltage / current / multiplier / number, value, unit, relative, word_begin, word_end, text, written}` ("give rover 45 more per cent" -> percent 45, relative +1); words inside list names are skipped |
| `conf`, `n_dropped` | mean word confidence; words the guard removed |

The guard (on by default; `predict(..., guard=False)` turns it off) removes words where the audio has no speech-level energy,
low-confidence repeats ("you you you") and low-confidence words in the trailing silence. `predict(..., almost=True)` adds names
records with `state: "almost"` for heard words that sound like a list entry but are not one of its spellings: `entity` stays
null, `candidate` is the entry and `candidate_p` its probability. Only uncertain words (confidence below 0.7) that are not common
vocabulary qualify, so the app can ask "did you mean ...?" rather than act.

### Room read-outs (act, addressee, safety, playback)

Packages with a `room` section in `nft2/config.json` (Arc Transcriber Space 2.0) add a small network on the same encoder frames,
the microphone audio and, when given, the device's own playback. `predict(..., transcript="nft2")` gives every transcript
utterance a `room` read-out; `arc.live_stream(...)` adds it to each final (using the audio from `feed_reference` as the
playback); `arc.room(audio, reference=None)` reads one turn and also returns the 80 ms frame tracks.

```python
r = arc.room(turn_wav, reference=tts_wav_aligned)   # reference: the playback, sample-aligned with turn_wav (or None)
r["act"]["label"]             # question / command / confirmation / chatter / statement
r["end_of_turn"]              # end-of-turn probability at the end of the audio
r["playback"], r["media"]     # share of frames that are the app's own playback / other loudspeaker speech
r["addressee"]                # BETA: device / other_person / reading_aloud
r["safety"]                   # BETA: probs per class, above_threshold = classes over the packaged thresholds
r["frames"]                   # t, labels (app_playback, media_playback, live_speech, end_of_turn, room_tone), probs, addressee, safety
arc.room_notes()              # what the BETA outputs are not yet reliable for
```

**BETA outputs.** `addressee` and `safety` carry `"beta": true`. The addressee read-out is not yet reliable for room talk
(people talking to each other near the device) or reading aloud. **The safety read-out is not a safety system**: on
children's voices it misses more than half of unsafe turns in our tests and marks some harmless ones. Never use it as the only check or
for any safety-critical or child-protection decision; keep your own safety checks (for example Arc Tagger, below) and treat it
as a hint. `predict(..., room=False)` / `live_stream(..., room=False)` turn the read-outs off. The network is also a plain ONNX
graph (`nft2/room.onnx`), so other runtimes can call it through ONNX Runtime directly: inputs `tap` (1, T, 512: the encoder
output after layer 9 without its first frame, the frames the Arc layer reads), `mic` and `ref` (1, L: 16 kHz audio, `ref`
sample-aligned, zeros when there is no playback), `ref_on` (1: 1.0 with a reference, else 0.0); outputs `frames` (1, T, 5),
`addressee` (1, T, 3), `act` (1, 5), `safety` (1, 15), `safety_frames` (1, T, 14), all probabilities, labels and
thresholds in `config.json` `room`.

### Arc Tagger (optional)

`arc.set_tagger("/path/to/arc-tagger-c/model", lib="/path/to/libarctag.so", domain="kids")` runs Arc Tagger (arc-tagger-c,
native backend) on each NFT2 final: utterances get `tagger`, its JSON read-out (entities, numbers, PII, safety, intent,
dialogue act, emotion). libarc does the same when built with `make ARC_WITH_ARCTAG=/path/to/arc-tagger-c`
(`arc_nft2_set_tagger`, `arc_utt.tagger_json`).

### Context and names lists

The names list is used three ways: it biases the RNNT words toward list names, the first ~10-15 entries go into the Arc decoder's
prompt (with their kinds) so it can type each mention, and the linker matches mentions to the whole list. Set it per call
(`names=`), for an `Arc` (`arc.set_context(names=..., topic=..., previous_text=...)`), or mid-stream
(`stream.set_context(names=...)`, which on an `nft2` stream replaces the bias, the prompt and the linking list together;
`stream.set_names(names)` replaces only the bias and the linking list).

Lists must follow the normalization in [docs/NAMES_LISTS.md](docs/NAMES_LISTS.md): always give a `kind`, put what is on screen
first, keep lists short, add aliases for names that are not said as written ("Artemis II" -> "Artemis 2") and ASCII aliases for
non-ASCII names. Mentions made only of generic words ("Mare", "North crater") are not linked unless they match an entry exactly
(`state: "abstain"`: ask the user).

### Output

The same schema as the other Arc models ([Output](#output)); the fields an app usually reads:

| key | content |
|---|---|
| `transcript` | utterances `{start, end, text, words, tags, source: "nft2", device_directed}`; `text` in spoken form ("Apollo twelve"), cased from the decoder; `words` `{text, start, end, tag, tag_begin}`; `tags` `[{tag, start, end, text}]` with the decoder's tag names (`ENTITY_LUNAR`, `ENTITY_SPACECRAFT`, `ENTITY_MISSION`, `ENTITY_CELESTIAL`, `ENTITY_PERSON`, ...) |
| `names` | `heard` (the words heard), `tag` (`entity.mention.lunar_feature` / `spacecraft` / `mission` / `celestial` / `person` / `place` / ...), `entity` (the list entry's `name`, or null), `state` (`linked` / `abstain` / `not_in_list`), `p`, `start`, `end`, `candidates` |
| `numbers` | `heard`, `kind`, `value`, `start`, `end` |
| `transcript_written` | the utterances in written form ("Apollo 12", "Ocean of Storms"): numbers as digits, linked names in your list's spelling |
| `audio_events`, `utterances` | sound / channel events; per-utterance acts and intents, `device_directed` |

Use `names[i].entity` to display and route, not the raw transcript words.

### Results

The app path (int8 ONNX encoder on CPU, this package); names list per clip = its own names plus 20 similar-sounding decoys.
Full numbers, room read-outs and the BETA notes are on the [model card](https://huggingface.co/WhissleAI/arc-transcriber-space).

| set | WER | names linked correctly | wrong links | other |
|---|---|---|---|---|
| Artemis II crew audio (1,141) | 12.3 % | 94.7 % | 0.8 % | name recall 93 %, space-tag F1 .89 |
| synthetic space questions (300) | 4.7 % | 91.4 % | 0.5 % | space-tag F1 .83 |
| Apollo mission audio (500) | 15.7 % | | | 0.2 false links per 100 utterances |
| real children's read speech (300) | 18.0 % | | | 0.67 false links per 100 utterances |
| Common Voice 17 English (2,000) | 14.9 % | | | |

CPU, int8 ONNX encoder: encoder + RNNT real-time factor about 0.13 on 4 threads, peak RAM about 2.3 GB.

## C library (Arc Mini)

[`c/`](c/README.md) holds **libarc**, a dependency-free C11 library (one header, `c/include/arc.h`) that runs Arc Mini from
the same release folder without Python: transcript with inline tags and word times, linked names, numbers with values,
utterance tags and sound events, checked against arc_audio (`make -C c parity MINI=/path/arc-v1.1/mini`). Arc Transcriber
(full) is not supported there yet.

It also streams (`arc_stream_*`): Arc's own transcript, heads only (tags, events and 100 ms frames without the decoder),
or another recogniser's words merged with Arc's tags. [`raspberry-pi/`](raspberry-pi/README.md) and
[`macos/`](macos/README.md) (Apple's on-device recogniser as the transcriber) have builds and examples.

Raspberry Pi 5, no cooler, 40 LibriSpeech test-clean clips, 100 ms chunks (latencies p50 after the end of speech):

| stream | threads | RTF | peak RSS | final | settled tags |
|---|---|---|---|---|---|
| heads only | 1 | 0.26 | 66 MB | 0.32 s (endpoint) | 1.02 s |
| full, partials every 1 s | 4 | 0.37 | 136 MB | 0.69 s | 1.01 s |
| full, no partials | 4 | 0.24 | 129 MB | 0.71 s | 1.01 s |

macOS, 100 LibriSpeech test-clean clips (latency and CPU: to be measured):

| transcript | WER clean | WER noisy (10 dB) |
|---|---|---|
| Apple SpeechTranscriber + Arc heads | 1.2 % | 2.2 % |
| Arc full | 6.2 % | to be measured |

## Tests

```bash
pip install "arc-transcriber[test] @ git+https://github.com/WhissleAI/arc_transcriber"
pytest -q                      # CPU, Arc Mini; downloads WhissleAI/arc-v1.1 on first use, skipped when offline
ARC_TEST_PARAKEET=1 pytest -q  # also Arc Transcriber (full) (ARC_MODEL=/local/arc-v1.1 to use a local copy of the model repo)
pip install -e ".[space,test]" && ARC_SPACE_MODEL=/local/arc-transcriber-space pytest -q tests/test_space.py   # Arc Transcriber Space (skipped without the model)
pytest -q tests/test_integrations.py   # cloud recognisers: recorded messages and a local mock server (no keys)
ARC_LIVE_ASR=deepgram,assemblyai,speechmatics pytest -q tests/test_integrations.py   # + live, with the keys set
pytest -q tests/test_written.py   # written form: text rules anywhere; spoken clips with macOS `say`; GPU feature path on MPS / CUDA
python scripts/check_parity.py --backbone mini --candidate out.json      # check an exported model against the golden outputs
python scripts/check_parity.py --run --backbone full -o full.json --model /local/arc-v1.1   # write a candidate with arc_audio
```

## License

Code: CC BY-NC 4.0 plus the Whissle terms in [LICENSE](LICENSE), non-commercial use only. Weights (downloaded from Hugging
Face): Arc weights CC BY-NC 4.0 under the same terms; the modified Moonshine Streaming tiny model MIT; Arc Transcriber (full)
uses NVIDIA Parakeet-TDT 0.6B v2 (CC-BY-4.0), downloaded from NVIDIA's repository. See [NOTICE](NOTICE).

- **Commercial use** (in a product or service, paid or hosted services, business use, or commercial use of outputs) needs a
  separate written licence from Whissle Inc.: ksingla@whissle.ai, https://whissle.ai.
- Whissle Inc. owns Arc and keeps all rights not expressly granted; the public licences do not limit its own use of Arc or its
  right to license Arc on other terms. No patent or trademark rights are granted; "Whissle", "Whissle Arc", "Arc Mini" and
  "Arc Transcriber" are Whissle Inc. trademarks.
- Contributions are welcome under [CONTRIBUTING.md](CONTRIBUTING.md) (includes a licence grant that lets Whissle Inc. relicense them).
