# Using arc_transcriber

The CLI, the Python API and the output schema for Arc Mini and Arc Transcriber (full). Arc Transcriber Space has its
own page: [SPACE.md](SPACE.md).

## Install

```bash
pip install git+https://github.com/WhissleAI/arc_transcriber
pip install "arc-transcriber[live] @ git+https://github.com/WhissleAI/arc_transcriber"   # + microphone and live external ASR
```

The first run downloads `WhissleAI/arc-v1.1` into the Hugging Face cache (only the folder of the model you use). The
model repo is gated with automatic approval: accept its terms on Hugging Face while logged in, then `hf auth login` or
set `HF_TOKEN`. Arc Transcriber (full) also downloads the public `nvidia/parakeet-tdt-0.6b-v2`; NeMo is not needed.

## CLI

```bash
arc-transcribe meeting.wav                                              # Arc Mini, text with inline tags
arc-transcribe game.wav --domain nba --names roster.txt                 # link names to your list
arc-transcribe game.wav --names roster.txt --format json -o game.json   # full records
arc-transcribe call.wav --backbone full --device cuda                   # Arc Transcriber (full) on a GPU
arc-transcribe meeting.wav --written                                    # "2:30 p.m.", "$45", "March 3"
arc-transcribe talk.wav --context terms.txt --topic "product launch"    # decoder context
arc-transcribe talk.wav --auto-context                                  # feed back what was said
arc-transcribe game.wav --format srt -o game.srt                        # or --format vtt
arc-transcribe game.wav --live                                          # stream the file at real-time pace
arc-transcribe --mic --domain talk                                      # microphone; Ctrl-C to stop
```

| option | values |
|---|---|
| `--backbone` | `mini` (default) or `full`; `edge` = `mini`, `parakeet` / `server` = `full` |
| `--domain` | `nba` `nhl` `nfl` `cricket` `soccer` `esports` `news` `talk` `stream_chat` `meeting` `court_appellate` `court_trial` `assistant` `assistant_noisy` `short_clips` `unknown` (default) |
| `--names` | names list: `.txt` (one per line, optionally `name<TAB>kind`) or `.json` (strings or `{"name", "kind", "aliases", "group"}`) |
| `--context`, `--topic`, `--auto-context` | decoder context: terms (`.txt`, `.json` or `a,b,c`), a topic hint, earlier text |
| `--events` | `default` (stable + beta labels), `all` (adds experimental labels), `none` |
| `--no-snap` | turn off the fuzzy / phonetic snap of mentions to `--names` |
| `--asr` | `builtin` (default), `deepgram`, `verbit` (see [External ASR](#external-asr-words)) |
| `--device` | `cpu` (default), `cuda`, `mps` (falls back to `cpu`), `auto` (cuda > mps > cpu) |
| `--format`, `-o` | `txt` (default), `json`, `srt`, `vtt`; output file (default stdout) |
| `--written` | txt / srt / vtt show the written form instead of inline tags |
| `--live`, `--mic`, `--duration S`, `--fast` | live mode from a file or the microphone |
| `--model`, `--revision` | another Hugging Face repo or local folder; a revision of the model repo |

Progress goes to stderr; the result goes to stdout or `-o`.

## Python

```python
from arc_audio import Arc, load_audio

arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone="mini", device="cpu", threads=4)
names = ["Marcus Delaney", {"name": "Theo Brandt", "aliases": ["Brandt"]}, {"name": "Harbor City Hawks", "kind": "team"}]

out = arc.predict("game.wav", domain="nba", names=names, events="default")           # whole file

arc.set_context(names=names, terms=["pick and roll"], topic="nba", auto_prev=True)     # decoder context

wav = load_audio("game.wav")                                                           # live: 100 ms chunks
s = arc.stream(domain="nba", names=names)
for i in range(0, len(wav), 1600):
    new = s.feed(wav[i:i + 1600])            # new transcript lines, names, numbers, events, ...
s.set_context(names=["Theo Brandt"], previous_text="...")   # applies to utterances closed after the call
final = s.flush()
```

Voice-agent turns: keep one stream open and end each turn with `end_turn()` (alias `finalize()`). It closes the open
utterance, returns its final transcript, and keeps the stream open.

```python
with arc.stream(domain="assistant", names=contacts) as s:
    for chunk in turn_audio:
        s.feed(chunk)
    out = s.end_turn()                     # or end_turn(context=...) for the next turn
```

### Options

| option | where | effect |
|---|---|---|
| `segmenter=dict(pause_s, short_pause_s, min_s, max_s)` | stream, predict | utterances close on a 0.4 s pause; ones shorter than 1.0 s after 0.8 s; at most 15 s |
| `live_hold=True` | stream | hold back low-confidence first guesses in the `live` view (`held: true`) |
| `command_complete_hold_s=0.5` | stream, predict | release `command_complete` only after that much silence (default 0) |
| `switch_interval_s=True` | stream | set `sys.setswitchinterval` to 0.5 ms for the stream's life (default: unchanged) |
| `events="all"` / `None` | predict | add experimental event labels / turn events off |
| `kb_snap=False` | predict | turn off the names-list snap (command words and stopwords are never snapped) |
| `alias_link=False` | predict | turn off the last alias / partial-name check before `not_in_list` |
| `name_candidates=5` | predict | ranked candidates per name record (0 = none) |
| `masks="default"` / `None` / list / dict | stream, predict | per-domain output masks; `assistant` domains drop profanity and bio events by default |
| `device="auto"` | from_pretrained | cuda, then mps, then cpu |

`arc.output_masks(domain)` shows what is masked; `arc.tags(domain, masks=...)` lists the remaining tags.

## Output

`--format json` and `predict()` return:

| key | content |
|---|---|
| `transcript` | utterances `{start, end, text, words, tags, device_directed}`, text as spoken ("two thirty") |
| `transcript_written` | the same utterances in written form, for display: numbers as digits, linked names in your list's spelling |
| `transcript_rewritten` | linked names replaced by the entity name; numbers unchanged |
| `names` | `heard`, `tag`, `entity` (or null), `state` (`linked` / `not_in_list` / `abstain`), `start`, `end`, `via`, `candidates`, `snap_candidates` |
| `numbers` | `heard`, `kind`, `value` (+ `value2`), `unit`, `start`, `end`; dates add `date: {year, month, day}` and `iso` |
| `speakers`, `diarization` | per-speaker role and tags; `{speaker, start, end}` turns |
| `utterances`, `segments`, `spans`, `events` | per-utterance acts and intents, segment labels, word-span tags, instant events |
| `audio_events` | `{label, start, end, p, status}`; status `stable` / `beta` / `experimental` |
| `bio_events` | cough, throat clear, breath (masked in `assistant` domains) |

Each `stream.feed()` / `flush()` result has the same keys for its new utterances.

### Device-directed: which number to use

Arc gives two device-directed numbers on different scales. Do not compare or mix them.

| field | what it is |
|---|---|
| `utterances[].tags["discourse.assistant.device_directed"]` | **use this one.** An utterance tag from the Arc network, the same on every runtime (libarc's `device_directed`), available before the utterance ends |
| `transcript[].device_directed` `{p, directed, threshold, ...}` | a raw auxiliary head score; `directed = p >= threshold` (Arc Mini 0.5, beta; Arc Transcriber 0.85, experimental) |

Neither is calibrated for always-listening use, and neither separates commands from people talking to each other near
the device. For now, require the device's name in the utterance as well as a score above a threshold you pick on your
own recordings.

### Written form and scoring

The written form is built from Arc's number tags, not by a second model. Only number spans change; a span whose value
is not certain keeps its words ("two thirty" with no time cue); single-digit counts and ordinals stay words.
`arc_audio.written.to_written("two thirty p.m.")` formats plain text.

To score WER against written references, score `transcript_written` (or `arc-transcribe --written --format txt`). Use
the spoken `transcript` for references written as words.

## External ASR words

Arc can take another recogniser's words as its transcript and still read everything else from the audio.

| `--asr` | needs (environment) | modes |
|---|---|---|
| `deepgram` | `DEEPGRAM_API_KEY` | file or `--live` |
| `verbit` | `VERBIT_WS_URL`, `VERBIT_API_KEY` | live only |
| either, with `--demo` | nothing (messages simulated from Arc's own transcript) | file or `--live` |

In Python: `arc_transcriber.asr` (`deepgram_file`, `DeepgramLive`, `VerbitLive`), then
`arc.predict(wav, transcript="external", words=words)` or `stream.add_words(words)`.

For streaming cloud recognisers with reconnects and fallback (Deepgram, AssemblyAI, Speechmatics), see
[arc_audio/integrations/](../arc_audio/integrations/README.md).

## Tests

```bash
pytest -q                                                   # Arc Mini, CPU; skipped when offline
ARC_TEST_PARAKEET=1 pytest -q                               # also Arc Transcriber (full); ARC_MODEL=/local/arc-v1.1 for a local copy
ARC_SPACE_MODEL=/local/arc-transcriber-space pytest -q tests/test_space.py    # Arc Transcriber Space
pytest -q tests/test_integrations.py                        # cloud recognisers, recorded messages, no keys
ARC_LIVE_ASR=deepgram,assemblyai,speechmatics pytest -q tests/test_integrations.py   # live, with keys set
python scripts/check_parity.py --backbone mini --candidate out.json   # check a port, see PORTING.md
```

Benchmarks against other systems are on the [model card](https://huggingface.co/WhissleAI/arc-v1.1).
