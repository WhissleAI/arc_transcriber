# arc_transcriber

A command-line tool and Python library for **Arc**, an all-in-one streaming audio model. Give it audio (a file or live), and
optionally your list of names and some context. One model returns, every 100 ms:

- a **transcript** with **inline tags** on names, numbers and entities;
- **names linked to your own list**, and **numbers with their values** (scores, clocks, money, dates, durations, ...);
- **speakers** and their **roles**, per-utterance **acts / intents**, segment labels;
- **sound and channel events** (silence, telephone / narrowband audio, degraded audio, clipping; more labels as experimental);
- **device-directed speech** (is this utterance addressed to an assistant?);
- **context memory**: the decoder can take your names, terms, a topic, and what was said earlier.

Two models, one API and one output schema (versions as loaded by `arc-transcriber` 1.2.0):

| model | version | `--backbone` | download | runs on | use it for |
|---|---|---|---|---|---|
| **Arc Mini** | 1.1.1 | `mini` (default; alias `edge`) | about 71 MB | CPU (files at about 0.3x real time on 4 cores), GPU optional | apps, laptops, on-device, lowest latency |
| **Arc Transcriber** (full) | **1.2.0** | `full` (aliases `parakeet`, `server`) | about 77 MB + NVIDIA Parakeet-TDT 0.6B v2 encoder (about 2.5 GB, on first use) | GPU recommended, CPU for files | best transcript and name accuracy |

Coming next: **Arc Transcriber Edge**, a 4-bit version of the full model for Android, Mac and cloud.

- **Model weights, model card and evaluation results:** https://huggingface.co/WhissleAI/arc-v1.1 (downloaded on first use; the
  repository name stays `arc-v1.1`, versions are git revisions of it)
- **This repo:** inference code (`arc_audio` 1.2.0) and the `arc-transcribe` CLI, CC BY-NC 4.0 with the Whissle terms (non-commercial only; commercial licences: ksingla@whissle.ai). Bugs go in [Issues](../../issues).
- English, 16 kHz mono (other rates are resampled).

## Install

```bash
pip install "arc-transcriber @ git+https://github.com/WhissleAI/arc_transcriber@v1.2.0"
pip install "arc-transcriber[live] @ git+https://github.com/WhissleAI/arc_transcriber@v1.2.0"     # + websockets, sounddevice (live / mic)
```

The first run downloads `WhissleAI/arc-v1.1` at revision `v1.2.0` into the Hugging Face cache (only the folder of the model you
use). The model repository asks you to accept its terms (approval is automatic): open https://huggingface.co/WhissleAI/arc-v1.1
while logged in, accept, then `hf auth login` (or set `HF_TOKEN`). Arc Transcriber (full) also downloads `nvidia/parakeet-tdt-0.6b-v2`
(public, CC-BY-4.0); NeMo is not needed. Python >= 3.10, torch >= 2.1, transformers >= 5.19.

### Versions

Each package version loads the model revision in this table (pass `--revision` / `revision=` to choose another). Code-only releases
keep the previous model revision:

| `arc-transcriber` | model revision | Arc Mini | Arc Transcriber (full) |
|---|---|---|---|
| 1.2.1 | `v1.2.0` | 1.1.1 | 1.2.0 |
| 1.2.0 | `v1.2.0` | 1.1.1 | **1.2.0** |
| 1.1.2 | `v1.1.2` | 1.1.1 | 1.1.1 |
| 1.1.1 | `v1.1.1` | 1.1.1 | 1.1.1 |
| 1.1.0 | `v1.1.0` | 1.1.0 | 1.1.0 |

Benchmarks and the comparison of Arc Transcriber 1.2.0 with 1.1.1 are on the
[model card](https://huggingface.co/WhissleAI/arc-v1.1). Changes per version: [CHANGELOG.md](CHANGELOG.md).

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
| `--model`, `--revision` | another Hugging Face repo / local folder, or revision (default `WhissleAI/arc-v1.1` at `v1.2.0`) |

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
| `transcript` | utterances `{start, end, text, words, tags, device_directed: {p, directed}}`: the text as spoken ("two thirty", "forty five dollars") |
| `transcript_written` | the same utterances in written form, the text to show: numbers from Arc's number tags as digits ("2:30 p.m.", "$45", "€12.50", "March 3, 2024", "12.5%", "21st", "555-1234", "3–1" in sports domains) and linked names in your list's spelling. A rewritten span is one word `{text, start, end, written_from, kind}` |
| `transcript_rewritten` | (older) linked names replaced by the full entity name; numbers unchanged |
| `names` | `heard`, `tag`, `entity` (your list's spelling) or null, `state` (`linked` / `not_in_list` / `abstain`), `start`, `end`, `via` (`snap` = filled by the name-list snap) |
| `numbers` | `heard`, `kind`, `value` (+ `value2` for pairs), `unit`, `start`, `end`; dates also `date: {year, month, day}` and `iso` (`"--03-03"` for "March third", `"2024-03-03"`, `"2024"`) |
| `speakers`, `diarization` | per-speaker role and tags; `{speaker, start, end}` segments |
| `utterances`, `segments`, `spans`, `events` | per-utterance acts / intents, segment labels, word-span tags, instant events (speaker change, ...) |
| `audio_events` | `{label, start, end, p, status}` (status `stable` / `beta` / `experimental`) |
| `bio_events` | cough / throat clear / breath |

## Python

```python
from arc_audio import Arc, load_audio

arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone="mini", revision="v1.2.0", device="cpu", threads=4)  # backbone="full": Arc Transcriber
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
s.set_context(kb=["Theo Brandt"], kb_type=["person"])         # context can change at any time
final = s.flush()
```

- `predict(...)["transcript_written"]` is the written form; `arc.transcribe(wav, written=True)` adds `text_written` / `words_written` to
  each utterance; each `stream.feed()` / `flush()` result has `transcript_written` for its new utterances (and `flush()` also
  `transcript_written_all` for the session). `arc_audio.written.to_written("two thirty p.m.")` formats plain text.
- `device="auto"` picks cuda, then mps (Apple GPU), then cpu; `device="mps"` falls back to cpu with a warning when MPS is not there.
- `predict(..., events="all")` turns on experimental event labels; `events=None` turns events off.
- `predict(..., kb_snap=False)` turns off the name-list snap; `context=dict(kb=[...], kb_type=[...], dom=..., prev=...)` sets the
  decoder context for one call.
### Written form and benchmarks

Arc's decoder writes numbers as they are spoken, and tags each one with its kind; the written form is built from those tags, not by a
second model. Rules: only number spans change; a span whose value is not certain from its words keeps its words ("two thirty" with no
time tag, no "at" and no "p.m."; "twelve fifty" with no currency); single-digit counts, durations and ordinals stay words ("two kids",
"first"), as in common style guides (`spell_small=False` in `arc_audio.written` writes them as digits). Words Arc already wrote as
digits are left as they are.

**What to score.** For word error rate against written references, or any check that looks for digits, score
`" ".join(u["text"] for u in out["transcript_written"])` (or `arc-transcribe --written --format txt`). The spoken-form `transcript`
is right for references written as words. With Whisper's `EnglishTextNormalizer` the two give nearly the same WER on plain speech,
but the normalizer does not undo every spoken form ("two thirty" -> "230", "March third" -> "march 3rd", while "2:30" -> "2 30"
and "March 3" -> "march 3"), so number-heavy sets score better on `transcript_written`. Measured in
[`evals/written_form`](evals/written_form/).

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

## What's new in 1.2.1

- **Written form**: `transcript_written` in `predict()` and the stream, `transcribe(written=True)`, `arc-transcribe --written`.
  Numbers become digits from Arc's own number tags ("two thirty p.m." -> "2:30 p.m.", "forty five dollars" -> "$45", "March third"
  -> "March 3"); linked names take your list's spelling without adding words that were not said.
- **Date records keep the month**: a `numbers` record for "March third" now has `heard` "March third", `date` `{month: 3, day: 3}`
  and `iso` `"--03-03"` (it was `heard` "third", value 3). `value` keeps its meaning (the day, or the year when only a year was said).
- **Apple GPU**: `device="mps"` works for Arc Mini (same outputs as CPU: `tests/golden/mini.json` passes on MPS) and for Arc
  Transcriber's feature path (it failed with "Passed CPU tensor to MPS op"); `device="auto"` picks cuda > mps > cpu.
- Model revision unchanged (`v1.2.0`); same weights, same golden outputs.

## What's new in 1.2.0

- **Arc Transcriber 1.2.0** (`--backbone full`): lower word error rate (15.7 % -> 13.7 % on our held-out multi-domain dev set),
  more names and numbers found, better assistant intents and diarization, fewer repetition loops; device-directed default
  threshold 0.85 (was 0.50) and re-calibrated audio-event thresholds (safety labels set for a low false-alarm rate). Short assistant
  commands are transcribed slightly worse than by 1.1.1. Details and numbers: [model card](https://huggingface.co/WhissleAI/arc-v1.1).
- **Arc Mini is unchanged** (1.1.1). Code behaviour, API and output schema are unchanged; the default model revision is `v1.2.0`.
- Golden reference outputs for Arc Transcriber (`tests/golden/full.json`) match 1.2.0; `tests/golden/mini.json` is unchanged.

Earlier versions: [CHANGELOG.md](CHANGELOG.md).

## C library (Arc Mini)

[`c/`](c/README.md) holds **libarc**, a dependency-free C11 library (one header, `c/include/arc.h`) that runs Arc Mini from
the same release folder without Python: transcript with inline tags and word times, linked names, numbers with values,
utterance tags and sound events, checked against arc_audio (`make -C c parity MINI=/path/arc-v1.1/mini`). Arc Transcriber
(full) is not supported there yet.

## Tests

```bash
pip install "arc-transcriber[test] @ git+https://github.com/WhissleAI/arc_transcriber@v1.2.0"
pytest -q                      # CPU, Arc Mini; downloads WhissleAI/arc-v1.1 on first use, skipped when offline
ARC_TEST_PARAKEET=1 pytest -q  # also Arc Transcriber (full) (ARC_MODEL=/local/arc-v1.1 to use a local copy of the model repo)
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
