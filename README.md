# arc_transcriber

A command-line tool and Python library for **Arc v1.1**, an all-in-one streaming audio model. Give it audio (a file or live), and
optionally your list of names and some context. One model returns, every 100 ms:

- a **transcript** with **inline tags** on names, numbers and entities;
- **names linked to your own list**, and **numbers with their values** (scores, clocks, money, dates, durations, ...);
- **speakers** and their **roles**, per-utterance **acts / intents**, segment labels;
- **sound and channel events** (silence, telephone / narrowband audio, degraded audio, clipping; more labels as experimental);
- **device-directed speech** (is this utterance addressed to an assistant?);
- **context memory**: the decoder can take your names, terms, a topic, and what was said earlier.

Two backbones, one API and one output schema:

| `--backbone` | download | runs on | use it for |
|---|---|---|---|
| `edge` (default) | about 71 MB | CPU (real time on 4 cores), GPU optional | apps, laptops, on-device, lowest latency |
| `parakeet` | about 77 MB + NVIDIA Parakeet-TDT 0.6B v2 encoder (about 2.5 GB, on first use) | GPU recommended, CPU for files | best transcript and name accuracy |

- **Model weights, model card and evaluation results:** https://huggingface.co/WhissleAI/arc-v1.1 (downloaded on first use)
- **This repo:** inference code (`arc_audio` 1.1.0) and the `arc-transcribe` CLI, CC BY-NC 4.0 (non-commercial). Bugs go in [Issues](../../issues).
- English, 16 kHz mono (other rates are resampled).

## Install

```bash
pip install "arc-transcriber @ git+https://github.com/WhissleAI/arc_transcriber@v1.1.0"
pip install "arc-transcriber[live] @ git+https://github.com/WhissleAI/arc_transcriber@v1.1.0"     # + websockets, sounddevice (live / mic)
```

The first run downloads `WhissleAI/arc-v1.1` at revision `v1.1.0` into the Hugging Face cache (only the folder of the backbone you
use). The model repository asks you to accept its terms (approval is automatic): open https://huggingface.co/WhissleAI/arc-v1.1
while logged in, accept, then `hf auth login` (or set `HF_TOKEN`). The parakeet backbone also downloads `nvidia/parakeet-tdt-0.6b-v2`
(public, CC-BY-4.0); NeMo is not needed. Python >= 3.10, torch >= 2.1, transformers >= 5.19.

## CLI

```bash
arc-transcribe meeting.wav                                                     # edge backbone, text with inline tags
arc-transcribe game.wav --domain nba --names roster.txt                        # link names to your list
arc-transcribe game.wav --names roster.txt --format json -o game.json          # full records (JSON)
arc-transcribe call.wav --backbone parakeet --device cuda                      # server backbone on a GPU
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
| `--backbone` | `edge` (default) or `parakeet` |
| `--domain` | `nba` `nhl` `nfl` `cricket` `soccer` `esports` `news` `talk` `stream_chat` `meeting` `court_appellate` `court_trial` `assistant` `assistant_noisy` `short_clips` `unknown` (default) |
| `--names` | your name list for linking: `.txt` (one name per line, optionally `name<TAB>kind`) or `.json` (strings or `{"name", "kind", "aliases", "group"}`) |
| `--context`, `--topic`, `--auto-context` | decoder context: terms (`.txt`, `.json` or `a,b,c`), a topic hint, and feeding back earlier text |
| `--events` | `default` (stable + beta labels), `all` (adds experimental labels), `none` |
| `--no-snap` | turn off the fuzzy / phonetic snap of mentions to `--names` |
| `--asr` | `builtin` (default), `deepgram`, `verbit` (external transcript words) |
| `--device` | `cpu` (default) or `cuda` |
| `--format` | `txt` (default), `json`, `srt`, `vtt`; `-o FILE` (default stdout) |
| `--live`, `--mic`, `--duration S`, `--fast` | live mode from a file at real-time pace, or from the microphone |
| `--model`, `--revision` | another Hugging Face repo / local folder, or revision (default `WhissleAI/arc-v1.1` at `v1.1.0`) |

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
| `transcript` | utterances `{start, end, text, words, tags, device_directed: {p, directed}}` |
| `names` | `heard`, `tag`, `entity` (your list's spelling) or null, `state` (`linked` / `not_in_list` / `abstain`), `start`, `end`, `via` (`snap` = filled by the name-list snap) |
| `numbers` | `heard`, `kind`, `value` (+ `value2` for pairs), `unit`, `start`, `end` |
| `speakers`, `diarization` | per-speaker role and tags; `{speaker, start, end}` segments |
| `utterances`, `segments`, `spans`, `events` | per-utterance acts / intents, segment labels, word-span tags, instant events (speaker change, ...) |
| `audio_events` | `{label, start, end, p, status}` (status `stable` / `beta` / `experimental`) |
| `bio_events` | cough / throat clear / breath |

## Python

```python
from arc_audio import Arc, load_audio

arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone="edge", device="cpu", threads=4)    # backbone="parakeet" for the server model
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

- `predict(..., events="all")` turns on experimental event labels; `events=None` turns events off.
- `predict(..., kb_snap=False)` turns off the name-list snap; `context=dict(kb=[...], kb_type=[...], dom=..., prev=...)` sets the
  decoder context for one call.
- External ASR words (`transcript="external"`, Deepgram / Verbit clients in `arc_transcriber.asr`) work as in 1.0.

## What's new in 1.1.0

- Two backbones: `edge` (new on-device model, 4-bit packed weights, about 71 MB download) and `parakeet` (server model on the NVIDIA
  Parakeet-TDT 0.6B v2 encoder).
- 34 inline tag classes in the transcript (new: law, product, term, phone, email, URL, address), repetition-loop guard, per-class
  tag calibration.
- Decoder context (names, terms, topic, earlier text) and context memory (`auto_prev`).
- Name-list snap post-pass (fuzzy / phonetic), sound / channel events with a status per label, device-directed speech.
- The default model is `WhissleAI/arc-v1.1`. The API is backward compatible: 1.0 calls keep working (they use the edge backbone).

## C API (coming in 1.1.x)

A dependency-free C11 library for the edge model is in preparation in [`c/`](c/). It reads the same packed weights; the format is
documented in [`docs/WEIGHTS_FORMAT.md`](docs/WEIGHTS_FORMAT.md).

## Tests

```bash
pip install "arc-transcriber[test] @ git+https://github.com/WhissleAI/arc_transcriber@v1.1.0"
pytest -q                      # CPU, edge backbone; downloads WhissleAI/arc-v1.1 on first use, skipped when offline
ARC_TEST_PARAKEET=1 pytest -q  # also the parakeet backbone (ARC_MODEL=/local/arc-v1.1 to use a local copy of the model repo)
```

## License

Code: CC BY-NC 4.0 ([LICENSE](LICENSE)), non-commercial use only; commercial use requires a separate licence from WhissleAI
(https://whissle.ai). Weights (downloaded from Hugging Face): Arc weights CC BY-NC 4.0; the modified Moonshine Streaming tiny model
MIT; the parakeet backbone uses NVIDIA Parakeet-TDT 0.6B v2 (CC-BY-4.0), downloaded from NVIDIA's repository. See [NOTICE](NOTICE).
