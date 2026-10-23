# arc_transcriber

A command-line and Python transcriber built on **Arc v1**, a streaming audio understanding model. Arc listens to the audio and,
every 100 ms, reports a transcript plus **names** linked to your own name list, **numbers** with their values (scores, clocks,
money, dates, ...), **speakers** and their roles, and tags and sound events. Transcript words can come from Arc's built-in decoder or
from an external ASR service (Deepgram, Verbit); Arc always reads the audio itself.

- **Model weights, model card and evaluation results:** https://huggingface.co/WhissleAI/arc-v1 (weights download on first use)
- **This repo:** the inference code (`arc_audio` 1.0.1) and the `arc-transcribe` CLI, CC BY-NC 4.0 (non-commercial). Bugs go in [Issues](../../issues).
- English, 16 kHz mono (other rates are resampled). CPU or GPU.

## Install

```bash
pip install "arc-transcriber @ git+https://github.com/WhissleAI/arc_transcriber"
pip install "arc-transcriber[live] @ git+https://github.com/WhissleAI/arc_transcriber"     # + websockets, sounddevice (live / mic)
```

The first run downloads `WhissleAI/arc-v1` at the revision `v1.0.1` (about 550 MB, model files only) into the Hugging Face cache.
The model repository asks you to accept its terms (approval is automatic): open https://huggingface.co/WhissleAI/arc-v1 while logged
in, accept, then `hf auth login` (or set `HF_TOKEN`). Python >= 3.10, torch >= 2.1, transformers >= 5.19.

## CLI

```bash
arc-transcribe game.wav --domain nba --names roster.txt                        # built-in transcript, text with inline tags
arc-transcribe game.wav --domain nba --names roster.txt --format json -o game.json
arc-transcribe game.wav --format srt -o game.srt                               # or --format vtt
arc-transcribe game.wav --live                                                 # stream the file through Arc at real-time pace
arc-transcribe --mic --domain talk                                             # microphone (needs sounddevice), Ctrl-C to stop
arc-transcribe game.wav --device cuda                                          # GPU

export DEEPGRAM_API_KEY=...                                                    # keys come from the environment only
arc-transcribe game.wav --asr deepgram --names roster.txt                      # Deepgram pre-recorded REST -> Arc
arc-transcribe game.wav --asr deepgram --live                                  # Deepgram live websocket, real-time pace
export VERBIT_WS_URL=... VERBIT_API_KEY=...                                    # websocket URL + token of a booked Verbit live order
arc-transcribe game.wav --asr verbit                                           # Verbit realtime captions (always live-paced)

arc-transcribe game.wav --asr deepgram --demo                                  # no key, no network: simulated Deepgram messages
arc-transcribe game.wav --asr verbit --demo                                    # simulated Verbit captions
```

Try it on the bundled synthetic clip (fictional names): `arc-transcribe examples/example_synth.wav --domain nba --names examples/example_names.txt`.

| option | |
|---|---|
| `--domain` | `nba` `nhl` `nfl` `cricket` `soccer` `esports` `news` `talk` `stream_chat` `meeting` `court_appellate` `court_trial` `assistant` `assistant_noisy` `short_clips` `unknown` (default) |
| `--names` | your name list: `.txt` (one name per line, optionally `name<TAB>kind`, kind = person, team, organization, place, ...) or `.json` (strings or `{"name", "kind", "aliases", "group"}`) |
| `--asr` | `builtin` (default), `deepgram`, `verbit` |
| `--device` | `cpu` (default) or `cuda` |
| `--format` | `txt` (default), `json`, `srt`, `vtt`; `-o FILE` (default stdout) |
| `--live`, `--mic`, `--duration S`, `--fast` | live mode from a file at real-time pace, or from the microphone |
| `--strict` | external words only: turn off the audio fallback for names (half the compute) |
| `--demo` | simulate the Deepgram / Verbit messages from Arc's own transcript (no key, no network) |
| `--model`, `--revision` | another Hugging Face repo / local folder, or revision (default `WhissleAI/arc-v1` at `v1.0.1`) |

Live progress (transcript lines, name and number records as they are decided) goes to stderr; the result goes to stdout or `-o`.

## Output

`--format txt` prints one line per utterance, with the speaker and inline tags after each name / number:

```
[   2.40-   7.00] S1: [Marcus Delaney](name: Marcus Delaney) brings the ball up the floor fourth quarter, [three minutes](duration: 180 s) left.
[   8.70-   9.90] S1: Delaney to Theo [Brandt.](name: Theo Brandt)
[  15.40-  19.40] S1: That puts the Harbor City Hawks ahead, [102 to 98.](number: 102-98)
[  20.90-  24.40] S1: [Brand](name) has [26 points](stat: 26) tonight, and the crowd is on its feet.
```

- `[words](name: Entity)`: a name linked to your list, in the list's spelling; `[words](name)`: a name that is not in your list.
- `[words](kind: value)`: a number with its kind (score, clock, time_of_day, date, duration, money, percent, ordinal, count, stat,
  measure, ...) and value (pairs such as scores as `102-98`, times of day as `HH:MM`, durations in seconds).

`--format srt` / `vtt` gives the same lines as subtitle cues. `--format json` gives Arc's full records plus `lines`:

| key | content |
|---|---|
| `lines` | `{start, end, speaker, text, tagged}` |
| `transcript` | utterances `{start, end, text, words}` (built-in) or the external words `{start, end, text}` |
| `names` | one per name mention: `heard`, `entity` (your list's spelling) or null, `state` (`linked` / `not_in_list` / `abstain`), `p`, `start`, `end`, `via` (`exact` = spelled exactly in the external transcript, `fallback` = from Arc's audio path) |
| `numbers` | `heard`, `kind`, `value` (+ `value2` for pairs), `unit`, `start`, `end` |
| `speakers`, `diarization` | per-speaker role and tags; `{speaker, start, end}` segments |
| `events`, `utterances`, `spans`, `segments`, `bio_events` | events (speaker change, ...), per-utterance acts / intents, word-span tags, segment labels, cough / breath / throat clear |

## Python

```python
import os
from arc_audio import Arc, load_audio

arc = Arc.from_pretrained("WhissleAI/arc-v1", revision="v1.0.1", device="cpu", threads=4)
names = ["Marcus Delaney", {"name": "Theo Brandt", "aliases": ["Brandt"]}, {"name": "Harbor City Hawks", "kind": "team"}]

# 1. built-in transcript: offline, then live
out = arc.predict("game.wav", domain="nba", names=names, transcript="builtin")
print(out["names"], out["numbers"])
wav = load_audio("game.wav")                        # float32 mono 16 kHz
s = arc.stream(domain="nba", names=names, transcript="builtin")
for i in range(0, len(wav), 1600):                  # 100 ms chunks
    new = s.feed(wav[i:i + 1600])                   # records that became available
final = s.flush()

# 2. Deepgram words (pre-recorded REST; DEEPGRAM_API_KEY from the environment)
from arc_transcriber.asr import deepgram_file
words = deepgram_file(wav, os.environ["DEEPGRAM_API_KEY"])
out = arc.predict(wav, domain="nba", names=names, transcript="external", words=words, name_source="word")

# 3. Verbit realtime captions: parse each websocket message and push the words into a stream
from arc_transcriber.asr import verbit_words
s = arc.stream(domain="nba", names=names, transcript="external", name_source="word", interim_source="verbit")
# in the websocket receive loop:  words, _ = verbit_words(message); s.add_words(words)
# in the audio loop (the audio you send to Verbit):  new = s.feed(chunk)
```

`arc_transcriber.asr` also has `DeepgramLive` and `VerbitLive` (async websocket clients) as used by the CLI's `--live` mode.

### External words: what Arc does with them (1.0.1)

- A name that the external transcript spells exactly as in your list (or an alias, or a surname unique in the list) is always kept
  and linked (`via: "exact"`). Other mentions go through Arc's own name linking on the external words.
- Number values are parsed from the external words' text ("25 minutes" = 1500 s, "6 p.m." = 18:00, "102 to 98" = 102-98), with
  Arc's number kind. Numbers the text shows but Arc did not detect are added from the text alone.
- Audio fallback (default): Arc's built-in transcript path runs alongside and adds the names it links where the external words gave
  no name record (`via: "fallback"`). This doubles the compute; `strict=True` (CLI `--strict`) turns it off.

Measured results (names linked right, numbers with the exact value, by transcript source) are on the
[model card](https://huggingface.co/WhissleAI/arc-v1).

## Tests

```bash
pip install "arc-transcriber[test] @ git+https://github.com/WhissleAI/arc_transcriber"
pytest -q            # CPU; downloads WhissleAI/arc-v1 on first use, skipped when offline (ARC_MODEL=/local/arc-v1 to use a local copy)
```

## License

Code: CC BY-NC 4.0 ([LICENSE](LICENSE)), non-commercial use only; commercial use requires a separate licence from
WhissleAI (https://whissle.ai). Weights (downloaded from Hugging Face): Arc weights CC BY-NC 4.0; the bundled modified
Moonshine Streaming tiny model MIT. See [NOTICE](NOTICE).
