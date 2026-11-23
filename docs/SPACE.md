# Arc Transcriber Space (`nft2`)

An Arc model for voice apps about the Moon and space flight: questions to a lunar map or a mission guide, crew and
mission-control audio. It links what is said to the **names list** your app sends (craters, sites, spacecraft,
missions, crew, planets) and tags each mention (`entity.mention.lunar_feature`, `.spacecraft`, `.mission`,
`.celestial`, `.person`, ...).

- Backbone: a streaming FastConformer encoder (int8 ONNX, CPU) whose RNNT decoder writes the words, with the Arc layer
  on the same encoder.
- Weights and model card: https://huggingface.co/WhissleAI/arc-transcriber-space (CC BY-NC 4.0 with the Whissle terms).
- English, 16 kHz mono.
- Names lists: [NAMES_LISTS.md](NAMES_LISTS.md).

## Install

```bash
pip install "arc-transcriber[space] @ git+https://github.com/WhissleAI/arc_transcriber"
python examples/quickstart_space.py          # offline + streaming on the model repo's sample clips
```

`Arc.from_pretrained("WhissleAI/arc-transcriber-space", backbone="nft2")` downloads only the `nft2/` folder; a local
folder works too.

## Offline

```python
from arc_audio import Arc, load_audio

arc = Arc.from_pretrained("WhissleAI/arc-transcriber-space", backbone="nft2", device="cpu", threads=4)
NAMES = [
    {"name": "Tycho", "kind": "lunar_feature"},
    {"name": "Sea of Tranquility", "kind": "lunar_feature", "aliases": ["Mare Tranquillitatis"]},
    {"name": "Apollo 12", "kind": "mission"},
]
out = arc.predict(load_audio("question.wav"), domain="assistant", transcript="nft2", names=NAMES)
for n in out["names"]:
    print(n["heard"], n["tag"], n["entity"], n["state"])     # Tycho entity.mention.lunar_feature Tycho linked
```

`transcript="nft2"` is the RNNT transcript, biased toward the names list (`bias_names=False` turns biasing off).
Use `names[i].entity` to display and route, not the raw transcript words.

## Streaming

```python
s = arc.stream(domain="assistant", transcript="nft2", names=NAMES)
for i in range(0, len(wav), 1600):            # 100 ms chunks, float32 mono 16 kHz
    new = s.feed(wav[i:i + 1600])             # finals of utterances closed so far
    if screen_changed:
        s.set_context(names=NEW_NAMES)        # applies to utterances that close after the call
final = s.flush(); s.close()
```

Utterances close on a 0.4 s pause (or at 15 s). For partials, early entities, end of turn and echo handling, use
`arc.live_stream`.

The model is built to ignore the app's own spoken answers leaking into the microphone, so synthetic TTS voices can be
partly or fully ignored. Test with real voices. Real children's speech is transcribed normally.

## Live events, echo and barge-in

```python
ls = arc.live_stream(names=NAMES, context={"dom": "space"}, almost_names=True)
ls.feed_reference(tts_pcm, t=0.0)             # optional: what the device plays, and when
out = ls.feed(chunk)                          # {"utterances": [...], "events": [...]}
ls.set_names(NEW_NAMES)                       # the open utterance's bias switches at once
ls.end_turn(); ls.flush()
```

| event | when | fields |
|---|---|---|
| `partial` | the open utterance's words changed | `text`, `n_stable`, `start`, `end` |
| `entity` | a list name, a sound-alike or a quantity appears in the stable words | `kind` (`name` / `almost` / `quantity`), `text`, `entity`, `p` |
| `eot` | every feed while an utterance is open | `p`: end-of-turn probability |
| `barge_in` | speech the playback does not explain, while the device plays | `p` |
| `final` | an utterance closed | `text`; `kind` `playback` when its `playback_leak` is 0.7 or more |

With a reference, an echo canceller (512 ms echo path) runs on the microphone and each utterance gets `playback_leak`
(0-1). It needs about a second of playback before its scores are reliable. `echo_decode=True` decodes finals from the
echo-cancelled audio.

## Extra fields on `nft2` utterances

| key | content |
|---|---|
| `words[i].conf` | the RNNT's unbiased probability of the word's least likely token |
| `words[i].alts` | other spellings heard `[{text, p}]` |
| `words[i].disfl` | `filler` / `repeat` / `fragment`; `clean_text` is the text without them |
| `alts` | with `nbest=4`: other texts of the utterance `[{text, p}]` |
| `quantities` | typed values: percent, distance, speed, duration, temperature, ... with `value`, `unit`, `relative` |
| `conf`, `n_dropped` | mean word confidence; words the guard removed |

- **Guard** (on; `guard=False` turns it off): removes words with no speech energy, low-confidence repeats and
  low-confidence words in trailing silence.
- **Sound-alike names** (`almost=True`): names records with `state: "almost"`, `candidate` and `candidate_p`, so the
  app can ask "did you mean ...?".

## Room read-outs (beta)

Packages with a `room` section in `nft2/config.json` add a small network on the same encoder frames.
`predict(..., transcript="nft2")` gives each utterance a `room` read-out; `arc.room(audio, reference=None)` reads one
turn.

| field | content |
|---|---|
| `act` | question / command / confirmation / chatter / statement |
| `end_of_turn` | end-of-turn probability at the end of the audio |
| `playback`, `media` | share of frames that are the app's own playback / other loudspeaker speech |
| `addressee` | **beta**: device / other_person / reading_aloud |
| `safety` | **beta**: per-class probabilities |

**The safety read-out is not a safety system.** On children's voices it misses more than half of unsafe turns in our
tests. Never use it as the only check or for any child-protection decision; keep your own checks (for example Arc
Tagger, below). `arc.room_notes()` lists what the beta outputs are not reliable for. `room=False` turns them off. The
network is also a plain ONNX graph (`nft2/room.onnx`); its inputs, outputs and thresholds are in `config.json` `room`.

## Arc Tagger (optional)

`arc.set_tagger("/path/to/arc-tagger-c/model", lib="/path/to/libarctag.so", domain="kids")` runs
[Arc Tagger](https://github.com/WhissleAI/arc-tagger-c) on each final: utterances get `tagger` (entities, numbers, PII,
safety, intent, act, emotion). libarc does the same when built with `make ARC_WITH_ARCTAG=/path/to/arc-tagger-c`.

## Names lists

The list biases the RNNT words, its first ~10-15 entries go into the decoder prompt with their kinds, and the linker
matches mentions to the whole list. Set it per call (`names=`), per `Arc` (`arc.set_context(...)`), or mid-stream
(`stream.set_context(names=...)` replaces bias, prompt and linking list; `stream.set_names` only bias and linking).
Follow [NAMES_LISTS.md](NAMES_LISTS.md): always give a `kind`, put what is on screen first, keep lists short, add
aliases for names not said as written.

## Results

App path (int8 ONNX encoder, CPU); names list per clip = its own names plus 20 similar-sounding decoys. Full numbers
are on the [model card](https://huggingface.co/WhissleAI/arc-transcriber-space).

| set | WER | names linked correctly | wrong links |
|---|---|---|---|
| Artemis II crew audio (1,141) | 12.3 % | 94.7 % | 0.8 % |
| synthetic space questions (300) | 4.7 % | 91.4 % | 0.5 % |
| Apollo mission audio (500) | 15.7 % | | 0.2 per 100 utterances |
| real children's read speech (300) | 18.0 % | | 0.67 per 100 utterances |
| Common Voice 17 English (2,000) | 14.9 % | | |

Encoder + RNNT real-time factor about 0.13 on 4 CPU threads; peak RAM about 2.3 GB (Python).
