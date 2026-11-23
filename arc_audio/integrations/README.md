# Cloud streaming recognisers with Arc (`arc_audio.integrations`)

Arc's EXTERNAL stream takes its words from another recogniser and reads everything else from the audio: utterance
tags (device-directed, acts, intents), end of turn, names and numbers on the words, speakers, events. This folder
connects it to three cloud streaming services.

| provider | module | key (environment) |
|---|---|---|
| Deepgram (nova-3) | `arc_audio.integrations.deepgram` | `DEEPGRAM_API_KEY` |
| AssemblyAI (Universal Streaming v3) | `arc_audio.integrations.assemblyai` | `ASSEMBLYAI_API_KEY` |
| Speechmatics (real-time) | `arc_audio.integrations.speechmatics` | `SPEECHMATICS_API_KEY` (`SPEECHMATICS_RT_URL` for another region; EU by default) |

```bash
pip install "arc-transcriber[integrations] @ git+https://github.com/WhissleAI/arc_transcriber"
export DEEPGRAM_API_KEY=...
python -m arc_audio.integrations.deepgram clip.wav                     # words + Arc's read-out per utterance
python -m arc_audio.integrations.assemblyai clip.wav --domain assistant --json > out.json
python examples/cloud_asr.py deepgram clip.wav                         # the Python API, printed live
```

CLI options: `--language`, `--model`, `--endpoint-ms`, `--timeout`, `--retries`, `--no-fallback`, `--domain`,
`--names`, `--speed`, `--json`.

## Python

```python
from arc_audio import Arc
from arc_audio.integrations import Session

arc = Arc.from_pretrained("WhissleAI/arc-v1.1")
async with Session(arc, "speechmatics", domain="assistant", language="en") as s:
    for chunk in chunks:                    # 16 kHz mono float32, 100 ms
        new = await s.feed(chunk)           # the keys of Stream.feed()
    out = await s.finish()                  # finalise at the provider, end its session, flush Arc
```

`s.end_turn()` finalises a turn and keeps the session open. Each adapter also works alone, as an async context with
`send(chunk)`, `finalize()` and `close()`.

Each adapter:

- streams 16 kHz PCM16 over the provider's websocket (the key goes in the handshake header only);
- turns interim and final results into Arc's external words, each final word once, and reports how far the provider
  has finalised, so Arc's utterances close as soon as the provider is done;
- finalises and ends the session at the end of the input, with timeouts;
- reconnects with backoff and re-sends audio the provider had not finalised, without repeating finals;
- on errors that will not pass, falls back to Arc's own decoder from the first unfinalised audio
  (`out["asr"]["status"] == "fallback"`), so the output keeps every key.

To add a provider, subclass `base.StreamingASR` and list it in `registry.py`. `parse` is a pure function, tested with
recorded messages in `tests/test_integrations.py`.

## Provider notes

| provider | defaults | billing |
|---|---|---|
| Deepgram | `nova-3`, interim results, endpointing 300 ms; `smart_format=True` for written numbers | per second of audio streamed |
| AssemblyAI | `universal-streaming-english` (multilingual when `language` is not English), spoken-form turns; send audio at about real-time pace | per hour of session time: close idle sessions |
| Speechmatics | `enhanced` model, partials on, `max_delay` 0.7 s | per hour of audio, by model |

Check each provider's pricing page for current rates. Low tiers limit concurrent sessions; the adapters wait and retry.

## Bench

`scripts/bench_integrations.py` streams 50 LibriSpeech test-clean clips (and the same clips with pink noise at
10 dB SNR) to each provider at real-time pace, records every message, and replays the recordings into Arc EXTERNAL
streams (libarc) on the same clock. WER uses Whisper's English normaliser.

| provider | audio | WER provider | WER Arc EXTERNAL | time to final p50 / p90 |
|---|---|---|---|---|
| Deepgram | clean / noisy | 2.80 % / 3.61 % | same | 0.05 / 0.12 s |
| AssemblyAI | clean / noisy | 2.71 % / 3.97 % | same | 0.04 / 0.24 s |
| Speechmatics | clean / noisy | 2.44 % / 2.89 % | same | 0.27 / 0.37 s |

The merged text is the provider's, word for word, on every clip. libarc's utterance tags come from the audio only, so
acts, intents and device-directed scores match Arc's own stream exactly. arc_audio's Python EXTERNAL stream also reads
the words it is given, so its tags are close to, not identical with, those over Arc's own transcript.
