# Cloud streaming recognisers with Arc (`arc_audio.integrations`)

Arc's EXTERNAL stream mode takes its words from another recogniser and keeps everything it gets from the audio itself:
utterance tags (device-directed, acts, intents), end of turn (`command_complete`), names and numbers tagged on the
words, speakers, sound and channel events. This folder connects that mode to three cloud streaming services:

| provider | module | key (environment) | protocol |
|---|---|---|---|
| Deepgram (nova-3, live streaming) | `arc_audio.integrations.deepgram` | `DEEPGRAM_API_KEY` | `wss://api.deepgram.com/v1/listen` |
| AssemblyAI (Universal Streaming v3) | `arc_audio.integrations.assemblyai` | `ASSEMBLYAI_API_KEY` | `wss://streaming.assemblyai.com/v3/ws` |
| Speechmatics (real-time) | `arc_audio.integrations.speechmatics` | `SPEECHMATICS_API_KEY` | `wss://eu.rt.speechmatics.com/v2` |

```bash
pip install "arc-transcriber[integrations] @ git+https://github.com/WhissleAI/arc_transcriber"   # + websockets
export DEEPGRAM_API_KEY=...
python -m arc_audio.integrations.deepgram clip.wav                    # words + Arc's read-out per utterance
python -m arc_audio.integrations.assemblyai clip.wav --domain assistant --json > out.json
python -m arc_audio.integrations.speechmatics clip.wav --language de
python examples/cloud_asr.py deepgram clip.wav                        # the same with the Python API, printed live
```

Options of the CLI: `--language`, `--model`, `--endpoint-ms` (the provider's end-of-utterance silence), `--timeout`
(seconds to wait for the last finals), `--retries`, `--no-fallback`, `--domain`, `--names`, `--speed`, `--json`.

## Python

```python
from arc_audio import Arc
from arc_audio.integrations import Session

arc = Arc.from_pretrained("WhissleAI/arc-v1.1")
async with Session(arc, "speechmatics", domain="assistant", language="en") as s:
    for chunk in chunks:                    # 16 kHz mono float32, 100 ms
        new = await s.feed(chunk)           # the keys of Stream.feed(): transcript, utterances, events, names, ...
    out = await s.finish()                  # finalise at the provider, end its session, flush Arc
out["text"], out["words"], out["asr"]       # merged text, the words Arc used, {provider, status, time_to_final_s, ...}
```

`s.end_turn()` finalises a turn (push-to-talk released) and keeps the session open. Each adapter can also be used
alone: `Deepgram(on_words=..., on_covered=...)` as an async context with `send(chunk)`, `finalize()`, `close()`.

What each adapter does:

- streams 16 kHz PCM16 over the provider's websocket; auth goes in the handshake header only;
- turns interim and final results into Arc's external words `{text, start, end, is_final, confidence}` (seconds on
  the audio clock), each final word once, and reports how far the provider has finalised (`covered`), so Arc's
  utterances close as soon as the provider is done with them;
- finalises at the end of the input (Deepgram `Finalize`, AssemblyAI `ForceEndpoint`, Speechmatics
  `ForceEndOfUtterance`), then ends the session (`CloseStream`, `Terminate`, `EndOfStream`), each with a timeout;
- reconnects with exponential backoff (and the provider's wait on quota / concurrency limits), re-sending the audio
  the provider had not finalised, on the same clock, without repeating finals;
- gives up on errors that will not pass (bad key, bad request) and on running out of retries; the `Session` then
  feeds words from Arc's own decoder into the same stream from the first audio the provider had not finalised
  (`out["asr"]["status"] == "fallback"`), so the output keeps every key.

Adding a provider: subclass `base.StreamingASR` (URL, headers, start / finalise / end messages, `parse(message) ->
Parsed`) and list it in `registry.py`. `parse` is a pure function, so it can be tested with recorded messages
(`tests/test_integrations.py`).

## Providers

### Deepgram

`DEEPGRAM_API_KEY`. Defaults: `model=nova-3`, `interim_results=true`, `endpointing=300` ms, `utterance_end_ms=1000`,
`punctuate=true` (add `smart_format=True` as an option for written-form numbers). Finals arrive as `is_final` results
during speech; `Finalize` returns the rest at once.
Latency: BENCH_DG. Cost: billed per second of audio streamed (pay-as-you-go nova-3 streaming is listed at
about $0.0077 per minute at the time of writing; check Deepgram's pricing page).

### AssemblyAI

`ASSEMBLYAI_API_KEY`. Defaults: `speech_model=universal-streaming-english` (`universal-streaming-multilingual` when
`language` is not English; any other model with `--model`), `format_turns=false` (spoken form, which Arc's number and
name tags read directly; `format_turns=True` takes the finals from the formatted copy of each turn, a little later).
Words become final one by one inside a turn (`word_is_final`) and go to Arc as they do. Send audio at about real-time
pace in 50-1000 ms chunks (the service asks for it). The service ignores unknown parameters, so the adapter logs the
model the session reports in `Begin`. Free and low tiers limit concurrent sessions; the adapter waits and retries.
Latency: BENCH_AA. Cost: billed per hour of session (connection) time, not only speech: close sessions you do not
use (about $0.15 per hour for Universal Streaming at the time of writing; check AssemblyAI's pricing page).

### Speechmatics

`SPEECHMATICS_API_KEY`; `SPEECHMATICS_RT_URL` (or the `url` option) for another region (default EU). Defaults:
`model=enhanced` (`standard` costs less), `enable_partials=true`, `max_delay=0.7` s (the lowest; finals then come 0.7-2 s after
the word), `max_delay_mode=flexible`. `--endpoint-ms` sets `end_of_utterance_silence_trigger`. Accounts limit
concurrent sessions (the server's `Info` message says how many); on `quota_exceeded` the adapter waits 5 s and retries.
Latency: BENCH_SM. Cost: billed per hour of audio, by model (enhanced costs more than standard; check
Speechmatics' pricing page).

## Bench

`scripts/bench_integrations.py` streams 50 LibriSpeech test-clean clips (and the same clips with pink noise at
10 dB SNR) to each provider at real-time pace, records every message with its arrival time, and replays the
recordings into Arc EXTERNAL streams on the same clock.

BENCH_TABLE
