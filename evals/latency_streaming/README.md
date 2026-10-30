# Latency and streaming

**What it measures.** How fast Arc Mini answers a short command, offline and in a stream fed at real-time pace:
- **Offline:** `transcribe()` and `predict()` time on the clip plus a 0.4 s tail (what a press-and-hold key sends), and the real-time factor.
- **Stream:**
  - The clip, followed by 6 s of quiet room, is fed in 100 ms chunks at real-time pace with the default `arc.stream(domain="assistant")`, background worker on.
  - The run records when each output first arrives, in seconds after the speech ends (on the audio clock):
    - the first partial that already has the final words;
    - the **final transcript record**;
    - the utterance record (act, intent, device-directed tag);
    - the transcript's device-directed record;
    - `command_complete` events, and whether one fires more than 0.5 s before the end ("early, mid-command").
  - It also records each feed's compute time.
- **App-ended turn:** the clip plus a 0.3 s tail, then `flush()`. This is what the Pipecat integration does on its VAD's
  end of turn. The run records how long `flush()` takes.

Lines (`data/commands.tsv`): Steer's ten signal-study commands, and two longer assistant commands from the voice-assistant replay
("Set an alarm at nine am on next Sunday", "Create a new list called groceries").

## Run

```bash
python evals/latency_streaming/run.py             # 12 commands x 2 voices, real-time pace, about 6 min
python evals/latency_streaming/run.py --tail 20   # a longer silent tail
python evals/api_issues/run.py                    # the minimal repro of the late final (and the other API issues)
```

## Results

**Steer (libarc, app path)**: Steer for Mac 1.6.x on an M3 Pro, Arc Mini 1.1.1:

| Step | Time |
|---|---|
| Arc on a 1.4–2.2 s command (offline decode of the held clip) | **0.05–0.12 s** (0.37 s worst) |
| Bench p50 / p95 per command (7,648 clips) | 39–64 ms / 61–84 ms |
| Book search: live partial decode, median / p90 | 20.9 / 63.8 ms |
| Book search: final decode with live context (film take) | 122 ms in Arc; title shown 0.1 s after the last word (the bench's "7 ms" times only the relisten step) |
| Hand-off to the main thread | usually < 0.05 s; 0.4–3.6 s spikes when the app was in the background (App Nap, not Arc) |

**Python reference, earlier runs**:
- The signal study (`arc_audio` 1.1.1, M3 Pro, other jobs running):
  - offline 0.18–0.7× real time;
  - streaming, a 100 ms feed took 88–214 ms on average (p95 up to 456 ms, spikes over 1 s);
  - utterance records arrived 0.8 s after the utterance ended, and the transcript 0.9–1.8 s after.
- The Pipecat integration (`arc-transcriber` 1.1.1, M3 Pro):
  - Arc's own end-of-utterance final came about **2.7 s** after speech ended, too slow for an agent;
  - so the service makes a new `arc.stream()` per VAD turn and calls `flush()`;
  - from end of speech to the final transcript (Pipecat's STT TTFB, which includes the VAD's stop wait) took 0.42–0.79 s, median 0.56 s;
  - a 100 ms feed took 18 ms (median).
- The voice-assistant replay (Arc v1.4 demo server; ASR comparison film):
  - `command_complete` fired early, mid-command: "set alarm at nine am | on next sunday" got a complete at "am";
  - on a command-routing page, `command_complete` recall was 1.0 and precision 0.91 (34 calls).

**Python reference (this PR):** `arc_audio` 1.2.0, model rev `v1.2.0` (Arc Mini 1.1.1 files), CPU, 4 threads, M3 Pro, 2026-10-29.

| Measure | Result |
|---|---|
| Offline `transcribe()` on a command plus a 0.4 s tail, median / p90 | 167 / 246 ms (real-time factor 0.076) |
| Offline `predict()` (all heads), median / p90 | 190 / 279 ms |
| Stream: one 100 ms feed, median / p90 / max compute | 13 / 31 / 93 ms |
| Stream: first partial with the final words, after the speech ends | median +0.11 s, p90 +0.30 s (19/24 partials ever matched the final) |
| Stream: `command_complete` emitted, after the speech ends | median +0.18 s, p90 +0.36 s; **missing on 5/24** ("Pause." ×2, "Read faster.", "Skip to chapter twelve.", "Set an alarm at nine am on next Sunday."); 1 more than 0.5 s before the end ("Sleep in twenty minutes", −0.51 s, on its last word) |
| Stream: utterance record (act, intent, directed tag) | median +0.79 s, p90 +0.89 s, all 24 |
| Stream: **final transcript record** | **15/24 at +0.58–0.68 s; 9/24 not within the 6 s of silence**: every command under ~1 s of speech ("Pause.", "Read faster.", "Say that again.", "What time is it?", "Summarise this.", "Open WhatsApp.") |
| `flush()` after speech + 0.3 s (app ends the turn), median / p90 | 155 / 201 ms |
| Minimal repro (`api_issues`, 20 s tail, no worker thread) | see `api_issues/README.md` |

## The late final, explained

`arc_audio/transcribe.py`, `UttSegmenter.feed`, closes an utterance on a pause only if it is long enough. The gate is `n - low_run >= min_n`, with `min_s` = 1.0 s in Mini's config.
- An utterance shorter than 1 s ("Pause.", "Open WhatsApp.") is not closed at the pause.
- The quiet frames pile onto it until `max_s` = 15 s closes it.
- So its final transcript record arrives about 14–15 s after the speech, or when the app calls `flush()`. Longer commands
  close at the pause and arrive about 0.6–0.8 s after it.
- `evals/api_issues/run.py` reproduces this (`stream(background=False)`, 20 s of silence after each command).

## Known limits

- The pace is real time with the default background worker, so the times include this machine's scheduling. Each run used 4 threads on a laptop.
- The speech end is measured from the audio: the last 10 ms frame within 35 dB of the peak.
