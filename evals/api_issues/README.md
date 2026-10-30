# API issues: minimal reproductions

`run.py` reproduces four issues found while building on `arc_audio`: in Steer for Mac, and in the Pipecat speech-to-text
service (`pipecat-whissle`). It runs in about a minute and writes `results/run.json`.

```bash
python evals/api_issues/run.py
```

1. **Short commands get their final transcript about 15 s late.**
   - `UttSegmenter` only closes an utterance on a pause when it has at least `min_s` (1.0 s) of frames.
   - Shorter commands wait for `max_s` (15 s) or `flush()`.
2. **`transcript_rewritten` can add tokens.**
   - In the Pipecat session, "Priya Raymond" was linked to "Priya Raman", but the linked span covered only "Priya".
   - The rewrite then produced "Priya Raman Raymond".
   - The script checks the same sentence (rendered with macOS voices) with a name list.
3. **`Arc.stream()` changes the process-wide `sys.setswitchinterval`** while a stream with a background worker runs, and restores it on `flush()`. In a host process such as an agent framework, that affects every thread.
4. **Two device-directed numbers per utterance.**
   - `utterances[].tags['discourse.assistant.device_directed']` and `transcript[].device_directed.p` can be far apart on the same clip, for example 0.27 and 0.003.
   - The docs don't say which one an app should use.

## Results (this PR)

Run on 2026-10-29: `arc_audio` 1.2.0, model rev `v1.2.0` (Arc Mini 1.1.1 files), CPU, macOS Samantha voice unless noted.

| # | Result |
|---|---|
| 1 | "Pause." (0.46 s of speech): final **+14.5 s** after the speech. "Open WhatsApp." (0.92 s): final **+14.1 s**. "Sleep in twenty minutes." (1.33 s) and "Stop reading Slack for an hour." (1.72 s): final +0.4 s. In every case, a first partial had arrived by the end of the speech. |
| 2 | **Not reproduced** on these renders. With all four voices, `transcript_rewritten` equalled `transcript`. "Priya Raman" was heard right by three voices and linked whole. Flo's "Queen Raymond" wasn't linked, so nothing was rewritten. The Pipecat case was a Steer TTS voice where only "Priya" of "Priya Raymond" was linked. |
| 3 | `sys.getswitchinterval()`: 0.005 before, **0.0005** while a stream runs, 0.005 after `flush()`. A stream that is never flushed leaves it changed. |
| 4 | "Sleep in twenty minutes.": utterance tag **0.32**, transcript head **0.0006**. |
